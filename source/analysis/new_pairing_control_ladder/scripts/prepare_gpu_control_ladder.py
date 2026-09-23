#!/usr/bin/env python3
"""Resolve GPU embeddings for the broad structure-ready control-ladder table."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-bytes", type=int, default=1000)
    return parser.parse_args()


def norm(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("InChIKey=", "").upper()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def valid_path(path, min_bytes):
    try:
        return bool(path) and os.path.isfile(path) and os.path.getsize(path) > min_bytes
    except OSError:
        return False


def normalize_path(path, project_root):
    text = str(path).strip()
    for old in ["/shared_data/11.HS_allostery", "/disk9/13.Heesu_Allostery"]:
        if text.startswith(old + "/"):
            return str(project_root) + text[len(old):]
    return text


def extract_paths(value):
    paths = []
    if isinstance(value, str):
        paths.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            paths.extend(extract_paths(nested))
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            paths.extend(extract_paths(nested))
    return paths


def choose_index_path(index, full_key, project_root, min_bytes):
    values = []
    for key in [full_key, full_key[:14]]:
        if key in index:
            values.extend(extract_paths(index[key]))
    candidates = []
    for path in values:
        normalized = normalize_path(path, project_root)
        if valid_path(normalized, min_bytes):
            candidates.append(normalized)
    if not candidates:
        return ""
    return sorted(set(candidates), key=lambda path: (
        0 if full_key in os.path.basename(path).upper() else 1, path
    ))[0]


def tensor_from_object(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        tensors = [item for item in value.values() if torch.is_tensor(item) and item.ndim == 2]
        if tensors:
            return tensors[0]
    raise TypeError("embedding file does not contain a 2D tensor")


def validate_embedding(path, expected_dim):
    tensor = tensor_from_object(torch.load(path, map_location="cpu"))
    if tensor.ndim != 2 or int(tensor.shape[-1]) != expected_dim or int(tensor.shape[0]) < 1:
        raise ValueError("unexpected tensor shape {} at {}".format(tuple(tensor.shape), path))
    return tuple(int(value) for value in tensor.shape)


def validate_graph(path, protein_length):
    with np.load(path, allow_pickle=False) as graph:
        required = {"embedding_index", "ca_xyz", "neighbor_index", "neighbor_distance"}
        if required - set(graph.files):
            raise ValueError("graph cache missing arrays")
        index = np.asarray(graph["embedding_index"], dtype=np.int64)
        xyz = np.asarray(graph["ca_xyz"], dtype=np.float32)
        neighbor = np.asarray(graph["neighbor_index"], dtype=np.int64)
        distance = np.asarray(graph["neighbor_distance"], dtype=np.float32)
    n = int(len(index))
    if n < 10 or xyz.shape != (n, 3) or neighbor.ndim != 2 or distance.shape != neighbor.shape:
        raise ValueError("invalid graph shapes")
    if int(index.min()) < 0 or int(index.max()) >= int(protein_length):
        raise ValueError("graph embedding index is outside protein tensor")
    if int(neighbor.min()) < 0 or int(neighbor.max()) >= n:
        raise ValueError("graph neighbor index is outside graph")
    if not np.isfinite(xyz).all() or not np.isfinite(distance).all():
        raise ValueError("graph has non-finite values")
    return {"n_residues": n, "n_neighbors": int(neighbor.shape[1])}


def bake_missing(records, output_root, batch_size):
    if not records:
        return {}, {}
    from unimol_tools import UniMolRepr

    model = UniMolRepr(data_type="molecule", remove_hs=True, use_gpu=True, batch_size=batch_size)
    completed = {}
    failed = {}

    def save_one(key, atom_repr):
        tensor = torch.as_tensor(atom_repr, dtype=torch.float32)
        if tensor.ndim != 2 or int(tensor.shape[-1]) != 512 or int(tensor.shape[0]) < 1:
            raise ValueError("unexpected UniMol atomic representation {}".format(tuple(tensor.shape)))
        directory = output_root / key[:2]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "{}.pt".format(key)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(tensor, temporary)
        os.replace(temporary, path)
        completed[key] = str(path)

    items = list(records.items())
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        try:
            result = model.get_repr([value for _, value in batch], return_atomic_reprs=True)
            atomic = result["atomic_reprs"]
            if len(atomic) != len(batch):
                raise RuntimeError("UniMol batch output length mismatch")
            for (key, _), value in zip(batch, atomic):
                save_one(key, value)
            print("UniMol baked {}/{}".format(min(start + len(batch), len(items)), len(items)), flush=True)
        except Exception as batch_error:
            print("UniMol batch fallback: {}".format(batch_error), flush=True)
            for key, smiles in batch:
                try:
                    result = model.get_repr([smiles], return_atomic_reprs=True)
                    save_one(key, result["atomic_reprs"][0])
                except Exception as error:
                    failed[key] = "{}: {}".format(type(error).__name__, error)
    return completed, failed


def main():
    args = parse_args()
    package = args.project_root / "analysis/new_pairing_control_ladder"
    source = package / "data/BROAD_UNCONTROLLED_STRUCTURE_READY.tsv.gz"
    index_path = args.project_root / "17.paired_dataset/legacy/0.ligand_path_index.pkl"
    output_root = package / "gpu_cache/ligand_embeddings"
    output = package / "gpu_cache/BASE_MODEL_READY.tsv.gz"
    validation = package / "gpu_cache/BASE_EMBEDDING_VALIDATION.json"
    if not source.is_file() or not index_path.is_file():
        raise FileNotFoundError("missing broad dataset or frozen ligand index")

    frame = pd.read_csv(source, sep="\t", low_memory=False)
    with index_path.open("rb") as handle:
        raw_index = pickle.load(handle)
    index = {}
    for key, value in raw_index.items():
        normalized = norm(key)
        if normalized:
            index.setdefault(normalized, value)
    del raw_index

    records = frame[["full_inchikey", "canonical_smiles"]].drop_duplicates("full_inchikey")
    resolved = {}
    resolved_from_index = set()
    resolved_from_baked_cache = set()
    invalid_existing_ligand_tensors = {}
    missing_with_smiles = {}
    missing_without_smiles = {}
    for row in records.itertuples(index=False):
        key = norm(row.full_inchikey)
        path = choose_index_path(index, key, args.project_root, args.min_bytes)
        if path:
            try:
                validate_embedding(path, 512)
                resolved[key] = path
                resolved_from_index.add(key)
                continue
            except Exception as error:
                invalid_existing_ligand_tensors[key] = {
                    "path": path,
                    "error": "{}: {}".format(type(error).__name__, error),
                }
        baked_path = output_root / key[:2] / "{}.pt".format(key)
        if valid_path(str(baked_path), args.min_bytes):
            try:
                validate_embedding(str(baked_path), 512)
                resolved[key] = str(baked_path)
                resolved_from_baked_cache.add(key)
                continue
            except Exception as error:
                invalid_existing_ligand_tensors[key] = {
                    "path": str(baked_path),
                    "error": "{}: {}".format(type(error).__name__, error),
                }
        smiles = "" if pd.isna(row.canonical_smiles) else str(row.canonical_smiles).strip()
        if smiles:
            missing_with_smiles[key] = smiles
        else:
            missing_without_smiles[key] = "blank canonical_smiles"

    baked, bake_failed = bake_missing(missing_with_smiles, output_root, args.batch_size)
    resolved.update(baked)
    failed = dict(missing_without_smiles)
    failed.update(bake_failed)

    frame["ligand_embedding_path"] = frame["full_inchikey"].map(lambda value: resolved.get(norm(value), ""))
    frame["protein_embedding_path"] = frame["protein_embedding_path"].map(lambda path: normalize_path(path, args.project_root))
    frame["structure_graph_path"] = frame["structure_graph_path"].map(lambda path: normalize_path(path, args.project_root))
    ligand_ok = frame["ligand_embedding_path"].map(lambda path: valid_path(path, args.min_bytes))
    protein_ok = frame["protein_embedding_path"].map(lambda path: valid_path(path, args.min_bytes))
    graph_ok = frame["structure_graph_path"].map(lambda path: valid_path(path, 200))
    ready = frame[ligand_ok & protein_ok & graph_ok].copy()

    ligand_shapes = {}
    invalid_resolved_ligand_paths = {}
    for path in sorted(ready["ligand_embedding_path"].unique()):
        try:
            ligand_shapes[path] = validate_embedding(path, 512)
        except Exception as error:
            invalid_resolved_ligand_paths[path] = "{}: {}".format(type(error).__name__, error)
    if invalid_resolved_ligand_paths:
        affected = ready["ligand_embedding_path"].isin(invalid_resolved_ligand_paths)
        for row in ready.loc[affected, ["full_inchikey", "ligand_embedding_path"]].drop_duplicates().itertuples(index=False):
            failed[norm(row.full_inchikey)] = invalid_resolved_ligand_paths[str(row.ligand_embedding_path)]
        ready = ready[~affected].copy()

    protein_shapes = {}
    invalid_protein_paths = {}
    for path in sorted(ready["protein_embedding_path"].unique()):
        try:
            protein_shapes[path] = validate_embedding(path, 1536)
        except Exception as error:
            invalid_protein_paths[path] = "{}: {}".format(type(error).__name__, error)
    if invalid_protein_paths:
        ready = ready[~ready["protein_embedding_path"].isin(invalid_protein_paths)].copy()
    invalid_graph_proteins = set()
    graph_failures = {}
    graph_shapes = {}
    for row in ready[["uniprot", "protein_embedding_path", "structure_graph_path"]].drop_duplicates("uniprot").itertuples(index=False):
        try:
            graph_shapes[str(row.structure_graph_path)] = validate_graph(
                str(row.structure_graph_path), protein_shapes[str(row.protein_embedding_path)][0]
            )
        except Exception as error:
            invalid_graph_proteins.add(str(row.uniprot))
            graph_failures[str(row.uniprot)] = "{}: {}".format(type(error).__name__, error)
    ready = ready[~ready["uniprot"].astype(str).isin(invalid_graph_proteins)].copy().reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    ready.to_csv(output, sep="\t", index=False, compression="gzip")

    labels = ready["class_label"].value_counts().to_dict()
    report = {
        "status": "validated",
        "source_rows": int(len(frame)),
        "source_proteins": int(frame["uniprot"].nunique()),
        "model_ready_rows": int(len(ready)),
        "model_ready_proteins": int(ready["uniprot"].nunique()),
        "model_ready_ligands": int(ready["full_inchikey"].nunique()),
        "model_ready_allosteric": int(labels.get("allosteric", 0)),
        "model_ready_orthosteric": int(labels.get("orthosteric", 0)),
        "model_ready_proteins_with_both_labels": int(ready.groupby("uniprot")["binary_label"].nunique().eq(2).sum()),
        "resolved_from_existing_index": int(len(resolved_from_index)),
        "resolved_from_baked_cache": int(len(resolved_from_baked_cache)),
        "new_unimol_embeddings": int(len(baked)),
        "failed_unique_ligands": int(len(failed)),
        "failed_ligands": failed,
        "invalid_existing_ligand_tensors": invalid_existing_ligand_tensors,
        "invalid_resolved_ligand_paths": invalid_resolved_ligand_paths,
        "invalid_protein_paths": invalid_protein_paths,
        "invalid_graph_proteins": sorted(invalid_graph_proteins),
        "graph_failures": graph_failures,
        "unique_ligand_tensors_validated": int(len(ligand_shapes)),
        "unique_protein_tensors_validated": int(len(protein_shapes)),
        "unique_structure_graphs_validated": int(len(graph_shapes)),
        "output": str(output),
    }
    if not all([len(ready) > 0, set(ready["binary_label"]) == {0, 1}]):
        report["status"] = "failed"
    atomic_json(validation, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("base GPU embedding validation failed")


if __name__ == "__main__":
    main()
