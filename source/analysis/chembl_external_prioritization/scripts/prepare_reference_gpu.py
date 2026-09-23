#!/usr/bin/env python3
"""Resolve exact reference ligand tensors and valid full-protein tensors."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import torch


LIGAND_DIM = 512
PROTEIN_DIM = 1536


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def normalize_path(value, root):
    text = str(value).strip()
    for old in ["/shared_data/11.HS_allostery", "/disk9/13.Heesu_Allostery"]:
        if text.startswith(old + "/"):
            return str(root) + text[len(old):]
    return text


def tensor_ok(path, dimension):
    try:
        value = torch.load(str(path), map_location="cpu")
        if isinstance(value, dict):
            values = [x for x in value.values() if torch.is_tensor(x) and x.ndim == 2 and int(x.shape[-1]) == dimension]
            if not values:
                return False, None
            value = values[0]
        valid = bool(
            torch.is_tensor(value) and value.ndim == 2 and int(value.shape[0]) > 0
            and int(value.shape[-1]) == dimension and torch.isfinite(value).all()
        )
        return valid, [int(x) for x in value.shape] if valid else None
    except Exception:
        return False, None


def main():
    args = parse_args()
    root = args.project_root
    package = root / "analysis/chembl_external_prioritization"
    main_package = root / "analysis/allosteric_pair_benchmark_main"
    broad_package = root / "analysis/allosteric_pair_benchmark_broad_superset"
    input_path = package / "data/CHEMBL_DIRECT_REFERENCE_PAIRS.tsv.gz"
    output_path = package / "gpu_cache/REFERENCE_MODEL_READY.tsv.gz"
    ligand_manifest_path = package / "gpu_cache/REFERENCE_LIGAND_MANIFEST.tsv.gz"
    validation_path = package / "validation/GPU_REFERENCE_INPUT_VALIDATION.json"
    index_path = root / "17.paired_dataset/legacy/0.ligand_path_index.pkl"
    exact_script = main_package / "scripts/prepare_exact_gpu_inputs.py"
    old_protein_map_path = root / "14.Organized_input/Protein_Embedding_Map.tsv"
    for path in [input_path, index_path, exact_script, old_protein_map_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    exact = load_module(exact_script, "current_exact_ligand_contract")
    frame = pd.read_csv(input_path, sep="\t", low_memory=False)
    if output_path.is_file() and ligand_manifest_path.is_file() and validation_path.is_file():
        prior = json.loads(validation_path.read_text(encoding="utf-8"))
        if prior.get("status") == "validated" and prior.get("source_rows") == len(frame):
            print("SKIP completed reference input preparation")
            print(json.dumps(prior, indent=2, sort_keys=True))
            return
    frame["full_inchikey"] = frame["full_inchikey"].map(exact.norm)
    exact_smiles, smiles_audit = exact.exact_smiles_by_key(frame)
    with index_path.open("rb") as handle:
        raw_index = pickle.load(handle)
    ligand_index = {exact.norm(key): value for key, value in raw_index.items() if exact.norm(key)}
    ligand_root = package / "gpu_cache/exact_reference_ligands"
    resolved = {}
    sources = {}
    to_bake = {}
    failures = {}
    for full_key in sorted(frame["full_inchikey"].unique()):
        cached = exact.validated_generated_cache(ligand_root, full_key)
        if cached:
            resolved[full_key] = cached
            sources[full_key] = "generated_exact_cache"
            continue
        legacy, legacy_failures = exact.exact_legacy_path(ligand_index, full_key, root)
        if legacy:
            resolved[full_key] = legacy
            sources[full_key] = "legacy_exact_full_inchikey_filename"
        elif full_key in exact_smiles:
            to_bake[full_key] = exact_smiles[full_key]
        else:
            failures[full_key] = "no RDKit-exact SMILES and no exact legacy tensor"
        for path, error in legacy_failures.items():
            failures[full_key + "|" + path] = error
    generated, generation_failures = exact.bake(to_bake, ligand_root, args.batch_size)
    resolved.update(generated)
    sources.update({key: "generated_exact_smiles" for key in generated})
    failures.update(generation_failures)

    ligand_rows = []
    for key in sorted(set(frame["full_inchikey"])):
        path = resolved.get(key, "")
        valid, shape = tensor_ok(path, LIGAND_DIM) if path else (False, None)
        ligand_rows.append({
            "full_inchikey": key,
            "ligand_embedding_path": path,
            "source": sources.get(key, "unresolved"),
            "valid": int(valid),
            "shape": "x".join(map(str, shape)) if shape else "",
            "smiles_exact": int(bool(smiles_audit.get(key, {}).get("exact_smiles_found"))),
        })
    ligand_manifest = pd.DataFrame(ligand_rows)
    ligand_manifest.to_csv(ligand_manifest_path, sep="\t", index=False)
    ligand_path_by_key = dict(zip(ligand_manifest["full_inchikey"], ligand_manifest["ligand_embedding_path"]))
    ligand_valid_by_key = dict(zip(ligand_manifest["full_inchikey"], ligand_manifest["valid"].astype(int)))

    # Prefer current target-chain tensors when available, then fall back to the
    # established full-protein ESM3 map used to build the ChEMBL tensor cache.
    protein_paths = {}
    required_proteins = set(frame["uniprot"].astype(str))
    for model_ready_path in [
        main_package / "gpu_cache/MODEL_READY.tsv.gz",
        broad_package / "gpu_cache/BROAD_MODEL_READY.tsv.gz",
    ]:
        table = pd.read_csv(model_ready_path, sep="\t", usecols=["uniprot", "protein_embedding_path"])
        table = table[table["uniprot"].astype(str).isin(required_proteins)]
        for row in table.drop_duplicates("uniprot").itertuples(index=False):
            path = normalize_path(row.protein_embedding_path, root)
            valid, _ = tensor_ok(path, PROTEIN_DIM)
            if valid:
                protein_paths.setdefault(str(row.uniprot), path)
    old = pd.read_csv(old_protein_map_path, sep="\t", low_memory=False)
    old = old[old["UniProt_ID"].astype(str).str.split("-").str[0].isin(required_proteins)]
    for row in old[["UniProt_ID", "Embedding_Path"]].drop_duplicates("UniProt_ID").itertuples(index=False):
        uid = str(row.UniProt_ID).split("-")[0].strip()
        path = normalize_path(row.Embedding_Path, root)
        valid, _ = tensor_ok(path, PROTEIN_DIM)
        if valid:
            protein_paths.setdefault(uid, path)

    frame["ligand_embedding_path"] = frame["full_inchikey"].map(ligand_path_by_key).fillna("")
    frame["protein_embedding_path"] = frame["uniprot"].astype(str).map(protein_paths).fillna("")
    # Do not reload the same protein tensor once per pair.  Every path was
    # validated once while building the maps above.
    frame["ligand_ready"] = frame["full_inchikey"].map(ligand_valid_by_key).fillna(0).astype(int)
    frame["protein_ready"] = frame["uniprot"].astype(str).isin(protein_paths).astype(int)
    ready = frame[frame["ligand_ready"].eq(1) & frame["protein_ready"].eq(1)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ready.to_csv(output_path, sep="\t", index=False)
    report = {
        "status": "validated" if len(ready) and ready["weak2020_label"].isin([0, 1]).any() else "failed",
        "source_rows": int(len(frame)),
        "model_ready_rows": int(len(ready)),
        "model_ready_proteins": int(ready["uniprot"].nunique()),
        "model_ready_ligands": int(ready["full_inchikey"].nunique()),
        "missing_ligand_rows": int(frame["ligand_ready"].eq(0).sum()),
        "missing_protein_rows": int(frame["protein_ready"].eq(0).sum()),
        "weak2020_allosteric_rows": int(ready["weak2020_label"].eq(1).sum()),
        "weak2020_orthosteric_rows": int(ready["weak2020_label"].eq(0).sum()),
        "five_a_positive_rows": int(ready["is_5a_positive"].eq(1).sum()),
        "ligand_sources": ligand_manifest.groupby("source").size().astype(int).to_dict(),
        "n_recorded_failures": int(len(failures)),
        "output": str(output_path),
    }
    if not report["weak2020_allosteric_rows"] or not report["weak2020_orthosteric_rows"]:
        report["status"] = "failed"
    atomic_json(validation_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("reference input preparation failed")


if __name__ == "__main__":
    main()
