#!/usr/bin/env python3
"""Resolve exact full-InChIKey ligand tensors and validate the graph-free cohort."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path

import pandas as pd
import torch


LIGAND_DIM = 512
PROTEIN_DIM = 1536


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--minimum-proteins", type=int, default=400)
    parser.add_argument("--disable-pubchem-rescue", action="store_true")
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
    os.replace(str(temporary), str(path))


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


def tensor_from_object(value, expected_dim):
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        candidates = [
            item for item in value.values()
            if torch.is_tensor(item) and item.ndim == 2 and int(item.shape[-1]) == expected_dim
        ]
        if not candidates:
            raise TypeError("no compatible 2D tensor in dictionary")
        tensor = candidates[0]
    else:
        raise TypeError("unsupported embedding object")
    if tensor.ndim != 2 or int(tensor.shape[-1]) != expected_dim or int(tensor.shape[0]) < 1:
        raise ValueError("unexpected tensor shape {}".format(tuple(tensor.shape)))
    if not torch.isfinite(tensor).all():
        raise ValueError("embedding contains non-finite values")
    return tensor.detach().cpu().float().contiguous()


def validate_tensor(path, expected_dim):
    tensor = tensor_from_object(torch.load(path, map_location="cpu"), expected_dim)
    return tuple(int(value) for value in tensor.shape)


def rdkit_key(smiles):
    from rdkit import Chem
    from rdkit.Chem import inchi

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "", "RDKit MolFromSmiles returned None"
    try:
        return str(inchi.MolToInchiKey(molecule)).upper(), ""
    except Exception as error:
        return "", "{}: {}".format(type(error).__name__, error)


def pubchem_isomeric_smiles(full_key):
    """Resolve only an otherwise-unusable identifier; RDKit revalidates it."""
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
        "{}/property/IsomericSMILES/JSON".format(full_key)
    )
    try:
        request = Request(url, headers={"User-Agent": "allosteric-pair-benchmark/1.0"})
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        properties = payload.get("PropertyTable", {}).get("Properties", [])
        if not properties:
            return "", "PubChem response contains no property record"
        smiles = str(properties[0].get("SMILES", "")).strip()
        return (smiles, "") if smiles else ("", "PubChem response contains no isomeric SMILES")
    except HTTPError as error:
        return "", "PubChem HTTP {}".format(error.code)
    except URLError as error:
        return "", "PubChem URL error: {}".format(error.reason)
    except Exception as error:
        return "", "PubChem {}: {}".format(type(error).__name__, error)


def exact_smiles_by_key(frame):
    selected = {}
    audit = {}
    for key, group in frame.groupby("full_inchikey", sort=True):
        full_key = norm(key)
        values = []
        for value in group["canonical_smiles"]:
            text = "" if pd.isna(value) else str(value).strip()
            if text and text.lower() not in {"nan", "none", "null"} and text not in values:
                values.append(text)
        checks = []
        for smiles in values:
            observed, error = rdkit_key(smiles)
            checks.append({"smiles": smiles, "rdkit_inchikey": observed, "error": error})
            if observed == full_key and full_key not in selected:
                selected[full_key] = smiles
        audit[full_key] = {
            "n_smiles_candidates": len(values),
            "exact_smiles_found": full_key in selected,
            "checks": checks,
        }
    return selected, audit


def exact_legacy_path(index, full_key, project_root):
    candidates = []
    for path in extract_paths(index.get(full_key, [])):
        normalized = normalize_path(path, project_root)
        if full_key not in os.path.basename(normalized).upper():
            continue
        if os.path.isfile(normalized):
            candidates.append(normalized)
    failures = {}
    for path in sorted(set(candidates)):
        try:
            validate_tensor(path, LIGAND_DIM)
            return path, failures
        except Exception as error:
            failures[path] = "{}: {}".format(type(error).__name__, error)
    return "", failures


def validated_generated_cache(output_root, full_key):
    path = output_root / full_key[:2] / "{}.pt".format(full_key)
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return ""
    try:
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("full_inchikey") != full_key or info.get("rdkit_inchikey") != full_key:
            return ""
        validate_tensor(str(path), LIGAND_DIM)
        return str(path)
    except Exception:
        return ""


def save_generated(output_root, full_key, smiles, atomic_repr):
    tensor = torch.as_tensor(atomic_repr, dtype=torch.float32)
    if tensor.ndim != 2 or int(tensor.shape[-1]) != LIGAND_DIM or int(tensor.shape[0]) < 1:
        raise ValueError("unexpected UniMol atomic representation {}".format(tuple(tensor.shape)))
    if not torch.isfinite(tensor).all():
        raise ValueError("UniMol representation contains non-finite values")
    directory = output_root / full_key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "{}.pt".format(full_key)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(tensor, temporary)
    os.replace(str(temporary), str(path))
    atomic_json(path.with_suffix(".json"), {
        "canonical_smiles": smiles,
        "full_inchikey": full_key,
        "rdkit_inchikey": full_key,
        "tensor_shape": [int(value) for value in tensor.shape],
    })
    return str(path)


def bake(records, output_root, batch_size):
    if not records:
        return {}, {}
    from unimol_tools import UniMolRepr

    model = UniMolRepr(data_type="molecule", remove_hs=True, use_gpu=True, batch_size=batch_size)
    completed = {}
    failed = {}
    items = list(sorted(records.items()))
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        try:
            result = model.get_repr([smiles for _, smiles in batch], return_atomic_reprs=True)
            values = result["atomic_reprs"]
            if len(values) != len(batch):
                raise RuntimeError("UniMol batch output length mismatch")
            for (full_key, smiles), value in zip(batch, values):
                completed[full_key] = save_generated(output_root, full_key, smiles, value)
            print("Exact UniMol {}/{}".format(min(start + len(batch), len(items)), len(items)), flush=True)
        except Exception as batch_error:
            print("Exact UniMol batch fallback: {}".format(batch_error), flush=True)
            for full_key, smiles in batch:
                try:
                    result = model.get_repr([smiles], return_atomic_reprs=True)
                    completed[full_key] = save_generated(
                        output_root, full_key, smiles, result["atomic_reprs"][0]
                    )
                except Exception as error:
                    failed[full_key] = "{}: {}".format(type(error).__name__, error)
    return completed, failed


def main():
    args = parse_args()
    package = args.project_root / "analysis/allosteric_pair_benchmark_main"
    cohort_path = package / "data/MAIN_COHORT.tsv.gz"
    masks_path = package / "data/POCKET_INDICES.json"
    index_path = args.project_root / "17.paired_dataset/legacy/0.ligand_path_index.pkl"
    output_root = package / "gpu_cache/exact_ligand_embeddings"
    model_ready_path = package / "gpu_cache/MODEL_READY.tsv.gz"
    manifest_path = package / "gpu_cache/LIGAND_EMBEDDING_MANIFEST.tsv.gz"
    pubchem_path = package / "gpu_cache/PUBCHEM_EXACT_SMILES.tsv.gz"
    validation_path = package / "validation/GPU_INPUT_VALIDATION.json"
    for path in [cohort_path, masks_path, index_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_csv(cohort_path, sep="\t", low_memory=False)
    with masks_path.open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    with index_path.open("rb") as handle:
        raw_index = pickle.load(handle)
    index = {norm(key): value for key, value in raw_index.items() if norm(key)}
    del raw_index

    exact_smiles, smiles_audit = exact_smiles_by_key(frame)
    resolved = {}
    source = {}
    legacy_failures = {}
    to_bake = {}
    rows = []
    for full_key in sorted(frame["full_inchikey"].map(norm).unique()):
        smiles = exact_smiles.get(full_key, "")
        cached = validated_generated_cache(output_root, full_key)
        if cached:
            resolved[full_key] = cached
            source[full_key] = "generated_exact_cache"
            continue
        legacy, failures = exact_legacy_path(index, full_key, args.project_root)
        legacy_failures.update({"{}|{}".format(full_key, path): error for path, error in failures.items()})
        if legacy:
            resolved[full_key] = legacy
            source[full_key] = "legacy_exact_full_inchikey_filename"
        elif smiles:
            to_bake[full_key] = smiles

    generated, generation_failures = bake(to_bake, output_root, args.batch_size)
    resolved.update(generated)
    source.update({key: "generated_exact_smiles" for key in generated})
    # Some legacy sources retain an exact InChIKey but a non-isomeric or
    # standardised SMILES.  Resolve only those residual identifiers through
    # PubChem, then require RDKit to reproduce the original full key before a
    # tensor is generated.  PubChem never supplies a label or a replacement
    # ligand identity.
    pubchem_audit = []
    pubchem_to_bake = {}
    if not args.disable_pubchem_rescue:
        unresolved_before_pubchem = [
            full_key for full_key in sorted(frame["full_inchikey"].map(norm).unique())
            if full_key not in resolved
        ]
        for position, full_key in enumerate(unresolved_before_pubchem, 1):
            smiles, error = pubchem_isomeric_smiles(full_key)
            observed, rdkit_error = rdkit_key(smiles) if smiles else ("", "")
            accepted = bool(smiles and observed == full_key)
            pubchem_audit.append({
                "full_inchikey": full_key,
                "pubchem_isomeric_smiles": smiles,
                "rdkit_inchikey": observed,
                "status": "accepted" if accepted else "rejected",
                "reason": "" if accepted else (error or rdkit_error or "full-InChIKey mismatch"),
                "url": (
                    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
                    "{}/property/IsomericSMILES/JSON".format(full_key)
                ),
            })
            if accepted:
                pubchem_to_bake[full_key] = smiles
            if position % 20 == 0 or position == len(unresolved_before_pubchem):
                print("PubChem exact-identity rescue {}/{} accepted={}".format(
                    position, len(unresolved_before_pubchem), len(pubchem_to_bake)
                ), flush=True)
            time.sleep(0.20)
    generated_pubchem, pubchem_generation_failures = bake(
        pubchem_to_bake, output_root, args.batch_size
    )
    resolved.update(generated_pubchem)
    source.update({key: "generated_exact_pubchem_smiles" for key in generated_pubchem})

    failed = {}
    for full_key in sorted(frame["full_inchikey"].map(norm).unique()):
        if full_key in resolved:
            continue
        if full_key in generation_failures:
            failed[full_key] = generation_failures[full_key]
        elif full_key in pubchem_generation_failures:
            failed[full_key] = pubchem_generation_failures[full_key]
        elif not smiles_audit[full_key]["n_smiles_candidates"]:
            failed[full_key] = "no canonical SMILES and no validated exact-key legacy tensor"
        else:
            failed[full_key] = "no SMILES candidate reproduced the full InChIKey and no exact legacy tensor"

    for full_key in sorted(frame["full_inchikey"].map(norm).unique()):
        check = smiles_audit[full_key]
        rows.append({
            "full_inchikey": full_key,
            "connectivity_key": full_key[:14],
            "status": "resolved" if full_key in resolved else "failed",
            "embedding_source": source.get(full_key, ""),
            "ligand_embedding_path": resolved.get(full_key, ""),
            "exact_smiles_found": int(check["exact_smiles_found"]),
            "n_smiles_candidates": int(check["n_smiles_candidates"]),
            "failure_reason": failed.get(full_key, ""),
        })
    manifest = pd.DataFrame(rows)
    pubchem_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pubchem_audit).to_csv(pubchem_path, sep="\t", index=False, compression="gzip")

    frame["full_inchikey"] = frame["full_inchikey"].map(norm)
    frame["ligand_embedding_path"] = frame["full_inchikey"].map(resolved).fillna("")
    frame["protein_embedding_path"] = frame["protein_embedding_path"].map(
        lambda value: normalize_path(value, args.project_root)
    )
    protein_failures = {}
    protein_shapes = {}
    for row in frame[["uniprot", "protein_embedding_path"]].drop_duplicates("uniprot").itertuples(index=False):
        uid = str(row.uniprot)
        path = str(row.protein_embedding_path)
        try:
            shape = validate_tensor(path, PROTEIN_DIM)
            indices = [int(value) for value in pocket_indices[uid]]
            if not indices or min(indices) < 0 or max(indices) >= shape[0]:
                raise ValueError("pocket index outside protein tensor")
            protein_shapes[uid] = shape
        except Exception as error:
            protein_failures[uid] = "{}: {}".format(type(error).__name__, error)

    ligand_ok = frame["ligand_embedding_path"].astype(str).ne("")
    protein_ok = ~frame["uniprot"].astype(str).isin(protein_failures)
    ready = frame[ligand_ok & protein_ok].copy()
    before_pair_filter = int(len(ready))
    two_label = ready.groupby("uniprot")["binary_label"].nunique()
    retained_proteins = set(two_label[two_label.eq(2)].index.astype(str))
    ready = ready[ready["uniprot"].astype(str).isin(retained_proteins)].copy()
    ready = ready.sort_values(
        ["family_fold", "family_component_id", "uniprot", "class_label", "full_inchikey"]
    ).reset_index(drop=True)
    ready.to_csv(model_ready_path, sep="\t", index=False, compression="gzip")
    manifest.to_csv(manifest_path, sep="\t", index=False, compression="gzip")

    report = {
        "status": "validated",
        "exact_identity_policy": (
            "No connectivity-key tensor fallback. A tensor must either have an exact full-InChIKey filename "
            "or be regenerated from a SMILES whose RDKit InChIKey exactly matches the row key."
        ),
        "source_rows": int(len(frame)),
        "source_proteins": int(frame["uniprot"].nunique()),
        "source_ligands": int(frame["full_inchikey"].nunique()),
        "model_ready_rows": int(len(ready)),
        "model_ready_proteins": int(ready["uniprot"].nunique()),
        "model_ready_ligands": int(ready["full_inchikey"].nunique()),
        "model_ready_allosteric": int(ready["binary_label"].eq(1).sum()),
        "model_ready_orthosteric": int(ready["binary_label"].eq(0).sum()),
        "rows_removed_for_missing_ligand_or_protein": int(len(frame) - before_pair_filter),
        "rows_removed_to_preserve_two_labels_per_protein": int(before_pair_filter - len(ready)),
        "proteins_removed": sorted(set(frame["uniprot"].astype(str)) - set(ready["uniprot"].astype(str))),
        "resolved_ligands_by_source": manifest.loc[manifest["status"].eq("resolved"), "embedding_source"].value_counts().to_dict(),
        "pubchem_exact_identity_rescue": {
            "attempted": int(len(pubchem_audit)),
            "accepted_smiles": int(sum(item["status"] == "accepted" for item in pubchem_audit)),
            "generated_tensors": int(len(generated_pubchem)),
            "audit_path": str(pubchem_path),
        },
        "failed_unique_ligands": int(len(failed)),
        "failed_ligands": failed,
        "invalid_exact_legacy_candidates": legacy_failures,
        "protein_failures": protein_failures,
        "all_retained_proteins_have_both_labels": bool(ready.groupby("uniprot")["binary_label"].nunique().eq(2).all()),
        "all_five_family_folds_retained": bool(set(ready["family_fold"]) == set(range(5))),
        "all_five_row_folds_retained": bool(set(ready["row_fold"]) == set(range(5))),
        "outputs": {
            "model_ready": str(model_ready_path),
            "ligand_manifest": str(manifest_path),
        },
    }
    conditions = [
        report["model_ready_proteins"] >= args.minimum_proteins,
        report["all_retained_proteins_have_both_labels"],
        report["all_five_family_folds_retained"],
        report["all_five_row_folds_retained"],
        set(ready["binary_label"]) == {0, 1},
    ]
    if not all(conditions):
        report["status"] = "failed"
    atomic_json(validation_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("exact GPU input validation failed")


if __name__ == "__main__":
    main()
