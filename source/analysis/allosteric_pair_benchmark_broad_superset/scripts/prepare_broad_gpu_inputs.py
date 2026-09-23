#!/usr/bin/env python3
"""Resolve exact ligand tensors and freeze the Arm B model-ready superset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import time
from pathlib import Path

import pandas as pd


LIGAND_DIM = 512
PROTEIN_DIM = 1536
CLASS_TO_BINARY = {"orthosteric": 0, "allosteric": 1}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--disable-pubchem-rescue", action="store_true")
    return parser.parse_args()


def load_exact_helper(project_root):
    path = project_root / "analysis/allosteric_pair_benchmark_main/scripts/prepare_exact_gpu_inputs.py"
    spec = importlib.util.spec_from_file_location("frozen_exact_ligand_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def exact_existing_path(frame, full_key, helper, project_root):
    failures = {}
    values = frame.loc[
        frame["full_inchikey"].astype(str).eq(full_key), "ligand_embedding_path"
    ].dropna().astype(str)
    for raw_path in sorted(set(value for value in values if value.strip())):
        path = helper.normalize_path(raw_path, project_root)
        if full_key not in os.path.basename(path).upper():
            failures[path] = "filename does not contain the complete full InChIKey"
            continue
        try:
            helper.validate_tensor(path, LIGAND_DIM)
            return path, failures
        except Exception as error:
            failures[path] = "{}: {}".format(type(error).__name__, error)
    return "", failures


def main():
    args = parse_args()
    root = args.project_root
    package = root / "analysis/allosteric_pair_benchmark_broad_superset"
    source_path = package / "data/BROAD_ALIGNED_POOL.tsv.gz"
    core_path = package / "data/REFERENCE_ARM_A_MODEL_READY.tsv.gz"
    masks_path = package / "data/POCKET_INDICES.json"
    index_path = root / "17.paired_dataset/legacy/0.ligand_path_index.pkl"
    output_root = package / "gpu_cache/exact_ligand_embeddings"
    output_path = package / "gpu_cache/BROAD_MODEL_READY.tsv.gz"
    manifest_path = package / "gpu_cache/LIGAND_EMBEDDING_MANIFEST.tsv.gz"
    pubchem_path = package / "gpu_cache/PUBCHEM_EXACT_SMILES.tsv.gz"
    validation_path = package / "validation/GPU_INPUT_VALIDATION.json"
    helper = load_exact_helper(root)
    for path in [source_path, core_path, masks_path, index_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_csv(source_path, sep="\t", low_memory=False)
    core = pd.read_csv(core_path, sep="\t", low_memory=False)
    frame["full_inchikey"] = frame["full_inchikey"].map(helper.norm)
    frame["connectivity_key"] = frame["full_inchikey"].str[:14]
    frame["binary_label"] = frame["binary_label"].astype(int)
    core["full_inchikey"] = core["full_inchikey"].map(helper.norm)
    core["binary_label"] = core["binary_label"].astype(int)
    if not (frame["class_label"].map(CLASS_TO_BINARY).astype(int) == frame["binary_label"]).all():
        raise RuntimeError("class_label/binary_label mismatch in broad aligned pool")
    with masks_path.open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    with index_path.open("rb") as handle:
        raw_index = pickle.load(handle)
    index = {helper.norm(key): value for key, value in raw_index.items() if helper.norm(key)}
    del raw_index

    exact_smiles, smiles_audit = helper.exact_smiles_by_key(frame)
    resolved = {}
    source = {}
    failures_by_path = {}
    to_bake = {}
    unique_keys = sorted(frame["full_inchikey"].unique())
    for full_key in unique_keys:
        existing, existing_failures = exact_existing_path(frame, full_key, helper, root)
        failures_by_path.update({
            "{}|{}".format(full_key, path): error for path, error in existing_failures.items()
        })
        if existing:
            resolved[full_key] = existing
            source[full_key] = "frozen_arm_a_exact_tensor"
            continue
        cached = helper.validated_generated_cache(output_root, full_key)
        if cached:
            resolved[full_key] = cached
            source[full_key] = "generated_exact_cache"
            continue
        legacy, legacy_failures = helper.exact_legacy_path(index, full_key, root)
        failures_by_path.update({
            "{}|{}".format(full_key, path): error for path, error in legacy_failures.items()
        })
        if legacy:
            resolved[full_key] = legacy
            source[full_key] = "legacy_exact_full_inchikey_filename"
        elif full_key in exact_smiles:
            to_bake[full_key] = exact_smiles[full_key]

    generated, generation_failures = helper.bake(to_bake, output_root, args.batch_size)
    resolved.update(generated)
    source.update({key: "generated_exact_smiles" for key in generated})

    pubchem_audit = []
    pubchem_to_bake = {}
    if not args.disable_pubchem_rescue:
        unresolved = [key for key in unique_keys if key not in resolved]
        for position, full_key in enumerate(unresolved, 1):
            smiles, error = helper.pubchem_isomeric_smiles(full_key)
            observed, rdkit_error = helper.rdkit_key(smiles) if smiles else ("", "")
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
            if position % 20 == 0 or position == len(unresolved):
                print(
                    "PubChem exact rescue {}/{} accepted={}".format(
                        position, len(unresolved), len(pubchem_to_bake)
                    ),
                    flush=True,
                )
            time.sleep(0.20)
    generated_pubchem, pubchem_generation_failures = helper.bake(
        pubchem_to_bake, output_root, args.batch_size
    )
    resolved.update(generated_pubchem)
    source.update({key: "generated_exact_pubchem_smiles" for key in generated_pubchem})

    failed = {}
    for full_key in unique_keys:
        if full_key in resolved:
            continue
        if full_key in generation_failures:
            failed[full_key] = generation_failures[full_key]
        elif full_key in pubchem_generation_failures:
            failed[full_key] = pubchem_generation_failures[full_key]
        elif not smiles_audit[full_key]["n_smiles_candidates"]:
            failed[full_key] = "no canonical SMILES and no validated exact-key tensor"
        else:
            failed[full_key] = "no SMILES candidate reproduced the complete full InChIKey"

    manifest_rows = []
    for full_key in unique_keys:
        check = smiles_audit[full_key]
        manifest_rows.append({
            "full_inchikey": full_key,
            "connectivity_key": full_key[:14],
            "status": "resolved" if full_key in resolved else "failed",
            "embedding_source": source.get(full_key, ""),
            "ligand_embedding_path": resolved.get(full_key, ""),
            "exact_smiles_found": int(check["exact_smiles_found"]),
            "n_smiles_candidates": int(check["n_smiles_candidates"]),
            "failure_reason": failed.get(full_key, ""),
        })
    manifest = pd.DataFrame(manifest_rows)

    frame["ligand_embedding_path"] = frame["full_inchikey"].map(resolved).fillna("")
    frame["protein_embedding_path"] = frame["protein_embedding_path"].map(
        lambda value: helper.normalize_path(value, root)
    )
    protein_failures = {}
    protein_shapes = {}
    for row in frame[["uniprot", "protein_embedding_path"]].drop_duplicates("uniprot").itertuples(index=False):
        uid = str(row.uniprot)
        path = str(row.protein_embedding_path)
        try:
            shape = helper.validate_tensor(path, PROTEIN_DIM)
            indices = [int(value) for value in pocket_indices[uid]]
            if not indices or min(indices) < 0 or max(indices) >= shape[0]:
                raise ValueError("pocket index outside protein tensor")
            protein_shapes[uid] = shape
        except Exception as error:
            protein_failures[uid] = "{}: {}".format(type(error).__name__, error)

    ready = frame[
        frame["ligand_embedding_path"].astype(str).ne("")
        & ~frame["uniprot"].astype(str).isin(protein_failures)
    ].copy()
    ready = ready.sort_values(
        ["pool_membership", "uniprot", "class_label", "full_inchikey"]
    ).reset_index(drop=True)
    core_ids = set(core["main_row_id"].astype(str))
    retained_core = ready[ready["pool_membership"].eq("frozen_arm_a_core")]
    retained_core_ids = set(retained_core["main_row_id"].astype(str))
    missing_core_ids = sorted(core_ids - retained_core_ids)
    if ready["main_row_id"].duplicated().any():
        raise RuntimeError("model-ready broad pool contains duplicate row identifiers")

    output_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ready.to_csv(output_path, sep="\t", index=False, compression="gzip")
    manifest.to_csv(manifest_path, sep="\t", index=False, compression="gzip")
    pd.DataFrame(pubchem_audit).to_csv(pubchem_path, sep="\t", index=False, compression="gzip")

    report = {
        "status": "validated",
        "exact_identity_policy": (
            "No connectivity-key tensor fallback. Every accepted tensor has a complete full-InChIKey "
            "filename or was generated from a SMILES that reproduces that complete key with RDKit."
        ),
        "source_rows": int(len(frame)),
        "source_proteins": int(frame["uniprot"].nunique()),
        "model_ready_rows": int(len(ready)),
        "model_ready_proteins": int(ready["uniprot"].nunique()),
        "model_ready_allosteric": int(ready["binary_label"].eq(1).sum()),
        "model_ready_orthosteric": int(ready["binary_label"].eq(0).sum()),
        "arm_a_rows_retained": int(len(retained_core)),
        "arm_a_rows_expected": int(len(core)),
        "additional_rows_retained": int(ready["pool_membership"].eq("additional_input_ready_pair").sum()),
        "additional_proteins_retained": int(
            ready.loc[ready["pool_membership"].eq("additional_input_ready_pair"), "uniprot"].nunique()
        ),
        "missing_arm_a_row_ids": missing_core_ids,
        "failed_unique_ligands": int(len(failed)),
        "failed_ligands": failed,
        "invalid_tensor_candidates": failures_by_path,
        "protein_failures": protein_failures,
        "resolved_ligands_by_source": manifest.loc[
            manifest["status"].eq("resolved"), "embedding_source"
        ].value_counts().astype(int).to_dict(),
        "pubchem_exact_identity_rescue": {
            "attempted": int(len(pubchem_audit)),
            "accepted_smiles": int(sum(row["status"] == "accepted" for row in pubchem_audit)),
            "generated_tensors": int(len(generated_pubchem)),
        },
        "gates": {
            "arm_b_is_strict_superset_of_arm_a": not missing_core_ids and len(ready) > len(core),
            "all_labels_consistent": bool(
                (ready["class_label"].map(CLASS_TO_BINARY).astype(int) == ready["binary_label"]).all()
            ),
            "both_classes_present": set(ready["binary_label"]) == {0, 1},
            "all_protein_tensors_and_masks_valid": not protein_failures,
        },
        "outputs": {
            "model_ready": str(output_path),
            "ligand_manifest": str(manifest_path),
            "pubchem_audit": str(pubchem_path),
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "failed"
    atomic_json(validation_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("broad-superset GPU input validation failed")


if __name__ == "__main__":
    main()
