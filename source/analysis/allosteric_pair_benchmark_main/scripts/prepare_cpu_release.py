#!/usr/bin/env python3
"""Freeze the target-chain-aligned main benchmark without scanning the project tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


N_FOLDS = 5
ROW_FOLD_SEED = 20260817


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    parser.add_argument("--shared-root", type=Path, default=Path("/shared_data/11.HS_allostery"))
    parser.add_argument("--package-dir", type=Path, default=None)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def stable_hash(*values):
    payload = "|".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_pfams(value):
    if value is None:
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip().startswith("PF")}


def assign_row_folds(frame):
    result = frame.copy()
    result["row_fold"] = -1
    for label, indices in result.groupby("binary_label").groups.items():
        ordered = sorted(
            indices,
            key=lambda index: stable_hash(ROW_FOLD_SEED, label, result.at[index, "main_row_id"]),
        )
        for position, index in enumerate(ordered):
            result.at[index, "row_fold"] = int(position % N_FOLDS)
    result["row_fold"] = result["row_fold"].astype(int)
    return result


def fold_summary(frame, column):
    rows = []
    for fold in range(N_FOLDS):
        subset = frame[frame[column].eq(fold)]
        rows.append({
            "fold": fold,
            "n_rows": int(len(subset)),
            "n_proteins": int(subset["uniprot"].nunique()),
            "n_components": int(subset["family_component_id"].nunique()),
            "n_ligands": int(subset["full_inchikey"].nunique()),
            "n_allosteric": int(subset["binary_label"].eq(1).sum()),
            "n_orthosteric": int(subset["binary_label"].eq(0).sum()),
        })
    return rows


def unseen_compound_counts(frame):
    rows = []
    for test_fold in range(N_FOLDS):
        val_fold = (test_fold + 1) % N_FOLDS
        train = frame[~frame["family_fold"].isin([test_fold, val_fold])]
        validation = frame[frame["family_fold"].eq(val_fold)]
        test = frame[frame["family_fold"].eq(test_fold)]
        train_groups = set(train["connectivity_key"].astype(str))
        validation_novel = validation[~validation["connectivity_key"].astype(str).isin(train_groups)]
        development_groups = train_groups | set(validation["connectivity_key"].astype(str))
        test_novel = test[~test["connectivity_key"].astype(str).isin(development_groups)]
        paired = set(
            test_novel.groupby("uniprot")["binary_label"].nunique().loc[lambda value: value.eq(2)].index.astype(str)
        )
        rows.append({
            "test_fold": test_fold,
            "validation_fold": val_fold,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "validation_unseen_compound_rows": int(len(validation_novel)),
            "test_rows": int(len(test)),
            "test_unseen_compound_rows": int(len(test_novel)),
            "test_unseen_compound_allosteric": int(test_novel["binary_label"].eq(1).sum()),
            "test_unseen_compound_orthosteric": int(test_novel["binary_label"].eq(0).sum()),
            "test_unseen_compound_proteins": int(test_novel["uniprot"].nunique()),
            "test_unseen_compound_two_label_proteins": int(len(paired)),
        })
    return rows


def main():
    args = parse_args()
    package = args.package_dir or args.project_root / "analysis/allosteric_pair_benchmark_main"
    source = args.project_root / "analysis/new_pairing_pfam_only_pilot/data/PFAM_ONLY_POCKET_READY.tsv.gz"
    component_source = args.project_root / "analysis/new_pairing_pfam_only_pilot/data/PFAM_ONLY_COMPONENTS.tsv"
    target_chain_source = package / "data/TARGET_CHAIN_SEQUENCES.tsv"
    mask_source = package / "data/POCKET_INDICES.json"
    alignment_validation = package / "validation/POCKET_ALIGNMENT_VALIDATION.json"
    pfam_source = args.shared_root / "14.Organized_input/uniprot_pfam_cache.json"
    for path in [source, component_source, target_chain_source, mask_source, alignment_validation, pfam_source]:
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_csv(source, sep="\t", low_memory=False)
    target_chains = pd.read_csv(target_chain_source, sep="\t", low_memory=False)
    target_chains["uniprot"] = target_chains["uniprot"].astype(str)
    if target_chains["uniprot"].duplicated().any():
        raise RuntimeError("target-chain table contains duplicate proteins")
    alignment_report = json.loads(alignment_validation.read_text(encoding="utf-8"))
    if alignment_report.get("status") != "validated":
        raise RuntimeError("target-chain alignment contract is not validated")
    required = {
        "pilot_row_id", "dataset_row_id", "pair_key", "uniprot", "full_inchikey",
        "connectivity_key", "canonical_smiles", "class_label", "binary_label",
        "protein_embedding_path", "family_component_id", "family_fold",
    }
    if required - set(frame.columns):
        raise RuntimeError("source is missing fields: {}".format(sorted(required - set(frame.columns))))
    frame["uniprot"] = frame["uniprot"].astype(str)
    frame = frame[frame["uniprot"].isin(set(target_chains["uniprot"]))].copy()
    if frame.empty:
        raise RuntimeError("no source rows survived the target-chain contract")
    target_by_uid = target_chains.set_index("uniprot")
    frame["protein_embedding_path"] = frame["uniprot"].map(target_by_uid["gpu_embedding_path"])
    frame["target_chain_pdb"] = frame["uniprot"].map(target_by_uid["mapped_pdb"])
    frame["target_chain_id"] = frame["uniprot"].map(target_by_uid["selected_chain"])
    frame["target_chain_length"] = frame["uniprot"].map(target_by_uid["target_chain_length"]).astype(int)
    frame["target_chain_sequence_sha256"] = frame["uniprot"].map(target_by_uid["sequence_sha256"])
    frame["representation_unit"] = "structure_matched_target_chain"
    if frame["protein_embedding_path"].isna().any():
        raise RuntimeError("a retained row lacks a target-chain embedding path")
    frame["main_row_id"] = frame["pilot_row_id"].astype(str)
    frame = assign_row_folds(frame)
    frame["reader_split_row"] = "Row-random"
    frame["reader_split_family"] = "Unseen-family"

    with mask_source.open(encoding="utf-8") as handle:
        raw_masks = json.load(handle)
    proteins = sorted(frame["uniprot"].astype(str).unique())
    masks = {uid: [int(value) for value in raw_masks.get(uid, [])] for uid in proteins}
    invalid_masks = {
        uid: values for uid, values in masks.items()
        if (
            not values or min(values) < 0 or len(values) != len(set(values))
            or max(values) >= int(target_by_uid.at[uid, "target_chain_length"])
        )
    }
    if invalid_masks:
        raise RuntimeError("invalid pocket masks for {} proteins".format(len(invalid_masks)))

    with pfam_source.open(encoding="utf-8") as handle:
        pfam_cache = json.load(handle)
    uid_pfams = {uid: parse_pfams(pfam_cache.get(uid, "")) for uid in proteins}
    if any(not values for values in uid_pfams.values()):
        raise RuntimeError("at least one benchmark protein lacks Pfam annotation")
    pfam_overlap = {}
    component_overlap = {}
    protein_overlap = {}
    for left in range(N_FOLDS):
        for right in range(left + 1, N_FOLDS):
            key = "{}_{}".format(left, right)
            left_rows = frame[frame["family_fold"].eq(left)]
            right_rows = frame[frame["family_fold"].eq(right)]
            left_uids = set(left_rows["uniprot"].astype(str))
            right_uids = set(right_rows["uniprot"].astype(str))
            protein_overlap[key] = len(left_uids & right_uids)
            component_overlap[key] = len(
                set(left_rows["family_component_id"].astype(str))
                & set(right_rows["family_component_id"].astype(str))
            )
            pfam_overlap[key] = len(
                {pfam for uid in left_uids for pfam in uid_pfams[uid]}
                & {pfam for uid in right_uids for pfam in uid_pfams[uid]}
            )

    protein_label_counts = frame.groupby("uniprot")["binary_label"].nunique()
    conditions = [
        frame["uniprot"].nunique() >= 430,
        set(frame["family_fold"]) == set(range(N_FOLDS)),
        set(frame["row_fold"]) == set(range(N_FOLDS)),
        protein_label_counts.eq(2).all(),
        not any(protein_overlap.values()),
        not any(component_overlap.values()),
        not any(pfam_overlap.values()),
    ]
    if not all(conditions):
        raise RuntimeError("CPU release contract failed")

    data_dir = package / "data"
    validation_dir = package / "validation"
    data_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = data_dir / "MAIN_COHORT.tsv.gz"
    masks_path = data_dir / "POCKET_INDICES.json"
    components_path = data_dir / "PFAM_COMPONENTS.tsv"
    validation_path = validation_dir / "CPU_INPUT_VALIDATION.json"
    frame.sort_values(["family_fold", "family_component_id", "uniprot", "class_label", "full_inchikey"]).to_csv(
        cohort_path, sep="\t", index=False, compression="gzip"
    )
    atomic_json(masks_path, masks)
    components = pd.read_csv(component_source, sep="\t", low_memory=False)
    components.to_csv(components_path, sep="\t", index=False)

    all_blank = (
        frame.assign(_smiles=frame["canonical_smiles"].fillna("").astype(str).str.strip())
        .groupby("full_inchikey")["_smiles"]
        .agg(lambda values: not any(values))
    )
    report = {
        "status": "validated",
        "reader_facing_evaluations": {
            "row_random": "Row-random",
            "unseen_family": "Unseen-family",
            "unseen_compound": "Unseen-family + unseen-compound",
        },
        "unseen_compound_definition": (
            "A test pair is retained only when its first InChIKey block (molecular connectivity) "
            "is absent from both training and validation."
        ),
        "rows": int(len(frame)),
        "proteins": int(frame["uniprot"].nunique()),
        "family_components": int(frame["family_component_id"].nunique()),
        "full_inchikeys": int(frame["full_inchikey"].nunique()),
        "connectivity_groups": int(frame["connectivity_key"].nunique()),
        "allosteric_rows": int(frame["binary_label"].eq(1).sum()),
        "orthosteric_rows": int(frame["binary_label"].eq(0).sum()),
        "all_blank_smiles_unique_ligands": int(all_blank.sum()),
        "family_fold_counts": fold_summary(frame, "family_fold"),
        "row_fold_counts": fold_summary(frame, "row_fold"),
        "prospective_unseen_compound_counts": unseen_compound_counts(frame),
        "protein_overlap_between_family_folds": protein_overlap,
        "component_overlap_between_family_folds": component_overlap,
        "pfam_overlap_between_family_folds": pfam_overlap,
        "inputs": {
            "cohort": {"path": str(source), "sha256": sha256(source)},
            "components": {"path": str(component_source), "sha256": sha256(component_source)},
            "target_chain_table": {"path": str(target_chain_source), "sha256": sha256(target_chain_source)},
            "pocket_masks": {"path": str(mask_source), "sha256": sha256(mask_source)},
            "pocket_alignment_validation": {"path": str(alignment_validation), "sha256": sha256(alignment_validation)},
            "pfam_cache": {"path": str(pfam_source), "sha256": sha256(pfam_source)},
        },
        "outputs": {
            "cohort": {"path": str(cohort_path), "sha256": sha256(cohort_path)},
            "components": {"path": str(components_path), "sha256": sha256(components_path)},
            "pocket_indices": {"path": str(masks_path), "sha256": sha256(masks_path)},
        },
    }
    atomic_json(validation_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
