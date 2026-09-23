#!/usr/bin/env python3
"""Independently validate checkpoint-8 protein and Pfam-family folds."""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

FEATURES = DATA / "CHECKPOINT7_REFERENCE_FEATURES.tsv.gz"
PFAM_CACHE = DATA / "CHECKPOINT8_UNIPROT_PFAM_CACHE.json"
PROTEIN_ASSIGNMENTS = DATA / "CHECKPOINT8_PROTEIN_FAMILY_ASSIGNMENTS.tsv"
OBSERVATION_FOLDS = DATA / "CHECKPOINT8_OBSERVATION_FOLDS.tsv.gz"
FOLD_AUDIT = DATA / "CHECKPOINT8_FOLD_AUDIT.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT8_SPLIT_INPUT_HASHES.tsv"
SPLIT_SPEC = MANIFESTS / "CHECKPOINT8_SPLIT_SPEC.json"
BUILD = VALIDATION / "CHECKPOINT8_SPLIT_BUILD_SUMMARY.json"
OUTPUT = VALIDATION / "CHECKPOINT8_SPLIT_VALIDATION.json"

EXPECTED_FOLDS = {
    "protein_held_out": [
        (1171, 343, 828, 68, 10),
        (1160, 342, 818, 75, 18),
        (1155, 341, 814, 75, 20),
        (1157, 341, 816, 75, 18),
        (1155, 341, 814, 77, 17),
    ],
    "family_held_out": [
        (1184, 345, 839, 50, 7),
        (1147, 339, 808, 76, 21),
        (1158, 342, 816, 87, 13),
        (1149, 340, 809, 59, 16),
        (1160, 342, 818, 98, 26),
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def independent_components(records: dict):
    proteins = sorted(records)
    by_pfam = {}
    for protein in proteins:
        for pfam_id in records[protein]["pfam_ids"]:
            by_pfam.setdefault(pfam_id, set()).add(protein)
    neighbors = {protein: set() for protein in proteins}
    for members in by_pfam.values():
        for protein in members:
            neighbors[protein].update(members - {protein})
    groups = []
    unseen = set(proteins)
    while unseen:
        start = min(unseen)
        queue = deque([start])
        component = set()
        while queue:
            protein = queue.popleft()
            if protein in component:
                continue
            component.add(protein)
            queue.extend(sorted(neighbors[protein] - component))
        unseen -= component
        groups.append(sorted(component))
    return groups, by_pfam


def main() -> None:
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    if build.get("status") != "complete_pending_independent_validation":
        raise RuntimeError("checkpoint-8 split build status changed")
    for asset_id, record in build["outputs"].items():
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint-8 split output hash mismatch: {asset_id}")

    inputs = pd.read_csv(INPUT_HASHES, sep="\t", dtype=str, keep_default_na=False)
    if len(inputs) != 4 or inputs.asset_id.duplicated().any():
        raise RuntimeError("checkpoint-8 split input manifest changed")
    for row in inputs.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or str(path.stat().st_size) != row.size_bytes or sha256(path) != row.sha256:
            raise RuntimeError(f"checkpoint-8 split input hash mismatch: {row.asset_id}")

    features = pd.read_csv(FEATURES, sep="\t", low_memory=False)
    protein = pd.read_csv(PROTEIN_ASSIGNMENTS, sep="\t", keep_default_na=False)
    observations = pd.read_csv(OBSERVATION_FOLDS, sep="\t")
    audit = pd.read_csv(FOLD_AUDIT, sep="\t")
    cache = json.loads(PFAM_CACHE.read_text(encoding="utf-8"))
    spec = json.loads(SPLIT_SPEC.read_text(encoding="utf-8"))

    if len(features) != 5798 or features.observation_id.duplicated().any():
        raise RuntimeError("checkpoint-8 reference universe changed")
    if len(protein) != 370 or protein.uniprot.duplicated().any():
        raise RuntimeError("checkpoint-8 protein assignment universe changed")
    if len(observations) != len(features) or observations.observation_id.duplicated().any():
        raise RuntimeError("checkpoint-8 observation-fold universe changed")
    if set(features.observation_id) != set(observations.observation_id):
        raise RuntimeError("checkpoint-8 observation IDs changed")
    if set(features.uniprot) != set(protein.uniprot):
        raise RuntimeError("checkpoint-8 protein IDs changed")

    records = cache.get("records", {})
    if set(records) != set(protein.uniprot):
        raise RuntimeError("checkpoint-8 Pfam cache accession set changed")
    if cache.get("retrieval", {}).get("uniprot_release") != "2026_02":
        raise RuntimeError("checkpoint-8 UniProt release changed")
    if cache.get("retrieval", {}).get("verified_tls") is not True:
        raise RuntimeError("checkpoint-8 UniProt retrieval did not verify TLS")
    if any(record["annotation_status"] != "annotated" for record in records.values()):
        raise RuntimeError("checkpoint-8 reference contains a Pfam-unavailable protein")
    if any(record["accession_resolution"] != "exact_primary_accession" for record in records.values()):
        raise RuntimeError("checkpoint-8 reference contains a non-exact accession resolution")

    groups, by_pfam = independent_components(records)
    if len(by_pfam) != 481 or len(groups) != 157 or max(map(len, groups)) != 31:
        raise RuntimeError("checkpoint-8 independently rebuilt Pfam components changed")
    assigned_groups = sorted(
        sorted(group.uniprot.tolist())
        for _component, group in protein.groupby("family_component_id")
    )
    if sorted(groups) != assigned_groups:
        raise RuntimeError("checkpoint-8 assigned family components differ from shared-Pfam components")

    merged = features[["observation_id", "site_ligand_signature_id", "uniprot", "binary_label"]].merge(
        observations,
        on=["observation_id", "site_ligand_signature_id", "uniprot", "binary_label"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features):
        raise RuntimeError("checkpoint-8 observation fold metadata mismatch")
    protein_lookup = protein.set_index("uniprot")
    if not np.array_equal(
        merged.protein_fold.astype(int).to_numpy(),
        protein_lookup.loc[merged.uniprot, "protein_fold"].astype(int).to_numpy(),
    ):
        raise RuntimeError("checkpoint-8 observation protein folds differ from protein assignments")
    if not np.array_equal(
        merged.family_fold.astype(int).to_numpy(),
        protein_lookup.loc[merged.uniprot, "family_fold"].astype(int).to_numpy(),
    ):
        raise RuntimeError("checkpoint-8 observation family folds differ from protein assignments")
    if set(merged.protein_fold.astype(int)) != set(range(5)) or set(merged.family_fold.astype(int)) != set(range(5)):
        raise RuntimeError("checkpoint-8 fold labels changed")

    reconstructed_rows = []
    for regime, fold_column in (
        ("protein_held_out", "protein_fold"),
        ("family_held_out", "family_fold"),
    ):
        for fold in range(5):
            test = merged[merged[fold_column].astype(int) == fold]
            train = merged[merged[fold_column].astype(int) != fold]
            both = sum(group.binary_label.nunique() == 2 for _key, group in test.groupby("uniprot"))
            if set(train.uniprot) & set(test.uniprot):
                raise RuntimeError(f"checkpoint-8 protein leakage: {regime}/{fold}")
            if regime == "family_held_out" and set(train.family_component_id) & set(test.family_component_id):
                raise RuntimeError(f"checkpoint-8 family leakage: {fold}")
            reconstructed_rows.append({
                "regime": regime,
                "fold": fold,
                "test_rows": len(test),
                "test_allosteric_rows": int(test.binary_label.sum()),
                "test_orthosteric_rows": int((test.binary_label == 0).sum()),
                "test_proteins": int(test.uniprot.nunique()),
                "test_both_label_proteins": int(both),
            })
            observed_tuple = (
                len(test), int(test.binary_label.sum()), int((test.binary_label == 0).sum()),
                int(test.uniprot.nunique()), int(both),
            )
            if observed_tuple != EXPECTED_FOLDS[regime][fold]:
                raise RuntimeError(f"checkpoint-8 frozen fold count changed: {regime}/{fold}")

    reconstructed = pd.DataFrame(reconstructed_rows)
    check = audit.merge(
        reconstructed,
        on=["regime", "fold"],
        suffixes=("_reported", "_independent"),
        validate="one_to_one",
    )
    for column in (
        "test_rows", "test_allosteric_rows", "test_orthosteric_rows",
        "test_proteins", "test_both_label_proteins",
    ):
        if not np.array_equal(check[f"{column}_reported"], check[f"{column}_independent"]):
            raise RuntimeError(f"checkpoint-8 fold audit mismatch: {column}")

    if spec.get("recursive_directory_scan_performed") is not False:
        raise RuntimeError("checkpoint-8 bounded-I/O contract changed")
    if spec.get("fold_assignment", {}).get("model_scores_used") is not False:
        raise RuntimeError("checkpoint-8 fold assignment used model scores")
    if spec.get("family_held_out", {}).get("unit") != "connected component formed by exact shared Pfam accessions":
        raise RuntimeError("checkpoint-8 family definition changed")

    result = {
        "checkpoint": 8,
        "stage": "heldout_split_preparation",
        "status": "validated_heldout_splits",
        "reference_rows": len(features),
        "reference_proteins": len(protein),
        "uniprot_release": cache["retrieval"]["uniprot_release"],
        "uniprot_release_date": cache["retrieval"]["uniprot_release_date"],
        "pfam_annotated_proteins": len(protein),
        "unique_pfam_ids": len(by_pfam),
        "family_components": len(groups),
        "largest_family_component_proteins": max(map(len, groups)),
        "protein_fold_test_rows": EXPECTED_FOLDS["protein_held_out"],
        "family_fold_test_rows": EXPECTED_FOLDS["family_held_out"],
        "protein_leakage": 0,
        "family_component_leakage": 0,
        "recursive_directory_scan_performed": False,
        "validated_output_hashes": {
            asset_id: record["sha256"] for asset_id, record in build["outputs"].items()
        },
        "build_summary_sha256": sha256(BUILD),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
