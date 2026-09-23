#!/usr/bin/env python3
"""Freeze the complete pre-audit orthosteric source ledger for BioLiP ranking."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
SOURCE = (
    ROOT
    / "analysis/allosteric_orthosteric_preprocessing_v1_participant_kegg_expansion/data"
    / "UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC_PARTICIPANT_KEGG_AUGMENTED.tsv.gz"
)
MAIN = ROOT / "analysis/allosteric_pair_benchmark_main/data/MAIN_COHORT.tsv.gz"
BROAD = (
    ROOT
    / "analysis/allosteric_pair_benchmark_broad_superset/gpu_cache/BROAD_MODEL_READY.tsv.gz"
)

EXPECTED_SOURCE_DATABASES = {
    "BRENDA",
    "GtoPdb",
    "KLIFS",
    "BioLiP",
    "UniProt",
    "PubChem",
    "RCSB",
    "ChEBI",
    "KEGG",
}
FULL_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def split_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def ordered_unique(values: list[object], split_semicolon: bool = False) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        tokens = split_tokens(value) if split_semicolon else ([] if pd.isna(value) else [str(value).strip()])
        for token in tokens:
            if token and token not in seen:
                seen.add(token)
                result.append(token)
    return result


def stable_pair_id(key: str) -> str:
    return "ORTHOREF_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def write_tsv(frame: pd.DataFrame, path: Path, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if compressed:
        with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=6, mtime=0) as raw:
            frame.to_csv(raw, sep="\t", index=False)
    else:
        frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def pair_keys(frame: pd.DataFrame) -> set[str]:
    return set(frame["uniprot"].astype(str) + "|" + frame["full_inchikey"].astype(str))


def benchmark_subset_audit(path: Path, master_keys: set[str]) -> dict[str, object]:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    orthosteric = frame.loc[frame["class_label"].eq("orthosteric")].copy()
    keys = pair_keys(orthosteric)
    missing = sorted(keys - master_keys)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows_total": int(len(frame)),
        "orthosteric_rows": int(len(orthosteric)),
        "orthosteric_pair_keys": int(len(keys)),
        "missing_from_checkpoint1_master": int(len(missing)),
        "missing_examples": missing[:10],
    }


def main() -> None:
    for path in (SOURCE, MAIN, BROAD):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = pd.read_csv(SOURCE, sep="\t", low_memory=False)
    required = {
        "dataset_row_id",
        "uniprot",
        "full_inchikey",
        "connectivity_key",
        "canonical_smiles",
        "fingerprint_eligible",
        "class_label",
        "binary_label",
        "source_databases",
        "source_lanes",
        "source_pair_ids",
        "evidence_subtypes",
        "n_source_pair_rows",
        "n_evidence_rows",
        "class_conflict_status",
        "decoy_included",
        "claim_limit",
    }
    missing_columns = sorted(required - set(source.columns))
    if missing_columns:
        raise RuntimeError(f"augmented source is missing columns: {missing_columns}")

    label_counts = source["class_label"].value_counts(dropna=False).to_dict()
    raw_orthosteric = source.loc[source["class_label"].eq("orthosteric")].copy()
    raw_orthosteric["uniprot"] = raw_orthosteric["uniprot"].astype(str).str.strip()
    raw_orthosteric["full_inchikey"] = raw_orthosteric["full_inchikey"].astype(str).str.strip()
    raw_orthosteric["connectivity_key"] = raw_orthosteric["connectivity_key"].astype(str).str.strip()
    raw_orthosteric["orthosteric_pair_key"] = (
        raw_orthosteric["uniprot"] + "|" + raw_orthosteric["full_inchikey"]
    )
    duplicate_input_rows = raw_orthosteric.loc[
        raw_orthosteric["orthosteric_pair_key"].duplicated(keep=False)
    ].copy()

    invalid = {
        "duplicate_dataset_row_id": int(raw_orthosteric["dataset_row_id"].duplicated().sum()),
        "blank_uniprot": int(raw_orthosteric["uniprot"].eq("").sum()),
        "invalid_full_inchikey": int((~raw_orthosteric["full_inchikey"].map(lambda x: bool(FULL_INCHIKEY_RE.fullmatch(x)))).sum()),
        "connectivity_key_mismatch": int((raw_orthosteric["connectivity_key"] != raw_orthosteric["full_inchikey"].str[:14]).sum()),
        "binary_label_not_zero": int((pd.to_numeric(raw_orthosteric["binary_label"], errors="coerce") != 0).sum()),
        "class_conflict_not_nonconflicting": int((raw_orthosteric["class_conflict_status"] != "nonconflicting").sum()),
        "decoy_included_true": int(raw_orthosteric["decoy_included"].astype(str).str.lower().eq("true").sum()),
        "blank_source_databases": int(raw_orthosteric["source_databases"].fillna("").astype(str).str.strip().eq("").sum()),
    }
    if any(invalid.values()):
        raise RuntimeError(f"checkpoint-1 pair identity/label contract failed: {invalid}")

    # The augmented source unexpectedly contains nine exact-pair duplicates.
    # Preserve the input rows verbatim, then merge evidence deterministically at
    # the natural (UniProt, full InChIKey) pair grain.  No audit disposition is
    # applied here; checkpoint 2 owns that decision.
    master_rows: list[dict[str, object]] = []
    for key, group in raw_orthosteric.groupby("orthosteric_pair_key", sort=True):
        group = group.sort_values("dataset_row_id")
        first = group.iloc[0]
        smiles = ordered_unique(group["canonical_smiles"].tolist())
        databases = ordered_unique(group["source_databases"].tolist(), split_semicolon=True)
        lanes = ordered_unique(group["source_lanes"].tolist(), split_semicolon=True)
        source_pair_ids = ordered_unique(group["source_pair_ids"].tolist(), split_semicolon=True)
        evidence_subtypes = ordered_unique(group["evidence_subtypes"].tolist(), split_semicolon=True)
        claim_limits = ordered_unique(group["claim_limit"].tolist())
        master_rows.append(
            {
                "checkpoint1_pair_id": stable_pair_id(key),
                "orthosteric_pair_key": key,
                "uniprot": first["uniprot"],
                "full_inchikey": first["full_inchikey"],
                "connectivity_key": first["connectivity_key"],
                "canonical_smiles": smiles[0] if smiles else "",
                "canonical_smiles_candidates": ";".join(smiles),
                "canonical_smiles_count": len(smiles),
                "fingerprint_eligible": bool(
                    group["fingerprint_eligible"].astype(str).str.lower().eq("true").any()
                ),
                "class_label": "orthosteric",
                "binary_label": 0,
                "source_databases": ";".join(databases),
                "source_lanes": ";".join(lanes),
                "source_pair_ids": ";".join(source_pair_ids),
                "evidence_subtypes": ";".join(evidence_subtypes),
                "n_source_pair_rows": int(pd.to_numeric(group["n_source_pair_rows"], errors="coerce").fillna(0).sum()),
                "n_evidence_rows": int(pd.to_numeric(group["n_evidence_rows"], errors="coerce").fillna(0).sum()),
                "input_dataset_row_ids": ";".join(group["dataset_row_id"].astype(str)),
                "input_row_count": int(len(group)),
                "protein_control_status": ";".join(ordered_unique(group["protein_control_status"].tolist())),
                "ligand_control_status": ";".join(ordered_unique(group["ligand_control_status"].tolist())),
                "class_conflict_status": "nonconflicting",
                "decoy_included": False,
                "claim_limit": " || ".join(claim_limits),
                "checkpoint1_status": "frozen_pre_manual_multisite_audit",
            }
        )
    master = pd.DataFrame(master_rows)
    if master["orthosteric_pair_key"].duplicated().any():
        raise RuntimeError("pair merge failed to produce unique orthosteric keys")

    incidence_rows: list[dict[str, object]] = []
    for row in raw_orthosteric.itertuples(index=False):
        for database in split_tokens(row.source_databases):
            incidence_rows.append(
                {
                    "orthosteric_pair_key": row.orthosteric_pair_key,
                    "dataset_row_id": row.dataset_row_id,
                    "uniprot": row.uniprot,
                    "full_inchikey": row.full_inchikey,
                    "connectivity_key": row.connectivity_key,
                    "source_database": database,
                    "source_lanes_verbatim": row.source_lanes,
                    "source_pair_ids_verbatim": row.source_pair_ids,
                    "evidence_subtypes_verbatim": row.evidence_subtypes,
                    "note": "Source databases are exploded independently; semicolon fields are retained verbatim and are not assumed to align positionally.",
                }
            )
    incidence = pd.DataFrame(incidence_rows)
    incidence = (
        incidence.groupby(
            [
                "orthosteric_pair_key",
                "uniprot",
                "full_inchikey",
                "connectivity_key",
                "source_database",
            ],
            as_index=False,
        )
        .agg(
            dataset_row_id=("dataset_row_id", lambda x: ";".join(ordered_unique(list(x)))),
            source_lanes_verbatim=("source_lanes_verbatim", lambda x: ";".join(ordered_unique(list(x), split_semicolon=True))),
            source_pair_ids_verbatim=("source_pair_ids_verbatim", lambda x: ";".join(ordered_unique(list(x), split_semicolon=True))),
            evidence_subtypes_verbatim=("evidence_subtypes_verbatim", lambda x: ";".join(ordered_unique(list(x), split_semicolon=True))),
            note=("note", "first"),
        )
    )
    observed_sources = set(incidence["source_database"].astype(str))
    if observed_sources != EXPECTED_SOURCE_DATABASES:
        raise RuntimeError(
            "source database contract failed: "
            f"expected={sorted(EXPECTED_SOURCE_DATABASES)}, observed={sorted(observed_sources)}"
        )
    if incidence.duplicated(["orthosteric_pair_key", "source_database"]).any():
        raise RuntimeError("duplicate pair/source incidence rows")

    source_counts = (
        incidence.groupby("source_database", as_index=False)
        .agg(
            pair_rows=("orthosteric_pair_key", "nunique"),
            unique_proteins=("uniprot", "nunique"),
            unique_full_inchikeys=("full_inchikey", "nunique"),
            unique_connectivity_keys=("connectivity_key", "nunique"),
        )
        .sort_values(["pair_rows", "source_database"], ascending=[False, True])
    )
    combinations = (
        master.groupby("source_databases", as_index=False)
        .agg(
            pair_rows=("orthosteric_pair_key", "size"),
            unique_proteins=("uniprot", "nunique"),
            unique_full_inchikeys=("full_inchikey", "nunique"),
        )
        .sort_values(["pair_rows", "source_databases"], ascending=[False, True])
    )

    master_keys = set(master["orthosteric_pair_key"])
    benchmark_audits = {
        "main": benchmark_subset_audit(MAIN, master_keys),
        "broad_superset": benchmark_subset_audit(BROAD, master_keys),
    }
    if any(item["missing_from_checkpoint1_master"] for item in benchmark_audits.values()):
        raise RuntimeError(f"a benchmark orthosteric pair is absent from the master: {benchmark_audits}")

    master = master.sort_values(["uniprot", "full_inchikey", "checkpoint1_pair_id"]).reset_index(drop=True)
    incidence = incidence.sort_values(
        ["source_database", "uniprot", "full_inchikey", "dataset_row_id"]
    ).reset_index(drop=True)

    data_dir = PACKAGE / "data"
    report_dir = PACKAGE / "reports"
    validation_dir = PACKAGE / "validation"
    manifest_dir = PACKAGE / "manifests"
    master_path = data_dir / "ORTHOSTERIC_PAIR_MASTER_PRE_AUDIT.tsv.gz"
    raw_rows_path = data_dir / "ORTHOSTERIC_INPUT_ROWS_PRE_AUDIT.tsv.gz"
    incidence_path = data_dir / "ORTHOSTERIC_SOURCE_INCIDENCE_PRE_AUDIT.tsv.gz"
    duplicate_path = data_dir / "CHECKPOINT1_DUPLICATE_INPUT_PAIR_ROWS.tsv"
    counts_path = data_dir / "CHECKPOINT1_SOURCE_COUNTS.tsv"
    combinations_path = data_dir / "CHECKPOINT1_SOURCE_COMBINATIONS.tsv"
    write_tsv(master, master_path, compressed=True)
    write_tsv(
        raw_orthosteric.sort_values(["orthosteric_pair_key", "dataset_row_id"]).reset_index(drop=True),
        raw_rows_path,
        compressed=True,
    )
    write_tsv(incidence, incidence_path, compressed=True)
    write_tsv(
        duplicate_input_rows.sort_values(["orthosteric_pair_key", "dataset_row_id"]).reset_index(drop=True),
        duplicate_path,
    )
    write_tsv(source_counts, counts_path)
    write_tsv(combinations, combinations_path)

    input_hashes = pd.DataFrame(
        [
            {"role": "augmented_pair_universe", "path": str(SOURCE), "sha256": sha256(SOURCE)},
            {"role": "main_benchmark_subset_check", "path": str(MAIN), "sha256": sha256(MAIN)},
            {"role": "broad_benchmark_subset_check", "path": str(BROAD), "sha256": sha256(BROAD)},
        ]
    )
    hashes_path = manifest_dir / "CHECKPOINT1_INPUT_HASHES.tsv"
    write_tsv(input_hashes, hashes_path)

    validation = {
        "status": "validated",
        "checkpoint": 1,
        "scope": "complete pre-manual-audit orthosteric source freeze",
        "manual_biolip_multisite_audit_applied": False,
        "source_rows_total": int(len(source)),
        "source_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "orthosteric_input_rows": int(len(raw_orthosteric)),
        "orthosteric_pair_rows": int(len(master)),
        "duplicate_input_pair_keys": int(duplicate_input_rows["orthosteric_pair_key"].nunique()),
        "duplicate_input_rows": int(len(duplicate_input_rows)),
        "duplicate_input_excess_rows": int(len(raw_orthosteric) - len(master)),
        "orthosteric_unique_proteins": int(master["uniprot"].nunique()),
        "orthosteric_unique_full_inchikeys": int(master["full_inchikey"].nunique()),
        "orthosteric_unique_connectivity_keys": int(master["connectivity_key"].nunique()),
        "fingerprint_eligible_rows": int(master["fingerprint_eligible"].astype(str).str.lower().eq("true").sum()),
        "fingerprint_unresolved_rows": int(master["fingerprint_eligible"].astype(str).str.lower().ne("true").sum()),
        "source_database_tokens": sorted(observed_sources),
        "source_incidence_rows": int(len(incidence)),
        "identity_and_label_anomalies": invalid,
        "benchmark_subset_audits": benchmark_audits,
        "inputs": {row["role"]: {"path": row["path"], "sha256": row["sha256"]} for row in input_hashes.to_dict("records")},
        "outputs": {},
    }
    for key, path in {
        "pair_master": master_path,
        "raw_input_rows": raw_rows_path,
        "source_incidence": incidence_path,
        "duplicate_input_rows": duplicate_path,
        "source_counts": counts_path,
        "source_combinations": combinations_path,
        "input_hashes": hashes_path,
    }.items():
        validation["outputs"][key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    validation_path = validation_dir / "CHECKPOINT1_VALIDATION.json"
    atomic_json(validation_path, validation)

    source_table = "\n".join(
        f"| {row.source_database} | {int(row.pair_rows):,} | {int(row.unique_proteins):,} | "
        f"{int(row.unique_full_inchikeys):,} | {int(row.unique_connectivity_keys):,} |"
        for row in source_counts.itertuples(index=False)
    )
    report = f"""# Checkpoint 1 — orthosteric source freeze

## Result

PASS.  The complete pre-manual-audit orthosteric pair universe has been frozen
from the participant/KEGG-augmented source without modifying any upstream file.

- augmented source rows: **{len(source):,}**
- source labels: **{label_counts}**
- orthosteric input rows: **{len(raw_orthosteric):,}**
- frozen unique orthosteric pairs: **{len(master):,}**
- duplicate exact-pair keys / excess input rows: **{validation['duplicate_input_pair_keys']} / {validation['duplicate_input_excess_rows']}**
- proteins: **{master['uniprot'].nunique():,}**
- full InChIKeys: **{master['full_inchikey'].nunique():,}**
- connectivity keys: **{master['connectivity_key'].nunique():,}**
- source incidence rows: **{len(incidence):,}**
- fingerprint eligible / unresolved: **{validation['fingerprint_eligible_rows']:,} / {validation['fingerprint_unresolved_rows']:,}**

Manual BioLiP same-chemical multisite dispositions are intentionally **not yet
applied**.  That is checkpoint 2.

## Source composition

Counts overlap because one pair can carry multiple source databases.

| source database | pair rows | proteins | full InChIKeys | connectivity keys |
|---|---:|---:|---:|---:|
{source_table}

## Validation

- duplicate dataset row IDs: **{invalid['duplicate_dataset_row_id']}**
- duplicate input exact-pair keys: **{validation['duplicate_input_pair_keys']}** (merged with all provenance retained)
- invalid full InChIKeys: **{invalid['invalid_full_inchikey']}**
- connectivity-key mismatches: **{invalid['connectivity_key_mismatch']}**
- class conflicts / decoys: **{invalid['class_conflict_not_nonconflicting']} / {invalid['decoy_included_true']}**
- main benchmark missing from master: **{benchmark_audits['main']['missing_from_checkpoint1_master']}**
- broad benchmark missing from master: **{benchmark_audits['broad_superset']['missing_from_checkpoint1_master']}**

## Data shape

`ORTHOSTERIC_PAIR_MASTER_PRE_AUDIT.tsv.gz` is one row per exact
`(UniProt, full InChIKey)` pair and preserves the upstream source/lane/evidence
fields.  The nine duplicated input keys are merged deterministically while all
input row IDs, source IDs, source lanes, evidence subtypes, SMILES candidates,
and claim limits are retained.

`ORTHOSTERIC_INPUT_ROWS_PRE_AUDIT.tsv.gz` is the verbatim 18,590-row
orthosteric slice of the augmented input, with only the natural pair key added.
`CHECKPOINT1_DUPLICATE_INPUT_PAIR_ROWS.tsv` exposes all duplicate source rows.

`ORTHOSTERIC_SOURCE_INCIDENCE_PRE_AUDIT.tsv.gz` is one row per pair and source
database token.  Semicolon-separated lane, source-pair-ID, and evidence fields
are retained verbatim because those lists are not guaranteed to align
positionally with the database list.
"""
    report_path = report_dir / "CHECKPOINT1_ORTHOSTERIC_SOURCE_FREEZE.md"
    atomic_text(report_path, report)
    validation["outputs"]["report"] = {
        "path": str(report_path),
        "bytes": report_path.stat().st_size,
        "sha256": sha256(report_path),
    }
    atomic_json(validation_path, validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
