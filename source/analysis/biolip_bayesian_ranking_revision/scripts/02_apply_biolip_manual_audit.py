#!/usr/bin/env python3
"""Apply the frozen BioLiP manual audit to the orthosteric pair master.

The audit decision is propagated at the natural (UniProt, full InChIKey)
pair grain.  Therefore, a pair excluded or held because of the BioLiP audit is
also removed from the active reference when the same pair entered through a
second provenance row (for example UniProt/ChEBI/KEGG).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"

PAIR_MASTER_PRE = DATA / "ORTHOSTERIC_PAIR_MASTER_PRE_AUDIT.tsv.gz"
RAW_ROWS_PRE = DATA / "ORTHOSTERIC_INPUT_ROWS_PRE_AUDIT.tsv.gz"
SOURCE_INCIDENCE_PRE = DATA / "ORTHOSTERIC_SOURCE_INCIDENCE_PRE_AUDIT.tsv.gz"
CHECKPOINT1_VALIDATION = PACKAGE / "validation/CHECKPOINT1_VALIDATION.json"

AUDIT_ROOT = (
    ROOT
    / "analysis/allosteric_orthosteric_preprocessing_v1_strict_pair_multisite_audit/data"
)
AUDIT_LEDGER = AUDIT_ROOT / "STRICT_PAIR_MULTISITE_AUDIT_LEDGER.tsv"
AUDIT_EXCLUDED_HELD = AUDIT_ROOT / "EXCLUDED_OR_HELD_MULTISITE_PAIRS.tsv"
AUDIT_FILTERED_RETAINED = AUDIT_ROOT / "FILTERED_STRICT_ORTHOSTERIC_PAIRS.tsv"
AUDIT_COUNTS = AUDIT_ROOT / "AUDIT_COUNT_SUMMARY.tsv"

MAIN = ROOT / "analysis/allosteric_pair_benchmark_main/data/MAIN_COHORT.tsv.gz"
BROAD = (
    ROOT
    / "analysis/allosteric_pair_benchmark_broad_superset/gpu_cache/BROAD_MODEL_READY.tsv.gz"
)

RETAIN = "retain_strict_orthosteric"
EXCLUDE = "exclude_same_chemical_distinct_functional_site"
HOLD = "hold_same_chemical_multisite_uncertain"
EXPECTED_DISPOSITIONS = {RETAIN: 247, EXCLUDE: 13, HOLD: 6}


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


def write_tsv(frame: pd.DataFrame, path: Path, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if compressed:
        with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=6, mtime=0) as raw:
            frame.to_csv(raw, sep="\t", index=False)
    else:
        frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def split_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def pair_key(frame: pd.DataFrame) -> pd.Series:
    return frame["uniprot"].astype(str).str.strip() + "|" + frame["full_inchikey"].astype(str).str.strip()


def benchmark_subset_audit(path: Path, active_keys: set[str], quarantine_keys: set[str]) -> dict[str, object]:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    orthosteric = frame.loc[frame["class_label"].eq("orthosteric")].copy()
    keys = set(pair_key(orthosteric))
    missing = sorted(keys - active_keys)
    quarantined = sorted(keys & quarantine_keys)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows_total": int(len(frame)),
        "orthosteric_rows": int(len(orthosteric)),
        "orthosteric_pair_keys": int(len(keys)),
        "missing_from_audited_master": int(len(missing)),
        "quarantined_key_overlap": int(len(quarantined)),
        "missing_examples": missing[:10],
    }


def main() -> None:
    inputs = [
        PAIR_MASTER_PRE,
        RAW_ROWS_PRE,
        SOURCE_INCIDENCE_PRE,
        CHECKPOINT1_VALIDATION,
        AUDIT_LEDGER,
        AUDIT_EXCLUDED_HELD,
        AUDIT_FILTERED_RETAINED,
        AUDIT_COUNTS,
        MAIN,
        BROAD,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint1 = json.loads(CHECKPOINT1_VALIDATION.read_text())
    if checkpoint1.get("status") != "validated" or checkpoint1.get("checkpoint") != 1:
        raise RuntimeError("checkpoint 1 is not validated")
    for record in checkpoint1.get("outputs", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint-1 output hash mismatch: {path}")

    master = pd.read_csv(PAIR_MASTER_PRE, sep="\t", low_memory=False)
    raw = pd.read_csv(RAW_ROWS_PRE, sep="\t", low_memory=False)
    incidence = pd.read_csv(SOURCE_INCIDENCE_PRE, sep="\t", low_memory=False)
    audit = pd.read_csv(AUDIT_LEDGER, sep="\t", low_memory=False)
    excluded_held = pd.read_csv(AUDIT_EXCLUDED_HELD, sep="\t", low_memory=False)
    filtered = pd.read_csv(AUDIT_FILTERED_RETAINED, sep="\t", low_memory=False)

    if len(master) != checkpoint1["orthosteric_pair_rows"]:
        raise RuntimeError("checkpoint-1 master row count changed")
    if len(raw) != checkpoint1["orthosteric_input_rows"]:
        raise RuntimeError("checkpoint-1 raw input row count changed")
    if len(incidence) != checkpoint1["source_incidence_rows"]:
        raise RuntimeError("checkpoint-1 source incidence row count changed")
    if master["orthosteric_pair_key"].duplicated().any():
        raise RuntimeError("pre-audit master key is not unique")
    if incidence.duplicated(["orthosteric_pair_key", "source_database"]).any():
        raise RuntimeError("pre-audit source incidence key is not unique")

    disposition_counts = audit["manual_disposition"].value_counts().to_dict()
    if disposition_counts != EXPECTED_DISPOSITIONS:
        raise RuntimeError(
            f"manual audit disposition contract changed: {disposition_counts}"
        )
    if len(audit) != 266 or audit["source_pair_id"].duplicated().any():
        raise RuntimeError("manual audit must contain 266 unique source_pair_ids")
    if set(excluded_held["source_pair_id"]) != set(
        audit.loc[audit["manual_disposition"].isin([EXCLUDE, HOLD]), "source_pair_id"]
    ):
        raise RuntimeError("excluded/held audit export does not match the full ledger")
    if set(filtered["source_pair_id"]) != set(
        audit.loc[audit["manual_disposition"].eq(RETAIN), "source_pair_id"]
    ):
        raise RuntimeError("retained audit export does not match the full ledger")

    # The 266 manually audited source pairs must map one-to-one to the 266
    # BioLiP-origin input rows.  Do not infer identity from protein or CCD.
    biolip_raw = raw.loc[
        raw["source_databases"].map(lambda value: "BioLiP" in split_tokens(value))
    ].copy()
    if len(biolip_raw) != 266:
        raise RuntimeError(f"expected 266 BioLiP input rows, observed {len(biolip_raw)}")

    audit_id_set = set(audit["source_pair_id"].astype(str))
    mapped_source_ids: list[str] = []
    for value in biolip_raw["source_pair_ids"]:
        matches = [token for token in split_tokens(value) if token in audit_id_set]
        if len(matches) != 1:
            raise RuntimeError(
                f"BioLiP input row does not map to exactly one audit source_pair_id: {value} -> {matches}"
            )
        mapped_source_ids.append(matches[0])
    biolip_raw["audit_source_pair_id"] = mapped_source_ids
    if biolip_raw["audit_source_pair_id"].duplicated().any():
        raise RuntimeError("multiple BioLiP input rows mapped to one audit source_pair_id")
    if set(biolip_raw["audit_source_pair_id"]) != audit_id_set:
        raise RuntimeError("audit/BioLiP input source_pair_id sets differ")

    identity = biolip_raw[
        [
            "audit_source_pair_id",
            "orthosteric_pair_key",
            "dataset_row_id",
            "uniprot",
            "full_inchikey",
            "connectivity_key",
            "canonical_smiles",
        ]
    ].rename(
        columns={
            "dataset_row_id": "biolip_input_dataset_row_id",
            "canonical_smiles": "checkpoint1_canonical_smiles",
        }
    )
    crosswalk = audit.merge(
        identity,
        left_on="source_pair_id",
        right_on="audit_source_pair_id",
        how="left",
        validate="one_to_one",
    )
    if crosswalk["orthosteric_pair_key"].isna().any():
        raise RuntimeError("an audit row failed to map to a checkpoint-1 pair")
    if (crosswalk["uniprot_x"].astype(str) != crosswalk["uniprot_y"].astype(str)).any():
        raise RuntimeError("audit/BioLiP UniProt mismatch")
    crosswalk = crosswalk.rename(columns={"uniprot_x": "uniprot"}).drop(columns=["uniprot_y"])

    # Independently verify the retained export's exact chemical identity.
    retained_identity = crosswalk.loc[crosswalk["manual_disposition"].eq(RETAIN), [
        "source_pair_id", "uniprot", "full_inchikey", "connectivity_key"
    ]].merge(
        filtered[["source_pair_id", "uniprot", "full_inchikey", "connectivity_key"]],
        on="source_pair_id",
        suffixes=("_crosswalk", "_filtered"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    identity_mismatch = retained_identity.loc[
        retained_identity["_merge"].ne("both")
        | retained_identity["uniprot_crosswalk"].ne(retained_identity["uniprot_filtered"])
        | retained_identity["full_inchikey_crosswalk"].ne(retained_identity["full_inchikey_filtered"])
        | retained_identity["connectivity_key_crosswalk"].ne(retained_identity["connectivity_key_filtered"])
    ]
    if len(identity_mismatch):
        raise RuntimeError(f"retained exact-identity mismatch: {len(identity_mismatch)}")

    master_context = master[
        [
            "orthosteric_pair_key",
            "checkpoint1_pair_id",
            "source_databases",
            "source_lanes",
            "source_pair_ids",
            "input_dataset_row_ids",
            "input_row_count",
        ]
    ].rename(
        columns={
            "source_databases": "checkpoint1_all_source_databases",
            "source_lanes": "checkpoint1_all_source_lanes",
            "source_pair_ids": "checkpoint1_all_source_pair_ids",
            "input_dataset_row_ids": "checkpoint1_all_input_dataset_row_ids",
            "input_row_count": "checkpoint1_pair_input_row_count",
        }
    )
    crosswalk = crosswalk.merge(
        master_context,
        on="orthosteric_pair_key",
        how="left",
        validate="one_to_one",
    )
    if crosswalk["checkpoint1_pair_id"].isna().any():
        raise RuntimeError("manual audit pair absent from checkpoint-1 master")

    crosswalk["pairwide_action"] = crosswalk["manual_disposition"].map(
        {
            RETAIN: "retain_in_orthosteric_reference",
            EXCLUDE: "exclude_from_orthosteric_reference",
            HOLD: "quarantine_from_orthosteric_reference",
        }
    )
    crosswalk["has_second_input_provenance_row"] = (
        pd.to_numeric(crosswalk["checkpoint1_pair_input_row_count"], errors="raise") > 1
    )
    crosswalk["checkpoint2_reference_eligible"] = crosswalk["manual_disposition"].eq(RETAIN)

    bad_crosswalk = crosswalk.loc[crosswalk["manual_disposition"].isin([EXCLUDE, HOLD])].copy()
    quarantine_keys = set(bad_crosswalk["orthosteric_pair_key"])
    retain_audit_keys = set(
        crosswalk.loc[crosswalk["manual_disposition"].eq(RETAIN), "orthosteric_pair_key"]
    )
    if len(quarantine_keys) != 19 or len(retain_audit_keys) != 247:
        raise RuntimeError("manual audit pair-key counts are not 19/247")
    if quarantine_keys & retain_audit_keys:
        raise RuntimeError("audit retain and quarantine key sets overlap")

    disposition_by_key = crosswalk.set_index("orthosteric_pair_key")["manual_disposition"]
    action_by_key = crosswalk.set_index("orthosteric_pair_key")["pairwide_action"]
    audit_id_by_key = crosswalk.set_index("orthosteric_pair_key")["audit_id"]

    def annotate(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["checkpoint2_audit_scope"] = result["orthosteric_pair_key"].map(
            lambda key: "biolip_manual_pair_audit" if key in disposition_by_key.index else "not_a_biolip_manual_audit_pair"
        )
        result["checkpoint2_audit_id"] = result["orthosteric_pair_key"].map(audit_id_by_key).fillna("")
        result["checkpoint2_manual_disposition"] = (
            result["orthosteric_pair_key"].map(disposition_by_key).fillna("not_applicable")
        )
        result["checkpoint2_pairwide_action"] = (
            result["orthosteric_pair_key"].map(action_by_key).fillna("retain_not_audit_target")
        )
        result["checkpoint2_reference_eligible"] = ~result["orthosteric_pair_key"].isin(quarantine_keys)
        return result

    master_annotated = annotate(master)
    raw_annotated = annotate(raw)
    incidence_annotated = annotate(incidence)

    active_master = master_annotated.loc[master_annotated["checkpoint2_reference_eligible"]].copy()
    quarantine_master = master_annotated.loc[~master_annotated["checkpoint2_reference_eligible"]].copy()
    active_raw = raw_annotated.loc[raw_annotated["checkpoint2_reference_eligible"]].copy()
    quarantine_raw = raw_annotated.loc[~raw_annotated["checkpoint2_reference_eligible"]].copy()
    active_incidence = incidence_annotated.loc[incidence_annotated["checkpoint2_reference_eligible"]].copy()
    quarantine_incidence = incidence_annotated.loc[~incidence_annotated["checkpoint2_reference_eligible"]].copy()

    active_keys = set(active_master["orthosteric_pair_key"])
    if active_keys & quarantine_keys:
        raise RuntimeError("quarantined pair leaked into active master")
    if set(quarantine_master["orthosteric_pair_key"]) != quarantine_keys:
        raise RuntimeError("quarantine master does not contain exactly 19 audited pair keys")
    if set(quarantine_raw["orthosteric_pair_key"]) != quarantine_keys:
        raise RuntimeError("not every quarantined pair has a raw source row")
    if set(quarantine_incidence["orthosteric_pair_key"]) != quarantine_keys:
        raise RuntimeError("not every quarantined pair has source incidence")
    if active_master["orthosteric_pair_key"].duplicated().any():
        raise RuntimeError("audited active master key is not unique")
    if active_incidence.duplicated(["orthosteric_pair_key", "source_database"]).any():
        raise RuntimeError("audited active source incidence key is not unique")

    # This is the critical provenance-propagation audit: nine excluded/held
    # pairs have an additional non-BioLiP input row and that row must also be
    # quarantined.
    second_provenance_bad_keys = set(
        bad_crosswalk.loc[bad_crosswalk["has_second_input_provenance_row"], "orthosteric_pair_key"]
    )
    non_biolip_quarantine_raw = quarantine_raw.loc[
        ~quarantine_raw["source_databases"].map(lambda value: "BioLiP" in split_tokens(value))
    ]
    if len(second_provenance_bad_keys) != 9 or len(non_biolip_quarantine_raw) != 9:
        raise RuntimeError(
            "pair-wide provenance propagation contract failed: "
            f"keys={len(second_provenance_bad_keys)}, rows={len(non_biolip_quarantine_raw)}"
        )
    if set(non_biolip_quarantine_raw["orthosteric_pair_key"]) != second_provenance_bad_keys:
        raise RuntimeError("the nine second-provenance rows do not match the audited pair keys")

    pre_source_counts = (
        incidence.groupby("source_database", as_index=False)
        .agg(
            pre_audit_pair_rows=("orthosteric_pair_key", "nunique"),
            pre_audit_unique_proteins=("uniprot", "nunique"),
        )
    )
    post_source_counts = (
        active_incidence.groupby("source_database", as_index=False)
        .agg(
            post_audit_pair_rows=("orthosteric_pair_key", "nunique"),
            post_audit_unique_proteins=("uniprot", "nunique"),
        )
    )
    source_counts = pre_source_counts.merge(post_source_counts, on="source_database", how="outer").fillna(0)
    for column in source_counts.columns[1:]:
        source_counts[column] = source_counts[column].astype(int)
    source_counts["removed_pair_rows"] = (
        source_counts["pre_audit_pair_rows"] - source_counts["post_audit_pair_rows"]
    )
    source_counts["removed_unique_proteins_delta"] = (
        source_counts["pre_audit_unique_proteins"] - source_counts["post_audit_unique_proteins"]
    )
    source_counts = source_counts.sort_values(
        ["post_audit_pair_rows", "source_database"], ascending=[False, True]
    ).reset_index(drop=True)

    disposition_summary = (
        crosswalk.groupby(["manual_disposition", "pairwide_action"], as_index=False)
        .agg(
            audited_pairs=("orthosteric_pair_key", "nunique"),
            proteins=("uniprot", "nunique"),
            full_inchikeys=("full_inchikey", "nunique"),
            pairs_with_second_input_provenance=("has_second_input_provenance_row", "sum"),
        )
        .sort_values("manual_disposition")
    )

    benchmark_audits = {
        "main": benchmark_subset_audit(MAIN, active_keys, quarantine_keys),
        "broad_superset": benchmark_subset_audit(BROAD, active_keys, quarantine_keys),
    }
    if any(item["missing_from_audited_master"] for item in benchmark_audits.values()):
        raise RuntimeError(f"audited master lost a benchmark reference pair: {benchmark_audits}")

    # Stable ordering before deterministic gzip output.
    active_master = active_master.sort_values(["uniprot", "full_inchikey"]).reset_index(drop=True)
    quarantine_master = quarantine_master.sort_values(["uniprot", "full_inchikey"]).reset_index(drop=True)
    active_raw = active_raw.sort_values(["orthosteric_pair_key", "dataset_row_id"]).reset_index(drop=True)
    quarantine_raw = quarantine_raw.sort_values(["orthosteric_pair_key", "dataset_row_id"]).reset_index(drop=True)
    active_incidence = active_incidence.sort_values(
        ["source_database", "uniprot", "full_inchikey"]
    ).reset_index(drop=True)
    quarantine_incidence = quarantine_incidence.sort_values(
        ["orthosteric_pair_key", "source_database"]
    ).reset_index(drop=True)
    crosswalk = crosswalk.sort_values(["manual_disposition", "uniprot", "full_inchikey"]).reset_index(drop=True)

    active_master_path = DATA / "ORTHOSTERIC_PAIR_MASTER_AUDITED.tsv.gz"
    active_raw_path = DATA / "ORTHOSTERIC_INPUT_ROWS_AUDITED.tsv.gz"
    active_incidence_path = DATA / "ORTHOSTERIC_SOURCE_INCIDENCE_AUDITED.tsv.gz"
    quarantine_master_path = DATA / "ORTHOSTERIC_PAIR_AUDIT_QUARANTINE.tsv.gz"
    quarantine_raw_path = DATA / "ORTHOSTERIC_INPUT_ROWS_AUDIT_QUARANTINE.tsv.gz"
    quarantine_incidence_path = DATA / "ORTHOSTERIC_SOURCE_INCIDENCE_AUDIT_QUARANTINE.tsv.gz"
    crosswalk_path = DATA / "BIOLIP_MANUAL_AUDIT_PAIR_CROSSWALK.tsv.gz"
    source_counts_path = DATA / "CHECKPOINT2_SOURCE_COUNTS_PRE_POST.tsv"
    disposition_path = DATA / "CHECKPOINT2_DISPOSITION_COUNTS.tsv"

    write_tsv(active_master, active_master_path, compressed=True)
    write_tsv(active_raw, active_raw_path, compressed=True)
    write_tsv(active_incidence, active_incidence_path, compressed=True)
    write_tsv(quarantine_master, quarantine_master_path, compressed=True)
    write_tsv(quarantine_raw, quarantine_raw_path, compressed=True)
    write_tsv(quarantine_incidence, quarantine_incidence_path, compressed=True)
    write_tsv(crosswalk, crosswalk_path, compressed=True)
    write_tsv(source_counts, source_counts_path)
    write_tsv(disposition_summary, disposition_path)

    input_hashes = pd.DataFrame(
        [
            {"role": "checkpoint1_pair_master", "path": str(PAIR_MASTER_PRE), "sha256": sha256(PAIR_MASTER_PRE)},
            {"role": "checkpoint1_raw_rows", "path": str(RAW_ROWS_PRE), "sha256": sha256(RAW_ROWS_PRE)},
            {"role": "checkpoint1_source_incidence", "path": str(SOURCE_INCIDENCE_PRE), "sha256": sha256(SOURCE_INCIDENCE_PRE)},
            {"role": "checkpoint1_validation", "path": str(CHECKPOINT1_VALIDATION), "sha256": sha256(CHECKPOINT1_VALIDATION)},
            {"role": "manual_audit_ledger", "path": str(AUDIT_LEDGER), "sha256": sha256(AUDIT_LEDGER)},
            {"role": "manual_audit_excluded_held", "path": str(AUDIT_EXCLUDED_HELD), "sha256": sha256(AUDIT_EXCLUDED_HELD)},
            {"role": "manual_audit_retained", "path": str(AUDIT_FILTERED_RETAINED), "sha256": sha256(AUDIT_FILTERED_RETAINED)},
            {"role": "manual_audit_counts", "path": str(AUDIT_COUNTS), "sha256": sha256(AUDIT_COUNTS)},
            {"role": "main_benchmark_subset_check", "path": str(MAIN), "sha256": sha256(MAIN)},
            {"role": "broad_benchmark_subset_check", "path": str(BROAD), "sha256": sha256(BROAD)},
        ]
    )
    input_hashes_path = PACKAGE / "manifests/CHECKPOINT2_INPUT_HASHES.tsv"
    write_tsv(input_hashes, input_hashes_path)

    output_paths = {
        "active_pair_master": active_master_path,
        "active_raw_input_rows": active_raw_path,
        "active_source_incidence": active_incidence_path,
        "quarantine_pair_master": quarantine_master_path,
        "quarantine_raw_input_rows": quarantine_raw_path,
        "quarantine_source_incidence": quarantine_incidence_path,
        "audit_crosswalk": crosswalk_path,
        "source_counts_pre_post": source_counts_path,
        "disposition_counts": disposition_path,
        "input_hashes": input_hashes_path,
    }
    validation = {
        "status": "validated",
        "checkpoint": 2,
        "scope": "pair-wide application of the BioLiP manual retain/exclude/hold audit",
        "audit_rows": int(len(audit)),
        "audit_disposition_counts": {str(k): int(v) for k, v in disposition_counts.items()},
        "audit_source_pair_id_unmatched": 0,
        "audit_exact_identity_mismatches": int(len(identity_mismatch)),
        "pre_audit_pair_rows": int(len(master)),
        "active_pair_rows": int(len(active_master)),
        "quarantine_pair_rows": int(len(quarantine_master)),
        "active_raw_input_rows": int(len(active_raw)),
        "quarantine_raw_input_rows": int(len(quarantine_raw)),
        "active_source_incidence_rows": int(len(active_incidence)),
        "quarantine_source_incidence_rows": int(len(quarantine_incidence)),
        "active_unique_proteins": int(active_master["uniprot"].nunique()),
        "active_unique_full_inchikeys": int(active_master["full_inchikey"].nunique()),
        "active_unique_connectivity_keys": int(active_master["connectivity_key"].nunique()),
        "active_fingerprint_eligible_rows": int(active_master["fingerprint_eligible"].astype(str).str.lower().eq("true").sum()),
        "active_fingerprint_unresolved_rows": int(active_master["fingerprint_eligible"].astype(str).str.lower().ne("true").sum()),
        "pairwide_second_provenance_keys_quarantined": int(len(second_provenance_bad_keys)),
        "non_biolip_input_rows_quarantined_by_pairwide_propagation": int(len(non_biolip_quarantine_raw)),
        "quarantined_keys_leaked_to_active_master": int(len(active_keys & quarantine_keys)),
        "benchmark_subset_audits": benchmark_audits,
        "inputs": {
            row["role"]: {"path": row["path"], "sha256": row["sha256"]}
            for row in input_hashes.to_dict("records")
        },
        "outputs": {},
    }
    for key, path in output_paths.items():
        validation["outputs"][key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    source_table = "\n".join(
        f"| {row.source_database} | {int(row.pre_audit_pair_rows):,} | "
        f"{int(row.post_audit_pair_rows):,} | {int(row.removed_pair_rows):,} |"
        for row in source_counts.itertuples(index=False)
    )
    report = f"""# Checkpoint 2 — BioLiP manual audit application

## Result

PASS.  All 266 manually audited BioLiP source pairs mapped one-to-one to the
checkpoint-1 pair master.  Audit decisions were applied to the entire exact
`(UniProt, full InChIKey)` pair, not only to its BioLiP input row.

- retained audited BioLiP pairs: **{disposition_counts[RETAIN]:,}**
- excluded audited BioLiP pairs: **{disposition_counts[EXCLUDE]:,}**
- held audited BioLiP pairs: **{disposition_counts[HOLD]:,}**
- active orthosteric pair master: **{len(active_master):,}**
- quarantined exact pairs: **{len(quarantine_master):,}**
- active / quarantined raw input rows: **{len(active_raw):,} / {len(quarantine_raw):,}**
- active / quarantined source-incidence rows: **{len(active_incidence):,} / {len(quarantine_incidence):,}**

## Pair-wide provenance propagation

Nine excluded/held exact pairs had a second input row from the
UniProt/ChEBI/KEGG participant expansion.  All nine second-provenance rows were
quarantined.  Removing only the BioLiP rows would have incorrectly allowed
these pairs to re-enter the active orthosteric reference.

## Source counts before and after audit

Counts overlap across databases.

| source database | pre-audit pairs | post-audit pairs | removed pairs |
|---|---:|---:|---:|
{source_table}

## Active data shape

`ORTHOSTERIC_PAIR_MASTER_AUDITED.tsv.gz` is the checkpoint-2 orthosteric pair
reference, one row per exact pair.  It contains {len(active_master):,} rows and
adds the audit scope, audit ID, disposition, pair-wide action, and reference
eligibility fields.

`ORTHOSTERIC_INPUT_ROWS_AUDITED.tsv.gz` retains the source-row grain, while
`ORTHOSTERIC_SOURCE_INCIDENCE_AUDITED.tsv.gz` retains one row per exact pair and
source database.  The corresponding `*_AUDIT_QUARANTINE*` files preserve every
excluded or held record rather than deleting it.

`BIOLIP_MANUAL_AUDIT_PAIR_CROSSWALK.tsv.gz` contains all 266 audited BioLiP
pairs with their exact checkpoint-1 identity and manual rationale.

## Validation

- unmatched audit source-pair IDs: **0**
- retained exact-identity mismatches: **0**
- quarantined keys leaked into the active master: **0**
- main benchmark orthosteric pairs missing after audit: **{benchmark_audits['main']['missing_from_audited_master']}**
- broad benchmark orthosteric pairs missing after audit: **{benchmark_audits['broad_superset']['missing_from_audited_master']}**
"""
    report_path = PACKAGE / "reports/CHECKPOINT2_BIOLIP_MANUAL_AUDIT.md"
    atomic_text(report_path, report)
    validation["outputs"]["report"] = {
        "path": str(report_path),
        "bytes": report_path.stat().st_size,
        "sha256": sha256(report_path),
    }
    validation_path = PACKAGE / "validation/CHECKPOINT2_VALIDATION.json"
    atomic_json(validation_path, validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
