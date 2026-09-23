#!/usr/bin/env python3
"""Independent validator for checkpoint-5 exact reference linkage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

BUILD = VALIDATION / "CHECKPOINT5_BUILD_SUMMARY.json"
MASTER = DATA / "BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz"
ELIGIBILITY = DATA / "BIOLIP_SITE_MAPPING_ELIGIBILITY.tsv.gz"
ALLOBENCH = ROOT / "data/allobench/AlloBench.csv"
STRICT_PAIRS = ROOT / (
    "analysis/allosteric_orthosteric_preprocessing_v1_strict_pair_multisite_audit/"
    "data/FILTERED_STRICT_ORTHOSTERIC_PAIRS.tsv"
)
OLD_BRIDGE = ROOT / "analysis/structure_known_candidate_rebuild/data/exact_anchor_label_bridge.parquet"
LEGACY_EXACT_ALLOSTERIC_EXPECTED = 2247

REGISTRY = DATA / "CHECKPOINT5_ALLOBENCH_SOURCE_REGISTRY.tsv.gz"
ALLO_LINKS = DATA / "CHECKPOINT5_ALLOSTERIC_EXACT_LINKS.tsv.gz"
ALLO_OBS = DATA / "CHECKPOINT5_EXACT_ALLOSTERIC_OBSERVATIONS.tsv.gz"
RECON = DATA / "CHECKPOINT5_LEGACY_ALLOSTERIC_RECONCILIATION.tsv.gz"
ORTHO_LINKS = DATA / "CHECKPOINT5_ORTHOSTERIC_EXACT_LINKS.tsv.gz"
ORTHO_COVERAGE = DATA / "CHECKPOINT5_ORTHOSTERIC_PAIR_COVERAGE.tsv.gz"
CONFLICTS = DATA / "CHECKPOINT5_EXACT_REFERENCE_CONFLICTS.tsv.gz"
UNIFIED = DATA / "BIOLIP_EXACT_SITE_REFERENCE_OBSERVATIONS.tsv.gz"
INPUT_HASHES = MANIFESTS / "CHECKPOINT5_INPUT_HASHES.tsv"
VALIDATION_OUT = VALIDATION / "CHECKPOINT5_VALIDATION.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def true_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def normalize_accession(value: object) -> str:
    return str(value).upper().split("-", 1)[0]


def checkpoint4_identity_counts(unified: pd.DataFrame) -> tuple:
    """Validate the frozen reference against checkpoint 4 without optional DuckDB."""
    wanted = set(unified["observation_id"])
    master_columns = [
        "source_ordinal", "observation_id", "pdb_id", "receptor_chain",
        "uniprot_resolved", "ligand_ccd", "ligand_chain",
        "ligand_auth_seq_id", "full_inchikey",
    ]
    master_rows = []
    for chunk in pd.read_csv(
        MASTER, sep="\t", dtype=str, keep_default_na=False,
        usecols=master_columns, chunksize=100000,
    ):
        selected = chunk.loc[chunk["observation_id"].isin(wanted)]
        if not selected.empty:
            master_rows.append(selected)
    master = pd.concat(master_rows, ignore_index=True) if master_rows else pd.DataFrame(columns=master_columns)
    if master["observation_id"].duplicated().any():
        raise RuntimeError("checkpoint-4 master contains duplicate requested observation IDs")

    eligibility_rows = []
    for chunk in pd.read_csv(
        ELIGIBILITY, sep="\t", dtype=str, keep_default_na=False,
        usecols=["observation_id", "final_site_mapping_usable"], chunksize=100000,
    ):
        selected = chunk.loc[chunk["observation_id"].isin(wanted)]
        if not selected.empty:
            eligibility_rows.append(selected)
    eligibility = (
        pd.concat(eligibility_rows, ignore_index=True)
        if eligibility_rows else pd.DataFrame(columns=["observation_id", "final_site_mapping_usable"])
    )
    if eligibility["observation_id"].duplicated().any():
        raise RuntimeError("checkpoint-4 eligibility table contains duplicate requested observation IDs")

    joined = unified.merge(master, on="observation_id", how="left", suffixes=("_reference", "_master"))
    joined = joined.merge(eligibility, on="observation_id", how="left")
    missing_master = int(joined["source_ordinal_master"].isna().sum())
    mismatch = (
        joined["source_ordinal_reference"].astype(str).ne(joined["source_ordinal_master"].astype(str))
        | joined["pdb_id_reference"].str.lower().ne(joined["pdb_id_master"].str.lower())
        | joined["receptor_chain_reference"].ne(joined["receptor_chain_master"])
        | joined["uniprot"].map(normalize_accession).ne(joined["uniprot_resolved"].map(normalize_accession))
        | joined["ligand_ccd_reference"].str.upper().ne(joined["ligand_ccd_master"].str.upper())
        | joined["ligand_chain_reference"].ne(joined["ligand_chain_master"])
        | joined["ligand_auth_seq_id_reference"].ne(joined["ligand_auth_seq_id_master"])
        | joined["full_inchikey_reference"].str.upper().ne(joined["full_inchikey_master"].str.upper())
    )
    identity_mismatch = int((mismatch & joined["source_ordinal_master"].notna()).sum())
    mapping_ineligible = int((~true_series(joined["final_site_mapping_usable"])).sum())
    return len(joined), missing_master, identity_mismatch, mapping_ineligible


def main() -> None:
    if not BUILD.is_file():
        raise FileNotFoundError(BUILD)
    build = json.loads(BUILD.read_text())
    if build.get("checkpoint") != 5 or build.get("status") != "complete_pending_independent_validation":
        raise RuntimeError("checkpoint-5 builder state is invalid")
    for name, record in build["outputs"].items():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"builder output hash mismatch: {name}")

    input_hashes = pd.read_csv(INPUT_HASHES, sep="\t", dtype=str, keep_default_na=False)
    if input_hashes["asset_id"].duplicated().any():
        raise RuntimeError("checkpoint-5 input asset IDs are not unique")
    for row in input_hashes.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or sha256(path) != row.sha256:
            raise RuntimeError(f"checkpoint-5 input hash mismatch: {row.asset_id}")

    source = pd.read_csv(ALLOBENCH, dtype=str, keep_default_na=False)
    registry = pd.read_csv(REGISTRY, sep="\t", dtype=str, keep_default_na=False)
    allo_links = pd.read_csv(ALLO_LINKS, sep="\t", dtype=str, keep_default_na=False)
    allo_obs = pd.read_csv(ALLO_OBS, sep="\t", dtype=str, keep_default_na=False)
    reconciliation = pd.read_csv(RECON, sep="\t", dtype=str, keep_default_na=False)
    ortho = pd.read_csv(ORTHO_LINKS, sep="\t", dtype=str, keep_default_na=False)
    coverage = pd.read_csv(ORTHO_COVERAGE, sep="\t", dtype=str, keep_default_na=False)
    conflicts = pd.read_csv(CONFLICTS, sep="\t", dtype=str, keep_default_na=False)
    unified = pd.read_csv(UNIFIED, sep="\t", dtype=str, keep_default_na=False)
    strict = pd.read_csv(STRICT_PAIRS, sep="\t", dtype=str, keep_default_na=False)

    if len(registry) != len(source):
        raise RuntimeError("AlloBench source registry does not preserve every source row")
    if registry["allobench_source_id"].duplicated().any():
        raise RuntimeError("AlloBench source IDs are not unique")
    if sorted(registry["allobench_row_index"].astype(int)) != list(range(len(source))):
        raise RuntimeError("AlloBench row indices are not preserved")
    if allo_links["allosteric_link_id"].duplicated().any():
        raise RuntimeError("allosteric link IDs are not unique")
    if allo_links.duplicated(["allobench_source_id", "observation_id"]).any():
        raise RuntimeError("duplicate source-to-observation allosteric link")

    supported_links = allo_links.loc[
        allo_links["link_status"].eq("exact_instance_site_residue_supported")
    ]
    if (supported_links["author_position_overlap_count"].astype(int) < 1).any():
        raise RuntimeError("site-supported allosteric link has no residue overlap")
    unsupported = allo_links.loc[
        ~allo_links["link_status"].eq("exact_instance_site_residue_supported")
    ]
    if (unsupported["author_position_overlap_count"].astype(int) != 0).any():
        raise RuntimeError("unsupported allosteric link contains a residue overlap")
    if allo_obs["observation_id"].duplicated().any():
        raise RuntimeError("exact allosteric observation table is not unique")
    if set(allo_obs.observation_id) != set(supported_links.observation_id):
        raise RuntimeError("exact allosteric observation table differs from supported links")
    eligible_allo = set(
        allo_obs.loc[true_series(allo_obs["eligible_exact_allosteric"]), "observation_id"]
    )
    if not true_series(
        allo_obs.loc[allo_obs.observation_id.isin(eligible_allo), "checkpoint4_mapping_usable"]
    ).all():
        raise RuntimeError("eligible allosteric observation failed checkpoint-4 mapping")

    if ortho["orthosteric_link_id"].duplicated().any() or ortho["record_id"].duplicated().any():
        raise RuntimeError("orthosteric exact observation/link IDs are not unique")
    if ortho["observation_id"].duplicated().any():
        raise RuntimeError("an exact orthosteric observation maps to more than one pair")
    if not true_series(ortho["checkpoint4_identity_match"]).all():
        raise RuntimeError("orthosteric source identity mismatch")
    if not true_series(ortho["active_broad_orthosteric_pair_reference"]).all():
        raise RuntimeError("audited exact orthosteric observation missing from broad pair reference")
    expected_ortho_eligible = true_series(ortho["checkpoint4_mapping_usable"])
    if not true_series(ortho["eligible_exact_orthosteric"]).equals(expected_ortho_eligible):
        raise RuntimeError("orthosteric eligibility is inconsistent with checkpoint-4 mapping")
    if set(coverage.cumulative_pair_id) != set(strict.cumulative_pair_id):
        raise RuntimeError("orthosteric pair coverage differs from the 247 retained pairs")
    linked_pair_counts = ortho.groupby("cumulative_pair_id")["observation_id"].nunique().astype(int)
    coverage_counts = coverage.set_index("cumulative_pair_id")["exact_observations"].astype(int)
    if not linked_pair_counts.sort_index().equals(coverage_counts.sort_index()):
        raise RuntimeError("orthosteric pair coverage counts are inconsistent")
    eligible_ortho = set(
        ortho.loc[true_series(ortho["eligible_exact_orthosteric"]), "observation_id"]
    )

    # The old parquet is hash-pinned in CHECKPOINT5_INPUT_HASHES.tsv.  Its exact
    # positive denominator was independently frozen before this validator was
    # written, avoiding an optional parquet-engine dependency here.
    old_exact_count = LEGACY_EXACT_ALLOSTERIC_EXPECTED
    if int(true_series(reconciliation["legacy_allo_exact"]).sum()) != old_exact_count:
        raise RuntimeError("legacy exact-allosteric reconciliation denominator changed")
    if set(reconciliation["reconciliation_state"]) - {
        "retained_with_site_residue_support", "removed_by_stricter_exact_site_rule",
        "newly_added_by_stricter_exact_site_rule",
    }:
        raise RuntimeError("unexpected reconciliation state")

    conflict_ids = set(conflicts.observation_id)
    if conflict_ids & set(unified.observation_id):
        raise RuntimeError("conflicting observation leaked into unified reference")
    if unified["observation_id"].duplicated().any():
        raise RuntimeError("unified reference observation IDs are not unique")
    unified_allo = set(unified.loc[unified.reference_label.eq("allosteric"), "observation_id"])
    unified_ortho = set(unified.loc[unified.reference_label.eq("orthosteric"), "observation_id"])
    if unified_allo != eligible_allo - conflict_ids:
        raise RuntimeError("unified allosteric set is inconsistent")
    if unified_ortho != eligible_ortho - conflict_ids:
        raise RuntimeError("unified orthosteric set is inconsistent")
    if unified_allo & unified_ortho:
        raise RuntimeError("unified allosteric and orthosteric labels overlap")

    # Independent identity join to the checkpoint-4 frozen master and eligibility.
    identity = checkpoint4_identity_counts(unified)
    if identity != (len(unified), 0, 0, 0):
        raise RuntimeError(f"unified reference identity validation failed: {identity}")

    label_counts = {str(k): int(v) for k, v in unified.reference_label.value_counts().items()}
    relation_counts = {
        str(k): int(v)
        for k, v in unified.loc[unified.reference_label.eq("allosteric"), "source_site_relation"].value_counts().items()
    }
    source_status_counts = {
        str(k): int(v) for k, v in registry.source_link_status.value_counts().items()
    }
    reconciliation_counts = {
        str(k): int(v) for k, v in reconciliation.reconciliation_state.value_counts().items()
    }
    result = {
        "checkpoint": 5,
        "status": "validated",
        "scope": "independent validation of exact allosteric and exact audited orthosteric observation linkage",
        "allobench_source_rows": len(registry),
        "allobench_source_link_status_counts": source_status_counts,
        "allosteric_candidate_links": len(allo_links),
        "allosteric_site_supported_links": len(supported_links),
        "allosteric_site_supported_observations": len(allo_obs),
        "allosteric_eligible_exact_observations_before_conflict_exclusion": len(eligible_allo),
        "allosteric_site_relation_counts_after_conflict_exclusion": relation_counts,
        "legacy_exact_allosteric_observations": old_exact_count,
        "legacy_reconciliation_counts": reconciliation_counts,
        "orthosteric_retained_pairs": len(coverage),
        "orthosteric_exact_observations": len(ortho),
        "orthosteric_eligible_exact_observations": len(eligible_ortho),
        "reference_conflicts_excluded": len(conflicts),
        "unified_reference_label_counts": label_counts,
        "unified_reference_observations": len(unified),
        "checkpoint4_identity_join": {
            "joined": int(identity[0]), "missing_master": int(identity[1]),
            "identity_mismatch": int(identity[2]), "mapping_ineligible": int(identity[3]),
        },
        "broad_orthosteric_sources_not_promoted_without_exact_biolip_audit": True,
        "large_structure_directory_recursively_scanned": False,
        "build_summary_sha256": sha256(BUILD),
        "validated_output_hashes": {
            name: record["sha256"] for name, record in build["outputs"].items()
        },
    }
    atomic_json(VALIDATION_OUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
