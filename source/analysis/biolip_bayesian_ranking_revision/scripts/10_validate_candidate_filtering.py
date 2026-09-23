#!/usr/bin/env python3
"""Independent contract checks for checkpoint-10 candidate filtering."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
REPORTS = PACKAGE / "reports"
VALIDATION = PACKAGE / "validation"

SITE_AUDIT = DATA / "CHECKPOINT10_SITE_CANDIDATE_AUDIT.tsv.gz"
PAIR_SHORTLIST = DATA / "CHECKPOINT10_PAIR_SHORTLIST.tsv.gz"
FILTER_COUNTS = DATA / "CHECKPOINT10_FILTER_COUNTS.tsv"
SENSITIVITY = DATA / "CHECKPOINT10_FILTER_SENSITIVITY.tsv"
BUILD = VALIDATION / "CHECKPOINT10_BUILD.json"
OUTPUT = VALIDATION / "CHECKPOINT10_VALIDATION.json"
REPORT = REPORTS / "CHECKPOINT10_FILTERED_CANDIDATES.md"

EXPECTED_COUNTS = {
    "all_ranked_sites": (5150, 2890, 153),
    "spatially_separate": (1768, 939, 117),
    "spatial_and_compound_filter": (1448, 787, 98),
    "review_filter": (1343, 736, 85),
    "ranking_consensus": (79, 53, 29),
    "multi_structure_shortlist": (13, 6, 6),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    errors = []
    build = json.loads(BUILD.read_text())
    if build.get("status") != "built_awaiting_independent_validation":
        errors.append("build status is not awaiting validation")
    site = pd.read_csv(SITE_AUDIT, sep="\t", low_memory=False)
    pairs = pd.read_csv(PAIR_SHORTLIST, sep="\t", low_memory=False)
    counts = pd.read_csv(FILTER_COUNTS, sep="\t")
    sensitivity = pd.read_csv(SENSITIVITY, sep="\t")

    count_lookup = {
        row.filter_step: (int(row.site_rows), int(row.protein_ligand_pairs), int(row.proteins))
        for row in counts.itertuples(index=False)
    }
    for name, expected in EXPECTED_COUNTS.items():
        if count_lookup.get(name) != expected:
            errors.append(f"frozen funnel count changed for {name}: {count_lookup.get(name)}")

    if site.candidate_site_ligand_id.duplicated().any() or len(site) != 5150:
        errors.append("site audit does not preserve 5,150 unique site rows")
    if not site.loc[site.passes_ranking_consensus, "passes_review_filter"].all():
        errors.append("ranking consensus contains a row that failed the review filter")
    if not site.loc[site.passes_multi_structure_shortlist, "passes_ranking_consensus"].all():
        errors.append("multi-structure shortlist contains a non-consensus row")
    if site.loc[site.passes_review_filter, "known_allosteric_by_any_broad_link"].any():
        errors.append("a broad known allosteric link survived the review filter")
    if site.loc[site.passes_review_filter, "same_protein_exact_orthosteric_connectivity"].any():
        errors.append("an exact orthosteric connectivity match survived the review filter")
    if site.loc[site.passes_review_filter, "max_same_protein_orthosteric_tanimoto"].ge(.70).any():
        errors.append("an orthosteric analogue at or above 0.70 survived the review filter")

    if len(pairs) != 6 or pairs.candidate_pair_id.nunique() != 6:
        errors.append("pair shortlist is not six unique pairs")
    if int(pairs.retained_after_manual_review.sum()) != 5:
        errors.append("manual review does not retain exactly five pairs")
    excluded = pairs.loc[~pairs.retained_after_manual_review]
    if len(excluded) != 1 or excluded.ligand_ccds.iloc[0] != "1PS":
        errors.append("manual experimental-additive exclusion is not exactly 1PS")

    expected_sensitivity_rows = 3 * 3 * 3 * 3
    if len(sensitivity) != expected_sensitivity_rows:
        errors.append("filter sensitivity grid is incomplete")
    if set(sensitivity.summary_level) != {"review_filter", "ranking_consensus", "multi_structure"}:
        errors.append("filter sensitivity summary levels changed")

    for path in (SITE_AUDIT, PAIR_SHORTLIST, FILTER_COUNTS, SENSITIVITY, REPORT):
        expected_hash = build.get("output_hashes", {}).get(path.name)
        if expected_hash != sha256(path):
            errors.append(f"output hash mismatch: {path.name}")

    result = {
        "status": "validated_filtered_candidate_shortlist" if not errors else "failed",
        "errors": errors,
        "site_rows": len(site),
        "review_filter_sites": int(site.passes_review_filter.sum()),
        "ranking_consensus_sites": int(site.passes_ranking_consensus.sum()),
        "multi_structure_sites": int(site.passes_multi_structure_shortlist.sum()),
        "multi_structure_pairs": len(pairs),
        "manually_retained_pairs": int(pairs.retained_after_manual_review.sum()),
        "manual_exclusions": pairs.loc[~pairs.retained_after_manual_review, "ligand_ccds"].tolist(),
        "ranking_models_refit": False,
    }
    atomic_json(result, OUTPUT)
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
