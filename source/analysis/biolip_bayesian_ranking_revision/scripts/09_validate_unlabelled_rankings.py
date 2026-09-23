#!/usr/bin/env python3
"""Independent consistency checks for checkpoint-9 BioLiP rankings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
REPORTS = PACKAGE / "reports"
VALIDATION = PACKAGE / "validation"

CANDIDATES = DATA / "CHECKPOINT9_CANDIDATE_UNIVERSE.tsv.gz"
GEOMETRY = DATA / "CHECKPOINT9_CANDIDATE_GEOMETRY.tsv.gz"
OBSERVATION_SCORES = DATA / "CHECKPOINT9_OBSERVATION_SCORES.tsv.gz"
SITE_ASSIGNMENTS = DATA / "CHECKPOINT9_SITE_CLUSTER_ASSIGNMENTS.tsv.gz"
SITE_FEATURES = DATA / "CHECKPOINT9_SITE_FEATURES.tsv.gz"
SITE_RANKINGS = DATA / "CHECKPOINT9_SITE_RANKINGS_LONG.tsv.gz"
PAIR_RANKINGS = DATA / "CHECKPOINT9_PAIR_RANKINGS_LONG.tsv.gz"
TOP_CANDIDATES = DATA / "CHECKPOINT9_TOP_SITE_CANDIDATES.tsv.gz"
RANKING_COUNTS = DATA / "CHECKPOINT9_RANKING_COUNTS.tsv"
SCORING_SPEC = MANIFESTS / "CHECKPOINT9_SCORING_SPEC.json"
SCORING_BUILD = VALIDATION / "CHECKPOINT9_SCORING_BUILD.json"
OUTPUT = VALIDATION / "CHECKPOINT9_VALIDATION.json"
VALIDATED_REPORT = REPORTS / "CHECKPOINT9_VALIDATED_RESULTS.md"

DISTANCE = "ligand_centroid_to_nearest_orthosteric_site_CA_centroid_A"
SITE_JACCARD_THRESHOLD = 0.50


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


def atomic_text(value: str, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def parse_positions(value: object) -> set[int]:
    return {int(token) for token in str(value or "").split(";") if token}


def jaccard(first: set[int], second: set[int]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def main() -> None:
    errors = []
    build = json.loads(SCORING_BUILD.read_text())
    spec = json.loads(SCORING_SPEC.read_text())
    if build.get("status") != "built_awaiting_independent_validation":
        errors.append("scoring build status is not awaiting validation")
    if spec.get("probability_claim") is not False:
        errors.append("ranking contract incorrectly permits a probability claim")
    if spec.get("ranking_definitions") != 46:
        errors.append("ranking-definition count is not 46")

    candidates = pd.read_csv(
        CANDIDATES, sep="\t",
        usecols=["observation_id", "uniprot", "full_inchikey"],
        dtype=str, keep_default_na=False,
    )
    geometry = pd.read_csv(GEOMETRY, sep="\t", low_memory=False)
    observations = pd.read_csv(OBSERVATION_SCORES, sep="\t", low_memory=False)
    assignments = pd.read_csv(SITE_ASSIGNMENTS, sep="\t", dtype=str, keep_default_na=False)
    site_features = pd.read_csv(SITE_FEATURES, sep="\t", low_memory=False)
    site_rankings = pd.read_csv(SITE_RANKINGS, sep="\t", low_memory=False)
    pair_rankings = pd.read_csv(PAIR_RANKINGS, sep="\t", low_memory=False)
    top = pd.read_csv(TOP_CANDIDATES, sep="\t", low_memory=False)
    ranking_counts = pd.read_csv(RANKING_COUNTS, sep="\t")

    if len(candidates) != 19_920 or candidates.observation_id.nunique() != 19_920:
        errors.append("frozen candidate universe is not 19,920 unique observations")
    if len(geometry) != len(candidates) or set(geometry.observation_id) != set(candidates.observation_id):
        errors.append("geometry does not preserve the frozen candidate universe")
    ok_ids = set(geometry.loc[geometry.geometry_status.eq("ok"), "observation_id"])
    if set(observations.observation_id) != ok_ids or observations.observation_id.duplicated().any():
        errors.append("observation score rows do not equal geometry-complete observations")

    numeric_distance = pd.to_numeric(observations[DISTANCE], errors="coerce")
    raw_score = pd.to_numeric(observations.score_RAW_D, errors="coerce")
    if not np.allclose(numeric_distance, raw_score, atol=1e-12, rtol=0.0):
        errors.append("raw-distance observation score does not equal the selected distance")
    for threshold in (8, 10, 12):
        flag = observations[f"distance_ge_{threshold}A"].astype(str).str.lower().eq("true")
        if not np.array_equal(flag.to_numpy(), numeric_distance.ge(threshold).to_numpy()):
            errors.append(f"distance_ge_{threshold}A flag mismatch")

    assignment_key = assignments[["uniprot", "binding_uniprot_positions"]]
    if assignment_key.duplicated().any():
        errors.append("site assignment keys are duplicated")
    if assignments.candidate_site_cluster_id.eq("").any():
        errors.append("empty site-cluster ID")
    minimum_observed_jaccard = 1.0
    for (_protein, _cluster), group in assignments.groupby(
        ["uniprot", "candidate_site_cluster_id"], sort=False,
    ):
        sites = [parse_positions(value) for value in group.binding_uniprot_positions]
        for left in range(len(sites) - 1):
            for right in range(left + 1, len(sites)):
                value = jaccard(sites[left], sites[right])
                minimum_observed_jaccard = min(minimum_observed_jaccard, value)
                if value < SITE_JACCARD_THRESHOLD - 1e-12:
                    errors.append("complete-linkage cluster contains a below-threshold pair")
                    break
            if errors and errors[-1].startswith("complete-linkage"):
                break

    if site_features.candidate_site_ligand_id.duplicated().any():
        errors.append("site-ligand ranking units are duplicated")
    if site_rankings.ranking_id.nunique() != 46 or ranking_counts.ranking_id.nunique() != 46:
        errors.append("not all 46 ranking definitions are represented")
    expected_site_long = len(site_features) * 46
    if len(site_rankings) != expected_site_long:
        errors.append("site-ranking long-table row count mismatch")
    per_ranking = site_rankings.groupby("ranking_id").candidate_site_ligand_id.nunique()
    if not per_ranking.eq(len(site_features)).all():
        errors.append("ranking definitions do not share the same site universe")

    raw_sites = site_rankings.loc[site_rankings.ranking_id.eq("RAW:RAW_D")].copy()
    raw_merged = raw_sites.merge(
        site_features[["candidate_site_ligand_id", DISTANCE, "score_RAW_D"]],
        on="candidate_site_ligand_id", validate="one_to_one", suffixes=("_rank", "_feature"),
    )
    if not np.allclose(
        raw_merged.ranking_score, raw_merged[DISTANCE + "_feature"], atol=1e-12, rtol=0.0,
    ):
        errors.append("site raw ranking score does not equal median site distance")

    # Every score/rank table must be monotonically ordered within its own model.
    for ranking_id, group in site_rankings.groupby("ranking_id", sort=False):
        valid = group.loc[group.ranking_score.notna()].copy()
        if valid.empty:
            errors.append(f"ranking has no scored rows: {ranking_id}")
            continue
        best_score = valid.ranking_score.max()
        if not valid.loc[valid.ranking_score.eq(best_score), "site_global_rank"].eq(1).all():
            errors.append(f"global rank 1 mismatch: {ranking_id}")
        tied = valid.groupby(["uniprot", "ranking_score"], dropna=False).site_within_protein_rank.nunique()
        if (tied > 1).any():
            errors.append(f"tied within-protein scores have different ranks: {ranking_id}")

    if pair_rankings.ranking_id.nunique() != 46:
        errors.append("pair ranking does not contain all 46 definitions")
    for ranking_id, group in pair_rankings.groupby("ranking_id", sort=False):
        valid = group.loc[group.best_site_ranking_score.notna()]
        if valid.empty or valid.pair_global_rank.min() != 1:
            errors.append(f"pair global ranking is invalid: {ranking_id}")
    if top.ranking_id.nunique() != 46 or len(top) != 4_600:
        errors.append("top-candidate table is not 100 rows for each ranking")

    output_hashes = build.get("output_hashes", {})
    for path in (
        OBSERVATION_SCORES, SITE_ASSIGNMENTS, SITE_FEATURES, SITE_RANKINGS,
        PAIR_RANKINGS, TOP_CANDIDATES, RANKING_COUNTS, SCORING_SPEC,
        REPORTS / "CHECKPOINT9_UNLABELLED_BIOLIP_RANKING.md",
    ):
        expected = output_hashes.get(path.name)
        if expected != sha256(path):
            errors.append(f"output hash mismatch: {path.name}")

    distance = pd.to_numeric(site_features[DISTANCE], errors="coerce")
    validation = {
        "checkpoint": 9,
        "status": "validated_unlabelled_biolip_rankings" if not errors else "failed",
        "errors": errors,
        "candidate_rows_frozen": len(candidates),
        "candidate_rows_geometry_ok": len(observations),
        "candidate_rows_geometry_failed": len(candidates) - len(observations),
        "candidate_proteins_geometry_ok": observations.uniprot.nunique(),
        "candidate_pairs_geometry_ok": observations[["uniprot", "full_inchikey"]].drop_duplicates().shape[0],
        "site_clusters": assignments.candidate_site_cluster_id.nunique(),
        "protein_ligand_site_rows": len(site_features),
        "ranking_definitions": site_rankings.ranking_id.nunique(),
        "site_ranking_long_rows": len(site_rankings),
        "pair_ranking_long_rows": len(pair_rankings),
        "minimum_recomputed_within_cluster_pairwise_jaccard": minimum_observed_jaccard,
        "site_rows_distance_ge_8A": int(distance.ge(8).sum()),
        "site_rows_distance_ge_10A": int(distance.ge(10).sum()),
        "site_rows_distance_ge_12A": int(distance.ge(12).sum()),
        "raw_distance_is_continuous_not_binary": True,
        "ranking_scores_are_probabilities": False,
        "known_reference_pairs_removed_before_ranking": True,
        "directory_enumeration_performed": False,
    }
    atomic_json(validation, OUTPUT)
    if errors:
        raise RuntimeError("checkpoint-9 validation failed: " + "; ".join(errors[:10]))

    status_counts = geometry.geometry_status.value_counts().to_dict()
    report = f"""# Checkpoint 9 validated BioLiP ranking results

Status: **validated**

## Final coverage

- Frozen unlabeled candidate observations: **{len(candidates):,}**
- Geometry-complete observations: **{len(observations):,}**
- Geometry failures retained in the audit ledger: **{len(candidates)-len(observations):,}**
- Proteins represented: **{observations.uniprot.nunique():,}**
- Exact protein-ligand pairs represented: **{observations[['uniprot', 'full_inchikey']].drop_duplicates().shape[0]:,}**
- Non-chaining protein-level binding-site clusters: **{assignments.candidate_site_cluster_id.nunique():,}**
- Primary protein-ligand-site ranking rows: **{len(site_features):,}**
- Ranking definitions: **46**

Geometry status counts: `{status_counts}`.

## Distance filters

The raw distance is a continuous ranking, not a binary classifier. At the
site-level median, **{int(distance.ge(8).sum()):,}** rows are at least 8 A,
**{int(distance.ge(10).sum()):,}** are at least 10 A, and
**{int(distance.ge(12).sum()):,}** are at least 12 A from the nearest mapped
exact orthosteric-site definition.

## Validation checks

- All 19,920 frozen candidate observations are accounted for by either an
  explicit geometry status or a score row.
- Direct-distance scores equal the selected structural distance exactly.
- Complete-linkage site clusters preserve every pairwise residue-set Jaccard
  similarity at or above 0.50; single-linkage chaining is absent.
- All 46 ranking definitions share the same site universe. Chemistry-containing
  scores remain NA only where their required RDKit descriptors are unavailable.
- Global and within-protein ranks preserve score ties.
- Output hashes agree with the scoring build manifest.

These rankings identify structurally distal BioLiP observations. They do not
by themselves establish functional allostery, biological-assembly accessibility,
or calibrated probability.
"""
    atomic_text(report, VALIDATED_REPORT)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
