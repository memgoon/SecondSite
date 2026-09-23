#!/usr/bin/env python3
"""Checkpoint 9e: score and rank the unlabeled BioLiP candidate universe.

The 46 checkpoint-8 ranking definitions are fitted once on the complete frozen
reference and applied to candidate observations. No score is a probability.
Repeated observations are additionally collapsed into non-chaining, complete-
linkage residue-set site clusters before the primary public ranking is formed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import gaussian_kde


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
REPORTS = PACKAGE / "reports"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

REFERENCE = DATA / "CHECKPOINT7_REFERENCE_FEATURES.tsv.gz"
REFERENCE_SCORES = DATA / "CHECKPOINT7_METHOD_SCORE_MATRIX.tsv.gz"
METHOD_VALIDATION = VALIDATION / "CHECKPOINT7_METHOD_VALIDATION.json"
CHECKPOINT8_VALIDATION = VALIDATION / "CHECKPOINT8_VALIDATION.json"
CANDIDATES = DATA / "CHECKPOINT9_CANDIDATE_UNIVERSE.tsv.gz"
GEOMETRY = DATA / "CHECKPOINT9_CANDIDATE_GEOMETRY.tsv.gz"
GEOMETRY_BUILD = VALIDATION / "CHECKPOINT9_GEOMETRY_BUILD.json"
DESCRIPTORS = DATA / "CHECKPOINT9_CANDIDATE_LIGAND_DESCRIPTORS.tsv.gz"
DESCRIPTOR_BUILD = VALIDATION / "CHECKPOINT9_DESCRIPTOR_BUILD.json"

FEATURES_OUTPUT = DATA / "CHECKPOINT9_CANDIDATE_FEATURES.tsv.gz"
OBSERVATION_SCORES = DATA / "CHECKPOINT9_OBSERVATION_SCORES.tsv.gz"
SITE_ASSIGNMENTS = DATA / "CHECKPOINT9_SITE_CLUSTER_ASSIGNMENTS.tsv.gz"
SITE_FEATURES = DATA / "CHECKPOINT9_SITE_FEATURES.tsv.gz"
SITE_RANKINGS = DATA / "CHECKPOINT9_SITE_RANKINGS_LONG.tsv.gz"
PAIR_RANKINGS = DATA / "CHECKPOINT9_PAIR_RANKINGS_LONG.tsv.gz"
TOP_CANDIDATES = DATA / "CHECKPOINT9_TOP_SITE_CANDIDATES.tsv.gz"
COMPONENT_MODELS = DATA / "CHECKPOINT9_FULL_REFERENCE_COMPONENT_MODELS.tsv"
RANKING_COUNTS = DATA / "CHECKPOINT9_RANKING_COUNTS.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT9_SCORING_INPUT_HASHES.tsv"
SCORING_SPEC = MANIFESTS / "CHECKPOINT9_SCORING_SPEC.json"
REPORT = REPORTS / "CHECKPOINT9_UNLABELLED_BIOLIP_RANKING.md"
BUILD = VALIDATION / "CHECKPOINT9_SCORING_BUILD.json"

SOURCE_DISTANCE = "ligand_centroid_to_nearest_orthosteric_site_CA_centroid_A"
REFERENCE_DISTANCE = "ligand_centroid_to_orthosteric_site_CA_centroid_A"
FEATURES: Tuple[Tuple[str, str, str], ...] = (
    ("D", SOURCE_DISTANCE, "selected structural distance"),
    ("MW", "molecular_weight", "molecular weight"),
    ("LP", "clogp", "calculated LogP"),
    ("AR", "aromatic_ring_count", "aromatic ring count"),
)
REFERENCE_FEATURE = {
    "D": REFERENCE_DISTANCE,
    "MW": "molecular_weight",
    "LP": "clogp",
    "AR": "aromatic_ring_count",
}
FEATURE_BY_CODE = {code: column for code, column, _description in FEATURES}
CONTINUOUS = {"D", "MW", "LP"}
ESTIMATORS = ("QNB", "KDE_LEGACY", "KDE100")
ALPHA = 0.5
DENSITY_FLOOR = 1e-12
DECIMALS = 12
SITE_JACCARD_THRESHOLD = 0.50
CAPS = {
    "KDE_LEGACY": {
        "D": math.log(100.0),
        "MW": math.log(10.0),
        "LP": math.log(10.0),
        "AR": math.log(10.0),
    },
    "KDE100": {code: math.log(100.0) for code, _column, _description in FEATURES},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_tsv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(value: str, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def stable_id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def parse_positions(value: object) -> Tuple[int, ...]:
    return tuple(sorted({int(token) for token in str(value or "").split(";") if token}))


def numeric_or_nan(value: object) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else np.nan


def jaccard(first: Sequence[int], second: Sequence[int]) -> float:
    left, right = set(first), set(second)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def feature_subsets() -> List[Tuple[str, ...]]:
    codes = [item[0] for item in FEATURES]
    return [
        subset
        for size in range(1, len(codes) + 1)
        for subset in itertools.combinations(codes, size)
    ]


def smoothed_bin_model(bin_ids: np.ndarray, labels: np.ndarray, n_bins: int):
    negative = np.bincount(bin_ids[labels == 0], minlength=n_bins).astype(int)
    positive = np.bincount(bin_ids[labels == 1], minlength=n_bins).astype(int)
    negative_probability = (negative + ALPHA) / (int(np.sum(labels == 0)) + ALPHA * n_bins)
    positive_probability = (positive + ALPHA) / (int(np.sum(labels == 1)) + ALPHA * n_bins)
    return negative, positive, np.log(positive_probability / negative_probability)


def fit_qnb(values: np.ndarray, labels: np.ndarray, code: str) -> dict:
    if code in CONTINUOUS:
        requested = int(math.ceil(len(values) ** (1.0 / 3.0)))
        internal_edges = np.unique(np.quantile(
            values, np.linspace(0.0, 1.0, requested + 1), method="linear",
        )[1:-1])
        bins = np.searchsorted(internal_edges, values, side="right")
        n_bins = len(internal_edges) + 1
        categories = []
        edges = [float("-inf"), *[float(value) for value in internal_edges], float("inf")]
        estimator = "pooled_equal_frequency_bins_cube_root_rule"
    else:
        categories = sorted(np.unique(values.astype(int)).tolist())
        lookup = {category: index for index, category in enumerate(categories)}
        bins = np.asarray([lookup[int(value)] for value in values], dtype=int)
        n_bins = len(categories) + 1
        requested = n_bins
        internal_edges = np.asarray([], dtype=float)
        edges = []
        estimator = "observed_categories_plus_unseen_category_bin"
    negative, positive, log_lr = smoothed_bin_model(bins, labels, n_bins)
    return {
        "estimator": estimator,
        "requested_bins": requested,
        "n_bins": n_bins,
        "internal_edges": internal_edges,
        "edges": edges,
        "categories": categories,
        "negative_counts": negative,
        "positive_counts": positive,
        "bin_log_lr": log_lr,
    }


def transform_qnb(model: dict, values: np.ndarray, code: str):
    if code in CONTINUOUS:
        bins = np.searchsorted(model["internal_edges"], values, side="right")
        outside = int(np.sum((values < model["edges"][1]) | (values > model["edges"][-2])))
    else:
        lookup = {category: index for index, category in enumerate(model["categories"])}
        unseen = model["n_bins"] - 1
        bins = np.asarray([lookup.get(int(value), unseen) for value in values], dtype=int)
        outside = int(np.sum(bins == unseen))
    return model["bin_log_lr"][bins], outside


def fit_kde(values: np.ndarray, labels: np.ndarray, code: str) -> dict:
    if code in CONTINUOUS:
        negative = gaussian_kde(values[labels == 0], bw_method="scott")
        positive = gaussian_kde(values[labels == 1], bw_method="scott")
        return {
            "estimator": "gaussian_kde_scott_full_reference",
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "negative_fit": negative,
            "positive_fit": positive,
            "negative_kde_factor": float(negative.factor),
            "positive_kde_factor": float(positive.factor),
            "categories": [],
            "n_bins": np.nan,
            "negative_counts": np.asarray([], dtype=int),
            "positive_counts": np.asarray([], dtype=int),
            "bin_log_lr": np.asarray([], dtype=float),
        }
    categories = sorted(np.unique(values.astype(int)).tolist())
    lookup = {category: index for index, category in enumerate(categories)}
    bins = np.asarray([lookup[int(value)] for value in values], dtype=int)
    n_bins = len(categories) + 1
    negative, positive, log_lr = smoothed_bin_model(bins, labels, n_bins)
    return {
        "estimator": "categorical_frequency_with_unseen_category_bin",
        "minimum": np.nan,
        "maximum": np.nan,
        "negative_fit": None,
        "positive_fit": None,
        "negative_kde_factor": np.nan,
        "positive_kde_factor": np.nan,
        "categories": categories,
        "n_bins": n_bins,
        "negative_counts": negative,
        "positive_counts": positive,
        "bin_log_lr": log_lr,
    }


def transform_kde(model: dict, values: np.ndarray, code: str):
    if code in CONTINUOUS:
        evaluated = np.clip(values, model["minimum"], model["maximum"])
        negative = model["negative_fit"](evaluated) + DENSITY_FLOOR
        positive = model["positive_fit"](evaluated) + DENSITY_FLOOR
        return np.log(positive / negative), int(np.sum(evaluated != values))
    lookup = {category: index for index, category in enumerate(model["categories"])}
    unseen = int(model["n_bins"]) - 1
    bins = np.asarray([lookup.get(int(value), unseen) for value in values], dtype=int)
    return model["bin_log_lr"][bins], int(np.sum(bins == unseen))


def model_record(estimator_id: str, code: str, model: dict, outside_rows: int) -> dict:
    return {
        "estimator_id": estimator_id,
        "feature_code": code,
        "feature_column_reference": REFERENCE_FEATURE[code],
        "feature_column_candidate": FEATURE_BY_CODE[code],
        "estimator": model["estimator"],
        "training_rows": 5_798,
        "requested_bins": model.get("requested_bins", np.nan),
        "actual_bins_including_unseen_bin": model.get("n_bins", np.nan),
        "bin_edges_json": json.dumps(model.get("edges", [model.get("minimum"), model.get("maximum")])),
        "categories_json": json.dumps(model.get("categories", [])),
        "negative_bin_counts_json": json.dumps(model.get("negative_counts", np.asarray([])).tolist()),
        "positive_bin_counts_json": json.dumps(model.get("positive_counts", np.asarray([])).tolist()),
        "bin_log_lr_json": json.dumps(model.get("bin_log_lr", np.asarray([])).tolist()),
        "negative_kde_factor": model.get("negative_kde_factor", np.nan),
        "positive_kde_factor": model.get("positive_kde_factor", np.nan),
        "candidate_rows_clipped_or_unseen": outside_rows,
    }


def cluster_sites(frame: pd.DataFrame):
    exact = (
        frame.groupby(["uniprot", "binding_uniprot_positions"], sort=True)
        .agg(source_observations=("observation_id", "size"))
        .reset_index()
    )
    assignments = []
    for uniprot, group in exact.groupby("uniprot", sort=True):
        group = group.sort_values("binding_uniprot_positions", kind="mergesort").reset_index(drop=True)
        sets = [parse_positions(value) for value in group.binding_uniprot_positions]
        if len(group) == 1:
            labels = np.ones(1, dtype=int)
        else:
            condensed = np.empty(len(group) * (len(group) - 1) // 2, dtype=float)
            cursor = 0
            for left in range(len(group) - 1):
                for right in range(left + 1, len(group)):
                    condensed[cursor] = 1.0 - jaccard(sets[left], sets[right])
                    cursor += 1
            hierarchy = linkage(condensed, method="complete")
            labels = fcluster(hierarchy, t=1.0 - SITE_JACCARD_THRESHOLD, criterion="distance")
        for temporary_label in sorted(set(labels)):
            indices = np.flatnonzero(labels == temporary_label).tolist()
            members = [sets[index] for index in indices]
            weights = group.loc[indices, "source_observations"].to_numpy(dtype=float)
            medoid_candidates = []
            for member_index, member in enumerate(members):
                weighted_distance = np.average(
                    [1.0 - jaccard(member, other) for other in members], weights=weights,
                )
                medoid_candidates.append((weighted_distance, ";".join(map(str, member)), member_index))
            _distance, medoid_text, _member_index = min(medoid_candidates)
            member_texts = sorted(group.loc[indices, "binding_uniprot_positions"].tolist())
            cluster_id = stable_id("BLC_", uniprot + "|" + "|".join(member_texts))
            pairwise_minimum = 1.0
            if len(members) > 1:
                pairwise_minimum = min(
                    jaccard(members[left], members[right])
                    for left in range(len(members) - 1)
                    for right in range(left + 1, len(members))
                )
            for index in indices:
                assignments.append({
                    "uniprot": uniprot,
                    "binding_uniprot_positions": group.loc[index, "binding_uniprot_positions"],
                    "candidate_site_cluster_id": cluster_id,
                    "cluster_representative_binding_uniprot_positions": medoid_text,
                    "cluster_exact_residue_sets": len(indices),
                    "cluster_source_observations": int(group.loc[indices, "source_observations"].sum()),
                    "cluster_minimum_pairwise_jaccard": pairwise_minimum,
                })
    assignments = pd.DataFrame(assignments)
    if (assignments.cluster_minimum_pairwise_jaccard < SITE_JACCARD_THRESHOLD - 1e-12).any():
        raise RuntimeError("complete-linkage site cluster violates the frozen Jaccard threshold")
    return assignments


def add_ranks(frame: pd.DataFrame, score_column: str, rank_prefix: str) -> pd.DataFrame:
    output = frame.copy()
    valid = output[score_column].notna()
    output.loc[valid, f"{rank_prefix}_global_rank"] = output.loc[valid, score_column].rank(
        method="min", ascending=False,
    )
    output.loc[valid, f"{rank_prefix}_global_percentile"] = (
        output.loc[valid, f"{rank_prefix}_global_rank"] - 1
    ) / max(1, int(valid.sum()) - 1)
    output.loc[valid, f"{rank_prefix}_within_protein_rank"] = output.loc[valid].groupby(
        "uniprot", sort=False
    )[score_column].rank(method="min", ascending=False)
    group_size = output.loc[valid].groupby("uniprot")[score_column].transform("size")
    output.loc[valid, f"{rank_prefix}_within_protein_percentile"] = np.where(
        group_size > 1,
        (output.loc[valid, f"{rank_prefix}_within_protein_rank"] - 1) / (group_size - 1),
        0.0,
    )
    output.loc[valid, f"{rank_prefix}_score_tie_size"] = output.loc[valid].groupby(
        ["uniprot", score_column], sort=False
    )[score_column].transform("size")
    return output


def collapse_site_features(frame: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    rows = []
    group_columns = ["uniprot", "full_inchikey", "candidate_site_cluster_id"]
    for (uniprot, full_inchikey, site_cluster), group in frame.groupby(group_columns, sort=True):
        distances = group[SOURCE_DISTANCE].to_numpy(dtype=float)
        median_distance = float(np.median(distances))
        resolution = pd.to_numeric(group.resolution, errors="coerce")
        representative_order = pd.DataFrame({
            "distance_delta": np.abs(distances - median_distance),
            "resolution": resolution.fillna(np.inf).to_numpy(),
            "observation_id": group.observation_id.to_numpy(dtype=str),
        }, index=group.index).sort_values(
            ["distance_delta", "resolution", "observation_id"], kind="mergesort",
        )
        representative = group.loc[representative_order.index[0]]
        row = {
            "candidate_site_ligand_id": stable_id(
                "BLSL_", f"{uniprot}|{full_inchikey}|{site_cluster}"
            ),
            "candidate_pair_id": representative.candidate_pair_id,
            "candidate_site_cluster_id": site_cluster,
            "uniprot": uniprot,
            "full_inchikey": full_inchikey,
            "connectivity_key": representative.connectivity_key,
            "canonical_smiles": representative.canonical_smiles,
            "ligand_ccds": ";".join(sorted(set(group.ligand_ccd))),
            "cluster_representative_binding_uniprot_positions": (
                representative.cluster_representative_binding_uniprot_positions
            ),
            "representative_observation_binding_uniprot_positions": (
                representative.binding_uniprot_positions
            ),
            "representative_observation_id": representative.observation_id,
            "representative_pdb_id": representative.pdb_id,
            "representative_receptor_chain": representative.receptor_chain,
            "representative_ligand_chain": representative.ligand_chain,
            "representative_ligand_auth_seq_id": representative.ligand_auth_seq_id,
            "representative_resolution": (
                float(representative.resolution) if str(representative.resolution) not in {"", "?"}
                else np.nan
            ),
            "source_observations": len(group),
            "source_pdbs": group.pdb_id.nunique(),
            "exact_residue_set_signatures": group.exact_site_ligand_signature_id.nunique(),
            SOURCE_DISTANCE: median_distance,
            "distance_min_across_observations_A": float(np.min(distances)),
            "distance_max_across_observations_A": float(np.max(distances)),
            "distance_IQR_across_observations_A": float(np.quantile(distances, .75) - np.quantile(distances, .25)),
            "fraction_observations_ge_8A": float(np.mean(distances >= 8.0)),
            "fraction_observations_ge_10A": float(np.mean(distances >= 10.0)),
            "fraction_observations_ge_12A": float(np.mean(distances >= 12.0)),
            "median_ligand_to_site_min_heavy_A": float(np.nanmedian(
                pd.to_numeric(group.ligand_to_nearest_orthosteric_site_min_heavy_A, errors="coerce")
            )),
            "any_candidate_site_residue_overlap": bool(
                (pd.to_numeric(
                    group.candidate_nearest_orthosteric_site_residue_overlap_count,
                    errors="coerce",
                ) > 0).any()
            ),
            "nearest_orthosteric_site_definition_ids": ";".join(sorted(set(
                group.nearest_orthosteric_site_definition_id
            ))),
            "molecular_weight": numeric_or_nan(representative.molecular_weight),
            "clogp": numeric_or_nan(representative.clogp),
            "aromatic_ring_count": numeric_or_nan(representative.aromatic_ring_count),
            "heavy_atom_count": numeric_or_nan(representative.heavy_atom_count),
            "descriptor_status": representative.descriptor_status,
        }
        for score_column in score_columns:
            values = pd.to_numeric(group[score_column], errors="coerce")
            row[score_column] = float(values.median()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["uniprot", "connectivity_key", "candidate_site_cluster_id"], kind="mergesort",
    )


def main() -> None:
    started = time.time()
    method_validation = json.loads(METHOD_VALIDATION.read_text())
    checkpoint8 = json.loads(CHECKPOINT8_VALIDATION.read_text())
    geometry_build = json.loads(GEOMETRY_BUILD.read_text())
    descriptor_build = json.loads(DESCRIPTOR_BUILD.read_text())
    if method_validation.get("status") != "validated_likelihood_estimators_awaiting_user_review":
        raise RuntimeError("checkpoint-7 method comparison is not validated")
    if checkpoint8.get("status") != "validated_heldout_ranking_matrix_awaiting_user_review":
        raise RuntimeError("checkpoint-8 evaluation is not validated")
    if geometry_build.get("status") != "validated_candidate_geometry":
        raise RuntimeError("checkpoint-9 geometry is not validated")
    if descriptor_build.get("status") not in {
        "validated_complete", "validated_with_explicit_parse_failures"
    }:
        raise RuntimeError("checkpoint-9 candidate descriptors are not validated")

    reference = pd.read_csv(REFERENCE, sep="\t")
    frozen_reference_scores = pd.read_csv(REFERENCE_SCORES, sep="\t")
    # Keep CCD codes such as the literal ligand identifier "NA" from being
    # converted into missing values by pandas' default NA parser.
    geometry = pd.read_csv(
        GEOMETRY, sep="\t", dtype=str, keep_default_na=False, low_memory=False
    )
    descriptors = pd.read_csv(DESCRIPTORS, sep="\t", dtype=str, keep_default_na=False)
    if len(reference) != 5_798 or len(frozen_reference_scores) != 5_798:
        raise RuntimeError("frozen reference scoring universe changed")
    candidate = geometry.loc[geometry.geometry_status.eq("ok")].copy()
    candidate = candidate.merge(descriptors, on="canonical_smiles", how="left", validate="many_to_one")
    candidate["descriptor_status"] = candidate.descriptor_status.fillna("descriptor_missing")
    for _code, column, _description in FEATURES:
        candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
    if not np.isfinite(candidate[SOURCE_DISTANCE]).all():
        raise RuntimeError("candidate selected distance contains non-finite values")

    reference_labels = reference.binary_label.to_numpy(dtype=int)
    component_records = []
    candidate_component_columns = []
    reference_component_columns = {}
    for code, candidate_column, _description in FEATURES:
        reference_values = reference[REFERENCE_FEATURE[code]].to_numpy(dtype=float)
        candidate_valid = candidate[candidate_column].notna().to_numpy()
        candidate_values = candidate.loc[candidate_valid, candidate_column].to_numpy(dtype=float)

        qnb = fit_qnb(reference_values, reference_labels, code)
        reference_qnb, _ = transform_qnb(qnb, reference_values, code)
        candidate_qnb = np.full(len(candidate), np.nan)
        candidate_qnb[candidate_valid], outside = transform_qnb(qnb, candidate_values, code)
        qnb_column = f"logLR_QNB_{code}"
        candidate[qnb_column] = np.round(candidate_qnb, DECIMALS)
        reference_component_columns[qnb_column] = np.round(reference_qnb, DECIMALS)
        candidate_component_columns.append(qnb_column)
        component_records.append(model_record("QNB", code, qnb, outside))

        kde = fit_kde(reference_values, reference_labels, code)
        reference_raw, _ = transform_kde(kde, reference_values, code)
        candidate_raw = np.full(len(candidate), np.nan)
        candidate_raw[candidate_valid], outside = transform_kde(kde, candidate_values, code)
        for estimator_id in ("KDE_LEGACY", "KDE100"):
            cap = CAPS[estimator_id][code]
            column = f"logLR_{estimator_id}_{code}"
            candidate[column] = np.round(np.clip(candidate_raw, -cap, cap), DECIMALS)
            reference_component_columns[column] = np.round(
                np.clip(reference_raw, -cap, cap), DECIMALS
            )
            candidate_component_columns.append(column)
            record = model_record(estimator_id, code, kde, outside)
            record["manual_likelihood_ratio_lower"] = float(math.exp(-cap))
            record["manual_likelihood_ratio_upper"] = float(math.exp(cap))
            component_records.append(record)

    # Exact reproduction of the checkpoint-7 full-reference component scores.
    frozen_lookup = frozen_reference_scores.set_index("observation_id")
    reference_order = reference.observation_id.to_numpy(dtype=str)
    for column, values in reference_component_columns.items():
        expected = frozen_lookup.loc[reference_order, column].to_numpy(dtype=float)
        if not np.allclose(values, expected, atol=1e-10, rtol=0.0):
            maximum = float(np.max(np.abs(values - expected)))
            raise RuntimeError(f"full-reference component reproduction failed for {column}: {maximum}")

    score_columns = []
    ranking_definitions = [{
        "ranking_id": "RAW:RAW_D",
        "estimator_id": "RAW",
        "feature_codes": "D",
        "score_column": "score_RAW_D",
    }]
    candidate["score_RAW_D"] = np.round(candidate[SOURCE_DISTANCE], DECIMALS)
    score_columns.append("score_RAW_D")
    for estimator_id in ESTIMATORS:
        for subset in feature_subsets():
            column = f"score_{estimator_id}_{'_'.join(subset)}"
            components = [f"logLR_{estimator_id}_{code}" for code in subset]
            candidate[column] = candidate[components].sum(axis=1, min_count=len(components))
            candidate[column] = candidate[column].round(DECIMALS)
            score_columns.append(column)
            ranking_definitions.append({
                "ranking_id": f"{estimator_id}:{'+'.join(subset)}",
                "estimator_id": estimator_id,
                "feature_codes": ";".join(subset),
                "score_column": column,
            })

    assignments = cluster_sites(candidate)
    candidate = candidate.merge(
        assignments,
        on=["uniprot", "binding_uniprot_positions"],
        how="left",
        validate="many_to_one",
    )
    if candidate.candidate_site_cluster_id.isna().any():
        raise RuntimeError("a scored observation was not assigned to a site cluster")

    feature_columns = [
        "observation_id", "source_ordinal", "candidate_pair_id",
        "exact_site_ligand_signature_id", "candidate_site_cluster_id", "uniprot",
        "pdb_id", "receptor_chain", "resolution", "binding_site_code", "ligand_ccd",
        "ligand_chain", "ligand_auth_seq_id", "full_inchikey", "connectivity_key",
        "canonical_smiles", "binding_uniprot_positions",
        "cluster_representative_binding_uniprot_positions", SOURCE_DISTANCE,
        "nearest_orthosteric_site_definition_id",
        "nearest_orthosteric_site_uniprot_positions",
        "nearest_orthosteric_site_residue_recovery_fraction",
        "ligand_to_nearest_orthosteric_site_min_CA_A",
        "ligand_to_nearest_orthosteric_site_min_heavy_A",
        "candidate_site_CA_centroid_to_nearest_orthosteric_site_CA_centroid_A",
        "candidate_nearest_orthosteric_site_residue_overlap_count",
        "candidate_nearest_orthosteric_site_residue_jaccard",
        "distance_ge_8A", "distance_ge_10A", "distance_ge_12A",
        "mapping_identity", "mapping_chain_coverage", "mapping_uniprot_coverage",
        "checkpoint4_binding_map_jaccard", "molecular_weight", "clogp",
        "aromatic_ring_count", "heavy_atom_count", "descriptor_status",
    ]
    atomic_tsv(candidate[feature_columns], FEATURES_OUTPUT, compression="gzip")
    atomic_tsv(
        candidate[[*feature_columns, *candidate_component_columns, *score_columns]],
        OBSERVATION_SCORES,
        compression="gzip",
    )
    atomic_tsv(assignments, SITE_ASSIGNMENTS, compression="gzip")
    atomic_tsv(pd.DataFrame(component_records), COMPONENT_MODELS)

    site_features = collapse_site_features(candidate, score_columns)
    atomic_tsv(site_features, SITE_FEATURES, compression="gzip")

    ranking_frames = []
    count_rows = []
    metadata_columns = [
        "candidate_site_ligand_id", "candidate_pair_id", "candidate_site_cluster_id",
        "uniprot", "full_inchikey", "connectivity_key", "canonical_smiles",
        "ligand_ccds", "cluster_representative_binding_uniprot_positions",
        "representative_observation_binding_uniprot_positions", "representative_observation_id",
        "representative_pdb_id", "representative_receptor_chain",
        "representative_ligand_chain", "representative_ligand_auth_seq_id",
        "representative_resolution", "source_observations", "source_pdbs",
        "exact_residue_set_signatures", SOURCE_DISTANCE, "distance_min_across_observations_A",
        "distance_max_across_observations_A", "distance_IQR_across_observations_A",
        "fraction_observations_ge_8A", "fraction_observations_ge_10A",
        "fraction_observations_ge_12A", "median_ligand_to_site_min_heavy_A",
        "any_candidate_site_residue_overlap", "nearest_orthosteric_site_definition_ids",
        "molecular_weight", "clogp", "aromatic_ring_count", "heavy_atom_count",
        "descriptor_status",
    ]
    for definition in ranking_definitions:
        score_column = definition["score_column"]
        ranked = site_features[metadata_columns + [score_column]].copy()
        ranked = ranked.rename(columns={score_column: "ranking_score"})
        ranked.insert(0, "ranking_id", definition["ranking_id"])
        ranked.insert(1, "estimator_id", definition["estimator_id"])
        ranked.insert(2, "feature_codes", definition["feature_codes"])
        ranked = add_ranks(ranked, "ranking_score", "site")
        ranking_frames.append(ranked)
        count_rows.append({
            "ranking_id": definition["ranking_id"],
            "estimator_id": definition["estimator_id"],
            "feature_codes": definition["feature_codes"],
            "site_rows": len(ranked),
            "site_rows_scored": int(ranked.ranking_score.notna().sum()),
            "proteins": ranked.loc[ranked.ranking_score.notna(), "uniprot"].nunique(),
            "pairs": ranked.loc[ranked.ranking_score.notna(), "candidate_pair_id"].nunique(),
        })
    site_rankings = pd.concat(ranking_frames, ignore_index=True)
    atomic_tsv(site_rankings, SITE_RANKINGS, compression="gzip")
    atomic_tsv(pd.DataFrame(count_rows), RANKING_COUNTS)

    pair_keys = ["ranking_id", "candidate_pair_id"]
    pair_summary = site_rankings.groupby(pair_keys, sort=True).agg(
        candidate_sites=("candidate_site_ligand_id", "nunique"),
        source_observations=("source_observations", "sum"),
        median_site_ranking_score=("ranking_score", "median"),
    ).reset_index()
    best_rows = (
        site_rankings.sort_values(
            ["ranking_id", "candidate_pair_id", "ranking_score", "candidate_site_ligand_id"],
            ascending=[True, True, False, True],
            na_position="last",
            kind="mergesort",
        )
        .drop_duplicates(pair_keys, keep="first")
        [[
            "ranking_id", "candidate_pair_id", "estimator_id", "feature_codes",
            "uniprot", "full_inchikey", "connectivity_key", "canonical_smiles",
            "ligand_ccds", "candidate_site_ligand_id", "ranking_score", SOURCE_DISTANCE,
        ]]
        .rename(columns={
            "candidate_site_ligand_id": "best_site_ligand_id",
            "ranking_score": "best_site_ranking_score",
            SOURCE_DISTANCE: "best_site_distance_A",
        })
    )
    no_score = best_rows.best_site_ranking_score.isna()
    best_rows.loc[no_score, "best_site_ligand_id"] = ""
    best_rows.loc[no_score, "best_site_distance_A"] = np.nan
    pair_rankings = pair_summary.merge(best_rows, on=pair_keys, validate="one_to_one")
    pair_rankings["pair_score_definition"] = (
        "maximum candidate-site score; candidate site count is reported"
    )
    pair_rankings = pd.concat(
        [
            add_ranks(group.copy(), "best_site_ranking_score", "pair")
            for _ranking_id, group in pair_rankings.groupby("ranking_id", sort=True)
        ],
        ignore_index=True,
    )
    atomic_tsv(pair_rankings, PAIR_RANKINGS, compression="gzip")

    top_frames = []
    for ranking_id, group in site_rankings.loc[site_rankings.ranking_score.notna()].groupby(
        "ranking_id", sort=True,
    ):
        top = group.sort_values(
            ["site_global_rank", "candidate_site_ligand_id"], kind="mergesort",
        ).head(100)
        top_frames.append(top)
    top_candidates = pd.concat(top_frames, ignore_index=True)
    atomic_tsv(top_candidates, TOP_CANDIDATES, compression="gzip")

    input_rows = []
    for asset_id, path in (
        ("reference_features", REFERENCE),
        ("reference_method_scores", REFERENCE_SCORES),
        ("method_validation", METHOD_VALIDATION),
        ("checkpoint8_validation", CHECKPOINT8_VALIDATION),
        ("candidate_universe", CANDIDATES),
        ("candidate_geometry", GEOMETRY),
        ("candidate_geometry_validation", GEOMETRY_BUILD),
        ("candidate_descriptors", DESCRIPTORS),
        ("candidate_descriptor_validation", DESCRIPTOR_BUILD),
    ):
        input_rows.append({
            "asset_id": asset_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    atomic_tsv(pd.DataFrame(input_rows), INPUT_HASHES)

    scoring_spec = {
        "checkpoint": 9,
        "status": "scoring_contract_frozen",
        "ranking_definitions": len(ranking_definitions),
        "full_reference_fit_rows": len(reference),
        "full_reference_labels": {
            "allosteric": int(reference.binary_label.sum()),
            "orthosteric": int((1 - reference.binary_label).sum()),
        },
        "raw_distance": {
            "definition": SOURCE_DISTANCE,
            "direction": "larger is more spatially separated",
            "hard_classification": False,
            "descriptive_flags_A": [8, 10, 12],
        },
        "site_clustering": {
            "unit": "mapped BioLiP binding-residue set in UniProt coordinates",
            "method": "complete-linkage on one-minus-Jaccard distance within protein",
            "minimum_pairwise_jaccard": SITE_JACCARD_THRESHOLD,
            "single_linkage_chaining": False,
        },
        "site_score_collapse": "median observation score within protein-ligand-site cluster",
        "pair_score_collapse": "maximum site score with site multiplicity reported",
        "candidate_distance_site_choice": "nearest same-protein exact orthosteric-site definition",
        "model_selection_from_candidate_scores": False,
        "probability_claim": False,
        "primary_public_unit": "protein-ligand-site cluster",
    }
    atomic_json(scoring_spec, SCORING_SPEC)

    ok_distances = candidate[SOURCE_DISTANCE].to_numpy(dtype=float)
    raw_top = site_rankings.loc[site_rankings.ranking_id.eq("RAW:RAW_D")].sort_values(
        ["site_global_rank", "candidate_site_ligand_id"], kind="mergesort",
    ).head(10)
    top_text = "\n".join(
        f"- {row.uniprot} / {row.ligand_ccds} / {row.representative_pdb_id}: "
        f"{row[SOURCE_DISTANCE]:.2f} A"
        for _, row in raw_top.iterrows()
    )
    report = f"""# Checkpoint 9: unlabeled BioLiP ranking

Status: **built; independent validation required**

## Coverage

- Frozen candidate observations before geometry: **{len(geometry):,}**
- Geometry-complete observations: **{len(candidate):,}**
- Proteins represented after geometry: **{candidate.uniprot.nunique():,}**
- Exact protein-ligand pairs after geometry: **{candidate[['uniprot', 'full_inchikey']].drop_duplicates().shape[0]:,}**
- Non-chaining protein-level site clusters: **{assignments.candidate_site_cluster_id.nunique():,}**
- Protein-ligand-site ranking rows: **{len(site_features):,}**
- Ranking definitions retained: **{len(ranking_definitions)}**

The selected raw distance remains a continuous ranking value. The 8, 10, and
12 A columns are descriptive filters, not automatic allosteric labels. At the
observation level, {int(np.sum(ok_distances >= 8)):,}, {int(np.sum(ok_distances >= 10)):,},
and {int(np.sum(ok_distances >= 12)):,} rows respectively exceed these distances.

## Ranking construction

All 46 checkpoint-8 definitions were fitted on the complete 5,798-row frozen
reference and then applied without selecting a method from the candidate
results. Repeated observations were clustered by complete linkage on mapped
binding-residue Jaccard similarity; every pair of residue sets in a cluster
must have Jaccard similarity at least {SITE_JACCARD_THRESHOLD:.2f}. The public
site score is the median observation score. A pair-level convenience table is
also supplied, but its maximum-over-sites score must be read together with the
reported number of candidate sites.

Seventeen unusual organoboron, organometallic, or invalid-valence SMILES did
not parse in RDKit. Their observations remain in the direct-distance and
distance-only rankings; rankings containing a missing chemistry descriptor are
NA rather than imputed.

## Highest direct-distance rows (descriptive)

{top_text}

These are spatially distal observations, not confirmed functional allosteric
interactions. The ranking uses deposited asymmetric-unit coordinates and does
not by itself establish biological-assembly accessibility or functional
coupling.
"""
    atomic_text(report, REPORT)

    build = {
        "checkpoint": 9,
        "stage": "unlabelled_scoring",
        "status": "built_awaiting_independent_validation",
        "candidate_rows_frozen": len(geometry),
        "candidate_rows_geometry_ok": len(candidate),
        "candidate_proteins_geometry_ok": candidate.uniprot.nunique(),
        "candidate_pairs_geometry_ok": candidate[["uniprot", "full_inchikey"]].drop_duplicates().shape[0],
        "candidate_exact_signatures_geometry_ok": candidate.exact_site_ligand_signature_id.nunique(),
        "site_clusters": assignments.candidate_site_cluster_id.nunique(),
        "site_ligand_ranking_rows": len(site_features),
        "pair_rows_per_ranking": pair_rankings.loc[
            pair_rankings.ranking_id.eq("RAW:RAW_D"), "candidate_pair_id"
        ].nunique(),
        "ranking_definitions": len(ranking_definitions),
        "site_ranking_long_rows": len(site_rankings),
        "pair_ranking_long_rows": len(pair_rankings),
        "top_candidate_rows": len(top_candidates),
        "descriptor_parse_failure_observations_geometry_ok": int(
            candidate.descriptor_status.ne("ok").sum()
        ),
        "reference_component_reproduction": "exact within absolute tolerance 1e-10",
        "output_hashes": {
            path.name: sha256(path)
            for path in (
                FEATURES_OUTPUT, OBSERVATION_SCORES, SITE_ASSIGNMENTS, SITE_FEATURES,
                SITE_RANKINGS, PAIR_RANKINGS, TOP_CANDIDATES, COMPONENT_MODELS,
                RANKING_COUNTS, SCORING_SPEC, REPORT,
            )
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(build, BUILD)
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
