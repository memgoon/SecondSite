#!/usr/bin/env python3
"""Fit and evaluate every frozen BioLiP ranking definition on held-out folds."""

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
from scipy.stats import gaussian_kde, rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
REPORTS = PACKAGE / "reports"
VALIDATION = PACKAGE / "validation"
SCRIPTS = PACKAGE / "scripts"

FEATURES = DATA / "CHECKPOINT7_REFERENCE_FEATURES.tsv.gz"
FOLDS = DATA / "CHECKPOINT8_OBSERVATION_FOLDS.tsv.gz"
PROTEINS = DATA / "CHECKPOINT8_PROTEIN_FAMILY_ASSIGNMENTS.tsv"
SPLIT_VALIDATION = VALIDATION / "CHECKPOINT8_SPLIT_VALIDATION.json"
METHOD_SPEC = MANIFESTS / "CHECKPOINT7_METHOD_SPEC.json"
METHOD_VALIDATION = VALIDATION / "CHECKPOINT7_METHOD_VALIDATION.json"
METHOD_SCRIPT = SCRIPTS / "07b_compare_likelihood_estimators.py"
SPLIT_SCRIPT = SCRIPTS / "08a_prepare_heldout_splits.py"

OOF_SCORES = DATA / "CHECKPOINT8_OOF_SCORES.tsv.gz"
COMPONENT_MODELS = DATA / "CHECKPOINT8_FOLD_COMPONENT_MODELS.tsv"
FOLD_METRICS = DATA / "CHECKPOINT8_FOLD_METRICS.tsv"
HELDOUT_METRICS = DATA / "CHECKPOINT8_HELDOUT_METRICS.tsv"
PROTEIN_METRICS = DATA / "CHECKPOINT8_PROTEIN_RANK_METRICS.tsv.gz"
BOOTSTRAP = DATA / "CHECKPOINT8_BOOTSTRAP_CI.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT8_EVALUATION_INPUT_HASHES.tsv"
EVALUATION_SPEC = MANIFESTS / "CHECKPOINT8_EVALUATION_SPEC.json"
REPORT = REPORTS / "CHECKPOINT8_HELDOUT_EVALUATION.md"
BUILD = VALIDATION / "CHECKPOINT8_EVALUATION_BUILD_SUMMARY.json"

DISTANCE = "ligand_centroid_to_orthosteric_site_CA_centroid_A"
FEATURES_BY_CODE = {
    "D": DISTANCE,
    "MW": "molecular_weight",
    "LP": "clogp",
    "AR": "aromatic_ring_count",
}
FEATURE_CODES = tuple(FEATURES_BY_CODE)
CONTINUOUS = {"D", "MW", "LP"}
ESTIMATORS = ("QNB", "KDE_LEGACY", "KDE100")
REGIMES = {
    "protein_held_out": "protein_fold",
    "family_held_out": "family_fold",
}
N_FOLDS = 5
ALPHA = 0.5
DENSITY_FLOOR = 1e-12
DECIMALS = 12
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEEDS = {"protein_held_out": 20260901, "family_held_out": 20260902}
CAPS = {
    "KDE_LEGACY": {
        "D": math.log(100.0),
        "MW": math.log(10.0),
        "LP": math.log(10.0),
        "AR": math.log(10.0),
    },
    "KDE100": {code: math.log(100.0) for code in FEATURE_CODES},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_tsv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def feature_subsets():
    return [
        combination
        for size in range(1, len(FEATURE_CODES) + 1)
        for combination in itertools.combinations(FEATURE_CODES, size)
    ]


def smoothed_bin_model(bin_ids: np.ndarray, labels: np.ndarray, n_bins: int):
    negative = np.bincount(bin_ids[labels == 0], minlength=n_bins).astype(int)
    positive = np.bincount(bin_ids[labels == 1], minlength=n_bins).astype(int)
    negative_probability = (negative + ALPHA) / (int(np.sum(labels == 0)) + ALPHA * n_bins)
    positive_probability = (positive + ALPHA) / (int(np.sum(labels == 1)) + ALPHA * n_bins)
    return negative, positive, np.log(positive_probability / negative_probability)


def fit_qnb(train_values: np.ndarray, train_labels: np.ndarray, code: str):
    if code in CONTINUOUS:
        requested = int(math.ceil(len(train_values) ** (1.0 / 3.0)))
        internal_edges = np.unique(np.quantile(
            train_values,
            np.linspace(0.0, 1.0, requested + 1),
            method="linear",
        )[1:-1])
        train_bins = np.searchsorted(internal_edges, train_values, side="right")
        n_bins = len(internal_edges) + 1
        categories = []
        edges = [float("-inf"), *[float(value) for value in internal_edges], float("inf")]
        estimator = "pooled_equal_frequency_bins_cube_root_rule"
    else:
        categories = sorted(np.unique(train_values.astype(int)).tolist())
        lookup = {category: index for index, category in enumerate(categories)}
        train_bins = np.asarray([lookup[int(value)] for value in train_values], dtype=int)
        n_bins = len(categories) + 1
        requested = n_bins
        edges = []
        estimator = "observed_categories_plus_unseen_category_bin"
    negative, positive, log_lr = smoothed_bin_model(train_bins, train_labels, n_bins)
    return {
        "estimator": estimator,
        "requested_bins": requested,
        "n_bins": n_bins,
        "internal_edges": internal_edges if code in CONTINUOUS else np.asarray([], dtype=float),
        "edges": edges,
        "categories": categories,
        "negative_counts": negative,
        "positive_counts": positive,
        "bin_log_lr": log_lr,
    }


def transform_qnb(model: dict, test_values: np.ndarray, code: str):
    if code in CONTINUOUS:
        bin_ids = np.searchsorted(model["internal_edges"], test_values, side="right")
        unseen_rows = 0
    else:
        lookup = {category: index for index, category in enumerate(model["categories"])}
        unseen_bin = model["n_bins"] - 1
        bin_ids = np.asarray([lookup.get(int(value), unseen_bin) for value in test_values], dtype=int)
        unseen_rows = int(np.sum(bin_ids == unseen_bin))
    return model["bin_log_lr"][bin_ids], unseen_rows


def fit_kde_component(train_values: np.ndarray, train_labels: np.ndarray, code: str):
    if code in CONTINUOUS:
        negative_fit = gaussian_kde(train_values[train_labels == 0], bw_method="scott")
        positive_fit = gaussian_kde(train_values[train_labels == 1], bw_method="scott")
        return {
            "estimator": "gaussian_kde_scott_training_fold",
            "minimum": float(np.min(train_values)),
            "maximum": float(np.max(train_values)),
            "negative_fit": negative_fit,
            "positive_fit": positive_fit,
            "negative_kde_factor": float(negative_fit.factor),
            "positive_kde_factor": float(positive_fit.factor),
            "categories": [],
            "n_bins": np.nan,
            "negative_counts": np.asarray([], dtype=int),
            "positive_counts": np.asarray([], dtype=int),
            "bin_log_lr": np.asarray([], dtype=float),
        }
    categories = sorted(np.unique(train_values.astype(int)).tolist())
    lookup = {category: index for index, category in enumerate(categories)}
    train_bins = np.asarray([lookup[int(value)] for value in train_values], dtype=int)
    n_bins = len(categories) + 1
    negative, positive, log_lr = smoothed_bin_model(train_bins, train_labels, n_bins)
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


def transform_kde(model: dict, test_values: np.ndarray, code: str):
    if code in CONTINUOUS:
        evaluated = np.clip(test_values, model["minimum"], model["maximum"])
        negative_density = model["negative_fit"](evaluated) + DENSITY_FLOOR
        positive_density = model["positive_fit"](evaluated) + DENSITY_FLOOR
        return np.log(positive_density / negative_density), int(np.sum(evaluated != test_values))
    lookup = {category: index for index, category in enumerate(model["categories"])}
    unseen_bin = int(model["n_bins"]) - 1
    bin_ids = np.asarray([lookup.get(int(value), unseen_bin) for value in test_values], dtype=int)
    return model["bin_log_lr"][bin_ids], int(np.sum(bin_ids == unseen_bin))


def macro_metrics(frame: pd.DataFrame, score_column: str):
    aucs, aps, rows = [], [], 0
    for _protein, group in frame.groupby("uniprot", sort=False):
        if group.binary_label.nunique() != 2:
            continue
        aucs.append(roc_auc_score(group.binary_label, group[score_column]))
        aps.append(average_precision_score(group.binary_label, group[score_column]))
        rows += len(group)
    return float(np.mean(aucs)), float(np.mean(aps)), len(aucs), rows


def fold_macro(frame: pd.DataFrame, score_column: str):
    aucs, aps = [], []
    for _fold, group in frame.groupby("fold", sort=True):
        if group.binary_label.nunique() != 2:
            continue
        aucs.append(roc_auc_score(group.binary_label, group[score_column]))
        aps.append(average_precision_score(group.binary_label, group[score_column]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1)), float(np.mean(aps)), float(np.std(aps, ddof=1)), len(aucs)


def collapse_sites(frame: pd.DataFrame, score_column: str):
    grouped = frame.groupby("site_ligand_signature_id", sort=False)
    if grouped.binary_label.nunique().max() != 1 or grouped.uniprot.nunique().max() != 1:
        raise RuntimeError("site-ligand signature identity conflict")
    if grouped.fold.nunique().max() != 1 or grouped.family_component_id.nunique().max() != 1:
        raise RuntimeError("site-ligand signature crosses a held-out unit")
    return grouped.agg(
        observation_id=("observation_id", "min"),
        uniprot=("uniprot", "first"),
        family_component_id=("family_component_id", "first"),
        fold=("fold", "first"),
        binary_label=("binary_label", "first"),
        score=(score_column, "median"),
        source_observations=("observation_id", "size"),
    ).reset_index().rename(columns={"score": score_column})


def per_protein_ranks(collapsed: pd.DataFrame, score_column: str):
    rows = []
    for protein, group in collapsed.groupby("uniprot", sort=True):
        if group.binary_label.nunique() != 2:
            continue
        scores = group[score_column].to_numpy(dtype=float)
        labels = group.binary_label.to_numpy(dtype=int)
        ranks = rankdata(-scores, method="average")
        first = float(np.min(ranks[labels == 1]))
        denominator = max(len(group) - 1, 1)
        rows.append({
            "uniprot": protein,
            "family_component_id": group.family_component_id.iloc[0],
            "fold": int(group.fold.iloc[0]),
            "site_ligand_signatures": len(group),
            "allosteric_signatures": int(labels.sum()),
            "orthosteric_signatures": int(np.sum(labels == 0)),
            "protein_AUROC": float(roc_auc_score(labels, scores)),
            "protein_AUPRC": float(average_precision_score(labels, scores)),
            "first_allosteric_average_tie_rank": first,
            "reciprocal_rank": float(1.0 / first),
            "normalized_first_allosteric_rank": float((first - 1.0) / denominator),
            "recall_at_1": float(first <= 1.0),
            "recall_at_3": float(first <= 3.0),
            "recall_at_5": float(first <= 5.0),
        })
    return pd.DataFrame(rows)


def aggregate_metrics(frame: pd.DataFrame, score_column: str, protein_rank: pd.DataFrame):
    labels = frame.binary_label.to_numpy(dtype=int)
    macro_auc, macro_ap, macro_groups, macro_rows = macro_metrics(frame, score_column)
    fold_auc, fold_auc_sd, fold_ap, fold_ap_sd, folds = fold_macro(frame, score_column)
    collapsed = collapse_sites(frame, score_column)
    collapsed_labels = collapsed.binary_label.to_numpy(dtype=int)
    collapsed_macro_auc, collapsed_macro_ap, collapsed_groups, collapsed_rows = macro_metrics(
        collapsed, score_column
    )
    collapsed_fold_auc, collapsed_fold_auc_sd, collapsed_fold_ap, collapsed_fold_ap_sd, _ = fold_macro(
        collapsed, score_column
    )
    return {
        "observation_rows": len(frame),
        "allosteric_observation_rows": int(labels.sum()),
        "orthosteric_observation_rows": int(np.sum(labels == 0)),
        "observation_pooled_AUROC": float(roc_auc_score(labels, frame[score_column])),
        "observation_pooled_AUPRC": float(average_precision_score(labels, frame[score_column])),
        "observation_fold_macro_AUROC": fold_auc,
        "observation_fold_AUROC_SD": fold_auc_sd,
        "observation_fold_macro_AUPRC": fold_ap,
        "observation_fold_AUPRC_SD": fold_ap_sd,
        "observation_folds": folds,
        "observation_within_protein_macro_AUROC": macro_auc,
        "observation_within_protein_macro_AUPRC": macro_ap,
        "observation_within_protein_groups": macro_groups,
        "observation_within_protein_rows": macro_rows,
        "site_ligand_signatures": len(collapsed),
        "allosteric_site_ligand_signatures": int(collapsed_labels.sum()),
        "orthosteric_site_ligand_signatures": int(np.sum(collapsed_labels == 0)),
        "site_pooled_AUROC": float(roc_auc_score(collapsed_labels, collapsed[score_column])),
        "site_pooled_AUPRC": float(average_precision_score(collapsed_labels, collapsed[score_column])),
        "site_fold_macro_AUROC": collapsed_fold_auc,
        "site_fold_AUROC_SD": collapsed_fold_auc_sd,
        "site_fold_macro_AUPRC": collapsed_fold_ap,
        "site_fold_AUPRC_SD": collapsed_fold_ap_sd,
        "site_within_protein_macro_AUROC": collapsed_macro_auc,
        "site_within_protein_macro_AUPRC": collapsed_macro_ap,
        "site_within_protein_groups": collapsed_groups,
        "site_within_protein_rows": collapsed_rows,
        "protein_rank_groups": len(protein_rank),
        "mean_reciprocal_rank": float(protein_rank.reciprocal_rank.mean()),
        "mean_normalized_first_allosteric_rank": float(protein_rank.normalized_first_allosteric_rank.mean()),
        "recall_at_1": float(protein_rank.recall_at_1.mean()),
        "recall_at_3": float(protein_rank.recall_at_3.mean()),
        "recall_at_5": float(protein_rank.recall_at_5.mean()),
    }


def per_fold_metrics(frame: pd.DataFrame, score_column: str):
    rows = []
    for fold, group in frame.groupby("fold", sort=True):
        labels = group.binary_label.to_numpy(dtype=int)
        macro_auc, macro_ap, macro_groups, macro_rows = macro_metrics(group, score_column)
        collapsed = collapse_sites(group, score_column)
        collapsed_labels = collapsed.binary_label.to_numpy(dtype=int)
        collapsed_macro_auc, collapsed_macro_ap, collapsed_groups, collapsed_rows = macro_metrics(
            collapsed, score_column
        )
        ranks = per_protein_ranks(collapsed, score_column)
        rows.append({
            "fold": int(fold),
            "observation_rows": len(group),
            "observation_AUROC": float(roc_auc_score(labels, group[score_column])),
            "observation_AUPRC": float(average_precision_score(labels, group[score_column])),
            "observation_within_protein_macro_AUROC": macro_auc,
            "observation_within_protein_macro_AUPRC": macro_ap,
            "observation_within_protein_groups": macro_groups,
            "observation_within_protein_rows": macro_rows,
            "site_ligand_signatures": len(collapsed),
            "site_AUROC": float(roc_auc_score(collapsed_labels, collapsed[score_column])),
            "site_AUPRC": float(average_precision_score(collapsed_labels, collapsed[score_column])),
            "site_within_protein_macro_AUROC": collapsed_macro_auc,
            "site_within_protein_macro_AUPRC": collapsed_macro_ap,
            "site_within_protein_groups": collapsed_groups,
            "site_within_protein_rows": collapsed_rows,
            "mean_reciprocal_rank": float(ranks.reciprocal_rank.mean()) if len(ranks) else np.nan,
            "recall_at_1": float(ranks.recall_at_1.mean()) if len(ranks) else np.nan,
            "recall_at_3": float(ranks.recall_at_3.mean()) if len(ranks) else np.nan,
            "recall_at_5": float(ranks.recall_at_5.mean()) if len(ranks) else np.nan,
        })
    return rows


def bootstrap_primary(protein_metrics: pd.DataFrame, metrics: pd.DataFrame):
    output = []
    metric_index = metrics.set_index(["regime", "ranking_id"])
    for regime in REGIMES:
        regime_rows = protein_metrics[protein_metrics.regime == regime]
        cluster_column = "uniprot" if regime == "protein_held_out" else "family_component_id"
        raw = regime_rows[regime_rows.ranking_id == "RAW:RAW_D"].copy()
        clusters = sorted(raw[cluster_column].unique())
        cluster_lookup = {cluster: index for index, cluster in enumerate(clusters)}
        rng = np.random.default_rng(BOOTSTRAP_SEEDS[regime])
        draw_indices = rng.integers(
            0, len(clusters), size=(BOOTSTRAP_REPLICATES, len(clusters)), dtype=np.int32
        )

        def samples_for(ranking_id: str, column: str):
            ranked = regime_rows[regime_rows.ranking_id == ranking_id]
            aggregates = ranked.groupby(cluster_column)[column].agg(["sum", "count"])
            sums = np.zeros(len(clusters), dtype=float)
            counts = np.zeros(len(clusters), dtype=float)
            for cluster, row in aggregates.iterrows():
                index = cluster_lookup[cluster]
                sums[index] = float(row["sum"])
                counts[index] = float(row["count"])
            numerator = sums[draw_indices].sum(axis=1)
            denominator = counts[draw_indices].sum(axis=1)
            valid = denominator > 0
            return numerator[valid] / denominator[valid], int(np.sum(valid))

        raw_auc, _ = samples_for("RAW:RAW_D", "protein_AUROC")
        raw_mrr, _ = samples_for("RAW:RAW_D", "reciprocal_rank")
        for ranking_id in sorted(regime_rows.ranking_id.unique()):
            auc_samples, valid_auc = samples_for(ranking_id, "protein_AUROC")
            mrr_samples, valid_mrr = samples_for(ranking_id, "reciprocal_rank")
            if valid_auc != BOOTSTRAP_REPLICATES or valid_mrr != BOOTSTRAP_REPLICATES:
                raise RuntimeError(f"undefined checkpoint-8 bootstrap replicate: {regime}/{ranking_id}")
            auc_delta = auc_samples - raw_auc
            mrr_delta = mrr_samples - raw_mrr
            observed = metric_index.loc[(regime, ranking_id)]
            output.append({
                "regime": regime,
                "ranking_id": ranking_id,
                "cluster_unit": "exact_uniprot" if regime == "protein_held_out" else "pfam_family_component",
                "eligible_clusters": len(clusters),
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEEDS[regime],
                "observed_site_within_protein_macro_AUROC": float(observed.site_within_protein_macro_AUROC),
                "site_within_protein_macro_AUROC_ci_low": float(np.quantile(auc_samples, 0.025)),
                "site_within_protein_macro_AUROC_ci_high": float(np.quantile(auc_samples, 0.975)),
                "delta_AUROC_vs_raw": float(observed.site_within_protein_macro_AUROC - metric_index.loc[(regime, "RAW:RAW_D")].site_within_protein_macro_AUROC),
                "delta_AUROC_vs_raw_ci_low": float(np.quantile(auc_delta, 0.025)),
                "delta_AUROC_vs_raw_ci_high": float(np.quantile(auc_delta, 0.975)),
                "bootstrap_probability_delta_AUROC_gt_0": float(np.mean(auc_delta > 0)),
                "observed_mean_reciprocal_rank": float(observed.mean_reciprocal_rank),
                "mean_reciprocal_rank_ci_low": float(np.quantile(mrr_samples, 0.025)),
                "mean_reciprocal_rank_ci_high": float(np.quantile(mrr_samples, 0.975)),
                "delta_MRR_vs_raw": float(observed.mean_reciprocal_rank - metric_index.loc[(regime, "RAW:RAW_D")].mean_reciprocal_rank),
                "delta_MRR_vs_raw_ci_low": float(np.quantile(mrr_delta, 0.025)),
                "delta_MRR_vs_raw_ci_high": float(np.quantile(mrr_delta, 0.975)),
                "bootstrap_probability_delta_MRR_gt_0": float(np.mean(mrr_delta > 0)),
            })
    return pd.DataFrame(output)


def main() -> None:
    start = time.time()
    split_validation = json.loads(SPLIT_VALIDATION.read_text(encoding="utf-8"))
    if split_validation.get("status") != "validated_heldout_splits":
        raise RuntimeError("checkpoint-8 held-out splits are not validated")
    method_validation = json.loads(METHOD_VALIDATION.read_text(encoding="utf-8"))
    if method_validation.get("status") != "validated_likelihood_estimators_awaiting_user_review":
        raise RuntimeError("checkpoint-7 method grid is not validated")

    frame = pd.read_csv(FEATURES, sep="\t", low_memory=False)
    folds = pd.read_csv(FOLDS, sep="\t")
    proteins = pd.read_csv(PROTEINS, sep="\t", keep_default_na=False)
    frame = frame.merge(
        folds[["observation_id", "family_component_id", "protein_fold", "family_fold"]],
        on="observation_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != 5798 or frame.observation_id.duplicated().any():
        raise RuntimeError("checkpoint-8 evaluation universe changed")
    protein_to_family = proteins.set_index("uniprot").family_component_id.to_dict()
    if any(protein_to_family.get(protein) != component for protein, component in zip(frame.uniprot, frame.family_component_id)):
        raise RuntimeError("checkpoint-8 family metadata mismatch")

    subsets = feature_subsets()
    ranking_ids = ["RAW:RAW_D"] + [
        f"{estimator}:{'+'.join(subset)}"
        for estimator in ESTIMATORS
        for subset in subsets
    ]
    score_columns = ["score_RAW_D"] + [
        f"score_{estimator}_{'_'.join(subset)}"
        for estimator in ESTIMATORS
        for subset in subsets
    ]
    identity_columns = [
        "observation_id", "site_ligand_signature_id", "reference_label", "binary_label",
        "uniprot", "family_component_id", "pdb_id", "receptor_chain", "ligand_ccd",
        "ligand_chain", "ligand_auth_seq_id", "full_inchikey", "connectivity_key",
    ]

    oof_frames = []
    model_rows = []
    for regime, fold_column in REGIMES.items():
        for fold in range(N_FOLDS):
            train = frame[frame[fold_column].astype(int) != fold].copy()
            test = frame[frame[fold_column].astype(int) == fold].copy()
            if set(train.uniprot) & set(test.uniprot):
                raise RuntimeError(f"protein leakage during checkpoint-8 evaluation: {regime}/{fold}")
            if regime == "family_held_out" and set(train.family_component_id) & set(test.family_component_id):
                raise RuntimeError(f"family leakage during checkpoint-8 evaluation: {fold}")
            train_labels = train.binary_label.to_numpy(dtype=int)
            if set(train_labels) != {0, 1} or set(test.binary_label.astype(int)) != {0, 1}:
                raise RuntimeError(f"one-class checkpoint-8 fold: {regime}/{fold}")
            out = test[identity_columns].copy()
            out.insert(0, "fold", fold)
            out.insert(0, "regime", regime)
            out["score_RAW_D"] = np.round(test[DISTANCE].to_numpy(dtype=float), DECIMALS)

            fold_components: Dict[Tuple[str, str], np.ndarray] = {}
            for code, feature_column in FEATURES_BY_CODE.items():
                train_values = train[feature_column].to_numpy(dtype=float)
                test_values = test[feature_column].to_numpy(dtype=float)
                qnb_model = fit_qnb(train_values, train_labels, code)
                qnb_raw, qnb_unseen = transform_qnb(qnb_model, test_values, code)
                qnb_score = np.round(qnb_raw, DECIMALS)
                fold_components[("QNB", code)] = qnb_score
                model_rows.append({
                    "regime": regime,
                    "fold": fold,
                    "estimator_id": "QNB",
                    "feature_code": code,
                    "feature_column": feature_column,
                    "train_rows": len(train),
                    "train_allosteric_rows": int(train_labels.sum()),
                    "train_orthosteric_rows": int(np.sum(train_labels == 0)),
                    "test_rows": len(test),
                    "estimator": qnb_model["estimator"],
                    "requested_bins": qnb_model["requested_bins"],
                    "actual_bins_including_unseen_bin": qnb_model["n_bins"],
                    "bin_edges_json": json.dumps(qnb_model["edges"]),
                    "categories_json": json.dumps(qnb_model["categories"]),
                    "negative_bin_counts_json": json.dumps(qnb_model["negative_counts"].tolist()),
                    "positive_bin_counts_json": json.dumps(qnb_model["positive_counts"].tolist()),
                    "bin_log_lr_json": json.dumps(qnb_model["bin_log_lr"].tolist()),
                    "smoothing_alpha": ALPHA,
                    "training_minimum": float(np.min(train_values)),
                    "training_maximum": float(np.max(train_values)),
                    "negative_kde_factor": np.nan,
                    "positive_kde_factor": np.nan,
                    "manual_likelihood_ratio_lower": np.nan,
                    "manual_likelihood_ratio_upper": np.nan,
                    "test_unseen_or_outside_training_rows": qnb_unseen,
                    "test_positive_cap_rows": 0,
                    "test_negative_cap_rows": 0,
                })

                kde_model = fit_kde_component(train_values, train_labels, code)
                kde_raw, kde_unseen = transform_kde(kde_model, test_values, code)
                for estimator in ("KDE_LEGACY", "KDE100"):
                    limit = CAPS[estimator][code]
                    component = np.round(np.clip(kde_raw, -limit, limit), DECIMALS)
                    fold_components[(estimator, code)] = component
                    model_rows.append({
                        "regime": regime,
                        "fold": fold,
                        "estimator_id": estimator,
                        "feature_code": code,
                        "feature_column": feature_column,
                        "train_rows": len(train),
                        "train_allosteric_rows": int(train_labels.sum()),
                        "train_orthosteric_rows": int(np.sum(train_labels == 0)),
                        "test_rows": len(test),
                        "estimator": kde_model["estimator"],
                        "requested_bins": kde_model["n_bins"],
                        "actual_bins_including_unseen_bin": kde_model["n_bins"],
                        "bin_edges_json": json.dumps([kde_model["minimum"], kde_model["maximum"]]) if code in CONTINUOUS else json.dumps([]),
                        "categories_json": json.dumps(kde_model["categories"]),
                        "negative_bin_counts_json": json.dumps(kde_model["negative_counts"].tolist()),
                        "positive_bin_counts_json": json.dumps(kde_model["positive_counts"].tolist()),
                        "bin_log_lr_json": json.dumps(kde_model["bin_log_lr"].tolist()),
                        "smoothing_alpha": ALPHA if code == "AR" else np.nan,
                        "training_minimum": kde_model["minimum"],
                        "training_maximum": kde_model["maximum"],
                        "negative_kde_factor": kde_model["negative_kde_factor"],
                        "positive_kde_factor": kde_model["positive_kde_factor"],
                        "manual_likelihood_ratio_lower": float(math.exp(-limit)),
                        "manual_likelihood_ratio_upper": float(math.exp(limit)),
                        "test_unseen_or_outside_training_rows": kde_unseen,
                        "test_positive_cap_rows": int(np.sum(kde_raw >= limit)),
                        "test_negative_cap_rows": int(np.sum(kde_raw <= -limit)),
                    })

            for estimator in ESTIMATORS:
                for subset in subsets:
                    score_column = f"score_{estimator}_{'_'.join(subset)}"
                    out[score_column] = np.round(
                        np.column_stack([fold_components[(estimator, code)] for code in subset]).sum(axis=1),
                        DECIMALS,
                    )
            oof_frames.append(out)
            print(f"completed {regime} fold {fold}", flush=True)

    oof = pd.concat(oof_frames, ignore_index=True)
    expected_rows = len(frame) * len(REGIMES)
    if len(oof) != expected_rows or oof[["regime", "observation_id"]].duplicated().any():
        raise RuntimeError("checkpoint-8 OOF coverage mismatch")
    if not np.isfinite(oof[score_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("checkpoint-8 OOF score matrix contains non-finite values")

    heldout_rows = []
    fold_metric_rows = []
    protein_metric_frames = []
    ranking_to_column = dict(zip(ranking_ids, score_columns))
    for regime in REGIMES:
        regime_frame = oof[oof.regime == regime].copy()
        for ranking_id, score_column in ranking_to_column.items():
            collapsed = collapse_sites(regime_frame, score_column)
            protein_rank = per_protein_ranks(collapsed, score_column)
            protein_rank.insert(0, "ranking_id", ranking_id)
            protein_rank.insert(0, "regime", regime)
            protein_metric_frames.append(protein_rank)
            heldout_rows.append({
                "regime": regime,
                "ranking_id": ranking_id,
                "estimator_id": ranking_id.split(":", 1)[0],
                "combination_id": ranking_id.split(":", 1)[1],
                "feature_codes": "D" if ranking_id == "RAW:RAW_D" else ";".join(ranking_id.split(":", 1)[1].split("+")),
                "n_features": 1 if ranking_id == "RAW:RAW_D" else len(ranking_id.split(":", 1)[1].split("+")),
                "training_fold_fitted": ranking_id != "RAW:RAW_D",
                **aggregate_metrics(regime_frame, score_column, protein_rank),
            })
            for row in per_fold_metrics(regime_frame, score_column):
                fold_metric_rows.append({
                    "regime": regime,
                    "ranking_id": ranking_id,
                    **row,
                })

    heldout = pd.DataFrame(heldout_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    protein_metrics = pd.concat(protein_metric_frames, ignore_index=True)
    bootstrap = bootstrap_primary(protein_metrics, heldout)

    atomic_tsv(oof, OOF_SCORES, compression="gzip")
    atomic_tsv(pd.DataFrame(model_rows), COMPONENT_MODELS)
    atomic_tsv(fold_metrics, FOLD_METRICS)
    atomic_tsv(heldout, HELDOUT_METRICS)
    atomic_tsv(protein_metrics, PROTEIN_METRICS, compression="gzip")
    atomic_tsv(bootstrap, BOOTSTRAP)

    input_rows = []
    for asset_id, path in (
        ("checkpoint7_reference_features", FEATURES),
        ("checkpoint8_observation_folds", FOLDS),
        ("checkpoint8_protein_assignments", PROTEINS),
        ("checkpoint8_split_validation", SPLIT_VALIDATION),
        ("checkpoint7_method_spec", METHOD_SPEC),
        ("checkpoint7_method_validation", METHOD_VALIDATION),
        ("checkpoint7_method_script", METHOD_SCRIPT),
        ("checkpoint8_split_script", SPLIT_SCRIPT),
    ):
        input_rows.append({
            "asset_id": asset_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    atomic_tsv(pd.DataFrame(input_rows), INPUT_HASHES)

    spec = {
        "checkpoint": 8,
        "stage": "heldout_ranking_evaluation",
        "status": "built_pending_independent_validation",
        "regimes": REGIMES,
        "folds": N_FOLDS,
        "ranking_definitions": len(ranking_ids),
        "raw_distance_baselines": 1,
        "likelihood_estimators": list(ESTIMATORS),
        "feature_subsets_per_estimator": len(subsets),
        "fit_scope": "all bins, category frequencies, and KDE densities are fit only on the four training folds",
        "test_value_handling": {
            "QNB_continuous": "training-fold edge bins extend to negative and positive infinity",
            "QNB_categorical": "training-fold unseen-category bin with Jeffreys smoothing",
            "KDE_continuous": "test values outside training support are evaluated at the nearest training boundary",
            "KDE_categorical": "training-fold unseen-category bin with Jeffreys smoothing",
        },
        "primary_metric": "site-collapsed within-protein macro AUROC",
        "retrieval_metrics": ["mean reciprocal rank", "normalized first-allosteric rank", "Recall@1", "Recall@3", "Recall@5"],
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "protein_held_out_cluster": "exact UniProt",
            "family_held_out_cluster": "Pfam family component",
            "paired_raw_distance_comparison": True,
            "multiplicity_preserved": True,
        },
        "preferred_estimator": None,
        "preferred_feature_subset": None,
        "limitations": [
            "The structural distance definition was chosen after descriptive full-reference comparison in checkpoint 6.",
            "Allosteric and orthosteric exact references retain different source provenance, so chemistry features can encode source composition.",
            "Held-out metrics compare ranking definitions internally; they are not calibrated probabilities.",
        ],
        "recursive_directory_scan_performed": False,
    }
    atomic_json(EVALUATION_SPEC, spec)

    index = heldout.set_index(["regime", "ranking_id"])
    report_lines = [
        "# Checkpoint 8: held-out BioLiP ranking evaluation",
        "",
        "Status: **built; independent validation required; no single ranking selected**",
        "",
        "Every likelihood transformation was fitted on training folds only. The direct",
        "distance ranking was evaluated without fitting. All 46 ranking definitions are",
        "retained so the web resource can expose more than one ranking convention.",
        "",
        "## Key results before independent validation",
        "",
        "| Regime | Ranking | Site within-protein AUROC | MRR | Recall@1 | Site pooled AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    key_rankings = [
        "RAW:RAW_D", "QNB:D", "QNB:D+AR", "QNB:D+LP+AR", "QNB:D+MW+LP+AR",
        "KDE_LEGACY:D", "KDE_LEGACY:D+MW+AR", "KDE_LEGACY:D+MW+LP+AR",
        "KDE100:D", "KDE100:D+MW+AR", "KDE100:D+MW+LP+AR",
    ]
    for regime in REGIMES:
        for ranking_id in key_rankings:
            row = index.loc[(regime, ranking_id)]
            report_lines.append(
                f"| {regime} | {ranking_id} | {row.site_within_protein_macro_AUROC:.3f} | "
                f"{row.mean_reciprocal_rank:.3f} | {row.recall_at_1:.3f} | {row.site_pooled_AUROC:.3f} |"
            )
    report_lines.extend([
        "",
        "The primary within-protein metric uses one median score per exact",
        "protein-ligand-site signature, preventing repeated structures from dominating.",
        "Pooled metrics remain secondary because protein and source composition can drive",
        "cross-protein ordering. Final interpretation must use the independently validated",
        "tables and the cluster-bootstrap intervals.",
    ])
    atomic_text(REPORT, "\n".join(report_lines) + "\n")

    outputs = {
        "oof_scores": OOF_SCORES,
        "component_models": COMPONENT_MODELS,
        "fold_metrics": FOLD_METRICS,
        "heldout_metrics": HELDOUT_METRICS,
        "protein_metrics": PROTEIN_METRICS,
        "bootstrap": BOOTSTRAP,
        "input_hashes": INPUT_HASHES,
        "evaluation_spec": EVALUATION_SPEC,
        "report": REPORT,
    }
    build = {
        "checkpoint": 8,
        "stage": "heldout_ranking_evaluation",
        "status": "complete_pending_independent_validation",
        "elapsed_seconds": round(time.time() - start, 3),
        "reference_rows": len(frame),
        "regime_oof_rows": {regime: int((oof.regime == regime).sum()) for regime in REGIMES},
        "ranking_definitions": len(ranking_ids),
        "fold_component_models": len(model_rows),
        "heldout_metric_rows": len(heldout),
        "fold_metric_rows": len(fold_metrics),
        "protein_metric_rows": len(protein_metrics),
        "bootstrap_rows": len(bootstrap),
        "preferred_estimator": None,
        "preferred_feature_subset": None,
        "recursive_directory_scan_performed": False,
        "outputs": {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in outputs.items()
        },
    }
    atomic_json(BUILD, build)
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
