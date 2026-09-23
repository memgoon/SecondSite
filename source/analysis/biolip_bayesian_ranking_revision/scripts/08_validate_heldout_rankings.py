#!/usr/bin/env python3
"""Independently validate checkpoint-8 held-out BioLiP rankings and metrics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

FEATURES = DATA / "CHECKPOINT7_REFERENCE_FEATURES.tsv.gz"
FOLDS = DATA / "CHECKPOINT8_OBSERVATION_FOLDS.tsv.gz"
OOF_SCORES = DATA / "CHECKPOINT8_OOF_SCORES.tsv.gz"
COMPONENT_MODELS = DATA / "CHECKPOINT8_FOLD_COMPONENT_MODELS.tsv"
FOLD_METRICS = DATA / "CHECKPOINT8_FOLD_METRICS.tsv"
HELDOUT_METRICS = DATA / "CHECKPOINT8_HELDOUT_METRICS.tsv"
PROTEIN_METRICS = DATA / "CHECKPOINT8_PROTEIN_RANK_METRICS.tsv.gz"
BOOTSTRAP = DATA / "CHECKPOINT8_BOOTSTRAP_CI.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT8_EVALUATION_INPUT_HASHES.tsv"
SPEC = MANIFESTS / "CHECKPOINT8_EVALUATION_SPEC.json"
BUILD = VALIDATION / "CHECKPOINT8_EVALUATION_BUILD_SUMMARY.json"
OUTPUT = VALIDATION / "CHECKPOINT8_VALIDATION.json"

DISTANCE = "ligand_centroid_to_orthosteric_site_CA_centroid_A"
FEATURE_COLUMNS = {
    "D": DISTANCE,
    "MW": "molecular_weight",
    "LP": "clogp",
    "AR": "aromatic_ring_count",
}
CODES = tuple(FEATURE_COLUMNS)
CONTINUOUS = {"D", "MW", "LP"}
ESTIMATORS = ("QNB", "KDE_LEGACY", "KDE100")
REGIMES = {"protein_held_out": "protein_fold", "family_held_out": "family_fold"}
ALPHA = 0.5
FLOOR = 1e-12
DECIMALS = 12
REPLICATES = 10000
SEEDS = {"protein_held_out": 20260901, "family_held_out": 20260902}
CAPS = {
    "KDE_LEGACY": {
        "D": math.log(100.0),
        "MW": math.log(10.0),
        "LP": math.log(10.0),
        "AR": math.log(10.0),
    },
    "KDE100": {code: math.log(100.0) for code in CODES},
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


def close(first, second, tolerance=3e-10) -> bool:
    if pd.isna(first) and pd.isna(second):
        return True
    return math.isclose(float(first), float(second), rel_tol=tolerance, abs_tol=tolerance)


def subsets():
    return [
        combination
        for size in range(1, len(CODES) + 1)
        for combination in itertools.combinations(CODES, size)
    ]


def bin_statistics(ids: np.ndarray, labels: np.ndarray, count: int):
    negative = np.bincount(ids[labels == 0], minlength=count).astype(int)
    positive = np.bincount(ids[labels == 1], minlength=count).astype(int)
    p_negative = (negative + ALPHA) / (int(np.sum(labels == 0)) + ALPHA * count)
    p_positive = (positive + ALPHA) / (int(np.sum(labels == 1)) + ALPHA * count)
    return negative, positive, np.log(p_positive / p_negative)


def qnb_component(train: np.ndarray, labels: np.ndarray, test: np.ndarray, code: str):
    if code in CONTINUOUS:
        requested = int(math.ceil(np.cbrt(len(train))))
        edges = np.unique(np.quantile(
            train,
            np.arange(1, requested, dtype=float) / requested,
            method="linear",
        ))
        train_ids = np.digitize(train, edges, right=False)
        test_ids = np.digitize(test, edges, right=False)
        count = len(edges) + 1
        categories = []
        recorded_edges = [float("-inf"), *[float(value) for value in edges], float("inf")]
        unseen = 0
    else:
        categories = sorted(np.unique(train.astype(int)).tolist())
        lookup = {category: index for index, category in enumerate(categories)}
        unseen_id = len(categories)
        train_ids = np.asarray([lookup[int(value)] for value in train], dtype=int)
        test_ids = np.asarray([lookup.get(int(value), unseen_id) for value in test], dtype=int)
        count = len(categories) + 1
        requested = count
        recorded_edges = []
        unseen = int(np.sum(test_ids == unseen_id))
    negative, positive, ratios = bin_statistics(train_ids, labels, count)
    return np.round(ratios[test_ids], DECIMALS), {
        "requested": requested,
        "count": count,
        "edges": recorded_edges,
        "categories": categories,
        "negative": negative,
        "positive": positive,
        "ratios": ratios,
        "unseen": unseen,
    }


def kde_component(train: np.ndarray, labels: np.ndarray, test: np.ndarray, code: str):
    if code in CONTINUOUS:
        negative_fit = gaussian_kde(train[labels == 0], bw_method="scott")
        positive_fit = gaussian_kde(train[labels == 1], bw_method="scott")
        lower, upper = float(np.min(train)), float(np.max(train))
        evaluated = np.clip(test, lower, upper)
        raw = np.log((positive_fit(evaluated) + FLOOR) / (negative_fit(evaluated) + FLOOR))
        return raw, {
            "lower": lower,
            "upper": upper,
            "negative_factor": float(negative_fit.factor),
            "positive_factor": float(positive_fit.factor),
            "categories": [],
            "negative": [],
            "positive": [],
            "ratios": [],
            "count": np.nan,
            "outside": int(np.sum(evaluated != test)),
        }
    categories = sorted(np.unique(train.astype(int)).tolist())
    lookup = {category: index for index, category in enumerate(categories)}
    unseen_id = len(categories)
    train_ids = np.asarray([lookup[int(value)] for value in train], dtype=int)
    test_ids = np.asarray([lookup.get(int(value), unseen_id) for value in test], dtype=int)
    count = len(categories) + 1
    negative, positive, ratios = bin_statistics(train_ids, labels, count)
    return ratios[test_ids], {
        "lower": np.nan,
        "upper": np.nan,
        "negative_factor": np.nan,
        "positive_factor": np.nan,
        "categories": categories,
        "negative": negative.tolist(),
        "positive": positive.tolist(),
        "ratios": ratios.tolist(),
        "count": count,
        "outside": int(np.sum(test_ids == unseen_id)),
    }


def parsed(value):
    return json.loads(value) if isinstance(value, str) else []


def macro(frame: pd.DataFrame, score: str):
    aucs, aps, used_rows = [], [], 0
    for _protein, group in frame.groupby("uniprot", sort=False):
        if group.binary_label.nunique() != 2:
            continue
        aucs.append(roc_auc_score(group.binary_label, group[score]))
        aps.append(average_precision_score(group.binary_label, group[score]))
        used_rows += len(group)
    return float(np.mean(aucs)), float(np.mean(aps)), len(aucs), used_rows


def fold_summary(frame: pd.DataFrame, score: str):
    aucs, aps = [], []
    for _fold, group in frame.groupby("fold", sort=True):
        aucs.append(roc_auc_score(group.binary_label, group[score]))
        aps.append(average_precision_score(group.binary_label, group[score]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1)), float(np.mean(aps)), float(np.std(aps, ddof=1)), len(aucs)


def collapse(frame: pd.DataFrame, score: str):
    grouped = frame.groupby("site_ligand_signature_id", sort=False)
    if grouped.binary_label.nunique().max() != 1 or grouped.uniprot.nunique().max() != 1:
        raise RuntimeError("checkpoint-8 site identity conflict")
    return grouped.agg(
        observation_id=("observation_id", "min"),
        uniprot=("uniprot", "first"),
        family_component_id=("family_component_id", "first"),
        fold=("fold", "first"),
        binary_label=("binary_label", "first"),
        score=(score, "median"),
        source_observations=("observation_id", "size"),
    ).reset_index().rename(columns={"score": score})


def protein_ranks(frame: pd.DataFrame, score: str):
    rows = []
    for protein, group in frame.groupby("uniprot", sort=True):
        if group.binary_label.nunique() != 2:
            continue
        labels = group.binary_label.to_numpy(dtype=int)
        values = group[score].to_numpy(dtype=float)
        ranks = rankdata(-values, method="average")
        first = float(np.min(ranks[labels == 1]))
        rows.append({
            "uniprot": protein,
            "family_component_id": group.family_component_id.iloc[0],
            "fold": int(group.fold.iloc[0]),
            "site_ligand_signatures": len(group),
            "allosteric_signatures": int(labels.sum()),
            "orthosteric_signatures": int(np.sum(labels == 0)),
            "protein_AUROC": float(roc_auc_score(labels, values)),
            "protein_AUPRC": float(average_precision_score(labels, values)),
            "first_allosteric_average_tie_rank": first,
            "reciprocal_rank": float(1.0 / first),
            "normalized_first_allosteric_rank": float((first - 1.0) / max(len(group) - 1, 1)),
            "recall_at_1": float(first <= 1.0),
            "recall_at_3": float(first <= 3.0),
            "recall_at_5": float(first <= 5.0),
        })
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, score: str, ranks: pd.DataFrame):
    labels = frame.binary_label.to_numpy(dtype=int)
    macro_auc, macro_ap, groups, rows = macro(frame, score)
    fold_auc, fold_auc_sd, fold_ap, fold_ap_sd, fold_count = fold_summary(frame, score)
    sites = collapse(frame, score)
    site_labels = sites.binary_label.to_numpy(dtype=int)
    site_macro_auc, site_macro_ap, site_groups, site_rows = macro(sites, score)
    site_fold_auc, site_fold_auc_sd, site_fold_ap, site_fold_ap_sd, _ = fold_summary(sites, score)
    return {
        "observation_rows": len(frame),
        "allosteric_observation_rows": int(labels.sum()),
        "orthosteric_observation_rows": int(np.sum(labels == 0)),
        "observation_pooled_AUROC": float(roc_auc_score(labels, frame[score])),
        "observation_pooled_AUPRC": float(average_precision_score(labels, frame[score])),
        "observation_fold_macro_AUROC": fold_auc,
        "observation_fold_AUROC_SD": fold_auc_sd,
        "observation_fold_macro_AUPRC": fold_ap,
        "observation_fold_AUPRC_SD": fold_ap_sd,
        "observation_folds": fold_count,
        "observation_within_protein_macro_AUROC": macro_auc,
        "observation_within_protein_macro_AUPRC": macro_ap,
        "observation_within_protein_groups": groups,
        "observation_within_protein_rows": rows,
        "site_ligand_signatures": len(sites),
        "allosteric_site_ligand_signatures": int(site_labels.sum()),
        "orthosteric_site_ligand_signatures": int(np.sum(site_labels == 0)),
        "site_pooled_AUROC": float(roc_auc_score(site_labels, sites[score])),
        "site_pooled_AUPRC": float(average_precision_score(site_labels, sites[score])),
        "site_fold_macro_AUROC": site_fold_auc,
        "site_fold_AUROC_SD": site_fold_auc_sd,
        "site_fold_macro_AUPRC": site_fold_ap,
        "site_fold_AUPRC_SD": site_fold_ap_sd,
        "site_within_protein_macro_AUROC": site_macro_auc,
        "site_within_protein_macro_AUPRC": site_macro_ap,
        "site_within_protein_groups": site_groups,
        "site_within_protein_rows": site_rows,
        "protein_rank_groups": len(ranks),
        "mean_reciprocal_rank": float(ranks.reciprocal_rank.mean()),
        "mean_normalized_first_allosteric_rank": float(ranks.normalized_first_allosteric_rank.mean()),
        "recall_at_1": float(ranks.recall_at_1.mean()),
        "recall_at_3": float(ranks.recall_at_3.mean()),
        "recall_at_5": float(ranks.recall_at_5.mean()),
    }


def validate_row(actual: pd.Series, expected: dict, context: str):
    for column, value in expected.items():
        if column not in actual.index or not close(actual[column], value):
            raise RuntimeError(f"checkpoint-8 metric mismatch: {context}/{column}")


def bootstrap_samples(table: pd.DataFrame, regime: str, ranking: str, value_column: str):
    cluster_column = "uniprot" if regime == "protein_held_out" else "family_component_id"
    raw = table[(table.regime == regime) & (table.ranking_id == "RAW:RAW_D")]
    clusters = sorted(raw[cluster_column].unique())
    lookup = {cluster: index for index, cluster in enumerate(clusters)}
    chosen = table[(table.regime == regime) & (table.ranking_id == ranking)]
    aggregated = chosen.groupby(cluster_column)[value_column].agg(["sum", "count"])
    sums = np.zeros(len(clusters), dtype=float)
    counts = np.zeros(len(clusters), dtype=float)
    for cluster, row in aggregated.iterrows():
        index = lookup[cluster]
        sums[index] = float(row["sum"])
        counts[index] = float(row["count"])
    rng = np.random.default_rng(SEEDS[regime])
    draws = rng.integers(0, len(clusters), size=(REPLICATES, len(clusters)), dtype=np.int32)
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1), len(clusters)


def main() -> None:
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    if build.get("status") != "complete_pending_independent_validation":
        raise RuntimeError("checkpoint-8 evaluation build status changed")
    for asset_id, record in build["outputs"].items():
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint-8 evaluation output hash mismatch: {asset_id}")

    inputs = pd.read_csv(INPUT_HASHES, sep="\t", dtype=str, keep_default_na=False)
    if len(inputs) != 8 or inputs.asset_id.duplicated().any():
        raise RuntimeError("checkpoint-8 evaluation input manifest changed")
    for row in inputs.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or str(path.stat().st_size) != row.size_bytes or sha256(path) != row.sha256:
            raise RuntimeError(f"checkpoint-8 evaluation input hash mismatch: {row.asset_id}")

    features = pd.read_csv(FEATURES, sep="\t", low_memory=False)
    folds = pd.read_csv(FOLDS, sep="\t")
    oof = pd.read_csv(OOF_SCORES, sep="\t", low_memory=False)
    models = pd.read_csv(COMPONENT_MODELS, sep="\t")
    fold_metrics = pd.read_csv(FOLD_METRICS, sep="\t")
    heldout = pd.read_csv(HELDOUT_METRICS, sep="\t")
    reported_protein = pd.read_csv(PROTEIN_METRICS, sep="\t")
    reported_bootstrap = pd.read_csv(BOOTSTRAP, sep="\t")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))

    reference = features.merge(
        folds[["observation_id", "family_component_id", "protein_fold", "family_fold"]],
        on="observation_id", how="inner", validate="one_to_one"
    )
    if len(reference) != 5798 or len(oof) != 11596:
        raise RuntimeError("checkpoint-8 OOF universe changed")
    if oof[["regime", "observation_id"]].duplicated().any():
        raise RuntimeError("checkpoint-8 duplicate OOF predictions")
    for regime in REGIMES:
        if set(oof.loc[oof.regime == regime, "observation_id"]) != set(reference.observation_id):
            raise RuntimeError(f"checkpoint-8 OOF coverage changed: {regime}")
    if len(models) != 120 or models[["regime", "fold", "estimator_id", "feature_code"]].duplicated().any():
        raise RuntimeError("checkpoint-8 fold-component model grid changed")

    model_index = models.set_index(["regime", "fold", "estimator_id", "feature_code"])
    ranking_ids = ["RAW:RAW_D"] + [
        f"{estimator}:{'+'.join(subset)}" for estimator in ESTIMATORS for subset in subsets()
    ]
    score_columns = ["score_RAW_D"] + [
        f"score_{estimator}_{'_'.join(subset)}" for estimator in ESTIMATORS for subset in subsets()
    ]
    ranking_to_score = dict(zip(ranking_ids, score_columns))
    if len(heldout) != 92 or set(heldout.ranking_id) != set(ranking_ids):
        raise RuntimeError("checkpoint-8 held-out ranking grid changed")
    if len(fold_metrics) != 460 or len(reported_bootstrap) != 92:
        raise RuntimeError("checkpoint-8 metric table row count changed")

    for regime, fold_column in REGIMES.items():
        for fold in range(5):
            train = reference[reference[fold_column].astype(int) != fold]
            test = reference[reference[fold_column].astype(int) == fold]
            observed = oof[(oof.regime == regime) & (oof.fold.astype(int) == fold)].copy()
            observed = test[["observation_id"]].merge(observed, on="observation_id", validate="one_to_one")
            if len(observed) != len(test):
                raise RuntimeError(f"checkpoint-8 fold coverage mismatch: {regime}/{fold}")
            if not np.allclose(observed.score_RAW_D, test[DISTANCE], atol=2e-11, rtol=2e-11):
                raise RuntimeError(f"checkpoint-8 raw distance mismatch: {regime}/{fold}")
            labels = train.binary_label.to_numpy(dtype=int)
            recomputed = {}
            for code, column in FEATURE_COLUMNS.items():
                train_values = train[column].to_numpy(dtype=float)
                test_values = test[column].to_numpy(dtype=float)
                qnb, qmeta = qnb_component(train_values, labels, test_values, code)
                recomputed[("QNB", code)] = qnb
                qrow = model_index.loc[(regime, fold, "QNB", code)]
                if int(qrow.requested_bins) != qmeta["requested"] or int(qrow.actual_bins_including_unseen_bin) != qmeta["count"]:
                    raise RuntimeError(f"checkpoint-8 QNB bin count mismatch: {regime}/{fold}/{code}")
                if parsed(qrow.negative_bin_counts_json) != qmeta["negative"].tolist() or parsed(qrow.positive_bin_counts_json) != qmeta["positive"].tolist():
                    raise RuntimeError(f"checkpoint-8 QNB class-bin counts mismatch: {regime}/{fold}/{code}")
                if not np.allclose(parsed(qrow.bin_log_lr_json), qmeta["ratios"], atol=3e-11, rtol=3e-11):
                    raise RuntimeError(f"checkpoint-8 QNB log ratios mismatch: {regime}/{fold}/{code}")
                if parsed(qrow.categories_json) != qmeta["categories"] or int(qrow.test_unseen_or_outside_training_rows) != qmeta["unseen"]:
                    raise RuntimeError(f"checkpoint-8 QNB category handling mismatch: {regime}/{fold}/{code}")
                if not pd.isna(qrow.manual_likelihood_ratio_lower) or not pd.isna(qrow.manual_likelihood_ratio_upper):
                    raise RuntimeError(f"checkpoint-8 QNB unexpectedly capped: {regime}/{fold}/{code}")

                uncapped, kmeta = kde_component(train_values, labels, test_values, code)
                for estimator in ("KDE_LEGACY", "KDE100"):
                    limit = CAPS[estimator][code]
                    expected = np.round(np.clip(uncapped, -limit, limit), DECIMALS)
                    recomputed[(estimator, code)] = expected
                    krow = model_index.loc[(regime, fold, estimator, code)]
                    if not close(krow.manual_likelihood_ratio_lower, math.exp(-limit)) or not close(krow.manual_likelihood_ratio_upper, math.exp(limit)):
                        raise RuntimeError(f"checkpoint-8 KDE cap mismatch: {regime}/{fold}/{estimator}/{code}")
                    if int(krow.test_positive_cap_rows) != int(np.sum(uncapped >= limit)) or int(krow.test_negative_cap_rows) != int(np.sum(uncapped <= -limit)):
                        raise RuntimeError(f"checkpoint-8 KDE capped rows mismatch: {regime}/{fold}/{estimator}/{code}")
                    if int(krow.test_unseen_or_outside_training_rows) != kmeta["outside"]:
                        raise RuntimeError(f"checkpoint-8 KDE boundary handling mismatch: {regime}/{fold}/{estimator}/{code}")
                    if code in CONTINUOUS and (
                        not close(krow.negative_kde_factor, kmeta["negative_factor"])
                        or not close(krow.positive_kde_factor, kmeta["positive_factor"])
                    ):
                        raise RuntimeError(f"checkpoint-8 KDE bandwidth mismatch: {regime}/{fold}/{estimator}/{code}")
            for estimator in ESTIMATORS:
                for subset in subsets():
                    column = f"score_{estimator}_{'_'.join(subset)}"
                    expected = np.round(np.column_stack([
                        recomputed[(estimator, code)] for code in subset
                    ]).sum(axis=1), DECIMALS)
                    if not np.allclose(observed[column], expected, atol=4e-11, rtol=4e-11):
                        raise RuntimeError(f"checkpoint-8 OOF score mismatch: {regime}/{fold}/{estimator}/{subset}")

    heldout_index = heldout.set_index(["regime", "ranking_id"])
    fold_index = fold_metrics.set_index(["regime", "ranking_id", "fold"])
    independently_built_protein = []
    for regime in REGIMES:
        regime_frame = oof[oof.regime == regime].copy()
        for ranking_id, score in ranking_to_score.items():
            sites = collapse(regime_frame, score)
            ranks = protein_ranks(sites, score)
            expected = aggregate(regime_frame, score, ranks)
            validate_row(heldout_index.loc[(regime, ranking_id)], expected, f"{regime}/{ranking_id}")
            ranks.insert(0, "ranking_id", ranking_id)
            ranks.insert(0, "regime", regime)
            independently_built_protein.append(ranks)
            for fold in range(5):
                group = regime_frame[regime_frame.fold.astype(int) == fold]
                labels = group.binary_label.to_numpy(dtype=int)
                macro_auc, macro_ap, groups, rows = macro(group, score)
                site_group = collapse(group, score)
                site_labels = site_group.binary_label.to_numpy(dtype=int)
                site_macro_auc, site_macro_ap, site_groups, site_rows = macro(site_group, score)
                fold_ranks = protein_ranks(site_group, score)
                fold_expected = {
                    "observation_rows": len(group),
                    "observation_AUROC": float(roc_auc_score(labels, group[score])),
                    "observation_AUPRC": float(average_precision_score(labels, group[score])),
                    "observation_within_protein_macro_AUROC": macro_auc,
                    "observation_within_protein_macro_AUPRC": macro_ap,
                    "observation_within_protein_groups": groups,
                    "observation_within_protein_rows": rows,
                    "site_ligand_signatures": len(site_group),
                    "site_AUROC": float(roc_auc_score(site_labels, site_group[score])),
                    "site_AUPRC": float(average_precision_score(site_labels, site_group[score])),
                    "site_within_protein_macro_AUROC": site_macro_auc,
                    "site_within_protein_macro_AUPRC": site_macro_ap,
                    "site_within_protein_groups": site_groups,
                    "site_within_protein_rows": site_rows,
                    "mean_reciprocal_rank": float(fold_ranks.reciprocal_rank.mean()),
                    "recall_at_1": float(fold_ranks.recall_at_1.mean()),
                    "recall_at_3": float(fold_ranks.recall_at_3.mean()),
                    "recall_at_5": float(fold_ranks.recall_at_5.mean()),
                }
                validate_row(fold_index.loc[(regime, ranking_id, fold)], fold_expected, f"{regime}/{ranking_id}/{fold}")

    independent_protein = pd.concat(independently_built_protein, ignore_index=True)
    keys = ["regime", "ranking_id", "uniprot"]
    joined = reported_protein.merge(independent_protein, on=keys, suffixes=("_reported", "_independent"), validate="one_to_one")
    if len(joined) != len(reported_protein) or len(joined) != 7636:
        raise RuntimeError("checkpoint-8 per-protein metric coverage changed")
    for column in (
        "family_component_id", "fold", "site_ligand_signatures", "allosteric_signatures",
        "orthosteric_signatures", "protein_AUROC", "protein_AUPRC",
        "first_allosteric_average_tie_rank", "reciprocal_rank",
        "normalized_first_allosteric_rank", "recall_at_1", "recall_at_3", "recall_at_5",
    ):
        first = joined[f"{column}_reported"]
        second = joined[f"{column}_independent"]
        if first.dtype == object or second.dtype == object:
            if not first.astype(str).equals(second.astype(str)):
                raise RuntimeError(f"checkpoint-8 per-protein identity mismatch: {column}")
        elif not np.allclose(first, second, atol=3e-10, rtol=3e-10, equal_nan=True):
            raise RuntimeError(f"checkpoint-8 per-protein metric mismatch: {column}")

    bootstrap_index = reported_bootstrap.set_index(["regime", "ranking_id"])
    for regime in REGIMES:
        raw_auc, cluster_count = bootstrap_samples(independent_protein, regime, "RAW:RAW_D", "protein_AUROC")
        raw_mrr, _ = bootstrap_samples(independent_protein, regime, "RAW:RAW_D", "reciprocal_rank")
        for ranking_id in ranking_ids:
            auc, observed_clusters = bootstrap_samples(independent_protein, regime, ranking_id, "protein_AUROC")
            mrr, _ = bootstrap_samples(independent_protein, regime, ranking_id, "reciprocal_rank")
            delta_auc = auc - raw_auc
            delta_mrr = mrr - raw_mrr
            row = bootstrap_index.loc[(regime, ranking_id)]
            expected = {
                "eligible_clusters": observed_clusters,
                "replicates": REPLICATES,
                "seed": SEEDS[regime],
                "site_within_protein_macro_AUROC_ci_low": float(np.quantile(auc, 0.025)),
                "site_within_protein_macro_AUROC_ci_high": float(np.quantile(auc, 0.975)),
                "delta_AUROC_vs_raw_ci_low": float(np.quantile(delta_auc, 0.025)),
                "delta_AUROC_vs_raw_ci_high": float(np.quantile(delta_auc, 0.975)),
                "bootstrap_probability_delta_AUROC_gt_0": float(np.mean(delta_auc > 0)),
                "mean_reciprocal_rank_ci_low": float(np.quantile(mrr, 0.025)),
                "mean_reciprocal_rank_ci_high": float(np.quantile(mrr, 0.975)),
                "delta_MRR_vs_raw_ci_low": float(np.quantile(delta_mrr, 0.025)),
                "delta_MRR_vs_raw_ci_high": float(np.quantile(delta_mrr, 0.975)),
                "bootstrap_probability_delta_MRR_gt_0": float(np.mean(delta_mrr > 0)),
            }
            validate_row(row, expected, f"bootstrap/{regime}/{ranking_id}")
            if cluster_count != observed_clusters:
                raise RuntimeError(f"checkpoint-8 bootstrap cluster coverage mismatch: {regime}/{ranking_id}")

    if spec.get("preferred_estimator") is not None or spec.get("preferred_feature_subset") is not None:
        raise RuntimeError("checkpoint-8 selected a single ranking unexpectedly")
    if spec.get("primary_metric") != "site-collapsed within-protein macro AUROC":
        raise RuntimeError("checkpoint-8 primary metric changed")
    if spec.get("bootstrap", {}).get("multiplicity_preserved") is not True:
        raise RuntimeError("checkpoint-8 bootstrap multiplicity contract changed")
    if spec.get("recursive_directory_scan_performed") is not False or build.get("recursive_directory_scan_performed") is not False:
        raise RuntimeError("checkpoint-8 bounded-I/O contract changed")

    key_ids = [
        "RAW:RAW_D", "QNB:D", "QNB:D+AR", "QNB:D+LP+AR", "QNB:D+MW+LP+AR",
        "KDE_LEGACY:D", "KDE_LEGACY:D+MW+AR", "KDE_LEGACY:D+MW+LP+AR",
        "KDE100:D", "KDE100:D+MW+AR", "KDE100:D+MW+LP+AR",
    ]
    key_results = {}
    for regime in REGIMES:
        key_results[regime] = {
            ranking: {
                "site_within_protein_macro_AUROC": float(heldout_index.loc[(regime, ranking)].site_within_protein_macro_AUROC),
                "mean_reciprocal_rank": float(heldout_index.loc[(regime, ranking)].mean_reciprocal_rank),
                "recall_at_1": float(heldout_index.loc[(regime, ranking)].recall_at_1),
                "site_pooled_AUROC": float(heldout_index.loc[(regime, ranking)].site_pooled_AUROC),
            }
            for ranking in key_ids
        }
    result = {
        "checkpoint": 8,
        "status": "validated_heldout_ranking_matrix_awaiting_user_review",
        "reference_rows": len(reference),
        "reference_proteins": int(reference.uniprot.nunique()),
        "site_ligand_signatures": int(reference.site_ligand_signature_id.nunique()),
        "both_label_proteins": int(sum(
            group.binary_label.nunique() == 2 for _key, group in reference.groupby("uniprot")
        )),
        "regimes": list(REGIMES),
        "folds_per_regime": 5,
        "ranking_definitions": len(ranking_ids),
        "oof_rows_per_regime": len(reference),
        "fold_component_models": len(models),
        "heldout_metric_rows": len(heldout),
        "fold_metric_rows": len(fold_metrics),
        "protein_metric_rows": len(reported_protein),
        "bootstrap_rows": len(reported_bootstrap),
        "bootstrap_replicates": REPLICATES,
        "primary_metric": spec["primary_metric"],
        "key_results": key_results,
        "preferred_estimator": None,
        "preferred_feature_subset": None,
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
