#!/usr/bin/env python3
"""Paired family-component bootstrap for the Figure 2(c) split penalty.

Two fixed evaluation universes are analyzed:

1. all 4,637 benchmark rows; and
2. rows whose ligand connectivity is absent from training under both the
   row-random and unseen-family regimes.

Within each universe, row-random and unseen-family predictions are evaluated
on identical rows under the same resampled family-component weights.  The
three model seeds are averaged inside every bootstrap replicate.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

# Each worker performs NumPy reductions; nested BLAS threading is unnecessary.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


MODELS = ("ligand", "protein", "c1", "c2", "c3", "d1")
REGIMES = ("row_random", "unseen_family")
METRICS = ("auroc", "symmetric_auprc", "allosteric_auprc")
UNIVERSES = ("all_test_pairs", "dual_novel_pairs")

_FRAME = None
_UNIVERSE_IDS = None


def parse_args():
    root = Path(__file__).resolve().parents[1]
    aggregate = root / "gpu_output" / "main_benchmark" / "aggregate"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=aggregate / "ALL_OOF_PREDICTIONS.tsv.gz",
    )
    parser.add_argument(
        "--reported-metrics",
        type=Path,
        default=aggregate / "OOF_METRICS_BY_SEED.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=aggregate / "SPLIT_PENALTY_FAMILY_BOOTSTRAP.tsv",
    )
    parser.add_argument(
        "--observed-output",
        type=Path,
        default=aggregate / "SPLIT_PENALTY_OBSERVED_BY_SEED.tsv",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=aggregate / "SPLIT_PENALTY_FAMILY_BOOTSTRAP_VALIDATION.json",
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


class RankedMetric:
    """Weighted binary metrics with a fixed score order."""

    def __init__(self, labels, scores, cluster_index):
        self.labels = np.asarray(labels, dtype=np.int8)
        self.scores = np.asarray(scores, dtype=np.float64)
        self.cluster_index = np.asarray(cluster_index, dtype=np.int32)

    def _grouped(self, descending, positive_label, cluster_weights):
        effective_score = self.scores if positive_label == 1 else -self.scores
        order = np.argsort(effective_score, kind="mergesort")
        if descending:
            order = order[::-1]
        score = effective_score[order]
        label = (self.labels[order] == positive_label).astype(np.float64)
        weight = cluster_weights[self.cluster_index[order]].astype(np.float64)
        starts = np.r_[0, np.flatnonzero(np.diff(score)) + 1]
        positive = np.add.reduceat(weight * label, starts)
        negative = np.add.reduceat(weight * (1.0 - label), starts)
        return positive, negative

    def auroc(self, cluster_weights):
        positive, negative = self._grouped(False, 1, cluster_weights)
        total_positive = positive.sum()
        total_negative = negative.sum()
        if total_positive == 0 or total_negative == 0:
            return np.nan
        negative_before = np.cumsum(negative) - negative
        concordant = np.sum(positive * (negative_before + 0.5 * negative))
        return concordant / (total_positive * total_negative)

    def average_precision(self, positive_label, cluster_weights):
        positive, negative = self._grouped(True, positive_label, cluster_weights)
        total_positive = positive.sum()
        if total_positive == 0:
            return np.nan
        cumulative_positive = np.cumsum(positive)
        cumulative_total = np.cumsum(positive + negative)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        return np.sum((positive / total_positive) * precision)

    def all_metrics(self, cluster_weights):
        allosteric_ap = self.average_precision(1, cluster_weights)
        orthosteric_ap = self.average_precision(0, cluster_weights)
        return {
            "auroc": self.auroc(cluster_weights),
            "symmetric_auprc": 0.5 * (allosteric_ap + orthosteric_ap),
            "allosteric_auprc": allosteric_ap,
        }


def load_predictions(path):
    usecols = [
        "main_row_id",
        "uniprot",
        "family_component_id",
        "binary_label",
        "p_allosteric",
        "unseen_compound",
        "regime",
        "model",
        "seed",
    ]
    frame = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    if set(frame["model"]) != set(MODELS):
        raise RuntimeError("unexpected models: {}".format(sorted(frame["model"].unique())))
    if set(frame["regime"]) != set(REGIMES):
        raise RuntimeError("unexpected regimes: {}".format(sorted(frame["regime"].unique())))
    if frame.duplicated(["main_row_id", "regime", "model", "seed"]).any():
        raise RuntimeError("duplicate OOF prediction rows detected")
    return frame


def build_universes(frame):
    novelty = frame[["main_row_id", "regime", "unseen_compound"]].drop_duplicates()
    counts = novelty.groupby(["main_row_id", "regime"])["unseen_compound"].nunique()
    if int(counts.max()) != 1:
        raise RuntimeError("compound-novelty flag changes across model or seed")
    pivot = novelty.pivot(index="main_row_id", columns="regime", values="unseen_compound")
    if pivot.isna().any().any():
        raise RuntimeError("a row is missing from one evaluation regime")
    all_ids = set(pivot.index.astype(str))
    dual_novel = set(
        pivot.index[(pivot["row_random"] == 1) & (pivot["unseen_family"] == 1)].astype(str)
    )
    return {"all_test_pairs": all_ids, "dual_novel_pairs": dual_novel}


def metric_task(task):
    universe, model, replicates, random_seed = task
    frame = _FRAME
    ids = _UNIVERSE_IDS[universe]
    subset = frame[(frame["model"] == model) & frame["main_row_id"].astype(str).isin(ids)].copy()
    seeds = sorted(int(value) for value in subset["seed"].unique())

    reference = subset[
        (subset["seed"] == seeds[0]) & (subset["regime"] == REGIMES[0])
    ].sort_values("main_row_id")
    families = sorted(reference["family_component_id"].astype(str).unique())
    family_to_index = {family: index for index, family in enumerate(families)}
    proteins = int(reference["uniprot"].nunique())
    prevalence = float(reference["binary_label"].mean())

    calculators = {}
    aligned_ids = reference["main_row_id"].astype(str).tolist()
    aligned_labels = reference["binary_label"].astype(int).tolist()
    aligned_families = reference["family_component_id"].astype(str).tolist()
    for seed in seeds:
        for regime in REGIMES:
            current = subset[
                (subset["seed"] == seed) & (subset["regime"] == regime)
            ].sort_values("main_row_id")
            if current["main_row_id"].astype(str).tolist() != aligned_ids:
                raise RuntimeError("prediction universe mismatch for {} {} {}".format(model, seed, regime))
            if current["binary_label"].astype(int).tolist() != aligned_labels:
                raise RuntimeError("label mismatch for {} {} {}".format(model, seed, regime))
            if current["family_component_id"].astype(str).tolist() != aligned_families:
                raise RuntimeError("family mismatch for {} {} {}".format(model, seed, regime))
            calculators[(seed, regime)] = RankedMetric(
                current["binary_label"].to_numpy(),
                current["p_allosteric"].to_numpy(),
                current["family_component_id"].astype(str).map(family_to_index).to_numpy(),
            )

    unit_weights = np.ones(len(families), dtype=np.int32)
    observed_rows = []
    observed = {}
    for seed in seeds:
        for regime in REGIMES:
            values = calculators[(seed, regime)].all_metrics(unit_weights)
            for metric, value in values.items():
                observed[(seed, regime, metric)] = value
        for metric in METRICS:
            row_random = observed[(seed, "row_random", metric)]
            unseen_family = observed[(seed, "unseen_family", metric)]
            observed_rows.append(
                {
                    "universe": universe,
                    "model": model,
                    "seed": seed,
                    "metric": metric,
                    "row_random": row_random,
                    "unseen_family": unseen_family,
                    "delta_unseen_family_minus_row_random": unseen_family - row_random,
                }
            )

    # All models in a universe receive the same component draws.
    universe_offset = 0 if universe == "all_test_pairs" else 1
    rng = np.random.RandomState(random_seed + universe_offset)
    probability = np.full(len(families), 1.0 / len(families), dtype=np.float64)
    draws = rng.multinomial(len(families), probability, size=replicates)
    distributions = {
        metric: np.full(replicates, np.nan, dtype=np.float64) for metric in METRICS
    }
    for replicate in range(replicates):
        weights = draws[replicate]
        seed_deltas = {metric: [] for metric in METRICS}
        for seed in seeds:
            row_random = calculators[(seed, "row_random")].all_metrics(weights)
            unseen_family = calculators[(seed, "unseen_family")].all_metrics(weights)
            for metric in METRICS:
                if np.isfinite(row_random[metric]) and np.isfinite(unseen_family[metric]):
                    seed_deltas[metric].append(unseen_family[metric] - row_random[metric])
        for metric in METRICS:
            if len(seed_deltas[metric]) == len(seeds):
                distributions[metric][replicate] = np.mean(seed_deltas[metric])

    summary_rows = []
    for metric in METRICS:
        values = distributions[metric]
        valid = values[np.isfinite(values)]
        observed_row_random = np.mean(
            [observed[(seed, "row_random", metric)] for seed in seeds]
        )
        observed_unseen_family = np.mean(
            [observed[(seed, "unseen_family", metric)] for seed in seeds]
        )
        low, high = np.percentile(valid, [2.5, 97.5])
        summary_rows.append(
            {
                "universe": universe,
                "model": model,
                "metric": metric,
                "cluster_unit": "family_component_id",
                "n_rows": len(reference),
                "n_proteins": proteins,
                "n_family_components": len(families),
                "allosteric_prevalence": prevalence,
                "n_seeds_averaged_per_replicate": len(seeds),
                "n_bootstrap_replicates": replicates,
                "n_valid_replicates": len(valid),
                "observed_row_random": observed_row_random,
                "observed_unseen_family": observed_unseen_family,
                "observed_delta_unseen_family_minus_row_random": (
                    observed_unseen_family - observed_row_random
                ),
                "bootstrap_mean_delta": float(np.mean(valid)),
                "percentile_95_ci_low": float(low),
                "percentile_95_ci_high": float(high),
                "bootstrap_probability_delta_lt_0": float(np.mean(valid < 0)),
            }
        )
    return summary_rows, observed_rows


def validate_full_metrics(observed, reported_path, tolerance=1e-12):
    reported = pd.read_csv(reported_path, sep="\t")
    reported = reported[reported["subset"] == "all_test_pairs"].copy()
    evaluation_to_regime = {"Row-random": "row_random", "Unseen-family": "unseen_family"}
    metric_columns = {
        "auroc": "auroc",
        "symmetric_auprc": "symmetric_ap",
        "allosteric_auprc": "allosteric_positive_ap",
    }
    lookup = {
        (evaluation_to_regime[row.evaluation], row.model, int(row.seed), metric): float(
            getattr(row, column)
        )
        for row in reported.itertuples(index=False)
        if row.evaluation in evaluation_to_regime
        for metric, column in metric_columns.items()
    }
    errors = []
    full = observed[observed["universe"] == "all_test_pairs"]
    for row in full.itertuples(index=False):
        for regime, value in (
            ("row_random", row.row_random),
            ("unseen_family", row.unseen_family),
        ):
            expected = lookup[(regime, row.model, int(row.seed), row.metric)]
            if not np.isclose(value, expected, atol=tolerance, rtol=0.0):
                errors.append(
                    "{}|{}|{}|{} recomputed={} reported={}".format(
                        regime, row.model, row.seed, row.metric, value, expected
                    )
                )
    return errors


def main():
    global _FRAME, _UNIVERSE_IDS
    args = parse_args()
    if args.replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    _FRAME = load_predictions(args.predictions)
    _FRAME["main_row_id"] = _FRAME["main_row_id"].astype(str)
    _UNIVERSE_IDS = build_universes(_FRAME)

    tasks = [
        (universe, model, args.replicates, args.seed)
        for universe in UNIVERSES
        for model in MODELS
    ]
    workers = max(1, min(args.workers, len(tasks), os.cpu_count() or 1))
    context = mp.get_context("fork")
    results = []
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(metric_task, tasks):
            results.append(result)
            print("completed {}/{} model-universe tasks".format(len(results), len(tasks)), flush=True)

    summary = pd.DataFrame([row for result in results for row in result[0]])
    observed = pd.DataFrame([row for result in results for row in result[1]])
    summary = summary.sort_values(["universe", "model", "metric"]).reset_index(drop=True)
    observed = observed.sort_values(["universe", "model", "seed", "metric"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, sep="\t", index=False)
    observed.to_csv(args.observed_output, sep="\t", index=False)

    validation_errors = validate_full_metrics(observed, args.reported_metrics)
    expected_counts = {
        universe: {
            "n_rows": int(summary.loc[summary["universe"] == universe, "n_rows"].iloc[0]),
            "n_proteins": int(summary.loc[summary["universe"] == universe, "n_proteins"].iloc[0]),
            "n_family_components": int(
                summary.loc[summary["universe"] == universe, "n_family_components"].iloc[0]
            ),
            "allosteric_prevalence": float(
                summary.loc[summary["universe"] == universe, "allosteric_prevalence"].iloc[0]
            ),
        }
        for universe in UNIVERSES
    }
    all_valid = bool((summary["n_valid_replicates"] == args.replicates).all())
    validation = {
        "status": "validated" if not validation_errors and all_valid else "failed",
        "input": str(args.predictions),
        "output": str(args.output),
        "observed_output": str(args.observed_output),
        "cluster_unit": "family_component_id",
        "paired_regime_resampling": True,
        "dual_novel_definition": (
            "connectivity key absent from training under both row_random and unseen_family"
        ),
        "model_seeds_averaged_within_each_replicate": sorted(
            int(value) for value in _FRAME["seed"].unique()
        ),
        "random_seed": args.seed,
        "n_bootstrap_replicates": args.replicates,
        "workers": workers,
        "universe_counts": expected_counts,
        "reported_full_metric_tolerance": 1e-12,
        "reported_full_metric_validation_errors": validation_errors,
        "all_replicates_valid": all_valid,
    }
    args.validation.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if validation["status"] != "validated":
        raise RuntimeError("split-penalty bootstrap validation failed; inspect {}".format(args.validation))
    print(summary.to_string(index=False))
    print("wrote {}".format(args.output))
    print("wrote {}".format(args.observed_output))
    print("wrote {}".format(args.validation))


if __name__ == "__main__":
    main()
