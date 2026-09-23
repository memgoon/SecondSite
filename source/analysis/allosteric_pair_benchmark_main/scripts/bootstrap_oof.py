#!/usr/bin/env python3
"""Family-component cluster bootstrap for seed-ensemble OOF predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def weighted_ap(y, score, cluster_index, cluster_counts, batch_size=256):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")[::-1]
    ordered_y = y[order]
    ordered_cluster = np.asarray(cluster_index, dtype=np.int64)[order]
    ordered_score = score[order]
    ends = np.r_[np.where(np.diff(ordered_score))[0], len(ordered_score) - 1]
    values = []
    for start in range(0, len(cluster_counts), batch_size):
        counts = cluster_counts[start:start + batch_size]
        weight = counts[:, ordered_cluster].astype(np.float64, copy=False)
        cumulative_total = np.cumsum(weight, axis=1)[:, ends]
        cumulative_positive = np.cumsum(weight * ordered_y[None, :], axis=1)[:, ends]
        total_positive = cumulative_positive[:, -1]
        precision = np.divide(
            cumulative_positive, cumulative_total,
            out=np.zeros_like(cumulative_positive), where=cumulative_total > 0,
        )
        increment = np.diff(
            np.concatenate([np.zeros((len(counts), 1)), cumulative_positive], axis=1), axis=1
        )
        ap = np.divide(
            np.sum(increment * precision, axis=1), total_positive,
            out=np.full(len(counts), np.nan), where=total_positive > 0,
        )
        values.append(ap)
    return np.concatenate(values)


def observed_ap(y, score):
    counts = np.ones((1, 1), dtype=np.int16)
    return float(weighted_ap(y, score, np.zeros(len(y), dtype=int), counts)[0])


def two_label_subset(frame):
    proteins = set(
        frame.groupby("uniprot")["binary_label"].nunique().loc[lambda value: value.eq(2)].index.astype(str)
    )
    return frame[frame["uniprot"].astype(str).isin(proteins)].copy()


def summarize_distribution(values):
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    low = float(np.percentile(clean, 2.5))
    high = float(np.percentile(clean, 97.5))
    return {
        "n_valid_replicates": int(len(clean)),
        "bootstrap_mean": float(clean.mean()),
        "percentile_95_ci_low": low,
        "percentile_95_ci_high": high,
        "percentile_95_ci": [low, high],
    }


def main():
    args = parse_args()
    package = args.project_root / "analysis/allosteric_pair_benchmark_main"
    aggregate = package / "gpu_output/main_benchmark/aggregate"
    prediction_path = aggregate / "ALL_OOF_PREDICTIONS.tsv.gz"
    training_index_path = package / "gpu_output/main_benchmark/TRAINING_INDEX.json"
    if not prediction_path.is_file() or not training_index_path.is_file():
        raise FileNotFoundError("OOF predictions or training index is missing")
    predictions = pd.read_csv(prediction_path, sep="\t", low_memory=False)
    training = json.loads(training_index_path.read_text(encoding="utf-8"))
    models = training["models"]

    metadata_columns = [
        "main_row_id", "uniprot", "family_component_id", "full_inchikey",
        "connectivity_key", "class_label", "binary_label", "regime", "model",
    ]
    ensemble = predictions.groupby(metadata_columns, as_index=False).agg(
        p_allosteric=("p_allosteric", "mean"),
        unseen_compound=("unseen_compound", "min"),
        n_seeds=("seed", "nunique"),
    )
    if ensemble["n_seeds"].nunique() != 1 or int(ensemble["n_seeds"].iloc[0]) != len(training["seeds"]):
        raise RuntimeError("seed ensemble is incomplete")

    evaluations = [
        ("Row-random", ensemble[ensemble["regime"].eq("row_random")]),
        ("Unseen-family", ensemble[ensemble["regime"].eq("unseen_family")]),
        (
            "Unseen-family + unseen-compound",
            ensemble[ensemble["regime"].eq("unseen_family") & ensemble["unseen_compound"].astype(int).eq(1)],
        ),
    ]
    rng = np.random.RandomState(args.seed)
    output_rows = []
    delta_rows = []
    for evaluation, source in evaluations:
        for subset_name, subset in [
            ("all_test_pairs", source),
            ("two-label-proteins", two_label_subset(source)),
        ]:
            base = subset[subset["model"].eq(models[0])][
                ["main_row_id", "family_component_id", "binary_label"]
            ].drop_duplicates("main_row_id")
            clusters = sorted(base["family_component_id"].astype(str).unique())
            cluster_map = {value: index for index, value in enumerate(clusters)}
            cluster_index = base["family_component_id"].astype(str).map(cluster_map).to_numpy(dtype=int)
            y = base["binary_label"].to_numpy(dtype=int)
            counts = rng.multinomial(
                len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=args.replicates
            )
            distributions = {}
            for model in models:
                score_table = subset[subset["model"].eq(model)][["main_row_id", "p_allosteric"]]
                aligned = base[["main_row_id"]].merge(
                    score_table, on="main_row_id", how="left", validate="one_to_one"
                )
                if aligned["p_allosteric"].isna().any():
                    raise RuntimeError("model prediction universe differs")
                score = aligned["p_allosteric"].to_numpy(dtype=float)
                allosteric = weighted_ap(y, score, cluster_index, counts)
                orthosteric = weighted_ap(1 - y, 1.0 - score, cluster_index, counts)
                symmetric = (allosteric + orthosteric) / 2.0
                distributions[model] = {
                    "allosteric_positive_ap": allosteric,
                    "symmetric_ap": symmetric,
                }
                for metric, values, observed in [
                    ("allosteric_positive_ap", allosteric, observed_ap(y, score)),
                    (
                        "symmetric_ap", symmetric,
                        (observed_ap(y, score) + observed_ap(1 - y, 1.0 - score)) / 2.0,
                    ),
                ]:
                    row = {
                        "evaluation": evaluation,
                        "subset": subset_name,
                        "model": model,
                        "metric": metric,
                        "observed_seed_ensemble": observed,
                        "n_rows": int(len(base)),
                        "n_family_components": int(len(clusters)),
                    }
                    row.update(summarize_distribution(values))
                    output_rows.append(row)
            if "c1" in distributions:
                for model in [value for value in models if value != "c1"]:
                    for metric in ["allosteric_positive_ap", "symmetric_ap"]:
                        delta = distributions[model][metric] - distributions["c1"][metric]
                        clean = delta[np.isfinite(delta)]
                        row = {
                            "evaluation": evaluation,
                            "subset": subset_name,
                            "model": model,
                            "reference": "c1",
                            "metric": metric,
                            "bootstrap_probability_delta_gt_0": float(np.mean(clean > 0)),
                        }
                        row.update(summarize_distribution(clean))
                        delta_rows.append(row)

    result = pd.DataFrame(output_rows)
    deltas = pd.DataFrame(delta_rows)
    result_path = aggregate / "CLUSTER_BOOTSTRAP_METRICS.tsv"
    delta_path = aggregate / "CLUSTER_BOOTSTRAP_DELTAS_VS_C1.tsv"
    result.to_csv(result_path, sep="\t", index=False)
    deltas.to_csv(delta_path, sep="\t", index=False)
    report = {
        "status": "validated",
        "replicates": int(args.replicates),
        "seed": int(args.seed),
        "cluster_unit": "Pfam connected component",
        "prediction_aggregation": "mean probability across three training seeds before bootstrap",
        "outputs": {"metrics": str(result_path), "deltas_vs_c1": str(delta_path)},
    }
    atomic_json(aggregate / "CLUSTER_BOOTSTRAP_VALIDATION.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
