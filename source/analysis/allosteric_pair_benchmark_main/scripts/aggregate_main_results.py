#!/usr/bin/env python3
"""Aggregate explicit five-fold prediction files; no directory discovery is used."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def average_precision(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(score, kind="mergesort")[::-1]
    y = y[order]
    score = score[order]
    ends = np.r_[np.where(np.diff(score))[0], len(score) - 1]
    true_positive = np.cumsum(y)[ends]
    precision = true_positive / (ends + 1)
    recall = true_positive / int(y.sum())
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (ranks[y == 1].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def binary_metrics(frame):
    y = frame["binary_label"].astype(int).to_numpy()
    score = frame["p_allosteric"].astype(float).to_numpy()
    allosteric = average_precision(y, score)
    orthosteric = average_precision(1 - y, 1.0 - score)
    return {
        "n_rows": int(len(frame)),
        "n_proteins": int(frame["uniprot"].nunique()),
        "n_family_components": int(frame["family_component_id"].nunique()),
        "allosteric_prevalence": float(y.mean()),
        "allosteric_positive_ap": allosteric,
        "orthosteric_positive_ap": orthosteric,
        "symmetric_ap": None if allosteric is None or orthosteric is None else float((allosteric + orthosteric) / 2.0),
        "auroc": roc_auc(y, score),
    }


def grouped_metrics(frame, column, prefix):
    values = []
    skipped = 0
    for _, group in frame.groupby(column, sort=False):
        row = binary_metrics(group)
        if row["symmetric_ap"] is None:
            skipped += 1
        else:
            values.append(row)
    return {
        prefix + "_symmetric_ap": float(np.mean([row["symmetric_ap"] for row in values])) if values else None,
        prefix + "_allosteric_positive_ap": float(np.mean([row["allosteric_positive_ap"] for row in values])) if values else None,
        prefix + "_orthosteric_positive_ap": float(np.mean([row["orthosteric_positive_ap"] for row in values])) if values else None,
        prefix + "_groups_used": int(len(values)),
        prefix + "_groups_skipped_single_class": int(skipped),
    }


def metric_row(frame):
    result = binary_metrics(frame)
    result.update(grouped_metrics(frame, "uniprot", "protein_macro"))
    result.update(grouped_metrics(frame, "family_component_id", "family_macro"))
    return result


def two_label_subset(frame):
    proteins = set(
        frame.groupby("uniprot")["binary_label"].nunique().loc[lambda value: value.eq(2)].index.astype(str)
    )
    return frame[frame["uniprot"].astype(str).isin(proteins)].copy()


def prediction_path(output_root, regime, model, seed, fold):
    return (
        output_root / "fits" / regime / model / "seed_{}".format(seed)
        / "fold_{}".format(fold) / "predictions.tsv"
    )


def main():
    args = parse_args()
    package = args.project_root / "analysis/allosteric_pair_benchmark_main"
    output_root = package / "gpu_output/main_benchmark"
    index_path = output_root / "TRAINING_INDEX.json"
    model_ready_path = package / "gpu_cache/MODEL_READY.tsv.gz"
    if not index_path.is_file() or not model_ready_path.is_file():
        raise FileNotFoundError("training index or model-ready table is missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(model_ready_path, sep="\t", low_memory=False)
    expected_all = set(frame["main_row_id"].astype(str))

    all_predictions = []
    metric_rows = []
    universe_errors = []
    evaluation_universes = {}
    for regime in index["regimes"]:
        for model in index["models"]:
            for seed in index["seeds"]:
                parts = []
                for fold in index["folds"]:
                    path = prediction_path(output_root, regime, model, seed, fold)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    part = pd.read_csv(path, sep="\t", low_memory=False)
                    parts.append(part)
                prediction = pd.concat(parts, ignore_index=True)
                if prediction["main_row_id"].duplicated().any() or set(prediction["main_row_id"].astype(str)) != expected_all:
                    universe_errors.append("{}|{}|{}".format(regime, model, seed))
                all_predictions.append(prediction)

                evaluations = []
                if regime == "row_random":
                    evaluations.append(("Row-random", prediction))
                else:
                    evaluations.append(("Unseen-family", prediction))
                    evaluations.append((
                        "Unseen-family + unseen-compound",
                        prediction[prediction["unseen_compound"].astype(int).eq(1)].copy(),
                    ))
                for evaluation, subset in evaluations:
                    key = (evaluation, model, seed)
                    evaluation_universes[key] = tuple(sorted(subset["main_row_id"].astype(str)))
                    for subset_name, evaluated in [
                        ("all_test_pairs", subset),
                        ("two-label-proteins", two_label_subset(subset)),
                    ]:
                        if not len(evaluated) or evaluated["binary_label"].nunique() < 2:
                            continue
                        row = {
                            "evaluation": evaluation,
                            "subset": subset_name,
                            "model": model,
                            "seed": int(seed),
                        }
                        row.update(metric_row(evaluated))
                        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    numeric = [
        column for column in metrics.columns
        if column not in {"evaluation", "subset", "model", "seed"}
    ]
    summary_rows = []
    for keys, group in metrics.groupby(["evaluation", "subset", "model"], sort=False):
        row = {"evaluation": keys[0], "subset": keys[1], "model": keys[2], "n_seeds": int(len(group))}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
            row[column + "_mean"] = float(values.mean()) if len(values) else None
            row[column + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None
            row[column + "_min"] = float(values.min()) if len(values) else None
            row[column + "_max"] = float(values.max()) if len(values) else None
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    delta_rows = []
    reference = metrics[metrics["model"].eq("c1")]
    for model in [value for value in index["models"] if value != "c1"]:
        candidate = metrics[metrics["model"].eq(model)]
        merged = candidate.merge(
            reference,
            on=["evaluation", "subset", "seed"], suffixes=("_model", "_c1"), validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            delta_rows.append({
                "evaluation": row.evaluation,
                "subset": row.subset,
                "model": model,
                "reference": "c1",
                "seed": int(row.seed),
                "delta_allosteric_positive_ap": float(row.allosteric_positive_ap_model - row.allosteric_positive_ap_c1),
                "delta_symmetric_ap": float(row.symmetric_ap_model - row.symmetric_ap_c1),
                "delta_auroc": float(row.auroc_model - row.auroc_c1),
                "delta_family_macro_symmetric_ap": float(
                    row.family_macro_symmetric_ap_model - row.family_macro_symmetric_ap_c1
                ),
            })
    deltas = pd.DataFrame(delta_rows)

    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.concat(all_predictions, ignore_index=True)
    prediction_table.to_csv(aggregate_dir / "ALL_OOF_PREDICTIONS.tsv.gz", sep="\t", index=False, compression="gzip")
    metrics.to_csv(aggregate_dir / "OOF_METRICS_BY_SEED.tsv", sep="\t", index=False)
    summary.to_csv(aggregate_dir / "MODEL_SUMMARY.tsv", sep="\t", index=False)
    deltas.to_csv(aggregate_dir / "PAIRED_DELTAS_VS_C1.tsv", sep="\t", index=False)

    cross_model_universe_errors = []
    for evaluation in sorted(metrics["evaluation"].unique()):
        for seed in index["seeds"]:
            universes = {
                model: evaluation_universes.get((evaluation, model, seed), ())
                for model in index["models"]
            }
            if len(set(universes.values())) != 1:
                cross_model_universe_errors.append("{}|{}".format(evaluation, seed))
    report = {
        "status": "validated" if not universe_errors and not cross_model_universe_errors else "failed",
        "reader_facing_evaluations": [
            "Row-random", "Unseen-family", "Unseen-family + unseen-compound"
        ],
        "n_fit_prediction_tables": int(len(all_predictions)),
        "n_metric_rows": int(len(metrics)),
        "n_summary_rows": int(len(summary)),
        "prediction_universe_errors": universe_errors,
        "cross_model_universe_errors": cross_model_universe_errors,
        "outputs": {
            "all_oof_predictions": str(aggregate_dir / "ALL_OOF_PREDICTIONS.tsv.gz"),
            "metrics_by_seed": str(aggregate_dir / "OOF_METRICS_BY_SEED.tsv"),
            "model_summary": str(aggregate_dir / "MODEL_SUMMARY.tsv"),
            "paired_deltas": str(aggregate_dir / "PAIRED_DELTAS_VS_C1.tsv"),
        },
    }
    atomic_json(aggregate_dir / "AGGREGATE_VALIDATION.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("aggregate validation failed")


if __name__ == "__main__":
    main()
