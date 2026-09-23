#!/usr/bin/env python3
"""Aggregate Arm B OOF predictions and compare them with frozen Arm A."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    return parser.parse_args()


def load_metric_helper(project_root):
    path = project_root / "analysis/allosteric_pair_benchmark_main/scripts/aggregate_main_results.py"
    spec = importlib.util.spec_from_file_location("frozen_main_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def prediction_path(output_root, regime, model, seed, fold):
    return (
        output_root / "fits" / regime / model / "seed_{}".format(seed)
        / "fold_{}".format(fold) / "predictions.tsv"
    )


def summarize(metrics):
    identifier = {"evaluation", "subset", "training_pool", "model", "seed"}
    numeric = [column for column in metrics.columns if column not in identifier]
    rows = []
    for keys, group in metrics.groupby(
        ["evaluation", "subset", "training_pool", "model"], sort=False
    ):
        row = {
            "evaluation": keys[0], "subset": keys[1], "training_pool": keys[2],
            "model": keys[3], "n_seeds": int(len(group)),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
            row[column + "_mean"] = float(values.mean()) if len(values) else None
            row[column + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None
            row[column + "_min"] = float(values.min()) if len(values) else None
            row[column + "_max"] = float(values.max()) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = args.project_root
    package = root / "analysis/allosteric_pair_benchmark_broad_superset"
    output_root = package / "gpu_output/broad_superset"
    index_path = output_root / "TRAINING_INDEX.json"
    core_path = package / "data/REFERENCE_ARM_A_MODEL_READY.tsv.gz"
    reference_path = package / "data/REFERENCE_ARM_A_OOF_PREDICTIONS.tsv.gz"
    if not index_path.is_file() or not core_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("training index, frozen core, or Arm A reference predictions are missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    core = pd.read_csv(core_path, sep="\t", low_memory=False)
    reference = pd.read_csv(reference_path, sep="\t", low_memory=False)
    metric_helper = load_metric_helper(root)
    expected_ids = set(core["main_row_id"].astype(str))

    broad_parts = []
    universe_errors = []
    for regime in index["regimes"]:
        for model in index["models"]:
            for seed in index["seeds"]:
                parts = []
                for fold in index["folds"]:
                    path = prediction_path(output_root, regime, model, seed, fold)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    parts.append(pd.read_csv(path, sep="\t", low_memory=False))
                prediction = pd.concat(parts, ignore_index=True)
                if prediction["main_row_id"].duplicated().any() or set(
                    prediction["main_row_id"].astype(str)
                ) != expected_ids:
                    universe_errors.append("broad|{}|{}|{}".format(regime, model, seed))
                broad_parts.append(prediction)
    broad = pd.concat(broad_parts, ignore_index=True)
    broad["training_pool"] = "Arm B: core + additional input-ready pairs"
    broad["common_unseen_compound"] = broad["unseen_compound"].astype(int)

    reference = reference[
        reference["regime"].isin(index["regimes"])
        & reference["model"].isin(index["models"])
        & reference["seed"].isin(index["seeds"])
    ].copy()
    expected_reference_rows = len(expected_ids) * len(index["regimes"]) * len(index["models"]) * len(index["seeds"])
    if len(reference) != expected_reference_rows:
        raise RuntimeError("Arm A reference prediction count differs from the frozen contract")
    flag_table = broad[[
        "main_row_id", "regime", "model", "seed", "common_unseen_compound"
    ]]
    reference = reference.drop(columns=["common_unseen_compound"], errors="ignore").merge(
        flag_table,
        on=["main_row_id", "regime", "model", "seed"],
        how="left", validate="one_to_one",
    )
    if reference["common_unseen_compound"].isna().any():
        raise RuntimeError("could not transfer Arm B compound-novelty flags to Arm A")
    reference["common_unseen_compound"] = reference["common_unseen_compound"].astype(int)
    reference["training_pool"] = "Arm A: proteins with both labels"

    paired = pd.concat([reference, broad], ignore_index=True)
    metric_rows = []
    evaluation_universes = {}
    for pool_name, pool in paired.groupby("training_pool", sort=False):
        for regime in index["regimes"]:
            source = pool[pool["regime"].eq(regime)]
            evaluations = [
                ("Row-random", source) if regime == "row_random" else ("Unseen-family", source)
            ]
            if regime == "unseen_family":
                evaluations.append((
                    "Unseen-family + unseen-compound",
                    source[source["common_unseen_compound"].eq(1)].copy(),
                ))
            for evaluation, evaluated_source in evaluations:
                for model in index["models"]:
                    for seed in index["seeds"]:
                        prediction = evaluated_source[
                            evaluated_source["model"].eq(model) & evaluated_source["seed"].eq(seed)
                        ].copy()
                        for subset_name, subset in [
                            ("all_test_pairs", prediction),
                            ("two-label-proteins", metric_helper.two_label_subset(prediction)),
                        ]:
                            if not len(subset) or subset["binary_label"].nunique() < 2:
                                continue
                            evaluation_universes[
                                (evaluation, subset_name, pool_name, model, int(seed))
                            ] = tuple(sorted(subset["main_row_id"].astype(str)))
                            row = {
                                "evaluation": evaluation,
                                "subset": subset_name,
                                "training_pool": pool_name,
                                "model": model,
                                "seed": int(seed),
                            }
                            row.update(metric_helper.metric_row(subset))
                            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    summary = summarize(metrics)

    arm_a_name = "Arm A: proteins with both labels"
    arm_b_name = "Arm B: core + additional input-ready pairs"
    delta_rows = []
    for keys, group in metrics.groupby(["evaluation", "subset", "model", "seed"], sort=False):
        values = group.set_index("training_pool")
        if arm_a_name not in values.index or arm_b_name not in values.index:
            continue
        a = values.loc[arm_a_name]
        b = values.loc[arm_b_name]
        row = {
            "evaluation": keys[0], "subset": keys[1], "model": keys[2],
            "seed": int(keys[3]), "test": arm_b_name, "reference": arm_a_name,
        }
        for metric in [
            "allosteric_positive_ap", "orthosteric_positive_ap", "symmetric_ap", "auroc",
            "protein_macro_symmetric_ap", "family_macro_symmetric_ap",
        ]:
            row["delta_" + metric] = float(b[metric] - a[metric])
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)

    cross_pool_errors = []
    for evaluation in metrics["evaluation"].unique():
        for subset in metrics["subset"].unique():
            for model in index["models"]:
                for seed in index["seeds"]:
                    left = evaluation_universes.get((evaluation, subset, arm_a_name, model, int(seed)))
                    right = evaluation_universes.get((evaluation, subset, arm_b_name, model, int(seed)))
                    if left is not None and right is not None and left != right:
                        cross_pool_errors.append("{}|{}|{}|{}".format(evaluation, subset, model, seed))

    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    broad.to_csv(aggregate / "ARM_B_OOF_PREDICTIONS.tsv.gz", sep="\t", index=False, compression="gzip")
    paired.to_csv(aggregate / "PAIRED_ARM_A_B_OOF_PREDICTIONS.tsv.gz", sep="\t", index=False, compression="gzip")
    metrics.to_csv(aggregate / "POOL_METRICS_BY_SEED.tsv", sep="\t", index=False)
    summary.to_csv(aggregate / "POOL_MODEL_SUMMARY.tsv", sep="\t", index=False)
    deltas.to_csv(aggregate / "PAIRED_DELTAS_ARM_B_MINUS_A.tsv", sep="\t", index=False)
    report = {
        "status": "validated" if not universe_errors and not cross_pool_errors else "failed",
        "comparison": "Arm B minus Arm A on identical frozen Arm A OOF test rows",
        "common_unseen_compound_definition": (
            "Connectivity key absent from Arm B training and the locked Arm A validation fold; "
            "the identical resulting test rows are applied to both training pools."
        ),
        "n_arm_b_prediction_rows": int(len(broad)),
        "n_paired_prediction_rows": int(len(paired)),
        "n_metric_rows": int(len(metrics)),
        "prediction_universe_errors": universe_errors,
        "cross_training_pool_universe_errors": cross_pool_errors,
        "outputs": {
            "arm_b_oof": str(aggregate / "ARM_B_OOF_PREDICTIONS.tsv.gz"),
            "paired_oof": str(aggregate / "PAIRED_ARM_A_B_OOF_PREDICTIONS.tsv.gz"),
            "metrics_by_seed": str(aggregate / "POOL_METRICS_BY_SEED.tsv"),
            "summary": str(aggregate / "POOL_MODEL_SUMMARY.tsv"),
            "paired_deltas": str(aggregate / "PAIRED_DELTAS_ARM_B_MINUS_A.tsv"),
        },
    }
    atomic_json(aggregate / "AGGREGATE_VALIDATION.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("Arm A/B aggregate validation failed")


if __name__ == "__main__":
    main()
