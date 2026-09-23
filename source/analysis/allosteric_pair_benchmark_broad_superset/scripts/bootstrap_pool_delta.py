#!/usr/bin/env python3
"""Paired Pfam-component bootstrap for Arm B minus Arm A."""

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
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def load_bootstrap_helper(project_root):
    path = project_root / "analysis/allosteric_pair_benchmark_main/scripts/bootstrap_oof.py"
    spec = importlib.util.spec_from_file_location("frozen_main_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    root = args.project_root
    package = root / "analysis/allosteric_pair_benchmark_broad_superset"
    aggregate = package / "gpu_output/broad_superset/aggregate"
    path = aggregate / "PAIRED_ARM_A_B_OOF_PREDICTIONS.tsv.gz"
    index_path = package / "gpu_output/broad_superset/TRAINING_INDEX.json"
    if not path.is_file() or not index_path.is_file():
        raise FileNotFoundError("paired OOF predictions or training index are missing")
    predictions = pd.read_csv(path, sep="\t", low_memory=False)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    helper = load_bootstrap_helper(root)
    metadata = [
        "training_pool", "main_row_id", "uniprot", "family_component_id",
        "full_inchikey", "connectivity_key", "class_label", "binary_label",
        "regime", "model",
    ]
    ensemble = predictions.groupby(metadata, as_index=False).agg(
        p_allosteric=("p_allosteric", "mean"),
        common_unseen_compound=("common_unseen_compound", "min"),
        n_seeds=("seed", "nunique"),
    )
    if ensemble["n_seeds"].nunique() != 1 or int(ensemble["n_seeds"].iloc[0]) != len(index["seeds"]):
        raise RuntimeError("seed ensemble is incomplete")

    arm_a = "Arm A: proteins with both labels"
    arm_b = "Arm B: core + additional input-ready pairs"
    evaluations = [
        ("Row-random", ensemble[ensemble["regime"].eq("row_random")]),
        ("Unseen-family", ensemble[ensemble["regime"].eq("unseen_family")]),
        (
            "Unseen-family + unseen-compound",
            ensemble[
                ensemble["regime"].eq("unseen_family")
                & ensemble["common_unseen_compound"].astype(int).eq(1)
            ],
        ),
    ]
    rng = np.random.RandomState(args.seed)
    rows = []
    for evaluation, source in evaluations:
        for subset_name, subset in [
            ("all_test_pairs", source),
            ("two-label-proteins", helper.two_label_subset(source)),
        ]:
            for model in index["models"]:
                model_source = subset[subset["model"].eq(model)]
                base = model_source[model_source["training_pool"].eq(arm_a)][
                    ["main_row_id", "family_component_id", "binary_label", "p_allosteric"]
                ].drop_duplicates("main_row_id")
                test = model_source[model_source["training_pool"].eq(arm_b)][
                    ["main_row_id", "p_allosteric"]
                ].drop_duplicates("main_row_id")
                aligned = base.merge(
                    test, on="main_row_id", how="inner", validate="one_to_one",
                    suffixes=("_arm_a", "_arm_b"),
                )
                if len(aligned) != len(base) or not len(aligned):
                    raise RuntimeError("Arm A/B bootstrap prediction universes differ")
                clusters = sorted(aligned["family_component_id"].astype(str).unique())
                cluster_map = {value: index for index, value in enumerate(clusters)}
                cluster_index = aligned["family_component_id"].astype(str).map(cluster_map).to_numpy(dtype=int)
                counts = rng.multinomial(
                    len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=args.replicates
                )
                y = aligned["binary_label"].to_numpy(dtype=int)
                score_a = aligned["p_allosteric_arm_a"].to_numpy(dtype=float)
                score_b = aligned["p_allosteric_arm_b"].to_numpy(dtype=float)
                distributions = {}
                for pool_name, score in [(arm_a, score_a), (arm_b, score_b)]:
                    allo = helper.weighted_ap(y, score, cluster_index, counts)
                    ortho = helper.weighted_ap(1 - y, 1.0 - score, cluster_index, counts)
                    distributions[pool_name] = {
                        "allosteric_positive_ap": allo,
                        "symmetric_ap": (allo + ortho) / 2.0,
                    }
                for metric in ["allosteric_positive_ap", "symmetric_ap"]:
                    delta = distributions[arm_b][metric] - distributions[arm_a][metric]
                    clean = delta[np.isfinite(delta)]
                    observed_a = (
                        helper.observed_ap(y, score_a) if metric == "allosteric_positive_ap"
                        else (helper.observed_ap(y, score_a) + helper.observed_ap(1 - y, 1 - score_a)) / 2
                    )
                    observed_b = (
                        helper.observed_ap(y, score_b) if metric == "allosteric_positive_ap"
                        else (helper.observed_ap(y, score_b) + helper.observed_ap(1 - y, 1 - score_b)) / 2
                    )
                    summary = helper.summarize_distribution(clean)
                    rows.append({
                        "evaluation": evaluation,
                        "subset": subset_name,
                        "model": model,
                        "metric": metric,
                        "reference": arm_a,
                        "test": arm_b,
                        "observed_arm_a": float(observed_a),
                        "observed_arm_b": float(observed_b),
                        "observed_delta_arm_b_minus_a": float(observed_b - observed_a),
                        "bootstrap_probability_delta_gt_0": float(np.mean(clean > 0)),
                        "n_rows": int(len(aligned)),
                        "n_family_components": int(len(clusters)),
                        **summary,
                    })
    result = pd.DataFrame(rows)
    output_path = aggregate / "PAIRED_CLUSTER_BOOTSTRAP_ARM_B_MINUS_A.tsv"
    result.to_csv(output_path, sep="\t", index=False)
    report = {
        "status": "validated",
        "replicates": int(args.replicates),
        "seed": int(args.seed),
        "cluster_unit": "frozen Arm A Pfam connected component",
        "prediction_aggregation": "mean probability across three matched training seeds",
        "comparison": "Arm B minus Arm A on identical test rows",
        "output": str(output_path),
    }
    atomic_json(aggregate / "BOOTSTRAP_VALIDATION.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
