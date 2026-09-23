#!/usr/bin/env python3
"""Combine both model seeds and report random-draw directional stability."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["ligand", "c2", "c3"]
DRAWS = [
    "random_seed_20260823",
    "random_seed_20260824",
    "random_seed_20260825",
]
MODEL_SEEDS = [20260817, 20260818]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_row(frame, metric_module):
    y = frame["binary_label"].astype(int).to_numpy()
    score = frame["p_allosteric"].astype(float).to_numpy()
    per_protein = []
    for _, group in frame.groupby("uniprot", sort=False):
        if group["binary_label"].nunique() == 2:
            per_protein.append(metric_module.roc_auc(
                group["binary_label"].astype(int).to_numpy(),
                group["p_allosteric"].astype(float).to_numpy(),
            ))
    return {
        "n_rows": int(len(frame)),
        "pooled_allosteric_positive_ap": float(metric_module.average_precision(y, score)),
        "pooled_auroc": float(metric_module.roc_auc(y, score)),
        "protein_macro_auroc": float(np.mean(per_protein)),
        "protein_groups": int(len(per_protein)),
    }


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    args = parser.parse_args()
    root = args.project_root
    package = root / "analysis/property_balanced_chembl"
    ligand_package = root / "analysis/ligand_chemistry_balancing"
    metric_module = load_module(
        "extension_metrics",
        root / "analysis/allosteric_pair_benchmark_main/scripts/aggregate_main_results.py",
    )

    common = pd.read_csv(
        ligand_package / "gpu_output/common_evaluation/ALL_COMMON_EVALUATION_PREDICTIONS.tsv.gz",
        sep="\t", low_memory=False,
    )
    old = pd.read_csv(
        ligand_package / "gpu_output/random_sampling_sensitivity/ALL_PREDICTIONS.tsv.gz",
        sep="\t", low_memory=False,
    )
    extension = pd.read_csv(
        package / "gpu_output/random_extension_scoring/ALL_PREDICTIONS.tsv.gz",
        sep="\t", low_memory=False,
    )

    random_parts = []
    draw23 = common[
        common["training_arm"].eq("random")
        & common["evaluation_universe"].eq("property_balanced_rows")
        & common["seed"].isin(MODEL_SEEDS)
    ].copy()
    draw23["random_draw"] = "random_seed_20260823"
    draw23["model_seed"] = draw23["seed"].astype(int)
    random_parts.append(draw23)
    random_parts.append(old[
        old["random_draw"].isin(DRAWS[1:]) & old["model_seed"].eq(20260817)
    ].copy())
    random_parts.append(extension.copy())
    random_predictions = pd.concat(random_parts, ignore_index=True, sort=False)

    key = ["random_draw", "model_seed", "model", "main_row_id"]
    random_predictions = random_predictions.drop_duplicates(key, keep="last")
    expected = 3022 * 3 * 2 * 3
    if len(random_predictions) != expected or random_predictions.duplicated(key).any():
        raise RuntimeError("combined random prediction contract failed")

    property_predictions = common[
        common["training_arm"].eq("property_balanced")
        & common["evaluation_universe"].eq("property_balanced_rows")
        & common["seed"].isin(MODEL_SEEDS)
    ].copy()
    property_predictions["model_seed"] = property_predictions["seed"].astype(int)
    property_predictions = property_predictions.drop_duplicates(
        ["model_seed", "model", "main_row_id"]
    )
    if len(property_predictions) != 3022 * 2 * 3:
        raise RuntimeError("property comparator prediction contract failed")

    metric_rows = []
    for (draw, model_seed, model), group in random_predictions.groupby(
        ["random_draw", "model_seed", "model"], sort=True
    ):
        row = {"training_arm": "random", "random_draw": draw,
               "model_seed": int(model_seed), "model": model}
        row.update(metric_row(group, metric_module))
        metric_rows.append(row)
    for (model_seed, model), group in property_predictions.groupby(
        ["model_seed", "model"], sort=True
    ):
        row = {"training_arm": "property_balanced", "random_draw": "not_applicable",
               "model_seed": int(model_seed), "model": model}
        row.update(metric_row(group, metric_module))
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    contrast_rows = []
    for model_seed in MODEL_SEEDS:
        random_seed_metrics = metrics[
            metrics["training_arm"].eq("random")
            & metrics["model_seed"].eq(model_seed)
        ]
        property_seed_metrics = metrics[
            metrics["training_arm"].eq("property_balanced")
            & metrics["model_seed"].eq(model_seed)
        ].set_index("model")
        for draw in DRAWS:
            draw_metrics = random_seed_metrics[
                random_seed_metrics["random_draw"].eq(draw)
            ].set_index("model")
            for name, test, reference in [
                ("c2_minus_ligand", "c2", "ligand"),
                ("c3_minus_ligand", "c3", "ligand"),
            ]:
                delta = float(
                    draw_metrics.loc[test, "protein_macro_auroc"]
                    - draw_metrics.loc[reference, "protein_macro_auroc"]
                )
                contrast_rows.append({
                    "model_seed": model_seed,
                    "random_draw": draw,
                    "contrast": name,
                    "metric": "protein_macro_auroc",
                    "delta": delta,
                    "sign": sign(delta),
                })
            for model in MODELS:
                delta = float(
                    property_seed_metrics.loc[model, "protein_macro_auroc"]
                    - draw_metrics.loc[model, "protein_macro_auroc"]
                )
                contrast_rows.append({
                    "model_seed": model_seed,
                    "random_draw": draw,
                    "contrast": "property_minus_random_{}".format(model),
                    "metric": "protein_macro_auroc",
                    "delta": delta,
                    "sign": sign(delta),
                })
    contrasts = pd.DataFrame(contrast_rows)

    stability = []
    for (model_seed, contrast), group in contrasts.groupby(["model_seed", "contrast"]):
        signs = group.sort_values("random_draw")["sign"].astype(int).tolist()
        stability.append({
            "model_seed": int(model_seed),
            "contrast": contrast,
            "signs_across_draws": ",".join(str(x) for x in signs),
            "sign_stable": bool(len(set(signs)) == 1 and 0 not in signs),
            "minimum_delta": float(group["delta"].min()),
            "maximum_delta": float(group["delta"].max()),
        })
    stability_frame = pd.DataFrame(stability)
    initial = stability_frame[stability_frame["model_seed"].eq(20260817)]
    if initial["sign_stable"].all():
        raise RuntimeError("the recorded initial extension trigger is absent")

    output = package / "gpu_output/random_extension_analysis"
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "METRICS.tsv", sep="\t", index=False)
    contrasts.to_csv(output / "CONTRASTS.tsv", sep="\t", index=False)
    stability_frame.to_csv(output / "DIRECTIONAL_STABILITY.tsv", sep="\t", index=False)
    report = {
        "status": "validated",
        "initial_gate_triggered": True,
        "extension_completed": True,
        "additional_fits": 30,
        "model_seeds": MODEL_SEEDS,
        "random_draws": DRAWS,
        "primary_stability_metric": "protein_macro_auroc",
        "seed_20260817_all_contrasts_stable": bool(initial["sign_stable"].all()),
        "seed_20260818_all_contrasts_stable": bool(
            stability_frame[stability_frame["model_seed"].eq(20260818)]["sign_stable"].all()
        ),
        "further_adaptive_training_permitted": False,
        "interpretation": (
            "The extension reports whether directional sensitivity replicates "
            "under a second initialization; it does not replace the primary bootstrap."
        ),
    }
    (output / "PREREGISTERED_EXTENSION_RESULT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
