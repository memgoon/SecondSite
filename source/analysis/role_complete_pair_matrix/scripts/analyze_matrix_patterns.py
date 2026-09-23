#!/usr/bin/env python3
"""Derive split penalties, signal retention, and pair-model lift tables."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ACTIVE_REGIMES = {
    "every_pair": ("row_random", "unseen_family", "unseen_ligand"),
    "protein_anchored": ("row_random", "unseen_family", "unseen_ligand"),
    "protein_ligand_role_complete": (
        "row_random",
        "unseen_family",
        "unseen_ligand",
    ),
}
MODELS = ("ligand", "protein", "c1", "c2", "c3", "d1", "d2", "d3")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    package = args.project_root.resolve() / "analysis/role_complete_pair_matrix"
    output = package / "gpu_output/benchmark/aggregate"
    metrics = pd.read_csv(output / "MATRIX_METRICS.tsv", sep="\t", low_memory=False)
    pooled_secondary = metrics[
        metrics["evaluation_universe"].eq("common_role_complete")
        & metrics["endpoint"].eq("pooled")
    ].copy()
    pooled_required = {
        (arm, regime, model)
        for arm, regimes in ACTIVE_REGIMES.items()
        for regime in regimes
        for model in MODELS
    }
    observed = set(
        pooled_secondary[["cohort_arm", "regime", "model"]].itertuples(
            index=False, name=None
        )
    )
    if observed != pooled_required:
        raise RuntimeError("secondary pooled matrix is incomplete")
    pooled_secondary.to_csv(
        output / "COMMON_STRICT_POOLED_SECONDARY.tsv", sep="\t", index=False
    )

    common = metrics[metrics["evaluation_universe"].eq("common_role_complete")]
    claim_specs = [
        (
            "family_generalization_with_ligand_identity_conditioned_out",
            "ligand_macro_fold_restricted",
            ["row_random", "unseen_family"],
            "protein",
        ),
        (
            "ligand_generalization_with_protein_identity_conditioned_out",
            "protein_macro_fold_restricted",
            ["row_random", "unseen_ligand"],
            "ligand",
        ),
    ]
    primary_parts = []
    for claim_axis, endpoint, regimes, reference in claim_specs:
        value = common[
            common["endpoint"].eq(endpoint) & common["regime"].isin(regimes)
        ].copy()
        value["claim_axis"] = claim_axis
        value["reference_model"] = reference
        primary_parts.append(value)
    primary = pd.concat(primary_parts, ignore_index=True)
    expected_primary = {
        (claim_axis, arm, regime, model)
        for claim_axis, _, regimes, _ in claim_specs
        for arm in ACTIVE_REGIMES
        for regime in regimes
        for model in MODELS
    }
    observed_primary = set(
        primary[["claim_axis", "cohort_arm", "regime", "model"]].itertuples(
            index=False, name=None
        )
    )
    if observed_primary != expected_primary:
        raise RuntimeError("claim-aligned primary matrix is incomplete")
    primary.to_csv(
        output / "COMMON_STRICT_PERFORMANCE_LADDER.tsv", sep="\t", index=False
    )

    double_unseen = metrics[
        metrics["evaluation_universe"].eq("common_family_held_out_ligand_novel")
        & metrics["endpoint"].isin(
            ["pooled", "protein_macro_fold_restricted", "ligand_macro_fold_restricted"]
        )
    ].copy()
    expected_double = {
        (arm, model, endpoint)
        for arm in ["every_pair", "protein_anchored"]
        for model in ["ligand", "protein", "c1", "c2", "c3", "d1", "d2", "d3"]
        for endpoint in [
            "pooled",
            "protein_macro_fold_restricted",
            "ligand_macro_fold_restricted",
        ]
    }
    observed_double = set(
        double_unseen[["cohort_arm", "model", "endpoint"]].itertuples(
            index=False, name=None
        )
    )
    if observed_double != expected_double:
        raise RuntimeError("common double-unseen matrix is incomplete")
    double_unseen.to_csv(
        output / "COMMON_GENERAL_DOUBLE_UNSEEN_PERFORMANCE.tsv", sep="\t", index=False
    )
    hard_primary = double_unseen[
        double_unseen["endpoint"].eq("protein_macro_fold_restricted")
    ].copy()
    if len(hard_primary) != 2 * len(MODELS):
        raise RuntimeError("double-unseen within-protein primary matrix is incomplete")
    if (
        set(hard_primary["n_rows_input"].astype(int)) != {2721}
        or set(hard_primary["n_rows_used"].astype(int)) != {1805}
        or set(hard_primary["n_groups_used"].astype(int)) != {112}
    ):
        raise RuntimeError("double-unseen within-protein support contract changed")
    hard_primary["claim_axis"] = (
        "double_unseen_within_new_family_protein_and_unseen_ligands"
    )
    hard_primary["reference_model"] = "ligand"
    hard_primary["endpoint_role"] = "primary"
    hard_primary.to_csv(
        output / "COMMON_DOUBLE_UNSEEN_PRIMARY.tsv", sep="\t", index=False
    )
    hard_primary_lookup = hard_primary.set_index(["cohort_arm", "model"])
    hard_primary_lifts = []
    for arm in ["every_pair", "protein_anchored"]:
        for model in ["c1", "c2", "c3", "d1", "d2", "d3"]:
            test = float(hard_primary_lookup.loc[(arm, model), "auroc"])
            reference = float(hard_primary_lookup.loc[(arm, "ligand"), "auroc"])
            hard_primary_lifts.append(
                {
                    "cohort_arm": arm,
                    "model": model,
                    "endpoint": "protein_macro_fold_restricted.auroc",
                    "test_model": model,
                    "reference_model": "ligand",
                    "test_value": test,
                    "reference_value": reference,
                    "delta": test - reference,
                    "n_rows_input": 2721,
                    "n_rows_used": 1805,
                    "n_groups_used": 112,
                }
            )
    pd.DataFrame(hard_primary_lifts).to_csv(
        output / "COMMON_DOUBLE_UNSEEN_PAIR_LIFT.tsv", sep="\t", index=False
    )

    retention = []
    for claim_axis, endpoint, regimes, _ in claim_specs:
        lookup = primary[primary["claim_axis"].eq(claim_axis)].set_index(
            ["cohort_arm", "regime", "model"]
        )
        heldout_regime = [value for value in regimes if value != "row_random"][0]
        for arm in ACTIVE_REGIMES:
            for model in MODELS:
                random_auc = float(lookup.loc[(arm, "row_random", model), "auroc"])
                heldout = float(lookup.loc[(arm, heldout_regime, model), "auroc"])
                random_rows_used = int(
                    lookup.loc[(arm, "row_random", model), "n_rows_used"]
                )
                heldout_rows_used = int(
                    lookup.loc[(arm, heldout_regime, model), "n_rows_used"]
                )
                random_groups_used = int(
                    lookup.loc[(arm, "row_random", model), "n_groups_used"]
                )
                heldout_groups_used = int(
                    lookup.loc[(arm, heldout_regime, model), "n_groups_used"]
                )
                denominator = random_auc - 0.5
                retention.append(
                    {
                        "claim_axis": claim_axis,
                        "endpoint": endpoint,
                        "cohort_arm": arm,
                        "model": model,
                        "heldout_regime": heldout_regime,
                        "row_random_auroc": random_auc,
                        "heldout_auroc": heldout,
                        "row_random_rows_used": random_rows_used,
                        "heldout_rows_used": heldout_rows_used,
                        "row_random_groups_used": random_groups_used,
                        "heldout_groups_used": heldout_groups_used,
                        "effective_row_sets_identical": False,
                        "interpretation": (
                            "descriptive conditional change; single-class block "
                            "exclusion gives the two regimes different effective rows"
                        ),
                        "absolute_penalty": heldout - random_auc,
                        "above_chance_signal_retention": (heldout - 0.5) / denominator
                        if denominator > 0
                        else None,
                        "above_chance_signal_retention_is_inferential": False,
                    }
                )
    pd.DataFrame(retention).to_csv(output / "SPLIT_SIGNAL_RETENTION.tsv", sep="\t", index=False)
    pd.DataFrame(retention).to_csv(
        output / "CONDITIONAL_SPLIT_CHANGE_WITH_SUPPORT.tsv",
        sep="\t",
        index=False,
    )

    pooled_split_change = []
    pooled_lookup = pooled_secondary.set_index(
        ["cohort_arm", "regime", "model"]
    )
    for claim_axis, _, regimes, _ in claim_specs:
        heldout_regime = [value for value in regimes if value != "row_random"][0]
        for arm in ACTIVE_REGIMES:
            for model in MODELS:
                for metric in ["auroc", "allosteric_ap", "symmetric_ap"]:
                    random_value = float(
                        pooled_lookup.loc[(arm, "row_random", model), metric]
                    )
                    heldout_value = float(
                        pooled_lookup.loc[(arm, heldout_regime, model), metric]
                    )
                    pooled_split_change.append(
                        {
                            "claim_axis": claim_axis,
                            "cohort_arm": arm,
                            "model": model,
                            "metric": metric,
                            "evaluation_rows": 395,
                            "effective_row_sets_identical": True,
                            "row_random_value": random_value,
                            "heldout_regime": heldout_regime,
                            "heldout_value": heldout_value,
                            "heldout_minus_row_random": heldout_value
                            - random_value,
                            "interpretation": (
                                "identical-row pooled sensitivity; retains between-identity "
                                "prevalence and is not the conditional primary endpoint"
                            ),
                        }
                    )
    pd.DataFrame(pooled_split_change).to_csv(
        output / "POOLED_SPLIT_CHANGE_IDENTICAL_ROWS.tsv",
        sep="\t",
        index=False,
    )

    # Pooled AUROC is retained only as an identical-2,721-row split-change
    # sensitivity.  It is not the hard endpoint: the primary double-unseen
    # statistic above is within-protein AUROC on 1,805 rows / 112 groups.
    hard_metrics = metrics[
        metrics["evaluation_universe"].eq("common_general_hard_rows")
        & metrics["endpoint"].eq("pooled")
    ].copy()
    hard_lookup = hard_metrics.set_index(["cohort_arm", "regime", "model"])
    hard_pooled_split_sensitivity = []
    for arm in ["every_pair", "protein_anchored"]:
        for model in MODELS:
            easy = float(hard_lookup.loc[(arm, "row_random", model), "auroc"])
            hard = float(hard_lookup.loc[(arm, "unseen_family", model), "auroc"])
            hard_pooled_split_sensitivity.append(
                {
                    "cohort_arm": arm,
                    "model": model,
                    "evaluation_rows": 2721,
                    "endpoint": "pooled_auroc",
                    "easy_regime": "row_random",
                    "hard_regime": "family_held_out_and_ligand_unseen",
                    "row_random_auroc": easy,
                    "double_unseen_auroc": hard,
                    "absolute_penalty": hard - easy,
                    "above_chance_signal_retention": (hard - 0.5) / (easy - 0.5)
                    if easy > 0.5
                    else None,
                    "endpoint_role": "secondary_identical_row_split_sensitivity",
                    "interpretation": (
                        "uses all matched rows but retains between-protein prevalence; "
                        "not the primary double-unseen endpoint"
                    ),
                }
            )
    hard_pooled_frame = pd.DataFrame(hard_pooled_split_sensitivity)
    hard_pooled_frame.to_csv(
        output / "COMMON_HARD_ROW_POOLED_SPLIT_SENSITIVITY.tsv",
        sep="\t",
        index=False,
    )

    lifts = []
    for claim_axis, endpoint, regimes, reference in claim_specs:
        lookup = primary[primary["claim_axis"].eq(claim_axis)].set_index(
            ["cohort_arm", "regime", "model"]
        )
        for arm in ACTIVE_REGIMES:
            for regime in regimes:
                for model in ["c1", "c2", "c3", "d1", "d2", "d3"]:
                    for metric in ["auroc", "allosteric_ap", "symmetric_ap"]:
                        test = float(lookup.loc[(arm, regime, model), metric])
                        base = float(lookup.loc[(arm, regime, reference), metric])
                        lifts.append(
                            {
                                "claim_axis": claim_axis,
                                "endpoint": endpoint,
                                "cohort_arm": arm,
                                "regime": regime,
                                "metric": metric,
                                "test_model": model,
                                "reference_model": reference,
                                "test_value": test,
                                "reference_value": base,
                                "delta": test - base,
                            }
                        )
    pd.DataFrame(lifts).to_csv(output / "PAIR_MODEL_LIFT.tsv", sep="\t", index=False)
    report = {
        "status": "validated",
        "primary_common_rows_per_cell": 395,
        "primary_common_prediction_rows_per_cell": 395,
        "conditional_metrics_may_use_fewer_rows": True,
        "claim_aligned_performance_cells": int(len(primary)),
        "pooled_secondary_cells": int(len(pooled_secondary)),
        "retention_rows": int(len(retention)),
        "conditional_split_changes_are_descriptive": True,
        "conditional_split_effective_rows_identical": False,
        "pooled_identical_row_split_change_rows": int(len(pooled_split_change)),
        "pair_lift_rows": int(len(lifts)),
        "matched_hard_row_pooled_split_sensitivity_rows": int(
            len(hard_pooled_split_sensitivity)
        ),
        "common_general_double_unseen_rows": 2721,
        "common_general_double_unseen_performance_rows": int(len(double_unseen)),
        "hard_primary_rows": int(len(hard_primary)),
        "hard_primary_pair_lift_rows": int(len(hard_primary_lifts)),
        "hard_primary_endpoint": (
            "fold-restricted within-protein AUROC on 1,805/2,721 rows and 112 groups"
        ),
        "hard_primary_comparator": "ligand-only",
        "hard_pooled_metric_role": (
            "secondary identical-2,721-row split-change sensitivity"
        ),
        "hard_within_ligand_metric_role": (
            "limited-support sensitivity only (37 rows / 14 groups)"
        ),
        "raw_auroc_increase_on_a_subset_called_recovery": False,
        "recovery_definition": "retention of above-chance AUROC on identical common rows",
        "primary_family_endpoint": "fold-restricted within-ligand AUROC",
        "primary_ligand_endpoint": "fold-restricted within-protein AUROC",
        "primary_family_comparator": "protein-only (ligand-only is a structural 0.5 check)",
        "primary_ligand_comparator": "ligand-only (protein-only is a structural 0.5 check)",
        "pooled_metrics_are_secondary": True,
        "pooled_metric_primary_exception": None,
    }
    atomic_json(output / "PATTERN_ANALYSIS_VALIDATION.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
