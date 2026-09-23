#!/usr/bin/env python3
"""Claim-aligned paired cluster bootstrap on frozen common evaluation rows."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_MODELS = ("c1", "c2", "c3", "d1", "d2", "d3")
ALL_MODELS = ("ligand", "protein") + PAIR_MODELS
ATP_ADP = {"ZKHQWZAMYRWXGA", "XTWYTFMLZFPYCI"}
MIN_VALID_REPLICATE_FRACTION = 0.95


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if not positive or not negative:
        return np.nan
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
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def ap(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not y.sum() or y.sum() == len(y):
        return np.nan
    order = np.argsort(score, kind="mergesort")[::-1]
    y = y[order]
    score = score[order]
    ends = np.r_[np.where(np.diff(score))[0], len(score) - 1]
    precision = np.cumsum(y)[ends] / (ends + 1)
    recall = np.cumsum(y)[ends] / int(y.sum())
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def statistic(frame, score_column, endpoint, regime):
    if endpoint == "pooled_auroc":
        return auc(frame["binary_label"], frame[score_column])
    if endpoint == "pooled_symmetric_ap":
        y = frame["binary_label"].astype(int).to_numpy()
        p = frame[score_column].astype(float).to_numpy()
        return float((ap(y, p) + ap(1 - y, 1 - p)) / 2.0)
    if endpoint == "within_ligand_auroc":
        group = "connectivity_key"
    elif endpoint == "within_protein_auroc":
        group = "uniprot"
    else:
        raise ValueError("unknown endpoint: {}".format(endpoint))
    group_columns = ["outer_fold", group]
    if "__cluster_draw_id" in frame.columns:
        group_columns.append("__cluster_draw_id")
    values = []
    for _, part in frame.groupby(group_columns, sort=False):
        value = auc(part["binary_label"], part[score_column])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def statistic_support(frame, endpoint):
    if endpoint.startswith("pooled_"):
        valid = frame["binary_label"].nunique() == 2
        return {
            "n_rows_input": int(len(frame)),
            "n_rows_used": int(len(frame)) if valid else 0,
            "n_rows_skipped_single_class": 0 if valid else int(len(frame)),
            "n_groups_used": 1 if valid else 0,
            "n_groups_skipped_single_class": 0 if valid else 1,
            "row_coverage": 1.0 if valid and len(frame) else 0.0,
        }
    group = (
        "connectivity_key"
        if endpoint == "within_ligand_auroc"
        else "uniprot"
    )
    used_rows = 0
    used_groups = 0
    skipped_rows = 0
    skipped_groups = 0
    group_columns = ["outer_fold", group]
    if "__cluster_draw_id" in frame.columns:
        group_columns.append("__cluster_draw_id")
    for _, part in frame.groupby(group_columns, sort=False):
        if part["binary_label"].nunique() == 2:
            used_rows += int(len(part))
            used_groups += 1
        else:
            skipped_rows += int(len(part))
            skipped_groups += 1
    return {
        "n_rows_input": int(len(frame)),
        "n_rows_used": int(used_rows),
        "n_rows_skipped_single_class": int(skipped_rows),
        "n_groups_used": int(used_groups),
        "n_groups_skipped_single_class": int(skipped_groups),
        "row_coverage": float(used_rows / len(frame)) if len(frame) else 0.0,
    }


def conditional_groups_nested_in_cluster(frame, endpoint, cluster):
    """Whether every metric group belongs to exactly one bootstrap cluster."""
    if endpoint.startswith("pooled_"):
        return False
    group = (
        "connectivity_key"
        if endpoint == "within_ligand_auroc"
        else "uniprot"
    )
    counts = frame.groupby(["outer_fold", group], sort=False)[cluster].nunique()
    return bool(len(counts) and int(counts.max()) == 1)


def resample_clusters(frame, clusters, indices, sampled, preserve_occurrence):
    """Build one cluster-bootstrap sample, preserving repeated nested groups."""
    if not preserve_occurrence:
        positions = np.concatenate([indices[clusters[index]] for index in sampled])
        return frame.iloc[positions]
    pieces = []
    for draw_id, index in enumerate(sampled):
        part = frame.iloc[indices[clusters[index]]].copy()
        part["__cluster_draw_id"] = int(draw_id)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def multiplicity_self_test():
    """A,A,B must weight the family-A nested protein statistic twice."""
    frame = pd.DataFrame(
        {
            "outer_fold": [0, 0, 0, 0],
            "family_component_id": ["A", "A", "B", "B"],
            "uniprot": ["PA", "PA", "PB", "PB"],
            "connectivity_key": ["L0", "L1", "L2", "L3"],
            "binary_label": [0, 1, 0, 1],
            "test": [0.0, 1.0, 0.5, 0.5],
            "reference": [1.0, 0.0, 0.5, 0.5],
        }
    )
    clusters = ["A", "B"]
    indices = {
        key: np.flatnonzero(
            frame["family_component_id"].astype(str).eq(key).to_numpy()
        )
        for key in clusters
    }
    if not conditional_groups_nested_in_cluster(
        frame, "within_protein_auroc", "family_component_id"
    ):
        raise RuntimeError("bootstrap multiplicity self-test nesting failed")
    sampled = resample_clusters(
        frame, clusters, indices, np.asarray([0, 0, 1]), True
    )
    delta = statistic(sampled, "test", "within_protein_auroc", "unseen_family") - statistic(
        sampled, "reference", "within_protein_auroc", "unseen_family"
    )
    if not np.isclose(delta, 2.0 / 3.0, atol=1e-12):
        raise RuntimeError(
            "bootstrap multiplicity self-test failed: {} != {}".format(
                delta, 2.0 / 3.0
            )
        )
    return True


def one_job(job):
    (
        frame,
        arm,
        regime,
        universe,
        test_model,
        reference_model,
        endpoint,
        cluster,
        comparison_role,
        replicates,
    ) = job
    clusters = sorted(frame[cluster].astype(str).unique())
    indices = {key: np.flatnonzero(frame[cluster].astype(str).eq(key).to_numpy()) for key in clusters}
    preserve_occurrence = conditional_groups_nested_in_cluster(
        frame, endpoint, cluster
    )
    token = "|".join(
        [arm, regime, universe, test_model, reference_model, endpoint, cluster]
    )
    seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    observed_test = statistic(frame, test_model, endpoint, regime)
    observed_ref = statistic(frame, reference_model, endpoint, regime)
    values = []
    bootstrap_rows_used = []
    bootstrap_groups_used = []
    bootstrap_groups_skipped = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        value = resample_clusters(
            frame, clusters, indices, sampled, preserve_occurrence
        )
        delta = statistic(value, test_model, endpoint, regime) - statistic(
            value, reference_model, endpoint, regime
        )
        if not np.isfinite(delta):
            continue
        support = statistic_support(value, endpoint)
        values.append(float(delta))
        bootstrap_rows_used.append(int(support["n_rows_used"]))
        bootstrap_groups_used.append(int(support["n_groups_used"]))
        bootstrap_groups_skipped.append(
            int(support["n_groups_skipped_single_class"])
        )
    values = np.asarray(values, dtype=float)
    valid_fraction = float(len(values) / replicates) if replicates else 0.0
    if valid_fraction < MIN_VALID_REPLICATE_FRACTION:
        raise RuntimeError(
            "valid bootstrap fraction below {:.3f}: {:.3f} for {}".format(
                MIN_VALID_REPLICATE_FRACTION, valid_fraction, token
            )
        )
    result = {
        "cohort_arm": arm,
        "regime": regime,
        "evaluation_universe": universe,
        "endpoint": endpoint,
        "test_model": test_model,
        "reference_model": reference_model,
        "comparison_role": comparison_role,
        "cluster_unit": cluster,
        "cluster_role": "prespecified held-out generalization unit",
        "nested_metric_groups_in_cluster": bool(preserve_occurrence),
        "cluster_draw_multiplicity_preserved": True,
        "n_clusters": int(len(clusters)),
        "n_rows": int(len(frame)),
        "test_value": float(observed_test),
        "reference_value": float(observed_ref),
        "delta": float(observed_test - observed_ref),
        "ci95_low": float(np.quantile(values, 0.025)) if len(values) else None,
        "ci95_high": float(np.quantile(values, 0.975)) if len(values) else None,
        "probability_delta_gt_zero": float(np.mean(values > 0)) if len(values) else None,
        "valid_replicates": int(len(values)),
        "valid_replicate_fraction": valid_fraction,
        "minimum_valid_replicate_fraction": MIN_VALID_REPLICATE_FRACTION,
        "invalid_replicates_no_defined_statistic": int(replicates - len(values)),
        "bootstrap_rows_used_min": int(min(bootstrap_rows_used))
        if bootstrap_rows_used
        else None,
        "bootstrap_rows_used_median": float(np.median(bootstrap_rows_used))
        if bootstrap_rows_used
        else None,
        "bootstrap_rows_used_max": int(max(bootstrap_rows_used))
        if bootstrap_rows_used
        else None,
        "bootstrap_groups_used_min": int(min(bootstrap_groups_used))
        if bootstrap_groups_used
        else None,
        "bootstrap_groups_used_median": float(np.median(bootstrap_groups_used))
        if bootstrap_groups_used
        else None,
        "bootstrap_groups_used_max": int(max(bootstrap_groups_used))
        if bootstrap_groups_used
        else None,
        "bootstrap_groups_skipped_min": int(min(bootstrap_groups_skipped))
        if bootstrap_groups_skipped
        else None,
        "bootstrap_groups_skipped_median": float(
            np.median(bootstrap_groups_skipped)
        )
        if bootstrap_groups_skipped
        else None,
        "bootstrap_groups_skipped_max": int(max(bootstrap_groups_skipped))
        if bootstrap_groups_skipped
        else None,
    }
    result.update(statistic_support(frame, endpoint))
    return result


def main():
    args = parse_args()
    multiplicity_self_test()
    root = args.project_root.resolve()
    package = root / "analysis/role_complete_pair_matrix"
    path = package / "gpu_output/benchmark/aggregate/COMMON_ROLE_COMPLETE_PREDICTIONS.tsv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, sep="\t", low_memory=False)

    def to_wide(value):
        return value.pivot_table(
        index=[
            "cohort_arm",
            "regime",
            "main_row_id",
            "uniprot",
            "family_component_id",
            "connectivity_key",
            "binary_label",
            "outer_fold",
        ],
        columns="model",
        values="p_allosteric",
        aggfunc="first",
    ).reset_index()

    wide = to_wide(raw)
    jobs = []
    for (arm, regime), frame in wide[
        wide["regime"].isin(["unseen_family", "unseen_ligand"])
    ].groupby(["cohort_arm", "regime"], sort=True):
        if regime == "unseen_family":
            reference = "protein"
            endpoint = "within_ligand_auroc"
            cluster = "family_component_id"
        else:
            reference = "ligand"
            endpoint = "within_protein_auroc"
            cluster = "connectivity_key"
        for universe, subset in [
            ("common_role_complete", frame),
            (
                "common_role_complete_without_ATP_ADP",
                frame[~frame["connectivity_key"].astype(str).isin(ATP_ADP)],
            ),
        ]:
            for model in PAIR_MODELS:
                jobs.append(
                    (
                        subset.copy(),
                        arm,
                        regime,
                        universe,
                        model,
                        reference,
                        endpoint,
                        cluster,
                        "primary_nonconstant_single_input",
                        args.replicates,
                    )
                )
    double_path = (
        package
        / "gpu_output/benchmark/aggregate/COMMON_GENERAL_DOUBLE_UNSEEN_PREDICTIONS.tsv.gz"
    )
    if not double_path.is_file():
        raise FileNotFoundError(double_path)
    double_wide = to_wide(pd.read_csv(double_path, sep="\t", low_memory=False))
    for arm, frame in double_wide.groupby("cohort_arm", sort=True):
        for model in PAIR_MODELS:
            jobs.append(
                (
                    frame.copy(),
                    arm,
                    "unseen_family",
                    "common_family_held_out_ligand_novel",
                    model,
                    "ligand",
                    "within_protein_auroc",
                    "family_component_id",
                    "primary_nonconstant_single_input_double_unseen",
                    args.replicates,
                )
            )
    hard_path = (
        package
        / "gpu_output/benchmark/aggregate/COMMON_GENERAL_HARD_ROW_COMPARISON_PREDICTIONS.tsv.gz"
    )
    if not hard_path.is_file():
        raise FileNotFoundError(hard_path)
    hard_raw = pd.read_csv(hard_path, sep="\t", low_memory=False)
    hard_raw["score_key"] = (
        hard_raw["regime"].astype(str) + "__" + hard_raw["model"].astype(str)
    )
    hard_wide = hard_raw.pivot_table(
        index=[
            "cohort_arm",
            "main_row_id",
            "uniprot",
            "family_component_id",
            "connectivity_key",
            "binary_label",
        ],
        columns="score_key",
        values="p_allosteric",
        aggfunc="first",
    ).reset_index()
    hard_wide["outer_fold"] = 0  # pooled endpoints do not use fold identity
    for arm, frame in hard_wide.groupby("cohort_arm", sort=True):
        for model in ALL_MODELS:
            jobs.append(
                (
                    frame.copy(),
                    arm,
                    "unseen_family",
                    "common_general_hard_rows_split_penalty",
                    "unseen_family__" + model,
                    "row_random__" + model,
                    "pooled_auroc",
                    "family_component_id",
                    "paired_split_penalty_on_identical_2721_rows",
                    args.replicates,
                )
            )
    if len(jobs) != 100:
        raise RuntimeError(
            "bootstrap comparison contract changed: {} != 100".format(len(jobs))
        )
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        rows = list(executor.map(one_job, jobs))
    output = package / "cpu_analysis"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "PAIRED_CLUSTER_BOOTSTRAP.tsv", sep="\t", index=False)
    report = {
        "status": "validated",
        "comparisons": int(len(rows)),
        "replicates_requested": int(args.replicates),
        "workers": int(args.workers),
        "executor": "multiprocessing ProcessPoolExecutor",
        "family_claim_cluster": "family_component_id (held-out generalization unit)",
        "ligand_claim_cluster": "connectivity_key (held-out generalization unit)",
        "paired_resampling": True,
        "cluster_multiplicity_self_test": "A,A,B nested-protein delta = 2/3",
        "cluster_multiplicity_self_test_passed": True,
        "minimum_valid_replicate_fraction": MIN_VALID_REPLICATE_FRACTION,
        "double_unseen_universe": "common 2,721 rows for every-pair and protein-anchored training arms",
        "within_ligand_primary_for_family_claims": True,
        "within_protein_primary_for_ligand_claims": True,
        "family_claim_primary_comparator": "protein-only",
        "ligand_claim_primary_comparator": "ligand-only",
        "identity_constant_controls": (
            "ligand-only within ligand and protein-only within protein are "
            "reported as exact 0.5 implementation checks, not primary lift baselines"
        ),
        "conditional_support_columns_reported": True,
        "bootstrap_replicate_support_ranges_reported": True,
        "undefined_conditional_group_handling": (
            "single-class fold-by-group blocks are excluded within each replicate; "
            "replicates with no defined test/reference statistic are counted invalid"
        ),
        "double_unseen_primary_endpoint": (
            "fold-restricted within-protein AUROC on 1,805/2,721 rows and 112 groups"
        ),
        "double_unseen_primary_comparator": "ligand-only",
        "double_unseen_primary_cluster": (
            "family_component_id, the held-out generalization unit"
        ),
        "double_unseen_within_ligand_role": (
            "limited-support sensitivity (37 rows / 14 groups)"
        ),
        "double_unseen_pooled_role": (
            "secondary identical-row split-change sensitivity"
        ),
        "hard_row_split_penalty_bootstrapped": True,
        "bootstrap_scope": (
            "primary conditional AUROCs including double-unseen within-protein lift, "
            "plus secondary matched-row pooled split penalties; AP and sparse "
            "double-unseen within-ligand metrics are not bootstrapped"
        ),
    }
    atomic_json(output / "BOOTSTRAP_VALIDATION.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
