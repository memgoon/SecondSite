#!/usr/bin/env python3
"""Validate and aggregate the complete controlled pair matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("every_pair", "protein_anchored", "protein_ligand_role_complete")
REGIMES = ("row_random", "unseen_family", "unseen_ligand")
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
SEEDS = (20260817, 20260818, 20260819)
FOLDS = (0, 1, 2, 3, 4)
FOLD_COLUMNS = {
    "row_random": "matrix_row_fold",
    "unseen_family": "matrix_family_fold",
    "unseen_ligand": "matrix_ligand_fold",
}
ARM_FILES = {
    "every_pair": "EVERY_PAIR.tsv.gz",
    "protein_anchored": "PROTEIN_ANCHORED.tsv.gz",
    "protein_ligand_role_complete": "PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz",
}
ATP_ADP = {"ZKHQWZAMYRWXGA", "XTWYTFMLZFPYCI"}
ATP = "ZKHQWZAMYRWXGA"
ADP = "XTWYTFMLZFPYCI"
MAX_SINGLE_INPUT_NUMERICAL_SPAN = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_training_contract(cpu_contract, arm):
    return {
        "data_contract_id": "role_complete_pair_matrix_v2",
        "cohort_sha256": cpu_contract["files"][
            "data/" + ARM_FILES[arm]
        ]["sha256"],
        "model_implementation_sha256": cpu_contract["implementation_files"][
            "analysis/role_complete_pair_matrix/scripts/model_definitions.py"
        ]["sha256"],
        "base_trainer_sha256": cpu_contract["implementation_files"][
            "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
        ]["sha256"],
        "matrix_trainer_sha256": cpu_contract["implementation_files"][
            "analysis/role_complete_pair_matrix/scripts/train_matrix.py"
        ]["sha256"],
        "epochs": 25,
        "patience": 5,
        "min_epochs": 1,
        "hidden_dim": 256,
        "heads": 4,
        "dropout": 0.30,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 6,
        "eval_batch_size": 8,
        "full_bidirectional_batch_size": 1,
        "full_bidirectional_eval_batch_size": 2,
        "max_atoms": 120,
        "max_protein_residues": 4096,
        "training_weight": "equal total weight per protein-label group and then per label; no ligand balancing",
        "checkpoint_selection": (
            "model-aware: ligand-only=validation within-protein macro AUROC; "
            "protein-only=validation within-ligand macro AUROC; joint "
            "unseen_family=within-ligand, joint unseen_ligand=within-protein, "
            "joint row_random=unweighted mean of both; every requested metric "
            "requires at least 8 valid groups; pooled symmetric AP fallback otherwise"
        ),
        "evaluation_contract": (
            "protein-anchored rows for every_pair and protein_anchored; "
            "role-complete rows for protein_ligand_role_complete"
        ),
    }


def expected_fold_rows(frame, regime, fold):
    column = FOLD_COLUMNS[regime]
    validation_fold = (int(fold) + 1) % 5
    test = frame[
        frame[column].eq(fold) & frame["matrix_evaluation_eligible"].eq(1)
    ].copy()
    development = frame[~frame[column].isin([fold, validation_fold])].copy()
    validation = frame[
        frame[column].eq(validation_fold)
        & frame["matrix_evaluation_eligible"].eq(1)
    ]
    development_ligands = set(development["connectivity_key"].astype(str)) | set(
        validation["connectivity_key"].astype(str)
    )
    test["unseen_compound"] = (
        ~test["connectivity_key"].astype(str).isin(development_ligands)
    ).astype(int)
    test["outer_fold"] = int(fold)
    return test


def validate_prediction_metadata(value, expected, directory):
    fields = [
        "main_row_id",
        "uniprot",
        "family_component_id",
        "full_inchikey",
        "connectivity_key",
        "class_label",
        "binary_label",
        "unseen_compound",
        "outer_fold",
    ]
    missing = sorted(set(fields + ["p_allosteric"]) - set(value.columns))
    if missing:
        raise RuntimeError("prediction columns missing at {}: {}".format(directory, missing))
    left = value[fields].copy()
    right = expected[fields].copy()
    for field in fields:
        if field in {"binary_label", "unseen_compound", "outer_fold"}:
            left[field] = pd.to_numeric(left[field], errors="raise").astype(int)
            right[field] = pd.to_numeric(right[field], errors="raise").astype(int)
        else:
            left[field] = left[field].astype(str)
            right[field] = right[field].astype(str)
    left = left.sort_values("main_row_id").reset_index(drop=True)
    right = right.sort_values("main_row_id").reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("frozen test metadata mismatch at {}".format(directory))
    probability = pd.to_numeric(value["p_allosteric"], errors="coerce").to_numpy()
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise RuntimeError("invalid probability values at {}".format(directory))


def average_precision(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not len(y) or y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(score, kind="mergesort")[::-1]
    y = y[order]
    score = score[order]
    ends = np.r_[np.where(np.diff(score))[0], len(score) - 1]
    precision = np.cumsum(y)[ends] / (ends + 1)
    recall = np.cumsum(y)[ends] / int(y.sum())
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
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
        (ranks[y == 1].sum() - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def binary(frame):
    if not len(frame):
        return None
    y = frame["binary_label"].astype(int).to_numpy()
    p = frame["p_allosteric"].astype(float).to_numpy()
    allo = average_precision(y, p)
    ortho = average_precision(1 - y, 1.0 - p)
    return {
        "n": int(len(frame)),
        "n_rows_input": int(len(frame)),
        "n_rows_used": int(len(frame)),
        "n_rows_skipped_single_class": 0,
        "row_coverage": 1.0,
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "allosteric_ap": allo,
        "orthosteric_ap": ortho,
        "symmetric_ap": None if allo is None or ortho is None else float((allo + ortho) / 2),
        "auroc": roc_auc(y, p),
    }


def macro(frame, columns):
    rows = []
    skipped = 0
    rows_used = 0
    rows_skipped = 0
    for _, group in frame.groupby(columns, sort=False):
        value = binary(group)
        if value is None or value["auroc"] is None:
            skipped += 1
            rows_skipped += int(len(group))
        else:
            rows.append(value)
            rows_used += int(len(group))
    return {
        "n_groups": int(len(rows)),
        "n_skipped": int(skipped),
        "n_groups_used": int(len(rows)),
        "n_groups_skipped_single_class": int(skipped),
        "n_rows_input": int(len(frame)),
        "n_rows_used": int(rows_used),
        "n_rows_skipped_single_class": int(rows_skipped),
        "row_coverage": float(rows_used / len(frame)) if len(frame) else None,
        "auroc": float(np.mean([x["auroc"] for x in rows])) if rows else None,
        "allosteric_ap": float(np.mean([x["allosteric_ap"] for x in rows])) if rows else None,
        "symmetric_ap": float(np.mean([x["symmetric_ap"] for x in rows])) if rows else None,
    }


def canonicalize_single_input_predictions(frame, maximum_span):
    """Remove batch-shape roundoff from deterministic single-input scores.

    The same exact model input can land in batches with different padding
    shapes.  Mixed-precision reductions can then differ in the last digits,
    and rank metrics can turn that numerical noise into a non-chance AUC.
    Scores are averaged only within one OOF model (outer fold) and the exact
    input identity.  Ligand identity is full InChIKey, not connectivity, so
    stereoisomer-level ligand variation is retained.
    """
    result = frame.copy()
    audits = []
    for model, identity in [("ligand", "full_inchikey"), ("protein", "uniprot")]:
        mask = result["model"].eq(model)
        block = result.loc[mask].copy()
        keys = ["cohort_arm", "regime", "model", "outer_fold", identity]
        grouped = block.groupby(keys, sort=False)["p_allosteric"]
        spans = grouped.agg(["min", "max", "size"]).reset_index()
        spans["probability_span"] = spans["max"] - spans["min"]
        for (arm, regime), part in spans.groupby(
            ["cohort_arm", "regime"], sort=False
        ):
            maximum = float(part["probability_span"].max()) if len(part) else 0.0
            audits.append(
                {
                    "cohort_arm": arm,
                    "regime": regime,
                    "model": model,
                    "exact_input_identity": identity,
                    "identity_groups": int(len(part)),
                    "repeated_identity_groups": int(part["size"].gt(1).sum()),
                    "groups_with_nonzero_span": int(
                        part["probability_span"].gt(0).sum()
                    ),
                    "median_probability_span": float(
                        part["probability_span"].median()
                    ),
                    "p99_probability_span": float(
                        part["probability_span"].quantile(0.99)
                    ),
                    "max_probability_span": maximum,
                    "maximum_allowed_span": float(maximum_span),
                    "status": "within_numerical_tolerance"
                    if maximum <= maximum_span
                    else "failed_excessive_identity_span",
                }
            )
        canonical = grouped.transform("mean")
        result.loc[mask, "p_allosteric"] = canonical.to_numpy()
    audit = pd.DataFrame(audits)
    failed = audit[audit["status"].ne("within_numerical_tolerance")]
    if not failed.empty:
        raise RuntimeError(
            "single-input predictions differ excessively for identical inputs:\n{}".format(
                failed.to_string(index=False)
            )
        )
    return result, audit


def metric_rows(frame, arm, regime, model, universe):
    rows = []
    definitions = [
        ("pooled", None),
        ("protein_macro_fold_restricted", ["outer_fold", "uniprot"]),
        ("family_macro_fold_restricted", ["outer_fold", "family_component_id"]),
        ("ligand_macro_fold_restricted", ["outer_fold", "connectivity_key"]),
    ]
    for endpoint, columns in definitions:
        value = binary(frame) if columns is None else macro(frame, columns)
        row = {
            "cohort_arm": arm,
            "regime": regime,
            "model": model,
            "evaluation_universe": universe,
            "endpoint": endpoint,
        }
        if value is not None:
            row.update(value)
        rows.append(row)
    return rows


def main():
    args = parse_args()
    root = args.project_root.resolve()
    package = root / "analysis/role_complete_pair_matrix"
    cpu_contract = json.loads(
        (package / "validation/CPU_CONTRACT.json").read_text(encoding="utf-8")
    )
    if (
        cpu_contract.get("status") != "validated"
        or cpu_contract.get("contract_id") != "role_complete_pair_matrix_v2"
    ):
        raise RuntimeError("CPU contract is invalid")
    input_root = package / "gpu_output/benchmark/fits"
    output = package / "gpu_output/benchmark/aggregate"
    output.mkdir(parents=True, exist_ok=True)
    strict = pd.read_csv(
        package / "data/PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz", sep="\t", low_memory=False
    )
    strict_ids = set(strict["main_row_id"].astype(str))
    common_double_unseen_ids = set(
        pd.read_csv(
            package / "data/COMMON_GENERAL_DOUBLE_UNSEEN_ROWS.tsv", sep="\t"
        )["main_row_id"].astype(str)
    )
    frozen_frames = {
        arm: pd.read_csv(
            package / "data" / filename, sep="\t", low_memory=False
        )
        for arm, filename in ARM_FILES.items()
    }
    predictions = []
    inventory = []
    for arm in ARMS:
        for regime in ACTIVE_REGIMES[arm]:
            for model in MODELS:
                for seed in SEEDS:
                    observed_ids = set()
                    for fold in FOLDS:
                        directory = input_root / arm / regime / model / "seed_{}".format(seed) / "fold_{}".format(fold)
                        report_path = directory / "FIT_REPORT.json"
                        prediction_path = directory / "predictions.tsv.gz"
                        if not report_path.is_file() or not prediction_path.is_file():
                            raise FileNotFoundError("missing fit: {}".format(directory))
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                        if (
                            report.get("status") != "validated"
                            or report.get("cohort_arm") != arm
                            or report.get("regime") != regime
                            or report.get("model") != model
                            or int(report.get("seed", -1)) != seed
                            or int(report.get("outer_fold", -1)) != fold
                            or report.get("model_version") != "role_complete_matrix_v2"
                            or report.get("probability_conversion")
                            != "sigmoid applied after FP32 logit cast"
                            or report.get("training_contract")
                            != expected_training_contract(cpu_contract, arm)
                            or report.get("checkpoint_selection_contract")
                            != expected_training_contract(cpu_contract, arm)[
                                "checkpoint_selection"
                            ]
                            or report.get("checkpoint_selection_model_aware") is not True
                            or report.get(
                                "checkpoint_selection_structurally_constant_expected"
                            )
                            is not False
                            or int(
                                report.get(
                                    "checkpoint_selection_minimum_valid_groups", -1
                                )
                            )
                            != 8
                            or not isinstance(
                                report.get(
                                    "checkpoint_selection_requested_group_counts_at_best_epoch"
                                ),
                                dict,
                            )
                            or report.get("checkpoint_sha256")
                            != sha256(directory / "best.pt")
                            or report.get("prediction_sha256")
                            != sha256(prediction_path)
                        ):
                            raise RuntimeError("invalid fit: {}".format(directory))
                        value = pd.read_csv(prediction_path, sep="\t", low_memory=False)
                        for column, expected_value in [
                            ("cohort_arm", arm),
                            ("regime", regime),
                            ("model", model),
                            ("seed", seed),
                            ("outer_fold", fold),
                        ]:
                            if column not in value or set(value[column].astype(str)) != {
                                str(expected_value)
                            }:
                                raise RuntimeError(
                                    "prediction identity mismatch for {} at {}".format(
                                        column, directory
                                    )
                                )
                        if value["main_row_id"].duplicated().any():
                            raise RuntimeError("duplicate test row in {}".format(directory))
                        if observed_ids & set(value["main_row_id"].astype(str)):
                            raise RuntimeError("OOF row repeated across folds")
                        expected_fold = expected_fold_rows(
                            frozen_frames[arm], regime, fold
                        )
                        validate_prediction_metadata(value, expected_fold, directory)
                        observed_ids |= set(value["main_row_id"].astype(str))
                        predictions.append(value)
                        inventory.append(
                            {
                                "cohort_arm": arm,
                                "regime": regime,
                                "model": model,
                                "seed": seed,
                                "outer_fold": fold,
                                "rows": int(len(value)),
                                "best_epoch": int(report["best_epoch"]),
                                "n_parameters": int(report["n_parameters"]),
                                "hit_maximum_epoch": int(report["hit_maximum_epoch"]),
                                "checkpoint_selection_fallback": int(
                                    report[
                                        "checkpoint_selection_fallback_at_best_epoch"
                                    ]
                                ),
                                "checkpoint_selection_fallback_reason": report[
                                    "checkpoint_selection_fallback_reason_at_best_epoch"
                                ],
                                "checkpoint_selection_structurally_constant_expected": int(
                                    report[
                                        "checkpoint_selection_structurally_constant_expected"
                                    ]
                                ),
                                "checkpoint_sha256": report["checkpoint_sha256"],
                                "prediction_sha256": report["prediction_sha256"],
                                "probability_conversion": report["probability_conversion"],
                            }
                        )
                    arm_frame = frozen_frames[arm]
                    expected = set(
                        arm_frame.loc[
                            arm_frame["matrix_evaluation_eligible"].eq(1), "main_row_id"
                        ].astype(str)
                    )
                    if observed_ids != expected:
                        raise RuntimeError("OOF coverage mismatch for {} {} {} {}".format(arm, regime, model, seed))
    all_prediction = pd.concat(predictions, ignore_index=True)
    all_prediction.to_csv(output / "ALL_OOF_PREDICTIONS.tsv.gz", sep="\t", index=False, compression="gzip")
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(output / "FIT_INVENTORY.tsv", sep="\t", index=False)
    best_epoch_audit = (
        inventory_frame.assign(best_epoch_is_one=inventory_frame["best_epoch"].eq(1).astype(int))
        .groupby(["cohort_arm", "regime", "model"], as_index=False)
        .agg(
            fits=("best_epoch", "size"),
            median_best_epoch=("best_epoch", "median"),
            min_best_epoch=("best_epoch", "min"),
            max_best_epoch=("best_epoch", "max"),
            best_epoch_one_fits=("best_epoch_is_one", "sum"),
            best_epoch_one_fraction=("best_epoch_is_one", "mean"),
            hit_maximum_epoch_fits=("hit_maximum_epoch", "sum"),
            checkpoint_selection_fallback_fits=(
                "checkpoint_selection_fallback",
                "sum",
            ),
            checkpoint_selection_fallback_fraction=(
                "checkpoint_selection_fallback",
                "mean",
            ),
            structurally_constant_selection_fits=(
                "checkpoint_selection_structurally_constant_expected",
                "sum",
            ),
        )
    )
    fallback_contract = cpu_contract.get("checkpoint_selection", {})
    expected_fallback_lookup = {
        (row["cohort_arm"], row["regime"], row["model"]): int(
            row["expected_fallback_fits"]
        )
        for row in fallback_contract.get("nonzero_cells", [])
    }
    best_epoch_audit["expected_checkpoint_selection_fallback_fits"] = [
        expected_fallback_lookup.get((arm, regime, model), 0)
        for arm, regime, model in best_epoch_audit[
            ["cohort_arm", "regime", "model"]
        ].itertuples(index=False, name=None)
    ]
    if not best_epoch_audit["checkpoint_selection_fallback_fits"].astype(int).equals(
        best_epoch_audit[
            "expected_checkpoint_selection_fallback_fits"
        ].astype(int)
    ):
        mismatch = best_epoch_audit[
            best_epoch_audit["checkpoint_selection_fallback_fits"].astype(int)
            != best_epoch_audit[
                "expected_checkpoint_selection_fallback_fits"
            ].astype(int)
        ]
        raise RuntimeError(
            "checkpoint fallback incidence changed:\n{}".format(
                mismatch.to_string(index=False)
            )
        )
    if (
        int(best_epoch_audit["checkpoint_selection_fallback_fits"].sum())
        != int(fallback_contract.get("expected_fallback_fits", -1))
        or int(fallback_contract.get("expected_fallback_fits", -1)) != 132
    ):
        raise RuntimeError("checkpoint fallback total differs from frozen 132 fits")
    best_epoch_audit["small_sample_training_stability_flag"] = np.where(
        best_epoch_audit["best_epoch_one_fraction"] >= (1.0 / 3.0),
        "review_best_epoch_distribution",
        "none",
    )
    if int(best_epoch_audit["structurally_constant_selection_fits"].sum()) != 0:
        raise RuntimeError("a checkpoint was selected by a structurally constant metric")
    best_epoch_audit.to_csv(
        output / "BEST_EPOCH_CELL_AUDIT.tsv", sep="\t", index=False
    )

    group = [
        "cohort_arm",
        "regime",
        "model",
        "main_row_id",
        "uniprot",
        "family_component_id",
        "full_inchikey",
        "connectivity_key",
        "class_label",
        "binary_label",
        "outer_fold",
        "unseen_compound",
    ]
    ensemble = (
        all_prediction.groupby(group, as_index=False, sort=False)
        .agg(p_allosteric=("p_allosteric", "mean"), p_allosteric_seed_sd=("p_allosteric", "std"))
    )
    if ensemble.groupby(["cohort_arm", "regime", "model", "main_row_id"]).size().max() != 1:
        raise RuntimeError("ensemble row is not unique")
    numerical_contract = cpu_contract.get(
        "single_input_numerical_canonicalization", {}
    )
    maximum_span = float(
        numerical_contract.get(
            "maximum_allowed_probability_span", MAX_SINGLE_INPUT_NUMERICAL_SPAN
        )
    )
    if (
        numerical_contract.get("enabled") is not True
        or maximum_span != MAX_SINGLE_INPUT_NUMERICAL_SPAN
    ):
        raise RuntimeError("single-input numerical canonicalization contract missing")
    ensemble, single_input_audit = canonicalize_single_input_predictions(
        ensemble, maximum_span
    )
    single_input_audit.to_csv(
        output / "SINGLE_INPUT_NUMERICAL_EQUIVALENCE_AUDIT.tsv",
        sep="\t",
        index=False,
    )
    ensemble.to_csv(output / "ENSEMBLE_OOF_PREDICTIONS.tsv.gz", sep="\t", index=False, compression="gzip")
    common = ensemble[ensemble["main_row_id"].astype(str).isin(strict_ids)].copy()
    active_arm_regimes = sum(len(ACTIVE_REGIMES[arm]) for arm in ARMS)
    expected_common = active_arm_regimes * len(MODELS) * len(strict)
    if len(common) != expected_common:
        raise RuntimeError("common strict evaluation lost rows: {} != {}".format(len(common), expected_common))
    common.to_csv(output / "COMMON_ROLE_COMPLETE_PREDICTIONS.tsv.gz", sep="\t", index=False, compression="gzip")

    double_unseen = ensemble[
        ensemble["cohort_arm"].isin(["every_pair", "protein_anchored"])
        & ensemble["regime"].eq("unseen_family")
        & ensemble["main_row_id"].astype(str).isin(common_double_unseen_ids)
        & ensemble["unseen_compound"].eq(1)
    ].copy()
    expected_double = 2 * len(MODELS) * len(common_double_unseen_ids)
    if len(double_unseen) != expected_double:
        raise RuntimeError(
            "common double-unseen evaluation lost rows: {} != {}".format(
                len(double_unseen), expected_double
            )
        )
    double_unseen.to_csv(
        output / "COMMON_GENERAL_DOUBLE_UNSEEN_PREDICTIONS.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    # Preserve the exact same 2,721 rows under both random and family-held-out
    # fits.  Only the latter predictions are genuinely double-unseen; the
    # random predictions are the matched easy-condition comparator.
    hard_row_comparison = ensemble[
        ensemble["cohort_arm"].isin(["every_pair", "protein_anchored"])
        & ensemble["regime"].isin(["row_random", "unseen_family"])
        & ensemble["main_row_id"].astype(str).isin(common_double_unseen_ids)
    ].copy()
    expected_hard_comparison = 2 * 2 * len(MODELS) * len(common_double_unseen_ids)
    if len(hard_row_comparison) != expected_hard_comparison:
        raise RuntimeError(
            "matched hard-row comparison lost rows: {} != {}".format(
                len(hard_row_comparison), expected_hard_comparison
            )
        )
    hard_row_comparison.to_csv(
        output / "COMMON_GENERAL_HARD_ROW_COMPARISON_PREDICTIONS.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    metrics = []
    for (arm, regime, model), frame in ensemble.groupby(["cohort_arm", "regime", "model"], sort=False):
        metrics.extend(metric_rows(frame, arm, regime, model, "arm_evaluation"))
        strict_frame = frame[frame["main_row_id"].astype(str).isin(strict_ids)]
        metrics.extend(metric_rows(strict_frame, arm, regime, model, "common_role_complete"))
        strict_variants = [
            (
                "common_role_complete_without_ATP_ADP",
                strict_frame[~strict_frame["connectivity_key"].astype(str).isin(ATP_ADP)],
            ),
            (
                "common_role_complete_without_ATP",
                strict_frame[~strict_frame["connectivity_key"].astype(str).eq(ATP)],
            ),
            (
                "common_role_complete_without_ADP",
                strict_frame[~strict_frame["connectivity_key"].astype(str).eq(ADP)],
            ),
            (
                "common_role_complete_ATP_only",
                strict_frame[strict_frame["connectivity_key"].astype(str).eq(ATP)],
            ),
            (
                "common_role_complete_ADP_only",
                strict_frame[strict_frame["connectivity_key"].astype(str).eq(ADP)],
            ),
        ]
        for universe, subset in strict_variants:
            metrics.extend(metric_rows(subset, arm, regime, model, universe))
        if arm in ["every_pair", "protein_anchored"] and regime in [
            "row_random",
            "unseen_family",
        ]:
            hard_rows = frame[
                frame["main_row_id"].astype(str).isin(common_double_unseen_ids)
            ]
            metrics.extend(
                metric_rows(
                    hard_rows,
                    arm,
                    regime,
                    model,
                    "common_general_hard_rows",
                )
            )
        if regime == "unseen_family" and arm in ["every_pair", "protein_anchored"]:
            arm_novel = frame[frame["unseen_compound"].eq(1)]
            metrics.extend(
                metric_rows(
                    arm_novel,
                    arm,
                    regime,
                    model,
                    "family_held_out_ligand_novel_arm_specific",
                )
            )
            common_novel = frame[
                frame["main_row_id"].astype(str).isin(common_double_unseen_ids)
                & frame["unseen_compound"].eq(1)
            ]
            metrics.extend(
                metric_rows(
                    common_novel,
                    arm,
                    regime,
                    model,
                    "common_family_held_out_ligand_novel",
                )
            )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "MATRIX_METRICS.tsv", sep="\t", index=False)
    support_columns = [
        "cohort_arm",
        "regime",
        "model",
        "evaluation_universe",
        "endpoint",
        "n_rows_input",
        "n_rows_used",
        "n_rows_skipped_single_class",
        "row_coverage",
        "n_groups_used",
        "n_groups_skipped_single_class",
    ]
    conditional_support = metrics[
        metrics["endpoint"].ne("pooled")
    ][support_columns].copy()
    conditional_support.to_csv(
        output / "CONDITIONAL_METRIC_SUPPORT.tsv", sep="\t", index=False
    )
    frozen_support_expectations = [
        ("common_role_complete", "row_random", "ligand_macro_fold_restricted", 395, 249, 56),
        ("common_role_complete", "unseen_family", "ligand_macro_fold_restricted", 395, 240, 41),
        ("common_role_complete", "row_random", "protein_macro_fold_restricted", 395, 129, 53),
        ("common_role_complete", "unseen_ligand", "protein_macro_fold_restricted", 395, 87, 37),
        ("common_general_hard_rows", "row_random", "ligand_macro_fold_restricted", 2721, 8, 4),
        ("common_general_hard_rows", "unseen_family", "ligand_macro_fold_restricted", 2721, 37, 14),
        ("common_general_hard_rows", "row_random", "protein_macro_fold_restricted", 2721, 654, 94),
        ("common_general_hard_rows", "unseen_family", "protein_macro_fold_restricted", 2721, 1805, 112),
        ("common_family_held_out_ligand_novel", "unseen_family", "protein_macro_fold_restricted", 2721, 1805, 112),
    ]
    conditional_support_audit = []
    for universe, regime, endpoint, rows_input, rows_used, groups_used in frozen_support_expectations:
        subset = conditional_support[
            conditional_support["evaluation_universe"].eq(universe)
            & conditional_support["regime"].eq(regime)
            & conditional_support["endpoint"].eq(endpoint)
        ]
        if subset.empty or (
            set(subset["n_rows_input"].dropna().astype(int)) != {rows_input}
            or set(subset["n_rows_used"].dropna().astype(int)) != {rows_used}
            or set(subset["n_groups_used"].dropna().astype(int)) != {groups_used}
        ):
            raise RuntimeError(
                "conditional support contract changed: {} {} {}".format(
                    universe, regime, endpoint
                )
            )
        conditional_support_audit.append(
            {
                "evaluation_universe": universe,
                "regime": regime,
                "endpoint": endpoint,
                "n_rows_input": rows_input,
                "n_rows_used": rows_used,
                "n_groups_used": groups_used,
            }
        )

    conditional_control_checks = []
    for arm in ARMS:
        ligand_control = common[
            common["cohort_arm"].eq(arm)
            & common["regime"].eq("unseen_family")
            & common["model"].eq("ligand")
        ]
        family_protein_control = common[
            common["cohort_arm"].eq(arm)
            & common["regime"].eq("unseen_family")
            & common["model"].eq("protein")
        ]
        ligand_auc = macro(ligand_control, ["outer_fold", "connectivity_key"])["auroc"]
        family_protein_auc = macro(
            family_protein_control, ["outer_fold", "uniprot"]
        )["auroc"]
        if ligand_auc is None or abs(ligand_auc - 0.5) > 1e-6:
            raise RuntimeError("within-ligand ligand-only control failed for {}".format(arm))
        if family_protein_auc is None or abs(family_protein_auc - 0.5) > 1e-6:
            raise RuntimeError(
                "within-protein protein-only family control failed for {}".format(arm)
            )
        row = {
            "cohort_arm": arm,
            "ligand_only_within_ligand_auroc_family_holdout": float(ligand_auc),
            "protein_only_within_protein_auroc_family_holdout": float(
                family_protein_auc
            ),
            "protein_only_within_protein_auroc_ligand_holdout": None,
        }
        ligand_holdout_protein = common[
            common["cohort_arm"].eq(arm)
            & common["regime"].eq("unseen_ligand")
            & common["model"].eq("protein")
        ]
        ligand_holdout_protein_auc = macro(
            ligand_holdout_protein, ["outer_fold", "uniprot"]
        )["auroc"]
        if (
            ligand_holdout_protein_auc is None
            or abs(ligand_holdout_protein_auc - 0.5) > 1e-6
        ):
            raise RuntimeError(
                "within-protein protein-only ligand control failed for {}".format(
                    arm
                )
            )
        row["protein_only_within_protein_auroc_ligand_holdout"] = float(
            ligand_holdout_protein_auc
        )
        conditional_control_checks.append(row)
    pd.DataFrame(conditional_control_checks).to_csv(
        output / "CONDITIONAL_CONTROL_CHECKS.tsv", sep="\t", index=False
    )

    # Claim-aligned contrasts use the nonconstant single-input comparator as
    # primary.  The identity-constant comparator is also emitted as a
    # structural 0.5 control, but is not treated as evidence for pair-specific
    # signal by itself.
    contrasts = []
    common_metrics = metrics[
        metrics["evaluation_universe"].isin(
            [
                "common_role_complete",
                "common_role_complete_without_ATP_ADP",
                "common_family_held_out_ligand_novel",
            ]
        )
        & metrics["endpoint"].isin(
            ["pooled", "protein_macro_fold_restricted", "ligand_macro_fold_restricted"]
        )
    ]
    for (arm, regime, universe, endpoint), frame in common_metrics.groupby(
        ["cohort_arm", "regime", "evaluation_universe", "endpoint"], sort=False
    ):
        lookup = frame.set_index("model")
        for model in ["c1", "c2", "c3", "d1", "d2", "d3"]:
            if model not in lookup.index:
                continue
            if regime == "unseen_family" and endpoint == "ligand_macro_fold_restricted":
                references = [("protein", "primary_nonconstant_single_input"), ("ligand", "identity_constant_control")]
            elif regime == "unseen_ligand" and endpoint == "protein_macro_fold_restricted":
                references = [("ligand", "primary_nonconstant_single_input"), ("protein", "identity_constant_control")]
            elif (
                universe == "common_family_held_out_ligand_novel"
                and regime == "unseen_family"
                and endpoint == "protein_macro_fold_restricted"
            ):
                references = [
                    ("ligand", "primary_nonconstant_single_input_double_unseen"),
                    ("protein", "identity_constant_control"),
                ]
            else:
                references = [("ligand", "descriptive"), ("protein", "descriptive")]
            for reference, comparison_role in references:
                if reference not in lookup.index:
                    continue
                for metric in ["auroc", "allosteric_ap", "symmetric_ap"]:
                    if metric not in lookup.columns:
                        continue
                    test_value = lookup.loc[model, metric]
                    ref_value = lookup.loc[reference, metric]
                    if pd.isna(test_value) or pd.isna(ref_value):
                        continue
                    contrasts.append(
                        {
                            "cohort_arm": arm,
                            "regime": regime,
                            "evaluation_universe": universe,
                            "endpoint": endpoint,
                            "metric": metric,
                            "test_model": model,
                            "reference_model": reference,
                            "comparison_role": comparison_role,
                            "test_value": float(test_value),
                            "reference_value": float(ref_value),
                            "delta": float(test_value - ref_value),
                        }
                    )
    pd.DataFrame(contrasts).to_csv(output / "CLAIM_ALIGNED_CONTRASTS.tsv", sep="\t", index=False)
    validation = {
        "status": "validated",
        "contract_id": "role_complete_pair_matrix_v2",
        "model_version": "role_complete_matrix_v2",
        "expected_fits": int(
            active_arm_regimes * len(MODELS) * len(SEEDS) * len(FOLDS)
        ),
        "observed_fits": int(len(inventory)),
        "all_oof_prediction_rows": int(len(all_prediction)),
        "ensemble_rows": int(len(ensemble)),
        "common_role_complete_rows_per_cell": int(len(strict)),
        "common_role_complete_total_rows": int(len(common)),
        "common_general_double_unseen_rows_per_cell": int(len(common_double_unseen_ids)),
        "common_general_double_unseen_total_rows": int(len(double_unseen)),
        "common_general_hard_row_comparison_total_rows": int(
            len(hard_row_comparison)
        ),
        "atp_connectivity": "ZKHQWZAMYRWXGA",
        "adp_connectivity": "XTWYTFMLZFPYCI",
        "primary_cross_arm_universe": "common_role_complete",
        "primary_hard_generalization_universe": "common_family_held_out_ligand_novel",
        "primary_hard_generalization_endpoint": "protein_macro_fold_restricted.auroc",
        "primary_hard_generalization_effective_rows": 1805,
        "primary_hard_generalization_effective_groups": 112,
        "primary_family_generalization_endpoint": "ligand_macro_fold_restricted.auroc",
        "primary_ligand_generalization_endpoint": "protein_macro_fold_restricted.auroc",
        "conditional_metrics_report_effective_rows_and_groups": True,
        "conditional_support_audit": conditional_support_audit,
        "primary_family_generalization_comparator": "protein-only",
        "primary_ligand_generalization_comparator": "ligand-only",
        "identity_constant_single_input_controls_also_reported": True,
        "pooled_metrics_are_secondary": True,
        "double_unseen_pooled_metric_role": (
            "secondary identical-2,721-row split-change sensitivity; the primary "
            "hard endpoint is fold-restricted within-protein AUROC"
        ),
        "role_complete_double_unseen_supported": False,
        "active_training_regimes_by_cohort": {
            arm: list(regimes) for arm, regimes in ACTIVE_REGIMES.items()
        },
        "arm_evaluation_cross_arm_comparison_allowed": False,
        "every_pair_oof_scope": "protein_anchored evaluation rows after broad training augmentation",
        "conditional_control_checks": conditional_control_checks,
        "single_input_numerical_canonicalization": numerical_contract,
        "single_input_numerical_audit_rows": int(len(single_input_audit)),
        "single_input_numerical_audit_failed_rows": int(
            single_input_audit["status"]
            .ne("within_numerical_tolerance")
            .sum()
        ),
        "best_epoch_one_review_threshold": float(1.0 / 3.0),
        "best_epoch_flagged_cells": int(
            best_epoch_audit["small_sample_training_stability_flag"]
            .ne("none")
            .sum()
        ),
        "best_epoch_flag_is_descriptive_not_a_selection_rule": True,
        "checkpoint_selection_model_aware": True,
        "checkpoint_selection_minimum_valid_groups": 8,
        "checkpoint_selection_fallback_is_prespecified": True,
        "checkpoint_selection_expected_fallback_fits": int(
            fallback_contract["expected_fallback_fits"]
        ),
        "checkpoint_selection_observed_fallback_fits": int(
            best_epoch_audit["checkpoint_selection_fallback_fits"].sum()
        ),
        "checkpoint_selection_expected_fallback_fraction": float(
            fallback_contract["expected_fallback_fraction_all_fits"]
        ),
        "checkpoint_selection_role_complete_fallback_fraction": float(
            fallback_contract["expected_role_complete_fallback_fraction"]
        ),
        "checkpoint_selection_fallback_limitation": fallback_contract[
            "interpretation"
        ],
        "structurally_constant_checkpoint_selection_fits": int(
            best_epoch_audit["structurally_constant_selection_fits"].sum()
        ),
    }
    atomic_json(output / "AGGREGATE_VALIDATION.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
