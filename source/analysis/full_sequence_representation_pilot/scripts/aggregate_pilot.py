#!/usr/bin/env python3
"""Compare full-UniProt and selected-chain OOF predictions on identical rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("protein", "c1", "c2", "c3")
NEW_REPRESENTATIONS = ("rechunked_selected_structure_chain", "full_canonical_uniprot")
LEGACY_REPRESENTATION = "legacy_selected_structure_chain"
SEEDS = (20260817, 20260818, 20260819)
FOLDS = tuple(range(5))


def roc_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
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
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if int(y.sum()) in {0, len(y)}:
        return np.nan
    order = np.argsort(score, kind="mergesort")[::-1]
    y = y[order]
    score = score[order]
    ends = np.r_[np.where(np.diff(score))[0], len(score) - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    recall = tp / int(y.sum())
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def conditional_auc(frame, group_column):
    values, rows_used = [], 0
    columns = ["outer_fold", group_column]
    if "__cluster_draw_id" in frame.columns:
        columns.append("__cluster_draw_id")
    for _, group in frame.groupby(columns, sort=False):
        value = roc_auc(group["binary_label"], group["p_allosteric"])
        if np.isfinite(value):
            values.append(value)
            rows_used += len(group)
    return (float(np.mean(values)) if values else np.nan, len(values), int(rows_used))


def metrics(frame):
    ligand, n_ligand, ligand_rows = conditional_auc(frame, "connectivity_key")
    protein, n_protein, protein_rows = conditional_auc(frame, "uniprot")
    return {
        "n_rows": int(len(frame)),
        "n_proteins": int(frame["uniprot"].nunique()),
        "pooled_auroc": roc_auc(frame["binary_label"], frame["p_allosteric"]),
        "allosteric_ap": average_precision(frame["binary_label"], frame["p_allosteric"]),
        "within_ligand_auroc": ligand,
        "within_ligand_groups": int(n_ligand),
        "within_ligand_rows": int(ligand_rows),
        "within_protein_auroc": protein,
        "within_protein_groups": int(n_protein),
        "within_protein_rows": int(protein_rows),
    }


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def read_new_predictions(package: Path):
    pieces = []
    for representation in NEW_REPRESENTATIONS:
        for model in MODELS:
            for seed in SEEDS:
                for fold in FOLDS:
                    path = package / "gpu_output/benchmark/fits" / representation / model / "seed_{}".format(seed) / "fold_{}".format(fold) / "predictions.tsv.gz"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    piece = pd.read_csv(path, sep="\t")
                    if len(piece) == 0 or set(piece["representation"].astype(str)) != {representation}:
                        raise RuntimeError("invalid prediction file {}".format(path))
                    pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True)
    return result


def read_selected_baseline(root: Path, manifest: pd.DataFrame):
    path = root / "analysis/role_complete_pair_matrix/gpu_output/benchmark/aggregate/ALL_OOF_PREDICTIONS.tsv.gz"
    frame = pd.read_csv(path, sep="\t")
    frame = frame[
        frame["cohort_arm"].eq("protein_anchored")
        & frame["regime"].eq("unseen_family")
        & frame["model"].isin(MODELS)
        & frame["seed"].isin(SEEDS)
    ].copy()
    frame["representation"] = LEGACY_REPRESENTATION
    frame = frame.merge(
        manifest[["uniprot", "aligned_canonical_fraction", "coverage_stratum", "canonical_length_gt_4096"]],
        on="uniprot",
        how="left",
        validate="many_to_one",
    )
    return frame


def bootstrap_job(payload):
    model, metric_name, stratum, test_representation, reference_representation, frame, replicates, seed = payload
    if stratum != "all":
        frame = frame[frame["coverage_stratum"].eq(stratum)].copy()
    wide = frame.pivot_table(
        index=["main_row_id", "uniprot", "family_component_id", "connectivity_key", "binary_label", "outer_fold"],
        columns="representation",
        values="p_allosteric",
        aggfunc="first",
    ).reset_index()
    required = [test_representation, reference_representation]
    if any(column not in wide for column in required) or wide[required].isna().any().any():
        raise RuntimeError("unpaired representation predictions")
    families = sorted(wide["family_component_id"].astype(str).unique())
    pieces = {family: wide[wide["family_component_id"].astype(str).eq(family)] for family in families}

    def statistic(sample, probability):
        work = sample.rename(columns={probability: "p_allosteric"})
        if metric_name == "pooled_auroc":
            return roc_auc(work["binary_label"], work["p_allosteric"])
        if metric_name == "within_ligand_auroc":
            return conditional_auc(work, "connectivity_key")[0]
        if metric_name == "within_protein_auroc":
            return conditional_auc(work, "uniprot")[0]
        raise ValueError(metric_name)

    group_column = {
        "within_ligand_auroc": "connectivity_key",
        "within_protein_auroc": "uniprot",
    }.get(metric_name)
    nested = False
    if group_column is not None:
        counts = wide.groupby(["outer_fold", group_column], sort=False)["family_component_id"].nunique()
        nested = bool(len(counts) and int(counts.max()) == 1)

    observed = statistic(wide, test_representation) - statistic(wide, reference_representation)
    rng = np.random.default_rng(seed)
    values = []
    invalid = 0
    for _ in range(replicates):
        draw = rng.choice(families, size=len(families), replace=True)
        sampled_parts = []
        for draw_id, value in enumerate(draw):
            part = pieces[value]
            if nested:
                part = part.copy()
                part["__cluster_draw_id"] = int(draw_id)
            sampled_parts.append(part)
        sample = pd.concat(sampled_parts, ignore_index=True)
        delta = statistic(sample, test_representation) - statistic(sample, reference_representation)
        if np.isfinite(delta):
            values.append(float(delta))
        else:
            invalid += 1
    array = np.asarray(values, dtype=float)
    if len(array) < int(0.95 * replicates):
        raise RuntimeError("insufficient valid bootstrap replicates")
    return {
        "model": model,
        "test_representation": test_representation,
        "reference_representation": reference_representation,
        "metric": metric_name,
        "coverage_stratum": stratum,
        "observed_delta": float(observed),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "p_delta_gt_0": float(np.mean(array > 0)),
        "replicates_requested": int(replicates),
        "replicates_valid": int(len(array)),
        "replicates_invalid": int(invalid),
        "cluster": "family_component_id",
        "conditional_groups_nested_in_cluster": bool(nested),
        "cluster_draw_multiplicity_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.project_root.resolve()
    package = root / "analysis/full_sequence_representation_pilot"
    output = package / "gpu_output/aggregate"
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(package / "data/FULL_SEQUENCE_MANIFEST.tsv", sep="\t")
    full = read_new_predictions(package)
    selected = read_selected_baseline(root, manifest)
    expected_per_representation = 4637 * len(MODELS) * len(SEEDS)
    if len(selected) != expected_per_representation:
        raise RuntimeError("unexpected legacy selected-chain prediction row count")
    for representation in NEW_REPRESENTATIONS:
        if len(full[full["representation"].eq(representation)]) != expected_per_representation:
            raise RuntimeError("unexpected prediction row count for {}".format(representation))
    key = [
        "model", "seed", "main_row_id", "uniprot", "family_component_id",
        "connectivity_key", "binary_label", "outer_fold",
    ]
    if full[["representation"] + key[:3]].duplicated().any() or selected[key[:3]].duplicated().any():
        raise RuntimeError("duplicate OOF prediction keys")
    legacy_keys = set(map(tuple, selected[key].astype(str).to_numpy()))
    for representation in NEW_REPRESENTATIONS:
        new_keys = set(map(tuple, full.loc[full["representation"].eq(representation), key].astype(str).to_numpy()))
        if new_keys != legacy_keys:
            raise RuntimeError("{} and legacy predictions are not paired on identical OOF rows".format(representation))
    predictions = pd.concat([selected, full], ignore_index=True, sort=False)
    predictions.to_csv(output / "PAIRED_OOF_PREDICTIONS.tsv.gz", sep="\t", index=False, compression="gzip")

    metric_rows = []
    strata_rows = []
    for (representation, model, seed), group in predictions.groupby(["representation", "model", "seed"], sort=True):
        row = {"representation": representation, "model": model, "seed": int(seed)}
        row.update(metrics(group))
        metric_rows.append(row)
        for stratum, subset in group.groupby("coverage_stratum", sort=True):
            item = {"representation": representation, "model": model, "seed": int(seed), "coverage_stratum": stratum}
            item.update(metrics(subset))
            strata_rows.append(item)
    metric_table = pd.DataFrame(metric_rows)
    strata_table = pd.DataFrame(strata_rows)
    metric_table.to_csv(output / "REPRESENTATION_METRICS.tsv", sep="\t", index=False)
    strata_table.to_csv(output / "COVERAGE_STRATIFIED_METRICS.tsv", sep="\t", index=False)

    extreme_rows = []
    for (representation, model, seed), group in predictions.groupby(["representation", "model", "seed"], sort=True):
        extreme_flag = group["canonical_length_gt_4096"].astype(str).str.lower().isin(["true", "1"])
        subset = group[~extreme_flag].copy()
        item = {"representation": representation, "model": model, "seed": int(seed)}
        item.update(metrics(subset))
        item["excluded_extreme_canonical_length_rows"] = int(len(group) - len(subset))
        extreme_rows.append(item)
    pd.DataFrame(extreme_rows).to_csv(output / "EXTREME_LENGTH_EXCLUSION_METRICS.tsv", sep="\t", index=False)

    delta_rows = []
    value_columns = ["pooled_auroc", "allosteric_ap", "within_ligand_auroc", "within_protein_auroc"]
    for (model, seed), group in metric_table.groupby(["model", "seed"]):
        indexed = group.set_index("representation")
        item = {"model": model, "seed": int(seed)}
        for column in value_columns:
            item[column + "_rechunked_minus_legacy"] = float(
                indexed.loc["rechunked_selected_structure_chain", column] - indexed.loc[LEGACY_REPRESENTATION, column]
            )
            item[column + "_full_minus_rechunked"] = float(
                indexed.loc["full_canonical_uniprot", column] - indexed.loc["rechunked_selected_structure_chain", column]
            )
            item[column + "_full_minus_legacy"] = float(
                indexed.loc["full_canonical_uniprot", column] - indexed.loc[LEGACY_REPRESENTATION, column]
            )
        delta_rows.append(item)
    pd.DataFrame(delta_rows).to_csv(output / "SEED_PAIRED_DELTAS.tsv", sep="\t", index=False)

    ensemble = (
        predictions.groupby(
            ["representation", "model", "main_row_id", "uniprot", "family_component_id", "connectivity_key", "binary_label", "outer_fold", "coverage_stratum"],
            as_index=False,
        )["p_allosteric"].mean()
    )
    jobs = []
    strata = ["all", "lt_0.25", "0.25_to_lt_0.50", "0.50_to_lt_0.90", "ge_0.90"]
    for model in MODELS:
        subset = ensemble[ensemble["model"].eq(model)].copy()
        # The overall within-ligand endpoint preserves comparability with the
        # frozen family-generalization analysis.  Its coverage-stratified
        # versions have only 2--49 usable ligand-by-fold groups, so they are
        # reported as transparent point estimates rather than over-precise CIs.
        bootstrap_cells = [
            ("within_ligand_auroc", "all"),
            ("within_protein_auroc", "all"),
            ("pooled_auroc", "all"),
        ]
        bootstrap_cells.extend((metric, stratum) for metric in ["within_protein_auroc", "pooled_auroc"] for stratum in strata[1:])
        # The first contrast is the primary biological representation test.
        # The second is a direct check that overlap chunking alone did not
        # create the observed full-sequence result.
        contrasts = [
            ("full_canonical_uniprot", "rechunked_selected_structure_chain", bootstrap_cells),
            ("rechunked_selected_structure_chain", LEGACY_REPRESENTATION, [
                ("within_ligand_auroc", "all"),
                ("within_protein_auroc", "all"),
                ("pooled_auroc", "all"),
            ]),
        ]
        for test_representation, reference_representation, cells in contrasts:
            for metric_name, stratum in cells:
                token = "{}|{}|{}|{}|{}".format(
                    model, metric_name, stratum, test_representation, reference_representation
                )
                seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
                jobs.append((
                    model, metric_name, stratum, test_representation,
                    reference_representation, subset, args.bootstrap_replicates, seed,
                ))
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as executor:
        bootstrap = list(executor.map(bootstrap_job, jobs))
    pd.DataFrame(bootstrap).to_csv(output / "BOOTSTRAP_REPRESENTATION_DELTAS.tsv", sep="\t", index=False)
    validation = {
        "status": "validated",
        "representations": [LEGACY_REPRESENTATION, *NEW_REPRESENTATIONS],
        "models": list(MODELS),
        "regime": "protein_anchored unseen_family",
        "fits_rechunked_selected_chain": 60,
        "fits_full_sequence": 60,
        "selected_chain_legacy_fits_reused": 60,
        "identical_oof_rows_per_model_seed": 4637,
        "primary_metric": "fold-restricted within-ligand AUROC",
        "secondary_metrics": ["fold-restricted within-protein AUROC", "pooled AUROC"],
        "coverage_strata": strata[1:],
        "coverage_stratified_within_ligand_role": "point-estimate sensitivity only because conditional support is sparse",
        "extreme_canonical_length_sensitivity": "metrics excluding the two proteins with canonical length >4096 are reported separately",
        "paired_family_cluster_bootstrap_replicates": int(args.bootstrap_replicates),
        "primary_representation_contrast": "full_canonical_uniprot minus rechunked_selected_structure_chain",
        "chunking_control_contrast": "rechunked_selected_structure_chain minus legacy_selected_structure_chain",
        "claim_limit": "This tests sequence-representation sensitivity, not biological-assembly completeness, processed-chain correctness, or pair-specific site correctness.",
    }
    atomic_json(output / "AGGREGATE_VALIDATION.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
