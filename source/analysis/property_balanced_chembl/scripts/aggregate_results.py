#!/usr/bin/env python3
"""Aggregate FP32 reference metrics, enrichment, and candidate priorities."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--scope", choices=["reference", "full"], required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--candidate-count", type=int, default=5000)
    return parser.parse_args()


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


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    result = {
        "n": int(len(y)),
        "positives": int(y.sum()) if len(y) else 0,
        "prevalence": float(y.mean()) if len(y) else None,
        "auprc": None,
        "auroc": None,
    }
    if len(y) and len(np.unique(y)) == 2:
        result["auprc"] = float(average_precision(y, score))
        result["auroc"] = float(roc_auc(y, score))
    return result


def reference_analysis(package, output, models, replicates):
    path = package / "gpu_output/reference/reference_predictions.tsv.gz"
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    primary = frame[frame["weak2020_label"].isin([0, 1])].copy()
    if primary["weak2020_label"].nunique() != 2:
        raise RuntimeError("2020 reference lost a class")
    allowed_family_status = {"seen", "unseen", "annotation_unavailable"}
    observed_family_status = set(primary["pfam_family_overlap_status"].dropna())
    if not observed_family_status <= allowed_family_status:
        raise RuntimeError(
            "invalid Pfam family-overlap status: {}".format(
                sorted(observed_family_status - allowed_family_status)
            )
        )
    if primary["pfam_family_overlap_status"].isna().any():
        raise RuntimeError("missing Pfam family-overlap status")

    if "c3_pocket_available" not in primary:
        raise RuntimeError("reference predictions lack the frozen C3 pocket scope")
    if primary.loc[
        primary["c3_pocket_available"].eq(0), "p_property_c3_mean"
    ].notna().any():
        raise RuntimeError("C3 scores exist outside the pocket-available subset")
    if primary.loc[
        primary["c3_pocket_available"].eq(1), "p_property_c3_mean"
    ].isna().any():
        raise RuntimeError("C3 scores are missing inside the pocket-available subset")

    pooled_rows = []
    stratum_rows = []
    target_rows = []
    per_target = {}
    strata = {
        "all": pd.Series(True, index=primary.index),
        "target_identity_unseen": primary["target_seen_property"].eq(0),
        "ligand_connectivity_unseen": primary["compound_seen_property"].eq(0),
        "double_novel_target_identity_and_connectivity": primary[
            "double_novel_target_identity_and_connectivity"
        ].eq(1),
        "pfam_family_seen": primary["pfam_family_overlap_status"].eq("seen"),
        "pfam_family_unseen": primary["pfam_family_overlap_status"].eq("unseen"),
        "pfam_annotation_unavailable": primary[
            "pfam_family_overlap_status"
        ].eq("annotation_unavailable"),
    }
    if not pd.concat(
        [strata["pfam_family_seen"], strata["pfam_family_unseen"],
         strata["pfam_annotation_unavailable"]], axis=1
    ).sum(axis=1).eq(1).all():
        raise RuntimeError("Pfam family strata do not partition the primary reference")
    evaluation_scopes = [
        (
            "all_rows",
            pd.Series(True, index=primary.index),
            [model for model in models if model != "property_c3"],
        ),
        (
            "pocket_available_matched",
            primary["c3_pocket_available"].eq(1),
            list(models),
        ),
    ]
    for evaluation_subset, subset_mask, subset_models in evaluation_scopes:
        subset = primary.loc[subset_mask].copy()
        if not len(subset) or subset["weak2020_label"].nunique() != 2:
            raise RuntimeError("evaluation subset lost a class: {}".format(evaluation_subset))
        for model in subset_models:
            score_column = "p_{}_mean".format(model)
            if subset[score_column].isna().any():
                raise RuntimeError(
                    "missing score for {} in {}".format(model, evaluation_subset)
                )
            row = metrics(subset["weak2020_label"], subset[score_column])
            row.update({
                "model": model,
                "evaluation_subset": evaluation_subset,
                "endpoint": "pooled_2020_allosteric_positive",
            })
            target_values = []
            for target, group in subset.groupby("uniprot"):
                if group["weak2020_label"].nunique() != 2:
                    continue
                value = metrics(group["weak2020_label"], group[score_column])
                value.update({
                    "model": model,
                    "evaluation_subset": evaluation_subset,
                    "uniprot": target,
                })
                target_rows.append(value)
                target_values.append((target, value["auprc"], value["auroc"]))
            per_target[(model, evaluation_subset)] = {
                target: ap for target, ap, _ in target_values
            }
            row["target_macro_auprc"] = (
                float(np.mean([x[1] for x in target_values])) if target_values else None
            )
            row["target_macro_auroc"] = (
                float(np.mean([x[2] for x in target_values])) if target_values else None
            )
            row["targets_with_both_labels"] = int(len(target_values))
            pooled_rows.append(row)
            for stratum, stratum_mask in strata.items():
                group = primary.loc[subset_mask & stratum_mask]
                value = metrics(group["weak2020_label"], group[score_column])
                value.update({
                    "model": model,
                    "evaluation_subset": evaluation_subset,
                    "stratum": stratum,
                })
                stratum_rows.append(value)

    pd.DataFrame(pooled_rows).to_csv(output / "REFERENCE_MODEL_METRICS.tsv", sep="\t", index=False)
    pd.DataFrame(stratum_rows).to_csv(output / "REFERENCE_NOVELTY_STRATA.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(output / "REFERENCE_PER_TARGET_METRICS.tsv", sep="\t", index=False)

    comparisons = [
        ("all_rows", "property_c2", "property_ligand"),
        ("all_rows", "property_c2", "source_c2"),
        ("all_rows", "source_c2", "source_ligand"),
        ("all_rows", "property_c2", "source_protein"),
        ("pocket_available_matched", "property_c3", "property_c2"),
        ("pocket_available_matched", "property_c3", "property_ligand"),
        ("pocket_available_matched", "property_c2", "property_ligand"),
        ("pocket_available_matched", "property_c3", "source_c2"),
    ]
    bootstrap_rows = []
    rng = np.random.default_rng(20260821)
    for evaluation_subset, test, reference in comparisons:
        test_values = per_target[(test, evaluation_subset)]
        reference_values = per_target[(reference, evaluation_subset)]
        targets = sorted(set(test_values) & set(reference_values))
        if not targets:
            raise RuntimeError("no paired targets for {} minus {}".format(test, reference))
        difference = np.asarray(
            [test_values[target] - reference_values[target] for target in targets],
            dtype=float,
        )
        values = np.empty(replicates, dtype=float)
        for index in range(replicates):
            values[index] = difference[
                rng.integers(0, len(difference), size=len(difference))
            ].mean()
        bootstrap_rows.append({
            "metric": "target_macro_allosteric_positive_auprc",
            "test_model": test,
            "reference_model": reference,
            "evaluation_subset": evaluation_subset,
            "n_targets": int(len(targets)),
            "delta": float(difference.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_probability_delta_gt_0": float(np.mean(values > 0)),
            "replicates": int(replicates),
        })
    pd.DataFrame(bootstrap_rows).to_csv(
        output / "REFERENCE_PAIRED_TARGET_BOOTSTRAP.tsv", sep="\t", index=False
    )
    return frame, primary, pooled_rows, bootstrap_rows, stratum_rows


def enrichment_rows_for(frame, models, stratum_name, evaluation_subset):
    fractions = [0.001, 0.005, 0.01, 0.05]
    rows = []
    positive_targets = set(frame.loc[frame["is_5a_positive"].eq(1), "uniprot"].astype(str))
    for model in models:
        score_column = "p_{}_mean".format(model)
        ranked = frame.sort_values(score_column, ascending=False).reset_index(drop=True)
        prevalence = float(ranked["is_5a_positive"].mean()) if len(ranked) else 0.0
        for fraction in fractions:
            n_top = max(1, int(np.ceil(len(ranked) * fraction)))
            top = ranked.iloc[:n_top]
            hits = int(top["is_5a_positive"].sum())
            rate = hits / float(n_top)
            rows.append({
                "model": model,
                "stratum": stratum_name,
                "evaluation_subset": evaluation_subset,
                "ranking": "pooled",
                "top_fraction": fraction,
                "universe_rows": int(len(ranked)),
                "positive_rows": int(ranked["is_5a_positive"].sum()),
                "top_rows": n_top,
                "top_positive_rows": hits,
                "top_positive_rate": rate,
                "background_prevalence": prevalence,
                "enrichment": rate / prevalence if prevalence > 0 else None,
            })

        within = frame[frame["uniprot"].astype(str).isin(positive_targets)].copy()
        within["target_percentile"] = within.groupby("uniprot")[score_column].rank(
            method="first", ascending=False, pct=True
        )
        prevalence = float(within["is_5a_positive"].mean()) if len(within) else 0.0
        for fraction in fractions:
            top = within[within["target_percentile"].le(fraction)]
            hits = int(top["is_5a_positive"].sum())
            rate = hits / float(len(top)) if len(top) else 0.0
            rows.append({
                "model": model,
                "stratum": stratum_name,
                "evaluation_subset": evaluation_subset,
                "ranking": "within_target",
                "top_fraction": fraction,
                "universe_rows": int(len(within)),
                "positive_rows": int(within["is_5a_positive"].sum()),
                "top_rows": int(len(top)),
                "top_positive_rows": hits,
                "top_positive_rate": rate,
                "background_prevalence": prevalence,
                "enrichment": rate / prevalence if prevalence > 0 else None,
            })
    return rows


def full_analysis(package, output, reference, models, candidate_count):
    paths = [
        package / "gpu_output/full/full_predictions_shard{:02d}of04.tsv.gz".format(index)
        for index in range(4)
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    full = pd.concat(
        [pd.read_csv(path, sep="\t", low_memory=False) for path in paths],
        ignore_index=True,
    )
    score_columns = []
    for model in models:
        score_columns.extend(["p_{}_mean".format(model), "p_{}_sd".format(model)])
    novelty_columns = [
        "target_seen_property", "compound_seen_property", "pair_seen_property",
        "double_novel_target_identity_and_connectivity",
        "pfam_ids", "pfam_annotation_status", "pfam_family_annotation_available",
        "family_seen_property", "family_unseen_property",
        "pfam_family_overlap_status",
    ]
    direct_columns = [
        "stable_pair_key", "target_chembl_id", "ligand_chembl_id", "uniprot",
        "connectivity_key", "target_name", "pchembl_numeric", "weak2020_label",
        "is_5a_positive", "c3_pocket_available",
        "c3_pocket_unavailable_reason",
    ] + novelty_columns + score_columns
    direct = reference[direct_columns].copy()
    direct["legacy_ood_label"] = -999
    direct["pchembl_like"] = direct["pchembl_numeric"]
    direct["direct_reference"] = 1
    full["direct_reference"] = 0
    full["weak2020_label"] = -1
    full["is_5a_positive"] = 0
    combined = pd.concat([direct, full], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["stable_pair_key", "direct_reference"], ascending=[True, False]
    ).drop_duplicates("stable_pair_key", keep="first")
    combined = combined[~combined["legacy_ood_label"].eq(0)].copy().reset_index(drop=True)
    if combined["pair_seen_property"].astype(int).any():
        raise RuntimeError("combined screen retained a property training pair")
    if combined.loc[
        combined["c3_pocket_available"].eq(0), "p_property_c3_mean"
    ].notna().any():
        raise RuntimeError("combined screen has C3 scores without a pocket")
    if combined.loc[
        combined["c3_pocket_available"].eq(1), "p_property_c3_mean"
    ].isna().any():
        raise RuntimeError("combined screen lacks C3 scores for an available pocket")

    strata = {
        "all": pd.Series(True, index=combined.index),
        "target_identity_unseen": combined["target_seen_property"].eq(0),
        "ligand_connectivity_unseen": combined["compound_seen_property"].eq(0),
        "double_novel_target_identity_and_connectivity": combined[
            "double_novel_target_identity_and_connectivity"
        ].eq(1),
        "pfam_family_seen": combined["pfam_family_overlap_status"].eq("seen"),
        "pfam_family_unseen": combined["pfam_family_overlap_status"].eq("unseen"),
        "pfam_annotation_unavailable": combined[
            "pfam_family_overlap_status"
        ].eq("annotation_unavailable"),
    }
    if not pd.concat(
        [strata["pfam_family_seen"], strata["pfam_family_unseen"],
         strata["pfam_annotation_unavailable"]], axis=1
    ).sum(axis=1).eq(1).all():
        raise RuntimeError("Pfam family strata do not partition the full screen")
    enrichment = []
    base_models = [model for model in models if model != "property_c3"]
    pocket = combined["c3_pocket_available"].eq(1)
    for name, mask in strata.items():
        enrichment.extend(enrichment_rows_for(
            combined.loc[mask].copy(), base_models, name, "all_rows"
        ))
        enrichment.extend(enrichment_rows_for(
            combined.loc[mask & pocket].copy(), models, name,
            "pocket_available_matched",
        ))
    pd.DataFrame(enrichment).to_csv(output / "FULL_5A_ENRICHMENT.tsv", sep="\t", index=False)

    labeled_keys = set(reference.loc[
        reference["weak2020_label"].isin([0, 1]) | reference["is_5a_positive"].eq(1),
        "stable_pair_key",
    ].astype(str))
    candidates = combined[~combined["stable_pair_key"].astype(str).isin(labeled_keys)].copy()
    candidate_columns = [
        "stable_pair_key", "target_chembl_id", "ligand_chembl_id", "uniprot",
        "connectivity_key", "target_name", "pchembl_like",
        "c3_pocket_available", "c3_pocket_unavailable_reason",
    ] + novelty_columns + [
        "p_property_c2_mean", "p_property_c2_sd", "p_property_ligand_mean",
        "p_property_c3_mean", "p_property_c3_sd", "p_source_c2_mean",
        "p_source_ligand_mean", "p_source_protein_mean",
    ]
    candidates.sort_values("p_property_c2_mean", ascending=False).head(candidate_count)[
        candidate_columns
    ].to_csv(output / "TOP_UNLABELED_CANDIDATES.tsv.gz", sep="\t", index=False)
    candidates[
        candidates["double_novel_target_identity_and_connectivity"].eq(1)
    ].sort_values("p_property_c2_mean", ascending=False).head(candidate_count)[
        candidate_columns
    ].to_csv(output / "TOP_DOUBLE_NOVEL_CANDIDATES.tsv.gz", sep="\t", index=False)
    c3_candidates = candidates[candidates["c3_pocket_available"].eq(1)].copy()
    c3_candidates.sort_values("p_property_c3_mean", ascending=False).head(
        candidate_count
    )[candidate_columns].to_csv(
        output / "TOP_C3_POCKET_AVAILABLE_CANDIDATES.tsv.gz",
        sep="\t", index=False,
    )
    c3_candidates[
        c3_candidates["double_novel_target_identity_and_connectivity"].eq(1)
    ].sort_values("p_property_c3_mean", ascending=False).head(candidate_count)[
        candidate_columns
    ].to_csv(
        output / "TOP_C3_POCKET_AVAILABLE_DOUBLE_NOVEL_CANDIDATES.tsv.gz",
        sep="\t", index=False,
    )
    return (
        int(len(combined)),
        int(combined["is_5a_positive"].sum()),
        enrichment,
        {
            "pocket_available_rows": int(pocket.sum()),
            "pocket_unavailable_rows": int((~pocket).sum()),
            "pocket_available_proteins": int(
                combined.loc[pocket, "uniprot"].nunique()
            ),
        },
    )


def main():
    args = parse_args()
    package = args.project_root / "analysis/property_balanced_chembl"
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    models = config["inference_models"]
    output = package / "gpu_output/aggregate"
    output.mkdir(parents=True, exist_ok=True)
    reference, primary, metric_rows, bootstrap_rows, stratum_rows = reference_analysis(
        package, output, models, args.bootstrap
    )
    report = {
        "status": "validated",
        "contract_id": config["contract_id"],
        "scope": args.scope,
        "models": models,
        "reference_rows": int(len(reference)),
        "primary_2020_rows": int(len(primary)),
        "primary_2020_allosteric": int(primary["weak2020_label"].eq(1).sum()),
        "primary_2020_orthosteric": int(primary["weak2020_label"].eq(0).sum()),
        "primary_metrics": metric_rows,
        "paired_target_bootstrap": bootstrap_rows,
        "novelty_strata": stratum_rows,
        "five_a_treated_as_positive_only": True,
        "pfam_family_overlap_stratification_reported": True,
        "pfam_family_overlap_definition": config["family_novelty"]["definition"],
        "pfam_annotation_unavailable_never_counted_as_unseen": True,
        "sequence_family_novelty_claimed": False,
        "c3_metric_scope": "pocket_available_matched",
        "c3_unavailable_scores_are_na": True,
        "primary_c3_pocket_accounting": {
            "available_rows": int(primary["c3_pocket_available"].eq(1).sum()),
            "unavailable_rows": int(primary["c3_pocket_available"].eq(0).sum()),
            "available_proteins": int(
                primary.loc[primary["c3_pocket_available"].eq(1), "uniprot"].nunique()
            ),
        },
        "primary_pfam_family_accounting": {
            "seen_rows": int(primary["pfam_family_overlap_status"].eq("seen").sum()),
            "unseen_rows": int(primary["pfam_family_overlap_status"].eq("unseen").sum()),
            "annotation_unavailable_rows": int(
                primary["pfam_family_overlap_status"].eq(
                    "annotation_unavailable"
                ).sum()
            ),
        },
    }
    if args.scope == "full":
        rows, positives, enrichment, c3_accounting = full_analysis(
            package, output, reference, models, args.candidate_count
        )
        report.update({
            "full_screen_rows": rows,
            "full_screen_5a_positive_rows": positives,
            "full_5a_enrichment": enrichment,
            "full_c3_pocket_accounting": c3_accounting,
        })
    if not report["primary_2020_allosteric"] or not report["primary_2020_orthosteric"]:
        report["status"] = "failed"
    atomic_json(output / "AGGREGATE_{}.json".format(args.scope.upper()), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("aggregate validation failed")


if __name__ == "__main__":
    main()
