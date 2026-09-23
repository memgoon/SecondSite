#!/usr/bin/env python3
"""Aggregate weak-label discrimination, enrichment, and candidate outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


MODELS = ["ligand", "protein", "c2"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--scope", choices=["reference", "full"], required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--candidate-count", type=int, default=5000)
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) != 2:
        return {"n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()) if len(y) else None, "auprc": None, "auroc": None}
    return {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
    }


def reference_analysis(package, output, replicates):
    path = package / "gpu_output/reference/reference_predictions.tsv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    primary = frame[frame["weak2020_label"].isin([0, 1])].copy()
    if not primary["weak2020_label"].eq(1).any() or not primary["weak2020_label"].eq(0).any():
        raise RuntimeError("2020 reference lost a class")
    arm_a_proteins = set(
        pd.read_csv(package / "data/CURRENT_BENCHMARK_PROTEINS.tsv", sep="\t")
        .loc[lambda x: x["seen_in_arm_a"].eq(1), "uniprot"].astype(str)
    )
    primary["target_seen_arm_a"] = primary["uniprot"].astype(str).isin(arm_a_proteins).astype(int)
    pooled_rows = []
    stratum_rows = []
    target_rows = []
    per_target = {}
    for model in MODELS:
        score_col = "p_{}_mean".format(model)
        row = metrics(primary["weak2020_label"], primary[score_col])
        row.update({"model": model, "endpoint": "pooled_2020_allosteric_positive"})
        pooled_rows.append(row)
        for seen, group in primary.groupby("target_seen_arm_a"):
            value = metrics(group["weak2020_label"], group[score_col])
            value.update({"model": model, "target_seen_arm_a": int(seen)})
            stratum_rows.append(value)
        target_values = []
        for target, group in primary.groupby("uniprot"):
            if group["weak2020_label"].nunique() != 2:
                continue
            value = metrics(group["weak2020_label"], group[score_col])
            value.update({"model": model, "uniprot": target})
            target_rows.append(value)
            target_values.append((target, value["auprc"], value["auroc"], value["n"]))
        per_target[model] = {target: ap for target, ap, _, _ in target_values}
        pooled_rows[-1]["target_macro_auprc"] = float(np.mean([x[1] for x in target_values])) if target_values else None
        pooled_rows[-1]["target_macro_auroc"] = float(np.mean([x[2] for x in target_values])) if target_values else None
        pooled_rows[-1]["targets_with_both_labels"] = int(len(target_values))
    pd.DataFrame(pooled_rows).to_csv(output / "REFERENCE_MODEL_METRICS.tsv", sep="\t", index=False)
    pd.DataFrame(stratum_rows).to_csv(output / "REFERENCE_TARGET_SEEN_STRATA.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(output / "REFERENCE_PER_TARGET_METRICS.tsv", sep="\t", index=False)

    bootstrap_rows = []
    rng = np.random.default_rng(20260818)
    for test, reference in [("c2", "ligand"), ("c2", "protein")]:
        targets = sorted(set(per_target[test]) & set(per_target[reference]))
        difference = np.asarray([per_target[test][x] - per_target[reference][x] for x in targets], dtype=float)
        observed = float(difference.mean())
        values = np.empty(replicates, dtype=float)
        for i in range(replicates):
            values[i] = difference[rng.integers(0, len(difference), size=len(difference))].mean()
        bootstrap_rows.append({
            "metric": "target_macro_allosteric_positive_auprc",
            "test_model": test, "reference_model": reference,
            "n_targets": int(len(targets)), "delta": observed,
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_probability_delta_gt_0": float(np.mean(values > 0)),
            "replicates": int(replicates),
        })
    pd.DataFrame(bootstrap_rows).to_csv(output / "REFERENCE_PAIRED_TARGET_BOOTSTRAP.tsv", sep="\t", index=False)
    return frame, primary, pooled_rows, bootstrap_rows


def full_analysis(package, output, reference, candidate_count):
    paths = [package / "gpu_output/full/full_predictions_shard{:02d}of04.tsv.gz".format(i) for i in range(4)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    full = pd.concat([pd.read_csv(path, sep="\t", low_memory=False) for path in paths], ignore_index=True)
    # The direct reference was reconstructed without the obsolete legacy
    # blacklist and takes precedence over a duplicate legacy OOD row.
    ref_columns = [
        "stable_pair_key", "target_chembl_id", "ligand_chembl_id", "uniprot",
        "connectivity_key", "target_name", "pchembl_numeric", "weak2020_label",
        "is_5a_positive",
    ] + ["p_{}_mean".format(x) for x in MODELS] + ["p_{}_sd".format(x) for x in MODELS]
    direct = reference[ref_columns].copy()
    direct["legacy_ood_label"] = -999
    direct["pchembl_like"] = direct["pchembl_numeric"]
    direct["direct_reference"] = 1
    full["direct_reference"] = 0
    full["weak2020_label"] = -1
    full["is_5a_positive"] = 0
    combined = pd.concat([direct, full], ignore_index=True, sort=False)
    combined = combined.sort_values(["stable_pair_key", "direct_reference"], ascending=[True, False])
    combined = combined.drop_duplicates("stable_pair_key", keep="first").reset_index(drop=True)
    # The binary model is conditional on an active pair; the 4,020 legacy
    # weak-binding decoys are not part of the screening universe.
    combined = combined[~combined["legacy_ood_label"].eq(0)].copy()
    positive_targets = set(combined.loc[combined["is_5a_positive"].eq(1), "uniprot"].astype(str))
    fractions = [0.001, 0.005, 0.01, 0.05]
    enrichment_rows = []
    for model in ["ligand", "protein", "c2"]:
        score = "p_{}_mean".format(model)
        ranked = combined.sort_values(score, ascending=False).reset_index(drop=True)
        prevalence = float(ranked["is_5a_positive"].mean())
        for fraction in fractions:
            n_top = max(1, int(np.ceil(len(ranked) * fraction)))
            hits = int(ranked.iloc[:n_top]["is_5a_positive"].sum())
            rate = hits / float(n_top)
            enrichment_rows.append({
                "model": model, "ranking": "pooled_full_universe", "top_fraction": fraction,
                "universe_rows": int(len(ranked)), "positive_rows": int(ranked["is_5a_positive"].sum()),
                "top_rows": n_top, "top_positive_rows": hits, "top_positive_rate": rate,
                "background_prevalence": prevalence,
                "enrichment": rate / prevalence if prevalence > 0 else None,
            })
        within = combined[combined["uniprot"].astype(str).isin(positive_targets)].copy()
        within["target_percentile"] = within.groupby("uniprot")[score].rank(method="first", ascending=False, pct=True)
        prevalence = float(within["is_5a_positive"].mean())
        for fraction in fractions:
            top = within[within["target_percentile"].le(fraction)]
            hits = int(top["is_5a_positive"].sum())
            rate = hits / float(len(top)) if len(top) else 0.0
            enrichment_rows.append({
                "model": model, "ranking": "within_each_5A_positive_target", "top_fraction": fraction,
                "universe_rows": int(len(within)), "positive_rows": int(within["is_5a_positive"].sum()),
                "top_rows": int(len(top)), "top_positive_rows": hits, "top_positive_rate": rate,
                "background_prevalence": prevalence,
                "enrichment": rate / prevalence if prevalence > 0 else None,
            })
    pd.DataFrame(enrichment_rows).to_csv(output / "FULL_5A_ENRICHMENT.tsv", sep="\t", index=False)

    labeled_keys = set(reference.loc[
        reference["weak2020_label"].isin([0, 1]) | reference["is_5a_positive"].eq(1), "stable_pair_key"
    ].astype(str))
    candidates = combined[~combined["stable_pair_key"].astype(str).isin(labeled_keys)].copy()
    candidates = candidates.sort_values("p_c2_mean", ascending=False).head(candidate_count)
    candidate_cols = [
        "stable_pair_key", "target_chembl_id", "ligand_chembl_id", "uniprot",
        "connectivity_key", "target_name", "pchembl_like",
        "p_c2_mean", "p_c2_sd", "p_ligand_mean", "p_protein_mean",
    ]
    candidates[candidate_cols].to_csv(output / "TOP_UNLABELED_CANDIDATES.tsv.gz", sep="\t", index=False)
    return int(len(combined)), int(combined["is_5a_positive"].sum()), enrichment_rows


def main():
    args = parse_args()
    package = args.project_root / "analysis/chembl_external_prioritization"
    output = package / "gpu_output/aggregate"
    output.mkdir(parents=True, exist_ok=True)
    reference, primary, metrics_rows, bootstrap_rows = reference_analysis(package, output, args.bootstrap)
    report = {
        "status": "validated", "scope": args.scope,
        "reference_rows": int(len(reference)),
        "primary_2020_rows": int(len(primary)),
        "primary_2020_allosteric": int(primary["weak2020_label"].eq(1).sum()),
        "primary_2020_orthosteric": int(primary["weak2020_label"].eq(0).sum()),
        "primary_metrics": metrics_rows,
        "paired_target_bootstrap": bootstrap_rows,
        "five_a_treated_as_positive_only": True,
    }
    if args.scope == "full":
        universe_rows, positive_rows, enrichment = full_analysis(package, output, reference, args.candidate_count)
        report.update({
            "full_screen_rows": universe_rows,
            "full_screen_5a_positive_rows": positive_rows,
            "full_5a_enrichment": enrichment,
        })
    if not report["primary_2020_allosteric"] or not report["primary_2020_orthosteric"]:
        report["status"] = "failed"
    atomic_json(output / ("AGGREGATE_{}.json".format(args.scope.upper())), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("aggregate validation failed")


if __name__ == "__main__":
    main()
