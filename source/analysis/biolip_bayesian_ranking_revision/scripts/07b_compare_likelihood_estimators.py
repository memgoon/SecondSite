#!/usr/bin/env python3
"""Checkpoint-7 addendum: compare empirical-bin and capped-KDE likelihood scores."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
REPORTS = PACKAGE / "reports"
VALIDATION = PACKAGE / "validation"
SCRIPTS = PACKAGE / "scripts"

FEATURE_TABLE = DATA / "CHECKPOINT7_REFERENCE_FEATURES.tsv.gz"
CHECKPOINT7_VALIDATION = VALIDATION / "CHECKPOINT7_VALIDATION.json"
BASE_FEATURE_SPEC = MANIFESTS / "CHECKPOINT7_FEATURE_SPEC.json"
BASE_SCRIPT = SCRIPTS / "07_compare_ranking_features.py"

COMPONENTS = DATA / "CHECKPOINT7_METHOD_COMPONENTS.tsv"
ABLATION = DATA / "CHECKPOINT7_METHOD_ABLATION.tsv"
SCORES = DATA / "CHECKPOINT7_METHOD_SCORE_MATRIX.tsv.gz"
INPUT_HASHES = MANIFESTS / "CHECKPOINT7_METHOD_INPUT_HASHES.tsv"
METHOD_SPEC = MANIFESTS / "CHECKPOINT7_METHOD_SPEC.json"
REPORT = REPORTS / "CHECKPOINT7_METHOD_COMPARISON.md"
BUILD = VALIDATION / "CHECKPOINT7_METHOD_BUILD_SUMMARY.json"

DISTANCE = "ligand_centroid_to_orthosteric_site_CA_centroid_A"
FEATURES: Tuple[Tuple[str, str, str], ...] = (
    ("D", DISTANCE, "selected structural distance"),
    ("MW", "molecular_weight", "molecular weight"),
    ("LP", "clogp", "calculated LogP"),
    ("AR", "aromatic_ring_count", "aromatic ring count"),
)
FEATURE_BY_CODE = {code: column for code, column, _label in FEATURES}
CONTINUOUS_CODES = {"D", "MW", "LP"}
ESTIMATORS = ("QNB", "KDE_LEGACY", "KDE100")
JEFFREYS_ALPHA = 0.5
DENSITY_FLOOR = 1e-12
SCORE_DECIMALS = 12
KDE_LIMITS = {
    "KDE_LEGACY": {
        "D": math.log(100.0),
        "MW": math.log(10.0),
        "LP": math.log(10.0),
        "AR": math.log(10.0),
    },
    "KDE100": {code: math.log(100.0) for code, _column, _label in FEATURES},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_tsv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def combinations() -> List[Tuple[str, ...]]:
    codes = [item[0] for item in FEATURES]
    return [
        subset
        for size in range(1, len(codes) + 1)
        for subset in itertools.combinations(codes, size)
    ]


def smoothed_bin_log_lr(bin_ids: np.ndarray, labels: np.ndarray, n_bins: int):
    negative_counts = np.bincount(bin_ids[labels == 0], minlength=n_bins).astype(int)
    positive_counts = np.bincount(bin_ids[labels == 1], minlength=n_bins).astype(int)
    negative_probability = (
        negative_counts + JEFFREYS_ALPHA
    ) / (int(np.sum(labels == 0)) + JEFFREYS_ALPHA * n_bins)
    positive_probability = (
        positive_counts + JEFFREYS_ALPHA
    ) / (int(np.sum(labels == 1)) + JEFFREYS_ALPHA * n_bins)
    bin_log_lr = np.log(positive_probability / negative_probability)
    return bin_log_lr[bin_ids], negative_counts, positive_counts, bin_log_lr


def quantile_component(values: np.ndarray, labels: np.ndarray, code: str):
    if code in CONTINUOUS_CODES:
        requested_bins = int(math.ceil(len(values) ** (1.0 / 3.0)))
        probabilities = np.linspace(0.0, 1.0, requested_bins + 1)
        quantiles = np.quantile(values, probabilities, method="linear")
        internal_edges = np.unique(quantiles[1:-1])
        bin_ids = np.searchsorted(internal_edges, values, side="right")
        n_bins = len(internal_edges) + 1
        edge_values = [float("-inf")] + [float(value) for value in internal_edges] + [float("inf")]
        method = "pooled_equal_frequency_bins_cube_root_rule"
        categories = []
    else:
        categories = sorted(set(int(value) for value in values))
        category_to_bin = {value: index for index, value in enumerate(categories)}
        # The final unobserved-category bin carries its Jeffreys pseudo-count and
        # is available for a held-out category in checkpoint 8.
        bin_ids = np.asarray([category_to_bin[int(value)] for value in values], dtype=int)
        n_bins = len(categories) + 1
        requested_bins = n_bins
        edge_values = []
        method = "observed_categories_plus_unseen_category_bin"
    log_lr, negative_counts, positive_counts, bin_log_lr = smoothed_bin_log_lr(
        bin_ids, labels, n_bins
    )
    model = {
        "estimator_id": "QNB",
        "feature_code": code,
        "feature_column": FEATURE_BY_CODE[code],
        "estimator": method,
        "rows": len(values),
        "requested_bins": requested_bins,
        "actual_bins_including_unseen_bin": n_bins,
        "bin_edges_json": json.dumps(edge_values),
        "categories_json": json.dumps(categories),
        "negative_bin_counts_json": json.dumps(negative_counts.tolist()),
        "positive_bin_counts_json": json.dumps(positive_counts.tolist()),
        "bin_log_lr_json": json.dumps(bin_log_lr.tolist()),
        "smoothing_alpha": JEFFREYS_ALPHA,
        "manual_likelihood_ratio_lower": np.nan,
        "manual_likelihood_ratio_upper": np.nan,
        "positive_cap_rows": 0,
        "negative_cap_rows": 0,
    }
    return log_lr, model


def kde_uncapped_component(values: np.ndarray, labels: np.ndarray, code: str):
    if code in CONTINUOUS_CODES:
        negative = values[labels == 0]
        positive = values[labels == 1]
        negative_kde = gaussian_kde(negative, bw_method="scott")
        positive_kde = gaussian_kde(positive, bw_method="scott")
        lower, upper = float(np.min(values)), float(np.max(values))
        evaluated = np.clip(values, lower, upper)
        negative_density = negative_kde.evaluate(evaluated) + DENSITY_FLOOR
        positive_density = positive_kde.evaluate(evaluated) + DENSITY_FLOOR
        raw_log_lr = np.log(positive_density / negative_density)
        detail = {
            "estimator": "gaussian_kde_scott_full_training_support",
            "requested_bins": np.nan,
            "actual_bins_including_unseen_bin": np.nan,
            "bin_edges_json": json.dumps([lower, upper]),
            "categories_json": json.dumps([]),
            "negative_bin_counts_json": json.dumps([]),
            "positive_bin_counts_json": json.dumps([]),
            "bin_log_lr_json": json.dumps([]),
            "smoothing_alpha": np.nan,
            "negative_kde_factor": float(negative_kde.factor),
            "positive_kde_factor": float(positive_kde.factor),
        }
    else:
        categories = sorted(set(int(value) for value in values))
        category_to_bin = {value: index for index, value in enumerate(categories)}
        bin_ids = np.asarray([category_to_bin[int(value)] for value in values], dtype=int)
        n_bins = len(categories) + 1
        raw_log_lr, negative_counts, positive_counts, bin_log_lr = smoothed_bin_log_lr(
            bin_ids, labels, n_bins
        )
        detail = {
            "estimator": "categorical_frequency_with_unseen_category_bin",
            "requested_bins": n_bins,
            "actual_bins_including_unseen_bin": n_bins,
            "bin_edges_json": json.dumps([]),
            "categories_json": json.dumps(categories),
            "negative_bin_counts_json": json.dumps(negative_counts.tolist()),
            "positive_bin_counts_json": json.dumps(positive_counts.tolist()),
            "bin_log_lr_json": json.dumps(bin_log_lr.tolist()),
            "smoothing_alpha": JEFFREYS_ALPHA,
            "negative_kde_factor": np.nan,
            "positive_kde_factor": np.nan,
        }
    return raw_log_lr, detail


def macro_metrics(frame: pd.DataFrame, score_column: str):
    aucs, aps, rows = [], [], 0
    for _protein, group in frame.groupby("uniprot", sort=False):
        if group.binary_label.nunique() != 2:
            continue
        aucs.append(roc_auc_score(group.binary_label, group[score_column]))
        aps.append(average_precision_score(group.binary_label, group[score_column]))
        rows += len(group)
    return float(np.mean(aucs)), float(np.mean(aps)), len(aucs), rows


def metrics(frame: pd.DataFrame, score_column: str, prefix: str = "") -> dict:
    y = frame.binary_label.to_numpy(dtype=int)
    score = frame[score_column].to_numpy(dtype=float)
    prevalence = float(np.mean(y))
    macro_auc, macro_ap, macro_groups, macro_rows = macro_metrics(frame, score_column)
    result = {
        f"{prefix}rows": len(frame),
        f"{prefix}allosteric_rows": int(np.sum(y)),
        f"{prefix}prevalence": prevalence,
        f"{prefix}AUROC_descriptive": float(roc_auc_score(y, score)),
        f"{prefix}AUPRC_descriptive": float(average_precision_score(y, score)),
        f"{prefix}within_protein_macro_AUROC_descriptive": macro_auc,
        f"{prefix}within_protein_macro_AUPRC_descriptive": macro_ap,
        f"{prefix}within_protein_groups": macro_groups,
        f"{prefix}within_protein_rows": macro_rows,
    }
    return result


def collapse(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    grouped = frame.groupby("site_ligand_signature_id", sort=False)
    if grouped.binary_label.nunique().max() != 1:
        raise RuntimeError("site-ligand signature label conflict")
    return grouped.agg(
        observation_id=("observation_id", "min"),
        uniprot=("uniprot", "first"),
        binary_label=("binary_label", "first"),
        score=(score_column, "median"),
    ).reset_index().rename(columns={"score": score_column})


def main() -> None:
    start = time.time()
    base_validation = json.loads(CHECKPOINT7_VALIDATION.read_text())
    if base_validation.get("status") != "validated_feature_subsets_awaiting_user_review":
        raise RuntimeError("base checkpoint-7 feature ablation is not validated")
    frame = pd.read_csv(FEATURE_TABLE, sep="\t")
    if len(frame) != 5798 or frame.observation_id.duplicated().any():
        raise RuntimeError("checkpoint-7 method-comparison universe changed")
    labels = frame.binary_label.to_numpy(dtype=int)

    component_rows = []
    component_columns = []
    for code, column, _description in FEATURES:
        values = frame[column].to_numpy(dtype=float)
        qnb_log_lr, qnb_model = quantile_component(values, labels, code)
        qnb_column = f"logLR_QNB_{code}"
        frame[qnb_column] = np.round(qnb_log_lr, SCORE_DECIMALS)
        component_columns.append(qnb_column)
        component_rows.append({
            **qnb_model,
            **metrics(frame, qnb_column, prefix="component_"),
        })

        kde_raw_log_lr, kde_detail = kde_uncapped_component(values, labels, code)
        for estimator_id in ("KDE_LEGACY", "KDE100"):
            limit = KDE_LIMITS[estimator_id][code]
            capped = np.clip(kde_raw_log_lr, -limit, limit)
            score_column = f"logLR_{estimator_id}_{code}"
            frame[score_column] = np.round(capped, SCORE_DECIMALS)
            component_columns.append(score_column)
            component_rows.append({
                "estimator_id": estimator_id,
                "feature_code": code,
                "feature_column": column,
                "rows": len(values),
                **kde_detail,
                "manual_likelihood_ratio_lower": float(math.exp(-limit)),
                "manual_likelihood_ratio_upper": float(math.exp(limit)),
                "positive_cap_rows": int(np.sum(kde_raw_log_lr >= limit)),
                "negative_cap_rows": int(np.sum(kde_raw_log_lr <= -limit)),
                **metrics(frame, score_column, prefix="component_"),
            })

    subsets = combinations()
    raw_column = "score_RAW_D"
    frame[raw_column] = np.round(frame[DISTANCE], SCORE_DECIMALS)
    score_columns = [
        "observation_id", "site_ligand_signature_id", "reference_label", "binary_label",
        "uniprot", "pdb_id", "receptor_chain", "ligand_ccd", "ligand_chain",
        "ligand_auth_seq_id", "full_inchikey", "connectivity_key",
    ] + component_columns + [raw_column]
    ablation_rows = [{
        "ranking_id": "RAW:RAW_D",
        "estimator_id": "RAW",
        "combination_id": "RAW_D",
        "feature_codes": "D",
        "n_features": 1,
        "contains_distance": True,
        "contains_chemistry": False,
        "manual_likelihood_cap": False,
        "selection_status": "sanity_baseline_for_checkpoint8",
        **metrics(frame, raw_column),
        **metrics(collapse(frame, raw_column), raw_column, prefix="site_collapsed_"),
    }]
    for estimator_id in ESTIMATORS:
        for subset in subsets:
            combination = "+".join(subset)
            score_column = f"score_{estimator_id}_{'_'.join(subset)}"
            component_subset = [f"logLR_{estimator_id}_{code}" for code in subset]
            frame[score_column] = np.round(frame[component_subset].sum(axis=1), SCORE_DECIMALS)
            score_columns.append(score_column)
            ablation_rows.append({
                "ranking_id": f"{estimator_id}:{combination}",
                "estimator_id": estimator_id,
                "combination_id": combination,
                "feature_codes": ";".join(subset),
                "n_features": len(subset),
                "contains_distance": "D" in subset,
                "contains_chemistry": any(code != "D" for code in subset),
                "manual_likelihood_cap": estimator_id != "QNB",
                "selection_status": "candidate_for_checkpoint8",
                **metrics(frame, score_column),
                **metrics(collapse(frame, score_column), score_column, prefix="site_collapsed_"),
            })
    ablation = pd.DataFrame(ablation_rows)
    raw = ablation.set_index("ranking_id").loc["RAW:RAW_D"]
    ablation["delta_AUROC_vs_raw_distance_descriptive"] = ablation.AUROC_descriptive - raw.AUROC_descriptive
    ablation["delta_within_protein_macro_AUROC_vs_raw_distance_descriptive"] = (
        ablation.within_protein_macro_AUROC_descriptive
        - raw.within_protein_macro_AUROC_descriptive
    )
    ablation["delta_site_collapsed_AUROC_vs_raw_distance_descriptive"] = (
        ablation.site_collapsed_AUROC_descriptive - raw.site_collapsed_AUROC_descriptive
    )

    components = pd.DataFrame(component_rows)
    score_matrix = frame[score_columns].sort_values("observation_id", kind="mergesort")
    atomic_tsv(components, COMPONENTS)
    atomic_tsv(ablation, ABLATION)
    atomic_tsv(score_matrix, SCORES, compression="gzip")

    inputs = []
    for asset_id, path in (
        ("checkpoint7_reference_features", FEATURE_TABLE),
        ("checkpoint7_validation", CHECKPOINT7_VALIDATION),
        ("checkpoint7_feature_spec", BASE_FEATURE_SPEC),
        ("checkpoint7_base_script", BASE_SCRIPT),
    ):
        inputs.append({
            "asset_id": asset_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    atomic_tsv(pd.DataFrame(inputs), INPUT_HASHES)

    requested_bins = int(math.ceil(len(frame) ** (1.0 / 3.0)))
    spec = {
        "checkpoint": 7,
        "addendum": "likelihood_estimator_comparison",
        "status": "validated_candidates_awaiting_user_review",
        "reference_rows": len(frame),
        "features": [code for code, _column, _description in FEATURES],
        "feature_subsets_per_estimator": len(subsets),
        "estimators": {
            "QNB": {
                "continuous_rule": "pooled equal-frequency bins formed on training rows only",
                "bin_count_rule": "ceil(cube_root(training_rows))",
                "requested_bins_full_reference": requested_bins,
                "categorical_rule": "observed categories plus one unseen-category bin",
                "smoothing": "Jeffreys alpha 0.5 per bin and class",
                "manual_likelihood_ratio_cap": None,
            },
            "KDE_LEGACY": {
                "continuous_rule": "Gaussian KDE with Scott bandwidth",
                "distance_likelihood_ratio_range": [0.01, 100.0],
                "chemistry_likelihood_ratio_range_per_feature": [0.1, 10.0],
            },
            "KDE100": {
                "continuous_rule": "Gaussian KDE with Scott bandwidth",
                "all_feature_likelihood_ratio_range": [0.01, 100.0],
            },
        },
        "raw_distance_baseline": True,
        "score_round_decimals": SCORE_DECIMALS,
        "class_prior": "omitted because it is constant within a ranked universe",
        "probability_claim": False,
        "preferred_estimator": None,
        "preferred_feature_subset": None,
        "checkpoint8_gate": "no method is selected from these in-sample descriptive results alone",
    }
    atomic_json(METHOD_SPEC, spec)

    key_rows = ablation.set_index("ranking_id")
    report = f"""# Checkpoint 7 addendum: likelihood-estimator comparison

Status: **built; independent validation required; no estimator selected**

All methods use the same {len(frame):,} exact observations, the same four frozen
features, and the same 15 non-empty feature subsets.

## Methods

- `RAW`: selected structural distance ranked directly; not Bayesian.
- `QNB`: pooled equal-frequency bins with the bin count fixed by
  `ceil(cube_root(training rows))` and Jeffreys 0.5 smoothing. No manual
  likelihood-ratio cap is applied.
- `KDE_LEGACY`: Gaussian KDE; distance ratio limited to 0.01–100 and each
  chemistry ratio to 0.1–10.
- `KDE100`: Gaussian KDE; every feature ratio limited to 0.01–100, reproducing
  the first checkpoint-7 implementation.

## Key descriptive results

| Ranking | Observation AUROC | Within-protein macro AUROC | Site-collapsed AUROC |
|---|---:|---:|---:|
| Raw distance | {key_rows.loc['RAW:RAW_D'].AUROC_descriptive:.3f} | {key_rows.loc['RAW:RAW_D'].within_protein_macro_AUROC_descriptive:.3f} | {key_rows.loc['RAW:RAW_D'].site_collapsed_AUROC_descriptive:.3f} |
| QNB distance | {key_rows.loc['QNB:D'].AUROC_descriptive:.3f} | {key_rows.loc['QNB:D'].within_protein_macro_AUROC_descriptive:.3f} | {key_rows.loc['QNB:D'].site_collapsed_AUROC_descriptive:.3f} |
| QNB all features | {key_rows.loc['QNB:D+MW+LP+AR'].AUROC_descriptive:.3f} | {key_rows.loc['QNB:D+MW+LP+AR'].within_protein_macro_AUROC_descriptive:.3f} | {key_rows.loc['QNB:D+MW+LP+AR'].site_collapsed_AUROC_descriptive:.3f} |
| Legacy-capped KDE all features | {key_rows.loc['KDE_LEGACY:D+MW+LP+AR'].AUROC_descriptive:.3f} | {key_rows.loc['KDE_LEGACY:D+MW+LP+AR'].within_protein_macro_AUROC_descriptive:.3f} | {key_rows.loc['KDE_LEGACY:D+MW+LP+AR'].site_collapsed_AUROC_descriptive:.3f} |
| Uniform-100 KDE all features | {key_rows.loc['KDE100:D+MW+LP+AR'].AUROC_descriptive:.3f} | {key_rows.loc['KDE100:D+MW+LP+AR'].within_protein_macro_AUROC_descriptive:.3f} | {key_rows.loc['KDE100:D+MW+LP+AR'].site_collapsed_AUROC_descriptive:.3f} |

These values are descriptive fits on the complete reference. They show how the
score definitions behave but cannot select the final estimator. Protein- and
family-held-out fitting remains checkpoint 8.
"""
    atomic_text(REPORT, report)

    output_paths = {
        "components": COMPONENTS,
        "ablation": ABLATION,
        "score_matrix": SCORES,
        "input_hashes": INPUT_HASHES,
        "method_spec": METHOD_SPEC,
        "report": REPORT,
    }
    build = {
        "checkpoint": 7,
        "addendum": "likelihood_estimator_comparison",
        "status": "complete_pending_independent_validation",
        "elapsed_seconds": round(time.time() - start, 3),
        "reference_rows": len(frame),
        "estimators": list(ESTIMATORS),
        "feature_subsets_per_estimator": len(subsets),
        "ablation_rows": len(ablation),
        "raw_distance_baselines": 1,
        "preferred_estimator": None,
        "preferred_feature_subset": None,
        "recursive_directory_scan_performed": False,
        "outputs": {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in output_paths.items()
        },
    }
    atomic_json(BUILD, build)
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
