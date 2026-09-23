#!/usr/bin/env python3
"""Independently validate the checkpoint-7 likelihood-estimator comparison."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

FEATURE_TABLE = DATA / "CHECKPOINT7_REFERENCE_FEATURES.tsv.gz"
COMPONENTS = DATA / "CHECKPOINT7_METHOD_COMPONENTS.tsv"
ABLATION = DATA / "CHECKPOINT7_METHOD_ABLATION.tsv"
SCORES = DATA / "CHECKPOINT7_METHOD_SCORE_MATRIX.tsv.gz"
INPUT_HASHES = MANIFESTS / "CHECKPOINT7_METHOD_INPUT_HASHES.tsv"
METHOD_SPEC = MANIFESTS / "CHECKPOINT7_METHOD_SPEC.json"
BUILD = VALIDATION / "CHECKPOINT7_METHOD_BUILD_SUMMARY.json"
OUTPUT = VALIDATION / "CHECKPOINT7_METHOD_VALIDATION.json"

DISTANCE = "ligand_centroid_to_orthosteric_site_CA_centroid_A"
FEATURE_COLUMNS = {
    "D": DISTANCE,
    "MW": "molecular_weight",
    "LP": "clogp",
    "AR": "aromatic_ring_count",
}
FEATURE_CODES = tuple(FEATURE_COLUMNS)
CONTINUOUS = {"D", "MW", "LP"}
ESTIMATORS = ("QNB", "KDE_LEGACY", "KDE100")
ALPHA = 0.5
DENSITY_FLOOR = 1e-12
DECIMALS = 12
CAPS = {
    "KDE_LEGACY": {
        "D": math.log(100.0),
        "MW": math.log(10.0),
        "LP": math.log(10.0),
        "AR": math.log(10.0),
    },
    "KDE100": {code: math.log(100.0) for code in FEATURE_CODES},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def close(first, second, tolerance: float = 2e-10) -> bool:
    if pd.isna(first) and pd.isna(second):
        return True
    return math.isclose(float(first), float(second), rel_tol=tolerance, abs_tol=tolerance)


def subsets():
    return [
        combination
        for size in range(1, len(FEATURE_CODES) + 1)
        for combination in itertools.combinations(FEATURE_CODES, size)
    ]


def smoothed_log_ratios(bin_ids: np.ndarray, labels: np.ndarray, bin_count: int):
    negative = np.bincount(bin_ids[labels == 0], minlength=bin_count).astype(int)
    positive = np.bincount(bin_ids[labels == 1], minlength=bin_count).astype(int)
    negative_probability = (negative + ALPHA) / (int(np.sum(labels == 0)) + ALPHA * bin_count)
    positive_probability = (positive + ALPHA) / (int(np.sum(labels == 1)) + ALPHA * bin_count)
    per_bin = np.log(positive_probability / negative_probability)
    return per_bin[bin_ids], negative, positive, per_bin


def independently_recompute_qnb(values: np.ndarray, labels: np.ndarray, code: str):
    if code in CONTINUOUS:
        requested = int(math.ceil(len(values) ** (1.0 / 3.0)))
        candidate_edges = np.quantile(
            values,
            np.linspace(0.0, 1.0, requested + 1),
            method="linear",
        )[1:-1]
        internal_edges = np.unique(candidate_edges)
        bin_ids = np.digitize(values, internal_edges, right=False)
        bin_count = len(internal_edges) + 1
        edges = [float("-inf"), *[float(value) for value in internal_edges], float("inf")]
        categories = []
    else:
        categories = sorted(np.unique(values.astype(int)).tolist())
        lookup = {category: index for index, category in enumerate(categories)}
        bin_ids = np.asarray([lookup[int(value)] for value in values], dtype=int)
        bin_count = len(categories) + 1
        requested = bin_count
        edges = []
    score, negative, positive, per_bin = smoothed_log_ratios(
        bin_ids, labels, bin_count
    )
    return {
        "score": np.round(score, DECIMALS),
        "requested_bins": requested,
        "bin_count": bin_count,
        "edges": edges,
        "categories": categories,
        "negative": negative,
        "positive": positive,
        "per_bin": per_bin,
    }


def independently_recompute_uncapped(values: np.ndarray, labels: np.ndarray, code: str):
    if code in CONTINUOUS:
        negative_fit = gaussian_kde(values[labels == 0], bw_method="scott")
        positive_fit = gaussian_kde(values[labels == 1], bw_method="scott")
        evaluated = np.clip(values, float(np.min(values)), float(np.max(values)))
        negative_density = negative_fit(evaluated) + DENSITY_FLOOR
        positive_density = positive_fit(evaluated) + DENSITY_FLOOR
        score = np.log(positive_density / negative_density)
        return score, float(negative_fit.factor), float(positive_fit.factor)
    categories = sorted(np.unique(values.astype(int)).tolist())
    lookup = {category: index for index, category in enumerate(categories)}
    bin_ids = np.asarray([lookup[int(value)] for value in values], dtype=int)
    score, _negative, _positive, _per_bin = smoothed_log_ratios(
        bin_ids, labels, len(categories) + 1
    )
    return score, np.nan, np.nan


def macro_metrics(frame: pd.DataFrame, score_column: str):
    aucs, aps, rows = [], [], 0
    for _protein, group in frame.groupby("uniprot", sort=False):
        if group.binary_label.nunique() != 2:
            continue
        aucs.append(roc_auc_score(group.binary_label, group[score_column]))
        aps.append(average_precision_score(group.binary_label, group[score_column]))
        rows += len(group)
    return float(np.mean(aucs)), float(np.mean(aps)), len(aucs), rows


def validate_metrics(frame: pd.DataFrame, score_column: str, row: pd.Series, prefix=""):
    labels = frame.binary_label.to_numpy(dtype=int)
    macro_auc, macro_ap, groups, rows = macro_metrics(frame, score_column)
    expected = {
        f"{prefix}rows": len(frame),
        f"{prefix}allosteric_rows": int(labels.sum()),
        f"{prefix}prevalence": float(labels.mean()),
        f"{prefix}AUROC_descriptive": float(roc_auc_score(labels, frame[score_column])),
        f"{prefix}AUPRC_descriptive": float(average_precision_score(labels, frame[score_column])),
        f"{prefix}within_protein_macro_AUROC_descriptive": macro_auc,
        f"{prefix}within_protein_macro_AUPRC_descriptive": macro_ap,
        f"{prefix}within_protein_groups": groups,
        f"{prefix}within_protein_rows": rows,
    }
    for column, value in expected.items():
        if column not in row.index or not close(row[column], value):
            raise RuntimeError(f"method-comparison metric mismatch: {row.name}/{column}")


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


def parsed_json_list(value):
    return json.loads(value) if isinstance(value, str) else []


def main() -> None:
    build = json.loads(BUILD.read_text())
    if build.get("checkpoint") != 7 or build.get("addendum") != "likelihood_estimator_comparison":
        raise RuntimeError("method-comparison build identity is invalid")
    if build.get("status") != "complete_pending_independent_validation":
        raise RuntimeError("method-comparison build status is invalid")
    for asset_id, record in build["outputs"].items():
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"method-comparison output hash mismatch: {asset_id}")

    inputs = pd.read_csv(INPUT_HASHES, sep="\t", dtype=str, keep_default_na=False)
    if len(inputs) != 4 or inputs.asset_id.duplicated().any():
        raise RuntimeError("method-comparison input manifest changed")
    for row in inputs.itertuples(index=False):
        path = Path(row.path)
        if (
            not path.is_file()
            or str(path.stat().st_size) != row.size_bytes
            or sha256(path) != row.sha256
        ):
            raise RuntimeError(f"method-comparison input hash mismatch: {row.asset_id}")

    features = pd.read_csv(FEATURE_TABLE, sep="\t")
    scores = pd.read_csv(SCORES, sep="\t")
    components = pd.read_csv(COMPONENTS, sep="\t")
    ablation = pd.read_csv(ABLATION, sep="\t")
    spec = json.loads(METHOD_SPEC.read_text())

    if len(features) != 5798 or features.observation_id.duplicated().any():
        raise RuntimeError("method-comparison reference universe changed")
    if features.reference_label.value_counts().to_dict() != {"orthosteric": 4090, "allosteric": 1708}:
        raise RuntimeError("method-comparison reference labels changed")
    if features.uniprot.nunique() != 370 or features.site_ligand_signature_id.nunique() != 3549:
        raise RuntimeError("method-comparison reference identities changed")
    numeric = features[list(FEATURE_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("method-comparison feature matrix is incomplete")

    identity = [
        "observation_id", "site_ligand_signature_id", "reference_label", "binary_label",
        "uniprot", "pdb_id", "receptor_chain", "ligand_ccd", "ligand_chain",
        "ligand_auth_seq_id", "full_inchikey", "connectivity_key",
    ]
    merged = features[[
        "observation_id", "site_ligand_signature_id", "reference_label", "binary_label",
        "uniprot", "pdb_id", "receptor_chain", "ligand_ccd", "ligand_chain",
        "ligand_auth_seq_id", "full_inchikey", "connectivity_key",
    ]].merge(scores, on=identity, how="inner", validate="one_to_one")
    if len(merged) != len(features) or set(merged.observation_id) != set(features.observation_id):
        raise RuntimeError("method-comparison score identities differ from the frozen universe")
    if len(components) != 12 or components[["estimator_id", "feature_code"]].duplicated().any():
        raise RuntimeError("method-comparison component-model grid changed")
    if set(components.estimator_id) != set(ESTIMATORS) or set(components.feature_code) != set(FEATURE_CODES):
        raise RuntimeError("method-comparison component-model identities changed")

    labels = features.binary_label.to_numpy(dtype=int)
    independent_components = {}
    qnb_ranges = {}
    cap_counts = {}
    component_index = components.set_index(["estimator_id", "feature_code"])
    for code, column in FEATURE_COLUMNS.items():
        values = features[column].to_numpy(dtype=float)
        qnb = independently_recompute_qnb(values, labels, code)
        qnb_score = qnb["score"]
        independent_components[("QNB", code)] = qnb_score
        output_column = f"logLR_QNB_{code}"
        if not np.allclose(merged[output_column], qnb_score, atol=2e-11, rtol=2e-11):
            raise RuntimeError(f"QNB component score mismatch: {code}")
        model = component_index.loc[("QNB", code)]
        if int(model.requested_bins) != qnb["requested_bins"] or int(model.actual_bins_including_unseen_bin) != qnb["bin_count"]:
            raise RuntimeError(f"QNB bin-count mismatch: {code}")
        if parsed_json_list(model.negative_bin_counts_json) != qnb["negative"].tolist():
            raise RuntimeError(f"QNB negative-bin counts differ: {code}")
        if parsed_json_list(model.positive_bin_counts_json) != qnb["positive"].tolist():
            raise RuntimeError(f"QNB positive-bin counts differ: {code}")
        if not np.allclose(parsed_json_list(model.bin_log_lr_json), qnb["per_bin"], atol=2e-11, rtol=2e-11):
            raise RuntimeError(f"QNB per-bin log ratios differ: {code}")
        recorded_edges = np.asarray(parsed_json_list(model.bin_edges_json), dtype=float)
        expected_edges = np.asarray(qnb["edges"], dtype=float)
        if recorded_edges.shape != expected_edges.shape or not np.allclose(
            recorded_edges, expected_edges, atol=2e-11, rtol=2e-11
        ):
            raise RuntimeError(f"QNB bin edges differ: {code}")
        if parsed_json_list(model.categories_json) != qnb["categories"]:
            raise RuntimeError(f"QNB categories differ: {code}")
        if not close(model.smoothing_alpha, ALPHA):
            raise RuntimeError(f"QNB smoothing differs: {code}")
        if not pd.isna(model.manual_likelihood_ratio_lower) or not pd.isna(model.manual_likelihood_ratio_upper):
            raise RuntimeError(f"QNB unexpectedly has a manual cap: {code}")
        if int(model.positive_cap_rows) != 0 or int(model.negative_cap_rows) != 0:
            raise RuntimeError(f"QNB unexpectedly reports capped rows: {code}")
        component_metric_frame = merged[[
            "observation_id", "site_ligand_signature_id", "uniprot", "binary_label", output_column
        ]].copy()
        validate_metrics(component_metric_frame, output_column, model, prefix="component_")
        qnb_ranges[code] = {
            "requested_bins": int(qnb["requested_bins"]),
            "actual_bins": int(qnb["bin_count"]),
            "minimum_bin_rows": int(np.min(qnb["negative"] + qnb["positive"])),
            "maximum_bin_rows": int(np.max(qnb["negative"] + qnb["positive"])),
            "minimum_log_likelihood_ratio": float(np.min(qnb["per_bin"])),
            "maximum_log_likelihood_ratio": float(np.max(qnb["per_bin"])),
            "minimum_likelihood_ratio": float(np.exp(np.min(qnb["per_bin"]))),
            "maximum_likelihood_ratio": float(np.exp(np.max(qnb["per_bin"]))),
        }

        uncapped, negative_factor, positive_factor = independently_recompute_uncapped(
            values, labels, code
        )
        for estimator in ("KDE_LEGACY", "KDE100"):
            limit = CAPS[estimator][code]
            expected_score = np.round(np.clip(uncapped, -limit, limit), DECIMALS)
            independent_components[(estimator, code)] = expected_score
            output_column = f"logLR_{estimator}_{code}"
            if not np.allclose(merged[output_column], expected_score, atol=2e-11, rtol=2e-11):
                raise RuntimeError(f"{estimator} component score mismatch: {code}")
            model = component_index.loc[(estimator, code)]
            if not close(model.manual_likelihood_ratio_lower, math.exp(-limit)):
                raise RuntimeError(f"{estimator} lower cap differs: {code}")
            if not close(model.manual_likelihood_ratio_upper, math.exp(limit)):
                raise RuntimeError(f"{estimator} upper cap differs: {code}")
            positive_count = int(np.sum(uncapped >= limit))
            negative_count = int(np.sum(uncapped <= -limit))
            if int(model.positive_cap_rows) != positive_count or int(model.negative_cap_rows) != negative_count:
                raise RuntimeError(f"{estimator} capped-row count differs: {code}")
            if code in CONTINUOUS:
                if not close(model.negative_kde_factor, negative_factor) or not close(model.positive_kde_factor, positive_factor):
                    raise RuntimeError(f"{estimator} KDE bandwidth differs: {code}")
            component_metric_frame = merged[[
                "observation_id", "site_ligand_signature_id", "uniprot", "binary_label", output_column
            ]].copy()
            validate_metrics(component_metric_frame, output_column, model, prefix="component_")
            cap_counts[f"{estimator}:{code}"] = {
                "positive_cap_rows": positive_count,
                "negative_cap_rows": negative_count,
            }

    expected_rankings = ["RAW:RAW_D"] + [
        f"{estimator}:{'+'.join(combination)}"
        for estimator in ESTIMATORS
        for combination in subsets()
    ]
    if ablation.ranking_id.tolist() != expected_rankings or len(ablation) != 46:
        raise RuntimeError("method-comparison exhaustive ablation grid changed")
    raw_expected = np.round(features[DISTANCE].to_numpy(dtype=float), DECIMALS)
    if not np.allclose(merged.score_RAW_D, raw_expected, atol=2e-11, rtol=2e-11):
        raise RuntimeError("raw-distance baseline differs from the selected distance")

    ablation_index = ablation.set_index("ranking_id")
    raw_frame = merged[[
        "observation_id", "site_ligand_signature_id", "uniprot", "binary_label", "score_RAW_D"
    ]].copy()
    validate_metrics(raw_frame, "score_RAW_D", ablation_index.loc["RAW:RAW_D"])
    validate_metrics(
        collapse(raw_frame, "score_RAW_D"),
        "score_RAW_D",
        ablation_index.loc["RAW:RAW_D"],
        prefix="site_collapsed_",
    )
    for estimator in ESTIMATORS:
        for combination in subsets():
            combination_id = "+".join(combination)
            ranking_id = f"{estimator}:{combination_id}"
            score_column = f"score_{estimator}_{'_'.join(combination)}"
            expected_score = np.round(
                np.column_stack([
                    independent_components[(estimator, code)] for code in combination
                ]).sum(axis=1),
                DECIMALS,
            )
            if not np.allclose(merged[score_column], expected_score, atol=4e-11, rtol=4e-11):
                raise RuntimeError(f"method-comparison score sum differs: {ranking_id}")
            metric_frame = merged[[
                "observation_id", "site_ligand_signature_id", "uniprot", "binary_label", score_column
            ]].copy()
            validate_metrics(metric_frame, score_column, ablation_index.loc[ranking_id])
            validate_metrics(
                collapse(metric_frame, score_column),
                score_column,
                ablation_index.loc[ranking_id],
                prefix="site_collapsed_",
            )

    raw = ablation_index.loc["RAW:RAW_D"]
    for ranking_id, row in ablation_index.iterrows():
        if not close(row.delta_AUROC_vs_raw_distance_descriptive, row.AUROC_descriptive - raw.AUROC_descriptive):
            raise RuntimeError(f"pooled raw-distance delta differs: {ranking_id}")
        if not close(
            row.delta_within_protein_macro_AUROC_vs_raw_distance_descriptive,
            row.within_protein_macro_AUROC_descriptive - raw.within_protein_macro_AUROC_descriptive,
        ):
            raise RuntimeError(f"within-protein raw-distance delta differs: {ranking_id}")
        if not close(
            row.delta_site_collapsed_AUROC_vs_raw_distance_descriptive,
            row.site_collapsed_AUROC_descriptive - raw.site_collapsed_AUROC_descriptive,
        ):
            raise RuntimeError(f"site-collapsed raw-distance delta differs: {ranking_id}")

    if spec.get("preferred_estimator") is not None or spec.get("preferred_feature_subset") is not None:
        raise RuntimeError("method or feature subset was selected before checkpoint 8")
    if spec["estimators"]["QNB"].get("manual_likelihood_ratio_cap") is not None:
        raise RuntimeError("QNB method specification contains a manual cap")
    if spec["estimators"]["KDE_LEGACY"].get("distance_likelihood_ratio_range") != [0.01, 100.0]:
        raise RuntimeError("legacy distance cap changed")
    if spec["estimators"]["KDE_LEGACY"].get("chemistry_likelihood_ratio_range_per_feature") != [0.1, 10.0]:
        raise RuntimeError("legacy chemistry cap changed")
    if spec["estimators"]["KDE100"].get("all_feature_likelihood_ratio_range") != [0.01, 100.0]:
        raise RuntimeError("uniform KDE cap changed")
    if build.get("preferred_estimator") is not None or build.get("preferred_feature_subset") is not None:
        raise RuntimeError("method-comparison build selected a method prematurely")
    if build.get("recursive_directory_scan_performed") is not False:
        raise RuntimeError("bounded-I/O contract changed")

    key_ids = [
        "RAW:RAW_D",
        "QNB:D",
        "QNB:D+MW+LP+AR",
        "KDE_LEGACY:D",
        "KDE_LEGACY:D+MW+LP+AR",
        "KDE100:D",
        "KDE100:D+MW+LP+AR",
    ]
    key_results = {
        ranking_id: {
            "AUROC_descriptive": float(ablation_index.loc[ranking_id].AUROC_descriptive),
            "within_protein_macro_AUROC_descriptive": float(
                ablation_index.loc[ranking_id].within_protein_macro_AUROC_descriptive
            ),
            "site_collapsed_AUROC_descriptive": float(
                ablation_index.loc[ranking_id].site_collapsed_AUROC_descriptive
            ),
        }
        for ranking_id in key_ids
    }
    result = {
        "checkpoint": 7,
        "addendum": "likelihood_estimator_comparison",
        "status": "validated_likelihood_estimators_awaiting_user_review",
        "reference_rows": len(features),
        "allosteric_rows": int(labels.sum()),
        "orthosteric_rows": int(np.sum(labels == 0)),
        "proteins": int(features.uniprot.nunique()),
        "site_ligand_signatures": int(features.site_ligand_signature_id.nunique()),
        "estimators": list(ESTIMATORS),
        "feature_subsets_per_estimator": len(subsets()),
        "ablation_rows": len(ablation),
        "raw_distance_baselines": 1,
        "qnb_manual_likelihood_ratio_cap": None,
        "qnb_component_ranges": qnb_ranges,
        "kde_cap_counts": cap_counts,
        "key_descriptive_results": key_results,
        "preferred_estimator": None,
        "preferred_feature_subset": None,
        "metrics_role": "full-reference descriptive comparison; not held-out model selection",
        "recursive_directory_scan_performed": False,
        "validated_output_hashes": {
            asset_id: record["sha256"] for asset_id, record in build["outputs"].items()
        },
        "build_summary_sha256": sha256(BUILD),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
