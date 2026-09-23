#!/usr/bin/env python3
"""Independent validation of checkpoint-7 ranking-feature ablation."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

BUILD = VALIDATION / "CHECKPOINT7_BUILD_SUMMARY.json"
GEOMETRY = DATA / "CHECKPOINT6_EXACT_REFERENCE_GEOMETRY.tsv.gz"
FEATURE_TABLE = DATA / "CHECKPOINT7_REFERENCE_FEATURES.tsv.gz"
COMPONENT_MODELS = DATA / "CHECKPOINT7_FEATURE_COMPONENT_MODELS.tsv"
ABLATION = DATA / "CHECKPOINT7_FEATURE_ABLATION.tsv"
SCORES = DATA / "CHECKPOINT7_FEATURE_SCORE_MATRIX.tsv.gz"
IDENTITY_AUDIT = DATA / "CHECKPOINT7_LIGAND_IDENTITY_AUDIT.json"
INPUT_HASHES = MANIFESTS / "CHECKPOINT7_INPUT_HASHES.tsv"
FEATURE_SPEC = MANIFESTS / "CHECKPOINT7_FEATURE_SPEC.json"
OUTPUT = VALIDATION / "CHECKPOINT7_VALIDATION.json"

DISTANCE = "ligand_centroid_to_orthosteric_site_CA_centroid_A"
CODES = ("D", "MW", "LP", "AR")
FEATURE_COLUMNS = {
    "D": DISTANCE,
    "MW": "molecular_weight",
    "LP": "clogp",
    "AR": "aromatic_ring_count",
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


def expected_combinations():
    return [
        "+".join(combination)
        for size in range(1, len(CODES) + 1)
        for combination in itertools.combinations(CODES, size)
    ]


def close(first, second, tolerance=1e-11):
    if pd.isna(first) and pd.isna(second):
        return True
    return math.isclose(float(first), float(second), rel_tol=tolerance, abs_tol=tolerance)


def macro_metrics(frame: pd.DataFrame, score_column: str):
    aucs, aps = [], []
    rows = 0
    for _protein, group in frame.groupby("uniprot", sort=False):
        if group.binary_label.nunique() != 2:
            continue
        aucs.append(roc_auc_score(group.binary_label, group[score_column]))
        aps.append(average_precision_score(group.binary_label, group[score_column]))
        rows += len(group)
    return float(np.mean(aucs)), float(np.mean(aps)), len(aucs), rows


def validate_metrics(frame: pd.DataFrame, score_column: str, row: pd.Series, prefix=""):
    y = frame.binary_label.to_numpy(dtype=int)
    score = frame[score_column].to_numpy(dtype=float)
    prevalence = float(np.mean(y))
    auc = roc_auc_score(y, score)
    ap = average_precision_score(y, score)
    macro_auc, macro_ap, groups, rows = macro_metrics(frame, score_column)
    expected = {
        f"{prefix}rows": len(frame),
        f"{prefix}allosteric_rows": int(np.sum(y)),
        f"{prefix}prevalence": prevalence,
        f"{prefix}AUROC_descriptive": auc,
        f"{prefix}AUPRC_descriptive": ap,
        f"{prefix}AUPRC_lift_descriptive": ap / prevalence,
        f"{prefix}within_protein_macro_AUROC_descriptive": macro_auc,
        f"{prefix}within_protein_macro_AUPRC_descriptive": macro_ap,
        f"{prefix}within_protein_groups": groups,
        f"{prefix}within_protein_rows": rows,
    }
    order = np.lexsort((frame.observation_id.astype(str).to_numpy(), -score))
    for fraction in (0.01, 0.05):
        count = max(1, int(math.ceil(len(frame) * fraction)))
        hits = int(np.sum(y[order[:count]]))
        label = f"top_{int(fraction * 100)}pct"
        expected[f"{prefix}{label}_rows"] = count
        expected[f"{prefix}{label}_allosteric_rows"] = hits
        expected[f"{prefix}{label}_enrichment_descriptive"] = hits / count / prevalence
    for column, value in expected.items():
        if not close(row[column], value):
            raise RuntimeError(f"checkpoint-7 metric mismatch: {row.name}/{column}")


def main() -> None:
    build = json.loads(BUILD.read_text())
    if build.get("checkpoint") != 7 or build.get("status") != "complete_pending_independent_validation":
        raise RuntimeError("checkpoint-7 build state is invalid")
    for name, record in build["outputs"].items():
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint-7 output hash mismatch: {name}")

    inputs = pd.read_csv(INPUT_HASHES, sep="\t", dtype=str, keep_default_na=False)
    if inputs.asset_id.duplicated().any():
        raise RuntimeError("checkpoint-7 input asset IDs are duplicated")
    for row in inputs.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or str(path.stat().st_size) != row.size_bytes or sha256(path) != row.sha256:
            raise RuntimeError(f"checkpoint-7 input hash mismatch: {row.asset_id}")

    features = pd.read_csv(FEATURE_TABLE, sep="\t")
    scores = pd.read_csv(SCORES, sep="\t")
    ablation = pd.read_csv(ABLATION, sep="\t")
    component_models = pd.read_csv(COMPONENT_MODELS, sep="\t")
    geometry = pd.read_csv(
        GEOMETRY,
        sep="\t",
        usecols=["observation_id", DISTANCE],
    )
    identity_audit = json.loads(IDENTITY_AUDIT.read_text())
    spec = json.loads(FEATURE_SPEC.read_text())

    if len(features) != 5798 or features.observation_id.duplicated().any():
        raise RuntimeError("checkpoint-7 reference feature rows changed")
    if features.reference_label.value_counts().to_dict() != {"orthosteric": 4090, "allosteric": 1708}:
        raise RuntimeError("checkpoint-7 reference label counts changed")
    if int(features.uniprot.nunique()) != 370:
        raise RuntimeError("checkpoint-7 protein count changed")
    if int(features.site_ligand_signature_id.nunique()) != build["site_ligand_signatures"]:
        raise RuntimeError("site-ligand signature count differs from build summary")
    if features.groupby("site_ligand_signature_id").binary_label.nunique().max() != 1:
        raise RuntimeError("site-ligand signature label conflict")
    numeric = features[list(FEATURE_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("checkpoint-7 feature matrix is incomplete")

    distance_check = features[["observation_id", DISTANCE]].merge(
        geometry, on="observation_id", suffixes=("_feature", "_geometry"), validate="one_to_one"
    )
    if not np.allclose(
        distance_check[f"{DISTANCE}_feature"],
        distance_check[f"{DISTANCE}_geometry"],
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("checkpoint-7 selected distance differs from checkpoint 6")

    if set(component_models.feature_code) != set(CODES) or len(component_models) != 4:
        raise RuntimeError("checkpoint-7 component-model set changed")
    expected_estimators = {
        "D": "gaussian_kde_scott_full_training_support",
        "MW": "gaussian_kde_scott_full_training_support",
        "LP": "gaussian_kde_scott_full_training_support",
        "AR": "laplace_smoothed_categorical_frequency",
    }
    if component_models.set_index("feature_code").estimator.to_dict() != expected_estimators:
        raise RuntimeError("checkpoint-7 component estimator changed")

    merged = features[[
        "observation_id", "site_ligand_signature_id", "uniprot", "binary_label",
    ]].merge(scores, on=["observation_id", "site_ligand_signature_id", "uniprot", "binary_label"], validate="one_to_one")
    if len(merged) != len(features) or set(merged.observation_id) != set(features.observation_id):
        raise RuntimeError("checkpoint-7 score matrix identity mismatch")
    component_score_columns = [f"logLR_{code}" for code in CODES]
    if not np.isfinite(merged[component_score_columns].to_numpy()).all():
        raise RuntimeError("checkpoint-7 component scores are not finite")
    if (np.abs(merged[component_score_columns].to_numpy()) > math.log(100.0) + 1e-10).any():
        raise RuntimeError("checkpoint-7 component score exceeds the frozen clip")

    combinations = expected_combinations()
    expected_ablation_rows = ["RAW_D"] + combinations
    if list(ablation.combination_id) != expected_ablation_rows or len(ablation) != 16:
        raise RuntimeError("checkpoint-7 exhaustive feature-subset order changed")
    raw_score_column = "score_RAW_D"
    if not np.allclose(
        merged[raw_score_column],
        features.set_index("observation_id").loc[merged.observation_id, DISTANCE].to_numpy(),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("raw-distance sanity baseline differs from selected distance")
    raw_row = ablation.set_index("combination_id").loc["RAW_D"]
    raw_metric_frame = merged[[
        "observation_id", "site_ligand_signature_id", "uniprot", "binary_label",
        raw_score_column,
    ]].copy()
    validate_metrics(raw_metric_frame, raw_score_column, raw_row)
    raw_collapsed = raw_metric_frame.groupby("site_ligand_signature_id", sort=False).agg(
        observation_id=("observation_id", "min"),
        uniprot=("uniprot", "first"),
        binary_label=("binary_label", "first"),
        score=(raw_score_column, "median"),
    ).reset_index().rename(columns={"score": raw_score_column})
    validate_metrics(raw_collapsed, raw_score_column, raw_row, prefix="site_collapsed_")
    for identifier in combinations:
        codes = identifier.split("+")
        score_column = "score_" + identifier.replace("+", "_")
        reconstructed = np.round(
            merged[[f"logLR_{code}" for code in codes]].sum(axis=1), 12
        )
        if not np.allclose(merged[score_column], reconstructed, atol=1e-12, rtol=1e-12):
            raise RuntimeError(f"checkpoint-7 score sum mismatch: {identifier}")
        row = ablation.set_index("combination_id").loc[identifier]
        metric_frame = merged[[
            "observation_id", "site_ligand_signature_id", "uniprot", "binary_label", score_column,
        ]].copy()
        validate_metrics(metric_frame, score_column, row)
        collapsed = metric_frame.groupby("site_ligand_signature_id", sort=False).agg(
            observation_id=("observation_id", "min"),
            uniprot=("uniprot", "first"),
            binary_label=("binary_label", "first"),
            score=(score_column, "median"),
        ).reset_index().rename(columns={"score": score_column})
        validate_metrics(collapsed, score_column, row, prefix="site_collapsed_")

    connectivity_prevalence = features.groupby("connectivity_key").binary_label.mean()
    oracle = features.connectivity_key.map(connectivity_prevalence)
    oracle_auc = roc_auc_score(features.binary_label, oracle)
    both_label = int((features.groupby("connectivity_key").binary_label.nunique() == 2).sum())
    if not close(identity_audit["connectivity_in_sample_prevalence_oracle_AUROC"], oracle_auc):
        raise RuntimeError("ligand-identity audit AUROC mismatch")
    if identity_audit["both_label_connectivity_keys"] != both_label:
        raise RuntimeError("ligand-identity both-label count mismatch")

    if spec.get("selected_structural_distance") != DISTANCE:
        raise RuntimeError("checkpoint-7 structural distance changed")
    if spec.get("ranking_feature_subsets") != combinations or spec.get("preferred_subset") is not None:
        raise RuntimeError("checkpoint-7 feature subset contract changed")
    if spec.get("sanity_baselines") != ["RAW_D: selected structural distance ranked directly"]:
        raise RuntimeError("checkpoint-7 raw-distance sanity baseline changed")
    if spec.get("score_round_decimals") != 12:
        raise RuntimeError("checkpoint-7 deterministic score precision changed")
    if spec.get("tanimoto_role") != "excluded from ranking; retained for later substrate/cofactor similarity QC":
        raise RuntimeError("Tanimoto was reintroduced into the ranking score")
    if build.get("preferred_feature_subset") is not None or build.get("recursive_directory_scan_performed"):
        raise RuntimeError("checkpoint-7 selection or bounded-I/O contract changed")

    raw_distance_row = ablation.set_index("combination_id").loc["RAW_D"]
    distance_row = ablation.set_index("combination_id").loc["D"]
    chemistry_row = ablation.set_index("combination_id").loc["MW+LP+AR"]
    full_row = ablation.set_index("combination_id").loc["D+MW+LP+AR"]
    result = {
        "checkpoint": 7,
        "status": "validated_feature_subsets_awaiting_user_review",
        "reference_rows": len(features),
        "allosteric_rows": int(features.binary_label.sum()),
        "orthosteric_rows": int((features.binary_label == 0).sum()),
        "proteins": int(features.uniprot.nunique()),
        "both_label_proteins": int(sum(
            group.binary_label.nunique() == 2 for _key, group in features.groupby("uniprot")
        )),
        "site_ligand_signatures": int(features.site_ligand_signature_id.nunique()),
        "likelihood_ratio_feature_subsets": len(combinations),
        "sanity_baselines": 1,
        "ablation_rows": len(ablation),
        "selected_structural_distance": DISTANCE,
        "preferred_feature_subset": None,
        "raw_distance_AUROC_descriptive": float(raw_distance_row.AUROC_descriptive),
        "raw_distance_within_protein_macro_AUROC_descriptive": float(
            raw_distance_row.within_protein_macro_AUROC_descriptive
        ),
        "distance_only_AUROC_descriptive": float(distance_row.AUROC_descriptive),
        "distance_only_within_protein_macro_AUROC_descriptive": float(
            distance_row.within_protein_macro_AUROC_descriptive
        ),
        "chemistry_only_AUROC_descriptive": float(chemistry_row.AUROC_descriptive),
        "chemistry_only_within_protein_macro_AUROC_descriptive": float(
            chemistry_row.within_protein_macro_AUROC_descriptive
        ),
        "full_AUROC_descriptive": float(full_row.AUROC_descriptive),
        "full_within_protein_macro_AUROC_descriptive": float(
            full_row.within_protein_macro_AUROC_descriptive
        ),
        "connectivity_in_sample_prevalence_oracle_AUROC": float(oracle_auc),
        "both_label_connectivity_keys": both_label,
        "metrics_role": "descriptive full-reference feature ablation; no held-out performance claim",
        "validated_output_hashes": {
            name: record["sha256"] for name, record in build["outputs"].items()
        },
        "build_summary_sha256": sha256(BUILD),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
