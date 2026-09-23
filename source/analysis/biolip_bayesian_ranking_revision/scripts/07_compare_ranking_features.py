#!/usr/bin/env python3
"""Checkpoint 7: compare ranking-feature subsets on the frozen exact references.

This is a full-reference descriptive ablation. It freezes no preferred feature
subset and performs no protein/family-held-out claim; those belong to checkpoint 8.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
REPORTS = PACKAGE / "reports"
VALIDATION = PACKAGE / "validation"

MASTER = DATA / "BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz"
GEOMETRY = DATA / "CHECKPOINT6_EXACT_REFERENCE_GEOMETRY.tsv.gz"
DESCRIPTORS = DATA / "CHECKPOINT7_LIGAND_DESCRIPTORS.tsv.gz"
DESCRIPTOR_BUILD = VALIDATION / "CHECKPOINT7_DESCRIPTOR_BUILD.json"
CHECKPOINT6_VALIDATION = VALIDATION / "CHECKPOINT6_VALIDATION.json"
DISTANCE_SELECTION = MANIFESTS / "CHECKPOINT6_DISTANCE_SELECTION.json"

FEATURE_TABLE = DATA / "CHECKPOINT7_REFERENCE_FEATURES.tsv.gz"
COMPONENT_MODELS = DATA / "CHECKPOINT7_FEATURE_COMPONENT_MODELS.tsv"
ABLATION = DATA / "CHECKPOINT7_FEATURE_ABLATION.tsv"
SCORES = DATA / "CHECKPOINT7_FEATURE_SCORE_MATRIX.tsv.gz"
DISTRIBUTIONS = DATA / "CHECKPOINT7_FEATURE_DISTRIBUTIONS.tsv"
CORRELATIONS = DATA / "CHECKPOINT7_FEATURE_CORRELATIONS.tsv"
IDENTITY_AUDIT = DATA / "CHECKPOINT7_LIGAND_IDENTITY_AUDIT.json"
INPUT_HASHES = MANIFESTS / "CHECKPOINT7_INPUT_HASHES.tsv"
FEATURE_SPEC = MANIFESTS / "CHECKPOINT7_FEATURE_SPEC.json"
REPORT = REPORTS / "CHECKPOINT7_FEATURE_ABLATION.md"
BUILD = VALIDATION / "CHECKPOINT7_BUILD_SUMMARY.json"

DISTANCE = "ligand_centroid_to_orthosteric_site_CA_centroid_A"
FEATURES: Tuple[Tuple[str, str, str], ...] = (
    ("D", DISTANCE, "selected structural distance"),
    ("MW", "molecular_weight", "molecular weight"),
    ("LP", "clogp", "calculated LogP"),
    ("AR", "aromatic_ring_count", "aromatic ring count"),
)
FEATURE_BY_CODE = {code: column for code, column, _label in FEATURES}
FEATURE_LABEL = {code: label for code, _column, label in FEATURES}
CONTINUOUS_CODES = {"D", "MW", "LP"}
LOG_LR_LIMIT = math.log(100.0)
CATEGORICAL_ALPHA = 0.5
DENSITY_FLOOR = 1e-12


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


def feature_combinations() -> List[Tuple[str, ...]]:
    codes = [item[0] for item in FEATURES]
    return [
        combination
        for size in range(1, len(codes) + 1)
        for combination in itertools.combinations(codes, size)
    ]


def combination_id(codes: Sequence[str]) -> str:
    return "+".join(codes)


def fit_continuous_component(values: np.ndarray, labels: np.ndarray, code: str):
    negative = values[labels == 0]
    positive = values[labels == 1]
    if min(len(negative), len(positive)) < 30:
        raise RuntimeError(f"too few values to fit {code}")
    if len(np.unique(negative)) < 2 or len(np.unique(positive)) < 2:
        raise RuntimeError(f"continuous component {code} is degenerate")
    negative_kde = gaussian_kde(negative, bw_method="scott")
    positive_kde = gaussian_kde(positive, bw_method="scott")
    lower = float(np.min(values))
    upper = float(np.max(values))
    evaluated = np.clip(values, lower, upper)
    negative_density = negative_kde.evaluate(evaluated) + DENSITY_FLOOR
    positive_density = positive_kde.evaluate(evaluated) + DENSITY_FLOOR
    log_lr = np.clip(np.log(positive_density / negative_density), -LOG_LR_LIMIT, LOG_LR_LIMIT)
    model = {
        "feature_code": code,
        "feature_column": FEATURE_BY_CODE[code],
        "estimator": "gaussian_kde_scott_full_training_support",
        "negative_rows": len(negative),
        "positive_rows": len(positive),
        "negative_unique_values": len(np.unique(negative)),
        "positive_unique_values": len(np.unique(positive)),
        "evaluation_clip_min": lower,
        "evaluation_clip_max": upper,
        "negative_kde_factor": float(negative_kde.factor),
        "positive_kde_factor": float(positive_kde.factor),
        "categorical_alpha": np.nan,
        "log_lr_clip_abs": LOG_LR_LIMIT,
    }
    return log_lr, model


def fit_categorical_component(values: np.ndarray, labels: np.ndarray, code: str):
    categories = sorted(set(int(value) for value in values))
    category_count = len(categories)
    scores: Dict[int, float] = {}
    for category in categories:
        negative_count = int(np.sum((values == category) & (labels == 0)))
        positive_count = int(np.sum((values == category) & (labels == 1)))
        negative_probability = (
            negative_count + CATEGORICAL_ALPHA
        ) / (int(np.sum(labels == 0)) + CATEGORICAL_ALPHA * category_count)
        positive_probability = (
            positive_count + CATEGORICAL_ALPHA
        ) / (int(np.sum(labels == 1)) + CATEGORICAL_ALPHA * category_count)
        scores[category] = float(np.clip(
            math.log(positive_probability / negative_probability),
            -LOG_LR_LIMIT,
            LOG_LR_LIMIT,
        ))
    log_lr = np.asarray([scores[int(value)] for value in values], dtype=float)
    model = {
        "feature_code": code,
        "feature_column": FEATURE_BY_CODE[code],
        "estimator": "laplace_smoothed_categorical_frequency",
        "negative_rows": int(np.sum(labels == 0)),
        "positive_rows": int(np.sum(labels == 1)),
        "negative_unique_values": int(len(set(values[labels == 0]))),
        "positive_unique_values": int(len(set(values[labels == 1]))),
        "evaluation_clip_min": int(min(categories)),
        "evaluation_clip_max": int(max(categories)),
        "negative_kde_factor": np.nan,
        "positive_kde_factor": np.nan,
        "categorical_alpha": CATEGORICAL_ALPHA,
        "log_lr_clip_abs": LOG_LR_LIMIT,
    }
    return log_lr, model


def macro_metrics(frame: pd.DataFrame, score_column: str):
    aucs, aps = [], []
    rows = 0
    for _protein, group in frame.groupby("uniprot", sort=False):
        if group.binary_label.nunique() != 2:
            continue
        aucs.append(roc_auc_score(group.binary_label, group[score_column]))
        aps.append(average_precision_score(group.binary_label, group[score_column]))
        rows += len(group)
    return (
        float(np.mean(aucs)) if aucs else np.nan,
        float(np.mean(aps)) if aps else np.nan,
        len(aucs),
        rows,
    )


def ranking_metrics(frame: pd.DataFrame, score_column: str, prefix: str = "") -> dict:
    y = frame.binary_label.to_numpy(dtype=int)
    score = frame[score_column].to_numpy(dtype=float)
    prevalence = float(np.mean(y))
    auc = float(roc_auc_score(y, score))
    ap = float(average_precision_score(y, score))
    macro_auc, macro_ap, macro_groups, macro_rows = macro_metrics(frame, score_column)
    order = np.lexsort((frame.observation_id.astype(str).to_numpy(), -score))
    result = {
        f"{prefix}rows": len(frame),
        f"{prefix}allosteric_rows": int(np.sum(y)),
        f"{prefix}prevalence": prevalence,
        f"{prefix}AUROC_descriptive": auc,
        f"{prefix}AUPRC_descriptive": ap,
        f"{prefix}AUPRC_lift_descriptive": ap / prevalence if prevalence else np.nan,
        f"{prefix}within_protein_macro_AUROC_descriptive": macro_auc,
        f"{prefix}within_protein_macro_AUPRC_descriptive": macro_ap,
        f"{prefix}within_protein_groups": macro_groups,
        f"{prefix}within_protein_rows": macro_rows,
    }
    for fraction in (0.01, 0.05):
        count = max(1, int(math.ceil(len(frame) * fraction)))
        hits = int(np.sum(y[order[:count]]))
        label = f"top_{int(fraction * 100)}pct"
        result[f"{prefix}{label}_rows"] = count
        result[f"{prefix}{label}_allosteric_rows"] = hits
        result[f"{prefix}{label}_enrichment_descriptive"] = (
            hits / count / prevalence if prevalence else np.nan
        )
    return result


def collapse_site_signatures(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    grouped = frame.groupby("site_ligand_signature_id", sort=False, dropna=False)
    conflicts = grouped.binary_label.nunique()
    if (conflicts > 1).any():
        raise RuntimeError("site-ligand signatures carry conflicting labels")
    collapsed = grouped.agg(
        observation_id=("observation_id", "min"),
        uniprot=("uniprot", "first"),
        binary_label=("binary_label", "first"),
        score=(score_column, "median"),
        source_observations=("observation_id", "size"),
    ).reset_index()
    return collapsed.rename(columns={"score": score_column})


def main() -> None:
    start = time.time()
    validation = json.loads(CHECKPOINT6_VALIDATION.read_text())
    selection = json.loads(DISTANCE_SELECTION.read_text())
    descriptor_build = json.loads(DESCRIPTOR_BUILD.read_text())
    if validation.get("status") != "validated_candidates_awaiting_user_selection":
        raise RuntimeError("checkpoint 6 validation is not complete")
    if selection.get("selected_distance") != DISTANCE:
        raise RuntimeError("checkpoint-7 selected distance differs from the user decision")
    if descriptor_build.get("status") != "validated":
        raise RuntimeError("checkpoint-7 ligand descriptors are not validated")

    geometry = pd.read_csv(GEOMETRY, sep="\t", dtype=str, keep_default_na=False)
    frame = geometry.loc[
        geometry.geometry_status.eq("ok")
        & (
            geometry.reference_label.eq("orthosteric")
            | geometry.source_site_relation.eq("distinct_from_orthosteric_site")
        )
    ].copy()
    master = pd.read_csv(
        MASTER,
        sep="\t",
        usecols=[
            "observation_id", "canonical_smiles", "resolution", "ec_number",
            "active_orthosteric_pair_reference",
        ],
        dtype=str,
        keep_default_na=False,
    )
    descriptors = pd.read_csv(DESCRIPTORS, sep="\t")
    frame = frame.merge(master, on="observation_id", validate="one_to_one")
    frame = frame.merge(descriptors, on="canonical_smiles", validate="many_to_one")
    frame["binary_label"] = frame.reference_label.eq("allosteric").astype(int)
    for _code, column, _label in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if len(frame) != 5798 or int(frame.binary_label.sum()) != 1708:
        raise RuntimeError("checkpoint-7 comparison universe changed")
    if frame.observation_id.duplicated().any():
        raise RuntimeError("checkpoint-7 observation IDs are duplicated")
    if not np.isfinite(frame[[item[1] for item in FEATURES]].to_numpy(dtype=float)).all():
        raise RuntimeError("checkpoint-7 feature matrix contains missing values")
    if not np.allclose(
        frame[DISTANCE].to_numpy(dtype=float),
        pd.to_numeric(frame[DISTANCE], errors="coerce").to_numpy(dtype=float),
    ):
        raise RuntimeError("selected distance conversion failed")

    signature_text = (
        frame.uniprot.astype(str) + "|" + frame.connectivity_key.astype(str) + "|"
        + frame.binding_uniprot_positions.astype(str)
    )
    frame["site_ligand_signature_id"] = signature_text.map(
        lambda value: "BLSITE_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    )

    labels = frame.binary_label.to_numpy(dtype=int)
    component_rows = []
    for code, column, _label in FEATURES:
        values = frame[column].to_numpy(dtype=float)
        if code in CONTINUOUS_CODES:
            log_lr, model = fit_continuous_component(values, labels, code)
        else:
            log_lr, model = fit_categorical_component(values, labels, code)
        score_column = f"logLR_{code}"
        frame[score_column] = np.round(log_lr, 12)
        component_metrics = ranking_metrics(frame, score_column, prefix="component_")
        component_rows.append({**model, **component_metrics})

    combinations = feature_combinations()
    raw_distance_score_column = "score_RAW_D"
    frame[raw_distance_score_column] = np.round(frame[DISTANCE], 12)
    raw_metrics = ranking_metrics(frame, raw_distance_score_column)
    raw_collapsed = collapse_site_signatures(frame, raw_distance_score_column)
    raw_collapsed_metrics = ranking_metrics(
        raw_collapsed, raw_distance_score_column, prefix="site_collapsed_"
    )
    ablation_rows = [{
        "combination_id": "RAW_D",
        "feature_codes": "D",
        "feature_columns": DISTANCE,
        "feature_description": "untransformed monotonic structural distance sanity baseline",
        "n_features": 1,
        "contains_distance": True,
        "contains_chemistry": False,
        "score_estimator": "raw_distance_larger_is_more_distal",
        "selection_status": "sanity_baseline_for_checkpoint8",
        **raw_metrics,
        **raw_collapsed_metrics,
    }]
    score_columns = [
        "observation_id", "site_ligand_signature_id", "reference_label", "binary_label",
        "uniprot", "pdb_id", "receptor_chain", "ligand_ccd", "ligand_chain",
        "ligand_auth_seq_id", "full_inchikey", "connectivity_key",
    ] + [f"logLR_{code}" for code, _column, _label in FEATURES] + [raw_distance_score_column]
    for codes in combinations:
        identifier = combination_id(codes)
        score_column = "score_" + identifier.replace("+", "_")
        frame[score_column] = np.round(
            frame[[f"logLR_{code}" for code in codes]].sum(axis=1), 12
        )
        score_columns.append(score_column)
        metrics = ranking_metrics(frame, score_column)
        collapsed = collapse_site_signatures(frame, score_column)
        collapsed_metrics = ranking_metrics(collapsed, score_column, prefix="site_collapsed_")
        ablation_rows.append({
            "combination_id": identifier,
            "feature_codes": ";".join(codes),
            "feature_columns": ";".join(FEATURE_BY_CODE[code] for code in codes),
            "feature_description": "; ".join(FEATURE_LABEL[code] for code in codes),
            "n_features": len(codes),
            "contains_distance": "D" in codes,
            "contains_chemistry": any(code != "D" for code in codes),
            "score_estimator": "sum_of_feature_log_likelihood_ratios",
            "selection_status": "candidate_for_checkpoint8",
            **metrics,
            **collapsed_metrics,
        })
    ablation = pd.DataFrame(ablation_rows)
    kde_distance_baseline = ablation.set_index("combination_id").loc["D"]
    raw_distance_baseline = ablation.set_index("combination_id").loc["RAW_D"]
    ablation["delta_AUROC_vs_kde_distance_only_descriptive"] = (
        ablation.AUROC_descriptive - kde_distance_baseline.AUROC_descriptive
    )
    ablation["delta_within_protein_macro_AUROC_vs_kde_distance_only_descriptive"] = (
        ablation.within_protein_macro_AUROC_descriptive
        - kde_distance_baseline.within_protein_macro_AUROC_descriptive
    )
    ablation["delta_AUROC_vs_raw_distance_descriptive"] = (
        ablation.AUROC_descriptive - raw_distance_baseline.AUROC_descriptive
    )
    ablation["delta_within_protein_macro_AUROC_vs_raw_distance_descriptive"] = (
        ablation.within_protein_macro_AUROC_descriptive
        - raw_distance_baseline.within_protein_macro_AUROC_descriptive
    )
    ablation["delta_site_collapsed_AUROC_vs_raw_distance_descriptive"] = (
        ablation.site_collapsed_AUROC_descriptive
        - raw_distance_baseline.site_collapsed_AUROC_descriptive
    )

    distribution_rows = []
    for code, column, label in FEATURES:
        for reference_label, group in frame.groupby("reference_label"):
            values = group[column]
            distribution_rows.append({
                "feature_code": code,
                "feature_column": column,
                "feature_description": label,
                "reference_label": reference_label,
                "rows": len(group),
                "unique_values": int(values.nunique()),
                "minimum": float(values.min()),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std()),
            })
    distributions = pd.DataFrame(distribution_rows)

    correlation_rows = []
    for population, group in [("all", frame)] + list(frame.groupby("reference_label")):
        for left_index, (left_code, left_column, _left_label) in enumerate(FEATURES):
            for right_code, right_column, _right_label in FEATURES[left_index + 1:]:
                correlation_rows.append({
                    "population": population,
                    "feature_1": left_code,
                    "feature_2": right_code,
                    "rows": len(group),
                    "spearman_rho": float(
                        spearmanr(group[left_column], group[right_column]).statistic
                    ),
                })
    correlations = pd.DataFrame(correlation_rows)

    connectivity_prevalence = frame.groupby("connectivity_key").binary_label.mean()
    frame["connectivity_in_sample_prevalence"] = frame.connectivity_key.map(connectivity_prevalence)
    identity_oracle_auc = float(roc_auc_score(
        frame.binary_label, frame.connectivity_in_sample_prevalence
    ))
    connectivity_groups = frame.groupby("connectivity_key")
    both_label_connectivities = int((connectivity_groups.binary_label.nunique() == 2).sum())
    rows_in_both_label_connectivities = int(sum(
        len(group) for _key, group in connectivity_groups if group.binary_label.nunique() == 2
    ))
    identity_audit = {
        "status": "descriptive_data_structure_audit_not_a_model_baseline",
        "rows": len(frame),
        "allosteric_rows": int(frame.binary_label.sum()),
        "orthosteric_rows": int((frame.binary_label == 0).sum()),
        "proteins": int(frame.uniprot.nunique()),
        "proteins_with_both_labels": int(sum(
            group.binary_label.nunique() == 2 for _key, group in frame.groupby("uniprot")
        )),
        "unique_full_inchikey": int(frame.full_inchikey.nunique()),
        "unique_connectivity_key": int(frame.connectivity_key.nunique()),
        "allosteric_unique_connectivity_key": int(
            frame.loc[frame.binary_label.eq(1), "connectivity_key"].nunique()
        ),
        "orthosteric_unique_connectivity_key": int(
            frame.loc[frame.binary_label.eq(0), "connectivity_key"].nunique()
        ),
        "both_label_connectivity_keys": both_label_connectivities,
        "rows_in_both_label_connectivity_keys": rows_in_both_label_connectivities,
        "connectivity_in_sample_prevalence_oracle_AUROC": identity_oracle_auc,
        "interpretation": "quantifies label information in repeated ligand identity; it is not an achievable held-out model baseline",
    }

    feature_columns = [
        "observation_id", "site_ligand_signature_id", "source_ordinal", "reference_label",
        "binary_label", "reference_source", "pdb_id", "receptor_chain", "uniprot",
        "ligand_ccd", "ligand_chain", "ligand_auth_seq_id", "full_inchikey",
        "connectivity_key", "canonical_smiles", "binding_uniprot_positions",
        "source_site_relation", "candidate_orthosteric_site_residue_overlap_count",
        "ligand_to_orthosteric_site_min_heavy_A", DISTANCE, "molecular_weight", "clogp",
        "aromatic_ring_count", "heavy_atom_count", "resolution", "ec_number",
        "active_broad_orthosteric_pair_reference", "descriptor_id", "descriptor_status",
    ]
    feature_table = frame[feature_columns].sort_values(
        ["source_ordinal", "observation_id"], kind="mergesort"
    )
    score_matrix = frame[score_columns].sort_values("observation_id", kind="mergesort")
    component_models = pd.DataFrame(component_rows)

    atomic_tsv(feature_table, FEATURE_TABLE, compression="gzip")
    atomic_tsv(component_models, COMPONENT_MODELS)
    atomic_tsv(ablation, ABLATION)
    atomic_tsv(score_matrix, SCORES, compression="gzip")
    atomic_tsv(distributions, DISTRIBUTIONS)
    atomic_tsv(correlations, CORRELATIONS)
    atomic_json(IDENTITY_AUDIT, identity_audit)

    input_rows = []
    for asset_id, path in (
        ("checkpoint6_geometry", GEOMETRY),
        ("checkpoint6_validation", CHECKPOINT6_VALIDATION),
        ("checkpoint6_distance_selection", DISTANCE_SELECTION),
        ("exact_observation_master", MASTER),
        ("checkpoint7_ligand_descriptors", DESCRIPTORS),
        ("checkpoint7_descriptor_build", DESCRIPTOR_BUILD),
    ):
        input_rows.append({
            "asset_id": asset_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    atomic_tsv(pd.DataFrame(input_rows), INPUT_HASHES)

    feature_spec = {
        "checkpoint": 7,
        "status": "feature_subsets_frozen_for_checkpoint8_awaiting_user_review",
        "selected_structural_distance": DISTANCE,
        "features": [
            {"code": code, "column": column, "description": label}
            for code, column, label in FEATURES
        ],
        "ranking_feature_subsets": [combination_id(item) for item in combinations],
        "sanity_baselines": ["RAW_D: selected structural distance ranked directly"],
        "ranking_score": "sum of per-feature allosteric-versus-orthosteric log likelihood ratios",
        "continuous_estimator": "Gaussian KDE with Scott bandwidth, full training support, and evaluation clipped to training min/max",
        "aromatic_ring_estimator": "categorical frequency with Jeffreys/Laplace alpha 0.5",
        "per_feature_log_likelihood_ratio_clip": [-LOG_LR_LIMIT, LOG_LR_LIMIT],
        "score_round_decimals": 12,
        "class_prior_in_ranking_score": "omitted because a constant prior does not change rank",
        "output_interpretation": "uncalibrated ranking score, never probability",
        "tanimoto_role": "excluded from ranking; retained for later substrate/cofactor similarity QC",
        "missing_feature_policy": "not exercised in the exact reference because all four features are complete; final ranking must report support rather than silently impute",
        "checkpoint7_metrics_role": "descriptive full-reference ablation only",
        "checkpoint8_requirement": "the raw-distance baseline and all 15 frozen likelihood-ratio subsets must be evaluated using training-fold fits only",
        "preferred_subset": None,
    }
    atomic_json(FEATURE_SPEC, feature_spec)

    sorted_auc = ablation.sort_values("AUROC_descriptive", ascending=False)
    sorted_macro = ablation.sort_values("within_protein_macro_AUROC_descriptive", ascending=False)
    raw_distance_row = ablation.set_index("combination_id").loc["RAW_D"]
    distance_row = ablation.set_index("combination_id").loc["D"]
    chemistry_row = ablation.set_index("combination_id").loc["MW+LP+AR"]
    full_row = ablation.set_index("combination_id").loc["D+MW+LP+AR"]
    report = f"""# Checkpoint 7: ranking-feature ablation

Status: **built; independent validation required; no preferred subset selected**

## Frozen comparison universe

- Exact observations: **{len(frame):,}** ({int(frame.binary_label.sum()):,} allosteric,
  {int((frame.binary_label == 0).sum()):,} orthosteric)
- Proteins: **{frame.uniprot.nunique():,}**; proteins carrying both labels: **{identity_audit['proteins_with_both_labels']:,}**
- Exact site–ligand signatures after collapsing repeated structures: **{frame.site_ligand_signature_id.nunique():,}**
- All four ranking features are complete for every row.

## Compared ranking features

The exhaustive 15 non-empty subsets of four features were scored with one
unchanged likelihood-ratio ranking rule. Direct monotonic distance ranking was
added as a required sanity baseline:

- `D`: exact ligand centroid to orthosteric-site C-alpha centroid distance;
- `MW`: molecular weight;
- `LP`: calculated LogP;
- `AR`: aromatic ring count.

Tanimoto similarity is not a ranking feature. It remains a later biochemical-role
quality-control variable. Scores are uncalibrated log likelihood ratios, not
probabilities.

## Descriptive results before held-out evaluation

- Raw distance: AUROC **{raw_distance_row.AUROC_descriptive:.3f}**; within-protein macro AUROC
  **{raw_distance_row.within_protein_macro_AUROC_descriptive:.3f}**.
- KDE-transformed distance alone: AUROC **{distance_row.AUROC_descriptive:.3f}**; within-protein macro AUROC
  **{distance_row.within_protein_macro_AUROC_descriptive:.3f}**.
- Chemistry alone (`MW+LP+AR`): AUROC **{chemistry_row.AUROC_descriptive:.3f}**;
  within-protein macro AUROC **{chemistry_row.within_protein_macro_AUROC_descriptive:.3f}**.
- Distance plus all chemistry: AUROC **{full_row.AUROC_descriptive:.3f}**;
  within-protein macro AUROC **{full_row.within_protein_macro_AUROC_descriptive:.3f}**.
- Highest observation AUROC: `{sorted_auc.iloc[0].combination_id}` =
  **{sorted_auc.iloc[0].AUROC_descriptive:.3f}**.
- Highest within-protein macro AUROC: `{sorted_macro.iloc[0].combination_id}` =
  **{sorted_macro.iloc[0].within_protein_macro_AUROC_descriptive:.3f}**.

These are same-reference descriptive values. They are not used to select a final
subset; checkpoint 8 must repeat all 15 subsets with protein- and family-held-out
fitting.

## Ligand-identity limitation exposed by this reference

There are **{identity_audit['allosteric_unique_connectivity_key']:,}** distinct
allosteric ligand connectivities but only
**{identity_audit['orthosteric_unique_connectivity_key']:,}** orthosteric
connectivities. Only **{both_label_connectivities:,}/{identity_audit['unique_connectivity_key']:,}**
connectivities occur with both labels. An in-sample lookup of each connectivity's
label prevalence gives AUROC **{identity_oracle_auc:.3f}**. This is a data-structure
diagnostic, not a realizable held-out baseline, but it shows that chemistry-only
performance cannot be interpreted as site-specific discrimination.

## Scope boundary

Checkpoint 7 compares feature content only. It does not rank the unlabeled BioLiP
universe and does not choose the final ranking subset. Checkpoint 8 remains blocked
until this result is reviewed.
"""
    atomic_text(REPORT, report)

    output_paths = {
        "reference_features": FEATURE_TABLE,
        "component_models": COMPONENT_MODELS,
        "feature_ablation": ABLATION,
        "score_matrix": SCORES,
        "feature_distributions": DISTRIBUTIONS,
        "feature_correlations": CORRELATIONS,
        "ligand_identity_audit": IDENTITY_AUDIT,
        "input_hashes": INPUT_HASHES,
        "feature_spec": FEATURE_SPEC,
        "report": REPORT,
    }
    build = {
        "checkpoint": 7,
        "status": "complete_pending_independent_validation",
        "elapsed_seconds": round(time.time() - start, 3),
        "reference_rows": len(frame),
        "allosteric_rows": int(frame.binary_label.sum()),
        "orthosteric_rows": int((frame.binary_label == 0).sum()),
        "proteins": int(frame.uniprot.nunique()),
        "both_label_proteins": identity_audit["proteins_with_both_labels"],
        "site_ligand_signatures": int(frame.site_ligand_signature_id.nunique()),
        "features": [item[0] for item in FEATURES],
        "likelihood_ratio_feature_subsets": len(combinations),
        "sanity_baselines": 1,
        "ablation_rows": len(ablation),
        "selected_structural_distance": DISTANCE,
        "preferred_feature_subset": None,
        "ligand_identity_audit": identity_audit,
        "recursive_directory_scan_performed": False,
        "outputs": {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in output_paths.items()
        },
    }
    atomic_json(BUILD, build)
    print(json.dumps(build, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
