#!/usr/bin/env python3
"""Aggregate general and source-linked biochemical ChEMBL screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


GENERAL_BASE_MODELS = ("ligand", "protein", "c1", "c2")
GENERAL_MODELS = GENERAL_BASE_MODELS + ("d1", "d2", "d3")
BIOCHEMICAL_MODELS = GENERAL_BASE_MODELS + ("c3", "d1", "d2", "d3")
POCKET_MODELS = {"d1", "d2", "d3"}
GENERAL_TRAINING_ARMS = ("every_pair", "general")
ENRICHMENT_COLUMNS = [
    "deploy_arm",
    "model",
    "evaluation_universe",
    "novelty_stratum",
    "novelty_reference",
    "ranking",
    "top_fraction",
    "universe_rows",
    "positive_rows",
    "top_rows",
    "top_positive_rows",
    "background_prevalence",
    "top_positive_rate",
    "enrichment",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--candidate-count", type=int, default=5000)
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


def canonical_fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def validate_inference_run_contract(package, report):
    value = report.get("run_contract")
    if not isinstance(value, dict):
        raise RuntimeError("inference report lacks a run contract")
    if report.get("run_fingerprint") != canonical_fingerprint(value):
        raise RuntimeError("inference run fingerprint mismatch")
    source_path = package / "validation/CHEMBL_SOURCE_FINGERPRINT.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_payload = dict(source)
    stored_source_fingerprint = source_payload.pop("source_fingerprint", None)
    source_status = source_payload.pop("status", None)
    if (
        source_status != "validated"
        or stored_source_fingerprint != canonical_fingerprint(source_payload)
        or source_payload.get("cpu_contract_sha256")
        != sha256(package / "validation/CPU_CONTRACT.json")
        or value.get("source_fingerprint") != stored_source_fingerprint
        or value.get("inference_script_sha256")
        != sha256(package / "scripts/infer_chembl.py")
    ):
        raise RuntimeError("inference source contract mismatch")
    for record in value.get("checkpoint_records", []):
        path = (
            package
            / "gpu_output/deploy"
            / record["deploy_arm"]
            / record["model"]
            / "seed_{}".format(record["seed"])
            / "deploy.pt"
        )
        if not path.is_file() or sha256(path) != record["checkpoint_sha256"]:
            raise RuntimeError("deploy checkpoint changed: {}".format(path))


def validate_prediction_frame(frame, models_per_arm, source):
    if "selected_chain_available" not in frame:
        raise RuntimeError("{} predictions lack selected-chain status".format(source))
    available = frame["selected_chain_available"].astype(int).eq(1)
    pocket_available = frame["pocket_available"].astype(int).eq(1)
    for arm, models in models_per_arm.items():
        for model in models:
            for suffix in ["mean", "sd"]:
                column = "p_{}_{}_{}".format(arm, model, suffix)
                if column not in frame:
                    raise RuntimeError("{} predictions lack {}".format(source, column))
                values = pd.to_numeric(frame[column], errors="coerce")
                if model == "ligand":
                    expected = pd.Series(True, index=frame.index)
                elif model in POCKET_MODELS:
                    expected = pocket_available
                else:
                    expected = available
                if not values.loc[expected].notna().all() or not values.loc[~expected].isna().all():
                    raise RuntimeError("{} availability contract failed for {}".format(source, column))
                finite = values.loc[expected].to_numpy(dtype=float)
                if not np.isfinite(finite).all():
                    raise RuntimeError("{} has nonfinite {}".format(source, column))
                if suffix == "mean" and not ((finite >= 0) & (finite <= 1)).all():
                    raise RuntimeError("{} has invalid probabilities in {}".format(source, column))
                if suffix == "sd" and not (finite >= 0).all():
                    raise RuntimeError("{} has negative ensemble SD in {}".format(source, column))


def binary_metrics(frame, label, score):
    frame = frame[frame[score].notna()].copy()
    y = pd.to_numeric(frame[label], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(frame[score], errors="coerce").astype(float).to_numpy()
    if len(np.unique(y)) != 2:
        return {
            "n": int(len(y)),
            "positives": int(y.sum()),
            "prevalence": float(y.mean()) if len(y) else None,
            "auprc": None,
            "auroc": None,
        }
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }


def reference_analysis(package, output):
    path = package / "gpu_output/chembl/reference_predictions.tsv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    report_path = package / "gpu_output/chembl/REFERENCE_INFERENCE.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_inference_run_contract(package, report)
    if (
        report.get("status") != "validated"
        or report.get("contract_id") != "role_complete_pair_matrix_v2"
        or report.get("model_version") != "role_complete_matrix_v2"
        or report.get("output_sha256") != sha256(path)
        or report.get("models_per_arm")
        != {arm: list(GENERAL_MODELS) for arm in GENERAL_TRAINING_ARMS}
    ):
        raise RuntimeError("reference inference contract is invalid")
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    validate_prediction_frame(
        frame,
        {arm: list(GENERAL_MODELS) for arm in GENERAL_TRAINING_ARMS},
        "reference",
    )
    labeled = frame[frame["weak2020_label"].isin([0, 1])].copy()
    rows = []
    for arm in GENERAL_TRAINING_ARMS:
        for model in GENERAL_MODELS:
            score = "p_{}_{}_mean".format(arm, model)
            universes = [("model_available", labeled[labeled[score].notna()])]
            availability_column = (
                "pocket_available" if model in POCKET_MODELS else "selected_chain_available"
            )
            universes.append(
                (
                    availability_column + "_matched",
                    labeled[labeled[availability_column].eq(1)],
                )
            )
            for universe, subset in universes:
                # Both deployment arms are evaluated on identical strata
                # defined relative to the prespecified primary
                # protein-anchored training reference.  Arm-specific Pfam
                # strata are not merged as if they contained the same rows.
                family_column = "pfam_family_status_general"
                family_statuses = ["seen", "unseen", "annotation_unavailable"]
                strata = [
                    ("all", "all", subset),
                    (
                        "exact_target",
                        "exact_target_unseen",
                        subset[subset["protein_seen_general"].eq(0)],
                    ),
                    (
                        "ligand_connectivity",
                        "ligand_connectivity_unseen",
                        subset[subset["ligand_seen_general"].eq(0)],
                    ),
                    (
                        "exact_target_and_ligand",
                        "exact_target_and_ligand_double_novel",
                        subset[
                            subset["protein_seen_general"].eq(0)
                            & subset["ligand_seen_general"].eq(0)
                        ],
                    ),
                ] + [
                    (
                        "pfam",
                        "pfam_" + status,
                        subset[subset[family_column].eq(status)],
                    )
                    for status in family_statuses
                ]
                for novelty_axis, stratum, stratum_frame in strata:
                    value = binary_metrics(stratum_frame, "weak2020_label", score)
                    value.update(
                        {
                            "deploy_arm": arm,
                            "model": model,
                            "endpoint": "Burggraaff_2020",
                            "evaluation_universe": universe,
                            "novelty_axis": novelty_axis,
                            "novelty_stratum": stratum,
                            "family_stratum": stratum,
                            "family_reference_arm": "protein_anchored",
                            "domain_interpretation": (
                                "every_pair_training_sensitivity"
                                if arm == "every_pair"
                                else "primary_protein_anchored"
                            ),
                        }
                    )
                    target_values = []
                    for _, group in stratum_frame.groupby("uniprot"):
                        group = group[group[score].notna()]
                        if group["weak2020_label"].nunique() == 2:
                            target_values.append(binary_metrics(group, "weak2020_label", score))
                    value["target_macro_auprc"] = (
                        float(np.mean([x["auprc"] for x in target_values if x["auprc"] is not None]))
                        if target_values and any(x["auprc"] is not None for x in target_values)
                        else None
                    )
                    value["target_macro_auroc"] = (
                        float(np.mean([x["auroc"] for x in target_values if x["auroc"] is not None]))
                        if target_values and any(x["auroc"] is not None for x in target_values)
                        else None
                    )
                    value["targets_with_both_labels"] = int(len(target_values))
                    rows.append(value)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "REFERENCE_METRICS.tsv", sep="\t", index=False)
    keys = [
        "model",
        "endpoint",
        "evaluation_universe",
        "novelty_axis",
        "novelty_stratum",
        "family_stratum",
        "family_reference_arm",
    ]
    every = metrics[metrics["deploy_arm"].eq("every_pair")].copy()
    anchored = metrics[metrics["deploy_arm"].eq("general")].copy()
    paired = every.merge(
        anchored,
        on=keys,
        how="inner",
        suffixes=("_every_pair", "_protein_anchored"),
        validate="one_to_one",
    )
    if not (
        paired["n_every_pair"].eq(paired["n_protein_anchored"]).all()
        and paired["positives_every_pair"].eq(
            paired["positives_protein_anchored"]
        ).all()
    ):
        raise RuntimeError("reference cohort delta is not evaluated on identical rows")
    for metric in ["auprc", "auroc", "target_macro_auprc", "target_macro_auroc"]:
        paired["delta_every_pair_minus_protein_anchored_" + metric] = (
            paired[metric + "_every_pair"] - paired[metric + "_protein_anchored"]
        )
    paired.to_csv(
        output / "REFERENCE_TRAINING_COHORT_DELTA.tsv", sep="\t", index=False
    )
    return frame, rows


def enrichment_rows(
    frame,
    arm,
    models,
    label="has_allosteric_text",
    evaluation_universe="model_available",
    novelty_stratum="all",
    novelty_reference="none",
):
    fractions = [0.001, 0.005, 0.01, 0.05]
    rows = []
    for model in models:
        score = "p_{}_{}_mean".format(arm, model)
        ranked = frame[frame[score].notna()].sort_values(score, ascending=False).reset_index(drop=True)
        prevalence = float(ranked[label].mean()) if len(ranked) else 0.0
        for fraction in fractions:
            count = max(1, int(np.ceil(len(ranked) * fraction)))
            top = ranked.iloc[:count]
            rate = float(top[label].mean()) if len(top) else 0.0
            rows.append(
                {
                    "deploy_arm": arm,
                    "model": model,
                    "evaluation_universe": evaluation_universe,
                    "novelty_stratum": novelty_stratum,
                    "novelty_reference": novelty_reference,
                    "ranking": "pooled",
                    "top_fraction": fraction,
                    "universe_rows": int(len(ranked)),
                    "positive_rows": int(ranked[label].sum()),
                    "top_rows": int(len(top)),
                    "top_positive_rows": int(top[label].sum()),
                    "background_prevalence": prevalence,
                    "top_positive_rate": rate,
                    "enrichment": rate / prevalence if prevalence else None,
                }
            )
        targets = set(frame.loc[frame[label].eq(1), "uniprot"].astype(str))
        within = frame[frame["uniprot"].astype(str).isin(targets) & frame[score].notna()].copy()
        prevalence = float(within[label].mean()) if len(within) else 0.0
        for fraction in fractions:
            selected = []
            for _, group in within.groupby("uniprot", sort=False):
                count = max(1, int(np.ceil(len(group) * fraction)))
                selected.append(
                    group.sort_values(score, ascending=False, kind="mergesort").iloc[
                        :count
                    ]
                )
            top = (
                pd.concat(selected, ignore_index=True)
                if selected
                else within.iloc[0:0].copy()
            )
            rate = float(top[label].mean()) if len(top) else 0.0
            rows.append(
                {
                    "deploy_arm": arm,
                    "model": model,
                    "evaluation_universe": evaluation_universe,
                    "novelty_stratum": novelty_stratum,
                    "novelty_reference": novelty_reference,
                    "ranking": "within_text_positive_targets",
                    "top_fraction": fraction,
                    "universe_rows": int(len(within)),
                    "positive_rows": int(within[label].sum()),
                    "top_rows": int(len(top)),
                    "top_positive_rows": int(top[label].sum()),
                    "background_prevalence": prevalence,
                    "top_positive_rate": rate,
                    "enrichment": rate / prevalence if prevalence else None,
                }
            )
    return rows


def load_shards(package, scope):
    expected_models = (
        {
            "every_pair": list(BIOCHEMICAL_MODELS),
            "general": list(BIOCHEMICAL_MODELS),
            "role_complete": list(BIOCHEMICAL_MODELS),
        }
        if scope == "biochemical"
        else {arm: list(GENERAL_MODELS) for arm in GENERAL_TRAINING_ARMS}
    )
    paths = [
        package / "gpu_output/chembl" / scope / "{}_predictions_shard{:02d}of04.tsv.gz".format(scope, index)
        for index in range(4)
    ]
    frames = []
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        report_path = path.parent / "{}_SHARD{:02d}.json".format(scope.upper(), index)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_inference_run_contract(package, report)
        if (
            report.get("status") != "validated"
            or report.get("contract_id") != "role_complete_pair_matrix_v2"
            or report.get("model_version") != "role_complete_matrix_v2"
            or report.get("output_sha256") != sha256(path)
            or report.get("models_per_arm") != expected_models
        ):
            raise RuntimeError("invalid {} shard {} contract".format(scope, index))
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        validate_prediction_frame(frame, expected_models, "{} shard {}".format(scope, index))
        if int(report.get("rows", -1)) != len(frame):
            raise RuntimeError("{} shard {} row mismatch".format(scope, index))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if "OOD_Row_ID" in combined and combined["OOD_Row_ID"].astype(str).duplicated().any():
        raise RuntimeError("{} shards contain duplicate OOD row IDs".format(scope))
    return combined


def full_analysis(package, output, candidate_count):
    frame = load_shards(package, "full")
    frame = frame[~frame["legacy_ood_label"].eq(0)].copy()
    enrichment = []
    common_strata = [
        ("all", frame, "none"),
        (
            "exact_target_unseen",
            frame[frame["protein_seen_protein_anchored"].eq(0)],
            "protein_anchored_training",
        ),
        (
            "ligand_connectivity_unseen",
            frame[frame["ligand_seen_protein_anchored"].eq(0)],
            "protein_anchored_training",
        ),
        (
            "exact_target_and_ligand_double_novel",
            frame[
                frame["protein_seen_protein_anchored"].eq(0)
                & frame["ligand_seen_protein_anchored"].eq(0)
            ],
            "protein_anchored_training",
        ),
        (
            "pfam_seen",
            frame[frame["pfam_family_status_general"].eq("seen")],
            "protein_anchored_training",
        ),
        (
            "pfam_unseen",
            frame[frame["pfam_family_status_general"].eq("unseen")],
            "protein_anchored_training",
        ),
        (
            "pfam_annotation_unavailable",
            frame[
                frame["pfam_family_status_general"].eq("annotation_unavailable")
            ],
            "protein_anchored_training",
        ),
    ]
    for arm in GENERAL_TRAINING_ARMS:
        for stratum, subset, reference in common_strata:
            enrichment.extend(
                enrichment_rows(
                    subset,
                    arm,
                    GENERAL_BASE_MODELS,
                    evaluation_universe="full_model_available",
                    novelty_stratum=stratum,
                    novelty_reference=reference,
                )
            )
            enrichment.extend(
                enrichment_rows(
                    subset[subset["pocket_available"].eq(1)],
                    arm,
                    GENERAL_MODELS,
                    evaluation_universe="pocket_available_matched",
                    novelty_stratum=stratum,
                    novelty_reference=reference,
                )
            )
    enrichment_frame = pd.DataFrame(enrichment)
    enrichment_frame.to_csv(output / "FULL_TEXT_ENRICHMENT.tsv", sep="\t", index=False)

    # Arm-relative strata answer a different question and are therefore kept
    # out of the direct training-cohort delta table.  In particular, a Pfam
    # nonmatch for every-pair training cannot establish family novelty because
    # that training Pfam reference is incomplete.
    arm_relative = []
    for arm in GENERAL_TRAINING_ARMS:
        if arm == "every_pair":
            protein_seen = "protein_seen_every_pair"
            ligand_seen = "ligand_seen_every_pair"
            family_column = "pfam_family_status_every_pair"
            family_statuses = [
                "seen",
                "training_reference_incomplete",
                "annotation_unavailable",
            ]
            reference = "every_pair_training"
        else:
            protein_seen = "protein_seen_protein_anchored"
            ligand_seen = "ligand_seen_protein_anchored"
            family_column = "pfam_family_status_general"
            family_statuses = ["seen", "unseen", "annotation_unavailable"]
            reference = "protein_anchored_training"
        strata = [
            ("exact_target_unseen", frame[frame[protein_seen].eq(0)]),
            ("ligand_connectivity_unseen", frame[frame[ligand_seen].eq(0)]),
            (
                "exact_target_and_ligand_double_novel",
                frame[frame[protein_seen].eq(0) & frame[ligand_seen].eq(0)],
            ),
        ] + [
            ("pfam_" + status, frame[frame[family_column].eq(status)])
            for status in family_statuses
        ]
        for stratum, subset in strata:
            arm_relative.extend(
                enrichment_rows(
                    subset,
                    arm,
                    GENERAL_BASE_MODELS,
                    evaluation_universe="full_model_available",
                    novelty_stratum=stratum,
                    novelty_reference=reference,
                )
            )
            arm_relative.extend(
                enrichment_rows(
                    subset[subset["pocket_available"].eq(1)],
                    arm,
                    GENERAL_MODELS,
                    evaluation_universe="pocket_available_matched",
                    novelty_stratum=stratum,
                    novelty_reference=reference,
                )
            )
    pd.DataFrame(arm_relative).to_csv(
        output / "FULL_ARM_RELATIVE_NOVELTY_ENRICHMENT.tsv",
        sep="\t",
        index=False,
    )
    keys = [
        "model",
        "evaluation_universe",
        "ranking",
        "top_fraction",
        "novelty_stratum",
        "novelty_reference",
    ]
    every = enrichment_frame[
        enrichment_frame["deploy_arm"].eq("every_pair")
    ].copy()
    anchored = enrichment_frame[
        enrichment_frame["deploy_arm"].eq("general")
    ].copy()
    paired = every.merge(
        anchored,
        on=keys,
        how="inner",
        suffixes=("_every_pair", "_protein_anchored"),
        validate="one_to_one",
    )
    if not (
        paired["universe_rows_every_pair"].eq(
            paired["universe_rows_protein_anchored"]
        ).all()
        and paired["positive_rows_every_pair"].eq(
            paired["positive_rows_protein_anchored"]
        ).all()
        and paired["top_rows_every_pair"].eq(
            paired["top_rows_protein_anchored"]
        ).all()
    ):
        raise RuntimeError("full-screen cohort delta is not row matched")
    for metric in ["top_positive_rate", "enrichment"]:
        paired["delta_every_pair_minus_protein_anchored_" + metric] = (
            paired[metric + "_every_pair"] - paired[metric + "_protein_anchored"]
        )
    paired.to_csv(
        output / "FULL_TRAINING_COHORT_ENRICHMENT_DELTA.tsv",
        sep="\t",
        index=False,
    )
    unlabeled = frame[
        frame["has_allosteric_text"].eq(0) & frame["has_orthosteric_text"].eq(0)
    ].copy()
    for arm, label in [
        ("every_pair", "EVERY_PAIR"),
        ("general", "PROTEIN_ANCHORED"),
    ]:
        score = "p_{}_c2_mean".format(arm)
        candidates = unlabeled[unlabeled[score].notna()].sort_values(
            score, ascending=False, kind="mergesort"
        ).head(candidate_count)
        candidates.to_csv(
            output / "FULL_TOP_UNLABELED_CANDIDATES_{}.tsv.gz".format(label),
            sep="\t",
            index=False,
            compression="gzip",
        )
        protein_seen = (
            "protein_seen_every_pair"
            if arm == "every_pair"
            else "protein_seen_protein_anchored"
        )
        ligand_seen = (
            "ligand_seen_every_pair"
            if arm == "every_pair"
            else "ligand_seen_protein_anchored"
        )
        double_novel = unlabeled[
            unlabeled[protein_seen].eq(0)
            & unlabeled[ligand_seen].eq(0)
            & unlabeled[score].notna()
        ].sort_values(score, ascending=False, kind="mergesort").head(candidate_count)
        double_novel.to_csv(
            output
            / "FULL_TOP_UNLABELED_DOUBLE_NOVEL_CANDIDATES_{}.tsv.gz".format(
                label
            ),
            sep="\t",
            index=False,
            compression="gzip",
        )
    comparison_columns = [
        "stable_pair_key",
        "target_chembl_id",
        "ligand_chembl_id",
        "uniprot",
        "connectivity_key",
        "has_allosteric_text",
        "has_orthosteric_text",
    ]
    for model in GENERAL_MODELS:
        comparison_columns.extend(
            ["p_every_pair_{}_mean".format(model), "p_general_{}_mean".format(model)]
        )
    comparison = frame[comparison_columns].copy()
    for model in GENERAL_MODELS:
        comparison["delta_every_pair_minus_protein_anchored_{}".format(model)] = (
            comparison["p_every_pair_{}_mean".format(model)]
            - comparison["p_general_{}_mean".format(model)]
        )
    comparison.to_csv(
        output / "FULL_TRAINING_COHORT_SCORE_COMPARISON.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    return frame, enrichment


def biochemical_analysis(package, output):
    frame = load_shards(package, "biochemical")
    frame = frame[~frame["legacy_ood_label"].eq(0)].copy()
    # High-confidence biochemical keys are sparse in ChEMBL and do not supply a
    # validated negative class. Enrichment is only emitted if text positives exist.
    enrichment = []
    if frame["has_allosteric_text"].sum() > 0:
        for arm in ["every_pair", "general", "role_complete"]:
            enrichment.extend(
                enrichment_rows(
                    frame,
                    arm,
                    GENERAL_BASE_MODELS + ("c3",),
                    evaluation_universe="biochemical_model_available",
                )
            )
            enrichment.extend(
                enrichment_rows(
                    frame[frame["pocket_available"].eq(1)],
                    arm,
                    BIOCHEMICAL_MODELS,
                    evaluation_universe="biochemical_pocket_available_matched",
                )
            )
    enrichment_frame = pd.DataFrame(enrichment, columns=ENRICHMENT_COLUMNS)
    enrichment_frame.to_csv(
        output / "BIOCHEMICAL_TEXT_ENRICHMENT.tsv", sep="\t", index=False
    )

    double_exact = frame[
        frame["protein_seen_role_complete"].eq(0)
        & frame["ligand_seen_role_complete"].eq(0)
        & frame["has_allosteric_text"].eq(0)
        & frame["has_orthosteric_text"].eq(0)
    ].copy()
    for model in ["c2", "c3", "d1", "d2", "d3"]:
        score = "p_role_complete_{}_mean".format(model)
        selected = double_exact[double_exact[score].notna()]
        selected.sort_values(score, ascending=False).head(250).to_csv(
            output / "BIOCHEMICAL_EXACT_DOUBLE_UNSEEN_TOP_{}.tsv.gz".format(model.upper()),
            sep="\t",
            index=False,
            compression="gzip",
        )
    correlations = []
    for model in BIOCHEMICAL_MODELS:
        for left, right in [
            ("every_pair", "general"),
            ("general", "role_complete"),
            ("every_pair", "role_complete"),
        ]:
            left_column = "p_{}_{}_mean".format(left, model)
            right_column = "p_{}_{}_mean".format(right, model)
            available = frame[[left_column, right_column]].dropna()
            correlations.append(
                {
                    "model": model,
                    "left_training_arm": left,
                    "right_training_arm": right,
                    "n": int(len(available)),
                    "spearman": float(available.corr(method="spearman").iloc[0, 1])
                    if len(available) > 1
                    else None,
                }
            )
    pd.DataFrame(correlations).to_csv(output / "BIOCHEMICAL_MODEL_RANK_CORRELATION.tsv", sep="\t", index=False)
    summary = {
        "rows": int(len(frame)),
        "targets": int(frame["uniprot"].nunique()),
        "ligands": int(frame["connectivity_key"].nunique()),
        "allosteric_text_rows": int(frame["has_allosteric_text"].sum()),
        "orthosteric_text_rows": int(frame["has_orthosteric_text"].sum()),
        "text_enrichment_status": (
            "evaluated"
            if int(frame["has_allosteric_text"].sum()) > 0
            else "not_evaluated_no_positive_class"
        ),
        "text_enrichment_rows_emitted": int(len(enrichment_frame)),
        "exact_protein_and_ligand_unseen_rows": int(len(double_exact)),
        "pfam_family_unseen_general_rows": int(
            frame["pfam_family_status_general"].eq("unseen").sum()
        ),
        "pfam_family_seen_every_pair_rows": int(
            frame["pfam_family_status_every_pair"].eq("seen").sum()
        ),
        "pfam_training_reference_incomplete_every_pair_rows": int(
            frame["pfam_family_status_every_pair"].eq(
                "training_reference_incomplete"
            ).sum()
        ),
        "pfam_family_unseen_role_complete_rows": int(
            frame["pfam_family_status_role_complete"].eq("unseen").sum()
        ),
        "pfam_annotation_unavailable_rows": int(
            frame["pfam_family_status_general"].eq("annotation_unavailable").sum()
        ),
        "independent_metabolite_catalog": False,
        "claim_limit": (
            "this is an orthosteric-source-linked biochemical candidate subset, "
            "not an independent metabolite validation set"
        ),
    }
    atomic_json(output / "BIOCHEMICAL_SCREEN_SUMMARY.json", summary)
    return frame, summary


def main():
    args = parse_args()
    root = args.project_root.resolve()
    package = root / "analysis/role_complete_pair_matrix"
    output = package / "gpu_output/chembl/aggregate"
    output.mkdir(parents=True, exist_ok=True)
    reference, reference_metrics = reference_analysis(package, output)
    full, full_enrichment = full_analysis(package, output, args.candidate_count)
    biochemical, biochemical_summary = biochemical_analysis(package, output)
    reference_proteins = int(reference["uniprot"].astype(str).nunique())
    reference_selected_proteins = int(
        reference.loc[
            reference["selected_chain_available"].eq(1), "uniprot"
        ].astype(str).nunique()
    )
    full_proteins = int(full["uniprot"].astype(str).nunique())
    full_selected_proteins = int(
        full.loc[full["selected_chain_available"].eq(1), "uniprot"]
        .astype(str)
        .nunique()
    )
    report = {
        "status": "validated",
        "contract_id": "role_complete_pair_matrix_v2",
        "model_version": "role_complete_matrix_v2",
        "reference_rows": int(len(reference)),
        "reference_proteins": reference_proteins,
        "reference_selected_chain_available_proteins": reference_selected_proteins,
        "reference_selected_chain_protein_coverage": float(
            reference_selected_proteins / reference_proteins
        )
        if reference_proteins
        else None,
        "full_rows": int(len(full)),
        "full_proteins": full_proteins,
        "full_selected_chain_available_proteins": full_selected_proteins,
        "full_selected_chain_protein_coverage": float(
            full_selected_proteins / full_proteins
        )
        if full_proteins
        else None,
        "biochemical_rows": int(len(biochemical)),
        "biochemical_summary": biochemical_summary,
        "general_training_arms": list(GENERAL_TRAINING_ARMS),
        "general_full_models": list(GENERAL_MODELS),
        "biochemical_models": list(BIOCHEMICAL_MODELS),
        "pocket_models_scored_externally": True,
        "non_ligand_external_input": (
            "the same validated selected target-chain tensor used by internal training"
        ),
        "non_ligand_model_scope": "selected-chain-available rows; unavailable rows are NA",
        "pocket_model_scope": "pocket residues extracted from the same selected-chain tensor",
        "full_c3_scored": False,
        "full_c3_reason": (
            "whole-chain bidirectional atom-residue attention has quadratic cross-context "
            "memory/time cost and is reserved for the much smaller biochemical screen"
        ),
        "weak_text_treated_as_ground_truth": False,
        "exact_target_ligand_novelty_stratified": True,
        "reference_exact_target_ligand_novelty_stratified": True,
        "double_novel_candidate_tables_emitted": True,
        "family_novelty_stratified": True,
        "family_novelty_scope": (
            "reference and full-screen strata are frozen relative to the primary "
            "protein-anchored Pfam training union so both deployment arms use identical rows"
        ),
        "arm_relative_novelty_reported_separately": True,
        "full_screen_pfam_annotation_available_rows": int(
            full["pfam_family_status_general"].ne("annotation_unavailable").sum()
        ),
        "full_screen_pfam_annotation_unavailable_rows": int(
            full["pfam_family_status_general"].eq("annotation_unavailable").sum()
        ),
        "every_pair_family_unseen_claim_allowed": False,
        "every_pair_family_unseen_reason": (
            "frozen Pfam cache covers 575 of 940 every-pair training proteins"
        ),
        "training_cohort_comparison_on_identical_external_rows": True,
        "legacy_mixed_tensor_external_metrics_directly_comparable": False,
        "legacy_comparison_reason": (
            "the selected-chain pipeline changes both the protein representation "
            "and, especially for the reference set, the evaluated row/protein universe"
        ),
    }
    atomic_json(output / "CHEMBL_VALIDATION.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
