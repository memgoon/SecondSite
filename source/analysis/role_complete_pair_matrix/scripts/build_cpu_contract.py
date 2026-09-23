#!/usr/bin/env python3
"""Build the frozen cohort, evaluation, and external-resource contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
    "every_pair": {"rows": 6854, "proteins": 940},
    "protein_anchored": {"rows": 4637, "proteins": 426},
    "protein_ligand_role_complete": {
        "rows": 395,
        "proteins": 98,
        "full_ligands": 40,
        "connectivity_ligands": 40,
        "families": 47,
        "allosteric": 143,
        "orthosteric": 252,
    },
}

# All three cohorts retain the same three fitted split regimes.  For the two
# general cohorts, strict family+ligand novelty remains an additional endpoint
# derived from family-held-out fits; it does not replace ligand-held-out CV.
ACTIVE_REGIMES = {
    "every_pair": ("row_random", "unseen_family", "unseen_ligand"),
    "protein_anchored": ("row_random", "unseen_family", "unseen_ligand"),
    "protein_ligand_role_complete": (
        "row_random",
        "unseen_family",
        "unseen_ligand",
    ),
}
CHECKPOINT_MODELS = ("ligand", "protein", "c1", "c2", "c3", "d1", "d2", "d3")
CHECKPOINT_MIN_GROUPS = 8
MATRIX_FOLD_COLUMNS = {
    "row_random": "matrix_row_fold",
    "unseen_family": "matrix_family_fold",
    "unseen_ligand": "matrix_ligand_fold",
}

EXTERNAL_RESOURCES = {
    "chembl_reference_model_ready": (
        "analysis/chembl_external_prioritization/gpu_cache/"
        "REFERENCE_MODEL_READY.tsv.gz"
    ),
    "pfam_cache": "analysis/property_balanced_chembl/data/UNIPROT_PFAM_CACHE.json",
    "pocket_target_chains": (
        "analysis/property_balanced_chembl/data/chembl_pocket_extension/"
        "TARGET_CHAIN_SEQUENCES.tsv"
    ),
    "pocket_indices": (
        "analysis/property_balanced_chembl/data/chembl_pocket_extension/POCKET_INDICES.json"
    ),
    "pocket_alignment_audit": (
        "analysis/property_balanced_chembl/data/chembl_pocket_extension/"
        "POCKET_ALIGNMENT_AUDIT.tsv"
    ),
    "pocket_coverage": (
        "analysis/property_balanced_chembl/data/chembl_pocket_extension/POCKET_COVERAGE.json"
    ),
    "pocket_embedding_validation": (
        "analysis/property_balanced_chembl/gpu_output/pocket_extension/"
        "C3_TARGET_CHAIN_EMBEDDING_VALIDATION.json"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery")
    )
    return parser.parse_args()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role_complete_fixed_point(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Retain rows whose exact protein and exact ligand each carry both roles."""
    current = frame.copy()
    iterations = 0
    while True:
        before = len(current)
        proteins = current.groupby("uniprot")["binary_label"].nunique()
        ligands = current.groupby("full_inchikey")["binary_label"].nunique()
        current = current[
            current["uniprot"].isin(proteins[proteins.eq(2)].index)
            & current["full_inchikey"].isin(ligands[ligands.eq(2)].index)
        ].copy()
        iterations += 1
        if len(current) == before:
            return current, iterations


def checkpoint_fallback_contract(arms):
    """Freeze support-only fallback counts before any model outcome exists."""
    nonzero_cells = []
    expected_total = 0
    expected_by_arm = {arm: 0 for arm in arms}
    for arm, frame in arms.items():
        for regime, fold_column in MATRIX_FOLD_COLUMNS.items():
            fallback_folds_by_model = {model: [] for model in CHECKPOINT_MODELS}
            for outer_fold in range(5):
                validation_fold = (outer_fold + 1) % 5
                validation = frame[
                    frame[fold_column].eq(validation_fold)
                    & frame["matrix_evaluation_eligible"].eq(1)
                ]
                group_counts = {
                    "protein": int(
                        sum(
                            group["binary_label"].nunique() == 2
                            for _, group in validation.groupby("uniprot")
                        )
                    ),
                    "ligand": int(
                        sum(
                            group["binary_label"].nunique() == 2
                            for _, group in validation.groupby("connectivity_key")
                        )
                    ),
                }
                for model in CHECKPOINT_MODELS:
                    if model == "ligand":
                        requested = ["protein"]
                    elif model == "protein":
                        requested = ["ligand"]
                    elif regime == "unseen_family":
                        requested = ["ligand"]
                    elif regime == "unseen_ligand":
                        requested = ["protein"]
                    else:
                        requested = ["ligand", "protein"]
                    if any(
                        group_counts[group] < CHECKPOINT_MIN_GROUPS
                        for group in requested
                    ):
                        fallback_folds_by_model[model].append(int(outer_fold))
            for model, outer_folds in fallback_folds_by_model.items():
                if not outer_folds:
                    continue
                expected_fits = int(len(outer_folds) * 3)
                expected_total += expected_fits
                expected_by_arm[arm] += expected_fits
                nonzero_cells.append(
                    {
                        "cohort_arm": arm,
                        "regime": regime,
                        "model": model,
                        "outer_folds": outer_folds,
                        "expected_fallback_fits": expected_fits,
                    }
                )
    if expected_total != 132 or expected_by_arm != {
        "every_pair": 0,
        "protein_anchored": 0,
        "protein_ligand_role_complete": 132,
    }:
        raise RuntimeError(
            "checkpoint fallback support contract changed: {} {}".format(
                expected_total, expected_by_arm
            )
        )
    return {
        "model_aware": True,
        "minimum_valid_groups": CHECKPOINT_MIN_GROUPS,
        "fallback_metric": "pooled symmetric AP",
        "fallback_determined_from_label_support_not_model_performance": True,
        "expected_fallback_fits": int(expected_total),
        "expected_fallback_fraction_all_fits": float(expected_total / 1080.0),
        "expected_fallback_fits_by_arm": expected_by_arm,
        "expected_role_complete_fallback_fraction": float(expected_total / 360.0),
        "nonzero_cells": nonzero_cells,
        "interpretation": (
            "fallback can reintroduce between-identity prevalence into epoch "
            "selection; evaluation remains conditional and fallback incidence "
            "must be reported"
        ),
    }


def check_subset(core: pd.DataFrame, broad: pd.DataFrame):
    compare = [
        "main_row_id",
        "uniprot",
        "full_inchikey",
        "connectivity_key",
        "binary_label",
        "family_component_id",
        "family_fold",
        "row_fold",
    ]
    left = core[compare].astype(str).sort_values("main_row_id").reset_index(drop=True)
    right = (
        broad[broad["main_row_id"].astype(str).isin(set(left["main_row_id"]))][compare]
        .astype(str)
        .sort_values("main_row_id")
        .reset_index(drop=True)
    )
    if len(right) != len(left) or not left.equals(right):
        raise RuntimeError("protein-anchored rows are not an exact metadata-preserving subset")


def group_statistics(broad: pd.DataFrame, core: pd.DataFrame, strict: pd.DataFrame):
    arms = {"broad": broad, "core": core, "strict": strict}
    keys = sorted(set(broad["connectivity_key"].astype(str)))
    rows = []
    for key in keys:
        row = {"connectivity_key": key}
        for name, frame in arms.items():
            part = frame[frame["connectivity_key"].astype(str).eq(key)]
            row[name + "_n"] = int(len(part))
            row[name + "_pos"] = int(part["binary_label"].eq(1).sum())
            row[name + "_neg"] = int(part["binary_label"].eq(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def assign_ligand_folds(stats: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Deterministic multi-cohort load balancing, strict groups assigned first."""
    stats = stats.copy()
    stats["strict_present"] = stats["strict_n"].gt(0).astype(int)
    stats = stats.sort_values(
        ["strict_present", "strict_n", "core_n", "broad_n", "connectivity_key"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    loads = {
        arm: np.zeros((n_folds, 3), dtype=float) for arm in ["strict", "core", "broad"]
    }
    targets = {}
    for arm in loads:
        targets[arm] = np.asarray(
            [
                max(float(stats[arm + "_n"].sum()) / n_folds, 1.0),
                max(float(stats[arm + "_pos"].sum()) / n_folds, 1.0),
                max(float(stats[arm + "_neg"].sum()) / n_folds, 1.0),
            ]
        )
    group_load = np.zeros(n_folds, dtype=float)
    assignments = []
    for index, row in stats.iterrows():
        values = {
            arm: np.asarray(
                [row[arm + "_n"], row[arm + "_pos"], row[arm + "_neg"]], dtype=float
            )
            for arm in loads
        }
        scores = []
        for fold in range(n_folds):
            score = 0.0
            # Strict balance is primary; core and broad prevent pathological outer folds.
            for arm, importance in [("strict", 8.0), ("core", 2.0), ("broad", 1.0)]:
                if values[arm][0] <= 0:
                    continue
                proposed = (loads[arm][fold] + values[arm]) / targets[arm]
                score += importance * float(np.square(proposed).sum())
            score += 0.02 * float((group_load[fold] + 1.0) ** 2)
            scores.append((score, float(loads["strict"][fold, 0]), group_load[fold], fold))
        fold = min(scores)[-1]
        for arm in loads:
            loads[arm][fold] += values[arm]
        group_load[fold] += 1
        assignments.append(
            {
                "connectivity_key": str(row["connectivity_key"]),
                "ligand_fold": int(fold),
                "strict_role_complete_ligand": int(row["strict_present"]),
            }
        )
    return pd.DataFrame(assignments).sort_values("connectivity_key").reset_index(drop=True)


def add_contract_columns(
    frame: pd.DataFrame,
    arm: str,
    fold_map: dict[str, int],
    evaluation_ids: set[str],
):
    value = frame.copy()
    value["cohort_arm"] = arm
    value["matrix_row_fold"] = pd.to_numeric(value["row_fold"], errors="raise").astype(int)
    value["matrix_family_fold"] = pd.to_numeric(value["family_fold"], errors="raise").astype(int)
    value["matrix_ligand_fold"] = value["connectivity_key"].astype(str).map(fold_map)
    if value["matrix_ligand_fold"].isna().any():
        raise RuntimeError("one or more ligand groups lack a fold")
    value["matrix_ligand_fold"] = value["matrix_ligand_fold"].astype(int)
    value["matrix_evaluation_eligible"] = (
        value["main_row_id"].astype(str).isin(evaluation_ids).astype(int)
    )
    return value


def fold_audit(arms: dict[str, pd.DataFrame]):
    rows = []
    for arm, frame in arms.items():
        definitions = dict([
            ("row_random", "matrix_row_fold"),
            ("unseen_family", "matrix_family_fold"),
            ("unseen_ligand", "matrix_ligand_fold"),
        ])
        for regime in ACTIVE_REGIMES[arm]:
            column = definitions[regime]
            for fold in range(5):
                # The broad arm is training-only augmentation outside the
                # protein-anchored evaluation universe.  Its extra rows can
                # affect training but are never silently called OOF tests.
                test = frame[
                    frame[column].eq(fold)
                    & frame["matrix_evaluation_eligible"].eq(1)
                ]
                development = frame[~frame[column].eq(fold)]
                row = {
                    "cohort_arm": arm,
                    "regime": regime,
                    "outer_fold": fold,
                    "test_rows": int(len(test)),
                    "test_allosteric": int(test["binary_label"].eq(1).sum()),
                    "test_orthosteric": int(test["binary_label"].eq(0).sum()),
                    "test_proteins": int(test["uniprot"].nunique()),
                    "test_ligands": int(test["connectivity_key"].nunique()),
                    "test_families": int(test["family_component_id"].nunique()),
                    "development_protein_overlap": int(
                        len(set(test["uniprot"].astype(str)) & set(development["uniprot"].astype(str)))
                    ),
                    "development_family_overlap": int(
                        len(
                            set(test["family_component_id"].astype(str))
                            & set(development["family_component_id"].astype(str))
                        )
                    ),
                    "development_ligand_overlap": int(
                        len(
                            set(test["connectivity_key"].astype(str))
                            & set(development["connectivity_key"].astype(str))
                        )
                    ),
                }
                if not len(test) or test["binary_label"].nunique() != 2:
                    raise RuntimeError("{} {} fold {} lost a class".format(arm, regime, fold))
                if regime == "unseen_family" and row["development_family_overlap"]:
                    raise RuntimeError("family leakage in {} fold {}".format(arm, fold))
                if regime == "unseen_ligand" and row["development_ligand_overlap"]:
                    raise RuntimeError("ligand leakage in {} fold {}".format(arm, fold))
                rows.append(row)
            expected = int(frame["matrix_evaluation_eligible"].sum())
            observed = int(sum(row["test_rows"] for row in rows if row["cohort_arm"] == arm and row["regime"] == regime))
            if observed != expected:
                raise RuntimeError(
                    "{} {} evaluation coverage mismatch: {} != {}".format(
                        arm, regime, observed, expected
                    )
                )
    return pd.DataFrame(rows)


def source_linked_biochemical_keys(broad: pd.DataFrame, strict: pd.DataFrame):
    """Return the legacy orthosteric-source-linked biochemical key set.

    This is intentionally not named a metabolite catalog.  Membership derives
    from the evidence provenance used to construct orthosteric examples and is
    therefore unsuitable as an independent metabolite-domain definition.
    """
    source = broad["source_databases"].fillna("").astype(str)
    selected = broad[source.str.contains("KEGG|BRENDA|ChEBI|UniProt", case=False, regex=True)].copy()
    if selected.empty:
        raise RuntimeError("orthosteric-source-linked biochemical key source is empty")
    if not selected["binary_label"].astype(int).eq(0).all():
        counts = selected["binary_label"].astype(int).value_counts().to_dict()
        raise RuntimeError(
            "biochemical source provenance is no longer orthosteric-only: {}".format(
                counts
            )
        )
    strict_keys = set(strict["connectivity_key"].astype(str))
    rows = []
    for key, group in selected.groupby("connectivity_key", sort=True):
        sources = sorted(
            {
                token.strip()
                for value in group["source_databases"].fillna("").astype(str)
                for token in value.split(";")
                if token.strip()
            }
        )
        rows.append(
            {
                "connectivity_key": str(key),
                "reference_sources": ";".join(sources),
                "reference_rows": int(len(group)),
                "reference_allosteric_rows": int(
                    group["binary_label"].astype(int).eq(1).sum()
                ),
                "reference_orthosteric_rows": int(
                    group["binary_label"].astype(int).eq(0).sum()
                ),
                "reference_proteins": int(group["uniprot"].nunique()),
                "seen_in_role_complete_training": int(str(key) in strict_keys),
                "domain_label": "orthosteric_source_linked_biochemical",
            }
        )
    return pd.DataFrame(rows)


def rank_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (ranks[labels == 1].sum()
         - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def role_completeness_audit(strict: pd.DataFrame):
    ligand = (
        strict.groupby("full_inchikey", as_index=False)
        .agg(
            allosteric_rows=("binary_label", "sum"),
            total_rows=("binary_label", "size"),
            proteins=("uniprot", "nunique"),
            connectivity_key=("connectivity_key", "first"),
        )
    )
    ligand["orthosteric_rows"] = ligand["total_rows"] - ligand["allosteric_rows"]
    ligand["allosteric_prevalence"] = ligand["allosteric_rows"] / ligand["total_rows"]
    lookup = ligand.set_index("full_inchikey")["allosteric_prevalence"]
    scores = strict["full_inchikey"].map(lookup).astype(float).to_numpy()
    labels = strict["binary_label"].astype(int).to_numpy()
    oracle = rank_auc(labels, scores)
    summary = {
        "ligands": int(len(ligand)),
        "ligands_exactly_one_to_one": int(
            ligand["allosteric_rows"].eq(ligand["orthosteric_rows"]).sum()
        ),
        "minimum_ligand_allosteric_prevalence": float(ligand["allosteric_prevalence"].min()),
        "maximum_ligand_allosteric_prevalence": float(ligand["allosteric_prevalence"].max()),
        "ligand_identity_in_sample_prevalence_oracle_auroc": oracle,
        "interpretation": (
            "role completeness removes deterministic identity rules but not "
            "between-ligand prevalence information; within-ligand evaluation "
            "is the primary ligand-control endpoint"
        ),
    }
    return ligand.sort_values("full_inchikey").reset_index(drop=True), summary


def double_unseen_audit(arms: dict[str, pd.DataFrame]):
    rows = []
    ids_by_arm = {}
    for arm, frame in arms.items():
        arm_ids = set()
        for fold in range(5):
            test = frame[
                frame["matrix_family_fold"].eq(fold)
                & frame["matrix_evaluation_eligible"].eq(1)
            ].copy()
            development = frame[~frame["matrix_family_fold"].eq(fold)]
            seen = set(development["connectivity_key"].astype(str))
            novel = test[~test["connectivity_key"].astype(str).isin(seen)]
            arm_ids.update(novel["main_row_id"].astype(str))
            rows.append(
                {
                    "cohort_arm": arm,
                    "outer_fold": fold,
                    "family_held_out_rows": int(len(test)),
                    "family_held_out_ligand_novel_rows": int(len(novel)),
                    "allosteric_rows": int(novel["binary_label"].eq(1).sum()),
                    "orthosteric_rows": int(novel["binary_label"].eq(0).sum()),
                    "proteins": int(novel["uniprot"].nunique()),
                    "ligands": int(novel["connectivity_key"].nunique()),
                }
            )
        ids_by_arm[arm] = arm_ids
    common_general = ids_by_arm["every_pair"] & ids_by_arm["protein_anchored"]
    summary = {
        "every_pair_rows": int(len(ids_by_arm["every_pair"])),
        "protein_anchored_rows": int(len(ids_by_arm["protein_anchored"])),
        "common_general_rows": int(len(common_general)),
        "role_complete_rows": int(len(ids_by_arm["protein_ligand_role_complete"])),
        "role_complete_supported": False,
        "role_complete_reason": "only 11 rows and three of five folds contain zero rows",
    }
    return pd.DataFrame(rows), summary, common_general


def joint_component_audit(arms: dict[str, pd.DataFrame]):
    """Quantify why a true family-plus-ligand joint partition degenerates."""
    rows = []
    summary = {}
    for arm, frame in arms.items():
        parent = {}

        def find(value):
            parent.setdefault(value, value)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left, right):
            left, right = find(left), find(right)
            if left != right:
                parent[right] = left

        for row in frame.itertuples(index=False):
            union("family:" + str(row.family_component_id), "ligand:" + str(row.connectivity_key))
        grouped = {}
        for row in frame.itertuples(index=False):
            key = find("family:" + str(row.family_component_id))
            value = grouped.setdefault(key, {"rows": 0, "families": set(), "ligands": set()})
            value["rows"] += 1
            value["families"].add(str(row.family_component_id))
            value["ligands"].add(str(row.connectivity_key))
        ordered = sorted(grouped.values(), key=lambda value: (-value["rows"], sorted(value["families"])))
        for index, value in enumerate(ordered):
            rows.append(
                {
                    "cohort_arm": arm,
                    "joint_component": index,
                    "rows": int(value["rows"]),
                    "row_fraction": float(value["rows"] / len(frame)),
                    "families": int(len(value["families"])),
                    "ligands": int(len(value["ligands"])),
                }
            )
        summary[arm] = {
            "components": int(len(ordered)),
            "largest_component_rows": int(ordered[0]["rows"]),
            "largest_component_row_fraction": float(ordered[0]["rows"] / len(frame)),
        }
    return pd.DataFrame(rows), summary


def main():
    args = parse_args()
    root = args.project_root.resolve()
    package = root / "analysis/role_complete_pair_matrix"
    data_dir = package / "data"
    validation_dir = package / "validation"
    data_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    main_path = root / "analysis/allosteric_pair_benchmark_main/gpu_cache/MODEL_READY.tsv.gz"
    broad_path = root / "analysis/allosteric_pair_benchmark_broad_superset/gpu_cache/BROAD_MODEL_READY.tsv.gz"
    core = pd.read_csv(main_path, sep="\t", low_memory=False)
    broad = pd.read_csv(broad_path, sep="\t", low_memory=False)
    for name, frame in [("main", core), ("broad", broad)]:
        required = {
            "main_row_id",
            "uniprot",
            "full_inchikey",
            "connectivity_key",
            "binary_label",
            "family_component_id",
            "family_fold",
            "row_fold",
            "protein_embedding_path",
            "ligand_embedding_path",
        }
        if required - set(frame.columns):
            raise RuntimeError("{} table lacks required fields: {}".format(name, sorted(required - set(frame.columns))))
        if frame["main_row_id"].duplicated().any():
            raise RuntimeError("{} main_row_id is not unique".format(name))
    check_subset(core, broad)
    broad_extra = broad[~broad["main_row_id"].astype(str).isin(set(core["main_row_id"].astype(str)))]
    if set(broad_extra["uniprot"].astype(str)) & set(core["uniprot"].astype(str)):
        raise RuntimeError("broad training-only rows overlap evaluation proteins")
    if set(broad_extra["family_component_id"].astype(str)) & set(
        core["family_component_id"].astype(str)
    ):
        raise RuntimeError("broad training-only rows overlap evaluation families")
    strict, iterations = role_complete_fixed_point(core)

    observed = {
        "every_pair": {"rows": int(len(broad)), "proteins": int(broad["uniprot"].nunique())},
        "protein_anchored": {"rows": int(len(core)), "proteins": int(core["uniprot"].nunique())},
        "protein_ligand_role_complete": {
            "rows": int(len(strict)),
            "proteins": int(strict["uniprot"].nunique()),
            "full_ligands": int(strict["full_inchikey"].nunique()),
            "connectivity_ligands": int(strict["connectivity_key"].nunique()),
            "families": int(strict["family_component_id"].nunique()),
            "allosteric": int(strict["binary_label"].eq(1).sum()),
            "orthosteric": int(strict["binary_label"].eq(0).sum()),
        },
    }
    if observed != EXPECTED:
        raise RuntimeError("cohort contract changed: {}".format(json.dumps(observed, sort_keys=True)))
    if not strict.groupby("uniprot")["binary_label"].nunique().eq(2).all():
        raise RuntimeError("strict cohort lost protein role completeness")
    if not strict.groupby("full_inchikey")["binary_label"].nunique().eq(2).all():
        raise RuntimeError("strict cohort lost exact-ligand role completeness")
    if strict.groupby("full_inchikey")["connectivity_key"].nunique().max() != 1:
        raise RuntimeError("one exact ligand maps to multiple connectivity groups")

    stats = group_statistics(broad, core, strict)
    assignments = assign_ligand_folds(stats)
    fold_map = assignments.set_index("connectivity_key")["ligand_fold"].astype(int).to_dict()
    core_ids = set(core["main_row_id"].astype(str))
    strict_ids = set(strict["main_row_id"].astype(str))
    arms = {
        # The 2,217 broad-only rows are training augmentation.  All broad-arm
        # tests use the same protein-anchored rows as the core arm.
        "every_pair": add_contract_columns(broad, "every_pair", fold_map, core_ids),
        "protein_anchored": add_contract_columns(core, "protein_anchored", fold_map, core_ids),
        "protein_ligand_role_complete": add_contract_columns(
            strict, "protein_ligand_role_complete", fold_map, strict_ids
        ),
    }
    audit = fold_audit(arms)
    checkpoint_fallback = checkpoint_fallback_contract(arms)
    double_unseen, double_unseen_summary, common_double_unseen_ids = double_unseen_audit(arms)
    joint_components, joint_component_summary = joint_component_audit(arms)
    ligand_role_audit, ligand_role_summary = role_completeness_audit(strict)
    if double_unseen_summary != {
        "every_pair_rows": 2721,
        "protein_anchored_rows": 2854,
        "common_general_rows": 2721,
        "role_complete_rows": 11,
        "role_complete_supported": False,
        "role_complete_reason": "only 11 rows and three of five folds contain zero rows",
    }:
        raise RuntimeError("double-unseen support contract changed: {}".format(double_unseen_summary))

    output_paths = {}
    names = {
        "every_pair": "EVERY_PAIR.tsv.gz",
        "protein_anchored": "PROTEIN_ANCHORED.tsv.gz",
        "protein_ligand_role_complete": "PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz",
    }
    for arm, frame in arms.items():
        path = data_dir / names[arm]
        frame.to_csv(
            path,
            sep="\t",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        output_paths[arm] = path
    assignments.to_csv(data_dir / "LIGAND_FOLD_ASSIGNMENTS.tsv", sep="\t", index=False)
    stats.merge(assignments, on="connectivity_key", how="left", validate="one_to_one").to_csv(
        validation_dir / "LIGAND_FOLD_BALANCE.tsv", sep="\t", index=False
    )
    audit.to_csv(validation_dir / "FOLD_AUDIT.tsv", sep="\t", index=False)
    double_unseen.to_csv(validation_dir / "DOUBLE_UNSEEN_SUPPORT.tsv", sep="\t", index=False)
    pd.DataFrame({"main_row_id": sorted(common_double_unseen_ids)}).to_csv(
        data_dir / "COMMON_GENERAL_DOUBLE_UNSEEN_ROWS.tsv", sep="\t", index=False
    )
    joint_components.to_csv(validation_dir / "JOINT_SPLIT_COMPONENTS.tsv", sep="\t", index=False)
    ligand_role_audit.to_csv(validation_dir / "LIGAND_ROLE_PREVALENCE_AUDIT.tsv", sep="\t", index=False)
    biochemical = source_linked_biochemical_keys(broad, strict)
    biochemical.to_csv(
        data_dir / "ORTHOSTERIC_SOURCE_LINKED_BIOCHEMICAL_KEYS.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            ("ligand", "ligand", "ligand-only baseline"),
            ("protein", "protein", "protein-only baseline"),
            ("c1", "new", "whole selected-target-chain concat"),
            ("c2", "c2", "whole selected-target-chain one-way attention"),
            ("c3", "new", "whole selected-target-chain bidirectional attention"),
            ("d1", "c1", "pocket concat"),
            ("d2", "c3", "pocket one-way attention"),
            ("d3", "d1", "pocket bidirectional attention"),
        ],
        columns=["new_model", "legacy_model", "definition"],
    ).to_csv(data_dir / "MODEL_NAME_MAPPING.tsv", sep="\t", index=False)

    pfam_payload = json.loads(
        (root / EXTERNAL_RESOURCES["pfam_cache"]).read_text(encoding="utf-8")
    )
    pfam_records = set(pfam_payload.get("records", {}))
    pfam_training_coverage = {}
    for arm, frame in arms.items():
        proteins = set(frame["uniprot"].astype(str).str.split("-").str[0])
        pfam_training_coverage[arm] = {
            "proteins": int(len(proteins)),
            "in_frozen_cache": int(len(proteins & pfam_records)),
            "missing_from_frozen_cache": int(len(proteins - pfam_records)),
            "complete": not bool(proteins - pfam_records),
        }

    report = {
        "status": "validated",
        "contract_id": "role_complete_pair_matrix_v2",
        "cohorts": observed,
        "strict_fixed_point_iterations": int(iterations),
        "cohort_relationship": "protein_ligand_role_complete is a subset of protein_anchored, which is a subset of every_pair",
        "role_anchor_identity": "standardized full InChIKey",
        "ligand_fold_group": "InChIKey connectivity block",
        "active_training_regimes_by_cohort": {
            arm: list(regimes) for arm, regimes in ACTIVE_REGIMES.items()
        },
        "expected_benchmark_fits": int(
            sum(len(regimes) for regimes in ACTIVE_REGIMES.values()) * 8 * 5 * 3
        ),
        "checkpoint_selection": checkpoint_fallback,
        "single_input_numerical_canonicalization": {
            "enabled": True,
            "scope": "OOF ensemble predictions within outer fold",
            "ligand_identity": "standardized full InChIKey",
            "protein_identity": "UniProt selected-target-chain identifier",
            "operation": "replace repeated exact-input scores by their within-fold mean before metric computation",
            "reason": "prevent mixed-precision batch-padding roundoff from creating artificial ranks for an identity-constant model",
            "maximum_allowed_probability_span": 0.01,
            "preserves_connectivity_level_stereoisomer_variation": True,
        },
        "primary_third_condition": {
            "every_pair": "ligand-held-out CV plus a separate family-and-ligand-unseen hard endpoint",
            "protein_anchored": "ligand-held-out CV plus a separate family-and-ligand-unseen hard endpoint",
            "protein_ligand_role_complete": "separate ligand-held-out folds",
        },
        "evaluation_rows": {
            "every_pair": int(arms["every_pair"]["matrix_evaluation_eligible"].sum()),
            "protein_anchored": int(arms["protein_anchored"]["matrix_evaluation_eligible"].sum()),
            "protein_ligand_role_complete": int(
                arms["protein_ligand_role_complete"]["matrix_evaluation_eligible"].sum()
            ),
        },
        "every_pair_extra_rows": {
            "rows": int(arms["every_pair"]["matrix_evaluation_eligible"].eq(0).sum()),
            "role": "training_only_augmentation",
            "oof_claim": False,
            "evaluation_protein_overlap": 0,
            "evaluation_family_overlap": 0,
            "evaluation_ligand_connectivity_overlap": int(
                len(
                    set(broad_extra["connectivity_key"].astype(str))
                    & set(core["connectivity_key"].astype(str))
                )
            ),
        },
        "double_unseen_evaluation": double_unseen_summary,
        "simultaneous_family_and_ligand_holdout": {
            "constructed": False,
            "reason": "family-ligand bipartite components degenerate in all three cohorts",
            "component_audit": joint_component_summary,
            "replacement": "family-held-out OOF rows restricted to ligands absent from train plus validation",
        },
        "additional_ligand_weighting": False,
        "ligand_role_prevalence_audit": ligand_role_summary,
        "source_linked_biochemical_keys": int(len(biochemical)),
        "source_linked_biochemical_source_rows": int(biochemical["reference_rows"].sum()),
        "source_linked_biochemical_training_label_origin": "orthosteric_only",
        "source_linked_biochemical_definition": (
            "connectivity keys selected through orthosteric KEGG, BRENDA, ChEBI, "
            "or UniProt provenance; not an independent metabolite catalog"
        ),
        "independent_metabolite_screen_available": False,
        "chembl_deploy_training_arms": [
            "every_pair",
            "protein_anchored",
            "protein_ligand_role_complete",
        ],
        "chembl_primary_general_arm": "protein_anchored",
        "chembl_every_pair_role": "prespecified training-coverage sensitivity",
        "external_pfam_training_coverage": pfam_training_coverage,
        "every_pair_pfam_unseen_claim_allowed": False,
        "whole_protein_representation": "structure_matched_selected_target_chain",
        "files": {
            str(path.relative_to(package)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in list(output_paths.values())
            + [
                data_dir / "LIGAND_FOLD_ASSIGNMENTS.tsv",
                data_dir / "ORTHOSTERIC_SOURCE_LINKED_BIOCHEMICAL_KEYS.tsv",
                data_dir / "COMMON_GENERAL_DOUBLE_UNSEEN_ROWS.tsv",
                data_dir / "MODEL_NAME_MAPPING.tsv",
                validation_dir / "FOLD_AUDIT.tsv",
                validation_dir / "LIGAND_FOLD_BALANCE.tsv",
                validation_dir / "DOUBLE_UNSEEN_SUPPORT.tsv",
                validation_dir / "JOINT_SPLIT_COMPONENTS.tsv",
                validation_dir / "LIGAND_ROLE_PREVALENCE_AUDIT.tsv",
            ]
        },
        "implementation_files": {
            str(path.relative_to(root)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(
                list((package / "scripts").glob("*.py"))
                + list((package / "scripts").glob("*.sh"))
                + [
                    root / "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py",
                    root / "analysis/allosteric_pair_benchmark_broad_superset/data/POCKET_INDICES.json",
                    package / "README.md",
                    package / "EXPERIMENT_CONTRACT.md",
                    package / "CODEX_GPU_TASK.md",
                    package / "methods/MATERIALS_AND_METHODS.md",
                    package / "requirements-cpu.txt",
                    package / "requirements-gpu.txt",
                ]
            )
        },
        "external_resources": {
            name: {
                "path": relative,
                "sha256": sha256(root / relative),
                "bytes": (root / relative).stat().st_size,
            }
            for name, relative in EXTERNAL_RESOURCES.items()
        },
    }
    report["contracted_data_and_validation_files"] = int(len(report["files"]))
    report["contracted_implementation_and_document_files"] = int(
        len(report["implementation_files"])
    )
    report["contracted_internal_files_total"] = int(
        len(report["files"]) + len(report["implementation_files"])
    )
    report["contracted_external_resources"] = int(
        len(report["external_resources"])
    )
    atomic_json(validation_dir / "CPU_CONTRACT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
