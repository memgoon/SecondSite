#!/usr/bin/env python3
"""Freeze uncontrolled, protein-controlled, and ligand-novel stage datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


N_FOLDS = 5
SEED = 20260817


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--pfam-cache", type=Path, default=None)
    parser.add_argument("--protein-template", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validation-path", type=Path, default=None)
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_pfams(value):
    if value is None:
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip().startswith("PF")}


def stable_hash(*values):
    payload = "|".join(map(str, values)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def component_id(members):
    payload = ";".join(sorted(members)).encode("utf-8")
    return "PFONLY_" + hashlib.sha256(payload).hexdigest()[:16]


def add_pfam_components(frame, pfam_cache):
    proteins = sorted(frame["uniprot"].astype(str).unique())
    uid_pfams = {uid: parse_pfams(pfam_cache.get(uid, "")) for uid in proteins}
    union_find = UnionFind(proteins)
    pfam_members = defaultdict(list)
    for uid in proteins:
        for pfam in uid_pfams[uid]:
            pfam_members[pfam].append(uid)
    for members in pfam_members.values():
        members = sorted(set(members))
        for member in members[1:]:
            union_find.union(members[0], member)
    roots = defaultdict(list)
    for uid in proteins:
        roots[union_find.find(uid)].append(uid)
    uid_component = {}
    for members in roots.values():
        identifier = component_id(members)
        for uid in members:
            uid_component[uid] = identifier
    result = frame.copy()
    result["family_component_id"] = result["uniprot"].astype(str).map(uid_component)
    return result, uid_pfams


def component_table(frame):
    rows = []
    for identifier, group in frame.groupby("family_component_id"):
        rows.append({
            "family_component_id": identifier,
            "n_proteins": int(group["uniprot"].nunique()),
            "n_rows": int(len(group)),
            "n_allosteric": int(group["binary_label"].eq(1).sum()),
            "n_orthosteric": int(group["binary_label"].eq(0).sum()),
        })
    return pd.DataFrame(rows)


def assign_component_folds(components):
    columns = ["n_proteins", "n_allosteric", "n_orthosteric", "n_rows"]
    weights = np.asarray([2.0, 1.5, 1.5, 0.5], dtype=float)
    values = components[columns].to_numpy(dtype=float)
    targets = values.sum(axis=0) / N_FOLDS
    targets[targets == 0] = 1.0
    peak = (values / targets).max(axis=1)
    order = sorted(range(len(components)), key=lambda index: (
        -peak[index], -float(components.iloc[index]["n_proteins"]),
        -float(components.iloc[index]["n_rows"]),
        str(components.iloc[index]["family_component_id"]),
    ))
    current = np.zeros((N_FOLDS, len(columns)), dtype=float)
    members = [[] for _ in range(N_FOLDS)]
    assignment = {}

    def objective(candidate):
        scaled = (candidate - targets[None, :]) / targets[None, :]
        return float(np.sum((scaled * scaled) * weights[None, :]))

    for index in order:
        vector = values[index]
        identifier = str(components.iloc[index]["family_component_id"])
        choices = []
        for fold in range(N_FOLDS):
            proposed = current.copy()
            proposed[fold] += vector
            choices.append((objective(proposed), len(members[fold]), fold))
        selected = min(choices)[-1]
        assignment[identifier] = selected
        current[selected] += vector
        members[selected].append(identifier)
    if any(not group for group in members):
        raise RuntimeError("component assignment produced an empty fold")
    return assignment


def choose_roles(components):
    giant = components.sort_values(
        ["n_proteins", "n_rows", "family_component_id"], ascending=[False, False, True]
    ).iloc[0]
    giant_fold = int(giant["family_fold"])
    fold = components.groupby("family_fold").agg(
        n_components=("family_component_id", "nunique"),
        n_proteins=("n_proteins", "sum"), n_rows=("n_rows", "sum"),
        n_allosteric=("n_allosteric", "sum"), n_orthosteric=("n_orthosteric", "sum"),
    ).reset_index()
    targets = {column: float(fold[column].sum()) / N_FOLDS for column in [
        "n_proteins", "n_rows", "n_allosteric", "n_orthosteric"
    ]}
    candidates = []
    for row in fold.itertuples(index=False):
        if int(row.family_fold) == giant_fold:
            continue
        score = sum(((float(getattr(row, column)) - targets[column]) / max(targets[column], 1.0)) ** 2 for column in targets)
        candidates.append((score, -int(row.n_components), int(row.family_fold)))
    candidates.sort()
    return candidates[0][2], candidates[1][2], giant


def stratified_random_split(frame):
    result = frame.copy()
    result["split"] = ""
    for label, indices in result.groupby("binary_label").groups.items():
        ordered = sorted(indices, key=lambda index: stable_hash(SEED, label, result.at[index, "ladder_row_id"]))
        n = len(ordered)
        n_test = int(round(n * 0.20))
        n_val = int(round(n * 0.10))
        result.loc[ordered[:n_test], "split"] = "test"
        result.loc[ordered[n_test:n_test + n_val], "split"] = "val"
        result.loc[ordered[n_test + n_val:], "split"] = "train"
    return result


def split_summary(frame):
    rows = []
    for split in ["train", "val", "test"]:
        subset = frame[frame["split"].eq(split)]
        paired = set(subset.groupby("uniprot")["binary_label"].nunique().loc[lambda value: value.eq(2)].index.astype(str))
        rows.append({
            "split": split,
            "n_rows": int(len(subset)),
            "n_proteins": int(subset["uniprot"].nunique()),
            "n_components": int(subset["family_component_id"].nunique()),
            "n_ligands": int(subset["full_inchikey"].nunique()),
            "n_connectivity_groups": int(subset["connectivity_key"].nunique()),
            "n_allosteric": int(subset["binary_label"].eq(1).sum()),
            "n_orthosteric": int(subset["binary_label"].eq(0).sum()),
            "n_two_label_proteins": int(len(paired)),
            "n_rows_on_two_label_proteins": int(subset[subset["uniprot"].astype(str).isin(paired)].shape[0]),
        })
    return rows


def overlap(frame, column):
    sets = {split: set(frame.loc[frame["split"].eq(split), column].astype(str)) for split in ["train", "val", "test"]}
    return {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }


def main():
    args = parse_args()
    package = args.project_root / "analysis/new_pairing_control_ladder"
    source = args.source or package / "gpu_cache/BASE_MODEL_READY.tsv.gz"
    pfam_path = args.pfam_cache or args.project_root / "14.Organized_input/uniprot_pfam_cache.json"
    protein_template_path = args.protein_template or args.project_root / "analysis/new_pairing_pfam_only_pilot/data/PFAM_ONLY_GPU_ELIGIBLE.tsv.gz"
    output_dir = args.output_dir or package / "gpu_cache/stages"
    validation_path = args.validation_path or package / "gpu_cache/CONTROL_STAGE_VALIDATION.json"
    if not source.is_file() or not pfam_path.is_file() or not protein_template_path.is_file():
        raise FileNotFoundError("missing base model-ready table, Pfam cache, or frozen protein template")
    base = pd.read_csv(source, sep="\t", low_memory=False)
    if "pilot_row_id" not in base.columns:
        base["pilot_row_id"] = base["ladder_row_id"].astype(str)
    with open(pfam_path, encoding="utf-8") as handle:
        pfam_cache = json.load(handle)

    uncontrolled, _ = add_pfam_components(base, pfam_cache)
    uncontrolled = stratified_random_split(uncontrolled)
    uncontrolled["control_stage"] = "uncontrolled"
    uncontrolled["paired_eval"] = uncontrolled.groupby(["split", "uniprot"])["binary_label"].transform("nunique").eq(2).astype(int)

    template = pd.read_csv(protein_template_path, sep="\t", low_memory=False)
    template_fields = template[["pair_key", "family_component_id", "family_fold", "split"]].drop_duplicates("pair_key")
    template_keys = set(template_fields["pair_key"].astype(str))
    base_keys = set(base["pair_key"].astype(str))
    if not template_keys.issubset(base_keys):
        raise RuntimeError("frozen Pfam-only template contains rows absent from the broad GPU-ready base")
    protein = base[base["pair_key"].astype(str).isin(template_keys)].copy()
    protein = protein.drop(columns=["family_component_id", "family_fold", "split"], errors="ignore").merge(
        template_fields, on="pair_key", how="left", validate="one_to_one"
    )
    if len(protein) != len(template_fields) or protein[["family_component_id", "family_fold", "split"]].isna().any().any():
        raise RuntimeError("failed to reproduce the frozen Pfam-only protein-controlled universe")
    uid_pfams = {
        uid: parse_pfams(pfam_cache.get(uid, ""))
        for uid in sorted(protein["uniprot"].astype(str).unique())
    }
    components = component_table(protein)
    component_folds = protein[["family_component_id", "family_fold"]].drop_duplicates()
    if component_folds["family_component_id"].duplicated().any():
        raise RuntimeError("one Pfam component appears in multiple frozen folds")
    components = components.merge(component_folds, on="family_component_id", how="left", validate="one_to_one")
    giant = components.sort_values(
        ["n_proteins", "n_rows", "family_component_id"], ascending=[False, False, True]
    ).iloc[0]
    test_folds = protein.loc[protein["split"].eq("test"), "family_fold"].unique()
    val_folds = protein.loc[protein["split"].eq("val"), "family_fold"].unique()
    if len(test_folds) != 1 or len(val_folds) != 1:
        raise RuntimeError("frozen template has ambiguous split roles")
    test_fold, val_fold = int(test_folds[0]), int(val_folds[0])
    protein["control_stage"] = "protein_controlled"
    protein["paired_eval"] = 1

    fully = protein.copy()
    train_groups = set(fully.loc[fully["split"].eq("train"), "connectivity_key"].astype(str))
    remove_val = fully["split"].eq("val") & fully["connectivity_key"].astype(str).isin(train_groups)
    fully = fully[~remove_val].copy()
    development_groups = set(fully.loc[fully["split"].isin(["train", "val"]), "connectivity_key"].astype(str))
    remove_test = fully["split"].eq("test") & fully["connectivity_key"].astype(str).isin(development_groups)
    fully = fully[~remove_test].copy()
    fully["control_stage"] = "fully_controlled"
    fully["paired_eval"] = fully.groupby(["split", "uniprot"])["binary_label"].transform("nunique").eq(2).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "uncontrolled": output_dir / "UNCONTROLLED.tsv.gz",
        "protein_controlled": output_dir / "PROTEIN_CONTROLLED.tsv.gz",
        "fully_controlled": output_dir / "FULLY_CONTROLLED.tsv.gz",
    }
    stages = {"uncontrolled": uncontrolled, "protein_controlled": protein, "fully_controlled": fully}
    for name, stage in stages.items():
        stage.sort_values(["split", "family_component_id", "uniprot", "class_label", "full_inchikey"]).to_csv(
            paths[name], sep="\t", index=False, compression="gzip"
        )

    protein_overlap = overlap(protein, "uniprot")
    protein_pfam_overlap = {
        left + "_" + right: len(
            {pfam for uid in set(protein.loc[protein["split"].eq(left), "uniprot"].astype(str)) for pfam in uid_pfams[uid]}
            & {pfam for uid in set(protein.loc[protein["split"].eq(right), "uniprot"].astype(str)) for pfam in uid_pfams[uid]}
        )
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    report = {
        "status": "validated",
        "definitions": {
            "uncontrolled": "broad structure/embedding-ready cohort with row-level stratified random split; protein and ligand overlap allowed",
            "protein_controlled": "proteins carrying both labels; exact-Pfam connected components held disjoint",
            "fully_controlled": "same protein-controlled Pfam split; validation/test connectivity groups seen earlier are removed",
        },
        "base_rows": int(len(base)),
        "base_proteins": int(base["uniprot"].nunique()),
        "protein_template": str(protein_template_path),
        "protein_template_exactly_reproduced": bool(set(protein["pair_key"].astype(str)) == template_keys),
        "stage_counts": {name: split_summary(stage) for name, stage in stages.items()},
        "overlaps": {
            name: {
                "protein": overlap(stage, "uniprot"),
                "full_inchikey": overlap(stage, "full_inchikey"),
                "connectivity_key": overlap(stage, "connectivity_key"),
                "family_component": overlap(stage, "family_component_id"),
            }
            for name, stage in stages.items()
        },
        "protein_controlled_exact_pfam_overlap": protein_pfam_overlap,
        "fully_removed_validation_rows": int(remove_val.sum()),
        "fully_removed_test_rows": int(remove_test.sum()),
        "largest_pfam_component": {
            "id": str(giant["family_component_id"]),
            "n_proteins": int(giant["n_proteins"]),
            "n_rows": int(giant["n_rows"]),
        },
        "test_fold": int(test_fold),
        "validation_fold": int(val_fold),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    conditions = [
        set(uncontrolled["split"]) == {"train", "val", "test"},
        all(value == 0 for value in protein_overlap.values()),
        all(value == 0 for value in protein_pfam_overlap.values()),
        all(value == 0 for value in overlap(fully, "connectivity_key").values()),
        report["protein_template_exactly_reproduced"],
        set(fully["binary_label"]) == {0, 1},
        all(len(fully[fully["split"].eq(split)]) > 0 for split in ["train", "val", "test"]),
    ]
    if not all(conditions):
        report["status"] = "failed"
    atomic_json(validation_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("control-stage validation failed")


if __name__ == "__main__":
    main()
