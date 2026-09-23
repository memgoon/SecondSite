#!/usr/bin/env python3
"""Build the broad uncontrolled structure candidate table using named inputs only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
SHARED = Path("/shared_data/11.HS_allostery")
REMOTE = Path("/disk1/11.HS_allostery")
PACKAGE = ROOT / "analysis/new_pairing_control_ladder"
RAW = ROOT / "analysis/allosteric_orthosteric_preprocessing_v1_participant_kegg_expansion/data/UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC_PARTICIPANT_KEGG_AUGMENTED.tsv.gz"
EXCLUSIONS = ROOT / "analysis/allosteric_orthosteric_preprocessing_v1_participant_kegg_expansion/data/AUDIT_EXCLUDED_BIOLIP_ROWS.tsv"
PROTEIN_MAP = SHARED / "14.Organized_input/Protein_Embedding_Map.tsv"
POCKET_MASKS = SHARED / "14.Organized_input/pocket_masks.json"
SEQUENCE_CACHE = SHARED / "14.Organized_input/uniprot_protein_cache.json"
OLD_GRAPH_AUDIT = ROOT / "analysis/new_pairing_d1_pilot/data/STRUCTURE_GRAPH_AUDIT.tsv"
OUTPUT = PACKAGE / "data/BROAD_UNCONTROLLED_PREGRAPH.tsv.gz"
VALIDATION = PACKAGE / "validation/BROAD_PREGRAPH_VALIDATION.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def clean(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def cpu_path(remote_path):
    text = clean(remote_path)
    prefix = str(REMOTE)
    return str(SHARED) + text[len(prefix):] if text.startswith(prefix + "/") else text


def remote_path(local_or_remote):
    text = clean(local_or_remote)
    prefix = str(SHARED)
    return str(REMOTE) + text[len(prefix):] if text.startswith(prefix + "/") else text


def canonical_embedding_paths(protein_map):
    paths = {}
    for row in protein_map.itertuples(index=False):
        uid = clean(row.UniProt_ID).split("-")[0]
        path = clean(row.Embedding_Path)
        expected = "/14.Organized_input/Protein_Embeddings/{}.pt".format(uid)
        physical = cpu_path(path)
        if expected not in path or not physical or not os.path.isfile(physical):
            continue
        paths[uid] = remote_path(path)
    return paths


def has_named_coordinate(group):
    for row in group.itertuples(index=False):
        for value in [clean(row.Mapped_PDB), clean(row.Original_Target_PDB)]:
            if len(value) >= 4:
                return True
        stem = Path(clean(row.Embedding_Path)).stem.lower()
        if len(stem) == 4 and stem[0].isdigit():
            return True
    return False


def main():
    raw = pd.read_csv(RAW, sep="\t", low_memory=False)
    excluded = pd.read_csv(EXCLUSIONS, sep="\t", dtype=str).fillna("")
    exclusion_keys = set(zip(excluded["uniprot"], excluded["full_inchikey"]))
    raw["_pair_key"] = list(zip(raw["uniprot"].astype(str), raw["full_inchikey"].astype(str)))
    cleaned = raw[~raw["_pair_key"].isin(exclusion_keys)].copy()
    conflicts = cleaned.groupby(["uniprot", "full_inchikey"])["class_label"].nunique()
    if int(conflicts.gt(1).sum()):
        raise RuntimeError("cross-class exact-pair conflicts remain")
    cleaned = cleaned.drop_duplicates(["uniprot", "full_inchikey"], keep="first")
    cleaned["uniprot"] = cleaned["uniprot"].astype(str).str.split("-").str[0]

    protein_map = pd.read_csv(PROTEIN_MAP, sep="\t", dtype=str).fillna("")
    protein_map["uid"] = protein_map["UniProt_ID"].str.split("-").str[0]
    embedding_paths = canonical_embedding_paths(protein_map)
    with open(POCKET_MASKS, encoding="utf-8") as handle:
        pocket_masks = json.load(handle)
    with open(SEQUENCE_CACHE, encoding="utf-8") as handle:
        sequence_cache = json.load(handle)
    old_audit = pd.read_csv(OLD_GRAPH_AUDIT, sep="\t", dtype=str).fillna("")
    old_ready = set(old_audit.loc[old_audit["status"].eq("ready"), "uniprot"])

    eligible = []
    coordinate_hint = {}
    for uid in sorted(set(cleaned["uniprot"])):
        if uid not in embedding_paths:
            continue
        if not pocket_masks.get(uid, []):
            continue
        sequence = sequence_cache.get(uid, {})
        sequence = sequence.get("seq", "") if isinstance(sequence, dict) else ""
        if not clean(sequence):
            continue
        rows = protein_map[protein_map["uid"].eq(uid)]
        named = has_named_coordinate(rows)
        reused = uid in old_ready
        if not named and not reused:
            continue
        eligible.append(uid)
        coordinate_hint[uid] = "old_validated_graph" if reused else "named_protein_map_coordinate"

    frame = cleaned[cleaned["uniprot"].isin(eligible)].copy()
    frame["binary_label"] = frame["class_label"].map({"orthosteric": 0, "allosteric": 1}).astype(int)
    frame["pair_key"] = frame["uniprot"] + "|" + frame["full_inchikey"].astype(str)
    frame["protein_embedding_path"] = frame["uniprot"].map(embedding_paths)
    frame["coordinate_candidate_source"] = frame["uniprot"].map(coordinate_hint)
    keep = [
        "dataset_row_id", "pair_key", "uniprot", "full_inchikey", "connectivity_key",
        "canonical_smiles", "class_label", "binary_label", "source_databases",
        "source_lanes", "evidence_subtypes", "n_evidence_rows", "protein_embedding_path",
        "coordinate_candidate_source",
    ]
    frame = frame[keep].sort_values(["uniprot", "class_label", "full_inchikey"]).reset_index(drop=True)
    frame.insert(0, "ladder_row_id", ["LADDER_{:06d}".format(index) for index in range(len(frame))])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, sep="\t", index=False, compression="gzip")

    labels = frame["class_label"].value_counts().to_dict()
    report = {
        "status": "validated",
        "no_directory_traversal": True,
        "clean_uncontrolled_rows": int(len(cleaned)),
        "clean_uncontrolled_proteins": int(cleaned["uniprot"].nunique()),
        "candidate_rows": int(len(frame)),
        "candidate_proteins": int(frame["uniprot"].nunique()),
        "candidate_ligands": int(frame["full_inchikey"].nunique()),
        "candidate_allosteric": int(labels.get("allosteric", 0)),
        "candidate_orthosteric": int(labels.get("orthosteric", 0)),
        "candidate_blank_smiles_rows": int(frame["canonical_smiles"].fillna("").astype(str).str.strip().eq("").sum()),
        "candidate_proteins_with_both_labels": int(frame.groupby("uniprot")["binary_label"].nunique().eq(2).sum()),
        "inputs": {
            "raw": {"path": str(RAW), "sha256": sha256(RAW)},
            "exclusions": {"path": str(EXCLUSIONS), "sha256": sha256(EXCLUSIONS)},
            "protein_map": {"path": str(PROTEIN_MAP), "sha256": sha256(PROTEIN_MAP)},
            "pocket_masks": {"path": str(POCKET_MASKS), "sha256": sha256(POCKET_MASKS)},
            "sequence_cache": {"path": str(SEQUENCE_CACHE), "sha256": sha256(SEQUENCE_CACHE)},
            "old_graph_audit": {"path": str(OLD_GRAPH_AUDIT), "sha256": sha256(OLD_GRAPH_AUDIT)},
        },
        "output": {"path": str(OUTPUT), "sha256": sha256(OUTPUT)},
    }
    if not all([
        len(cleaned) == 20640,
        frame["uniprot"].nunique() > 0,
        set(frame["binary_label"]) == {0, 1},
        frame["protein_embedding_path"].map(clean).ne("").all(),
    ]):
        report["status"] = "failed"
    atomic_json(VALIDATION, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("broad pregraph validation failed")


if __name__ == "__main__":
    main()
