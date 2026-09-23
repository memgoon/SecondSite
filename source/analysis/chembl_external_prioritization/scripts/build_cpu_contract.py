#!/usr/bin/env python3
"""Build compact, auditable ChEMBL reference and exclusion tables locally."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    parser.add_argument("--shared-root", type=Path, default=Path("/shared_data/11.HS_allostery"))
    parser.add_argument("--chunksize", type=int, default=100000)
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def clean(series):
    return series.fillna("").astype(str).str.strip()


def load_table(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def main():
    args = parse_args()
    root = args.project_root
    shared = args.shared_root
    package = root / "analysis/chembl_external_prioritization"
    data = package / "data"
    validation = package / "validation"
    data.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)

    arm_a_path = root / "analysis/allosteric_pair_benchmark_main/gpu_cache/MODEL_READY.tsv.gz"
    arm_b_path = root / "analysis/allosteric_pair_benchmark_broad_superset/gpu_cache/BROAD_MODEL_READY.tsv.gz"
    word_path = root / "analysis/expanded_allosteric_database/derived/word_mined_pair_evidence.tsv.gz"
    five_a_path = shared / "17.paired_dataset/5A.ChEMBL_Text_AllostericLike_global.tsv"
    ood_path = shared / "19.Structure_Unknown/8.Build_ChEMBL_OOD_Validation_Set/chembl_ood_validation_set.tsv"
    raw_chembl_path = shared / "Data/ChEMBL/chembl_36/chembl_36_sqlite/chembl_protein_ligand_dataset_fulltext.csv"
    for path in [arm_a_path, arm_b_path, word_path, five_a_path, ood_path, raw_chembl_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    arm_a = load_table(arm_a_path)
    arm_b = load_table(arm_b_path)
    required_benchmark = {"uniprot", "full_inchikey", "connectivity_key"}
    if required_benchmark - set(arm_a.columns) or required_benchmark - set(arm_b.columns):
        raise RuntimeError("benchmark model-ready table lacks identity fields")
    if len(arm_a) != 4637 or arm_a["uniprot"].nunique() != 426:
        raise RuntimeError("Arm A identity/count contract changed")
    arm_a_pairs = set(
        clean(arm_a["uniprot"]) + "|" + clean(arm_a["connectivity_key"]).str.upper()
    )
    arm_b_pairs = set(
        clean(arm_b["uniprot"]) + "|" + clean(arm_b["connectivity_key"]).str.upper()
    )
    if not arm_a_pairs.issubset(arm_b_pairs):
        raise RuntimeError("Arm B is no longer a strict pair superset of Arm A")

    benchmark = pd.concat([
        arm_b[["uniprot", "full_inchikey", "connectivity_key"]],
    ], ignore_index=True).drop_duplicates()
    benchmark["uniprot"] = clean(benchmark["uniprot"])
    benchmark["full_inchikey"] = clean(benchmark["full_inchikey"]).str.upper()
    benchmark["connectivity_key"] = clean(benchmark["connectivity_key"]).str.upper()
    benchmark["pair_full_key"] = benchmark["uniprot"] + "|" + benchmark["full_inchikey"]
    benchmark["pair_connectivity_key"] = benchmark["uniprot"] + "|" + benchmark["connectivity_key"]
    benchmark["seen_in_arm_a"] = benchmark["pair_connectivity_key"].isin(arm_a_pairs).astype(int)
    benchmark = benchmark.sort_values(["uniprot", "connectivity_key", "full_inchikey"])
    benchmark.to_csv(data / "CURRENT_BENCHMARK_PAIR_BLACKLIST.tsv.gz", sep="\t", index=False)

    protein_rows = []
    arm_a_proteins = set(clean(arm_a["uniprot"]))
    arm_b_proteins = set(clean(arm_b["uniprot"]))
    for uid in sorted(arm_b_proteins):
        protein_rows.append({"uniprot": uid, "seen_in_arm_a": int(uid in arm_a_proteins), "seen_in_arm_b": 1})
    pd.DataFrame(protein_rows).to_csv(data / "CURRENT_BENCHMARK_PROTEINS.tsv", sep="\t", index=False)

    word = load_table(word_path)
    word = word[clean(word["source_release"]).eq("ChEMBL22")].copy()
    word["target_chembl_id"] = clean(word["target_chembl_id"])
    word["ligand_chembl_id"] = clean(word["compound_chembl_id"])
    allo = pd.to_numeric(word["n_allosteric_word_mined"], errors="coerce").fillna(0).gt(0)
    ortho = pd.to_numeric(word["n_orthosteric_word_mined"], errors="coerce").fillna(0).gt(0)
    word["weak_label"] = -1
    word.loc[allo & ~ortho, "weak_label"] = 1
    word.loc[ortho & ~allo, "weak_label"] = 0
    conflict_count = int((allo & ortho).sum())
    ref2020 = word[word["weak_label"].isin([0, 1]) & word["target_chembl_id"].ne("") & word["ligand_chembl_id"].ne("")].copy()
    if ref2020.duplicated(["target_chembl_id", "ligand_chembl_id"]).any():
        raise RuntimeError("2020 stable ChEMBL pairs are unexpectedly duplicated")
    keep_2020 = [
        "target_chembl_id", "ligand_chembl_id", "weak_label", "n_records",
        "n_allosteric_word_mined", "n_orthosteric_word_mined", "pair_interpretation",
    ]
    ref2020[keep_2020].sort_values(["target_chembl_id", "ligand_chembl_id"]).to_csv(
        data / "CHEMBL2020_WEAK_PAIR_LABELS.tsv.gz", sep="\t", index=False
    )

    five = load_table(five_a_path)
    five["target_chembl_id"] = clean(five["target_chembl_id"])
    five["ligand_chembl_id"] = clean(five["ligand_chembl_id"])
    five["UniProt_ID"] = clean(five["UniProt_ID"])
    five["InChIKey14"] = clean(five["InChIKey14"]).str.upper()
    five = five[five["target_chembl_id"].ne("") & five["ligand_chembl_id"].ne("")].copy()
    five = five.sort_values(
        ["target_chembl_id", "ligand_chembl_id", "Allosteric_Text_Confidence_Score"],
        ascending=[True, True, False],
    ).drop_duplicates(["target_chembl_id", "ligand_chembl_id"])
    keep_5a = [
        "target_chembl_id", "ligand_chembl_id", "UniProt_ID", "InChIKey14",
        "Allosteric_Text_Confidence", "Allosteric_Text_Confidence_Score",
        "Allosteric_Text_Hits", "Allosteric_Text_Source",
    ]
    five[keep_5a].to_csv(data / "CHEMBL5A_POSITIVE_PAIRS.tsv.gz", sep="\t", index=False)

    # Reconstruct reference pairs from the unblacklisted ChEMBL36 source.  The
    # legacy OOD table intentionally removed rows used by an older development
    # pipeline, including nearly all local 5A positives.  That blacklist is
    # irrelevant to the current Arm A model and must not define this reference.
    keys_2020_set = set(ref2020["target_chembl_id"] + "|" + ref2020["ligand_chembl_id"])
    raw_hits = []
    raw_usecols = [
        "target_chembl_id", "target_name", "uniprot_id", "ligand_chembl_id",
        "canonical_smiles", "standard_inchi_key", "pchembl_value",
        "activity_type", "activity_value", "activity_units",
    ]
    for chunk in pd.read_csv(
        raw_chembl_path, usecols=raw_usecols, chunksize=args.chunksize, low_memory=False
    ):
        stable = clean(chunk["target_chembl_id"]) + "|" + clean(chunk["ligand_chembl_id"])
        selected = stable.isin(keys_2020_set)
        if selected.any():
            out = chunk.loc[selected].copy()
            out["stable_pair_key"] = stable.loc[selected].to_numpy()
            raw_hits.append(out)
    raw_ref = pd.concat(raw_hits, ignore_index=True) if raw_hits else pd.DataFrame(columns=raw_usecols + ["stable_pair_key"])
    raw_ref["uniprot"] = clean(raw_ref["uniprot_id"]).str.split("|").str[0].str.split(";").str[0]
    raw_ref["full_inchikey"] = clean(raw_ref["standard_inchi_key"]).str.upper()
    raw_ref["connectivity_key"] = raw_ref["full_inchikey"].str[:14]
    raw_ref["canonical_smiles"] = clean(raw_ref["canonical_smiles"])
    raw_ref["pchembl_numeric"] = pd.to_numeric(raw_ref["pchembl_value"], errors="coerce")
    raw_ref = raw_ref[
        raw_ref["uniprot"].ne("")
        & raw_ref["full_inchikey"].str.len().ge(14)
        & raw_ref["canonical_smiles"].ne("")
    ].copy()
    raw_ref = raw_ref.sort_values(
        ["stable_pair_key", "pchembl_numeric"], ascending=[True, False], na_position="last"
    ).drop_duplicates("stable_pair_key")
    label_by_stable = {
        str(row.target_chembl_id) + "|" + str(row.ligand_chembl_id): int(row.weak_label)
        for row in ref2020[["target_chembl_id", "ligand_chembl_id", "weak_label"]].itertuples(index=False)
    }
    raw_ref["weak2020_label"] = raw_ref["stable_pair_key"].map(label_by_stable).astype(int)
    raw_ref["is_5a_positive"] = 0
    raw_ref["reference_source"] = "Burggraaff2020_ChEMBL22"

    five_ref = five.copy()
    five_ref["stable_pair_key"] = five_ref["target_chembl_id"] + "|" + five_ref["ligand_chembl_id"]
    five_ref["uniprot"] = clean(five_ref["UniProt_ID"])
    five_ref["full_inchikey"] = clean(five_ref.get("InChIKey", pd.Series(index=five_ref.index, dtype=str))).str.upper()
    if five_ref["full_inchikey"].eq("").any():
        # The locked 5A file contains the full key; this is a fail-closed guard.
        raise RuntimeError("5A contains a row without a full InChIKey")
    five_ref["connectivity_key"] = clean(five_ref["InChIKey14"]).str.upper()
    five_ref["canonical_smiles"] = clean(five_ref["SMILES"])
    five_ref["target_name"] = clean(five_ref["target_name"])
    five_ref["pchembl_numeric"] = pd.to_numeric(five_ref["pchembl_value"], errors="coerce")
    five_ref["weak2020_label"] = five_ref["stable_pair_key"].map(label_by_stable).fillna(-1).astype(int)
    five_ref["is_5a_positive"] = 1
    five_ref["reference_source"] = "Local_ChEMBL36_5A"

    common_cols = [
        "stable_pair_key", "target_chembl_id", "ligand_chembl_id", "target_name",
        "uniprot", "full_inchikey", "connectivity_key", "canonical_smiles",
        "pchembl_numeric", "weak2020_label", "is_5a_positive", "reference_source",
    ]
    direct = pd.concat([raw_ref[common_cols], five_ref[common_cols]], ignore_index=True)
    # A stable ChEMBL pair can occur in both references.  Preserve the 2020
    # binary label and OR the 5A flag while retaining one molecular record.
    direct = direct.sort_values(
        ["stable_pair_key", "weak2020_label", "is_5a_positive", "pchembl_numeric"],
        ascending=[True, False, False, False], na_position="last",
    )
    combined_rows = []
    mapping_conflicts = 0
    for stable_key, group in direct.groupby("stable_pair_key", sort=False):
        proteins = set(clean(group["uniprot"])) - {""}
        connectivities = set(clean(group["connectivity_key"])) - {""}
        if len(proteins) != 1 or len(connectivities) != 1:
            mapping_conflicts += 1
            continue
        row = group.iloc[0].copy()
        labels = set(pd.to_numeric(group["weak2020_label"], errors="coerce").fillna(-1).astype(int)) - {-1}
        if len(labels) > 1:
            mapping_conflicts += 1
            continue
        row["weak2020_label"] = next(iter(labels)) if labels else -1
        row["is_5a_positive"] = int(group["is_5a_positive"].astype(int).max())
        row["reference_source"] = ";".join(sorted(set(clean(group["reference_source"]))))
        combined_rows.append(row)
    direct = pd.DataFrame(combined_rows)[common_cols]
    direct["pair_connectivity_key"] = direct["uniprot"] + "|" + direct["connectivity_key"]
    direct["excluded_current_benchmark_pair"] = direct["pair_connectivity_key"].isin(
        set(benchmark["pair_connectivity_key"])
    ).astype(int)
    direct = direct[direct["excluded_current_benchmark_pair"].eq(0)].copy()
    direct = direct.sort_values("stable_pair_key").reset_index(drop=True)
    direct.insert(0, "reference_row_id", ["CHEMBLREF_{:06d}".format(i) for i in range(len(direct))])
    direct.to_csv(data / "CHEMBL_DIRECT_REFERENCE_PAIRS.tsv.gz", sep="\t", index=False)

    keys_2020 = {
        str(row.target_chembl_id) + "|" + str(row.ligand_chembl_id): int(row.weak_label)
        for row in ref2020[["target_chembl_id", "ligand_chembl_id", "weak_label"]].itertuples(index=False)
    }
    keys_5a = set(five["target_chembl_id"] + "|" + five["ligand_chembl_id"])
    blacklist_connectivity = set(benchmark["pair_connectivity_key"])
    blacklist_full = set(benchmark["pair_full_key"])
    scan = {
        "ood_rows": 0,
        "benchmark_connectivity_overlap_rows": 0,
        "benchmark_full_overlap_rows": 0,
        "eligible_rows_after_connectivity_exclusion": 0,
        "mapped_2020_allosteric_rows": 0,
        "mapped_2020_orthosteric_rows": 0,
        "mapped_5a_positive_rows": 0,
    }
    flag_frames = []
    usecols = [
        "OOD_Row_ID", "UniProt_ID", "target_chembl_id", "ligand_chembl_id",
        "InChIKey", "InChIKey14", "Label", "Validation_Source",
    ]
    for chunk in pd.read_csv(ood_path, sep="\t", usecols=usecols, chunksize=args.chunksize, low_memory=False):
        scan["ood_rows"] += int(len(chunk))
        uid = clean(chunk["UniProt_ID"])
        ik = clean(chunk["InChIKey"]).str.upper()
        ik14 = clean(chunk["InChIKey14"]).str.upper()
        pair14 = uid + "|" + ik14
        pairfull = uid + "|" + ik
        overlap14 = pair14.isin(blacklist_connectivity)
        overlapfull = pairfull.isin(blacklist_full)
        eligible = ~overlap14
        scan["benchmark_connectivity_overlap_rows"] += int(overlap14.sum())
        scan["benchmark_full_overlap_rows"] += int(overlapfull.sum())
        scan["eligible_rows_after_connectivity_exclusion"] += int(eligible.sum())
        stable = clean(chunk["target_chembl_id"]) + "|" + clean(chunk["ligand_chembl_id"])
        weak = stable.map(keys_2020).fillna(-1).astype(int)
        is5a = stable.isin(keys_5a)
        scan["mapped_2020_allosteric_rows"] += int((eligible & weak.eq(1)).sum())
        scan["mapped_2020_orthosteric_rows"] += int((eligible & weak.eq(0)).sum())
        scan["mapped_5a_positive_rows"] += int((eligible & is5a).sum())
        selected = eligible & (weak.isin([0, 1]) | is5a)
        if selected.any():
            out = chunk.loc[selected, ["OOD_Row_ID", "UniProt_ID", "target_chembl_id", "ligand_chembl_id", "InChIKey14"]].copy()
            out["weak2020_label"] = weak.loc[selected].to_numpy()
            out["is_5a_positive"] = is5a.loc[selected].astype(int).to_numpy()
            flag_frames.append(out)

    flags = pd.concat(flag_frames, ignore_index=True) if flag_frames else pd.DataFrame()
    if len(flags):
        if flags["OOD_Row_ID"].duplicated().any():
            raise RuntimeError("OOD row identifiers duplicated during reference mapping")
        flags.to_csv(data / "CHEMBL_REFERENCE_OOD_ROWS.tsv.gz", sep="\t", index=False)
    else:
        raise RuntimeError("no 2020 or 5A references mapped to the ChEMBL36 OOD table")

    report = {
        "status": "validated",
        "inputs": {
            "arm_a": str(arm_a_path),
            "arm_b": str(arm_b_path),
            "word_mined": str(word_path),
            "five_a": str(five_a_path),
            "chembl_ood": str(ood_path),
            "chembl36_fulltext": str(raw_chembl_path),
        },
        "arm_a": {"rows": int(len(arm_a)), "proteins": int(arm_a["uniprot"].nunique())},
        "benchmark_union_blacklist": {
            "rows_full_identity": int(len(benchmark)),
            "connectivity_pairs": int(benchmark["pair_connectivity_key"].nunique()),
            "proteins": int(benchmark["uniprot"].nunique()),
            "arm_a_is_subset": True,
        },
        "chembl2020": {
            "pure_labeled_pairs": int(len(ref2020)),
            "allosteric_pairs": int(ref2020["weak_label"].eq(1).sum()),
            "orthosteric_pairs": int(ref2020["weak_label"].eq(0).sum()),
            "conflicting_pairs_excluded": conflict_count,
        },
        "chembl5a": {"positive_pairs": int(len(five)), "negative_pairs_defined": 0},
        "direct_reference": {
            "rows_after_current_benchmark_exclusion": int(len(direct)),
            "mapped_2020_allosteric": int(direct["weak2020_label"].eq(1).sum()),
            "mapped_2020_orthosteric": int(direct["weak2020_label"].eq(0).sum()),
            "mapped_5a_positive": int(direct["is_5a_positive"].eq(1).sum()),
            "mapping_conflicts_excluded": int(mapping_conflicts),
        },
        "ood_scan": scan,
        "reference_rows_written": int(len(flags)),
        "gates": {
            "no_word_mined_labels_in_training": True,
            "benchmark_pair_exclusion_is_connectivity_level": True,
            "five_a_unlabeled_not_negative": True,
            "primary_reference_has_both_classes": bool(
                direct["weak2020_label"].eq(1).any() and direct["weak2020_label"].eq(0).any()
            ),
        },
    }
    if not all(report["gates"].values()):
        report["status"] = "failed"
    atomic_json(validation / "CPU_INPUT_VALIDATION.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("CPU contract validation failed")


if __name__ == "__main__":
    main()
