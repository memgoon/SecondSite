#!/usr/bin/env python3
"""Build Arm B as a strict superset of the frozen Arm A benchmark.

Only explicitly named tables and fpocket directories are accessed.  The
project tree is never searched recursively.  Frozen Arm A target chains and
pocket masks are reused byte-for-byte; additional proteins are processed with
the same graph-free target-chain alignment contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests


CLASS_TO_BINARY = {"orthosteric": 0, "allosteric": 1}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    parser.add_argument("--shared-root", type=Path, default=Path("/shared_data/11.HS_allostery"))
    parser.add_argument("--remote-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--minimum-identity", type=float, default=0.70)
    parser.add_argument("--minimum-chain-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-aligned-residues", type=int, default=30)
    parser.add_argument("--minimum-pocket-residues", type=int, default=5)
    parser.add_argument("--max-pockets", type=int, default=5)
    parser.add_argument("--api-delay", type=float, default=0.05)
    return parser.parse_args()


def load_alignment_helper(project_root):
    path = project_root / "analysis/allosteric_pair_benchmark_main/scripts/build_aligned_target_chains.py"
    spec = importlib.util.spec_from_file_location("frozen_target_chain_alignment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*values):
    payload = "|".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20].upper()


def normalize_uid(value):
    return str(value).strip().split("-")[0]


def normalize_key(value):
    return str(value).strip().replace("InChIKey=", "").upper()


def parse_pfams(value):
    return sorted({token.strip() for token in str(value).split(";") if token.strip().startswith("PF")})


def exact_pair_series(frame):
    return frame["uniprot"].astype(str) + "|" + frame["full_inchikey"].astype(str)


def write_fasta(path, uid, sequence):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(">{}\n{}\n".format(uid, sequence), encoding="ascii")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    root = args.project_root
    shared = args.shared_root
    remote = args.remote_root
    package = root / "analysis/allosteric_pair_benchmark_broad_superset"
    main_package = root / "analysis/allosteric_pair_benchmark_main"
    raw_path = (
        root / "analysis/allosteric_orthosteric_preprocessing_v1_participant_kegg_expansion/data/"
        "UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC_PARTICIPANT_KEGG_AUGMENTED.tsv.gz"
    )
    exclusions_path = (
        root / "analysis/allosteric_orthosteric_preprocessing_v1_participant_kegg_expansion/data/"
        "AUDIT_EXCLUDED_BIOLIP_ROWS.tsv"
    )
    core_path = main_package / "gpu_cache/MODEL_READY.tsv.gz"
    core_target_path = main_package / "data/TARGET_CHAIN_SEQUENCES.tsv"
    core_masks_path = main_package / "data/POCKET_INDICES.json"
    core_sequence_cache_path = main_package / "data/UNIPROT_SEQUENCE_CACHE.json"
    reference_predictions_path = (
        main_package / "gpu_output/main_benchmark/aggregate/ALL_OOF_PREDICTIONS.tsv.gz"
    )
    reference_training_index_path = main_package / "gpu_output/main_benchmark/TRAINING_INDEX.json"
    fpocket_map_path = shared / "14.Organized_input/fpocket_Map.tsv"
    local_sequence_path = shared / "14.Organized_input/uniprot_protein_cache.json"
    pfam_path = shared / "14.Organized_input/uniprot_pfam_cache.json"
    fpocket_root = shared / "BioLiP_CIFs/fpocket"
    helper, helper_path = load_alignment_helper(root)
    required_paths = [
        raw_path, exclusions_path, core_path, core_target_path, core_masks_path,
        core_sequence_cache_path, reference_predictions_path, reference_training_index_path,
        fpocket_map_path, local_sequence_path, pfam_path, helper_path,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = pd.read_csv(raw_path, sep="\t", low_memory=False)
    exclusions = pd.read_csv(exclusions_path, sep="\t", dtype=str).fillna("")
    raw["uniprot"] = raw["uniprot"].map(normalize_uid)
    raw["full_inchikey"] = raw["full_inchikey"].map(normalize_key)
    raw["connectivity_key"] = raw["full_inchikey"].str[:14]
    exclusions["uniprot"] = exclusions["uniprot"].map(normalize_uid)
    exclusions["full_inchikey"] = exclusions["full_inchikey"].map(normalize_key)
    expected_binary = raw["class_label"].map(CLASS_TO_BINARY)
    label_mismatches = int((pd.to_numeric(raw["binary_label"], errors="coerce") != expected_binary).sum())
    if label_mismatches:
        raise RuntimeError("raw class_label/binary_label mismatch: {} rows".format(label_mismatches))
    exclusion_keys = set(zip(exclusions["uniprot"], exclusions["full_inchikey"]))
    raw_pairs = list(zip(raw["uniprot"], raw["full_inchikey"]))
    cleaned = raw[~pd.Series(raw_pairs, index=raw.index).isin(exclusion_keys)].copy()
    conflicts = cleaned.groupby(["uniprot", "full_inchikey"])["class_label"].nunique()
    if int(conflicts.gt(1).sum()):
        raise RuntimeError("strict-clean source contains cross-label exact-pair conflicts")
    cleaned = cleaned.drop_duplicates(["uniprot", "full_inchikey"], keep="first").copy()
    cleaned["binary_label"] = cleaned["class_label"].map(CLASS_TO_BINARY).astype(int)
    cleaned["pair_key"] = exact_pair_series(cleaned)
    if len(cleaned) != 20640:
        raise RuntimeError("strict-clean source count changed: {}".format(len(cleaned)))

    core = pd.read_csv(core_path, sep="\t", low_memory=False)
    core["uniprot"] = core["uniprot"].map(normalize_uid)
    core["full_inchikey"] = core["full_inchikey"].map(normalize_key)
    core["connectivity_key"] = core["full_inchikey"].str[:14]
    core["pair_key"] = exact_pair_series(core)
    core["binary_label"] = core["binary_label"].astype(int)
    if core["pair_key"].duplicated().any() or len(core) != 4637:
        raise RuntimeError("frozen Arm A identity/count contract changed")
    if not core.groupby("uniprot")["binary_label"].nunique().eq(2).all():
        raise RuntimeError("frozen Arm A no longer has both labels per protein")

    fpocket_map = pd.read_csv(fpocket_map_path, sep="\t", dtype=str).fillna("")
    fpocket_map["uid"] = fpocket_map["UniProt_ID"].map(normalize_uid)
    if fpocket_map["uid"].duplicated().any():
        raise RuntimeError("fpocket map is not unique by canonical UniProt accession")
    fpocket_map = fpocket_map.set_index("uid")
    candidate_proteins = sorted(set(cleaned["uniprot"]) & set(fpocket_map.index))
    core_proteins = set(core["uniprot"].astype(str))
    if not core_proteins.issubset(candidate_proteins):
        raise RuntimeError("raw mapped candidate universe does not contain every Arm A protein")

    core_targets = pd.read_csv(core_target_path, sep="\t", low_memory=False)
    core_targets["uniprot"] = core_targets["uniprot"].map(normalize_uid)
    core_targets = core_targets[core_targets["uniprot"].isin(core_proteins)].copy()
    if core_targets["uniprot"].nunique() != len(core_proteins):
        raise RuntimeError("frozen target-chain table does not cover final Arm A proteins")
    with core_masks_path.open(encoding="utf-8") as handle:
        core_masks = json.load(handle)
    with core_sequence_cache_path.open(encoding="utf-8") as handle:
        frozen_sequences = json.load(handle)
    with local_sequence_path.open(encoding="utf-8") as handle:
        local_sequences = json.load(handle)
    with pfam_path.open(encoding="utf-8") as handle:
        pfam_cache = json.load(handle)

    data_dir = package / "data"
    validation_dir = package / "validation"
    extra_fasta_root = data_dir / "extra_target_chain_fasta"
    data_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    extra_fasta_root.mkdir(parents=True, exist_ok=True)

    target_records = []
    extra_target_records = []
    audit = []
    masks = {}
    core_target_by_uid = core_targets.set_index("uniprot")
    for uid in sorted(core_proteins):
        row = core_target_by_uid.loc[uid].to_dict()
        row["uniprot"] = uid
        row["embedding_origin"] = "frozen_arm_a_reuse"
        row["gpu_embedding_path"] = str(
            remote / "analysis/allosteric_pair_benchmark_main/gpu_cache/target_chain_embeddings_v2" / (uid + ".pt")
        )
        target_records.append(row)
        mask = [int(value) for value in core_masks.get(uid, [])]
        if not mask:
            raise RuntimeError("missing frozen Arm A mask for {}".format(uid))
        masks[uid] = mask
        audit.append({
            "uniprot": uid,
            "status": "ready",
            "reason": "",
            "origin": "frozen_arm_a_reuse",
            "mapped_pdb": row.get("mapped_pdb", ""),
            "selected_chain": row.get("selected_chain", ""),
            "target_chain_length": int(row.get("target_chain_length", 0)),
            "pocket_residues": len(mask),
        })

    session = requests.Session()
    extra_candidates = [uid for uid in candidate_proteins if uid not in core_proteins]
    for position, uid in enumerate(extra_candidates, 1):
        status = "ready"
        reason = ""
        sequence_source = ""
        cached = frozen_sequences.get(uid, {})
        canonical = helper.normalize_sequence(cached.get("sequence", "") if isinstance(cached, dict) else "")
        if canonical:
            sequence_source = cached.get("source", "frozen_cache")
        if not canonical:
            local = local_sequences.get(uid, {})
            canonical = helper.normalize_sequence(local.get("seq", "") if isinstance(local, dict) else "")
            if canonical:
                sequence_source = "local_uniprot_cache"
        if not canonical:
            canonical, error = helper.fetch_uniprot_sequence(uid, session)
            sequence_source = "uniprot_rest" if canonical else ""
            if not canonical:
                status, reason = "excluded", "canonical_sequence_missing:{}".format(error)
            time.sleep(max(args.api_delay, 0.0))
        if canonical:
            frozen_sequences[uid] = {
                "sequence": canonical,
                "sequence_sha256": helper.sha256_text(canonical),
                "source": sequence_source,
            }

        mapped = fpocket_map.loc[uid]
        pdb_id = str(mapped["Mapped_PDB"]).strip().lower()[:4]
        directory = fpocket_root / (pdb_id + "_out") if pdb_id else Path("/")
        coordinate_path = directory / (pdb_id + "_out.cif") if pdb_id else Path("/")
        selected = None
        selected_pockets = []
        n_pocket_files = 0
        n_retained_pocket_files = 0
        n_chains = 0
        if status == "ready" and not pdb_id:
            status, reason = "excluded", "fpocket_mapping_missing_pdb"
        elif status == "ready" and not coordinate_path.is_file():
            status, reason = "excluded", "fpocket_coordinate_file_missing"
        elif status == "ready":
            try:
                chains = helper.parse_structure_chains(coordinate_path)
                pocket_residues, n_pocket_files, n_retained_pocket_files, selected_pockets = (
                    helper.parse_pocket_residues(directory, args.max_pockets)
                )
                n_chains = len(chains)
                selected = helper.choose_chain(canonical, chains, pocket_residues)
                if selected is None:
                    status, reason = "excluded", "no_protein_chain"
                elif selected["identity"] < args.minimum_identity:
                    status, reason = "excluded", "target_chain_alignment_identity_below_threshold"
                elif selected["aligned"] < args.minimum_aligned_residues:
                    status, reason = "excluded", "target_chain_alignment_too_short"
                elif selected["chain_coverage"] < args.minimum_chain_coverage:
                    status, reason = "excluded", "target_chain_alignment_coverage_below_threshold"
                elif selected["n_pocket_residues"] < args.minimum_pocket_residues:
                    status, reason = "excluded", "too_few_aligned_pocket_residues"
            except Exception as error:
                status, reason = "excluded", "{}:{}".format(type(error).__name__, error)

        record = {
            "uniprot": uid,
            "status": status,
            "reason": reason,
            "origin": "new_broad_alignment",
            "sequence_source": sequence_source,
            "canonical_length": len(canonical),
            "mapped_pdb": pdb_id,
            "fpocket_coordinate_file": str(coordinate_path) if pdb_id else "",
            "n_structure_chains": n_chains,
            "n_pocket_files": n_pocket_files,
            "n_retained_pocket_files": n_retained_pocket_files,
        }
        if selected is not None:
            record.update({
                "selected_chain": selected["chain"],
                "target_chain_length": len(selected["sequence"]),
                "alignment_matches": selected["matches"],
                "alignment_residues": selected["aligned"],
                "alignment_identity": selected["identity"],
                "target_chain_coverage": selected["chain_coverage"],
                "pocket_residues": selected["n_pocket_residues"],
            })
        audit.append(record)
        if status == "ready":
            sequence = selected["sequence"]
            indices = [int(value) for value in selected["pocket_index"]]
            fasta_path = extra_fasta_root / (uid + ".fasta")
            write_fasta(fasta_path, uid, sequence)
            masks[uid] = indices
            target = {
                "uniprot": uid,
                "mapped_pdb": pdb_id,
                "selected_chain": selected["chain"],
                "target_chain_sequence": sequence,
                "target_chain_length": len(sequence),
                "sequence_sha256": helper.sha256_text(sequence),
                "pocket_residues": len(indices),
                "pocket_fraction": len(indices) / max(len(sequence), 1),
                "fasta_path": str(fasta_path),
                "gpu_embedding_path": str(
                    remote / "analysis/allosteric_pair_benchmark_broad_superset/"
                    "gpu_cache/extra_target_chain_embeddings" / (uid + ".pt")
                ),
                "embedding_origin": "new_broad_alignment",
            }
            target_records.append(target)
            extra_target_records.append(target)
        if position % 50 == 0 or position == len(extra_candidates):
            print(
                "broad target-chain alignment {}/{} newly_ready={}".format(
                    position, len(extra_candidates), len(extra_target_records)
                ),
                flush=True,
            )

    audit_frame = pd.DataFrame(audit)
    target_frame = pd.DataFrame(target_records).sort_values("uniprot").reset_index(drop=True)
    extra_target_frame = pd.DataFrame(extra_target_records).sort_values("uniprot").reset_index(drop=True)
    ready_proteins = set(target_frame["uniprot"].astype(str))
    if not core_proteins.issubset(ready_proteins):
        raise RuntimeError("target-chain preparation lost an Arm A protein")

    pfams = {uid: parse_pfams(pfam_cache.get(uid, "")) for uid in ready_proteins}
    target_by_uid = target_frame.set_index("uniprot")
    core_pair_keys = set(core["pair_key"].astype(str))
    extra = cleaned[
        cleaned["uniprot"].isin(ready_proteins) & ~cleaned["pair_key"].isin(core_pair_keys)
    ].copy()
    extra["main_row_id"] = extra.apply(
        lambda row: "BROAD_" + stable_id(row["uniprot"], row["full_inchikey"]), axis=1
    )
    extra["pool_membership"] = "additional_input_ready_pair"
    extra["pfam_accessions"] = extra["uniprot"].map(lambda uid: ";".join(pfams.get(uid, [])))
    core_component_by_uid = core.drop_duplicates("uniprot").set_index("uniprot")["family_component_id"].to_dict()
    extra["family_component_id"] = extra["uniprot"].map(core_component_by_uid)
    missing_component = extra["family_component_id"].isna()
    extra.loc[missing_component, "family_component_id"] = extra.loc[missing_component, "uniprot"].map(
        lambda uid: "BROAD_" + stable_id("family", ";".join(pfams.get(uid, [])) or uid)
    )
    extra["family_fold"] = -1
    extra["row_fold"] = -1
    extra["protein_embedding_path"] = extra["uniprot"].map(target_by_uid["gpu_embedding_path"])
    extra["target_chain_pdb"] = extra["uniprot"].map(target_by_uid["mapped_pdb"])
    extra["target_chain_id"] = extra["uniprot"].map(target_by_uid["selected_chain"])
    extra["target_chain_length"] = extra["uniprot"].map(target_by_uid["target_chain_length"]).astype(int)
    extra["target_chain_sequence_sha256"] = extra["uniprot"].map(target_by_uid["sequence_sha256"])
    extra["representation_unit"] = "structure_matched_target_chain"
    extra["reader_split_row"] = "training addition only"
    extra["reader_split_family"] = "training addition subject to held-out Pfam exclusion"
    extra["ligand_embedding_path"] = ""

    core["pool_membership"] = "frozen_arm_a_core"
    core["pfam_accessions"] = core["uniprot"].map(lambda uid: ";".join(pfams.get(uid, [])))
    all_columns = list(core.columns) + [column for column in extra.columns if column not in core.columns]
    for column in all_columns:
        if column not in core:
            core[column] = ""
        if column not in extra:
            extra[column] = ""
    pool = pd.concat([core[all_columns], extra[all_columns]], ignore_index=True)
    pool["binary_label"] = pool["binary_label"].astype(int)
    if pool["main_row_id"].duplicated().any() or exact_pair_series(pool).duplicated().any():
        raise RuntimeError("unified broad pool is not unique by row or exact pair")

    core_check = pool[pool["pool_membership"].eq("frozen_arm_a_core")].copy()
    compare_columns = [
        "main_row_id", "pair_key", "binary_label", "class_label", "connectivity_key",
        "protein_embedding_path", "target_chain_sequence_sha256", "representation_unit",
        "family_component_id", "family_fold", "row_fold",
    ]
    left = core[compare_columns].sort_values("main_row_id").reset_index(drop=True).astype(str)
    right = core_check[compare_columns].sort_values("main_row_id").reset_index(drop=True).astype(str)
    if not left.equals(right):
        raise RuntimeError("Arm A metadata changed while constructing Arm B")

    invalid_masks = {}
    for row in target_frame.itertuples(index=False):
        uid = str(row.uniprot)
        values = [int(value) for value in masks.get(uid, [])]
        if (
            not values or min(values) < 0 or len(values) != len(set(values))
            or max(values) >= int(row.target_chain_length)
        ):
            invalid_masks[uid] = values
    if invalid_masks:
        raise RuntimeError("invalid target-chain pocket masks: {} proteins".format(len(invalid_masks)))

    aligned_path = data_dir / "BROAD_ALIGNED_POOL.tsv.gz"
    target_path = data_dir / "UNIFIED_TARGET_CHAIN_SEQUENCES.tsv"
    extra_target_path = data_dir / "EXTRA_TARGET_CHAIN_SEQUENCES.tsv"
    masks_path = data_dir / "POCKET_INDICES.json"
    audit_path = validation_dir / "TARGET_CHAIN_ALIGNMENT_AUDIT.tsv"
    sequence_cache_path = data_dir / "UNIPROT_SEQUENCE_CACHE.json"
    reference_core_path = data_dir / "REFERENCE_ARM_A_MODEL_READY.tsv.gz"
    reference_predictions_copy = data_dir / "REFERENCE_ARM_A_OOF_PREDICTIONS.tsv.gz"
    reference_training_index_copy = data_dir / "REFERENCE_ARM_A_TRAINING_INDEX.json"
    pool.sort_values(["pool_membership", "uniprot", "class_label", "full_inchikey"]).to_csv(
        aligned_path, sep="\t", index=False, compression="gzip"
    )
    target_frame.to_csv(target_path, sep="\t", index=False)
    extra_target_frame.to_csv(extra_target_path, sep="\t", index=False)
    audit_frame.to_csv(audit_path, sep="\t", index=False)
    atomic_json(masks_path, {uid: masks[uid] for uid in sorted(masks)})
    atomic_json(sequence_cache_path, frozen_sequences)
    core.sort_values("main_row_id").to_csv(reference_core_path, sep="\t", index=False, compression="gzip")
    reference_predictions = pd.read_csv(reference_predictions_path, sep="\t", low_memory=False)
    reference_predictions.to_csv(
        reference_predictions_copy, sep="\t", index=False, compression="gzip"
    )
    reference_training_index = json.loads(reference_training_index_path.read_text(encoding="utf-8"))
    atomic_json(reference_training_index_copy, reference_training_index)

    extra_pfams_missing = sorted({uid for uid in set(extra["uniprot"].astype(str)) if not pfams.get(uid)})
    ready_audit = audit_frame[audit_frame["status"].eq("ready")]
    report = {
        "status": "validated",
        "no_recursive_project_scan": True,
        "scientific_contract": "Arm B equals frozen Arm A plus input-ready additional exact pairs",
        "strict_clean": {
            "rows": int(len(cleaned)),
            "proteins": int(cleaned["uniprot"].nunique()),
            "allosteric": int(cleaned["binary_label"].eq(1).sum()),
            "orthosteric": int(cleaned["binary_label"].eq(0).sum()),
            "raw_label_mismatches": int(label_mismatches),
        },
        "mapped_candidate_proteins": int(len(candidate_proteins)),
        "target_chain_ready_proteins": int(len(ready_proteins)),
        "reused_arm_a_proteins": int(len(core_proteins)),
        "new_target_chain_ready_proteins": int(len(extra_target_frame)),
        "target_chain_exclusion_reasons": audit_frame.loc[
            audit_frame["status"].ne("ready"), "reason"
        ].value_counts().astype(int).to_dict(),
        "arm_a": {
            "rows": int(len(core)),
            "proteins": int(core["uniprot"].nunique()),
            "allosteric": int(core["binary_label"].eq(1).sum()),
            "orthosteric": int(core["binary_label"].eq(0).sum()),
        },
        "pre_gpu_arm_b": {
            "rows": int(len(pool)),
            "proteins": int(pool["uniprot"].nunique()),
            "additional_rows": int(len(extra)),
            "additional_proteins_not_in_arm_a": int(len(set(extra["uniprot"]) - core_proteins)),
            "allosteric": int(pool["binary_label"].eq(1).sum()),
            "orthosteric": int(pool["binary_label"].eq(0).sum()),
            "additional_proteins_without_pfam": extra_pfams_missing,
        },
        "alignment_thresholds": {
            "minimum_identity": float(args.minimum_identity),
            "minimum_chain_coverage": float(args.minimum_chain_coverage),
            "minimum_aligned_residues": int(args.minimum_aligned_residues),
            "minimum_pocket_residues": int(args.minimum_pocket_residues),
            "max_pockets": int(args.max_pockets),
        },
        "inclusion_gates": {
            "all_arm_a_rows_present": bool(set(core["main_row_id"]) <= set(pool["main_row_id"])),
            "arm_a_metadata_unchanged": True,
            "class_label_binary_label_consistent": bool(
                (pool["class_label"].map(CLASS_TO_BINARY).astype(int) == pool["binary_label"]).all()
            ),
            "all_masks_valid": not invalid_masks,
        },
        "inputs": {str(path): sha256(path) for path in required_paths},
        "outputs": {
            "broad_aligned_pool": {"path": str(aligned_path), "sha256": sha256(aligned_path)},
            "unified_target_chains": {"path": str(target_path), "sha256": sha256(target_path)},
            "extra_target_chains": {"path": str(extra_target_path), "sha256": sha256(extra_target_path)},
            "pocket_indices": {"path": str(masks_path), "sha256": sha256(masks_path)},
            "reference_arm_a": {"path": str(reference_core_path), "sha256": sha256(reference_core_path)},
            "reference_predictions": {
                "path": str(reference_predictions_copy), "sha256": sha256(reference_predictions_copy)
            },
        },
    }
    conditions = [
        report["inclusion_gates"]["all_arm_a_rows_present"],
        report["inclusion_gates"]["class_label_binary_label_consistent"],
        report["inclusion_gates"]["all_masks_valid"],
        len(extra) > 0,
        set(pool["binary_label"]) == {0, 1},
        int(ready_audit["uniprot"].nunique()) == len(ready_proteins),
    ]
    if not all(conditions):
        report["status"] = "failed"
    report_path = validation_dir / "CPU_INPUT_VALIDATION.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("broad-superset CPU input validation failed")


if __name__ == "__main__":
    main()
