#!/usr/bin/env python3
"""Build structure-matched ChEMBL target chains and exact pocket indices.

This is an outcome-blind extension of the frozen target-chain contract.  It
imports the chain parsing, fpocket ranking, local alignment, chain selection,
and threshold rules from ``build_aligned_target_chains.py``.  It does not scan
the fpocket tree: each coordinate directory is addressed only through the
frozen one-row-per-UniProt ``fpocket_Map.tsv``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery")
    )
    parser.add_argument(
        "--shared-root", type=Path, default=Path("/shared_data/11.HS_allostery")
    )
    parser.add_argument(
        "--remote-root", type=Path, default=Path("/disk1/11.HS_allostery")
    )
    parser.add_argument("--minimum-identity", type=float, default=0.70)
    parser.add_argument("--minimum-chain-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-aligned-residues", type=int, default=30)
    parser.add_argument("--minimum-pocket-residues", type=int, default=5)
    parser.add_argument("--max-pockets", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_helper(root):
    path = (
        root
        / "analysis/allosteric_pair_benchmark_main/scripts/"
        "build_aligned_target_chains.py"
    )
    spec = importlib.util.spec_from_file_location("frozen_chain_builder", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def atomic_tsv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(str(temporary), str(path))


def normalize_uid(value):
    return str(value).strip().split("-")[0]


def full_screen_proteins(cache_root):
    proteins = set()
    paths = []
    for shard in range(4):
        path = cache_root / (
            "ood_cache_mapping_chembl_ood_validation_set.limit0."
            "shard{:02d}of04.json".format(shard)
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        proteins.update(normalize_uid(value) for value in payload["uid_to_idx"])
        paths.append(path)
    return proteins, paths


def load_sequence_caches(root, shared):
    paths = [
        root
        / "analysis/allosteric_pair_benchmark_broad_superset/data/"
        "UNIPROT_SEQUENCE_CACHE.json",
        root
        / "analysis/allosteric_pair_benchmark_main/data/"
        "UNIPROT_SEQUENCE_CACHE.json",
        shared / "14.Organized_input/uniprot_protein_cache.json",
    ]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths], paths


def resolve_sequence(uid, caches, helper):
    broad, main, local = caches
    for name, cache, field in [
        ("frozen_broad_sequence_cache", broad, "sequence"),
        ("frozen_main_sequence_cache", main, "sequence"),
        ("local_uniprot_cache", local, "seq"),
    ]:
        record = cache.get(uid, {})
        if isinstance(record, dict):
            sequence = helper.normalize_sequence(record.get(field, ""))
            if sequence:
                return sequence, name
    return "", ""


def summarize_scope(audit, membership_column):
    scope = audit[audit[membership_column].eq(1)].copy()
    reasons = (
        scope.loc[scope["status"].ne("ready"), "reason"]
        .value_counts()
        .astype(int)
        .to_dict()
    )
    return {
        "proteins": int(len(scope)),
        "ready_proteins": int(scope["status"].eq("ready").sum()),
        "excluded_proteins": int(scope["status"].ne("ready").sum()),
        "pocket_coverage_fraction": float(scope["status"].eq("ready").mean()),
        "exclusion_reasons": reasons,
    }


def validate_existing(args, output_root, expected_sets):
    table_path = output_root / "TARGET_CHAIN_SEQUENCES.tsv"
    masks_path = output_root / "POCKET_INDICES.json"
    audit_path = output_root / "POCKET_ALIGNMENT_AUDIT.tsv"
    coverage_path = output_root / "POCKET_COVERAGE.json"
    table = pd.read_csv(table_path, sep="\t", low_memory=False)
    audit = pd.read_csv(audit_path, sep="\t", low_memory=False)
    masks = json.loads(masks_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    universe, reference, primary, full = expected_sets
    checks = {
        "coverage_status_pass": coverage.get("status") == "PASS",
        "audit_exact_universe": set(audit["uniprot"].astype(str)) == universe,
        "audit_unique_uniprot": not audit["uniprot"].duplicated().any(),
        "table_unique_uniprot": not table["uniprot"].duplicated().any(),
        "table_equals_ready_audit": (
            set(table["uniprot"].astype(str))
            == set(audit.loc[audit["status"].eq("ready"), "uniprot"].astype(str))
        ),
        "mask_keys_equal_table": set(masks) == set(table["uniprot"].astype(str)),
        "all_masks_nonempty": all(bool(values) for values in masks.values()),
        "all_indices_in_range": all(
            min(map(int, masks[str(row.uniprot)])) >= 0
            and max(map(int, masks[str(row.uniprot)])) < int(row.target_chain_length)
            for row in table.itertuples(index=False)
        ),
        "reference_count": int(audit["in_reference"].sum()) == len(reference),
        "primary_count": int(audit["in_two_label_primary"].sum()) == len(primary),
        "full_count": int(audit["in_full_screen"].sum()) == len(full),
        "full_missing_pdb_zero": coverage["full_screen"][
            "pdb_structure_absent_proteins"
        ] == 0,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "target_chain_rows": int(len(table)),
        "mask_proteins": int(len(masks)),
        "coverage": coverage,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


def main():
    args = parse_args()
    root = args.project_root.resolve()
    shared = args.shared_root.resolve()
    package = root / "analysis/property_balanced_chembl"
    output_root = package / "data/chembl_pocket_extension"
    fasta_root = output_root / "target_chain_fasta"
    reference_path = (
        root
        / "analysis/chembl_external_prioritization/gpu_cache/"
        "REFERENCE_MODEL_READY.tsv.gz"
    )
    cache_relative = Path(
        "19.Structure_Unknown/6.Build_Fully_Controlled_Model_Dataset_OODTrain/"
        "9.Train_Fully_Controlled_Embedding_Model_OODTrain/ood_tensor_cache"
    )
    cache_candidates = [shared / cache_relative, root / cache_relative]
    cache_root = next(
        (
            value for value in cache_candidates
            if (value / (
                "ood_cache_mapping_chembl_ood_validation_set.limit0."
                "shard00of04.json"
            )).is_file()
        ),
        cache_candidates[0],
    )
    fpocket_map_path = shared / "14.Organized_input/fpocket_Map.tsv"
    fpocket_root = shared / "BioLiP_CIFs/fpocket"
    helper, helper_path = load_helper(root)
    reference_frame = pd.read_csv(
        reference_path, sep="\t", usecols=["uniprot", "weak2020_label"]
    )
    reference = {normalize_uid(value) for value in reference_frame["uniprot"]}
    primary = {
        normalize_uid(value)
        for value in reference_frame.loc[
            reference_frame["weak2020_label"].isin([0, 1]), "uniprot"
        ]
    }
    full, mapping_paths = full_screen_proteins(cache_root)
    universe = reference | full
    expected_sets = universe, reference, primary, full
    if args.validate_only:
        validate_existing(args, output_root, expected_sets)
        return

    fpocket_map = pd.read_csv(fpocket_map_path, sep="\t", dtype=str).fillna("")
    fpocket_map["uid"] = fpocket_map["UniProt_ID"].map(normalize_uid)
    if fpocket_map["uid"].duplicated().any():
        raise RuntimeError("fpocket map is not unique by canonical UniProt accession")
    fpocket_map = fpocket_map.set_index("uid")
    sequence_caches, sequence_paths = load_sequence_caches(root, shared)
    broad_table_path = (
        root
        / "analysis/allosteric_pair_benchmark_broad_superset/data/"
        "UNIFIED_TARGET_CHAIN_SEQUENCES.tsv"
    )
    broad_masks_path = (
        root
        / "analysis/allosteric_pair_benchmark_broad_superset/data/POCKET_INDICES.json"
    )
    broad_table = pd.read_csv(broad_table_path, sep="\t", low_memory=False)
    broad_table = broad_table.set_index("uniprot")
    broad_masks = json.loads(broad_masks_path.read_text(encoding="utf-8"))

    fasta_root.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = []
    masks = {}
    for position, uid in enumerate(sorted(universe), 1):
        in_reference = int(uid in reference)
        in_primary = int(uid in primary)
        in_full = int(uid in full)
        status = "ready"
        reason = ""
        canonical, sequence_source = resolve_sequence(uid, sequence_caches, helper)
        mapped = fpocket_map.loc[uid] if uid in fpocket_map.index else None
        pdb_id = str(mapped["Mapped_PDB"]).strip().lower()[:4] if mapped is not None else ""
        directory = fpocket_root / (pdb_id + "_out") if pdb_id else Path("/")
        coordinate_path = directory / (pdb_id + "_out.cif") if pdb_id else Path("/")
        selected = None
        selected_pockets = []
        n_pocket_files = 0
        n_retained_pocket_files = 0
        n_chains = 0
        if mapped is None:
            status, reason = "excluded", "fpocket_mapping_missing"
        elif not pdb_id:
            status, reason = "excluded", "fpocket_mapping_missing_pdb"
        elif not coordinate_path.is_file():
            status, reason = "excluded", "fpocket_coordinate_file_missing"
        elif not canonical:
            status, reason = "excluded", "canonical_sequence_missing"
        else:
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
                    status, reason = (
                        "excluded",
                        "target_chain_alignment_identity_below_threshold",
                    )
                elif selected["aligned"] < args.minimum_aligned_residues:
                    status, reason = "excluded", "target_chain_alignment_too_short"
                elif selected["chain_coverage"] < args.minimum_chain_coverage:
                    status, reason = (
                        "excluded",
                        "target_chain_alignment_coverage_below_threshold",
                    )
                elif selected["n_pocket_residues"] < args.minimum_pocket_residues:
                    status, reason = "excluded", "too_few_aligned_pocket_residues"
            except Exception as error:
                status, reason = "excluded", "{}:{}".format(type(error).__name__, error)
        record = {
            "uniprot": uid,
            "in_reference": in_reference,
            "in_two_label_primary": in_primary,
            "in_full_screen": in_full,
            "status": status,
            "reason": reason,
            "mapping_origin": "frozen_fpocket_map" if mapped is not None else "unmapped",
            "sequence_source": sequence_source,
            "canonical_length": len(canonical),
            "mapped_pdb": pdb_id,
            "fpocket_coordinate_file": str(coordinate_path) if pdb_id else "",
            "n_structure_chains": n_chains,
            "n_pocket_files": n_pocket_files,
            "n_retained_pocket_files": n_retained_pocket_files,
            "max_selected_pockets": int(args.max_pockets),
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
                "pocket_fraction": (
                    selected["n_pocket_residues"]
                    / max(len(selected["sequence"]), 1)
                ),
                "selected_fpocket_ids": ";".join(
                    str(value["pocket_id"]) for value in selected_pockets
                ),
            })
        audit.append(record)
        if status == "ready":
            sequence = selected["sequence"]
            indices = [int(value) for value in selected["pocket_index"]]
            fasta_path = fasta_root / (uid + ".fasta")
            helper.write_fasta(
                fasta_path, uid, pdb_id, selected["chain"], sequence
            )
            masks[uid] = indices
            rows.append({
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
                    args.remote_root
                    / "analysis/property_balanced_chembl/gpu_cache/"
                    "chembl_target_chain_embeddings"
                    / (uid + ".pt")
                ),
            })
        if position % 50 == 0 or position == len(universe):
            print(
                "ChEMBL target chains {}/{} ready={}".format(
                    position, len(universe), len(rows)
                ),
                flush=True,
            )

    table = pd.DataFrame(rows).sort_values("uniprot").reset_index(drop=True)
    audit_frame = pd.DataFrame(audit).sort_values("uniprot").reset_index(drop=True)
    # Every target already frozen in the broad builder must reproduce exactly.
    shared_ready = sorted(set(masks) & set(broad_masks))
    broad_mismatches = []
    table_by_uid = table.set_index("uniprot")
    for uid in shared_ready:
        old = broad_table.loc[uid]
        new = table_by_uid.loc[uid]
        if (
            list(map(int, masks[uid])) != list(map(int, broad_masks[uid]))
            or str(old["mapped_pdb"]) != str(new["mapped_pdb"])
            or str(old["selected_chain"]) != str(new["selected_chain"])
            or str(old["sequence_sha256"]) != str(new["sequence_sha256"])
        ):
            broad_mismatches.append(uid)
    if broad_mismatches:
        raise RuntimeError(
            "frozen broad target-chain reproduction failed: {}".format(
                broad_mismatches[:10]
            )
        )

    full_summary = summarize_scope(audit_frame, "in_full_screen")
    full_summary["pdb_structure_absent_proteins"] = int(
        audit_frame.loc[
            audit_frame["in_full_screen"].eq(1), "reason"
        ].isin([
            "fpocket_mapping_missing",
            "fpocket_mapping_missing_pdb",
            "fpocket_coordinate_file_missing",
        ]).sum()
    )
    coverage = {
        "status": "PASS",
        "contract_id": "property_balanced_chembl_c3_pocket_extension_v1",
        "coordinate_contract": (
            "POCKET_INDICES.json is zero-based into the exact selected-chain "
            "sequence in TARGET_CHAIN_SEQUENCES.tsv; C3 must use newly generated "
            "ESM3 tensors from those FASTAs, not the legacy full-screen protein tensor."
        ),
        "selection_contract": {
            "pdb_mapping": "frozen one-row-per-UniProt fpocket_Map.tsv only",
            "chain_and_pocket_rules_imported_from": str(helper_path.relative_to(root)),
            "minimum_identity": float(args.minimum_identity),
            "minimum_chain_coverage": float(args.minimum_chain_coverage),
            "minimum_aligned_residues": int(args.minimum_aligned_residues),
            "minimum_pocket_residues": int(args.minimum_pocket_residues),
            "max_pockets": int(args.max_pockets),
        },
        "universe": {
            "union_proteins": len(universe),
            "reference_proteins": len(reference),
            "two_label_primary_proteins": len(primary),
            "full_screen_proteins": len(full),
        },
        "reference": summarize_scope(audit_frame, "in_reference"),
        "two_label_primary": summarize_scope(
            audit_frame, "in_two_label_primary"
        ),
        "full_screen": full_summary,
        "union": {
            "proteins": len(universe),
            "ready_proteins": int(audit_frame["status"].eq("ready").sum()),
            "excluded_proteins": int(audit_frame["status"].ne("ready").sum()),
        },
        "frozen_broad_reproduction": {
            "shared_ready_proteins": len(shared_ready),
            "mismatches": len(broad_mismatches),
        },
        "input_sha256": {
            "reference_model_ready": sha256_file(reference_path),
            "fpocket_map": sha256_file(fpocket_map_path),
            "broad_target_chains": sha256_file(broad_table_path),
            "broad_pocket_indices": sha256_file(broad_masks_path),
            **{
                "full_mapping_shard_{:02d}".format(index): sha256_file(path)
                for index, path in enumerate(mapping_paths)
            },
            **{
                "sequence_cache_{:02d}".format(index): sha256_file(path)
                for index, path in enumerate(sequence_paths)
            },
        },
    }
    atomic_tsv(output_root / "TARGET_CHAIN_SEQUENCES.tsv", table)
    atomic_json(output_root / "POCKET_INDICES.json", masks)
    atomic_tsv(output_root / "POCKET_ALIGNMENT_AUDIT.tsv", audit_frame)
    atomic_json(output_root / "POCKET_COVERAGE.json", coverage)
    validate_existing(args, output_root, expected_sets)


if __name__ == "__main__":
    main()
