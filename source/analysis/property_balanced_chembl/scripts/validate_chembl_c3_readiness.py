#!/usr/bin/env python3
"""Independently validate ChEMBL C3 pocket-extension readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from pathlib import Path

import pandas as pd


REQUIRED_TARGET_COLUMNS = {
    "uniprot", "mapped_pdb", "selected_chain", "target_chain_sequence",
    "target_chain_length", "sequence_sha256", "pocket_residues",
    "pocket_fraction", "fasta_path", "gpu_embedding_path",
}
REQUIRED_AUDIT_COLUMNS = {
    "uniprot", "in_reference", "in_two_label_primary", "in_full_screen",
    "status", "reason", "mapped_pdb", "selected_chain",
    "target_chain_length", "pocket_residues", "pocket_fraction",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery")
    )
    parser.add_argument(
        "--shared-root", type=Path, default=Path("/shared_data/11.HS_allostery")
    )
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_uid(value):
    return str(value).strip().split("-")[0]


def full_screen_set(root, shared):
    relative = (
        "19.Structure_Unknown/6.Build_Fully_Controlled_Model_Dataset_OODTrain/"
        "9.Train_Fully_Controlled_Embedding_Model_OODTrain/ood_tensor_cache"
    )
    candidates = [root / relative, shared / relative]
    cache = next(
        (
            value for value in candidates
            if (value / (
                "ood_cache_mapping_chembl_ood_validation_set.limit0."
                "shard00of04.json"
            )).is_file()
        ),
        candidates[0],
    )
    result = set()
    for shard in range(4):
        path = cache / (
            "ood_cache_mapping_chembl_ood_validation_set.limit0."
            "shard{:02d}of04.json".format(shard)
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        result.update(canonical_uid(value) for value in payload["uid_to_idx"])
    return result


def read_fasta(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError("invalid FASTA header")
    sequence = "".join(line.strip() for line in lines[1:] if line.strip())
    if not sequence or re.search(r"[^A-Z]", sequence):
        raise ValueError("invalid FASTA sequence")
    return lines[0], sequence


def derive_epoch_audit(root):
    fit_root = root / (
        "analysis/ligand_chemistry_balancing/gpu_output/balanced_benchmark/"
        "fits/unseen_family"
    )
    values = {}
    for model in ("ligand", "c2", "c3"):
        epochs = []
        for seed in (20260817, 20260818, 20260819):
            for fold in range(5):
                path = (
                    fit_root / model / "seed_{}".format(seed)
                    / "fold_{}".format(fold) / "FIT_REPORT.json"
                )
                epochs.append(
                    int(json.loads(path.read_text(encoding="utf-8"))["best_epoch"])
                )
        median = statistics.median(epochs)
        values[model] = {
            "best_epochs": epochs,
            "median": int(median),
            "fit_count": len(epochs),
        }
    return values


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    root = args.project_root.resolve()
    shared = args.shared_root.resolve()
    package = root / "analysis/property_balanced_chembl"
    data = package / "data/chembl_pocket_extension"
    table_path = data / "TARGET_CHAIN_SEQUENCES.tsv"
    masks_path = data / "POCKET_INDICES.json"
    audit_path = data / "POCKET_ALIGNMENT_AUDIT.tsv"
    coverage_path = data / "POCKET_COVERAGE.json"
    config_path = package / "config.json"
    epoch_path = package / "data/C3_DEPLOY_EPOCH_AUDIT.json"
    table = pd.read_csv(table_path, sep="\t", low_memory=False)
    audit = pd.read_csv(audit_path, sep="\t", low_memory=False).fillna("")
    masks = json.loads(masks_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    epoch_audit = json.loads(epoch_path.read_text(encoding="utf-8"))

    reference_path = (
        root / "analysis/chembl_external_prioritization/gpu_cache/"
        "REFERENCE_MODEL_READY.tsv.gz"
    )
    reference_frame = pd.read_csv(
        reference_path, sep="\t", usecols=["uniprot", "weak2020_label"]
    )
    reference = {canonical_uid(value) for value in reference_frame["uniprot"]}
    primary = {
        canonical_uid(value)
        for value in reference_frame.loc[
            reference_frame["weak2020_label"].isin([0, 1]), "uniprot"
        ]
    }
    full = full_screen_set(root, shared)
    universe = reference | full
    ready = set(audit.loc[audit["status"].eq("ready"), "uniprot"].astype(str))
    table_uids = set(table["uniprot"].astype(str))

    checks = {}
    checks["target_schema"] = REQUIRED_TARGET_COLUMNS <= set(table)
    checks["audit_schema"] = REQUIRED_AUDIT_COLUMNS <= set(audit)
    checks["audit_exact_union"] = set(audit["uniprot"].astype(str)) == universe
    checks["audit_unique"] = not audit["uniprot"].duplicated().any()
    checks["target_unique"] = not table["uniprot"].duplicated().any()
    checks["ready_equals_table"] = ready == table_uids
    checks["masks_equal_table"] = set(masks) == table_uids

    target_errors = []
    for row in table.itertuples(index=False):
        uid = str(row.uniprot)
        sequence = str(row.target_chain_sequence)
        values = [int(value) for value in masks.get(uid, [])]
        if (
            int(row.target_chain_length) != len(sequence)
            or str(row.sequence_sha256) != sha256_text(sequence)
            or len(values) != int(row.pocket_residues)
            or not values
            or values != sorted(set(values))
            or values[0] < 0
            or values[-1] >= len(sequence)
            or abs(float(row.pocket_fraction) - len(values) / len(sequence)) > 1e-12
        ):
            target_errors.append(uid + ":coordinate_or_hash")
            continue
        fasta = data / "target_chain_fasta" / (uid + ".fasta")
        try:
            header, observed_sequence = read_fasta(fasta)
            if (
                observed_sequence != sequence
                or header != ">" + uid
            ):
                target_errors.append(uid + ":fasta_content")
        except Exception as error:
            target_errors.append(uid + ":" + type(error).__name__)
    checks["target_coordinate_and_fasta_semantics"] = not target_errors
    fasta_paths = {
        path.name for path in (data / "target_chain_fasta").glob("*.fasta")
    }
    checks["no_stale_fasta"] = fasta_paths == {
        uid + ".fasta" for uid in table_uids
    }

    derived_scopes = {}
    for name, flag, expected in [
        ("reference", "in_reference", reference),
        ("two_label_primary", "in_two_label_primary", primary),
        ("full_screen", "in_full_screen", full),
    ]:
        scope = audit[audit[flag].astype(int).eq(1)]
        derived_scopes[name] = {
            "proteins": int(len(scope)),
            "ready_proteins": int(scope["status"].eq("ready").sum()),
            "excluded_proteins": int(scope["status"].ne("ready").sum()),
        }
        observed = coverage[name]
        checks[name + "_counts"] = all(
            int(observed[key]) == value
            for key, value in derived_scopes[name].items()
        ) and int(len(scope)) == len(expected)
    full_audit = audit[audit["in_full_screen"].astype(int).eq(1)]
    absent_reasons = {
        "fpocket_mapping_missing", "fpocket_mapping_missing_pdb",
        "fpocket_coordinate_file_missing",
    }
    full_absent = int(full_audit["reason"].isin(absent_reasons).sum())
    checks["full_screen_pdb_absent_zero"] = (
        full_absent == 0
        and int(coverage["full_screen"]["pdb_structure_absent_proteins"]) == 0
    )

    broad_table = pd.read_csv(
        root / "analysis/allosteric_pair_benchmark_broad_superset/data/"
        "UNIFIED_TARGET_CHAIN_SEQUENCES.tsv",
        sep="\t", low_memory=False,
    ).set_index("uniprot")
    broad_masks = json.loads((
        root / "analysis/allosteric_pair_benchmark_broad_superset/data/"
        "POCKET_INDICES.json"
    ).read_text(encoding="utf-8"))
    table_index = table.set_index("uniprot")
    shared_ready = set(table_uids) & set(broad_masks)
    frozen_mismatch = []
    for uid in sorted(shared_ready):
        old = broad_table.loc[uid]
        new = table_index.loc[uid]
        if (
            [int(value) for value in broad_masks[uid]]
            != [int(value) for value in masks[uid]]
            or str(old["mapped_pdb"]) != str(new["mapped_pdb"])
            or str(old["selected_chain"]) != str(new["selected_chain"])
            or str(old["sequence_sha256"]) != str(new["sequence_sha256"])
        ):
            frozen_mismatch.append(uid)
    checks["frozen_broad_exact_reproduction"] = not frozen_mismatch

    epoch_values = derive_epoch_audit(root)
    derived_epochs = {key: value["median"] for key, value in epoch_values.items()}
    checks["epoch_audit_15_fits_each"] = all(
        value["fit_count"] == 15 for value in epoch_values.values()
    )
    checks["epoch_audit_reproduces"] = (
        epoch_audit.get("deploy_epochs") == derived_epochs
        and epoch_audit.get("selection_independent_of_chembl") is True
    )
    checks["config_includes_c3"] = (
        config.get("deploy_epochs", {}).get("c3") == 6
        and config.get("deploy_models") == ["ligand", "c2", "c3"]
        and "property_c3" in config.get("inference_models", [])
    )
    pocket_contract = config.get("c3_pocket_extension", {})
    checks["config_pins_pocket_outputs"] = all([
        pocket_contract.get("target_chain_sha256") == sha256_file(table_path),
        pocket_contract.get("pocket_indices_sha256") == sha256_file(masks_path),
        pocket_contract.get("alignment_audit_sha256") == sha256_file(audit_path),
        pocket_contract.get("coverage_sha256") == sha256_file(coverage_path),
        pocket_contract.get("full_screen_ready_proteins")
        == derived_scopes["full_screen"]["ready_proteins"],
        pocket_contract.get("reference_ready_proteins")
        == derived_scopes["reference"]["ready_proteins"],
    ])
    checks["no_chembl_epoch_selection"] = (
        config.get("c3_deploy_epoch", {}).get("selection_source")
        == "chemistry_balanced_family_heldout_15_fits"
        and config.get("c3_deploy_epoch", {}).get("audit_sha256")
        == sha256_file(epoch_path)
    )

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "union_proteins": len(universe),
            "ready_union_proteins": len(ready),
            **derived_scopes,
            "full_screen_pdb_structure_absent": full_absent,
            "frozen_broad_common_proteins": len(shared_ready),
            "frozen_broad_mismatches": len(frozen_mismatch),
            "target_semantic_errors": len(target_errors),
        },
        "deploy_epoch_recalculation": epoch_values,
        "limitations": [
            "The frozen fpocket map supplies no mapping for 254 reference proteins.",
            "C3 metrics are valid only for rows whose target has a validated selected-chain pocket embedding.",
            "This is readiness validation; production C3 scores require the GPU ESM3 and inference run.",
        ],
    }
    if args.write_report:
        atomic_json(package / "data/C3_EXTENSION_READINESS_VALIDATION.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
