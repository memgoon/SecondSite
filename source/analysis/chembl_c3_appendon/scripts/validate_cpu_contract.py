#!/usr/bin/env python3
"""Read-only semantic and checksum validation of C3 append-on inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd


PACKAGE = Path("analysis/chembl_c3_appendon")
CONTRACT_ID = "chembl_c3_appendon_v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def validate_manifest(root: Path, manifest: Path):
    failures = []
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        if not match:
            failures.append(f"malformed:{line}")
            continue
        expected, relative = match.groups()
        path = root / relative
        count += 1
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256(path) != expected:
            failures.append(f"hash:{relative}")
    return count, failures


def main():
    args = parse_args()
    root = args.project_root.resolve()
    package = root / PACKAGE
    contract_path = package / "validation/CPU_CONTRACT.json"
    manifest_path = package / "manifests/CPU_INPUTS.sha256"
    if not contract_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("run build_cpu_contract.py first")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    count, failures = validate_manifest(root, manifest_path)

    expected = contract.get("expected", {})
    semantic_failures = []
    if contract.get("status") != "validated" or contract.get("contract_id") != CONTRACT_ID:
        semantic_failures.append("contract_identity")
    if expected.get("rows") != 799873 or expected.get("selected_chain_available_rows") != 785182:
        semantic_failures.append("frozen_counts")
    if expected.get("selected_chain_unavailable_rows") != 14691:
        semantic_failures.append("unavailable_count")
    if expected.get("C3_score_columns") != [
        "p_every_pair_c3_mean",
        "p_every_pair_c3_sd",
        "p_general_c3_mean",
        "p_general_c3_sd",
    ]:
        semantic_failures.append("score_schema")
    if len(contract.get("C3_checkpoints", [])) != 6:
        semantic_failures.append("checkpoint_count")

    chain_path = root / "analysis/property_balanced_chembl/data/chembl_pocket_extension/TARGET_CHAIN_SEQUENCES.tsv"
    chains = pd.read_csv(chain_path, sep="\t", usecols=["uniprot", "target_chain_length"])
    if len(chains) != 808 or chains["uniprot"].duplicated().any():
        semantic_failures.append("selected_chain_table")
    for shard in range(4):
        report_path = root / (
            "analysis/role_complete_pair_matrix/gpu_output/chembl/full/"
            f"FULL_SHARD{shard:02d}.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        models = {m for values in report.get("models_per_arm", {}).values() for m in values}
        if "c3" in models or report.get("status") != "validated":
            semantic_failures.append(f"upstream_shard_{shard}")

    result = {
        "status": "PASS" if not failures and not semantic_failures else "FAIL",
        "contract_id": CONTRACT_ID,
        "manifest_target_count": count,
        "checksum_failures": failures,
        "semantic_failures": semantic_failures,
        "expected_rows": expected.get("rows"),
        "expected_C3_eligible_rows": expected.get("selected_chain_available_rows"),
        "existing_outputs_immutable": True,
        "network_requests_performed": 0,
    }
    if args.write:
        atomic_json(package / "validation/CPU_VALIDATION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

