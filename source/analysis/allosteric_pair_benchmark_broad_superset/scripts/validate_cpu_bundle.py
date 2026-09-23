#!/usr/bin/env python3
"""Fail closed if the CPU-frozen Arm B handoff bundle is inconsistent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    package = args.project_root / "analysis/allosteric_pair_benchmark_broad_superset"
    report_path = package / "validation/CPU_INPUT_VALIDATION.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "validated":
        raise SystemExit("CPU input validation is not accepted")
    for value in report["outputs"].values():
        path = Path(value["path"])
        if not path.is_file() or sha256(path) != value["sha256"]:
            raise SystemExit("CPU output hash mismatch: {}".format(path))
    broad = pd.read_csv(package / "data/BROAD_ALIGNED_POOL.tsv.gz", sep="\t", low_memory=False)
    core = pd.read_csv(package / "data/REFERENCE_ARM_A_MODEL_READY.tsv.gz", sep="\t", low_memory=False)
    core_ids = set(core["main_row_id"].astype(str))
    broad_core = broad[broad["pool_membership"].eq("frozen_arm_a_core")]
    if len(core) != 4637 or set(broad_core["main_row_id"].astype(str)) != core_ids:
        raise SystemExit("Arm A is not exactly preserved in the handoff")
    if len(broad) <= len(core):
        raise SystemExit("Arm B contains no additional rows")
    print(json.dumps({
        "status": "validated",
        "arm_a_rows": int(len(core)),
        "pre_gpu_arm_b_rows": int(len(broad)),
        "pre_gpu_additional_rows": int(len(broad) - len(core)),
        "pre_gpu_arm_b_proteins": int(broad["uniprot"].nunique()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
