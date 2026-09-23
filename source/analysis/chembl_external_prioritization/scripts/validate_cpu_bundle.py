#!/usr/bin/env python3
"""Fail closed if the locally prepared handoff bundle is incomplete."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    args = parser.parse_args()
    package = args.project_root / "analysis/chembl_external_prioritization"
    report = json.loads((package / "validation/CPU_INPUT_VALIDATION.json").read_text())
    if report.get("status") != "validated" or not all(report.get("gates", {}).values()):
        raise SystemExit("CPU validation report failed")
    direct = pd.read_csv(package / "data/CHEMBL_DIRECT_REFERENCE_PAIRS.tsv.gz", sep="\t")
    blacklist = pd.read_csv(package / "data/CURRENT_BENCHMARK_PAIR_BLACKLIST.tsv.gz", sep="\t")
    overlap = set(direct["pair_connectivity_key"].astype(str)) & set(blacklist["pair_connectivity_key"].astype(str))
    if overlap or len(direct) != report["direct_reference"]["rows_after_current_benchmark_exclusion"]:
        raise SystemExit("direct reference identity/count validation failed")
    if direct["weak2020_label"].eq(1).sum() == 0 or direct["weak2020_label"].eq(0).sum() == 0:
        raise SystemExit("direct 2020 reference lacks a class")
    print(json.dumps({
        "status": "validated", "direct_reference_rows": int(len(direct)),
        "benchmark_blacklist_pairs": int(blacklist["pair_connectivity_key"].nunique()),
        "pair_overlap": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
