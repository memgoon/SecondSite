#!/usr/bin/env python3
"""Audit all ten locked split designs before the GPU handoff."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_contract import build_split  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    package = args.project_root / "analysis/allosteric_pair_benchmark_broad_superset"
    frame = pd.read_csv(package / "data/BROAD_ALIGNED_POOL.tsv.gz", sep="\t", low_memory=False)
    rows = []
    for regime in ["row_random", "unseen_family"]:
        for fold in range(5):
            _, _, _, _, audit = build_split(frame, regime, fold)
            rows.append(audit)
    failures = [
        row for row in rows
        if row["train_locked_pair_overlap"]
        or row["development_common_unseen_compound_overlap"]
        or (row["regime"] == "unseen_family" and (
            row["train_heldout_protein_overlap"] or row["train_heldout_pfam_overlap"]
        ))
    ]
    output = package / "validation/PRE_GPU_SPLIT_DESIGN.tsv"
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    report = {
        "status": "validated" if not failures else "failed",
        "n_splits": int(len(rows)),
        "failed_splits": failures,
        "audit_table": str(output),
    }
    atomic_json(package / "validation/PRE_GPU_SPLIT_VALIDATION.json", report)
    print(pd.DataFrame(rows)[[
        "regime", "test_fold", "arm_a_train_rows", "eligible_additional_rows",
        "arm_b_train_rows", "locked_validation_rows", "locked_test_rows",
        "common_unseen_compound_test_rows", "train_heldout_protein_overlap",
        "train_heldout_pfam_overlap",
    ]].to_string(index=False))
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("pre-GPU split design validation failed")


if __name__ == "__main__":
    main()
