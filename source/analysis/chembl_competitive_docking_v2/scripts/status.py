#!/usr/bin/env python3
"""Report hash-compatible docking progress and a rough remaining time."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def read_status(row: pd.Series) -> Tuple[str, float]:
    path = Path(row.output_dir) / "status.json"
    if not path.exists():
        return "pending", math.nan
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "invalid", math.nan
    if value.get("job_contract_sha256") != row.job_contract_sha256:
        return "stale", math.nan
    state = str(value.get("status", "invalid"))
    if state == "complete" and Path(row.output_pose).exists() and Path(row.output_pose).stat().st_size:
        return "complete", float(value.get("elapsed_seconds", math.nan))
    if state == "failed":
        return "failed", float(value.get("elapsed_seconds", math.nan))
    return "invalid", float(value.get("elapsed_seconds", math.nan))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    root = args.root.resolve()
    jobs = pd.read_csv(root / "inputs" / "docking_jobs.tsv", sep="\t")
    states: List[str] = []
    elapsed: List[float] = []
    completed_by_class: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    pending_by_class: Dict[Tuple[str, str], int] = defaultdict(int)
    for _, row in jobs.iterrows():
        state, seconds = read_status(row)
        states.append(state)
        elapsed.append(seconds)
        key = (str(row.target_id), str(row.scoring))
        if state == "complete" and np.isfinite(seconds):
            completed_by_class[key].append(seconds)
        elif state != "complete":
            pending_by_class[key] += 1

    counts = pd.Series(states).value_counts().to_dict()
    finite = np.asarray([value for value in elapsed if np.isfinite(value)], dtype=float)
    global_median = float(np.median(finite)) if len(finite) else math.nan
    remaining_job_seconds = 0.0
    estimable = bool(len(finite))
    for key, count in pending_by_class.items():
        values = completed_by_class.get(key, [])
        estimate = float(np.median(values)) if values else global_median
        if not np.isfinite(estimate):
            estimable = False
            break
        remaining_job_seconds += count * estimate
    eta_hours = remaining_job_seconds / args.workers / 3600.0 if estimable else math.nan
    output = {
        "manifest_jobs": int(len(jobs)),
        "complete": int(counts.get("complete", 0)),
        "pending": int(len(jobs) - counts.get("complete", 0) - counts.get("failed", 0)),
        "failed": int(counts.get("failed", 0)),
        "stale_or_invalid": int(counts.get("stale", 0) + counts.get("invalid", 0)),
        "percent_complete": round(100.0 * counts.get("complete", 0) / len(jobs), 2),
        "completed_job_median_seconds": global_median if np.isfinite(global_median) else None,
        "workers_used_for_eta": args.workers,
        "rough_eta_hours": eta_hours if np.isfinite(eta_hours) else None,
        "eta_caveat": "ETA uses observed target/scoring medians and ignores shared-resource contention.",
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
