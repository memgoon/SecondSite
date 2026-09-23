#!/usr/bin/env python3
"""Freeze deploy epochs from the completed chemistry-balanced family CV fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path


MODELS = ("ligand", "c2", "c3")
SEEDS = (20260817, 20260818, 20260819)
FOLDS = tuple(range(5))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery")
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(root):
    fit_root = root / (
        "analysis/ligand_chemistry_balancing/gpu_output/balanced_benchmark/"
        "fits/unseen_family"
    )
    rows = []
    medians = {}
    for model in MODELS:
        epochs = []
        for seed in SEEDS:
            for fold in FOLDS:
                path = (
                    fit_root
                    / model
                    / "seed_{}".format(seed)
                    / "fold_{}".format(fold)
                    / "FIT_REPORT.json"
                )
                report = json.loads(path.read_text(encoding="utf-8"))
                epoch = int(report["best_epoch"])
                epochs.append(epoch)
                rows.append({
                    "model": model,
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": epoch,
                    "fit_report": str(path.relative_to(root)),
                    "fit_report_sha256": sha256_file(path),
                })
        median = statistics.median(epochs)
        if int(median) != median:
            raise RuntimeError("nonintegral deploy median for {}".format(model))
        medians[model] = int(median)
    return rows, medians


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
    output = (
        root
        / "analysis/property_balanced_chembl/data/"
        "C3_DEPLOY_EPOCH_AUDIT.json"
    )
    rows, medians = derive(root)
    expected = {
        "status": "PASS",
        "selection_scope": "chemistry-balanced family-held-out fits only",
        "selection_independent_of_chembl": True,
        "rule": "integer median best epoch across 3 seeds x 5 folds",
        "fit_count_per_model": 15,
        "models": list(MODELS),
        "deploy_epochs": medians,
        "fits": rows,
    }
    if args.validate_only:
        observed = json.loads(output.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("C3 deploy-epoch audit does not reproduce")
    else:
        atomic_json(output, expected)
    print(json.dumps({
        "status": "PASS",
        "deploy_epochs": medians,
        "fit_count_per_model": 15,
        "validate_only": bool(args.validate_only),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
