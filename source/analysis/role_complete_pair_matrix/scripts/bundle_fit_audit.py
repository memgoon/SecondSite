#!/usr/bin/env python3
"""Bundle all lightweight fit reports and histories without checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path


ACTIVE_REGIMES = {
    "every_pair": ("row_random", "unseen_family", "unseen_ligand"),
    "protein_anchored": ("row_random", "unseen_family", "unseen_ligand"),
    "protein_ligand_role_complete": (
        "row_random",
        "unseen_family",
        "unseen_ligand",
    ),
}
MODELS = ("ligand", "protein", "c1", "c2", "c3", "d1", "d2", "d3")
SEEDS = (20260817, 20260818, 20260819)
FOLDS = range(5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    package = args.project_root.resolve() / "analysis/role_complete_pair_matrix"
    fit_root = package / "gpu_output/benchmark/fits"
    directories = [
        fit_root
        / arm
        / regime
        / model
        / "seed_{}".format(seed)
        / "fold_{}".format(fold)
        for arm, regimes in ACTIVE_REGIMES.items()
        for regime in regimes
        for model in MODELS
        for seed in SEEDS
        for fold in FOLDS
    ]
    if len(directories) != 1080:
        raise RuntimeError("expected 1,080 fit directories, observed {}".format(len(directories)))
    all_directories = set(fit_root.glob("*/*/*/seed_*/fold_*"))
    inactive_directories = sorted(all_directories - set(directories))
    files = []
    manifest = []
    for directory in directories:
        for name in ["FIT_REPORT.json", "history.tsv"]:
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(path)
            manifest.append(
                {
                    "path": str(path.relative_to(package)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    output = package / "gpu_output/benchmark/FIT_AUDIT_REPORTS.tar.gz"
    temporary = output.with_suffix(output.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(package)))
    os.replace(str(temporary), str(output))
    report = {
        "status": "validated",
        "fit_directories": len(directories),
        "inactive_stale_fit_directories_ignored": len(inactive_directories),
        "files": len(files),
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256(output),
        "manifest": manifest,
        "checkpoints_included": False,
        "predictions_included": False,
    }
    atomic_json(package / "gpu_output/benchmark/FIT_AUDIT_BUNDLE.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "manifest"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
