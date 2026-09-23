#!/usr/bin/env python3
"""Combine the two disjoint 15-fit extension indexes."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    args = parser.parse_args()
    package = args.project_root / "analysis/property_balanced_chembl"
    root = package / "gpu_output/random_extension"
    rows = []
    for selection_seed in [20260824, 20260825]:
        path = root / "selection_seed_{}".format(selection_seed) / "TRAINING_INDEX.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("status") != "validated"
            or row.get("selection_seed") != selection_seed
            or row.get("model_seed") != 20260818
            or row.get("models") != ["ligand", "c2", "c3"]
            or row.get("folds") != [0, 1, 2, 3, 4]
            or row.get("n_completed_fits") != 15
        ):
            raise SystemExit("extension index failed: {}".format(path))
        rows.append(row)
    report = {
        "status": "validated",
        "selection_seeds": [20260824, 20260825],
        "model_seed": 20260818,
        "models": ["ligand", "c2", "c3"],
        "folds": [0, 1, 2, 3, 4],
        "n_expected_fits": 30,
        "n_completed_fits": int(sum(x["n_completed_fits"] for x in rows)),
        "maximum_epoch_hits": int(sum(len(x.get("maximum_epoch_hits", [])) for x in rows)),
    }
    path = root / "RANDOM_EXTENSION_INDEX.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
