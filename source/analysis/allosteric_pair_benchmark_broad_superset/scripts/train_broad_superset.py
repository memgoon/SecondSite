#!/usr/bin/env python3
"""Train Arm B while keeping every Arm A validation and test row locked."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import split_contract  # noqa: E402


VALID_MODELS = {"ligand", "protein", "c1", "c2", "c3", "d1"}
VALID_REGIMES = {"row_random", "unseen_family"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--models", default="ligand,protein,c1,c2,c3,d1")
    parser.add_argument("--regimes", default="row_random,unseen_family")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="20260817,20260818,20260819")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-atoms", type=int, default=120)
    parser.add_argument("--max-protein-residues", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_main_trainer(project_root):
    path = project_root / "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
    spec = importlib.util.spec_from_file_location("frozen_main_training_code", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def comma_strings(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def comma_ints(value):
    return [int(item) for item in comma_strings(value)]


def main():
    args = parse_args()
    models = comma_strings(args.models)
    regimes = comma_strings(args.regimes)
    folds = comma_ints(args.folds)
    seeds = comma_ints(args.seeds)
    if set(models) - VALID_MODELS or set(regimes) - VALID_REGIMES:
        raise ValueError("unknown model or regime")
    if set(folds) - set(range(5)) or not seeds:
        raise ValueError("folds must be 0..4 and at least one seed is required")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    root = args.project_root
    package = root / "analysis/allosteric_pair_benchmark_broad_superset"
    frame_path = package / "gpu_cache/BROAD_MODEL_READY.tsv.gz"
    masks_path = package / "data/POCKET_INDICES.json"
    if not frame_path.is_file() or not masks_path.is_file():
        raise FileNotFoundError("broad model-ready table or pocket masks are missing")
    frame = pd.read_csv(frame_path, sep="\t", low_memory=False)
    with masks_path.open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    required = {
        "main_row_id", "pair_key", "pool_membership", "uniprot", "full_inchikey",
        "connectivity_key", "binary_label", "family_component_id", "family_fold",
        "row_fold", "pfam_accessions", "protein_embedding_path", "ligand_embedding_path",
    }
    if required - set(frame.columns):
        raise RuntimeError("broad model-ready table is missing required fields")
    core = frame[frame["pool_membership"].eq("frozen_arm_a_core")]
    if len(core) != 4637 or core["uniprot"].nunique() != 426:
        raise RuntimeError("Arm A core count changed before Arm B training")
    if frame["main_row_id"].duplicated().any() or frame["pair_key"].duplicated().any():
        raise RuntimeError("broad model-ready rows are not unique")

    trainer = load_main_trainer(root)
    split_audits = []
    split_cache = {}
    for regime in regimes:
        for fold in folds:
            train, validation, test, validation_fold, audit = split_contract.build_split(
                frame, regime, fold
            )
            split_cache[(regime, fold)] = (train, validation, test, validation_fold)
            split_audits.append(audit)
    failed_audits = [
        row for row in split_audits
        if row["train_locked_pair_overlap"]
        or row["development_common_unseen_compound_overlap"]
        or (row["regime"] == "unseen_family" and (
            row["train_heldout_protein_overlap"] or row["train_heldout_pfam_overlap"]
        ))
    ]
    if failed_audits:
        raise RuntimeError("one or more Arm B split gates failed")
    split_audit_path = package / "validation/SPLIT_AUDIT.tsv"
    split_audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_audits).to_csv(split_audit_path, sep="\t", index=False)

    def frozen_split(_frame, regime, test_fold):
        train, validation, test, validation_fold = split_cache[(regime, int(test_fold))]
        return train.copy(), validation.copy(), test.copy(), int(validation_fold)

    trainer.split_frame = frozen_split
    print("Preloading exact Arm B ligand, protein, and pocket tensors", flush=True)
    caches = trainer.load_caches(frame, pocket_indices, args.max_atoms)
    print("ligands={} proteins={} pockets={}".format(*(len(value) for value in caches)), flush=True)
    output_root = package / "gpu_output/broad_superset"
    device = torch.device(args.device)
    reports = []
    for regime in regimes:
        for model_name in models:
            for seed in seeds:
                for fold in folds:
                    reports.append(
                        trainer.train_fit(
                            model_name, regime, seed, fold, args, frame, caches, output_root, device
                        )
                    )

    index = {
        "status": "validated" if all(row.get("status") == "validated" for row in reports) else "failed",
        "experiment": "Arm B broad-superset training on locked Arm A validation/test rows",
        "models": models,
        "regimes": regimes,
        "folds": folds,
        "seeds": seeds,
        "n_expected_fits": int(len(models) * len(regimes) * len(folds) * len(seeds)),
        "n_completed_fits": int(len(reports)),
        "maximum_epoch_hits": [
            {
                "regime": row["regime"], "model": row["model"], "seed": row["seed"],
                "outer_fold": row["outer_fold"],
            }
            for row in reports if row.get("hit_maximum_epoch")
        ],
        "training_contract": {
            "frozen_model_implementation": str(
                root / "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
            ),
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "training_weight": "equal total weight per protein-label group and then per label",
            "checkpoint_selection": "locked validation Pfam-component-macro symmetric AP",
            "locked_evaluation": "all validation and test rows come only from frozen Arm A",
        },
        "split_audit": str(split_audit_path),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "TRAINING_INDEX.json", index)
    atomic_json(package / "validation/SPLIT_VALIDATION.json", {
        "status": "validated",
        "n_regime_fold_splits": int(len(split_audits)),
        "all_locked_pair_overlaps_zero": True,
        "all_unseen_family_protein_overlaps_zero": True,
        "all_unseen_family_pfam_overlaps_zero": True,
        "all_common_unseen_compound_overlaps_zero": True,
        "audit_table": str(split_audit_path),
    })
    print(json.dumps(index, indent=2, sort_keys=True))
    if index["status"] != "validated" or index["n_completed_fits"] != index["n_expected_fits"]:
        raise SystemExit("Arm B broad-superset training did not complete")


if __name__ == "__main__":
    main()
