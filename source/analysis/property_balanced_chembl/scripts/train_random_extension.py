#!/usr/bin/env python3
"""Train the prespecified second model seed on two random cohort draws."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


MODELS = ["ligand", "c2", "c3"]
FOLDS = list(range(5))
MODEL_SEED = 20260818
DRAW_CONTRACTS = {
    20260824: {
        "file": "PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260824.tsv.gz",
        "sha256": "346e6dcc1da7125a7ca6bb9abdc1e139cc2c1773a24c3a2ea909360c31f11d0f",
    },
    20260825: {
        "file": "PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260825.tsv.gz",
        "sha256": "c19e9b82a92f46cbe0b5a9cec57a37acd5ab87c1b8ba4e649a0164dcf54281d9",
    },
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--selection-seeds", default="20260824,20260825")
    parser.add_argument("--overall-index-name", default="RANDOM_EXTENSION_INDEX.json")
    parser.add_argument("--device", default="cuda:0")
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
    return parser.parse_args()


def main():
    args = parse_args()
    selection_seeds = [int(x.strip()) for x in args.selection_seeds.split(",") if x.strip()]
    if not selection_seeds or len(selection_seeds) != len(set(selection_seeds)):
        raise ValueError("selection seeds must be unique and nonempty")
    if not set(selection_seeds) <= set(DRAW_CONTRACTS):
        raise ValueError("only selection seeds 20260824 and 20260825 are permitted")
    if Path(args.overall_index_name).name != args.overall_index_name:
        raise ValueError("overall-index-name must be a basename")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    root = args.project_root
    package = root / "analysis/property_balanced_chembl"
    ligand_package = root / "analysis/ligand_chemistry_balancing"
    main_package = root / "analysis/allosteric_pair_benchmark_main"
    control = load_module(
        "frozen_ligand_control",
        ligand_package / "scripts/train_balanced_models.py",
    )
    trainer = control.load_main_trainer(main_package / "scripts/train_main_benchmark.py")
    control.install_fp32_probability_evaluator(trainer)
    device = torch.device(args.device)

    frames = {}
    observed_hashes = {}
    for selection_seed in selection_seeds:
        contract = DRAW_CONTRACTS[selection_seed]
        path = ligand_package / contract["file"]
        observed_hash = sha256(path)
        if observed_hash != contract["sha256"]:
            raise RuntimeError("cohort hash changed for {}".format(selection_seed))
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        counts = {
            "rows": int(len(frame)),
            "proteins": int(frame["uniprot"].nunique()),
            "family_components": int(frame["family_component_id"].nunique()),
            "allosteric_rows": int(frame["binary_label"].astype(int).sum()),
            "orthosteric_rows": int((1 - frame["binary_label"].astype(int)).sum()),
        }
        expected = {
            "rows": 3022,
            "proteins": 426,
            "family_components": 165,
            "allosteric_rows": 1511,
            "orthosteric_rows": 1511,
        }
        if counts != expected:
            raise RuntimeError("cohort count changed for {}: {}".format(selection_seed, counts))
        if frame["main_row_id"].duplicated().any():
            raise RuntimeError("duplicate row ID for {}".format(selection_seed))
        if not frame.groupby("uniprot")["binary_label"].nunique().eq(2).all():
            raise RuntimeError("protein anchoring lost for {}".format(selection_seed))
        if set(frame["family_fold"].astype(int)) != set(FOLDS):
            raise RuntimeError("family fold contract changed")
        frames[selection_seed] = frame
        observed_hashes[selection_seed] = observed_hash

    source = pd.read_csv(main_package / "gpu_cache/MODEL_READY.tsv.gz", sep="\t", low_memory=False)
    with (main_package / "data/POCKET_INDICES.json").open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    print("Preloading frozen tensors for random-draw extension", flush=True)
    caches = trainer.load_caches(source, pocket_indices, args.max_atoms)
    print("ligands={} proteins={} pockets={}".format(
        *(len(value) for value in caches)
    ), flush=True)

    indexes = {}
    for selection_seed in selection_seeds:
        output_root = (
            package / "gpu_output/random_extension"
            / "selection_seed_{}".format(selection_seed)
        )
        previous_path = output_root / "TRAINING_INDEX.json"
        if previous_path.is_file():
            try:
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                reusable = (
                    previous.get("status") == "validated"
                    and previous.get("selection_seed") == selection_seed
                    and previous.get("model_seed") == MODEL_SEED
                    and previous.get("models") == MODELS
                    and previous.get("folds") == FOLDS
                    and previous.get("n_completed_fits") == 15
                    and previous.get("cohort_sha256") == observed_hashes[selection_seed]
                    and previous.get("probability_conversion") == control.PROBABILITY_CONVERSION
                )
            except Exception:
                reusable = False
            if reusable:
                trainer.control_resume_models_by_root[str(output_root.resolve())] = set(MODELS)
                print("Authorized extension resume for {}".format(selection_seed), flush=True)

        reports = []
        for model_name in MODELS:
            for fold in FOLDS:
                reports.append(trainer.train_fit(
                    model_name,
                    "unseen_family",
                    MODEL_SEED,
                    fold,
                    args,
                    frames[selection_seed],
                    caches,
                    output_root,
                    device,
                ))
        index = {
            "status": "validated" if all(x.get("status") == "validated" for x in reports) else "failed",
            "selection_seed": int(selection_seed),
            "model_seed": int(MODEL_SEED),
            "models": MODELS,
            "folds": FOLDS,
            "n_expected_fits": 15,
            "n_completed_fits": int(len(reports)),
            "cohort_path": str(ligand_package / DRAW_CONTRACTS[selection_seed]["file"]),
            "cohort_sha256": observed_hashes[selection_seed],
            "probability_conversion": control.PROBABILITY_CONVERSION,
            "maximum_epoch_hits": [
                {"model": x["model"], "outer_fold": x["outer_fold"]}
                for x in reports if x.get("hit_maximum_epoch")
            ],
        }
        trainer.atomic_json(output_root / "TRAINING_INDEX.json", index)
        if index["status"] != "validated" or index["n_completed_fits"] != 15:
            raise SystemExit("extension training failed for {}".format(selection_seed))
        indexes[str(selection_seed)] = index

    overall = {
        "status": "validated",
        "selection_seeds": selection_seeds,
        "model_seed": MODEL_SEED,
        "models": MODELS,
        "folds": FOLDS,
        "n_expected_fits": int(15 * len(selection_seeds)),
        "n_completed_fits": int(sum(x["n_completed_fits"] for x in indexes.values())),
        "indexes": {
            key: str(
                package / "gpu_output/random_extension"
                / "selection_seed_{}".format(key) / "TRAINING_INDEX.json"
            )
            for key in indexes
        },
    }
    trainer.atomic_json(
        package / "gpu_output/random_extension" / args.overall_index_name,
        overall,
    )
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
