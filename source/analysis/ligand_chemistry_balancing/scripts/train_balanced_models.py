#!/usr/bin/env python3
"""Retrain ligand-only, C2, and C3 under frozen ligand-control contracts."""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


PROBABILITY_CONVERSION = (
    "sigmoid applied after FP32 logit cast to avoid FP16 rank ties"
)


def load_main_trainer(path):
    spec = importlib.util.spec_from_file_location("frozen_main_trainer", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--models", default="ligand,c2,c3")
    parser.add_argument("--arms", default="source,random,balanced")
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
    parser.add_argument("--overall-index-name", default="CONTROL_TRAINING_INDEX.json")
    return parser.parse_args()


def install_fp32_probability_evaluator(trainer):
    """Keep mixed-precision forward passes but compute sigmoid in FP32."""
    original_fit_complete = trainer.fit_complete

    def fit_complete_with_checkpoint(paths, expected_test_ids):
        # Common-row rescoring needs the checkpoint, so a report/prediction-only
        # directory is not resumable for this experiment. Reuse is also limited
        # to output roots whose completed training index matches this contract.
        output_root = str(paths["directory"].parents[4].resolve())
        model_name = paths["directory"].parents[1].name
        return (
            model_name in trainer.control_resume_models_by_root.get(output_root, set())
            and paths["checkpoint"].is_file()
            and original_fit_complete(paths, expected_test_ids)
        )

    def evaluate_fp32(model, loader, device):
        model.eval()
        records = []
        with torch.no_grad():
            for batch in loader:
                tensors = trainer.move_batch(batch, device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(tensors)
                probabilities = torch.sigmoid(logits.float()).cpu().numpy()
                for index, probability in enumerate(probabilities):
                    records.append({
                        "main_row_id": batch["main_row_id"][index],
                        "uniprot": batch["uniprot"][index],
                        "family_component_id": batch["family_component_id"][index],
                        "full_inchikey": batch["full_inchikey"][index],
                        "connectivity_key": batch["connectivity_key"][index],
                        "class_label": batch["class_label"][index],
                        "binary_label": int(batch["label"][index].item()),
                        "p_allosteric": float(probability),
                    })
        prediction = pd.DataFrame(records)
        return prediction, trainer.metric_bundle(prediction)

    trainer.control_resume_models_by_root = {}
    trainer.fit_complete = fit_complete_with_checkpoint
    trainer.evaluate = evaluate_fp32


def main():
    args = parse_args()
    package = args.project_root / "analysis/ligand_chemistry_balancing"
    main_package = args.project_root / "analysis/allosteric_pair_benchmark_main"
    trainer = load_main_trainer(main_package / "scripts/train_main_benchmark.py")
    install_fp32_probability_evaluator(trainer)
    models = trainer.comma_strings(args.models)
    arms = trainer.comma_strings(args.arms)
    folds = trainer.comma_ints(args.folds)
    seeds = trainer.comma_ints(args.seeds)
    if set(models) != {"ligand", "c2", "c3"} or len(models) != 3:
        raise ValueError("this control is frozen to ligand,c2,c3")
    main_arms = {"source", "random", "balanced"}
    sensitivity_arms = {"random_seed_20260824", "random_seed_20260825"}
    arm_set = set(arms)
    if not arms or len(arm_set) != len(arms):
        raise ValueError("arms must be a nonempty list without duplicates")
    if arm_set <= main_arms:
        if seeds != [20260817, 20260818, 20260819]:
            raise ValueError("the main control requires three frozen model seeds")
    elif arm_set <= sensitivity_arms:
        if seeds != [20260817]:
            raise ValueError("random-cohort sensitivity requires model seed 20260817 only")
    else:
        raise ValueError(
            "an invocation may contain only main arms or only random-sensitivity arms"
        )
    if set(folds) != set(range(5)):
        raise ValueError("five frozen family folds are required")
    if Path(args.overall_index_name).name != args.overall_index_name:
        raise ValueError("overall-index-name must be a basename")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    specifications = {
        "source": {
            "path": main_package / "gpu_cache/MODEL_READY.tsv.gz",
            "output": package / "gpu_output/source_fp32_benchmark",
            "label": "protein-anchored source",
            "expected": {
                "rows": 4637,
                "proteins": 426,
                "family_components": 165,
                "allosteric_rows": 1511,
                "orthosteric_rows": 3126,
            },
        },
        "balanced": {
            "path": package / "PROTEIN_ANCHORED_CHEMISTRY_BALANCED.tsv.gz",
            "output": package / "gpu_output/balanced_benchmark",
            "label": "protein-anchored, chemistry-balanced",
            "expected": {
                "rows": 3022,
                "proteins": 426,
                "family_components": 165,
                "allosteric_rows": 1511,
                "orthosteric_rows": 1511,
            },
        },
        "random": {
            "path": package / "PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM.tsv.gz",
            "output": package / "gpu_output/count_matched_random_benchmark",
            "label": "protein-anchored, count-matched random control",
            "expected": {
                "rows": 3022,
                "proteins": 426,
                "family_components": 165,
                "allosteric_rows": 1511,
                "orthosteric_rows": 1511,
            },
        },
        "random_seed_20260824": {
            "path": package / "PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260824.tsv.gz",
            "output": package / "gpu_output/count_matched_random_seed_20260824_benchmark",
            "label": "protein-anchored, count-matched random control seed 20260824",
            "expected": {
                "rows": 3022,
                "proteins": 426,
                "family_components": 165,
                "allosteric_rows": 1511,
                "orthosteric_rows": 1511,
            },
        },
        "random_seed_20260825": {
            "path": package / "PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260825.tsv.gz",
            "output": package / "gpu_output/count_matched_random_seed_20260825_benchmark",
            "label": "protein-anchored, count-matched random control seed 20260825",
            "expected": {
                "rows": 3022,
                "proteins": 426,
                "family_components": 165,
                "allosteric_rows": 1511,
                "orthosteric_rows": 1511,
            },
        },
    }
    frames = {}
    counts = {}
    for arm in arms:
        frame = pd.read_csv(specifications[arm]["path"], sep="\t", low_memory=False)
        observed = {
            "rows": int(len(frame)),
            "proteins": int(frame["uniprot"].nunique()),
            "family_components": int(frame["family_component_id"].nunique()),
            "allosteric_rows": int(frame["binary_label"].astype(int).sum()),
            "orthosteric_rows": int((1 - frame["binary_label"].astype(int)).sum()),
        }
        if observed != specifications[arm]["expected"]:
            raise RuntimeError("{} count contract changed: {}".format(arm, observed))
        if frame["main_row_id"].duplicated().any():
            raise RuntimeError("duplicate main_row_id in {}".format(arm))
        if not frame.groupby("uniprot")["binary_label"].nunique().eq(2).all():
            raise RuntimeError("protein anchoring was lost in {}".format(arm))
        if set(frame["family_fold"].astype(int)) != set(range(5)):
            raise RuntimeError("frozen family folds are incomplete in {}".format(arm))
        frames[arm] = frame
        counts[arm] = observed

    with (main_package / "data/POCKET_INDICES.json").open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    # The balanced arm is a row subset of the source arm. Loading the source
    # once gives an identical tensor universe to both training arms.
    source_frame = pd.read_csv(
        specifications["source"]["path"], sep="\t", low_memory=False
    )
    print("Preloading the frozen main-benchmark tensors", flush=True)
    caches = trainer.load_caches(source_frame, pocket_indices, args.max_atoms)
    print(
        "ligands={} proteins={} pockets={}".format(*(len(value) for value in caches)),
        flush=True,
    )

    device = torch.device(args.device)
    arm_indexes = {}
    for arm in arms:
        reports = []
        frame = frames[arm]
        output_root = specifications[arm]["output"]
        previous_index_path = output_root / "TRAINING_INDEX.json"
        if previous_index_path.is_file():
            try:
                previous = json.loads(previous_index_path.read_text(encoding="utf-8"))
                previous_models = previous.get("models", [])
                reusable = (
                    previous.get("status") == "validated"
                    and set(previous_models).issubset(set(models))
                    and previous.get("n_completed_fits")
                    == len(previous_models) * len(folds) * len(seeds)
                    and previous.get("folds") == folds
                    and previous.get("seeds") == seeds
                    and previous.get("cohort_counts") == counts[arm]
                    and previous.get("probability_conversion") == PROBABILITY_CONVERSION
                )
            except Exception:
                reusable = False
            if reusable:
                trainer.control_resume_models_by_root[
                    str(output_root.resolve())
                ] = set(previous_models)
                print("Authorized contract-matched resume for {}".format(arm), flush=True)
            else:
                print("Existing {} fits will be recomputed: contract index mismatch".format(arm), flush=True)
        for model in models:
            for seed in seeds:
                for fold in folds:
                    reports.append(
                        trainer.train_fit(
                            model, "unseen_family", seed, fold, args, frame,
                            caches, output_root, device,
                        )
                    )
        index = {
            "status": "validated" if all(row.get("status") == "validated" for row in reports) else "failed",
            "training_arm": arm,
            "cohort": specifications[arm]["label"],
            "cohort_path": str(specifications[arm]["path"]),
            "cohort_counts": counts[arm],
            "models": models,
            "regimes": ["unseen_family"],
            "folds": folds,
            "seeds": seeds,
            "n_expected_fits": int(len(models) * len(folds) * len(seeds)),
            "n_completed_fits": int(len(reports)),
            "maximum_epoch_hits": [
                {
                    "model": row["model"],
                    "seed": row["seed"],
                    "outer_fold": row["outer_fold"],
                }
                for row in reports if row.get("hit_maximum_epoch")
            ],
            "reused_main_trainer": str(main_package / "scripts/train_main_benchmark.py"),
            "probability_conversion": PROBABILITY_CONVERSION,
            "training_contract": {
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "training_weight": "equal total weight per protein-label group and then per label",
                "checkpoint_selection": "validation family-component-macro symmetric AP",
            },
        }
        trainer.atomic_json(output_root / "TRAINING_INDEX.json", index)
        arm_indexes[arm] = index
        print(json.dumps(index, indent=2, sort_keys=True))
        if (
            index["status"] != "validated"
            or index["n_completed_fits"] != index["n_expected_fits"]
        ):
            raise SystemExit("{} training did not complete its fit contract".format(arm))

    overall = {
        "status": "validated",
        "arms": arms,
        "models": models,
        "folds": folds,
        "seeds": seeds,
        "n_completed_fits": int(sum(value["n_completed_fits"] for value in arm_indexes.values())),
        "arm_indexes": {
            arm: str(specifications[arm]["output"] / "TRAINING_INDEX.json")
            for arm in arms
        },
    }
    overall["n_expected_fits"] = int(
        len(arms) * len(models) * len(folds) * len(seeds)
    )
    trainer.atomic_json(package / "gpu_output" / args.overall_index_name, overall)
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
