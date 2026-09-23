#!/usr/bin/env python3
"""Score seed-20260818 random-draw checkpoints on fixed property rows."""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


MODELS = ["ligand", "c2", "c3"]
FOLDS = list(range(5))
MODEL_SEED = 20260818


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_fp32(trainer, model, loader, device):
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
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--max-atoms", type=int, default=120)
    parser.add_argument("--max-protein-residues", type=int, default=4096)
    args = parser.parse_args()

    root = args.project_root
    package = root / "analysis/property_balanced_chembl"
    ligand_package = root / "analysis/ligand_chemistry_balancing"
    main_package = root / "analysis/allosteric_pair_benchmark_main"
    trainer = load_module(
        "extension_scoring_trainer",
        main_package / "scripts/train_main_benchmark.py",
    )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)

    draws = {
        "random_seed_20260824": (
            ligand_package / "PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260824.tsv.gz",
            package / "gpu_output/random_extension/selection_seed_20260824",
        ),
        "random_seed_20260825": (
            ligand_package / "PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260825.tsv.gz",
            package / "gpu_output/random_extension/selection_seed_20260825",
        ),
    }
    frames = {
        name: pd.read_csv(path, sep="\t", low_memory=False)
        for name, (path, _) in draws.items()
    }
    evaluation = pd.read_csv(
        ligand_package / "PROTEIN_ANCHORED_CHEMISTRY_BALANCED.tsv.gz",
        sep="\t", low_memory=False,
    )
    if len(evaluation) != 3022 or evaluation["main_row_id"].duplicated().any():
        raise RuntimeError("property evaluation contract changed")

    source = pd.read_csv(main_package / "gpu_cache/MODEL_READY.tsv.gz", sep="\t", low_memory=False)
    with (main_package / "data/POCKET_INDICES.json").open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    print("Preloading frozen tensors for extension rescoring", flush=True)
    caches = trainer.load_caches(source, pocket_indices, args.max_atoms)

    all_predictions = []
    for draw, (_, output_root) in draws.items():
        index = json.loads((output_root / "TRAINING_INDEX.json").read_text(encoding="utf-8"))
        if index.get("status") != "validated" or index.get("n_completed_fits") != 15:
            raise RuntimeError("incomplete extension draw: {}".format(draw))
        training_frame = frames[draw]
        for model_name in MODELS:
            for fold in FOLDS:
                train, validation, _, _ = trainer.split_frame(training_frame, "unseen_family", fold)
                development = set(train["connectivity_key"].astype(str)) | set(
                    validation["connectivity_key"].astype(str)
                )
                model = trainer.make_model(
                    model_name, args.hidden_dim, args.dropout, args.heads
                ).to(device)
                checkpoint = (
                    output_root / "fits/unseen_family" / model_name
                    / "seed_{}".format(MODEL_SEED) / "fold_{}".format(fold) / "best.pt"
                )
                saved = torch.load(checkpoint, map_location=device)
                model.load_state_dict(saved["model_state_dict"])
                test = evaluation[evaluation["family_fold"].astype(int).eq(fold)].copy()
                loader = DataLoader(
                    trainer.PairDataset(test, caches, args.max_protein_residues),
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=trainer.collate_pairs,
                )
                prediction = evaluate_fp32(trainer, model, loader, device)
                prediction["unseen_compound"] = (
                    ~prediction["connectivity_key"].astype(str).isin(development)
                ).astype(int)
                prediction["random_draw"] = draw
                prediction["model"] = model_name
                prediction["model_seed"] = MODEL_SEED
                prediction["outer_fold"] = int(fold)
                all_predictions.append(prediction)
                del saved, model

    predictions = pd.concat(all_predictions, ignore_index=True)
    expected_rows = 3022 * 2 * 3
    if len(predictions) != expected_rows:
        raise RuntimeError("extension scoring row count changed")
    key = ["random_draw", "model", "main_row_id"]
    if predictions.duplicated(key).any():
        raise RuntimeError("duplicate extension prediction")
    expected_ids = set(evaluation["main_row_id"].astype(str))
    for (draw, model_name), group in predictions.groupby(["random_draw", "model"]):
        if set(group["main_row_id"].astype(str)) != expected_ids:
            raise RuntimeError("evaluation row mismatch: {} {}".format(draw, model_name))

    output = package / "gpu_output/random_extension_scoring"
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / "ALL_PREDICTIONS.tsv.gz"
    predictions.to_csv(output_path, sep="\t", index=False, compression="gzip")
    flags = predictions[predictions["model"].eq("ligand")].pivot(
        index="main_row_id", columns="random_draw", values="unseen_compound"
    )
    common_unseen = set(flags.index[flags.astype(int).eq(1).all(axis=1)].astype(str))
    report = {
        "status": "validated",
        "random_draws": list(draws),
        "model_seed": MODEL_SEED,
        "models": MODELS,
        "folds": FOLDS,
        "evaluation_universe": "property_balanced_rows",
        "evaluation_rows": 3022,
        "prediction_rows": int(len(predictions)),
        "common_unseen_connectivity_rows": int(len(common_unseen)),
        "probability_conversion": "sigmoid applied after FP32 logit cast",
        "output": str(output_path),
    }
    (output / "VALIDATION.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
