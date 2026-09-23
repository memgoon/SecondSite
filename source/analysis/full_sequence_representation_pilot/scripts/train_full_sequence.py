#!/usr/bin/env python3
"""Retrain protein-only and C1-C3 under matched selected-chain and full-sequence representations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


MODELS = ("protein", "c1", "c2", "c3")
REPRESENTATIONS = ("full_canonical_uniprot", "rechunked_selected_structure_chain")
SEEDS = (20260817, 20260818, 20260819)
FOLDS = tuple(range(5))
MODEL_VERSION = "full_sequence_representation_pilot_v2"
PROBABILITY_CONVERSION = "sigmoid applied after FP32 logit cast"
PROTEIN_DIM = 1536
LIGAND_DIM = 512


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--folds", default=",".join(map(str, FOLDS)))
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--c3-batch-size", type=int, default=1)
    parser.add_argument("--c3-eval-batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-atoms", type=int, default=120)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def tensor_from_object(value, dimension: int) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        candidates = [x for x in value.values() if torch.is_tensor(x) and x.ndim == 2 and int(x.shape[-1]) == dimension]
        if len(candidates) != 1:
            raise TypeError("expected one compatible tensor")
        tensor = candidates[0]
    else:
        raise TypeError("unsupported tensor object")
    tensor = tensor.detach().cpu().float()
    if tensor.ndim != 2 or int(tensor.shape[1]) != dimension or int(tensor.shape[0]) < 1:
        raise ValueError("invalid tensor shape")
    return tensor.contiguous()


class FullSequenceDataset(Dataset):
    def __init__(self, frame, ligand_cache, protein_cache):
        self.frame = frame.reset_index(drop=True)
        self.ligand_cache = ligand_cache
        self.protein_cache = protein_cache

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return {
            "ligand": self.ligand_cache[str(row["ligand_embedding_path"])],
            "protein": self.protein_cache[str(row["uniprot"])],
            "label": float(row["binary_label"]),
            "weight": float(row.get("training_weight", 1.0)),
            "main_row_id": str(row["main_row_id"]),
            "uniprot": str(row["uniprot"]),
            "family_component_id": str(row["family_component_id"]),
            "full_inchikey": str(row["full_inchikey"]),
            "connectivity_key": str(row["connectivity_key"]),
            "class_label": str(row["class_label"]),
        }


def collate(items):
    batch = len(items)
    max_atoms = max(int(x["ligand"].shape[0]) for x in items)
    max_residues = max(int(x["protein"].shape[0]) for x in items)
    ligand = torch.zeros(batch, max_atoms, LIGAND_DIM)
    ligand_mask = torch.zeros(batch, max_atoms, dtype=torch.bool)
    protein = torch.zeros(batch, max_residues, PROTEIN_DIM)
    protein_mask = torch.zeros(batch, max_residues, dtype=torch.bool)
    for position, item in enumerate(items):
        na, nr = int(item["ligand"].shape[0]), int(item["protein"].shape[0])
        ligand[position, :na] = item["ligand"]
        ligand_mask[position, :na] = True
        protein[position, :nr] = item["protein"]
        protein_mask[position, :nr] = True
    result = {
        "ligand": ligand,
        "ligand_mask": ligand_mask,
        "protein": protein,
        "protein_mask": protein_mask,
        "label": torch.tensor([x["label"] for x in items], dtype=torch.float32),
        "weight": torch.tensor([x["weight"] for x in items], dtype=torch.float32),
    }
    for key in ["main_row_id", "uniprot", "family_component_id", "full_inchikey", "connectivity_key", "class_label"]:
        result[key] = [x[key] for x in items]
    return result


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def paths_for(output: Path, representation: str, model: str, seed: int, fold: int):
    directory = output / "fits" / representation / model / "seed_{}".format(seed) / "fold_{}".format(fold)
    return {
        "dir": directory,
        "checkpoint": directory / "best.pt",
        "prediction": directory / "predictions.tsv.gz",
        "history": directory / "history.tsv",
        "report": directory / "FIT_REPORT.json",
    }


def move(batch, device):
    return {key: batch[key].to(device, non_blocking=True) for key in ["ligand", "ligand_mask", "protein", "protein_mask"]}


@torch.no_grad()
def evaluate(model, loader, device, matrix):
    model.eval()
    rows = []
    for batch in loader:
        tensors = move(batch, device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(tensors)
        probability = torch.sigmoid(logits.float()).cpu().numpy()
        for index, value in enumerate(probability):
            rows.append(
                {
                    "main_row_id": batch["main_row_id"][index],
                    "uniprot": batch["uniprot"][index],
                    "family_component_id": batch["family_component_id"][index],
                    "full_inchikey": batch["full_inchikey"][index],
                    "connectivity_key": batch["connectivity_key"][index],
                    "class_label": batch["class_label"][index],
                    "binary_label": int(batch["label"][index].item()),
                    "p_allosteric": float(value),
                }
            )
    prediction = pd.DataFrame(rows)
    return prediction, matrix.metric_bundle(prediction)


def main() -> None:
    args = parse_args()
    models = [x for x in args.models.split(",") if x]
    representations = [x for x in args.representations.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    folds = [int(x) for x in args.folds.split(",") if x]
    if (
        not models or set(models) - set(MODELS) or not representations
        or set(representations) - set(REPRESENTATIONS) or not seeds or set(folds) - set(FOLDS)
    ):
        raise ValueError("invalid model/seed/fold selection")
    if args.world_size < 1 or not 0 <= args.worker_rank < args.world_size:
        raise ValueError("invalid worker geometry")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    root = args.project_root.resolve()
    package = root / "analysis/full_sequence_representation_pilot"
    role = root / "analysis/role_complete_pair_matrix"
    sys.path.insert(0, str(role / "scripts"))
    matrix = load_module("fullseq_matrix", role / "scripts/train_matrix.py")
    models_module = load_module("fullseq_models", role / "scripts/model_definitions.py")
    base = load_module("fullseq_base", root / "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py")
    contract = json.loads((package / "validation/CPU_CONTRACT.json").read_text())
    if contract.get("status") != "validated":
        raise RuntimeError("CPU contract is not validated")
    frame = pd.read_csv(package / "data/PILOT_COHORT.tsv.gz", sep="\t")
    ligand_cache = {
        path: tensor_from_object(torch.load(path, map_location="cpu"), LIGAND_DIM)[: args.max_atoms]
        for path in sorted(frame["ligand_embedding_path"].astype(str).unique())
    }
    layouts = {
        "full_canonical_uniprot": {
            "manifest": package / "data/FULL_SEQUENCE_MANIFEST.tsv",
            "embedding_directory": package / "gpu_cache/full_sequence_embeddings",
            "length_column": "canonical_length",
            "validation": package / "gpu_output/FULL_SEQUENCE_EMBEDDING_VALIDATION.json",
            "files": package / "gpu_output/FULL_SEQUENCE_EMBEDDING_FILES.tsv",
            "description": "full canonical UniProt sequence, overlap-chunked ESM3 and stitched without downstream truncation",
        },
        "rechunked_selected_structure_chain": {
            "manifest": package / "data/RECHUNKED_SELECTED_CHAIN_MANIFEST.tsv",
            "embedding_directory": package / "gpu_cache/rechunked_selected_chain_embeddings",
            "length_column": "selected_chain_length",
            "validation": package / "gpu_output/RECHUNKED_SELECTED_CHAIN_EMBEDDING_VALIDATION.json",
            "files": package / "gpu_output/RECHUNKED_SELECTED_CHAIN_EMBEDDING_FILES.tsv",
            "description": "frozen selected PDB target chain, re-embedded with the identical overlap-chunk and stitching procedure",
        },
    }
    full_coverage_manifest = pd.read_csv(
        package / "data/FULL_SEQUENCE_MANIFEST.tsv", sep="\t"
    ).set_index("uniprot")
    manifests, protein_caches = {}, {}
    for representation in representations:
        layout = layouts[representation]
        embedding_validation = json.loads(layout["validation"].read_text())
        if embedding_validation.get("status") != "validated" or embedding_validation.get("representation") != representation:
            raise RuntimeError("embedding contract is not validated for {}".format(representation))
        manifest = pd.read_csv(layout["manifest"], sep="\t").set_index("uniprot")
        cache = {}
        for uid in sorted(frame["uniprot"].astype(str).unique()):
            path = layout["embedding_directory"] / (uid + ".pt")
            tensor = tensor_from_object(torch.load(path, map_location="cpu"), PROTEIN_DIM)
            if int(tensor.shape[0]) != int(manifest.loc[uid, layout["length_column"]]):
                raise RuntimeError("{} sequence length mismatch for {}".format(representation, uid))
            cache[uid] = tensor
        manifests[representation] = manifest
        protein_caches[representation] = cache
    output = package / "gpu_output/benchmark"
    frozen_training_contract_base = {
        "data_contract_id": contract["contract_id"],
        "cpu_contract_sha256": sha256(package / "validation/CPU_CONTRACT.json"),
        "trainer_sha256": sha256(package / "scripts/train_full_sequence.py"),
        "model_definitions_sha256": sha256(role / "scripts/model_definitions.py"),
        "split": "frozen protein-anchored unseen-family folds",
        "epochs": args.epochs,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "c3_batch_size": args.c3_batch_size,
        "c3_eval_batch_size": args.c3_eval_batch_size,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_atoms": args.max_atoms,
        "protein_truncation": "none",
        "training_weight": "same protein-label balanced weights as selected-chain benchmark",
        "checkpoint_selection": "same model-aware unseen-family within-ligand validation rule as selected-chain benchmark",
        "probability_conversion": PROBABILITY_CONVERSION,
    }
    tasks = [
        (representation, model, seed, fold)
        for representation in representations for model in models for seed in seeds for fold in folds
    ]
    tasks = [task for index, task in enumerate(tasks) if index % args.world_size == args.worker_rank]
    print("worker {} of {} assigned {} fits".format(args.worker_rank, args.world_size, len(tasks)), flush=True)

    for representation, model_name, seed, fold in tasks:
        train, validation, test, validation_fold = matrix.split_frame(
            frame, "protein_anchored", "unseen_family", fold, base.class_and_protein_balanced_weights
        )
        paths = paths_for(output, representation, model_name, seed, fold)
        layout = layouts[representation]
        frozen_training_contract = dict(frozen_training_contract_base)
        frozen_training_contract.update(
            {
                "representation": layout["description"],
                "embedding_validation_sha256": sha256(layout["validation"]),
                "embedding_files_manifest_sha256": sha256(layout["files"]),
                "representation_manifest_sha256": sha256(layout["manifest"]),
            }
        )
        identity = {
            "status": "validated",
            "representation": representation,
            "model": model_name,
            "seed": int(seed),
            "outer_fold": int(fold),
            "validation_fold": int(validation_fold),
            "model_version": MODEL_VERSION,
        }
        if paths["report"].is_file() and paths["prediction"].is_file() and paths["checkpoint"].is_file():
            try:
                old = json.loads(paths["report"].read_text())
                observed = pd.read_csv(paths["prediction"], sep="\t", usecols=["main_row_id"])
                if (
                    all(old.get(k) == v for k, v in identity.items())
                    and old.get("probability_conversion") == PROBABILITY_CONVERSION
                    and old.get("training_contract") == frozen_training_contract
                    and old.get("checkpoint_sha256") == sha256(paths["checkpoint"])
                    and old.get("prediction_sha256") == sha256(paths["prediction"])
                    and set(observed["main_row_id"].astype(str)) == set(test["main_row_id"].astype(str))
                ):
                    print("SKIP {} {} seed={} fold={}".format(representation, model_name, seed, fold), flush=True)
                    continue
            except Exception:
                pass

        seed_all(seed)
        train_batch = args.c3_batch_size if model_name == "c3" else args.batch_size
        eval_batch = args.c3_eval_batch_size if model_name == "c3" else args.eval_batch_size
        loaders = {}
        for name, subset, shuffle, batch_size in [
            ("train", train, True, train_batch),
            ("validation", validation, False, eval_batch),
            ("test", test, False, eval_batch),
        ]:
            loaders[name] = DataLoader(
                FullSequenceDataset(subset, ligand_cache, protein_caches[representation]),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate,
                generator=torch.Generator().manual_seed(seed) if shuffle else None,
            )
        model = models_module.make_model(model_name, args.hidden_dim, args.dropout, args.heads).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        best_score, best_epoch, stale = -math.inf, -1, 0
        history = []
        paths["dir"].mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                tensors = move(batch, device)
                labels = batch["label"].to(device, non_blocking=True)
                weights = batch["weight"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(tensors)
                    per_row = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                    loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-8)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            _, validation_metrics = evaluate(model, loaders["validation"], device, matrix)
            selection, metric_name, fallback, fallback_reason, support = matrix.checkpoint_selection(
                validation_metrics, "unseen_family", model_name
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "selection_score": float(selection),
                    "selection_metric": metric_name,
                    "selection_fallback": bool(fallback),
                    "selection_fallback_reason": fallback_reason,
                    "selection_support": json.dumps(support, sort_keys=True),
                }
            )
            print(
                "{} unseen_family {} seed={} fold={} epoch={} loss={:.6f} selection={:.6f}".format(
                    representation, model_name, seed, fold, epoch, history[-1]["train_loss"], selection
                ),
                flush=True,
            )
            if selection > best_score + 1e-6:
                best_score, best_epoch, stale = float(selection), int(epoch), 0
                temporary = paths["checkpoint"].with_suffix(".pt.tmp")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": model_name,
                        "model_version": MODEL_VERSION,
                        "representation": representation,
                        "seed": seed,
                        "outer_fold": fold,
                        "epoch": epoch,
                    },
                    temporary,
                )
                os.replace(str(temporary), str(paths["checkpoint"]))
            else:
                stale += 1
            if epoch >= args.min_epochs and stale >= args.patience:
                break
        saved = torch.load(paths["checkpoint"], map_location=device)
        model.load_state_dict(saved["model_state_dict"])
        prediction, metrics = evaluate(model, loaders["test"], device, matrix)
        prediction["cohort_arm"] = "protein_anchored"
        prediction["regime"] = "unseen_family"
        prediction["model"] = model_name
        prediction["representation"] = representation
        prediction["seed"] = seed
        prediction["outer_fold"] = fold
        prediction["validation_fold"] = validation_fold
        prediction = prediction.merge(
            full_coverage_manifest[["aligned_canonical_fraction", "coverage_stratum", "canonical_length_gt_4096"]],
            left_on="uniprot",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
        prediction.to_csv(paths["prediction"], sep="\t", index=False, compression="gzip")
        pd.DataFrame(history).to_csv(paths["history"], sep="\t", index=False)
        report = dict(identity)
        report.update(
            {
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "test_metrics": metrics,
                "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
                "probability_conversion": PROBABILITY_CONVERSION,
                "training_contract": frozen_training_contract,
                "checkpoint_sha256": sha256(paths["checkpoint"]),
                "prediction_sha256": sha256(paths["prediction"]),
            }
        )
        atomic_json(paths["report"], report)


if __name__ == "__main__":
    main()
