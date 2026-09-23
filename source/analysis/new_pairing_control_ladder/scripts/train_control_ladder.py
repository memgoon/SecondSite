#!/usr/bin/env python3
"""Train C1/C2/C3/D1/graph-cross on the three frozen control stages."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--stages", default="uncontrolled,protein_controlled,fully_controlled")
    parser.add_argument("--models", default="c1,c2,c3,d1,graph_cross_pair")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
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


def load_model_helpers(project_root):
    path = project_root / "analysis/new_pairing_d1_pilot/scripts/train_family_model_pilot.py"
    spec = importlib.util.spec_from_file_location("pairing_model_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_and_protein_balanced_weights(train):
    size = train.groupby(["uniprot", "binary_label"])["pilot_row_id"].transform("size").astype(float)
    weight = 1.0 / size
    class_total = weight.groupby(train["binary_label"]).transform("sum")
    weight = weight / class_total.clip(lower=1e-12)
    return weight * (len(weight) / weight.sum())


def paired_subset_metrics(helper, prediction, stage_frame):
    paired_ids = set(stage_frame.loc[
        stage_frame["split"].eq("test") & stage_frame["paired_eval"].astype(int).eq(1), "pilot_row_id"
    ].astype(str))
    subset = prediction[prediction["pilot_row_id"].astype(str).isin(paired_ids)].copy()
    if not len(subset) or subset["binary_label"].nunique() < 2:
        return {"n_rows": int(len(subset)), "metrics": None}
    return {
        "n_rows": int(len(subset)),
        "n_proteins": int(subset["uniprot"].nunique()),
        "metrics": {
            "pooled": helper.binary_metrics(subset),
            "protein_macro": helper.grouped_metrics(subset, "uniprot"),
            "family_component_macro": helper.grouped_metrics(subset, "family_component_id"),
        },
    }


def train_one(helper, name, stage_name, args, frame, caches, output_dir, device):
    seed_everything(args.seed)
    train = frame[frame["split"].eq("train")].copy()
    validation = frame[frame["split"].eq("val")].copy()
    test = frame[frame["split"].eq("test")].copy()
    train["training_weight"] = class_and_protein_balanced_weights(train)
    validation["training_weight"] = 1.0
    test["training_weight"] = 1.0

    loaders = {}
    for split, subset, shuffle, batch_size in [
        ("train", train, True, args.batch_size),
        ("val", validation, False, args.eval_batch_size),
        ("test", test, False, args.eval_batch_size),
    ]:
        loaders[split] = DataLoader(
            helper.PairDataset(subset, caches, args.max_protein_residues),
            batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True,
            collate_fn=helper.collate_pairs,
        )

    model = helper.make_model(name, args.hidden_dim, args.dropout, args.heads).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    checkpoint = output_dir / "checkpoints/{}_best.pt".format(name)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_score = -math.inf
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            tensors = helper.move_batch(batch, device)
            label = batch["label"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(tensors)
                per_row = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
                loss = (per_row * weight).sum() / weight.sum().clamp_min(1e-8)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        _, metrics = helper.evaluate(model, loaders["val"], device)
        selection = metrics["family_component_macro"]["macro_symmetric_ap"]
        if selection is None:
            selection = metrics["pooled"]["symmetric_ap"]
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_pooled_symmetric_ap": metrics["pooled"]["symmetric_ap"],
            "val_protein_macro_symmetric_ap": metrics["protein_macro"]["macro_symmetric_ap"],
            "val_family_component_macro_symmetric_ap": metrics["family_component_macro"]["macro_symmetric_ap"],
            "selection_score": selection,
        })
        print("{} {} epoch={} loss={:.6f} selection={:.6f}".format(
            stage_name, name, epoch, history[-1]["train_loss"], selection
        ), flush=True)
        if selection > best_score + 1e-6:
            best_score = float(selection)
            best_epoch = epoch
            stale = 0
            torch.save({"model_state_dict": model.state_dict(), "model": name, "epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    prediction, test_metrics = helper.evaluate(model, loaders["test"], device)
    prediction_path = output_dir / "predictions/{}_test_predictions.tsv".format(name)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(prediction_path, sep="\t", index=False)
    history_path = output_dir / "metrics/{}_history.tsv".format(name)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, sep="\t", index=False)
    report = {
        "model": name,
        "n_parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "test": test_metrics,
        "test_two_label_protein_subset": paired_subset_metrics(helper, prediction, frame),
        "prediction_path": str(prediction_path),
        "checkpoint_path": str(checkpoint),
    }
    atomic_json(output_dir / "metrics/{}_metrics.json".format(name), report)
    return report, prediction


def main():
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("CUDA requested but unavailable")
    helper = load_model_helpers(args.project_root)
    package = args.project_root / "analysis/new_pairing_control_ladder"
    stage_dir = package / "gpu_cache/stages"
    stage_paths = {
        "uncontrolled": stage_dir / "UNCONTROLLED.tsv.gz",
        "protein_controlled": stage_dir / "PROTEIN_CONTROLLED.tsv.gz",
        "fully_controlled": stage_dir / "FULLY_CONTROLLED.tsv.gz",
    }
    requested_stages = [value.strip() for value in args.stages.split(",") if value.strip()]
    requested_models = [value.strip() for value in args.models.split(",") if value.strip()]
    if set(requested_stages) - set(stage_paths):
        raise ValueError("unknown stage")
    expected_models = {"c1", "c2", "c3", "d1", "graph_cross_pair"}
    if set(requested_models) - expected_models:
        raise ValueError("unknown model")

    frames = {name: pd.read_csv(stage_paths[name], sep="\t", low_memory=False) for name in requested_stages}
    cache_frame = pd.concat(frames.values(), ignore_index=True).drop_duplicates("pilot_row_id")
    print("Preloading exact embedding and graph caches", flush=True)
    caches = helper.load_caches(cache_frame, args.max_atoms)
    print("ligands={} proteins={} graphs={}".format(*(len(cache) for cache in caches)), flush=True)
    device = torch.device(args.device)
    final_stages = {}
    for stage_name in requested_stages:
        frame = frames[stage_name]
        output_dir = package / "gpu_output" / stage_name
        reports = {}
        predictions = {}
        for model_name in requested_models:
            reports[model_name], predictions[model_name] = train_one(
                helper, model_name, stage_name, args, frame, caches, output_dir, device
            )
        universes = {name: tuple(value["pilot_row_id"].astype(str)) for name, value in predictions.items()}
        prediction_universe_exact = len(set(universes.values())) == 1
        deltas = {}
        comparisons = [
            ("c2", "c1"), ("c3", "c1"), ("c3", "c2"),
            ("d1", "c1"), ("d1", "c3"),
            ("graph_cross_pair", "c1"), ("graph_cross_pair", "c3"), ("graph_cross_pair", "d1"),
        ]
        endpoints = [("pooled", "symmetric_ap"), ("protein_macro", "macro_symmetric_ap"), ("family_component_macro", "macro_symmetric_ap")]
        for model, reference in comparisons:
            if model not in reports or reference not in reports:
                continue
            for section, key in endpoints:
                left = reports[model]["test"][section][key]
                right = reports[reference]["test"][section][key]
                if left is not None and right is not None:
                    deltas["{}_minus_{}_{}_{}".format(model, reference, section, key)] = float(left - right)
        stage_report = {
            "status": "validated" if prediction_universe_exact else "failed",
            "stage": stage_name,
            "dataset_rows": int(len(frame)),
            "dataset_proteins": int(frame["uniprot"].nunique()),
            "split_counts": frame.groupby("split").size().astype(int).to_dict(),
            "models": reports,
            "deltas": deltas,
            "prediction_universe_exact": prediction_universe_exact,
        }
        atomic_json(output_dir / "STAGE_REPORT.json", stage_report)
        final_stages[stage_name] = stage_report

    final = {
        "status": "validated" if all(value["status"] == "validated" for value in final_stages.values()) else "failed",
        "seed": args.seed,
        "models": requested_models,
        "training_weight": "equal total weight per protein-label group, then equal total weight between labels",
        "selection_endpoint": "validation family-component-macro symmetric AP; pooled symmetric AP fallback",
        "graph_model_limit": "reduced LABind-style pair adaptation, not official LABind",
        "c1_definition": "mean-pooled UniMol atoms concatenated with attention-pooled union-pocket ESM3 residues; no protein-ligand cross-attention",
        "stages": final_stages,
    }
    atomic_json(package / "gpu_output/CONTROL_LADDER_REPORT.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    if final["status"] != "validated":
        raise SystemExit("control ladder training validation failed")


if __name__ == "__main__":
    main()
