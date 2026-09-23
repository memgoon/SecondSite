#!/usr/bin/env python3
"""Fit fixed-epoch deploy ensembles on every Arm A training row."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    return parser.parse_args()


def load_module(path):
    spec = importlib.util.spec_from_file_location("frozen_main_trainer", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    root = args.project_root
    package = root / "analysis/chembl_external_prioritization"
    main_package = root / "analysis/allosteric_pair_benchmark_main"
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    trainer = load_module(main_package / "scripts/train_main_benchmark.py")
    frame = pd.read_csv(main_package / "gpu_cache/MODEL_READY.tsv.gz", sep="\t", low_memory=False)
    if len(frame) != 4637 or frame["uniprot"].nunique() != 426:
        raise RuntimeError("Arm A deploy cohort changed")
    with (main_package / "data/POCKET_INDICES.json").open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    frame["training_weight"] = trainer.class_and_protein_balanced_weights(frame)
    caches = trainer.load_caches(frame, pocket_indices, max_atoms=120)
    dataset = trainer.PairDataset(frame, caches, max_full_residues=4096)
    output_root = package / "gpu_output/deploy_models"
    device = torch.device(args.device)
    reports = []

    for model_name in config["models"]:
        epochs = int(config["deploy_epochs"][model_name])
        for seed in config["seeds"]:
            directory = output_root / model_name / "seed_{}".format(seed)
            checkpoint = directory / "deploy.pt"
            report_path = directory / "DEPLOY_REPORT.json"
            if checkpoint.is_file() and report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("status") == "validated" and report.get("epochs") == epochs:
                    print("SKIP deploy {} seed={}".format(model_name, seed), flush=True)
                    reports.append(report)
                    continue
            trainer.seed_everything(int(seed))
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                pin_memory=True, collate_fn=trainer.collate_pairs,
                generator=torch.Generator().manual_seed(int(seed)),
            )
            model = trainer.make_model(model_name, hidden=256, dropout=0.30, heads=4).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
            history = []
            for epoch in range(1, epochs + 1):
                model.train()
                losses = []
                for batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    tensors = trainer.move_batch(batch, device)
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
                history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
                print("deploy {} seed={} epoch={}/{} loss={:.6f}".format(
                    model_name, seed, epoch, epochs, history[-1]["train_loss"]
                ), flush=True)
            directory.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model": model_name,
                "seed": int(seed),
                "epochs": epochs,
                "training_rows": int(len(frame)),
                "training_proteins": int(frame["uniprot"].nunique()),
            }, temporary)
            os.replace(str(temporary), str(checkpoint))
            pd.DataFrame(history).to_csv(directory / "history.tsv", sep="\t", index=False)
            report = {
                "status": "validated", "model": model_name, "seed": int(seed),
                "epochs": epochs, "training_rows": int(len(frame)),
                "training_proteins": int(frame["uniprot"].nunique()),
                "checkpoint": str(checkpoint),
                "epoch_rule": "median best epoch across 15 completed unseen-family CV fits",
            }
            atomic_json(report_path, report)
            reports.append(report)

    index = {
        "status": "validated" if len(reports) == 9 and all(x.get("status") == "validated" for x in reports) else "failed",
        "training_pool": "Arm A proteins with both labels",
        "training_rows": int(len(frame)),
        "training_proteins": int(frame["uniprot"].nunique()),
        "models": config["models"], "seeds": config["seeds"],
        "n_expected": 9, "n_completed": int(len(reports)),
    }
    atomic_json(output_root / "DEPLOY_INDEX.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    if index["status"] != "validated":
        raise SystemExit("deploy training failed")


if __name__ == "__main__":
    main()
