#!/usr/bin/env python3
"""Train fixed-epoch deploy ensembles on all property-balanced rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--models", default="ligand,c2,c3")
    parser.add_argument("--index-name", default="DEPLOY_INDEX.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.project_root
    package = root / "analysis/property_balanced_chembl"
    main_package = root / "analysis/allosteric_pair_benchmark_main"
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    epoch_contract = config["c3_deploy_epoch"]
    epoch_audit_path = root / epoch_contract["audit_path"]
    if sha256(epoch_audit_path) != epoch_contract["audit_sha256"]:
        raise RuntimeError("deploy-epoch audit hash changed")
    epoch_audit = json.loads(epoch_audit_path.read_text(encoding="utf-8"))
    if (
        epoch_audit.get("status") != "PASS"
        or epoch_audit.get("selection_independent_of_chembl") is not True
        or epoch_audit.get("deploy_epochs") != config["deploy_epochs"]
    ):
        raise RuntimeError("deploy-epoch audit contract failed")
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    allowed = config["deploy_models"]
    if not models or len(models) != len(set(models)) or not set(models) <= set(allowed):
        raise ValueError("models must be a unique nonempty subset of {}".format(allowed))
    if Path(args.index_name).name != args.index_name:
        raise ValueError("index-name must be a basename")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    cohort_path = root / config["property_cohort"]["path"]
    observed_hash = sha256(cohort_path)
    if observed_hash != config["property_cohort"]["sha256"]:
        raise RuntimeError("property cohort hash changed")
    frame = pd.read_csv(cohort_path, sep="\t", low_memory=False)
    observed = {
        "rows": int(len(frame)),
        "proteins": int(frame["uniprot"].nunique()),
        "family_components": int(frame["family_component_id"].nunique()),
        "allosteric_rows": int(frame["binary_label"].astype(int).sum()),
        "orthosteric_rows": int((1 - frame["binary_label"].astype(int)).sum()),
    }
    expected = {
        key: int(config["property_cohort"][key])
        for key in observed
    }
    if observed != expected:
        raise RuntimeError("property cohort count contract changed: {}".format(observed))
    if frame["main_row_id"].duplicated().any():
        raise RuntimeError("duplicate property cohort row")
    if not frame.groupby("uniprot")["binary_label"].nunique().eq(2).all():
        raise RuntimeError("property cohort lost protein anchoring")

    output_root = package / "gpu_output/deploy_models"
    seeds = [int(x) for x in config["seeds"]]
    expected_reports = []
    pending = []
    for model_name in models:
        epochs = int(config["deploy_epochs"][model_name])
        for seed in seeds:
            directory = output_root / model_name / "seed_{}".format(seed)
            checkpoint = directory / "deploy.pt"
            report_path = directory / "DEPLOY_REPORT.json"
            report = None
            if checkpoint.is_file() and report_path.is_file():
                try:
                    candidate = json.loads(report_path.read_text(encoding="utf-8"))
                    if (
                        candidate.get("status") == "validated"
                        and candidate.get("contract_id") == config["contract_id"]
                        and candidate.get("model") == model_name
                        and candidate.get("seed") == seed
                        and candidate.get("epochs") == epochs
                        and candidate.get("training_rows") == 3022
                        and candidate.get("cohort_sha256") == observed_hash
                    ):
                        report = candidate
                except Exception:
                    report = None
            expected_reports.append((model_name, seed, epochs, directory, checkpoint, report_path))
            if report is None:
                pending.append((model_name, seed, epochs, directory, checkpoint, report_path))

    trainer = load_module(
        "property_deploy_trainer",
        main_package / "scripts/train_main_benchmark.py",
    )
    frame["training_weight"] = trainer.class_and_protein_balanced_weights(frame)
    caches = None
    dataset = None
    if pending:
        with (main_package / "data/POCKET_INDICES.json").open(encoding="utf-8") as handle:
            pocket_indices = json.load(handle)
        print("Preloading property deploy tensors", flush=True)
        caches = trainer.load_caches(frame, pocket_indices, max_atoms=120)
        dataset = trainer.PairDataset(frame, caches, max_full_residues=4096)
    device = torch.device(args.device)

    for model_name, seed, epochs, directory, checkpoint, report_path in pending:
        trainer.seed_everything(seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=trainer.collate_pairs,
            generator=torch.Generator().manual_seed(seed),
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
            print(
                "property deploy {} seed={} epoch={}/{} loss={:.6f}".format(
                    model_name, seed, epoch, epochs, history[-1]["train_loss"]
                ),
                flush=True,
            )
        directory.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save({
            "model_state_dict": model.state_dict(),
            "contract_id": config["contract_id"],
            "model": model_name,
            "seed": seed,
            "epochs": epochs,
            "training_rows": 3022,
            "training_proteins": 426,
            "training_family_components": 165,
            "cohort_sha256": observed_hash,
        }, temporary)
        os.replace(str(temporary), str(checkpoint))
        pd.DataFrame(history).to_csv(directory / "history.tsv", sep="\t", index=False)
        report = {
            "status": "validated",
            "contract_id": config["contract_id"],
            "model": model_name,
            "seed": seed,
            "epochs": epochs,
            "training_rows": 3022,
            "training_proteins": 426,
            "training_family_components": 165,
            "cohort_sha256": observed_hash,
            "checkpoint": str(checkpoint),
            "epoch_rule": "median best epoch across 15 property-balanced family-held-out fits",
        }
        atomic_json(report_path, report)
        del model

    reports = []
    for model_name, seed, epochs, directory, checkpoint, report_path in expected_reports:
        if not checkpoint.is_file() or not report_path.is_file():
            raise RuntimeError("missing deploy output: {} {}".format(model_name, seed))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "validated"
            or report.get("contract_id") != config["contract_id"]
            or report.get("cohort_sha256") != observed_hash
        ):
            raise RuntimeError("invalid deploy report: {}".format(report_path))
        reports.append(report)

    index = {
        "status": "validated",
        "contract_id": config["contract_id"],
        "training_pool": "property-balanced proteins with both labels",
        "cohort_sha256": observed_hash,
        "training_rows": 3022,
        "training_proteins": 426,
        "training_family_components": 165,
        "models": models,
        "seeds": seeds,
        "deploy_epochs": {name: int(config["deploy_epochs"][name]) for name in models},
        "n_expected": int(len(models) * len(seeds)),
        "n_completed": int(len(reports)),
    }
    atomic_json(output_root / args.index_name, index)
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
