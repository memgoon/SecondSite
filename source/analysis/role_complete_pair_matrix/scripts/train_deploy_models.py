#!/usr/bin/env python3
"""Fit fixed-epoch deploy ensembles after the internal matrix is complete."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_definitions import (  # noqa: E402
    MODEL_VERSION,
    VALID_MODELS,
    make_model,
    model_contract,
)


DEPLOY_MODELS = tuple(VALID_MODELS)
DEPLOY_ARMS = {
    "general_every_pair": "EVERY_PAIR.tsv.gz",
    "general_protein_anchored": "PROTEIN_ANCHORED.tsv.gz",
    "biochemical_role_complete": "PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz",
}
SEEDS = (20260817, 20260818, 20260819)
EPOCH_RULE = (
    "median best epoch across the 15 unseen-family CV fits for the same cohort and model"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--full-bidirectional-batch-size", type=int, default=1)
    return parser.parse_args()


def load_base(root):
    path = root / "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
    spec = importlib.util.spec_from_file_location("deploy_base", str(path))
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


def canonical_fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def expected_cv_training_contract(cpu_contract, source_arm):
    filename = {
        "every_pair": "EVERY_PAIR.tsv.gz",
        "protein_anchored": "PROTEIN_ANCHORED.tsv.gz",
        "protein_ligand_role_complete": "PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz",
    }[source_arm]
    return {
        "data_contract_id": "role_complete_pair_matrix_v2",
        "cohort_sha256": cpu_contract["files"]["data/" + filename]["sha256"],
        "model_implementation_sha256": cpu_contract["implementation_files"][
            "analysis/role_complete_pair_matrix/scripts/model_definitions.py"
        ]["sha256"],
        "base_trainer_sha256": cpu_contract["implementation_files"][
            "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
        ]["sha256"],
        "matrix_trainer_sha256": cpu_contract["implementation_files"][
            "analysis/role_complete_pair_matrix/scripts/train_matrix.py"
        ]["sha256"],
        "epochs": 25,
        "patience": 5,
        "min_epochs": 1,
        "hidden_dim": 256,
        "heads": 4,
        "dropout": 0.30,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 6,
        "eval_batch_size": 8,
        "full_bidirectional_batch_size": 1,
        "full_bidirectional_eval_batch_size": 2,
        "max_atoms": 120,
        "max_protein_residues": 4096,
        "training_weight": "equal total weight per protein-label group and then per label; no ligand balancing",
        "checkpoint_selection": (
            "model-aware: ligand-only=validation within-protein macro AUROC; "
            "protein-only=validation within-ligand macro AUROC; joint "
            "unseen_family=within-ligand, joint unseen_ligand=within-protein, "
            "joint row_random=unweighted mean of both; every requested metric "
            "requires at least 8 valid groups; pooled symmetric AP fallback otherwise"
        ),
        "evaluation_contract": (
            "protein-anchored rows for every_pair and protein_anchored; "
            "role-complete rows for protein_ligand_role_complete"
        ),
    }


def epoch_contract(package, deploy_arm, model, cpu_contract):
    fit_root = package / "gpu_output/benchmark/fits"
    source_arm = {
        "general_every_pair": "every_pair",
        "general_protein_anchored": "protein_anchored",
        "biochemical_role_complete": "protein_ligand_role_complete",
    }[deploy_arm]
    # Use one prespecified generalization regime for every deployment arm.
    # Pooling family- and ligand-held-out epoch distributions would create an
    # arbitrary optimization target that was not used by either CV regime.
    regimes = ["unseen_family"]
    expected_training = expected_cv_training_contract(cpu_contract, source_arm)
    epochs = []
    records = []
    for regime in regimes:
        for seed in SEEDS:
            for fold in range(5):
                path = fit_root / source_arm / regime / model / "seed_{}".format(seed) / "fold_{}".format(fold) / "FIT_REPORT.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                report = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = path.parent / "best.pt"
                prediction = path.parent / "predictions.tsv.gz"
                if (
                    report.get("status") != "validated"
                    or report.get("model_version") != MODEL_VERSION
                    or report.get("cohort_arm") != source_arm
                    or report.get("regime") != regime
                    or report.get("model") != model
                    or int(report.get("seed", -1)) != seed
                    or int(report.get("outer_fold", -1)) != fold
                    or int(report.get("validation_fold", -1)) != (fold + 1) % 5
                    or report.get("model_contract") != model_contract(model)
                    or report.get("training_contract") != expected_training
                    or report.get("checkpoint_selection_contract")
                    != expected_training["checkpoint_selection"]
                    or report.get("checkpoint_selection_model_aware") is not True
                    or report.get(
                        "checkpoint_selection_structurally_constant_expected"
                    )
                    is not False
                    or int(
                        report.get(
                            "checkpoint_selection_minimum_valid_groups", -1
                        )
                    )
                    != 8
                    or not isinstance(
                        report.get(
                            "checkpoint_selection_requested_group_counts_at_best_epoch"
                        ),
                        dict,
                    )
                    or not checkpoint.is_file()
                    or not prediction.is_file()
                    or report.get("checkpoint_sha256") != sha256(checkpoint)
                    or report.get("prediction_sha256") != sha256(prediction)
                ):
                    raise RuntimeError("invalid CV report: {}".format(path))
                epochs.append(int(report["best_epoch"]))
                records.append(
                    {
                        "report_path": str(path.relative_to(package)),
                        "report_sha256": sha256(path),
                        "checkpoint_sha256": report["checkpoint_sha256"],
                        "prediction_sha256": report["prediction_sha256"],
                        "cohort_arm": source_arm,
                        "regime": regime,
                        "model": model,
                        "seed": int(seed),
                        "outer_fold": int(fold),
                        "validation_fold": int((fold + 1) % 5),
                        "best_epoch": int(report["best_epoch"]),
                    }
                )
    if len(records) != 15 or len(
        {
            (row["seed"], row["outer_fold"])
            for row in records
        }
    ) != 15:
        raise RuntimeError("deploy epoch source must contain exactly 15 unique CV fits")
    fingerprint_payload = {
        "epoch_rule": EPOCH_RULE,
        "source_arm": source_arm,
        "model": model,
        "records": records,
    }
    return {
        "epochs": max(1, int(np.median(epochs))),
        "source_best_epochs": epochs,
        "source_report_records": records,
        "source_fingerprint": canonical_fingerprint(fingerprint_payload),
    }


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    root = args.project_root.resolve()
    package = root / "analysis/role_complete_pair_matrix"
    cpu_contract = json.loads(
        (package / "validation/CPU_CONTRACT.json").read_text(encoding="utf-8")
    )
    if (
        cpu_contract.get("status") != "validated"
        or cpu_contract.get("contract_id") != "role_complete_pair_matrix_v2"
    ):
        raise RuntimeError("CPU deployment contract is invalid")
    base = load_base(root)
    with (root / "analysis/allosteric_pair_benchmark_broad_superset/data/POCKET_INDICES.json").open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    device = torch.device(args.device)
    output_root = package / "gpu_output/deploy"
    reports = []
    for deploy_arm, filename in DEPLOY_ARMS.items():
        frame = pd.read_csv(package / "data" / filename, sep="\t", low_memory=False)
        training_data_sha256 = cpu_contract["files"]["data/" + filename]["sha256"]
        implementation_sha256 = cpu_contract["implementation_files"][
            "analysis/role_complete_pair_matrix/scripts/model_definitions.py"
        ]["sha256"]
        deploy_trainer_sha256 = cpu_contract["implementation_files"][
            "analysis/role_complete_pair_matrix/scripts/train_deploy_models.py"
        ]["sha256"]
        base_trainer_sha256 = cpu_contract["implementation_files"][
            "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
        ]["sha256"]
        frame["training_weight"] = base.class_and_protein_balanced_weights(frame)
        caches = base.load_caches(frame, pocket_indices, max_atoms=120)
        dataset = base.PairDataset(frame, caches, max_full_residues=4096)
        for model_name in DEPLOY_MODELS:
            epoch_source = epoch_contract(
                package, deploy_arm, model_name, cpu_contract
            )
            epochs = int(epoch_source["epochs"])
            source_epochs = epoch_source["source_best_epochs"]
            source_records = epoch_source["source_report_records"]
            epoch_source_fingerprint = epoch_source["source_fingerprint"]
            for seed in SEEDS:
                directory = output_root / deploy_arm / model_name / "seed_{}".format(seed)
                checkpoint = directory / "deploy.pt"
                report_path = directory / "DEPLOY_REPORT.json"
                if checkpoint.is_file() and report_path.is_file():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if (
                        report.get("status") == "validated"
                        and report.get("deploy_arm") == deploy_arm
                        and report.get("model") == model_name
                        and int(report.get("seed", -1)) == seed
                        and report.get("epochs") == epochs
                        and report.get("model_version") == MODEL_VERSION
                        and report.get("training_rows") == len(frame)
                        and report.get("contract_id") == "role_complete_pair_matrix_v2"
                        and report.get("training_data_sha256") == training_data_sha256
                        and report.get("model_implementation_sha256")
                        == implementation_sha256
                        and report.get("deploy_trainer_sha256")
                        == deploy_trainer_sha256
                        and report.get("base_trainer_sha256")
                        == base_trainer_sha256
                        and report.get("checkpoint_sha256") == sha256(checkpoint)
                        and report.get("epoch_rule") == EPOCH_RULE
                        and report.get("epoch_source_regimes")
                        == ["unseen_family"]
                        and report.get("epoch_source_fingerprint")
                        == epoch_source_fingerprint
                        and report.get("source_report_records") == source_records
                    ):
                        reports.append(report)
                        print("SKIP deploy {} {} seed={}".format(deploy_arm, model_name, seed), flush=True)
                        continue
                base.seed_everything(int(seed))
                batch_size = args.full_bidirectional_batch_size if model_name == "c3" else args.batch_size
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=base.collate_pairs,
                    generator=torch.Generator().manual_seed(int(seed)),
                )
                model = make_model(model_name, hidden=256, dropout=0.30, heads=4).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
                history = []
                for epoch in range(1, epochs + 1):
                    model.train()
                    losses = []
                    for batch in loader:
                        optimizer.zero_grad(set_to_none=True)
                        tensors = base.move_batch(batch, device)
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
                        "deploy {} {} seed={} epoch={}/{} loss={:.6f}".format(
                            deploy_arm, model_name, seed, epoch, epochs, history[-1]["train_loss"]
                        ),
                        flush=True,
                    )
                directory.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": model_name,
                        "model_version": MODEL_VERSION,
                        "deploy_arm": deploy_arm,
                        "seed": int(seed),
                        "epochs": int(epochs),
                        "training_rows": int(len(frame)),
                        "contract_id": "role_complete_pair_matrix_v2",
                        "training_data_sha256": training_data_sha256,
                        "model_implementation_sha256": implementation_sha256,
                        "deploy_trainer_sha256": deploy_trainer_sha256,
                        "base_trainer_sha256": base_trainer_sha256,
                        "epoch_rule": EPOCH_RULE,
                        "epoch_source_fingerprint": epoch_source_fingerprint,
                    },
                    temporary,
                )
                os.replace(str(temporary), str(checkpoint))
                pd.DataFrame(history).to_csv(directory / "history.tsv", sep="\t", index=False)
                report = {
                    "status": "validated",
                    "contract_id": "role_complete_pair_matrix_v2",
                    "deploy_arm": deploy_arm,
                    "model": model_name,
                    "model_version": MODEL_VERSION,
                    "seed": int(seed),
                    "epochs": int(epochs),
                    "training_rows": int(len(frame)),
                    "training_proteins": int(frame["uniprot"].nunique()),
                    "training_ligands_connectivity": int(frame["connectivity_key"].nunique()),
                    "training_data_sha256": training_data_sha256,
                    "model_implementation_sha256": implementation_sha256,
                    "deploy_trainer_sha256": deploy_trainer_sha256,
                    "base_trainer_sha256": base_trainer_sha256,
                    "checkpoint_sha256": sha256(checkpoint),
                    "epoch_rule": EPOCH_RULE,
                    "epoch_source_regimes": ["unseen_family"],
                    "source_best_epochs": source_epochs,
                    "source_report_count": len(source_records),
                    "source_report_records": source_records,
                    "epoch_source_fingerprint": epoch_source_fingerprint,
                    "training_weight": "equal total weight per protein-label group and then per label; no ligand balancing",
                }
                atomic_json(report_path, report)
                reports.append(report)
        del caches
    expected = len(DEPLOY_ARMS) * len(DEPLOY_MODELS) * len(SEEDS)
    index = {
        "status": "validated" if len(reports) == expected else "failed",
        "contract_id": "role_complete_pair_matrix_v2",
        "model_version": MODEL_VERSION,
        "expected_models": int(expected),
        "completed_models": int(len(reports)),
        "deploy_arms": list(DEPLOY_ARMS),
        "models": list(DEPLOY_MODELS),
        "seeds": list(SEEDS),
        "epoch_rule": EPOCH_RULE,
        "epoch_source_regimes": ["unseen_family"],
        "general_full_screen_models": [
            "ligand", "protein", "c1", "c2", "d1", "d2", "d3"
        ],
        "general_full_screen_deploy_arms": [
            "general_every_pair",
            "general_protein_anchored",
        ],
        "source_linked_biochemical_screen_models": list(DEPLOY_MODELS),
        "source_linked_biochemical_screen_deploy_arms": list(DEPLOY_ARMS),
        "c3_full_screen_default": False,
        "pocket_models_external_default": True,
        "pocket_models_external_scope": "validated selected-chain pocket subset only",
    }
    atomic_json(output_root / "DEPLOY_INDEX.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    if index["status"] != "validated":
        raise SystemExit("deploy training incomplete")


if __name__ == "__main__":
    main()
