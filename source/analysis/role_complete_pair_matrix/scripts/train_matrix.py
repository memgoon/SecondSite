#!/usr/bin/env python3
"""Train the eight frozen heads under the cohort-specific split contract."""

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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_definitions import MODEL_VERSION, VALID_MODELS, make_model, model_contract  # noqa: E402


PROBABILITY_CONVERSION = "sigmoid applied after FP32 logit cast"
VALID_ARMS = ("every_pair", "protein_anchored", "protein_ligand_role_complete")
VALID_REGIMES = ("row_random", "unseen_family", "unseen_ligand")
ACTIVE_REGIMES = {
    "every_pair": ("row_random", "unseen_family", "unseen_ligand"),
    "protein_anchored": ("row_random", "unseen_family", "unseen_ligand"),
    "protein_ligand_role_complete": (
        "row_random",
        "unseen_family",
        "unseen_ligand",
    ),
}
ARM_FILES = {
    "every_pair": "EVERY_PAIR.tsv.gz",
    "protein_anchored": "PROTEIN_ANCHORED.tsv.gz",
    "protein_ligand_role_complete": "PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz",
}
FOLD_COLUMNS = {
    "row_random": "matrix_row_fold",
    "unseen_family": "matrix_family_fold",
    "unseen_ligand": "matrix_ligand_fold",
}
JOINT_MODELS = {"c1", "c2", "c3", "d1", "d2", "d3"}
MIN_CHECKPOINT_GROUPS = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
    parser.add_argument("--regimes", default=",".join(VALID_REGIMES))
    parser.add_argument("--models", default=",".join(VALID_MODELS))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="20260817,20260818,20260819")
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--full-bidirectional-batch-size", type=int, default=1)
    parser.add_argument("--full-bidirectional-eval-batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-atoms", type=int, default=120)
    parser.add_argument("--max-protein-residues", type=int, default=4096)
    return parser.parse_args()


def comma_strings(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def comma_ints(value):
    return [int(item) for item in comma_strings(value)]


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_trainer(root: Path):
    path = root / "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
    spec = importlib.util.spec_from_file_location("matrix_base_trainer", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_precision(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not len(y) or y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(score, kind="mergesort")[::-1]
    y = y[order]
    score = score[order]
    ends = np.r_[np.where(np.diff(score))[0], len(score) - 1]
    true_positive = np.cumsum(y)[ends]
    precision = true_positive / (ends + 1)
    recall = true_positive / int(y.sum())
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (ranks[y == 1].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def binary_metrics(frame):
    if not len(frame):
        return None
    y = frame["binary_label"].astype(int).to_numpy()
    p = frame["p_allosteric"].astype(float).to_numpy()
    ap_allo = average_precision(y, p)
    ap_ortho = average_precision(1 - y, 1.0 - p)
    return {
        "n": int(len(frame)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "allosteric_positive_ap": ap_allo,
        "orthosteric_positive_ap": ap_ortho,
        "symmetric_ap": None
        if ap_allo is None or ap_ortho is None
        else float((ap_allo + ap_ortho) / 2.0),
        "auroc": roc_auc(y, p),
    }


def grouped_metrics(frame, column):
    values = []
    skipped = 0
    rows_used = 0
    rows_skipped = 0
    for _, group in frame.groupby(column, sort=False):
        value = binary_metrics(group)
        if value is None or value["symmetric_ap"] is None:
            skipped += 1
            rows_skipped += int(len(group))
        else:
            values.append(value)
            rows_used += int(len(group))
    return {
        "macro_symmetric_ap": float(np.mean([x["symmetric_ap"] for x in values]))
        if values
        else None,
        "macro_allosteric_positive_ap": float(
            np.mean([x["allosteric_positive_ap"] for x in values])
        )
        if values
        else None,
        "macro_orthosteric_positive_ap": float(
            np.mean([x["orthosteric_positive_ap"] for x in values])
        )
        if values
        else None,
        "macro_auroc": float(np.mean([x["auroc"] for x in values])) if values else None,
        "n_groups_used": int(len(values)),
        "n_groups_skipped_single_class": int(skipped),
        "n_rows_input": int(len(frame)),
        "n_rows_used": int(rows_used),
        "n_rows_skipped_single_class": int(rows_skipped),
        "row_coverage": float(rows_used / len(frame)) if len(frame) else None,
    }


def metric_bundle(frame):
    return {
        "pooled": binary_metrics(frame),
        "protein_macro": grouped_metrics(frame, "uniprot"),
        "family_macro": grouped_metrics(frame, "family_component_id"),
        "ligand_macro": grouped_metrics(frame, "connectivity_key"),
    }


def checkpoint_selection(validation_metrics, regime, model_name):
    """Return a validation-only score aligned with the declared endpoint.

    Selection is model-aware so a single-input model is never selected by a
    metric on which its input is structurally constant.  Every requested
    conditional metric must have at least MIN_CHECKPOINT_GROUPS two-class
    groups.  Row-random joint models require both conditional metrics; one
    available metric is not silently substituted for their prespecified mean.
    Pooled symmetric AP is the recorded small-support fallback.
    """
    if model_name == "ligand":
        groups = ["protein_macro"]
    elif model_name == "protein":
        groups = ["ligand_macro"]
    elif model_name in JOINT_MODELS and regime == "unseen_family":
        groups = ["ligand_macro"]
    elif model_name in JOINT_MODELS and regime == "unseen_ligand":
        groups = ["protein_macro"]
    elif model_name in JOINT_MODELS and regime == "row_random":
        groups = ["ligand_macro", "protein_macro"]
    else:
        raise ValueError(
            "unknown model/regime selection: {} {}".format(model_name, regime)
        )
    requested = [
        {
            "name": group + ".macro_auroc",
            "value": validation_metrics[group]["macro_auroc"],
            "n_groups": int(validation_metrics[group]["n_groups_used"]),
        }
        for group in groups
    ]
    supported = all(
        item["value"] is not None
        and np.isfinite(item["value"])
        and item["n_groups"] >= MIN_CHECKPOINT_GROUPS
        for item in requested
    )
    support = {item["name"]: item["n_groups"] for item in requested}
    if supported:
        return (
            float(np.mean([float(item["value"]) for item in requested])),
            "mean(" + ",".join(item["name"] for item in requested) + ")",
            False,
            "none",
            support,
        )
    fallback = validation_metrics["pooled"]["symmetric_ap"]
    if fallback is None or not np.isfinite(fallback):
        raise RuntimeError("no valid checkpoint-selection metric")
    failed = [
        "{}={}".format(item["name"], item["n_groups"])
        for item in requested
        if item["value"] is None
        or not np.isfinite(item["value"])
        or item["n_groups"] < MIN_CHECKPOINT_GROUPS
    ]
    return (
        float(fallback),
        "pooled.symmetric_ap",
        True,
        "conditional_support_below_{}:{}".format(
            MIN_CHECKPOINT_GROUPS, ",".join(failed)
        ),
        support,
    )


def move_batch(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ["ligand", "ligand_mask", "protein", "protein_mask", "pocket", "pocket_mask"]
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        tensors = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(tensors)
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        for index, probability in enumerate(probabilities):
            rows.append(
                {
                    "main_row_id": batch["main_row_id"][index],
                    "uniprot": batch["uniprot"][index],
                    "family_component_id": batch["family_component_id"][index],
                    "full_inchikey": batch["full_inchikey"][index],
                    "connectivity_key": batch["connectivity_key"][index],
                    "class_label": batch["class_label"][index],
                    "binary_label": int(batch["label"][index].item()),
                    "p_allosteric": float(probability),
                }
            )
    prediction = pd.DataFrame(rows)
    return prediction, metric_bundle(prediction)


def split_frame(frame, arm, regime, test_fold, weight_function):
    column = FOLD_COLUMNS[regime]
    validation_fold = (int(test_fold) + 1) % 5
    train = frame[~frame[column].isin([test_fold, validation_fold])].copy()
    validation = frame[
        frame[column].eq(validation_fold)
        & frame["matrix_evaluation_eligible"].eq(1)
    ].copy()
    test = frame[
        frame[column].eq(test_fold)
        & frame["matrix_evaluation_eligible"].eq(1)
    ].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("empty train/validation/test partition")
    if regime == "unseen_family":
        parts = [train, validation, test]
        for field in ["uniprot", "family_component_id"]:
            sets = [set(x[field].astype(str)) for x in parts]
            if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
                raise RuntimeError("{} leakage in family-held-out split".format(field))
    if regime == "unseen_ligand":
        sets = [set(x["connectivity_key"].astype(str)) for x in [train, validation, test]]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise RuntimeError("connectivity leakage in ligand-held-out split")
    development = set(train["connectivity_key"].astype(str)) | set(
        validation["connectivity_key"].astype(str)
    )
    test["unseen_compound"] = (~test["connectivity_key"].astype(str).isin(development)).astype(int)
    train["training_weight"] = weight_function(train)
    validation["training_weight"] = 1.0
    test["training_weight"] = 1.0
    return train, validation, test, validation_fold


def paths_for(output_root, arm, regime, model_name, seed, fold):
    directory = (
        output_root
        / "fits"
        / arm
        / regime
        / model_name
        / "seed_{}".format(seed)
        / "fold_{}".format(fold)
    )
    return {
        "dir": directory,
        "checkpoint": directory / "best.pt",
        "prediction": directory / "predictions.tsv.gz",
        "history": directory / "history.tsv",
        "report": directory / "FIT_REPORT.json",
    }


def completed(paths, expected_ids, contract, identity):
    if not paths["report"].is_file() or not paths["prediction"].is_file() or not paths["checkpoint"].is_file():
        return False
    try:
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        prediction = pd.read_csv(paths["prediction"], sep="\t", usecols=["main_row_id"])
        return (
            report.get("status") == "validated"
            and all(report.get(key) == value for key, value in identity.items())
            and report.get("probability_conversion") == PROBABILITY_CONVERSION
            and report.get("model_version") == MODEL_VERSION
            and report.get("training_contract") == contract
            and set(prediction["main_row_id"].astype(str)) == set(expected_ids)
            and not prediction["main_row_id"].duplicated().any()
            and report.get("checkpoint_sha256") == sha256(paths["checkpoint"])
            and report.get("prediction_sha256") == sha256(paths["prediction"])
        )
    except Exception:
        return False


def fit_one(
    base,
    args,
    frame,
    caches,
    arm,
    regime,
    model_name,
    seed,
    fold,
    output_root,
    device,
    cpu_contract,
):
    train, validation, test, validation_fold = split_frame(
        frame, arm, regime, fold, base.class_and_protein_balanced_weights
    )
    contract = {
        "data_contract_id": "role_complete_pair_matrix_v2",
        "cohort_sha256": cpu_contract["files"][
            "data/" + ARM_FILES[arm]
        ]["sha256"],
        "model_implementation_sha256": cpu_contract["implementation_files"][
            "analysis/role_complete_pair_matrix/scripts/model_definitions.py"
        ]["sha256"],
        "base_trainer_sha256": cpu_contract["implementation_files"][
            "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
        ]["sha256"],
        "matrix_trainer_sha256": cpu_contract["implementation_files"][
            "analysis/role_complete_pair_matrix/scripts/train_matrix.py"
        ]["sha256"],
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "min_epochs": int(args.min_epochs),
        "hidden_dim": int(args.hidden_dim),
        "heads": int(args.heads),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "full_bidirectional_batch_size": int(
            args.full_bidirectional_batch_size
        ),
        "full_bidirectional_eval_batch_size": int(
            args.full_bidirectional_eval_batch_size
        ),
        "max_atoms": int(args.max_atoms),
        "max_protein_residues": int(args.max_protein_residues),
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
    paths = paths_for(output_root, arm, regime, model_name, seed, fold)
    expected_ids = set(test["main_row_id"].astype(str))
    identity = {
        "cohort_arm": arm,
        "regime": regime,
        "model": model_name,
        "seed": int(seed),
        "outer_fold": int(fold),
        "validation_fold": int(validation_fold),
    }
    if completed(paths, expected_ids, contract, identity):
        print("SKIP {} {} {} seed={} fold={}".format(arm, regime, model_name, seed, fold), flush=True)
        return json.loads(paths["report"].read_text(encoding="utf-8"))

    seed_everything(seed)
    train_batch = args.full_bidirectional_batch_size if model_name == "c3" else args.batch_size
    eval_batch = (
        args.full_bidirectional_eval_batch_size if model_name == "c3" else args.eval_batch_size
    )
    loaders = {}
    for name, subset, shuffle, batch in [
        ("train", train, True, train_batch),
        ("validation", validation, False, eval_batch),
        ("test", test, False, eval_batch),
    ]:
        loaders[name] = DataLoader(
            base.PairDataset(subset, caches, args.max_protein_residues),
            batch_size=batch,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            collate_fn=base.collate_pairs,
            generator=torch.Generator().manual_seed(int(seed)) if shuffle else None,
        )
    model = make_model(model_name, args.hidden_dim, args.dropout, args.heads).to(device)
    n_parameters = int(sum(x.numel() for x in model.parameters() if x.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_score = -math.inf
    best_epoch = -1
    stale = 0
    history = []
    paths["dir"].mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            tensors = move_batch(batch, device)
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
        _, validation_metrics = evaluate(model, loaders["validation"], device)
        (
            selection,
            selection_metric,
            selection_fallback,
            selection_fallback_reason,
            selection_support,
        ) = checkpoint_selection(validation_metrics, regime, model_name)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "selection_metric": selection_metric,
                "selection_fallback_to_pooled": bool(selection_fallback),
                "selection_fallback_reason": selection_fallback_reason,
                "selection_requested_group_counts": json.dumps(
                    selection_support, sort_keys=True
                ),
                "selection_minimum_valid_groups": MIN_CHECKPOINT_GROUPS,
                "selection_score": float(selection),
                "validation_pooled_symmetric_ap": validation_metrics["pooled"]["symmetric_ap"],
                "validation_family_macro_symmetric_ap": validation_metrics["family_macro"]["macro_symmetric_ap"],
                "validation_ligand_macro_symmetric_ap": validation_metrics["ligand_macro"]["macro_symmetric_ap"],
                "validation_ligand_macro_auroc": validation_metrics["ligand_macro"]["macro_auroc"],
                "validation_protein_macro_auroc": validation_metrics["protein_macro"]["macro_auroc"],
                "validation_ligand_macro_rows_used": validation_metrics["ligand_macro"]["n_rows_used"],
                "validation_protein_macro_rows_used": validation_metrics["protein_macro"]["n_rows_used"],
            }
        )
        print(
            "{} {} {} seed={} fold={} epoch={} loss={:.6f} selection={:.6f}".format(
                arm, regime, model_name, seed, fold, epoch, history[-1]["train_loss"], selection
            ),
            flush=True,
        )
        if selection > best_score + 1e-6:
            best_score = float(selection)
            best_epoch = int(epoch)
            stale = 0
            temporary = paths["checkpoint"].with_suffix(".pt.tmp")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": model_name,
                    "model_version": MODEL_VERSION,
                    "cohort_arm": arm,
                    "regime": regime,
                    "seed": int(seed),
                    "outer_fold": int(fold),
                    "epoch": int(epoch),
                    "selection_metric": selection_metric,
                    "selection_fallback_to_pooled": bool(selection_fallback),
                    "selection_fallback_reason": selection_fallback_reason,
                    "selection_requested_group_counts": selection_support,
                    "selection_minimum_valid_groups": MIN_CHECKPOINT_GROUPS,
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
    prediction, metrics = evaluate(model, loaders["test"], device)
    flags = test.set_index("main_row_id")["unseen_compound"].astype(int).to_dict()
    prediction["unseen_compound"] = prediction["main_row_id"].map(flags).astype(int)
    prediction["cohort_arm"] = arm
    prediction["regime"] = regime
    prediction["model"] = model_name
    prediction["seed"] = int(seed)
    prediction["outer_fold"] = int(fold)
    prediction["validation_fold"] = int(validation_fold)
    prediction.to_csv(paths["prediction"], sep="\t", index=False, compression="gzip")
    pd.DataFrame(history).to_csv(paths["history"], sep="\t", index=False)
    report = {
        "status": "validated",
        "cohort_arm": arm,
        "regime": regime,
        "model": model_name,
        "model_contract": model_contract(model_name),
        "model_version": MODEL_VERSION,
        "seed": int(seed),
        "outer_fold": int(fold),
        "validation_fold": int(validation_fold),
        "n_parameters": int(n_parameters),
        "best_epoch": int(best_epoch),
        "hit_maximum_epoch": bool(best_epoch == args.epochs),
        "best_validation_score": float(best_score),
        "checkpoint_selection_contract": contract["checkpoint_selection"],
        "checkpoint_selection_model_aware": True,
        "checkpoint_selection_structurally_constant_expected": False,
        "checkpoint_selection_minimum_valid_groups": MIN_CHECKPOINT_GROUPS,
        "checkpoint_selection_metric_at_best_epoch": str(
            pd.DataFrame(history).set_index("epoch").loc[best_epoch, "selection_metric"]
        ),
        "checkpoint_selection_fallback_at_best_epoch": bool(
            pd.DataFrame(history).set_index("epoch").loc[
                best_epoch, "selection_fallback_to_pooled"
            ]
        ),
        "checkpoint_selection_fallback_reason_at_best_epoch": str(
            pd.DataFrame(history).set_index("epoch").loc[
                best_epoch, "selection_fallback_reason"
            ]
        ),
        "checkpoint_selection_requested_group_counts_at_best_epoch": json.loads(
            pd.DataFrame(history).set_index("epoch").loc[
                best_epoch, "selection_requested_group_counts"
            ]
        ),
        "split_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "training_only_augmentation": int(
                train["matrix_evaluation_eligible"].eq(0).sum()
            ),
        },
        "test_metrics": metrics,
        "test_unseen_compound_rows": int(prediction["unseen_compound"].eq(1).sum()),
        "probability_conversion": PROBABILITY_CONVERSION,
        "training_contract": contract,
        "checkpoint_path": str(paths["checkpoint"]),
        "prediction_path": str(paths["prediction"]),
        "checkpoint_sha256": sha256(paths["checkpoint"]),
        "prediction_sha256": sha256(paths["prediction"]),
    }
    atomic_json(paths["report"], report)
    return report


def validate_selection(values, allowed, name):
    if not values or set(values) - set(allowed):
        raise ValueError("invalid {}: {}".format(name, values))


def main():
    args = parse_args()
    arms = comma_strings(args.arms)
    regimes = comma_strings(args.regimes)
    models = comma_strings(args.models)
    folds = comma_ints(args.folds)
    seeds = comma_ints(args.seeds)
    validate_selection(arms, VALID_ARMS, "arms")
    validate_selection(regimes, VALID_REGIMES, "regimes")
    validate_selection(models, VALID_MODELS, "models")
    validate_selection(folds, range(5), "folds")
    if not seeds or args.world_size < 1 or not 0 <= args.worker_rank < args.world_size:
        raise ValueError("invalid seeds or worker geometry")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.project_root.resolve()
    package = root / "analysis/role_complete_pair_matrix"
    output_root = package / "gpu_output/benchmark"
    contract = json.loads((package / "validation/CPU_CONTRACT.json").read_text(encoding="utf-8"))
    if (
        contract.get("status") != "validated"
        or contract.get("contract_id") != "role_complete_pair_matrix_v2"
    ):
        raise RuntimeError("CPU data contract is not validated")
    base = load_base_trainer(root)
    pocket_path = root / "analysis/allosteric_pair_benchmark_broad_superset/data/POCKET_INDICES.json"
    with pocket_path.open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    device = torch.device(args.device)
    reports = []
    tasks = [
        (arm, regime, model)
        for arm in arms
        for regime in regimes
        if regime in ACTIVE_REGIMES[arm]
        for model in models
    ]
    if not tasks:
        raise ValueError("the requested arm/regime selection contains no active fit")
    model_cost = {
        "ligand": 0.6,
        "protein": 1.0,
        "c1": 1.5,
        "c2": 2.0,
        "c3": 8.0,
        "d1": 1.2,
        "d2": 1.8,
        "d3": 2.5,
    }
    # Deterministic greedy scheduling avoids placing whole-sequence C3 and most
    # other heavy jobs on the same GPU.
    bins = [[] for _ in range(args.world_size)]
    loads = [0.0 for _ in range(args.world_size)]
    weighted = sorted(
        tasks,
        key=lambda task: (
            -contract["cohorts"][task[0]]["rows"] * model_cost[task[2]],
            task,
        ),
    )
    for task in weighted:
        rank = min(range(args.world_size), key=lambda value: (loads[value], value))
        bins[rank].append(task)
        loads[rank] += contract["cohorts"][task[0]]["rows"] * model_cost[task[2]]
    assigned = bins[args.worker_rank]
    assigned_by_arm = {arm: [task for task in assigned if task[0] == arm] for arm in arms}

    for arm in arms:
        arm_tasks = assigned_by_arm[arm]
        if not arm_tasks:
            continue
        frame = pd.read_csv(package / "data" / ARM_FILES[arm], sep="\t", low_memory=False)
        if len(frame) != contract["cohorts"][arm]["rows"]:
            raise RuntimeError("{} row count changed".format(arm))
        print("Preloading {} tensors for {} rows".format(arm, len(frame)), flush=True)
        caches = base.load_caches(frame, pocket_indices, args.max_atoms)
        for _, regime, model_name in arm_tasks:
            for seed in seeds:
                for fold in folds:
                    reports.append(
                        fit_one(
                            base,
                            args,
                            frame,
                            caches,
                            arm,
                            regime,
                            model_name,
                            seed,
                            fold,
                            output_root,
                            device,
                            contract,
                        )
                    )
        del caches

    expected = len(assigned) * len(seeds) * len(folds)
    index = {
        "status": "validated"
        if len(reports) == expected and all(x.get("status") == "validated" for x in reports)
        else "failed",
        "worker_rank": int(args.worker_rank),
        "world_size": int(args.world_size),
        "assigned_model_regime_arms": [list(x) for x in assigned],
        "estimated_worker_load": float(loads[args.worker_rank]),
        "estimated_all_worker_loads": [float(value) for value in loads],
        "expected_fits": int(expected),
        "completed_fits": int(len(reports)),
        "probability_conversion": PROBABILITY_CONVERSION,
        "model_version": MODEL_VERSION,
    }
    atomic_json(output_root / "workers" / "WORKER_{}_INDEX.json".format(args.worker_rank), index)
    print(json.dumps(index, indent=2, sort_keys=True))
    if index["status"] != "validated":
        raise SystemExit("worker did not complete its assigned matrix")


if __name__ == "__main__":
    main()
