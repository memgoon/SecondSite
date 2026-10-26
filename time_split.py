"""Construct publication-year cohorts, select epochs on earlier data, and refit."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from build_datasets import DATASET_NAMES
from train_models import (
    PairDataset,
    collate_pairs,
    make_model,
    predict,
    training_weights,
    file_hash,
)
from evaluate_models import binary_metrics, conditional_metrics
from screen_chembl import input_availability


def validate_temporal(frame, cutoff=2015):
    if (
        frame.main_row_id.duplicated().any()
        or frame.duplicated(["uniprot", "full_inchikey"]).any()
    ):
        raise ValueError("Duplicate benchmark identities")
    if frame.role_year.isna().any():
        raise ValueError("Undated records must be excluded")
    if not frame.role_year.le(cutoff).all():
        raise ValueError("Future label entered training cohort")
    if set(frame.binary_label) != {0, 1}:
        raise ValueError("Training cohort needs both labels")


def complete_cohorts(early):
    """Reconstruct using EARLY labels only, with exact full InChIKey identity."""
    protein = early[
        early.groupby("uniprot").binary_label.transform("nunique").eq(2)
    ].copy()
    ligand = early[
        early.groupby("full_inchikey").binary_label.transform("nunique").eq(2)
    ].copy()
    double = protein[
        protein.groupby("full_inchikey").binary_label.transform("nunique").eq(2)
    ].copy()
    history = [dict(iteration=0, rows=len(double))]
    while True:
        before = len(double)
        double = double[
            double.groupby("uniprot").binary_label.transform("nunique").eq(2)
        ].copy()
        double = double[
            double.groupby("full_inchikey").binary_label.transform("nunique").eq(2)
        ].copy()
        history.append(dict(iteration=len(history), rows=len(double)))
        if len(double) == before:
            break
    return (
        dict(
            every_pair=early.copy(),
            both_class_proteins=protein,
            both_class_ligands=ligand,
            both_class_proteins_ligands=double,
        ),
        history,
    )


def chronological_validation(frame, cutoff=2015):
    choices = []
    for year in sorted(set(frame.role_year.astype(int))):
        tr = frame[frame.role_year.le(year)]
        va = frame[frame.role_year.gt(year)]
        if (
            len(tr) < 10
            or len(va) < 6
            or set(tr.binary_label) != {0, 1}
            or set(va.binary_label) != {0, 1}
        ):
            continue
        choices.append((abs(len(va) / len(frame) - 0.2), -year, year))
    if not choices:
        raise ValueError("Cannot make chronological validation with both classes")
    boundary = min(choices)[2]
    result = frame.copy()
    result["development_partition"] = np.where(
        result.role_year.le(boundary), "train", "validation"
    )
    return result, boundary


def build(frame, cutoff=2015):
    required = {
        "main_row_id",
        "uniprot",
        "full_inchikey",
        "connectivity_key",
        "binary_label",
        "role_year",
    }
    if (
        not required <= set(frame.columns)
        or frame.main_row_id.isna().any()
        or frame.main_row_id.duplicated().any()
    ):
        raise ValueError(
            "Dated pair table needs unique row IDs and all required fields"
        )
    if (
        not frame.binary_label.isin([0, 1]).all()
        or frame.duplicated(["uniprot", "full_inchikey"]).any()
    ):
        raise ValueError("Invalid labels or duplicate exact pairs")
    early = frame[frame.role_year.le(cutoff)].copy()
    future = frame[frame.role_year.gt(cutoff)].copy()
    unknown = frame[frame.role_year.isna()].copy()
    block = set(zip(early.uniprot, early.connectivity_key))
    overlap = pd.Series(
        [(u, k) in block for u, k in zip(future.uniprot, future.connectivity_key)],
        index=future.index,
        dtype=bool,
    )
    removed, future = future[overlap].copy(), future[~overlap].copy()
    if future.binary_label.nunique() != 2:
        raise ValueError("Later curated test requires both class labels")
    cohorts, history = complete_cohorts(early)
    counts = []
    for name, pool in cohorts.items():
        validate_temporal(pool, cutoff)
        pool, year = chronological_validation(pool, cutoff)
        cohorts[name] = pool
        for partition, group in [
            ("refit", pool),
            ("train", pool[pool.development_partition == "train"]),
            ("validation", pool[pool.development_partition == "validation"]),
        ]:
            counts.append(
                dict(
                    dataset=DATASET_NAMES[name],
                    partition=partition,
                    rows=len(group),
                    allosteric=int(group.binary_label.sum()),
                    proteins=group.uniprot.nunique(),
                    ligands=group.full_inchikey.nunique(),
                    validation_boundary_year=year,
                )
            )
    return cohorts, future, unknown, removed, pd.DataFrame(counts), history


def select_epoch_score(frame, model):
    fields = (
        ["uniprot"]
        if model == "ligand"
        else ["full_inchikey"] if model == "protein" else ["uniprot", "full_inchikey"]
    )
    measures = {key: conditional_metrics(frame, key) for key in fields}
    fallback = any(
        v["n_groups_used"] < 8 or not np.isfinite(v["auroc"]) for v in measures.values()
    )
    score = (
        binary_metrics(frame)["symmetric_ap"]
        if fallback
        else float(np.mean([v["auroc"] for v in measures.values()]))
    )
    if not np.isfinite(score):
        raise ValueError("Undefined earlier-validation selection statistic")
    return float(score), dict(
        fallback=fallback, support={k: v["n_groups_used"] for k, v in measures.items()}
    )


def setup(args, frame, pockets, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = make_model(args.model).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    frame = frame.copy()
    frame["training_weight"] = training_weights(frame)
    loader = DataLoader(
        PairDataset(frame, pockets, args.tensor_root, args.model),
        batch_size=1 if args.model == "c3" else 6,
        shuffle=True,
        collate_fn=collate_pairs,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return net, optimizer, scaler, loader


def train_epoch(net, optimizer, scaler, loader, device):
    net.train()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            per = torch.nn.functional.binary_cross_entropy_with_logits(
                net(batch), batch["label"], reduction="none"
            )
            loss = (per * batch["weight"]).sum() / batch["weight"].sum().clamp_min(1e-8)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def fit(args):
    pool = pd.read_csv(args.pairs, sep="\t", dtype={"main_row_id": str})
    future = pd.read_csv(args.future, sep="\t", dtype={"main_row_id": str})
    validate_temporal(pool, args.cutoff)
    if (
        future.empty
        or not future.role_year.gt(args.cutoff).all()
        or future.main_row_id.duplicated().any()
    ):
        raise ValueError("Invalid later test records")
    if set(zip(pool.uniprot, pool.connectivity_key)) & set(
        zip(future.uniprot, future.connectivity_key)
    ):
        raise ValueError("Earlier/later exact connectivity pair overlap")
    expected, _ = chronological_validation(pool, args.cutoff)
    if not expected.development_partition.equals(pool.development_partition):
        raise ValueError("Changed earlier chronological validation assignments")
    pockets = json.loads(args.pockets.read_text())
    device = torch.device(args.device)
    train = pool[pool.development_partition == "train"]
    validation = pool[pool.development_partition == "validation"]
    args.output.mkdir(parents=True, exist_ok=False)
    net, opt, scaler, loader = setup(args, train, pockets, device)
    best, best_epoch, stale, history = -np.inf, 0, 0, []
    for epoch in range(1, 26):
        loss = train_epoch(net, opt, scaler, loader, device)
        pred = validation.copy()
        pred["score"] = predict(
            net,
            validation,
            pockets,
            args.tensor_root,
            device,
            args.model,
            batch_size=2 if args.model == "c3" else 8,
            constant_single_input=True,
        )
        score, detail = select_epoch_score(pred, args.model)
        history.append(dict(epoch=epoch, loss=loss, selection_score=score, **detail))
        if score > best + 1e-6:
            best, best_epoch, stale = score, epoch, 0
            pred.to_csv(
                args.output / "validation_predictions.tsv.gz", sep="\t", index=False
            )
        else:
            stale += 1
        if stale >= 5:
            break
    del net, opt, scaler, loader
    net, opt, scaler, loader = setup(args, pool, pockets, device)
    refit = [
        dict(epoch=e, loss=train_epoch(net, opt, scaler, loader, device))
        for e in range(1, best_epoch + 1)
    ]
    future["score"] = predict(
        net,
        future,
        pockets,
        args.tensor_root,
        device,
        args.model,
        batch_size=2 if args.model == "c3" else 8,
        constant_single_input=True,
    )
    future["dataset"], future["model"], future["seed"], future["outer_fold"] = (
        args.dataset,
        args.model,
        args.seed,
        0,
    )
    future["regime"] = "time_split_" + str(args.cutoff)
    future.to_csv(args.output / "predictions.tsv.gz", sep="\t", index=False)
    torch.save(
        dict(
            state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
            model=args.model,
            selected_epoch=best_epoch,
            cutoff=args.cutoff,
            dataset=args.dataset,
            seed=args.seed,
        ),
        args.output / "checkpoint.pt",
    )
    pd.DataFrame(refit).to_csv(args.output / "refit_history.tsv", sep="\t", index=False)
    (args.output / "fit.json").write_text(
        json.dumps(
            dict(
                model=args.model,
                dataset=args.dataset,
                seed=args.seed,
                cutoff=args.cutoff,
                input_sha256=file_hash(args.pairs),
                checkpoint_sha256=file_hash(args.output / "checkpoint.pt"),
                selected_epoch=best_epoch,
                train_rows=len(train),
                validation_rows=len(validation),
                refit_rows=len(pool),
                test_rows=len(future),
                history=history,
            ),
            indent=2,
        )
        + "\n"
    )


def infer(args):
    frame = pd.read_csv(args.pairs, sep="\t", dtype={"main_row_id": str})
    if (
        frame.main_row_id.duplicated().any()
        or not frame.first_pair_year.gt(args.cutoff).all()
    ):
        raise ValueError("External rows must be unique and later than the cutoff")
    blocked = pd.read_csv(args.blocked_pairs, sep="\t")
    forbidden = set(zip(blocked.uniprot, blocked.connectivity_key))
    keep = np.array(
        [(u, k) not in forbidden for u, k in zip(frame.uniprot, frame.connectivity_key)]
    )
    excluded = frame[~keep].copy()
    frame = frame[keep].copy().reset_index(drop=True)
    reports = [
        json.loads(p.with_name("fit.json").read_text()) for p in args.checkpoints
    ]
    if (
        len({r["seed"] for r in reports}) != 3
        or len({(r["model"], r["dataset"]) for r in reports}) != 1
    ):
        raise ValueError("Three seeds of the same time-split model are required")
    pockets = json.loads(args.pockets.read_text())
    model_name = reports[0]["model"]
    available = input_availability(frame, model_name, pockets, args.tensor_root)
    frame["score_status"] = np.where(available, "scored", "input_unavailable")
    device = torch.device(args.device)
    for path, report in zip(args.checkpoints, reports):
        if (
            report["cutoff"] != args.cutoff
            or file_hash(path) != report["checkpoint_sha256"]
        ):
            raise ValueError("Time-split checkpoint mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        net = make_model(model_name).to(device)
        net.load_state_dict(saved["state_dict"], strict=True)
        column = "seed_" + str(report["seed"])
        frame[column] = np.nan
        indexes = frame.index[available]
        for start in range(0, len(indexes), 2048):
            ix = indexes[start : start + 2048]
            frame.loc[ix, column] = predict(
                net,
                frame.loc[ix],
                pockets,
                args.tensor_root,
                device,
                model_name,
                batch_size=2 if model_name == "c3" else 8,
                constant_single_input=True,
            )
        del net
    frame["score"] = frame[["seed_" + str(r["seed"]) for r in reports]].mean(axis=1)
    frame["cohort"], frame["model"] = reports[0]["dataset"], model_name
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / "predictions.tsv.gz", sep="\t", index=False)
    excluded.to_csv(
        args.output / "excluded_earlier_or_undated_pairs.tsv.gz", sep="\t", index=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("build")
    prep.add_argument("--dated-pairs", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--cutoff", type=int, default=2015)
    run = sub.add_parser("fit")
    for name in ["pairs", "future", "pockets", "tensor-root", "output"]:
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--dataset", choices=list(DATASET_NAMES.values()), required=True)
    run.add_argument(
        "--model",
        choices=["ligand", "protein", "c1", "c2", "c3", "d1", "d2", "d3"],
        required=True,
    )
    run.add_argument("--seed", type=int, default=20260817)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--cutoff", type=int, default=2015)
    external = sub.add_parser("infer")
    for name in ["pairs", "blocked-pairs", "pockets", "tensor-root", "output"]:
        external.add_argument("--" + name, type=Path, required=True)
    external.add_argument("--checkpoints", type=Path, nargs=3, required=True)
    external.add_argument("--cutoff", type=int, default=2015)
    external.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "fit":
        fit(args)
    elif args.command == "infer":
        infer(args)
    else:
        frame = pd.read_csv(args.dated_pairs, sep="\t", dtype={"main_row_id": str})
        cohorts, future, unknown, removed, counts, history = build(frame, args.cutoff)
        args.output.mkdir(parents=True, exist_ok=False)
        for name, pool in cohorts.items():
            pool.to_csv(
                args.output / (DATASET_NAMES[name] + ".tsv.gz"), sep="\t", index=False
            )
        for name, data in [
            ("future_curated", future),
            ("undated", unknown),
            ("same_pair_excluded", removed),
            ("counts", counts),
        ]:
            data.to_csv(args.output / (name + ".tsv.gz"), sep="\t", index=False)
        (args.output / "filtering.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
