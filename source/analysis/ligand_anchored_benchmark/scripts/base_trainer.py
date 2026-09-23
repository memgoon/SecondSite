#!/usr/bin/env python3
"""Train six graph-free heads under row-random and unseen-family five-fold CV."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROTEIN_DIM = 1536
LIGAND_DIM = 512
VALID_MODELS = {"ligand", "protein", "c1", "c2", "c3", "d1"}
VALID_REGIMES = {"row_random", "unseen_family"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--models", default="ligand,protein,c1,c2,c3,d1")
    parser.add_argument("--regimes", default="row_random,unseen_family")
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
    return parser.parse_args()


def comma_strings(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def comma_ints(value):
    return [int(item) for item in comma_strings(value)]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_from_object(value, expected_dim):
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        candidates = [
            item for item in value.values()
            if torch.is_tensor(item) and item.ndim == 2 and int(item.shape[-1]) == expected_dim
        ]
        if not candidates:
            raise TypeError("no compatible tensor in dictionary")
        tensor = candidates[0]
    else:
        raise TypeError("unsupported tensor object")
    tensor = tensor.detach().cpu().float()
    if tensor.ndim != 2 or int(tensor.shape[-1]) != expected_dim or int(tensor.shape[0]) < 1:
        raise ValueError("unexpected tensor shape {}".format(tuple(tensor.shape)))
    return tensor.contiguous()


def uniform_truncate(tensor, maximum):
    if maximum <= 0 or int(tensor.shape[0]) <= maximum:
        return tensor
    index = torch.linspace(0, int(tensor.shape[0]) - 1, steps=maximum).round().long().unique(sorted=True)
    return tensor.index_select(0, index)


def load_caches(frame, pocket_indices, max_atoms):
    ligand_cache = {}
    for path in sorted(frame["ligand_embedding_path"].astype(str).unique()):
        ligand_cache[path] = tensor_from_object(torch.load(path, map_location="cpu"), LIGAND_DIM)[:max_atoms]
    protein_cache = {}
    pocket_cache = {}
    for row in frame[["uniprot", "protein_embedding_path"]].drop_duplicates("uniprot").itertuples(index=False):
        uid = str(row.uniprot)
        protein = tensor_from_object(torch.load(str(row.protein_embedding_path), map_location="cpu"), PROTEIN_DIM)
        index = torch.tensor([int(value) for value in pocket_indices[uid]], dtype=torch.long)
        if not len(index) or int(index.min()) < 0 or int(index.max()) >= int(protein.shape[0]):
            raise ValueError("invalid pocket indices for {}".format(uid))
        protein_cache[uid] = protein
        pocket_cache[uid] = protein.index_select(0, index).contiguous()
    return ligand_cache, protein_cache, pocket_cache


class PairDataset(Dataset):
    def __init__(self, frame, caches, max_full_residues):
        self.frame = frame.reset_index(drop=True)
        self.ligand_cache, self.protein_cache, self.pocket_cache = caches
        self.max_full_residues = max_full_residues

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        uid = str(row["uniprot"])
        return {
            "ligand": self.ligand_cache[str(row["ligand_embedding_path"])],
            "protein": uniform_truncate(self.protein_cache[uid], self.max_full_residues),
            "pocket": self.pocket_cache[uid],
            "label": float(row["binary_label"]),
            "weight": float(row.get("training_weight", 1.0)),
            "main_row_id": str(row["main_row_id"]),
            "uniprot": uid,
            "family_component_id": str(row["family_component_id"]),
            "full_inchikey": str(row["full_inchikey"]),
            "connectivity_key": str(row["connectivity_key"]),
            "class_label": str(row["class_label"]),
        }


def collate_pairs(items):
    batch = len(items)
    max_atoms = max(int(item["ligand"].shape[0]) for item in items)
    max_full = max(int(item["protein"].shape[0]) for item in items)
    max_pocket = max(int(item["pocket"].shape[0]) for item in items)
    ligand = torch.zeros(batch, max_atoms, LIGAND_DIM, dtype=torch.float32)
    ligand_mask = torch.zeros(batch, max_atoms, dtype=torch.bool)
    protein = torch.zeros(batch, max_full, PROTEIN_DIM, dtype=torch.float32)
    protein_mask = torch.zeros(batch, max_full, dtype=torch.bool)
    pocket = torch.zeros(batch, max_pocket, PROTEIN_DIM, dtype=torch.float32)
    pocket_mask = torch.zeros(batch, max_pocket, dtype=torch.bool)
    for position, item in enumerate(items):
        n_atoms = int(item["ligand"].shape[0])
        n_full = int(item["protein"].shape[0])
        n_pocket = int(item["pocket"].shape[0])
        ligand[position, :n_atoms] = item["ligand"]
        ligand_mask[position, :n_atoms] = True
        protein[position, :n_full] = item["protein"]
        protein_mask[position, :n_full] = True
        pocket[position, :n_pocket] = item["pocket"]
        pocket_mask[position, :n_pocket] = True
    output = {
        "ligand": ligand,
        "ligand_mask": ligand_mask,
        "protein": protein,
        "protein_mask": protein_mask,
        "pocket": pocket,
        "pocket_mask": pocket_mask,
        "label": torch.tensor([item["label"] for item in items], dtype=torch.float32),
        "weight": torch.tensor([item["weight"] for item in items], dtype=torch.float32),
    }
    for key in [
        "main_row_id", "uniprot", "family_component_id", "full_inchikey",
        "connectivity_key", "class_label",
    ]:
        output[key] = [item[key] for item in items]
    return output


def masked_mean(value, mask):
    weight = mask.unsqueeze(-1).to(value.dtype)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def masked_max(value, mask):
    filled = value.masked_fill(~mask.unsqueeze(-1), torch.finfo(value.dtype).min)
    return filled.max(dim=1).values


class FusionHead(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, ligand, protein):
        value = torch.cat([ligand, protein, ligand * protein, torch.abs(ligand - protein)], dim=-1)
        return self.network(value).squeeze(-1)


class LigandOnly(nn.Module):
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        del heads
        self.network = nn.Sequential(
            nn.Linear(LIGAND_DIM, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        return self.network(masked_mean(batch["ligand"], batch["ligand_mask"])).squeeze(-1)


class ProteinOnly(nn.Module):
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        del heads
        self.attention = nn.Sequential(nn.Linear(PROTEIN_DIM, 128), nn.Tanh(), nn.Linear(128, 1))
        self.network = nn.Sequential(
            nn.Linear(PROTEIN_DIM, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        score = self.attention(batch["protein"]).squeeze(-1)
        score = score.masked_fill(~batch["protein_mask"], torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (batch["protein"] * weight).sum(dim=1)
        return self.network(pooled).squeeze(-1)


class C1PocketConcat(nn.Module):
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        del heads
        self.protein_attention = nn.Sequential(nn.Linear(PROTEIN_DIM, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(
            nn.Linear(LIGAND_DIM + PROTEIN_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        ligand = masked_mean(batch["ligand"], batch["ligand_mask"])
        score = self.protein_attention(batch["pocket"]).squeeze(-1)
        score = score.masked_fill(~batch["pocket_mask"], torch.finfo(score.dtype).min)
        pocket = (batch["pocket"] * torch.softmax(score, dim=1).unsqueeze(-1)).sum(dim=1)
        return self.classifier(torch.cat([ligand, pocket], dim=-1)).squeeze(-1)


class C2WholeProtein(nn.Module):
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        self.ligand_projection = nn.Sequential(nn.Linear(LIGAND_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.protein_projection = nn.Sequential(nn.Linear(PROTEIN_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.head = FusionHead(hidden, dropout)

    def forward(self, batch):
        ligand = masked_mean(self.ligand_projection(batch["ligand"]), batch["ligand_mask"])
        protein = self.protein_projection(batch["protein"])
        context, _ = self.attention(
            ligand.unsqueeze(1), protein, protein,
            key_padding_mask=~batch["protein_mask"], need_weights=False,
        )
        return self.head(ligand, context.squeeze(1))


class C3PocketOneWay(nn.Module):
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        self.ligand_projection = nn.Sequential(nn.Linear(LIGAND_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.protein_projection = nn.Sequential(nn.Linear(PROTEIN_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.head = FusionHead(hidden, dropout)

    def forward(self, batch):
        ligand = masked_mean(self.ligand_projection(batch["ligand"]), batch["ligand_mask"])
        pocket = self.protein_projection(batch["pocket"])
        context, _ = self.attention(
            ligand.unsqueeze(1), pocket, pocket,
            key_padding_mask=~batch["pocket_mask"], need_weights=False,
        )
        return self.head(ligand, context.squeeze(1))


class D1BidirectionalPocket(nn.Module):
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        self.ligand_projection = nn.Sequential(nn.Linear(LIGAND_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.protein_projection = nn.Sequential(nn.Linear(PROTEIN_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.atom_to_residue = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.residue_to_atom = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.head = FusionHead(hidden, dropout)

    def forward(self, batch):
        ligand = self.ligand_projection(batch["ligand"])
        pocket = self.protein_projection(batch["pocket"])
        atom_context, _ = self.atom_to_residue(
            ligand, pocket, pocket, key_padding_mask=~batch["pocket_mask"], need_weights=False,
        )
        residue_context, _ = self.residue_to_atom(
            pocket, ligand, ligand, key_padding_mask=~batch["ligand_mask"], need_weights=False,
        )
        return self.head(
            masked_max(atom_context, batch["ligand_mask"]),
            masked_max(residue_context, batch["pocket_mask"]),
        )


def make_model(name, hidden, dropout, heads):
    constructors = {
        "ligand": LigandOnly,
        "protein": ProteinOnly,
        "c1": C1PocketConcat,
        "c2": C2WholeProtein,
        "c3": C3PocketOneWay,
        "d1": D1BidirectionalPocket,
    }
    return constructors[name](hidden, dropout, heads)


def average_precision(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
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
    y = frame["binary_label"].astype(int).to_numpy()
    probability = frame["p_allosteric"].astype(float).to_numpy()
    allosteric_ap = average_precision(y, probability)
    orthosteric_ap = average_precision(1 - y, 1.0 - probability)
    symmetric = None if allosteric_ap is None or orthosteric_ap is None else float(
        (allosteric_ap + orthosteric_ap) / 2.0
    )
    return {
        "n": int(len(frame)),
        "allosteric_positive_ap": allosteric_ap,
        "orthosteric_positive_ap": orthosteric_ap,
        "symmetric_ap": symmetric,
        "auroc": roc_auc(y, probability),
    }


def grouped_metrics(frame, column):
    values = []
    skipped = 0
    for _, group in frame.groupby(column, sort=False):
        row = binary_metrics(group)
        if row["symmetric_ap"] is None:
            skipped += 1
        else:
            values.append(row)
    return {
        "macro_symmetric_ap": float(np.mean([row["symmetric_ap"] for row in values])) if values else None,
        "macro_allosteric_positive_ap": float(np.mean([row["allosteric_positive_ap"] for row in values])) if values else None,
        "macro_orthosteric_positive_ap": float(np.mean([row["orthosteric_positive_ap"] for row in values])) if values else None,
        "n_groups_used": int(len(values)),
        "n_groups_skipped_single_class": int(skipped),
    }


def metric_bundle(frame):
    if not len(frame) or frame["binary_label"].nunique() < 2:
        return None
    return {
        "pooled": binary_metrics(frame),
        "protein_macro": grouped_metrics(frame, "uniprot"),
        "family_component_macro": grouped_metrics(frame, "family_component_id"),
    }


def two_label_subset(frame):
    proteins = set(
        frame.groupby("uniprot")["binary_label"].nunique().loc[lambda value: value.eq(2)].index.astype(str)
    )
    return frame[frame["uniprot"].astype(str).isin(proteins)].copy()


def move_batch(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ["ligand", "ligand_mask", "protein", "protein_mask", "pocket", "pocket_mask"]
    }


def evaluate(model, loader, device):
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            tensors = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(tensors)
            probabilities = torch.sigmoid(logits).float().cpu().numpy()
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
    return prediction, metric_bundle(prediction)


def class_and_protein_balanced_weights(train):
    size = train.groupby(["uniprot", "binary_label"])["main_row_id"].transform("size").astype(float)
    weight = 1.0 / size
    class_total = weight.groupby(train["binary_label"]).transform("sum")
    weight = weight / class_total.clip(lower=1e-12)
    return weight * (len(weight) / weight.sum())


def split_frame(frame, regime, test_fold):
    fold_column = "row_fold" if regime == "row_random" else "family_fold"
    validation_fold = (test_fold + 1) % 5
    train = frame[~frame[fold_column].isin([test_fold, validation_fold])].copy()
    validation = frame[frame[fold_column].eq(validation_fold)].copy()
    test = frame[frame[fold_column].eq(test_fold)].copy()
    if regime == "unseen_family":
        for column in ["uniprot", "family_component_id"]:
            sets = [set(part[column].astype(str)) for part in [train, validation, test]]
            if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
                raise RuntimeError("{} overlap in unseen-family split".format(column))
    development_compounds = set(train["connectivity_key"].astype(str)) | set(
        validation["connectivity_key"].astype(str)
    )
    test["unseen_compound"] = (~test["connectivity_key"].astype(str).isin(development_compounds)).astype(int)
    validation["unseen_compound"] = (
        ~validation["connectivity_key"].astype(str).isin(set(train["connectivity_key"].astype(str)))
    ).astype(int)
    train["unseen_compound"] = 0
    train["training_weight"] = class_and_protein_balanced_weights(train)
    validation["training_weight"] = 1.0
    test["training_weight"] = 1.0
    return train, validation, test, validation_fold


def fit_paths(output_root, regime, model_name, seed, fold):
    directory = output_root / "fits" / regime / model_name / "seed_{}".format(seed) / "fold_{}".format(fold)
    return {
        "directory": directory,
        "report": directory / "FIT_REPORT.json",
        "prediction": directory / "predictions.tsv",
        "history": directory / "history.tsv",
        "checkpoint": directory / "best.pt",
    }


def fit_complete(paths, expected_test_ids):
    if not paths["report"].is_file() or not paths["prediction"].is_file():
        return False
    try:
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        prediction = pd.read_csv(paths["prediction"], sep="\t")
        return (
            report.get("status") == "validated"
            and set(prediction["main_row_id"].astype(str)) == set(expected_test_ids)
            and not prediction["main_row_id"].duplicated().any()
        )
    except Exception:
        return False


def train_fit(model_name, regime, seed, test_fold, args, frame, caches, output_root, device):
    train, validation, test, validation_fold = split_frame(frame, regime, test_fold)
    paths = fit_paths(output_root, regime, model_name, seed, test_fold)
    expected_test_ids = set(test["main_row_id"].astype(str))
    if fit_complete(paths, expected_test_ids):
        print("SKIP completed {} {} seed={} fold={}".format(regime, model_name, seed, test_fold), flush=True)
        return json.loads(paths["report"].read_text(encoding="utf-8"))

    seed_everything(seed)
    loaders = {}
    for name, subset, shuffle, batch_size in [
        ("train", train, True, args.batch_size),
        ("validation", validation, False, args.eval_batch_size),
        ("test", test, False, args.eval_batch_size),
    ]:
        loaders[name] = DataLoader(
            PairDataset(subset, caches, args.max_protein_residues),
            batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True,
            collate_fn=collate_pairs,
        )

    model = make_model(model_name, args.hidden_dim, args.dropout, args.heads).to(device)
    n_parameters = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_score = -math.inf
    best_epoch = -1
    stale = 0
    history = []
    paths["directory"].mkdir(parents=True, exist_ok=True)

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
        selection = validation_metrics["family_component_macro"]["macro_symmetric_ap"]
        if selection is None:
            selection = validation_metrics["pooled"]["symmetric_ap"]
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_pooled_symmetric_ap": validation_metrics["pooled"]["symmetric_ap"],
            "validation_protein_macro_symmetric_ap": validation_metrics["protein_macro"]["macro_symmetric_ap"],
            "validation_family_macro_symmetric_ap": validation_metrics["family_component_macro"]["macro_symmetric_ap"],
            "selection_score": selection,
        })
        print(
            "{} {} seed={} fold={} epoch={} loss={:.6f} selection={:.6f}".format(
                regime, model_name, seed, test_fold, epoch, history[-1]["train_loss"], selection
            ),
            flush=True,
        )
        if selection > best_score + 1e-6:
            best_score = float(selection)
            best_epoch = epoch
            stale = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, paths["checkpoint"])
        else:
            stale += 1
            if stale >= args.patience:
                break

    saved = torch.load(paths["checkpoint"], map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    prediction, test_metrics = evaluate(model, loaders["test"], device)
    flags = test.set_index("main_row_id")["unseen_compound"].astype(int).to_dict()
    prediction["unseen_compound"] = prediction["main_row_id"].map(flags).astype(int)
    prediction["regime"] = regime
    prediction["model"] = model_name
    prediction["seed"] = int(seed)
    prediction["outer_fold"] = int(test_fold)
    prediction["validation_fold"] = int(validation_fold)
    all_paired = two_label_subset(prediction)
    unseen = prediction[prediction["unseen_compound"].eq(1)].copy()
    unseen_paired = two_label_subset(unseen)
    report = {
        "status": "validated",
        "regime": regime,
        "reader_facing_evaluation": "Row-random" if regime == "row_random" else "Unseen-family",
        "model": model_name,
        "seed": int(seed),
        "outer_fold": int(test_fold),
        "validation_fold": int(validation_fold),
        "n_parameters": n_parameters,
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "split_counts": {"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
        "test": test_metrics,
        "test_two_label_proteins": metric_bundle(all_paired),
        "test_two_label_rows": int(len(all_paired)),
        "test_unseen_compound": metric_bundle(unseen) if regime == "unseen_family" else None,
        "test_unseen_compound_rows": int(len(unseen)) if regime == "unseen_family" else 0,
        "test_unseen_compound_two_label_proteins": metric_bundle(unseen_paired) if regime == "unseen_family" else None,
        "test_unseen_compound_two_label_rows": int(len(unseen_paired)) if regime == "unseen_family" else 0,
        "prediction_path": str(paths["prediction"]),
        "history_path": str(paths["history"]),
        "checkpoint_path": str(paths["checkpoint"]),
        "hit_maximum_epoch": bool(best_epoch == args.epochs),
    }
    prediction.to_csv(paths["prediction"], sep="\t", index=False)
    pd.DataFrame(history).to_csv(paths["history"], sep="\t", index=False)
    atomic_json(paths["report"], report)
    return report


def main():
    args = parse_args()
    models = comma_strings(args.models)
    regimes = comma_strings(args.regimes)
    folds = comma_ints(args.folds)
    seeds = comma_ints(args.seeds)
    if set(models) - VALID_MODELS or set(regimes) - VALID_REGIMES:
        raise ValueError("unknown model or regime")
    if set(folds) - set(range(5)) or not seeds:
        raise ValueError("folds must be 0..4 and at least one seed is required")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    package = args.project_root / "analysis/allosteric_pair_benchmark_main"
    frame = pd.read_csv(package / "gpu_cache/MODEL_READY.tsv.gz", sep="\t", low_memory=False)
    with (package / "data/POCKET_INDICES.json").open(encoding="utf-8") as handle:
        pocket_indices = json.load(handle)
    required = {
        "main_row_id", "uniprot", "full_inchikey", "connectivity_key", "binary_label",
        "family_component_id", "family_fold", "row_fold", "protein_embedding_path",
        "ligand_embedding_path",
    }
    if required - set(frame.columns):
        raise RuntimeError("model-ready table is missing required columns")
    if frame["main_row_id"].duplicated().any() or not frame.groupby("uniprot")["binary_label"].nunique().eq(2).all():
        raise RuntimeError("model-ready cohort violates unique-row or paired-protein contract")

    print("Preloading exact ligand, protein, and pocket tensors", flush=True)
    caches = load_caches(frame, pocket_indices, args.max_atoms)
    print("ligands={} proteins={} pockets={}".format(*(len(value) for value in caches)), flush=True)
    output_root = package / "gpu_output/main_benchmark"
    device = torch.device(args.device)
    reports = []
    for regime in regimes:
        for model_name in models:
            for seed in seeds:
                for fold in folds:
                    reports.append(
                        train_fit(model_name, regime, seed, fold, args, frame, caches, output_root, device)
                    )
    index = {
        "status": "validated" if all(report.get("status") == "validated" for report in reports) else "failed",
        "models": models,
        "regimes": regimes,
        "folds": folds,
        "seeds": seeds,
        "n_expected_fits": int(len(models) * len(regimes) * len(folds) * len(seeds)),
        "n_completed_fits": int(len(reports)),
        "maximum_epoch_hits": [
            {
                "regime": report["regime"], "model": report["model"], "seed": report["seed"],
                "outer_fold": report["outer_fold"],
            }
            for report in reports if report.get("hit_maximum_epoch")
        ],
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
    atomic_json(output_root / "TRAINING_INDEX.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    if index["status"] != "validated":
        raise SystemExit("main benchmark did not validate")


if __name__ == "__main__":
    main()

