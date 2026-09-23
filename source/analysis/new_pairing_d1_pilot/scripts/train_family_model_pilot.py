#!/usr/bin/env python3
"""Single-run C1/C2/C3/D1/reduced-graph-cross comparison on the new cohort."""

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


PROT_DIM = 1536
LIG_DIM = 512


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--models", default="c2,c3,d1,graph_cross_pair")
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
    if tensor.ndim != 2 or int(tensor.shape[-1]) != expected_dim:
        raise ValueError("unexpected tensor shape {} expected dim {}".format(tuple(tensor.shape), expected_dim))
    return tensor.contiguous()


def uniform_truncate(tensor, maximum):
    if maximum <= 0 or int(tensor.shape[0]) <= maximum:
        return tensor
    index = torch.linspace(0, int(tensor.shape[0]) - 1, steps=maximum).round().long().unique(sorted=True)
    return tensor.index_select(0, index)


def load_caches(frame, max_atoms):
    ligand_cache = {}
    for path in sorted(frame["ligand_embedding_path"].astype(str).unique()):
        ligand_cache[path] = tensor_from_object(torch.load(path, map_location="cpu"), LIG_DIM)[:max_atoms].contiguous()
    protein_cache = {}
    for path in sorted(frame["protein_embedding_path"].astype(str).unique()):
        protein_cache[path] = tensor_from_object(torch.load(path, map_location="cpu"), PROT_DIM)
    graph_cache = {}
    for row in frame[["uniprot", "structure_graph_path"]].drop_duplicates("uniprot").itertuples(index=False):
        with np.load(str(row.structure_graph_path), allow_pickle=False) as graph:
            graph_cache[str(row.uniprot)] = {
                "embedding_index": torch.from_numpy(np.asarray(graph["embedding_index"], dtype=np.int64).copy()),
                "neighbor_index": torch.from_numpy(np.asarray(graph["neighbor_index"], dtype=np.int64).copy()),
                "neighbor_distance": torch.from_numpy(np.asarray(graph["neighbor_distance"], dtype=np.float32).copy()),
            }
    return ligand_cache, protein_cache, graph_cache


class PairDataset(Dataset):
    def __init__(self, frame, caches, max_full_residues):
        self.frame = frame.reset_index(drop=True)
        self.ligand_cache, self.protein_cache, self.graph_cache = caches
        self.max_full_residues = max_full_residues

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        uid = str(row["uniprot"])
        protein = self.protein_cache[str(row["protein_embedding_path"])]
        graph = self.graph_cache[uid]
        graph_index = graph["embedding_index"]
        if int(graph_index.max()) >= int(protein.shape[0]):
            raise RuntimeError("graph index out of bounds for {}".format(uid))
        pocket = protein.index_select(0, graph_index)
        return {
            "ligand": self.ligand_cache[str(row["ligand_embedding_path"])],
            "protein": uniform_truncate(protein, self.max_full_residues),
            "pocket": pocket,
            "neighbor_index": graph["neighbor_index"],
            "neighbor_distance": graph["neighbor_distance"],
            "label": float(row["binary_label"]),
            "weight": float(row.get("training_weight", 1.0)),
            "pilot_row_id": str(row["pilot_row_id"]),
            "uniprot": uid,
            "family_component_id": str(row["family_component_id"]),
            "inchikey": str(row["full_inchikey"]),
            "class_label": str(row["class_label"]),
        }


def collate_pairs(items):
    batch = len(items)
    max_atoms = max(int(item["ligand"].shape[0]) for item in items)
    max_full = max(int(item["protein"].shape[0]) for item in items)
    max_pocket = max(int(item["pocket"].shape[0]) for item in items)
    n_neighbors = int(items[0]["neighbor_index"].shape[1])
    ligand = torch.zeros(batch, max_atoms, LIG_DIM, dtype=torch.float32)
    ligand_mask = torch.zeros(batch, max_atoms, dtype=torch.bool)
    protein = torch.zeros(batch, max_full, PROT_DIM, dtype=torch.float32)
    protein_mask = torch.zeros(batch, max_full, dtype=torch.bool)
    pocket = torch.zeros(batch, max_pocket, PROT_DIM, dtype=torch.float32)
    pocket_mask = torch.zeros(batch, max_pocket, dtype=torch.bool)
    neighbor_index = torch.zeros(batch, max_pocket, n_neighbors, dtype=torch.long)
    neighbor_distance = torch.zeros(batch, max_pocket, n_neighbors, dtype=torch.float32)
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
        neighbor_index[position, :n_pocket] = item["neighbor_index"]
        neighbor_distance[position, :n_pocket] = item["neighbor_distance"]
    return {
        "ligand": ligand,
        "ligand_mask": ligand_mask,
        "protein": protein,
        "protein_mask": protein_mask,
        "pocket": pocket,
        "pocket_mask": pocket_mask,
        "neighbor_index": neighbor_index,
        "neighbor_distance": neighbor_distance,
        "label": torch.tensor([item["label"] for item in items], dtype=torch.float32),
        "weight": torch.tensor([item["weight"] for item in items], dtype=torch.float32),
        "pilot_row_id": [item["pilot_row_id"] for item in items],
        "uniprot": [item["uniprot"] for item in items],
        "family_component_id": [item["family_component_id"] for item in items],
        "inchikey": [item["inchikey"] for item in items],
        "class_label": [item["class_label"] for item in items],
    }


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


class C1PocketConcat(nn.Module):
    """Legacy-aligned C1: independently pool ligand and pocket, then concatenate."""
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        del heads
        self.protein_attention = nn.Sequential(
            nn.Linear(PROT_DIM, 128), nn.Tanh(), nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(LIG_DIM + PROT_DIM, hidden),
            nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        ligand = masked_mean(batch["ligand"], batch["ligand_mask"])
        score = self.protein_attention(batch["pocket"]).squeeze(-1)
        score = score.masked_fill(~batch["pocket_mask"], torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pocket = (batch["pocket"] * weight).sum(dim=1)
        return self.classifier(torch.cat([ligand, pocket], dim=-1)).squeeze(-1)


class C2WholeProtein(nn.Module):
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        self.ligand_projection = nn.Sequential(nn.Linear(LIG_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.protein_projection = nn.Sequential(nn.Linear(PROT_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
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
        self.ligand_projection = nn.Sequential(nn.Linear(LIG_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.protein_projection = nn.Sequential(nn.Linear(PROT_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
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
        self.ligand_projection = nn.Sequential(nn.Linear(LIG_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.protein_projection = nn.Sequential(nn.Linear(PROT_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
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


class GeometryNeighborAttention(nn.Module):
    def __init__(self, hidden, heads, dropout, n_rbf=16, dmax=20.0):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden dimension must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.n_rbf = n_rbf
        self.dmax = dmax
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.edge = nn.Linear(n_rbf, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node, node_mask, neighbor_index, neighbor_distance):
        batch_size, n_nodes, _ = node.shape
        batch_index = torch.arange(batch_size, device=node.device)[:, None, None].expand_as(neighbor_index)
        neighbor = node[batch_index, neighbor_index]
        centers = torch.linspace(0.0, self.dmax, self.n_rbf, device=node.device, dtype=node.dtype)
        sigma = self.dmax / self.n_rbf
        rbf = torch.exp(-((neighbor_distance.unsqueeze(-1) - centers) / sigma) ** 2)
        message = neighbor + self.edge(rbf)
        query = self.query(node).view(batch_size, n_nodes, self.heads, self.head_dim)
        key = self.key(message).view(batch_size, n_nodes, neighbor_index.shape[-1], self.heads, self.head_dim)
        value = self.value(message).view(batch_size, n_nodes, neighbor_index.shape[-1], self.heads, self.head_dim)
        logits = (query.unsqueeze(2) * key).sum(dim=-1) / math.sqrt(self.head_dim)
        neighbor_valid = node_mask[batch_index, neighbor_index]
        logits = logits.masked_fill(~neighbor_valid.unsqueeze(-1), torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=2)
        attention = self.dropout(attention)
        update = (attention.unsqueeze(-1) * value).sum(dim=2).reshape(batch_size, n_nodes, self.hidden)
        update = self.output(update)
        return update * node_mask.unsqueeze(-1).to(update.dtype)


class GraphCrossBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.neighbor = GeometryNeighborAttention(hidden, heads, dropout)
        self.residue_to_ligand = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ligand_to_residue = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.node_norm = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(hidden), nn.LayerNorm(hidden)])
        self.ligand_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 4, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node, node_mask, ligand, ligand_mask, neighbor_index, neighbor_distance):
        node = self.node_norm[0](node + self.dropout(self.neighbor(node, node_mask, neighbor_index, neighbor_distance)))
        cross_node, _ = self.residue_to_ligand(
            node, ligand, ligand, key_padding_mask=~ligand_mask, need_weights=False,
        )
        node = self.node_norm[1](node + self.dropout(cross_node))
        cross_ligand, _ = self.ligand_to_residue(
            ligand, node, node, key_padding_mask=~node_mask, need_weights=False,
        )
        ligand = self.ligand_norm(ligand + self.dropout(cross_ligand))
        node = self.node_norm[2](node + self.dropout(self.feedforward(node)))
        node = node * node_mask.unsqueeze(-1).to(node.dtype)
        ligand = ligand * ligand_mask.unsqueeze(-1).to(ligand.dtype)
        return node, ligand


class ReducedGraphCrossPair(nn.Module):
    """Pair-level adaptation of LABind's geometry-neighbor/cross-attention core."""
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        self.ligand_projection = nn.Sequential(nn.Linear(LIG_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.protein_projection = nn.Sequential(nn.Linear(PROT_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.ModuleList([GraphCrossBlock(hidden, heads, dropout) for _ in range(2)])
        self.head = FusionHead(hidden, dropout)

    def forward(self, batch):
        ligand = self.ligand_projection(batch["ligand"])
        node = self.protein_projection(batch["pocket"])
        for block in self.blocks:
            node, ligand = block(
                node, batch["pocket_mask"], ligand, batch["ligand_mask"],
                batch["neighbor_index"], batch["neighbor_distance"],
            )
        return self.head(masked_mean(ligand, batch["ligand_mask"]), masked_mean(node, batch["pocket_mask"]))


def make_model(name, hidden, dropout, heads):
    if name == "c1":
        return C1PocketConcat(hidden, dropout, heads)
    if name == "c2":
        return C2WholeProtein(hidden, dropout, heads)
    if name == "c3":
        return C3PocketOneWay(hidden, dropout, heads)
    if name == "d1":
        return D1BidirectionalPocket(hidden, dropout, heads)
    if name == "graph_cross_pair":
        return ReducedGraphCrossPair(hidden, dropout, heads)
    raise ValueError(name)


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
    return float((ranks[y == 1].sum() - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def binary_metrics(frame):
    y = frame["binary_label"].astype(int).to_numpy()
    probability = frame["p_allosteric"].astype(float).to_numpy()
    allosteric_ap = average_precision(y, probability)
    orthosteric_ap = average_precision(1 - y, 1.0 - probability)
    symmetric = None if allosteric_ap is None or orthosteric_ap is None else float((allosteric_ap + orthosteric_ap) / 2.0)
    return {
        "n": int(len(frame)),
        "allosteric_positive_ap": allosteric_ap,
        "orthosteric_positive_ap": orthosteric_ap,
        "symmetric_ap": symmetric,
        "auroc": roc_auc(y, probability),
    }


def grouped_metrics(frame, column):
    metrics = []
    skipped = 0
    for _, group in frame.groupby(column, sort=False):
        row = binary_metrics(group)
        if row["symmetric_ap"] is None:
            skipped += 1
        else:
            metrics.append(row)
    return {
        "macro_symmetric_ap": float(np.mean([row["symmetric_ap"] for row in metrics])) if metrics else None,
        "macro_allosteric_positive_ap": float(np.mean([row["allosteric_positive_ap"] for row in metrics])) if metrics else None,
        "macro_orthosteric_positive_ap": float(np.mean([row["orthosteric_positive_ap"] for row in metrics])) if metrics else None,
        "n_groups_used": int(len(metrics)),
        "n_groups_skipped_single_class": int(skipped),
    }


def move_batch(batch, device):
    names = [
        "ligand", "ligand_mask", "protein", "protein_mask", "pocket", "pocket_mask",
        "neighbor_index", "neighbor_distance",
    ]
    return {name: batch[name].to(device, non_blocking=True) for name in names}


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
                    "pilot_row_id": batch["pilot_row_id"][index],
                    "uniprot": batch["uniprot"][index],
                    "family_component_id": batch["family_component_id"][index],
                    "full_inchikey": batch["inchikey"][index],
                    "class_label": batch["class_label"][index],
                    "binary_label": int(batch["label"][index].item()),
                    "p_allosteric": float(probability),
                })
    prediction = pd.DataFrame(records)
    return prediction, {
        "pooled": binary_metrics(prediction),
        "protein_macro": grouped_metrics(prediction, "uniprot"),
        "family_component_macro": grouped_metrics(prediction, "family_component_id"),
    }


def train_one(name, args, frame, caches, output_dir, device):
    seed_everything(args.seed)
    train = frame[frame["split"].eq("train")].copy()
    validation = frame[frame["split"].eq("val")].copy()
    test = frame[frame["split"].eq("test")].copy()
    group_size = train.groupby(["uniprot", "binary_label"])["pilot_row_id"].transform("size").astype(float)
    train["training_weight"] = 1.0 / group_size
    train["training_weight"] *= len(train) / train["training_weight"].sum()
    validation["training_weight"] = 1.0
    test["training_weight"] = 1.0

    loaders = {}
    for split, subset, shuffle, batch_size in [
        ("train", train, True, args.batch_size),
        ("val", validation, False, args.eval_batch_size),
        ("test", test, False, args.eval_batch_size),
    ]:
        loaders[split] = DataLoader(
            PairDataset(subset, caches, args.max_protein_residues), batch_size=batch_size,
            shuffle=shuffle, num_workers=0, pin_memory=True, collate_fn=collate_pairs,
        )

    model = make_model(name, args.hidden_dim, args.dropout, args.heads).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_score = -math.inf
    best_epoch = -1
    stale = 0
    checkpoint = output_dir / "checkpoints/{}_best.pt".format(name)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            tensors = move_batch(batch, device)
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

        _, validation_metrics = evaluate(model, loaders["val"], device)
        selection = validation_metrics["family_component_macro"]["macro_symmetric_ap"]
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_pooled_symmetric_ap": validation_metrics["pooled"]["symmetric_ap"],
            "val_protein_macro_symmetric_ap": validation_metrics["protein_macro"]["macro_symmetric_ap"],
            "val_family_component_macro_symmetric_ap": selection,
        })
        print("{} epoch={} loss={:.6f} val_pool={:.6f} val_family_macro={:.6f}".format(
            name, epoch, history[-1]["train_loss"], history[-1]["val_pooled_symmetric_ap"], selection
        ), flush=True)
        if selection is not None and selection > best_score + 1e-6:
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
    prediction, test_metrics = evaluate(model, loaders["test"], device)
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
        "best_validation_family_component_macro_symmetric_ap": best_score,
        "test": test_metrics,
        "prediction_path": str(prediction_path),
        "checkpoint_path": str(checkpoint),
    }
    atomic_json(output_dir / "metrics/{}_metrics.json".format(name), report)
    return report, prediction


def main():
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    package = args.project_root / "analysis/new_pairing_d1_pilot"
    frame = pd.read_csv(package / "gpu_cache/MODEL_READY_PAIRING.tsv.gz", sep="\t", low_memory=False)
    required = {
        "pilot_row_id", "uniprot", "family_component_id", "full_inchikey", "binary_label",
        "split", "ligand_embedding_path", "protein_embedding_path", "structure_graph_path",
    }
    if required - set(frame.columns):
        raise RuntimeError("model-ready dataset missing columns: {}".format(sorted(required - set(frame.columns))))
    split_names = ["train", "val", "test"]
    if set(frame["split"].astype(str)) != set(split_names):
        raise RuntimeError("split universe is incomplete")
    split_components = {name: set(frame.loc[frame["split"].eq(name), "family_component_id"].astype(str)) for name in split_names}
    if any(split_components[left] & split_components[right] for left, right in [("train", "val"), ("train", "test"), ("val", "test")]):
        raise RuntimeError("family-component split leakage")

    print("Preloading exact embedding and graph caches", flush=True)
    caches = load_caches(frame, args.max_atoms)
    print("ligands={} proteins={} graphs={}".format(*(len(cache) for cache in caches)), flush=True)
    output_dir = package / "gpu_output/family_single_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    reports = {}
    predictions = {}
    for name in model_names:
        print("Starting model {}".format(name), flush=True)
        reports[name], predictions[name] = train_one(name, args, frame, caches, output_dir, device)

    test_universe = None
    for name, prediction in predictions.items():
        universe = tuple(prediction["pilot_row_id"].astype(str))
        if test_universe is None:
            test_universe = universe
        elif universe != test_universe:
            raise RuntimeError("test prediction universe differs across models")

    deltas = {}
    endpoints = [
        ("pooled", "symmetric_ap"),
        ("protein_macro", "macro_symmetric_ap"),
        ("family_component_macro", "macro_symmetric_ap"),
    ]
    comparisons = [
        ("c3", "c2"), ("d1", "c3"), ("d1", "c2"),
        ("graph_cross_pair", "c3"), ("graph_cross_pair", "d1"),
    ]
    for test_model, reference_model in comparisons:
        if test_model not in reports or reference_model not in reports:
            continue
        for section, key in endpoints:
            name = "{}_minus_{}_{}_{}".format(test_model, reference_model, section, key)
            deltas[name] = float(reports[test_model]["test"][section][key] - reports[reference_model]["test"][section][key])

    final = {
        "status": "validated",
        "scope": "new pairing cohort; Pfam/C9-component-disjoint structure-ready subset; one seed",
        "interpretation": "screening estimate only; no confidence interval or significance claim",
        "graph_model_limit": "graph_cross_pair is a reduced LABind-style pair adaptation, not official LABind",
        "dataset_rows": int(len(frame)),
        "dataset_proteins": int(frame["uniprot"].nunique()),
        "family_components": int(frame["family_component_id"].nunique()),
        "split_counts": frame.groupby("split").size().astype(int).to_dict(),
        "protein_counts": frame.groupby("split")["uniprot"].nunique().astype(int).to_dict(),
        "family_component_counts": frame.groupby("split")["family_component_id"].nunique().astype(int).to_dict(),
        "seed": args.seed,
        "models": reports,
        "deltas": deltas,
        "prediction_universe_exact": True,
    }
    atomic_json(output_dir / "PILOT_REPORT.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
