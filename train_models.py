"""Train the eight fixed-embedding classifiers and select deployment epochs."""

from pathlib import Path
import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from build_datasets import partition, leave_one_family_out
from evaluate_models import checkpoint_selection

PROTEIN_DIM = 1536
LIGAND_DIM = 512
VALID_MODELS = ("ligand", "protein", "c1", "c2", "c3", "d1", "d2", "d3")


def masked_mean(value, mask):
    weight = mask.unsqueeze(-1).to(value.dtype)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def masked_max(value, mask):
    filled = value.masked_fill(~mask.unsqueeze(-1), torch.finfo(value.dtype).min)
    return filled.max(dim=1).values


class AttentionPool(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dimension, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, value, mask):
        score = self.attention(value).squeeze(-1)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        return (value * torch.softmax(score, dim=1).unsqueeze(-1)).sum(dim=1)


class FusionHead(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, ligand, protein):
        value = torch.cat(
            [ligand, protein, ligand * protein, torch.abs(ligand - protein)], dim=-1
        )
        return self.network(value).squeeze(-1)


class LigandOnly(nn.Module):
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        del heads
        self.network = nn.Sequential(
            nn.Linear(LIGAND_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        return self.network(masked_mean(batch["ligand"], batch["ligand_mask"])).squeeze(
            -1
        )


class ProteinOnly(nn.Module):
    def __init__(self, hidden, dropout, heads):
        super().__init__()
        del heads
        self.pool = AttentionPool(PROTEIN_DIM)
        self.network = nn.Sequential(
            nn.Linear(PROTEIN_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        return self.network(self.pool(batch["protein"], batch["protein_mask"])).squeeze(
            -1
        )


class ConcatModel(nn.Module):
    def __init__(self, protein_field, mask_field, hidden, dropout, heads):
        super().__init__()
        del heads
        self.protein_field = protein_field
        self.mask_field = mask_field
        self.pool = AttentionPool(PROTEIN_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(LIGAND_DIM + PROTEIN_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        ligand = masked_mean(batch["ligand"], batch["ligand_mask"])
        protein = self.pool(batch[self.protein_field], batch[self.mask_field])
        return self.classifier(torch.cat([ligand, protein], dim=-1)).squeeze(-1)


class OneWayAttention(nn.Module):
    def __init__(self, protein_field, mask_field, hidden, dropout, heads):
        super().__init__()
        self.protein_field = protein_field
        self.mask_field = mask_field
        self.ligand_projection = nn.Sequential(
            nn.Linear(LIGAND_DIM, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.protein_projection = nn.Sequential(
            nn.Linear(PROTEIN_DIM, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.head = FusionHead(hidden, dropout)

    def forward(self, batch):
        ligand = masked_mean(
            self.ligand_projection(batch["ligand"]), batch["ligand_mask"]
        )
        protein = self.protein_projection(batch[self.protein_field])
        context, _ = self.attention(
            ligand.unsqueeze(1),
            protein,
            protein,
            key_padding_mask=~batch[self.mask_field],
            need_weights=False,
        )
        return self.head(ligand, context.squeeze(1))


class BidirectionalAttention(nn.Module):
    def __init__(self, protein_field, mask_field, hidden, dropout, heads):
        super().__init__()
        self.protein_field = protein_field
        self.mask_field = mask_field
        self.ligand_projection = nn.Sequential(
            nn.Linear(LIGAND_DIM, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.protein_projection = nn.Sequential(
            nn.Linear(PROTEIN_DIM, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.atom_to_residue = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.residue_to_atom = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.head = FusionHead(hidden, dropout)

    def forward(self, batch):
        ligand = self.ligand_projection(batch["ligand"])
        protein = self.protein_projection(batch[self.protein_field])
        atom_context, _ = self.atom_to_residue(
            ligand,
            protein,
            protein,
            key_padding_mask=~batch[self.mask_field],
            need_weights=False,
        )
        residue_context, _ = self.residue_to_atom(
            protein,
            ligand,
            ligand,
            key_padding_mask=~batch["ligand_mask"],
            need_weights=False,
        )
        return self.head(
            masked_max(atom_context, batch["ligand_mask"]),
            masked_max(residue_context, batch[self.mask_field]),
        )


def make_model(name, hidden=256, dropout=0.30, heads=4):
    if name == "ligand":
        return LigandOnly(hidden, dropout, heads)
    if name == "protein":
        return ProteinOnly(hidden, dropout, heads)
    if name == "c1":
        return ConcatModel("protein", "protein_mask", hidden, dropout, heads)
    if name == "c2":
        return OneWayAttention("protein", "protein_mask", hidden, dropout, heads)
    if name == "c3":
        return BidirectionalAttention("protein", "protein_mask", hidden, dropout, heads)
    if name == "d1":
        return ConcatModel("pocket", "pocket_mask", hidden, dropout, heads)
    if name == "d2":
        return OneWayAttention("pocket", "pocket_mask", hidden, dropout, heads)
    if name == "d3":
        return BidirectionalAttention("pocket", "pocket_mask", hidden, dropout, heads)
    raise ValueError("unknown model: {}".format(name))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_weights(frame):
    sizes = (
        frame.groupby(["uniprot", "binary_label"])
        .main_row_id.transform("size")
        .astype(float)
    )
    weights = 1 / sizes
    weights = weights / weights.groupby(frame.binary_label).transform("sum")
    return weights * (len(weights) / weights.sum())


def read_tensor(path, dimension):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        candidates = [
            v
            for v in obj.values()
            if torch.is_tensor(v) and v.ndim == 2 and v.shape[-1] == dimension
        ]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous tensor: {path}")
        obj = candidates[0]
    if (
        not torch.is_tensor(obj)
        or obj.ndim != 2
        or obj.shape[-1] != dimension
        or not len(obj)
        or not torch.isfinite(obj).all()
    ):
        raise ValueError(f"Invalid tensor: {path}")
    return obj.float()


class PairDataset(Dataset):
    def __init__(
        self,
        frame,
        pocket_indices,
        tensor_root,
        model,
        max_atoms=120,
        max_residues=4096,
    ):
        self.frame = frame.reset_index(drop=True)
        self.pockets = pocket_indices
        self.tensor_root = Path(tensor_root)
        self.model = model
        self.max_atoms = max_atoms
        self.max_residues = max_residues
        self.cache = {}

    def __len__(self):
        return len(self.frame)

    def tensor(self, value, dim):
        path = self.tensor_root / str(value)
        key = (str(path), dim)
        if key not in self.cache:
            self.cache[key] = read_tensor(path, dim)
        return self.cache[key]

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        ligand = (
            self.tensor(row.ligand_embedding_path, LIGAND_DIM)[: self.max_atoms]
            if self.model != "protein"
            else torch.zeros(1, LIGAND_DIM)
        )
        protein = (
            self.tensor(row.protein_embedding_path, PROTEIN_DIM)
            if self.model != "ligand"
            else torch.zeros(1, PROTEIN_DIM)
        )
        pocket = torch.zeros(1, PROTEIN_DIM)
        if self.model.startswith("d"):
            indices = np.asarray(self.pockets[str(row.uniprot)], int)
            if (
                not len(indices)
                or indices.min() < 0
                or indices.max() >= len(protein)
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError(f"Invalid pocket indices for {row.uniprot}")
            pocket = protein[torch.tensor(indices)]
        if len(protein) > self.max_residues:
            protein = protein[
                torch.linspace(0, len(protein) - 1, self.max_residues)
                .round()
                .long()
                .unique(sorted=True)
            ]
        return dict(
            ligand=ligand,
            protein=protein,
            pocket=pocket,
            label=float(row.get("binary_label", 0)),
            weight=float(row.get("training_weight", 1)),
        )


def collate_pairs(items):
    batch = {}
    for field, dimension in [
        ("ligand", LIGAND_DIM),
        ("protein", PROTEIN_DIM),
        ("pocket", PROTEIN_DIM),
    ]:
        maximum = max(len(item[field]) for item in items)
        batch[field] = torch.zeros(len(items), maximum, dimension)
        batch[field + "_mask"] = torch.zeros(len(items), maximum, dtype=torch.bool)
        for i, item in enumerate(items):
            n = len(item[field])
            batch[field][i, :n] = item[field]
            batch[field + "_mask"][i, :n] = True
    batch["label"] = torch.tensor([x["label"] for x in items])
    batch["weight"] = torch.tensor([x["weight"] for x in items])
    return batch


def predict(
    model,
    frame,
    pockets,
    tensor_root,
    device,
    model_name,
    batch_size=8,
    constant_single_input=False,
):
    """FP32 sigmoid; optional unique-input evaluation preserves exact baseline ties."""
    if frame.empty:
        return np.array([], float)
    key = (
        "ligand_embedding_path" if model_name == "ligand" else "protein_embedding_path"
    )
    if constant_single_input and model_name in {"ligand", "protein"}:
        unique = frame.drop_duplicates(key)
        values = predict(
            model, unique, pockets, tensor_root, device, model_name, 1, False
        )
        return frame[key].map(dict(zip(unique[key], values))).to_numpy(float)
    data = PairDataset(frame, pockets, tensor_root, model_name)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs
    )
    values = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inputs)
            values.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return np.asarray(values, float)


def deployment_epochs(paths, dataset, model):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    identities = {(r["seed"], r["outer_fold"]) for r in reports}
    expected = {(s, f) for s in [20260817, 20260818, 20260819] for f in range(5)}
    if len(reports) != 15 or identities != expected:
        raise ValueError("Deployment requires all 15 distinct Pfam-held-out fits")
    for report, path in zip(reports, paths):
        if (
            report["dataset"] != dataset
            or report["model"] != model
            or report["regime"] != "unseen_family"
        ):
            raise ValueError("Wrong source fit for deployment")
        if (
            file_hash(Path(path).parent / "checkpoint.pt")
            != report["checkpoint_sha256"]
        ):
            raise ValueError("Changed source checkpoint")
    return int(np.median([r["best_epoch"] for r in reports]))


def train(args):
    frame = pd.read_csv(args.pairs, sep="\t")
    if frame.main_row_id.duplicated().any():
        raise ValueError("Duplicate pair row")
    pockets = json.loads(args.pockets.read_text())
    deployment = bool(args.deployment_fit_reports)
    if sum([deployment, bool(args.leave_family), bool(args.scaffold_exclusion)]) > 1:
        raise ValueError(
            "Deployment, leave-family-out and scaffold sensitivity are separate protocols"
        )
    if deployment:
        epochs = deployment_epochs(
            args.deployment_fit_reports, args.dataset, args.model
        )
        if any(
            json.loads(Path(p).read_text())["input_sha256"] != file_hash(args.pairs)
            for p in args.deployment_fit_reports
        ):
            raise ValueError(
                "Deployment and source CV fits must use the same dataset file"
            )
        train, validation, test = (
            frame.copy(),
            frame.iloc[:0].copy(),
            frame.iloc[:0].copy(),
        )
    elif args.leave_family:
        if args.dataset != "double_anchored" or args.regime != "double_unseen":
            raise ValueError(
                "Leave-one-family-out protocol is for double-anchored double held-out"
            )
        train, validation, test = leave_one_family_out(frame, args.leave_family)
        epochs = 25
    else:
        train, validation, test = partition(
            frame, args.regime, args.fold, args.scaffold_exclusion
        )
        epochs = 25
    if train.binary_label.nunique() != 2 or (not deployment and test.empty):
        raise ValueError("Empty test or one-class training partition")
    if not deployment and not args.leave_family and validation.empty:
        raise ValueError("Empty validation partition")
    train["training_weight"] = training_weights(train)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model = make_model(args.model).to(device)
    loader = DataLoader(
        PairDataset(train, pockets, args.tensor_root, args.model),
        batch_size=1 if args.model == "c3" else 6,
        shuffle=True,
        collate_fn=collate_pairs,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=False)
    history = []
    best = -np.inf
    stale = 0
    best_epoch = 0
    constant = args.dataset == "ligand_anchored" or bool(args.leave_family)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                per_row = nn.functional.binary_cross_entropy_with_logits(
                    model(batch), batch["label"], reduction="none"
                )
                weighted = per_row * batch["weight"]
                loss = (
                    weighted.mean()
                    if args.leave_family
                    else weighted.sum() / batch["weight"].sum().clamp_min(1e-8)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        if len(validation):
            scored = validation.copy()
            scored["score"] = predict(
                model,
                validation,
                pockets,
                args.tensor_root,
                device,
                args.model,
                2 if args.model == "c3" else 8,
                constant,
            )
            ligand_identity = (
                "full_inchikey"
                if args.dataset == "ligand_anchored"
                else "connectivity_key"
            )
            value, selection = checkpoint_selection(
                scored, args.model, args.regime, ligand_identity
            )
            improved = value > best + 1e-6
        else:
            value = float(epoch)
            selection = dict(metric="fixed_epoch", fallback=False)
            improved = True
        history.append(
            dict(epoch=epoch, loss=float(np.mean(losses)), selection=value, **selection)
        )
        if improved:
            best = value
            best_epoch = epoch
            stale = 0
            torch.save(
                dict(
                    model_state_dict=model.state_dict(),
                    model=args.model,
                    hidden=256,
                    dropout=0.3,
                    heads=4,
                    dataset=args.dataset,
                    training_pair_ids=(
                        frame.main_row_id.tolist()
                        if deployment
                        else train.main_row_id.tolist()
                    ),
                ),
                args.output / "checkpoint.pt",
            )
        else:
            stale += 1
        if len(validation) and stale >= 5:
            break
    checkpoint = torch.load(
        args.output / "checkpoint.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if len(test):
        test = test.copy()
        test["score"] = predict(
            model,
            test,
            pockets,
            args.tensor_root,
            device,
            args.model,
            2 if args.model == "c3" else 8,
            constant,
        )
        test["model"] = args.model
        test["seed"] = args.seed
        test["outer_fold"] = args.leave_family or args.fold
        test["dataset"] = args.dataset
        test["regime"] = args.regime
        test.to_csv(args.output / "predictions.tsv.gz", sep="\t", index=False)
    report = dict(
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        outer_fold=args.leave_family or args.fold,
        regime="deployment" if deployment else args.regime,
        best_epoch=best_epoch,
        training_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        input_sha256=file_hash(args.pairs),
        pocket_sha256=file_hash(args.pockets),
        checkpoint_sha256=file_hash(args.output / "checkpoint.pt"),
        history=history,
        deployment_epoch_sources={
            str(p): file_hash(p) for p in args.deployment_fit_reports or []
        },
    )
    (args.output / "fit.json").write_text(
        json.dumps(report, indent=2, default=lambda x: float(x)) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pockets", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=[
            "every_pair",
            "protein_anchored",
            "ligand_anchored",
            "double_anchored",
        ],
        required=True,
    )
    parser.add_argument("--model", choices=VALID_MODELS, required=True)
    parser.add_argument(
        "--regime",
        choices=["row_random", "unseen_family", "unseen_ligand", "double_unseen"],
        default="unseen_family",
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--leave-family")
    parser.add_argument("--scaffold-exclusion", action="store_true")
    parser.add_argument("--deployment-fit-reports", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
