#!/usr/bin/env python3
"""Frozen eight-model naming contract for the controlled pair matrix."""

from __future__ import annotations

import torch
import torch.nn as nn


PROTEIN_DIM = 1536
LIGAND_DIM = 512
VALID_MODELS = ("ligand", "protein", "c1", "c2", "c3", "d1", "d2", "d3")
MODEL_VERSION = "role_complete_matrix_v2"


def masked_mean(value, mask):
    weight = mask.unsqueeze(-1).to(value.dtype)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def masked_max(value, mask):
    filled = value.masked_fill(~mask.unsqueeze(-1), torch.finfo(value.dtype).min)
    return filled.max(dim=1).values


class AttentionPool(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dimension, 128), nn.Tanh(), nn.Linear(128, 1))

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
        return self.network(masked_mean(batch["ligand"], batch["ligand_mask"])).squeeze(-1)


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
        return self.network(self.pool(batch["protein"], batch["protein_mask"])).squeeze(-1)


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


def model_contract(name):
    representation = "single_input"
    interaction = "baseline"
    if name.startswith("c"):
        representation = "whole_selected_target_chain"
    elif name.startswith("d"):
        representation = "pocket"
    if name.endswith("1"):
        interaction = "concat"
    elif name.endswith("2"):
        interaction = "one_way_attention"
    elif name.endswith("3"):
        interaction = "bidirectional_attention"
    return {
        "name": name,
        "model_version": MODEL_VERSION,
        "representation": representation,
        "interaction": interaction,
    }
