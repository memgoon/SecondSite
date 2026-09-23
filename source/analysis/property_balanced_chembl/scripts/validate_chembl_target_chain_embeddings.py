#!/usr/bin/env python3
"""Validate ChEMBL C3 ESM3 tensors against frozen chain sequences and masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch


PROTEIN_DIM = 1536


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/disk1/11.HS_allostery")
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def tensor_from_object(value):
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        candidates = [
            item for item in value.values()
            if torch.is_tensor(item)
            and item.ndim == 2
            and int(item.shape[-1]) == PROTEIN_DIM
        ]
        if len(candidates) != 1:
            raise TypeError("expected exactly one compatible embedding tensor")
        tensor = candidates[0]
    else:
        raise TypeError("unsupported embedding object")
    if (
        tensor.ndim != 2
        or int(tensor.shape[0]) < 1
        or int(tensor.shape[1]) != PROTEIN_DIM
        or not torch.isfinite(tensor).all()
    ):
        raise ValueError("invalid embedding tensor")
    return tensor


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    package = args.project_root / "analysis/property_balanced_chembl"
    data = package / "data/chembl_pocket_extension"
    table = pd.read_csv(data / "TARGET_CHAIN_SEQUENCES.tsv", sep="\t")
    masks = json.loads((data / "POCKET_INDICES.json").read_text(encoding="utf-8"))
    embedding_root = package / "gpu_cache/chembl_target_chain_embeddings"
    failures = {}
    observed_paths = set()
    for row in table.itertuples(index=False):
        uid = str(row.uniprot)
        path = embedding_root / (uid + ".pt")
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            observed_paths.add(path.name)
            tensor = tensor_from_object(torch.load(path, map_location="cpu"))
            if int(tensor.shape[0]) != int(row.target_chain_length):
                raise ValueError("tensor and selected-chain lengths differ")
            indices = [int(value) for value in masks[uid]]
            if not indices or min(indices) < 0 or max(indices) >= int(tensor.shape[0]):
                raise ValueError("pocket index outside tensor")
        except Exception as error:
            failures[uid] = "{}: {}".format(type(error).__name__, error)
    stale = []
    if embedding_root.is_dir():
        expected = {str(value) + ".pt" for value in table["uniprot"]}
        stale = sorted(
            path.name for path in embedding_root.glob("*.pt")
            if path.name not in expected
        )
    status = "validated" if not failures and not stale else "incomplete"
    report = {
        "status": status,
        "coordinate_contract": (
            "Each tensor has one row per residue of TARGET_CHAIN_SEQUENCES.tsv; "
            "POCKET_INDICES.json indexes this exact tensor."
        ),
        "expected_proteins": int(len(table)),
        "validated_proteins": int(len(table) - len(failures)),
        "failed_protein_count": int(len(failures)),
        "failed_proteins": failures,
        "stale_embedding_files": stale,
        "embedding_directory": str(embedding_root),
    }
    atomic_json(
        package / "gpu_output/pocket_extension/"
        "C3_TARGET_CHAIN_EMBEDDING_VALIDATION.json",
        report,
    )
    shown = report if not args.summary_only else {
        key: report[key] for key in [
            "status", "expected_proteins", "validated_proteins",
            "failed_protein_count", "stale_embedding_files",
        ]
    }
    print(json.dumps(shown, indent=2, sort_keys=True))
    if status != "validated" and not args.allow_incomplete:
        raise SystemExit("ChEMBL target-chain embedding validation failed")


if __name__ == "__main__":
    main()
