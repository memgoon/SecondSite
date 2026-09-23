#!/usr/bin/env python3
"""Validate ESM3 tensors for only the newly aligned Arm B target chains."""

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
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def tensor_from_object(value):
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        candidates = [
            item for item in value.values()
            if torch.is_tensor(item) and item.ndim == 2 and int(item.shape[-1]) == PROTEIN_DIM
        ]
        if len(candidates) != 1:
            raise TypeError("expected exactly one compatible ESM3 tensor")
        tensor = candidates[0]
    else:
        raise TypeError("unsupported embedding object")
    if tensor.ndim != 2 or int(tensor.shape[-1]) != PROTEIN_DIM or int(tensor.shape[0]) < 1:
        raise ValueError("unexpected tensor shape {}".format(tuple(tensor.shape)))
    if not torch.isfinite(tensor).all():
        raise ValueError("embedding contains non-finite values")
    return tensor


def main():
    args = parse_args()
    package = args.project_root / "analysis/allosteric_pair_benchmark_broad_superset"
    table_path = package / "data/EXTRA_TARGET_CHAIN_SEQUENCES.tsv"
    output_dir = package / "gpu_cache/extra_target_chain_embeddings"
    report_path = package / "validation/EXTRA_TARGET_EMBEDDING_VALIDATION.json"
    if not table_path.is_file():
        raise FileNotFoundError(table_path)
    table = pd.read_csv(table_path, sep="\t", low_memory=False)
    failures = {}
    successes = 0
    for row in table.itertuples(index=False):
        uid = str(row.uniprot)
        path = output_dir / (uid + ".pt")
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            tensor = tensor_from_object(torch.load(path, map_location="cpu"))
            if int(tensor.shape[0]) != int(row.target_chain_length):
                raise ValueError("tensor length differs from the frozen target-chain sequence")
            successes += 1
        except Exception as error:
            failures[uid] = "{}: {}".format(type(error).__name__, error)
    report = {
        "status": "validated" if not failures else "incomplete",
        "expected_extra_proteins": int(len(table)),
        "validated_extra_proteins": int(successes),
        "failed_extra_proteins": failures,
        "embedding_directory": str(output_dir),
        "coordinate_contract": (
            "One ESM3 row per residue of each frozen additional target-chain FASTA; "
            "the shared POCKET_INDICES.json indexes this exact tensor."
        ),
    }
    atomic_json(report_path, report)
    shown = report if not args.summary_only else {
        "status": report["status"],
        "expected_extra_proteins": report["expected_extra_proteins"],
        "validated_extra_proteins": report["validated_extra_proteins"],
        "failed_extra_protein_count": len(failures),
    }
    print(json.dumps(shown, indent=2, sort_keys=True))
    if failures and not args.allow_incomplete:
        raise SystemExit("additional target-chain ESM3 validation failed")


if __name__ == "__main__":
    main()
