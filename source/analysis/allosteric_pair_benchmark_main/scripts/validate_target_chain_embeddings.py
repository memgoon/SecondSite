#!/usr/bin/env python3
"""Validate ESM3 embeddings generated from the frozen target-chain FASTAs."""

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
            raise TypeError("expected exactly one compatible 2D tensor in embedding dictionary")
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
    package = args.project_root / "analysis/allosteric_pair_benchmark_main"
    target_table = package / "data/TARGET_CHAIN_SEQUENCES.tsv"
    output_directory = package / "gpu_cache/target_chain_embeddings_v2"
    report_path = package / "validation/TARGET_CHAIN_EMBEDDING_VALIDATION.json"
    if not target_table.is_file():
        raise FileNotFoundError(target_table)
    table = pd.read_csv(target_table, sep="\t", low_memory=False)
    failures = {}
    successes = 0
    for row in table.itertuples(index=False):
        uid = str(row.uniprot)
        expected_length = int(row.target_chain_length)
        path = output_directory / (uid + ".pt")
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            tensor = tensor_from_object(torch.load(path, map_location="cpu"))
            if int(tensor.shape[0]) != expected_length:
                raise ValueError(
                    "sequence length {} does not equal frozen target-chain length {}".format(
                        int(tensor.shape[0]), expected_length
                    )
                )
            successes += 1
        except Exception as error:
            failures[uid] = "{}: {}".format(type(error).__name__, error)
    report = {
        "status": "validated" if not failures else "incomplete",
        "coordinate_contract": (
            "Each validated ESM3 tensor has one row per residue of the frozen target-chain FASTA; "
            "POCKET_INDICES.json is indexed into this exact tensor."
        ),
        "expected_proteins": int(len(table)),
        "validated_proteins": int(successes),
        "failed_proteins": failures,
        "embedding_directory": str(output_directory),
    }
    atomic_json(report_path, report)
    if args.summary_only:
        print(json.dumps({
            "status": report["status"],
            "expected_proteins": report["expected_proteins"],
            "validated_proteins": report["validated_proteins"],
            "failed_protein_count": len(failures),
            "note": "An incomplete preflight is expected before first-time ESM3 generation.",
        }, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if failures and not args.allow_incomplete:
        raise SystemExit("target-chain ESM3 embedding validation failed")


if __name__ == "__main__":
    main()
