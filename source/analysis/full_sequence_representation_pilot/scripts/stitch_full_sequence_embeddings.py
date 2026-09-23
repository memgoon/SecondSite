#!/usr/bin/env python3
"""Stitch overlap-chunk ESM3 tensors into full canonical-sequence tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch


PROTEIN_DIM = 1536


def tensor_from_object(value):
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        candidates = [
            item
            for item in value.values()
            if torch.is_tensor(item) and item.ndim == 2 and int(item.shape[-1]) == PROTEIN_DIM
        ]
        if len(candidates) != 1:
            raise TypeError("expected exactly one compatible embedding tensor")
        tensor = candidates[0]
    else:
        raise TypeError("unsupported embedding object")
    tensor = tensor.detach().cpu().float()
    if tensor.ndim != 2 or int(tensor.shape[1]) != PROTEIN_DIM or not torch.isfinite(tensor).all():
        raise ValueError("invalid embedding tensor")
    return tensor.contiguous()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def find_chunk_tensor(root: Path, chunk_id: str) -> Path:
    candidates = [root / (chunk_id + suffix) for suffix in [".pt", ".pth"]]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError("expected one tensor for {} under {}".format(chunk_id, root))
    return found[0]


def linear_weights(length: int, left_overlap: int, right_overlap: int) -> torch.Tensor:
    weight = torch.ones(length, dtype=torch.float32)
    if left_overlap > 0:
        weight[:left_overlap] = torch.arange(1, left_overlap + 1, dtype=torch.float32) / float(left_overlap + 1)
    if right_overlap > 0:
        weight[-right_overlap:] = torch.minimum(
            weight[-right_overlap:],
            torch.arange(right_overlap, 0, -1, dtype=torch.float32) / float(right_overlap + 1),
        )
    return weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument(
        "--representation",
        choices=["full_canonical_uniprot", "rechunked_selected_structure_chain"],
        default="full_canonical_uniprot",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    package = root / "analysis/full_sequence_representation_pilot"
    data = package / "data"
    layouts = {
        "full_canonical_uniprot": {
            "sequence_manifest": "FULL_SEQUENCE_MANIFEST.tsv",
            "chunk_manifest": "CHUNK_MANIFEST.tsv",
            "raw_directory": "chunk_embeddings",
            "output_directory": "full_sequence_embeddings",
            "length_column": "canonical_length",
            "files_table": "FULL_SEQUENCE_EMBEDDING_FILES.tsv",
            "validation": "FULL_SEQUENCE_EMBEDDING_VALIDATION.json",
            "expected_residues": 289481,
        },
        "rechunked_selected_structure_chain": {
            "sequence_manifest": "RECHUNKED_SELECTED_CHAIN_MANIFEST.tsv",
            "chunk_manifest": "RECHUNKED_SELECTED_CHAIN_CHUNK_MANIFEST.tsv",
            "raw_directory": "rechunked_selected_chain_chunk_embeddings",
            "output_directory": "rechunked_selected_chain_embeddings",
            "length_column": "selected_chain_length",
            "files_table": "RECHUNKED_SELECTED_CHAIN_EMBEDDING_FILES.tsv",
            "validation": "RECHUNKED_SELECTED_CHAIN_EMBEDDING_VALIDATION.json",
            "expected_residues": 169680,
        },
    }
    layout = layouts[args.representation]
    raw_root = package / "gpu_cache" / layout["raw_directory"]
    output_root = package / "gpu_cache" / layout["output_directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    sequences = pd.read_csv(data / layout["sequence_manifest"], sep="\t")
    chunks = pd.read_csv(data / layout["chunk_manifest"], sep="\t")
    reports = []

    for seq in sequences.sort_values("uniprot").itertuples(index=False):
        uid = str(seq.uniprot)
        subset = chunks[chunks["uniprot"].astype(str).eq(uid)].sort_values("chunk_index")
        if len(subset) != int(seq.n_chunks):
            raise RuntimeError("chunk count mismatch for {}".format(uid))
        length = int(getattr(seq, layout["length_column"]))
        total = torch.zeros(length, PROTEIN_DIM, dtype=torch.float32)
        denominator = torch.zeros(length, dtype=torch.float32)
        rows = list(subset.itertuples(index=False))
        chunk_hashes = []
        for position, row in enumerate(rows):
            start, end = int(row.start_0based), int(row.end_exclusive)
            path = find_chunk_tensor(raw_root, str(row.chunk_id))
            tensor = tensor_from_object(torch.load(path, map_location="cpu"))
            if int(tensor.shape[0]) != end - start:
                raise ValueError("chunk tensor length mismatch for {}".format(row.chunk_id))
            previous_end = int(rows[position - 1].end_exclusive) if position else start
            next_start = int(rows[position + 1].start_0based) if position + 1 < len(rows) else end
            left_overlap = max(0, previous_end - start)
            right_overlap = max(0, end - next_start)
            weight = linear_weights(end - start, left_overlap, right_overlap)
            total[start:end] += tensor * weight.unsqueeze(1)
            denominator[start:end] += weight
            chunk_hashes.append(sha256(path))
        if bool((denominator <= 0).any()):
            raise RuntimeError("uncovered canonical residues for {}".format(uid))
        stitched = (total / denominator.unsqueeze(1)).contiguous()
        if tuple(stitched.shape) != (length, PROTEIN_DIM) or not torch.isfinite(stitched).all():
            raise RuntimeError("invalid stitched tensor for {}".format(uid))
        output = output_root / (uid + ".pt")
        temporary = output.with_suffix(".pt.tmp")
        torch.save(stitched, temporary)
        os.replace(str(temporary), str(output))
        reports.append(
            {
                "uniprot": uid,
                "canonical_length": length,
                "n_chunks": int(len(rows)),
                "output_path": str(output),
                "output_sha256": sha256(output),
                "chunk_sha256_digest": hashlib.sha256("\n".join(chunk_hashes).encode()).hexdigest(),
            }
        )
        print("stitched {} length={} chunks={}".format(uid, length, len(rows)), flush=True)

    table = pd.DataFrame(reports)
    table.to_csv(package / "gpu_output" / layout["files_table"], sep="\t", index=False)
    report = {
        "status": "validated",
        "representation": args.representation,
        "proteins": int(len(table)),
        "total_residues": int(table["canonical_length"].sum()),
        "embedding_dimension": PROTEIN_DIM,
        "coordinate_contract": "one stitched tensor row per residue of the frozen canonical UniProt sequence",
        "stitching": "linear overlap ramps",
    }
    if report["proteins"] != 426 or report["total_residues"] != layout["expected_residues"]:
        raise RuntimeError("unexpected stitched representation geometry")
    atomic_json(package / "gpu_output" / layout["validation"], report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
