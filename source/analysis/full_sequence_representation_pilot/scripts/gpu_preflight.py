#!/usr/bin/env python3
"""Fail-closed preflight for the full-sequence representation pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch


PROTEIN_DIM = 1536
EXPECTED_PARAMETERS = {
    "protein": 591106,
    "c1": 722178,
    "c2": 1052161,
    "c3": 1315329,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_from_object(value):
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        candidates = [
            item for item in value.values()
            if torch.is_tensor(item) and item.ndim == 2 and int(item.shape[-1]) == PROTEIN_DIM
        ]
        if len(candidates) != 1:
            raise TypeError("expected one compatible protein tensor")
        tensor = candidates[0]
    else:
        raise TypeError("unsupported tensor object")
    tensor = tensor.detach().cpu().float()
    if tensor.ndim != 2 or int(tensor.shape[1]) != PROTEIN_DIM or not torch.isfinite(tensor).all():
        raise ValueError("invalid protein tensor")
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--scope", choices=["cpu", "train", "aggregate"], default="train")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.project_root.resolve()
    package = root / "analysis/full_sequence_representation_pilot"
    contract_path = package / "validation/CPU_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "validated" or contract.get("contract_id") != "full_sequence_representation_pilot_v2":
        raise RuntimeError("invalid CPU contract")
    for relative, expected in {**contract["files"], **contract["implementation_files"]}.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise RuntimeError("CPU contract file mismatch: {}".format(relative))

    cohort = pd.read_csv(package / "data/PILOT_COHORT.tsv.gz", sep="\t")
    manifest = pd.read_csv(package / "data/FULL_SEQUENCE_MANIFEST.tsv", sep="\t")
    chunks = pd.read_csv(package / "data/CHUNK_MANIFEST.tsv", sep="\t")
    observed_counts = {
        "rows": int(len(cohort)),
        "proteins": int(manifest["uniprot"].nunique()),
        "families": int(cohort["family_component_id"].nunique()),
        "full_sequence_residues": int(manifest["canonical_length"].sum()),
        "full_sequence_embedding_chunks": int(len(chunks)),
    }
    for key, value in observed_counts.items():
        if contract["counts"].get(key) != value:
            raise RuntimeError("count mismatch for {}".format(key))
    selected_manifest = pd.read_csv(package / "data/RECHUNKED_SELECTED_CHAIN_MANIFEST.tsv", sep="\t")
    selected_chunks = pd.read_csv(package / "data/RECHUNKED_SELECTED_CHAIN_CHUNK_MANIFEST.tsv", sep="\t")
    if (
        len(selected_manifest) != 426
        or int(selected_manifest["selected_chain_length"].sum()) != 169680
        or len(selected_chunks) != contract["counts"].get("rechunked_selected_chain_embedding_chunks")
    ):
        raise RuntimeError("rechunked selected-chain manifest mismatch")
    if set(cohort["matrix_family_fold"].astype(int)) != set(range(5)):
        raise RuntimeError("unexpected family folds")
    if cohort.groupby("matrix_family_fold")["binary_label"].nunique().min() != 2:
        raise RuntimeError("single-class family fold")
    family_fold_counts = cohort.groupby("family_component_id")["matrix_family_fold"].nunique()
    if int(family_fold_counts.max()) != 1:
        raise RuntimeError("family component split leakage")
    for row in chunks.itertuples(index=False):
        fasta = package / "data/full_sequence_chunks" / str(row.fasta_name)
        if not fasta.is_file():
            raise FileNotFoundError(fasta)
        sequence = "".join(fasta.read_text(encoding="ascii").splitlines()[1:])
        if len(sequence) != int(row.chunk_length) or hashlib.sha256(sequence.encode("ascii")).hexdigest() != str(row.chunk_sequence_sha256):
            raise RuntimeError("full-sequence chunk FASTA hash mismatch: {}".format(row.fasta_name))
    for row in selected_chunks.itertuples(index=False):
        fasta = package / "data/rechunked_selected_chain_chunks" / str(row.fasta_name)
        if not fasta.is_file():
            raise FileNotFoundError(fasta)
        sequence = "".join(fasta.read_text(encoding="ascii").splitlines()[1:])
        if len(sequence) != int(row.chunk_length) or hashlib.sha256(sequence.encode("ascii")).hexdigest() != str(row.chunk_sequence_sha256):
            raise RuntimeError("selected-chain chunk FASTA hash mismatch: {}".format(row.fasta_name))

    report = {
        "status": "validated",
        "scope": args.scope,
        "cohort_rows": int(len(cohort)),
        "proteins": int(len(manifest)),
        "chunks": int(len(chunks)),
        "full_sequence_residues": int(manifest["canonical_length"].sum()),
        "selected_representation_baseline_available": False,
    }
    baseline = root / "analysis/role_complete_pair_matrix/gpu_output/benchmark/aggregate/ALL_OOF_PREDICTIONS.tsv.gz"
    report["selected_representation_baseline_available"] = baseline.is_file()

    if args.scope in {"train", "aggregate"}:
        representation_layouts = {
            "full_canonical_uniprot": {
                "validation": "FULL_SEQUENCE_EMBEDDING_VALIDATION.json",
                "files": "FULL_SEQUENCE_EMBEDDING_FILES.tsv",
                "directory": "full_sequence_embeddings",
                "lengths": manifest.set_index("uniprot")["canonical_length"].astype(int).to_dict(),
                "total": 289481,
            },
            "rechunked_selected_structure_chain": {
                "validation": "RECHUNKED_SELECTED_CHAIN_EMBEDDING_VALIDATION.json",
                "files": "RECHUNKED_SELECTED_CHAIN_EMBEDDING_FILES.tsv",
                "directory": "rechunked_selected_chain_embeddings",
                "lengths": selected_manifest.set_index("uniprot")["selected_chain_length"].astype(int).to_dict(),
                "total": 169680,
            },
        }
        for representation, layout in representation_layouts.items():
            embedding_report = json.loads((package / "gpu_output" / layout["validation"]).read_text())
            embedding_files = pd.read_csv(package / "gpu_output" / layout["files"], sep="\t")
            if (
                embedding_report.get("status") != "validated"
                or embedding_report.get("representation") != representation
                or embedding_report.get("proteins") != 426
                or embedding_report.get("total_residues") != layout["total"]
                or len(embedding_files) != 426
            ):
                raise RuntimeError("embedding validation mismatch for {}".format(representation))
            expected_hashes = embedding_files.set_index("uniprot")["output_sha256"].astype(str).to_dict()
            for uid, expected_length in sorted(layout["lengths"].items()):
                path = package / "gpu_cache" / layout["directory"] / (uid + ".pt")
                if uid not in expected_hashes or sha256(path) != expected_hashes[uid]:
                    raise RuntimeError("embedding file hash mismatch: {} {}".format(representation, uid))
                tensor = tensor_from_object(torch.load(path, map_location="cpu"))
                if int(tensor.shape[0]) != expected_length:
                    raise RuntimeError("tensor length mismatch: {} {}".format(representation, uid))
        report["validated_embedding_proteins_per_representation"] = 426

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        model_path = root / "analysis/role_complete_pair_matrix/scripts/model_definitions.py"
        models = load_module("fullseq_preflight_models", model_path)
        longest_uid = manifest.sort_values("canonical_length").iloc[-1]["uniprot"]
        longest = tensor_from_object(
            torch.load(package / "gpu_cache/full_sequence_embeddings" / (str(longest_uid) + ".pt"), map_location="cpu")
        )
        # A short synthetic ligand is sufficient to check every full-sequence
        # head against the longest frozen protein without changing model state.
        batch = {
            "protein": longest.unsqueeze(0).to(device),
            "protein_mask": torch.ones(1, len(longest), dtype=torch.bool, device=device),
            "ligand": torch.zeros(1, 16, 512, device=device),
            "ligand_mask": torch.ones(1, 16, dtype=torch.bool, device=device),
        }
        parameter_counts = {}
        with torch.no_grad():
            for name in ["protein", "c1", "c2", "c3"]:
                model = models.make_model(name).to(device).eval()
                parameter_counts[name] = int(sum(value.numel() for value in model.parameters()))
                if parameter_counts[name] != EXPECTED_PARAMETERS[name]:
                    raise RuntimeError("parameter count mismatch for {}".format(name))
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logit = model(batch)
                probability = torch.sigmoid(logit.float())
                if tuple(probability.shape) != (1,) or not torch.isfinite(probability).all():
                    raise RuntimeError("forward pass failed for {}".format(name))
                del model
        report["longest_sequence_forward_pass_length"] = int(len(longest))
        report["parameter_counts"] = parameter_counts
        report["all_model_forward_passes"] = True

    if args.scope == "aggregate" and not baseline.is_file():
        raise FileNotFoundError(baseline)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
