#!/usr/bin/env python3
"""Freeze the protein-anchored full-UniProt representation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
CONTRACT_ID = "full_sequence_representation_pilot_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def normalize_sequence(value: str) -> str:
    return "".join(str(value).split()).upper()


def chunk_bounds(length: int) -> list[tuple[int, int]]:
    if length <= CHUNK_SIZE:
        return [(0, length)]
    stride = CHUNK_SIZE - CHUNK_OVERLAP
    starts = list(range(0, max(length - CHUNK_SIZE, 0) + 1, stride))
    last = length - CHUNK_SIZE
    if starts[-1] != last:
        starts.append(last)
    return [(start, min(start + CHUNK_SIZE, length)) for start in starts]


def coverage_stratum(value: float) -> str:
    if value < 0.25:
        return "lt_0.25"
    if value < 0.50:
        return "0.25_to_lt_0.50"
    if value < 0.90:
        return "0.50_to_lt_0.90"
    return "ge_0.90"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    package = root / "analysis/full_sequence_representation_pilot"
    data = package / "data"
    chunks = data / "full_sequence_chunks"
    selected_chunks = data / "rechunked_selected_chain_chunks"
    validation = package / "validation"
    chunks.mkdir(parents=True, exist_ok=True)
    selected_chunks.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)

    source_cohort = root / "analysis/role_complete_pair_matrix/data/PROTEIN_ANCHORED.tsv.gz"
    sequence_cache = root / "analysis/allosteric_pair_benchmark_main/data/UNIPROT_SEQUENCE_CACHE.json"
    selected_manifest = root / "analysis/allosteric_pair_benchmark_main/data/TARGET_CHAIN_SEQUENCES.tsv"
    alignment_audit = root / "analysis/allosteric_pair_benchmark_main/validation/POCKET_ALIGNMENT_AUDIT.tsv"
    legacy_oof = root / "analysis/role_complete_pair_matrix/gpu_output/benchmark/aggregate/ALL_OOF_PREDICTIONS.tsv.gz"
    matrix_trainer = root / "analysis/role_complete_pair_matrix/scripts/train_matrix.py"
    model_definitions = root / "analysis/role_complete_pair_matrix/scripts/model_definitions.py"
    base_trainer = root / "analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
    for path in [
        source_cohort, sequence_cache, selected_manifest, alignment_audit,
        legacy_oof, matrix_trainer, model_definitions, base_trainer,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_csv(source_cohort, sep="\t")
    if len(frame) != 4637 or frame["uniprot"].nunique() != 426:
        raise RuntimeError("unexpected protein-anchored cohort geometry")
    sequences_raw = json.loads(sequence_cache.read_text())
    selected = pd.read_csv(selected_manifest, sep="\t").set_index("uniprot")
    alignment = pd.read_csv(alignment_audit, sep="\t").set_index("uniprot")
    proteins = frame[["uniprot", "target_chain_length", "target_chain_sequence_sha256"]].drop_duplicates("uniprot")

    sequence_rows = []
    chunk_rows = []
    selected_sequence_rows = []
    selected_chunk_rows = []
    expected_chunk_names = set()
    expected_selected_chunk_names = set()
    for row in proteins.sort_values("uniprot").itertuples(index=False):
        uid = str(row.uniprot)
        value = sequences_raw.get(uid, {})
        sequence = normalize_sequence(value.get("sequence", "") if isinstance(value, dict) else value)
        if not sequence:
            raise RuntimeError("missing canonical sequence for {}".format(uid))
        if uid not in selected.index:
            raise RuntimeError("missing selected-chain manifest row for {}".format(uid))
        if uid not in alignment.index or str(alignment.loc[uid, "status"]) != "ready":
            raise RuntimeError("missing ready alignment audit row for {}".format(uid))
        observed_selected_hash = str(selected.loc[uid, "sequence_sha256"])
        if observed_selected_hash != str(row.target_chain_sequence_sha256):
            raise RuntimeError("selected-chain sequence hash mismatch for {}".format(uid))
        full_hash = sha256_text(sequence)
        selected_length = int(row.target_chain_length)
        aligned_residues = int(alignment.loc[uid, "alignment_residues"])
        length_fraction = selected_length / float(len(sequence))
        coverage = aligned_residues / float(len(sequence))
        bounds = chunk_bounds(len(sequence))
        for chunk_index, (start, end) in enumerate(bounds):
            chunk_id = "{}__{:06d}_{:06d}".format(uid, start, end)
            filename = chunk_id + ".fasta"
            expected_chunk_names.add(filename)
            (chunks / filename).write_text(
                ">{}|{}:{}-{}\n{}\n".format(chunk_id, uid, start, end, sequence[start:end]),
                encoding="ascii",
            )
            chunk_rows.append(
                {
                    "uniprot": uid,
                    "chunk_id": chunk_id,
                    "chunk_index": int(chunk_index),
                    "start_0based": int(start),
                    "end_exclusive": int(end),
                    "chunk_length": int(end - start),
                    "chunk_sequence_sha256": sha256_text(sequence[start:end]),
                    "fasta_name": filename,
                }
            )
        selected_sequence = normalize_sequence(selected.loc[uid, "target_chain_sequence"])
        if len(selected_sequence) != selected_length or sha256_text(selected_sequence) != observed_selected_hash:
            raise RuntimeError("selected-chain sequence content mismatch for {}".format(uid))
        selected_bounds = chunk_bounds(len(selected_sequence))
        for chunk_index, (start, end) in enumerate(selected_bounds):
            chunk_id = "{}__{:06d}_{:06d}".format(uid, start, end)
            filename = chunk_id + ".fasta"
            expected_selected_chunk_names.add(filename)
            (selected_chunks / filename).write_text(
                ">{}|{}:{}-{}\n{}\n".format(chunk_id, uid, start, end, selected_sequence[start:end]),
                encoding="ascii",
            )
            selected_chunk_rows.append(
                {
                    "uniprot": uid,
                    "chunk_id": chunk_id,
                    "chunk_index": int(chunk_index),
                    "start_0based": int(start),
                    "end_exclusive": int(end),
                    "chunk_length": int(end - start),
                    "chunk_sequence_sha256": sha256_text(selected_sequence[start:end]),
                    "fasta_name": filename,
                }
            )
        sequence_rows.append(
            {
                "uniprot": uid,
                "canonical_length": int(len(sequence)),
                "canonical_length_gt_4096": bool(len(sequence) > 4096),
                "canonical_sequence_sha256": full_hash,
                "selected_chain_length": selected_length,
                "selected_chain_sequence_sha256": str(row.target_chain_sequence_sha256),
                "aligned_canonical_residues": aligned_residues,
                "selected_chain_length_fraction": float(length_fraction),
                "aligned_canonical_fraction": float(coverage),
                "coverage_stratum": coverage_stratum(coverage),
                "n_chunks": int(len(bounds)),
                "full_embedding_relative_path": "gpu_cache/full_sequence_embeddings/{}.pt".format(uid),
            }
        )
        selected_sequence_rows.append(
            {
                "uniprot": uid,
                "selected_chain_length": selected_length,
                "selected_chain_sequence_sha256": observed_selected_hash,
                "n_chunks": int(len(selected_bounds)),
                "rechunked_embedding_relative_path": "gpu_cache/rechunked_selected_chain_embeddings/{}.pt".format(uid),
            }
        )

    stale = sorted(path.name for path in chunks.glob("*.fasta") if path.name not in expected_chunk_names)
    if stale:
        raise RuntimeError("stale chunk FASTAs: {}".format(stale[:10]))
    stale_selected = sorted(path.name for path in selected_chunks.glob("*.fasta") if path.name not in expected_selected_chunk_names)
    if stale_selected:
        raise RuntimeError("stale selected-chain chunk FASTAs: {}".format(stale_selected[:10]))
    sequence_table = pd.DataFrame(sequence_rows)
    chunk_table = pd.DataFrame(chunk_rows)
    selected_sequence_table = pd.DataFrame(selected_sequence_rows)
    selected_chunk_table = pd.DataFrame(selected_chunk_rows)
    sequence_table.to_csv(data / "FULL_SEQUENCE_MANIFEST.tsv", sep="\t", index=False)
    chunk_table.to_csv(data / "CHUNK_MANIFEST.tsv", sep="\t", index=False)
    selected_sequence_table.to_csv(data / "RECHUNKED_SELECTED_CHAIN_MANIFEST.tsv", sep="\t", index=False)
    selected_chunk_table.to_csv(data / "RECHUNKED_SELECTED_CHAIN_CHUNK_MANIFEST.tsv", sep="\t", index=False)
    frame = frame.merge(
        sequence_table[
            [
                "uniprot",
                "canonical_length",
                "canonical_sequence_sha256",
                "canonical_length_gt_4096",
                "aligned_canonical_fraction",
                "coverage_stratum",
            ]
        ],
        on="uniprot",
        how="left",
        validate="many_to_one",
    )
    frame.to_csv(data / "PILOT_COHORT.tsv.gz", sep="\t", index=False, compression="gzip")

    if len(sequence_table) != 426 or len(chunk_table) != 530:
        raise RuntimeError("unexpected sequence/chunk counts")
    if int(sequence_table["canonical_length"].sum()) != 289481:
        raise RuntimeError("unexpected total canonical residues")
    if len(selected_sequence_table) != 426 or len(selected_chunk_table) != 433:
        raise RuntimeError("unexpected rechunked selected-chain counts")
    if int(selected_sequence_table["selected_chain_length"].sum()) != 169680:
        raise RuntimeError("unexpected total selected-chain residues")

    implementation = {}
    for path in sorted((package / "scripts").glob("*")):
        if path.is_file() and path.suffix in {".py", ".sh"}:
            implementation[str(path.relative_to(root))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    files = {}
    for path in [
        source_cohort,
        sequence_cache,
        selected_manifest,
        alignment_audit,
        legacy_oof,
        matrix_trainer,
        model_definitions,
        base_trainer,
        data / "PILOT_COHORT.tsv.gz",
        data / "FULL_SEQUENCE_MANIFEST.tsv",
        data / "CHUNK_MANIFEST.tsv",
        data / "RECHUNKED_SELECTED_CHAIN_MANIFEST.tsv",
        data / "RECHUNKED_SELECTED_CHAIN_CHUNK_MANIFEST.tsv",
    ]:
        files[str(path.relative_to(root))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    counts = {
        "rows": int(len(frame)),
        "proteins": int(sequence_table["uniprot"].nunique()),
        "families": int(frame["family_component_id"].nunique()),
        "full_sequence_residues": int(sequence_table["canonical_length"].sum()),
        "full_sequence_embedding_chunks": int(len(chunk_table)),
        "rechunked_selected_chain_embedding_chunks": int(len(selected_chunk_table)),
        "rechunked_selected_chain_residues": int(selected_sequence_table["selected_chain_length"].sum()),
        "aligned_canonical_fraction_lt_0.25_proteins": int((sequence_table["aligned_canonical_fraction"] < 0.25).sum()),
        "aligned_canonical_fraction_lt_0.50_proteins": int((sequence_table["aligned_canonical_fraction"] < 0.50).sum()),
        "aligned_canonical_fraction_lt_0.90_proteins": int((sequence_table["aligned_canonical_fraction"] < 0.90).sum()),
        "canonical_length_gt_4096_proteins": int((sequence_table["canonical_length"] > 4096).sum()),
    }
    contract = {
        "status": "validated",
        "contract_id": CONTRACT_ID,
        "scope": "protein-anchored cohort; unseen-family; protein-only and C1-C3; two retrained representations x 5 folds x 3 seeds = 120 fits",
        "representation_contrast": "legacy selected PDB chain versus the same selected chain re-embedded with overlap chunks versus full canonical UniProt sequence, on identical OOF rows",
        "full_sequence_is_not_assembly": True,
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "stitching": "linear overlap ramps; exact one tensor row per canonical residue",
        },
        "coverage_definition": "aligned residues from the frozen local alignment divided by canonical UniProt length; not selected-chain length divided by canonical length",
        "counts": counts,
        "files": files,
        "implementation_files": implementation,
    }
    atomic_json(validation / "CPU_CONTRACT.json", contract)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
