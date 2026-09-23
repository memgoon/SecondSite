#!/usr/bin/env python3
"""Append only the missing C3 predictions to the frozen ChEMBL full universe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CONTRACT_ID = "chembl_c3_appendon_v1"
PACKAGE_REL = Path("analysis/chembl_c3_appendon")
UPSTREAM_REL = Path("analysis/role_complete_pair_matrix")
NUM_SHARDS = 4
ARMS = {
    "every_pair": "general_every_pair",
    "general": "general_protein_anchored",
}
SCORE_COLUMNS = [
    "p_every_pair_c3_mean",
    "p_every_pair_c3_sd",
    "p_general_c3_mean",
    "p_general_c3_sd",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk1/11.HS_allostery"))
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pilot-rows", type=int, default=2500)
    parser.add_argument("--chunk-rows", type=int, default=10000)
    parser.add_argument("--max-batch-rows", type=int, default=8)
    parser.add_argument("--max-batch-residues", type=int, default=4096)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_id_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_tsv_gzip(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        sep="\t",
        index=False,
        na_rep="NA",
        float_format="%.9g",
        compression={"method": "gzip", "mtime": 0},
    )
    os.replace(str(temporary), str(path))


def load_upstream(root: Path):
    path = root / UPSTREAM_REL / "scripts/infer_chembl.py"
    scripts = str(path.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("c3_appendon_upstream_infer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import upstream implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_full_paths(upstream_package: Path, shard: int):
    directory = upstream_package / "gpu_output/chembl/full"
    return (
        directory / f"full_predictions_shard{shard:02d}of04.tsv.gz",
        directory / f"FULL_SHARD{shard:02d}.json",
    )


def make_run_contract(args, root: Path, upstream, upstream_package: Path):
    package = root / PACKAGE_REL
    old_output, old_report = old_full_paths(upstream_package, args.shard)
    checkpoint_records = []
    for arm, deploy_arm in ARMS.items():
        records = upstream.deploy_checkpoint_records(upstream_package, deploy_arm, ("c3",))
        for record in records:
            item = dict(record)
            item["output_arm"] = arm
            checkpoint_records.append(item)
    value = {
        "contract_id": CONTRACT_ID,
        "mode": args.mode,
        "shard": int(args.shard),
        "num_shards": int(args.num_shards),
        "chunk_rows": int(args.chunk_rows),
        "max_batch_rows": int(args.max_batch_rows),
        "max_batch_residues": int(args.max_batch_residues),
        "pilot_rows": int(args.pilot_rows) if args.mode == "pilot" else None,
        "probability_conversion": upstream.PROBABILITY_CONVERSION,
        "append_only_score_columns": SCORE_COLUMNS,
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "upstream_inference_sha256": sha256(upstream_package / "scripts/infer_chembl.py"),
        "upstream_model_sha256": sha256(upstream_package / "scripts/model_definitions.py"),
        "CPU_contract_sha256": sha256(package / "validation/CPU_CONTRACT.json"),
        "CPU_inputs_manifest_sha256": sha256(package / "manifests/CPU_INPUTS.sha256"),
        "accepted_full_output_sha256": sha256(old_output),
        "accepted_full_report_sha256": sha256(old_report),
        "C3_checkpoint_records": checkpoint_records,
    }
    return value, canonical_fingerprint(value)


def load_and_verify_membership(root: Path, upstream, upstream_package: Path, shard: int):
    paths = upstream.cache_paths(root, shard, NUM_SHARDS)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source = pd.read_pickle(paths["frame"])
    sets = upstream.benchmark_sets(root, upstream_package)
    uid = source["UniProt_ID"].fillna("").astype(str).str.split("-").str[0]
    connectivity = source["InChIKey14"].fillna("").astype(str).str.upper()
    excluded = (uid + "|" + connectivity).isin(sets["broad_pairs"])
    frame = source.loc[~excluded].copy().reset_index(drop=True)
    uid = frame["UniProt_ID"].fillna("").astype(str).str.split("-").str[0]
    connectivity = frame["InChIKey14"].fillna("").astype(str).str.upper()
    expected = pd.DataFrame(
        {
            "OOD_Row_ID": frame["OOD_Row_ID"].astype(str),
            "stable_pair_key": frame["target_chembl_id"].fillna("").astype(str)
            + "|"
            + frame["ligand_chembl_id"].fillna("").astype(str),
            "uniprot": uid,
            "connectivity_key": connectivity,
        }
    )
    old_output, old_report_path = old_full_paths(upstream_package, shard)
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    if old_report.get("status") != "validated" or old_report.get("output_sha256") != sha256(old_output):
        raise RuntimeError(f"accepted upstream full shard {shard} fails its hash")
    accepted = pd.read_csv(
        old_output,
        sep="\t",
        usecols=[
            "OOD_Row_ID",
            "stable_pair_key",
            "uniprot",
            "connectivity_key",
            "selected_chain_available",
        ],
        dtype={
            "OOD_Row_ID": str,
            "stable_pair_key": str,
            "uniprot": str,
            "connectivity_key": str,
        },
    )
    compare_columns = ["OOD_Row_ID", "stable_pair_key", "uniprot", "connectivity_key"]
    if len(expected) != len(accepted):
        raise RuntimeError(f"source/accepted row mismatch for shard {shard}")
    for column in compare_columns:
        left = expected[column].fillna("").astype(str).to_numpy()
        right = accepted[column].fillna("").astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise RuntimeError(f"source/accepted {column} mismatch for shard {shard}")
    if accepted["OOD_Row_ID"].duplicated().any():
        raise RuntimeError(f"duplicate accepted OOD_Row_ID in shard {shard}")
    expected["selected_chain_available"] = (
        pd.to_numeric(accepted["selected_chain_available"], errors="raise").astype(int)
    )
    return frame, expected, paths, uid


def deterministic_batches(positions, lengths, max_rows, max_residues):
    positions = np.asarray(positions, dtype=int)
    if not len(positions):
        return []
    ordered = positions[np.lexsort((positions, lengths[positions]))]
    batches = []
    current = []
    current_max = 0
    for position in ordered:
        length = int(lengths[position])
        proposed_max = max(current_max, length)
        proposed_rows = len(current) + 1
        if current and (
            proposed_rows > max_rows or proposed_rows * proposed_max > max_residues
        ):
            batches.append(np.asarray(current, dtype=int))
            current = []
            current_max = 0
            proposed_max = length
        current.append(int(position))
        current_max = proposed_max
    if current:
        batches.append(np.asarray(current, dtype=int))
    return batches


def score_positions(
    positions,
    frame,
    uid,
    row_ligand,
    ligand,
    ligand_mask,
    pocket_context,
    model_sets,
    device,
    max_batch_rows,
    max_batch_residues,
    upstream,
):
    positions = np.asarray(positions, dtype=int)
    lengths = np.asarray(
        [
            int(pocket_context["protein_cache"][value].shape[0])
            if value in pocket_context["protein_cache"]
            else 1
            for value in uid
        ],
        dtype=int,
    )
    batches = deterministic_batches(
        positions, lengths, max_batch_rows, max_batch_residues
    )
    result = {column: np.full(len(frame), np.nan, dtype=np.float32) for column in SCORE_COLUMNS}
    placeholder_protein = torch.zeros(1, 1, upstream.PROTEIN_DIM)
    placeholder_lengths = torch.ones(1, dtype=torch.long)
    row_protein = np.zeros(len(frame), dtype=int)
    scored = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for batch_number, selected in enumerate(batches):
        batch = upstream.make_batch(
            ligand,
            ligand_mask,
            placeholder_protein,
            placeholder_lengths,
            row_ligand[selected],
            row_protein[selected],
        )
        selected_uids = uid.iloc[selected].astype(str).tolist()
        selected_batch, local = upstream.attach_selected_chains(
            batch, selected_uids, pocket_context, require_pocket=False
        )
        if len(local) != len(selected):
            raise RuntimeError("an eligible selected-chain row became unavailable")
        values = upstream.score(model_sets, selected_batch, device)
        for key, value in values.items():
            column = "p_" + key
            if column not in result:
                raise RuntimeError(f"unexpected C3 output column: {column}")
            result[column][selected] = value
        scored += len(selected)
        if batch_number == 0 or scored % 10000 < len(selected):
            print(f"C3 scored {scored}/{len(positions)} eligible rows", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    elapsed = float(time.perf_counter() - start_time)
    for column in SCORE_COLUMNS:
        values = result[column][positions]
        if len(values) and (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
            raise RuntimeError(f"invalid probability output: {column}")
    return result, lengths, elapsed, peak, len(batches)


def build_output(membership, result, positions):
    start, stop = positions
    output = membership.iloc[start:stop].copy().reset_index(drop=True)
    available = output["selected_chain_available"].astype(int).eq(1)
    output["c3_status"] = np.where(available, "scored", "selected_chain_unavailable")
    for column in SCORE_COLUMNS:
        output[column] = result[column][start:stop]
        if output.loc[available, column].isna().any():
            raise RuntimeError(f"missing eligible score in {column}")
        if output.loc[~available, column].notna().any():
            raise RuntimeError(f"unexpected unavailable-row score in {column}")
    return output


def completed_file(output_path, report_path, fingerprint, rows, row_digest):
    if not output_path.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return (
            report.get("status") == "validated"
            and report.get("contract_id") == CONTRACT_ID
            and report.get("run_fingerprint") == fingerprint
            and int(report.get("rows", -1)) == int(rows)
            and report.get("row_id_sha256") == row_digest
            and report.get("output_sha256") == sha256(output_path)
        )
    except Exception:
        return False


def run_pilot(args, root, package, upstream, upstream_package, device, run_contract, fingerprint):
    frame, membership, paths, uid = load_and_verify_membership(
        root, upstream, upstream_package, args.shard
    )
    pocket_context = upstream.load_pocket_context(root, upstream_package, uid)
    expected_available = uid.isin(pocket_context["protein_cache"]).astype(int).to_numpy()
    if not np.array_equal(expected_available, membership["selected_chain_available"].to_numpy()):
        raise RuntimeError("selected-chain availability differs from accepted full output")
    available = np.flatnonzero(expected_available)
    lengths = np.asarray([int(pocket_context["protein_cache"][uid.iloc[i]].shape[0]) for i in available])
    ordered = available[np.lexsort((available, lengths))]
    if len(ordered) > args.pilot_rows:
        indices = np.linspace(0, len(ordered) - 1, num=args.pilot_rows).round().astype(int)
        pilot = ordered[np.unique(indices)]
    else:
        pilot = ordered
    pilot = np.sort(pilot)
    with paths["mapping"].open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    row_ligand = np.asarray(
        [mapping["lig_path_to_idx"][str(x)] for x in frame["Ligand_Embedding_Path"].astype(str)],
        dtype=int,
    )
    ligand = upstream.load_mmap(paths["ligand"])
    ligand_mask = upstream.load_mmap(paths["ligand_mask"])
    model_sets = {
        arm: upstream.load_models(upstream_package, deploy_arm, ("c3",), device)
        for arm, deploy_arm in ARMS.items()
    }
    result, all_lengths, elapsed, peak, batches = score_positions(
        pilot,
        frame,
        uid,
        row_ligand,
        ligand,
        ligand_mask,
        pocket_context,
        model_sets,
        device,
        args.max_batch_rows,
        args.max_batch_residues,
        upstream,
    )
    output = membership.iloc[pilot].copy().reset_index(drop=True)
    output["c3_status"] = "scored"
    for column in SCORE_COLUMNS:
        output[column] = result[column][pilot]
    out_dir = package / "gpu_output/timing"
    output_path = out_dir / f"C3_TIMING_PREDICTIONS_SHARD{args.shard:02d}.tsv.gz"
    report_path = out_dir / f"C3_TIMING_SHARD{args.shard:02d}.json"
    atomic_tsv_gzip(output_path, output)
    full_weight = int(all_lengths[available].sum())
    pilot_weight = int(all_lengths[pilot].sum())
    report = {
        "status": "validated",
        "contract_id": CONTRACT_ID,
        "mode": "pilot",
        "shard": int(args.shard),
        "run_fingerprint": fingerprint,
        "run_contract": run_contract,
        "rows": int(len(output)),
        "full_C3_eligible_rows": int(len(available)),
        "pilot_residue_weight": pilot_weight,
        "full_residue_weight": full_weight,
        "inference_seconds": elapsed,
        "estimated_full_shard_seconds": float(elapsed * full_weight / max(pilot_weight, 1)),
        "batches": int(batches),
        "rows_per_second": float(len(output) / elapsed),
        "peak_cuda_bytes": peak,
        "row_id_sha256": row_id_digest(output["OOD_Row_ID"]),
        "output_sha256": sha256(output_path),
        "output": str(output_path.relative_to(root)),
        "network_requests_performed": 0,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def merge_chunks(package, shard, fingerprint, membership, chunk_rows, run_contract):
    chunk_dir = package / "gpu_output/chunks" / f"shard{shard:02d}"
    parts = []
    total_chunks = int(math.ceil(len(membership) / float(chunk_rows)))
    for chunk in range(total_chunks):
        start = chunk * chunk_rows
        stop = min(start + chunk_rows, len(membership))
        output_path = chunk_dir / f"chunk{chunk:03d}.tsv.gz"
        report_path = chunk_dir / f"chunk{chunk:03d}.json"
        digest = row_id_digest(membership["OOD_Row_ID"].iloc[start:stop])
        if not completed_file(output_path, report_path, fingerprint, stop - start, digest):
            raise RuntimeError(f"chunk {chunk} failed final validation")
        part = pd.read_csv(output_path, sep="\t", dtype={"OOD_Row_ID": str})
        if row_id_digest(part["OOD_Row_ID"]) != digest:
            raise RuntimeError(f"chunk {chunk} membership changed")
        parts.append(part)
    merged = pd.concat(parts, ignore_index=True)
    if len(merged) != len(membership) or not np.array_equal(
        merged["OOD_Row_ID"].astype(str).to_numpy(), membership["OOD_Row_ID"].astype(str).to_numpy()
    ):
        raise RuntimeError("merged shard membership mismatch")
    output_dir = package / "gpu_output/shards"
    output_path = output_dir / f"full_c3_predictions_shard{shard:02d}of04.tsv.gz"
    report_path = output_dir / f"C3_SHARD{shard:02d}.json"
    atomic_tsv_gzip(output_path, merged)
    available = merged["selected_chain_available"].astype(int).eq(1)
    report = {
        "status": "validated",
        "contract_id": CONTRACT_ID,
        "mode": "full",
        "shard": int(shard),
        "rows": int(len(merged)),
        "C3_scored_rows": int(available.sum()),
        "C3_unavailable_rows": int((~available).sum()),
        "chunks": total_chunks,
        "run_fingerprint": fingerprint,
        "run_contract": run_contract,
        "row_id_sha256": row_id_digest(merged["OOD_Row_ID"]),
        "output_sha256": sha256(output_path),
        "output": str(output_path.relative_to(package.parents[1])),
        "network_requests_performed": 0,
    }
    atomic_json(report_path, report)
    return report


def run_full(args, root, package, upstream, upstream_package, device, run_contract, fingerprint):
    frame, membership, paths, uid = load_and_verify_membership(
        root, upstream, upstream_package, args.shard
    )
    output_dir = package / "gpu_output/shards"
    shard_output = output_dir / f"full_c3_predictions_shard{args.shard:02d}of04.tsv.gz"
    shard_report = output_dir / f"C3_SHARD{args.shard:02d}.json"
    if completed_file(
        shard_output,
        shard_report,
        fingerprint,
        len(membership),
        row_id_digest(membership["OOD_Row_ID"]),
    ):
        print(f"SKIP hash-valid completed C3 shard {args.shard}", flush=True)
        return
    pocket_context = upstream.load_pocket_context(root, upstream_package, uid)
    expected_available = uid.isin(pocket_context["protein_cache"]).astype(int).to_numpy()
    if not np.array_equal(expected_available, membership["selected_chain_available"].to_numpy()):
        raise RuntimeError("selected-chain availability differs from accepted full output")
    with paths["mapping"].open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    row_ligand = np.asarray(
        [mapping["lig_path_to_idx"][str(x)] for x in frame["Ligand_Embedding_Path"].astype(str)],
        dtype=int,
    )
    ligand = upstream.load_mmap(paths["ligand"])
    ligand_mask = upstream.load_mmap(paths["ligand_mask"])
    model_sets = {
        arm: upstream.load_models(upstream_package, deploy_arm, ("c3",), device)
        for arm, deploy_arm in ARMS.items()
    }
    chunk_dir = package / "gpu_output/chunks" / f"shard{args.shard:02d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = int(math.ceil(len(frame) / float(args.chunk_rows)))
    for chunk in range(total_chunks):
        start = chunk * args.chunk_rows
        stop = min(start + args.chunk_rows, len(frame))
        output_path = chunk_dir / f"chunk{chunk:03d}.tsv.gz"
        report_path = chunk_dir / f"chunk{chunk:03d}.json"
        digest = row_id_digest(membership["OOD_Row_ID"].iloc[start:stop])
        if completed_file(output_path, report_path, fingerprint, stop - start, digest):
            print(f"SKIP shard {args.shard} chunk {chunk}/{total_chunks}", flush=True)
            continue
        available = np.flatnonzero(expected_available[start:stop]) + start
        result, lengths, elapsed, peak, batches = score_positions(
            available,
            frame,
            uid,
            row_ligand,
            ligand,
            ligand_mask,
            pocket_context,
            model_sets,
            device,
            args.max_batch_rows,
            args.max_batch_residues,
            upstream,
        )
        output = build_output(membership, result, (start, stop))
        atomic_tsv_gzip(output_path, output)
        report = {
            "status": "validated",
            "contract_id": CONTRACT_ID,
            "mode": "full_chunk",
            "shard": int(args.shard),
            "chunk": int(chunk),
            "rows": int(len(output)),
            "C3_scored_rows": int(len(available)),
            "C3_unavailable_rows": int(len(output) - len(available)),
            "inference_seconds": elapsed,
            "batches": int(batches),
            "peak_cuda_bytes": peak,
            "run_fingerprint": fingerprint,
            "row_id_sha256": digest,
            "output_sha256": sha256(output_path),
            "output": str(output_path.relative_to(root)),
            "network_requests_performed": 0,
        }
        atomic_json(report_path, report)
        print(
            f"PASS shard {args.shard} chunk {chunk + 1}/{total_chunks}: "
            f"{len(available)} C3 rows in {elapsed:.1f}s",
            flush=True,
        )
    report = merge_chunks(
        package, args.shard, fingerprint, membership, args.chunk_rows, run_contract
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    args = parse_args()
    if args.num_shards != NUM_SHARDS or args.shard not in range(NUM_SHARDS):
        raise SystemExit("the frozen full screen has exactly shards 0--3 of 4")
    if args.chunk_rows <= 0 or args.max_batch_rows <= 0 or args.max_batch_residues <= 0:
        raise SystemExit("batch and chunk limits must be positive")
    root = args.project_root.resolve()
    package = root / PACKAGE_REL
    upstream_package = root / UPSTREAM_REL
    upstream = load_upstream(root)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("C3 append-on requires a CUDA device")
    run_contract, fingerprint = make_run_contract(
        args, root, upstream, upstream_package
    )
    if args.mode == "pilot":
        run_pilot(
            args, root, package, upstream, upstream_package, device, run_contract, fingerprint
        )
    else:
        run_full(
            args, root, package, upstream, upstream_package, device, run_contract, fingerprint
        )


if __name__ == "__main__":
    main()
