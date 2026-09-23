#!/usr/bin/env python3
"""Run one manifest row with hash-gated resumability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_complete(status_path: Path, pose_path: Path, row: pd.Series, vina_hash: str) -> bool:
    if not status_path.exists() or not pose_path.exists() or pose_path.stat().st_size == 0:
        return False
    try:
        status = json.loads(status_path.read_text())
        return (
            status.get("status") == "complete"
            and status.get("job_contract_sha256") == row.job_contract_sha256
            and status.get("receptor_sha256") == sha256(Path(row.receptor))
            and status.get("ligand_sha256") == sha256(Path(row.ligand))
            and status.get("vina_binary_sha256") == vina_hash
        )
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--job-index", required=True, type=int)
    parser.add_argument("--vina-bin", required=True, type=Path)
    args = parser.parse_args()
    jobs = pd.read_csv(args.manifest, sep="\t")
    row = jobs.iloc[args.job_index]
    receptor, ligand = Path(row.receptor), Path(row.ligand)
    output_directory = Path(row.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    pose_path, log_path = Path(row.output_pose), Path(row.output_log)
    status_path = output_directory / "status.json"
    vina_hash = sha256(args.vina_bin)
    if is_complete(status_path, pose_path, row, vina_hash):
        print(f"already-complete {row.job_id}", flush=True)
        return

    command = [
        str(args.vina_bin),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", str(row.center_x),
        "--center_y", str(row.center_y),
        "--center_z", str(row.center_z),
        "--size_x", str(row.size_x),
        "--size_y", str(row.size_y),
        "--size_z", str(row.size_z),
        "--scoring", str(row.scoring),
        "--seed", str(int(row.seed)),
        "--exhaustiveness", str(int(row.exhaustiveness)),
        "--cpu", str(int(row.cpu)),
        "--num_modes", str(int(row.num_modes)),
        "--energy_range", "6",
        "--out", str(pose_path),
    ]
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    start = time.monotonic()
    result = subprocess.run(command, text=True, capture_output=True, env=environment)
    log_path.write_text(
        "COMMAND\n{}\n\nSTDOUT\n{}\nSTDERR\n{}".format(" ".join(command), result.stdout, result.stderr)
    )
    complete = result.returncode == 0 and pose_path.exists() and pose_path.stat().st_size > 0
    status = {
        "job_id": row.job_id,
        "system_id": row.system_id,
        "target_id": row.target_id,
        "pocket_id": row.pocket_id,
        "scoring": row.scoring,
        "seed": int(row.seed),
        "status": "complete" if complete else "failed",
        "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - start,
        "job_contract_sha256": row.job_contract_sha256,
        "receptor_sha256": sha256(receptor),
        "ligand_sha256": sha256(ligand),
        "vina_binary_sha256": vina_hash,
        "command": command,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    if not complete:
        raise SystemExit(result.returncode or 1)
    print(f"complete {row.job_id} {row.system_id} {row.pocket_id} {row.scoring}", flush=True)


if __name__ == "__main__":
    main()
