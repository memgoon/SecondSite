#!/usr/bin/env python3
"""Execute pending Vina jobs with bounded process-level parallelism."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete(row: pd.Series, vina_hash: str) -> bool:
    status_path, pose_path = Path(row.output_dir) / "status.json", Path(row.output_pose)
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


def execute(command: List[str]) -> Tuple[int, str]:
    result = subprocess.run(command, text=True, capture_output=True)
    return result.returncode, (result.stdout + result.stderr)[-1200:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--env-bin", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    root = args.root.resolve()
    jobs = pd.read_csv(root / "inputs" / "docking_jobs.tsv", sep="\t")
    vina = args.env_bin / "vina"
    vina_hash = sha256(vina)
    pending = [index for index, row in jobs.iterrows() if not complete(row, vina_hash)]
    progress_path = root / "logs" / "docking_progress.log"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "event": "start",
        "manifest_jobs": int(len(jobs)),
        "already_complete": int(len(jobs) - len(pending)),
        "pending": int(len(pending)),
        "workers": args.workers,
    }
    with progress_path.open("a") as handle:
        handle.write(json.dumps(header) + "\n")
    print(header, flush=True)
    if not pending:
        return
    commands: Dict[int, List[str]] = {
        index: [
            str(args.env_bin / "python"),
            str(root / "scripts" / "run_one_job.py"),
            "--manifest", str(root / "inputs" / "docking_jobs.tsv"),
            "--job-index", str(index),
            "--vina-bin", str(vina),
        ]
        for index in pending
    }
    start = time.monotonic()
    done = failed = 0
    failures: List[Dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {executor.submit(execute, command): index for index, command in commands.items()}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            done += 1
            returncode, tail = future.result()
            if returncode:
                failed += 1
                failures.append(
                    {"job_id": jobs.iloc[index].job_id, "returncode": returncode, "output_tail": tail}
                )
            if done % 10 == 0 or done == len(commands):
                message = {
                    "event": "progress",
                    "completed_this_invocation": done,
                    "pending_at_start": len(commands),
                    "failed": failed,
                    "elapsed_seconds": round(time.monotonic() - start, 1),
                }
                with progress_path.open("a") as handle:
                    handle.write(json.dumps(message) + "\n")
                print(message, flush=True)
    if failures:
        (root / "results").mkdir(parents=True, exist_ok=True)
        (root / "results" / "failed_jobs.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(f"{failed} jobs failed; see results/failed_jobs.json")


if __name__ == "__main__":
    main()
