"""Run prespecified Vina/Vinardo jobs and summarize candidate/control poses."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_job(row, executable, output):
    job = str(row["job_id"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job):
        raise ValueError("Unsafe job identifier")
    directory = output / job
    directory.mkdir(parents=True, exist_ok=False)
    pose = directory / "poses.pdbqt"
    command = [
        str(executable),
        "--receptor",
        str(row["receptor"]),
        "--ligand",
        str(row["ligand"]),
    ]
    for field in [
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "scoring",
        "seed",
        "exhaustiveness",
        "cpu",
        "num_modes",
    ]:
        command.extend(["--" + field, str(row[field])])
    command.extend(["--energy_range", "6", "--out", str(pose)])
    process = subprocess.run(command, text=True, capture_output=True)
    (directory / "docking.log").write_text(process.stdout + "\n" + process.stderr)
    if process.returncode or not pose.exists():
        raise RuntimeError(f"Docking failed: {job}")
    scores = [
        float(m.group(1))
        for m in re.finditer(r"REMARK VINA RESULT:\s+(-?[0-9.]+)", pose.read_text())
    ]
    if not scores:
        raise ValueError(f"No scored pose: {job}")
    result = {
        **row,
        "best_score": min(scores),
        "poses": len(scores),
        "pose_path": str(pose),
        "receptor_sha256": digest(row["receptor"]),
        "ligand_sha256": digest(row["ligand"]),
        "vina_sha256": digest(executable),
        "pose_sha256": digest(pose),
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize(results):
    columns = ["system_id", "ligand_id", "ligand_role", "pocket_id", "scoring"]
    summary = (
        results.groupby(columns)
        .best_score.agg(["median", "min", "max", "count"])
        .reset_index()
    )
    summary["rank_within_pocket"] = summary.groupby(
        ["system_id", "pocket_id", "scoring"]
    )["median"].rank(method="average")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--vina", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = pd.read_csv(args.jobs, sep="\t")
    required = {
        "job_id",
        "system_id",
        "ligand_id",
        "ligand_role",
        "pocket_id",
        "scoring",
        "seed",
        "receptor",
        "ligand",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "exhaustiveness",
        "cpu",
        "num_modes",
    }
    if not required.issubset(jobs) or jobs.job_id.duplicated().any():
        raise ValueError("Invalid docking job table")
    if set(jobs.scoring) != {"vina", "vinardo"}:
        raise ValueError("Both scoring functions are required")
    if not set(jobs.ligand_role).issubset({"candidate", "background", "reference"}):
        raise ValueError("Unknown ligand role")
    for _, group in jobs.groupby(["system_id", "ligand_id", "pocket_id", "scoring"]):
        if len(group) != 5 or group.seed.nunique() != 5:
            raise ValueError(
                "Each ligand/site/scoring combination needs five distinct seeds"
            )
    args.output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(
                lambda row: run_job(row, args.vina, args.output),
                jobs.to_dict("records"),
            )
        )
    results = pd.DataFrame(results)
    results.to_csv(args.output / "scores.tsv", sep="\t", index=False)
    summarize(results).to_csv(args.output / "site_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
