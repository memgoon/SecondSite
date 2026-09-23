#!/usr/bin/env python3
"""Build a deterministic docking manifest for named and de-novo sites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return 1 + int(hashlib.sha256(payload).hexdigest()[:8], 16) % 2_000_000_000


def contract_hash(values: Dict[str, object]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def select_systems(systems: pd.DataFrame, profile: str) -> pd.DataFrame:
    if profile == "full":
        return systems.copy()
    if profile == "candidates":
        return systems[systems.system_role != "matched_background"].copy()
    if profile == "cd73":
        return systems[systems.target_id == "CD73"].copy()
    if profile == "cd73_candidate":
        return systems[systems.system_id == "CD73_CHEMBL4549571"].copy()
    raise ValueError(f"Unknown profile {profile!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--profile", choices=["full", "candidates", "cd73", "cd73_candidate"], default="full")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--exhaustiveness", type=int, default=24)
    parser.add_argument("--cpu-per-job", type=int, default=2)
    parser.add_argument("--num-modes", type=int, default=10)
    args = parser.parse_args()
    if args.seeds < 2 or min(args.exhaustiveness, args.cpu_per_job, args.num_modes) < 1:
        raise SystemExit("seeds must be >=2 and numeric docking settings must be positive")

    root = args.root.resolve()
    systems = select_systems(pd.read_csv(root / "systems.tsv", sep="\t"), args.profile)
    ligands = pd.read_csv(root / "ligands.tsv", sep="\t")
    boxes = pd.read_csv(root / "inputs" / "docking_boxes.tsv", sep="\t")
    if systems.empty:
        raise RuntimeError(f"Profile {args.profile} selected no systems")
    systems = systems.merge(ligands[["ligand_key", "ligand_chembl_id", "ligand_name", "ligand_role"]], on="ligand_key", how="left", validate="many_to_one")

    rows: List[Dict[str, object]] = []
    for system in systems.itertuples(index=False):
        receptor = root / "inputs" / "receptors" / system.target_id / "receptor.pdbqt"
        ligand = root / "inputs" / "ligands" / system.ligand_key / "ligand.pdbqt"
        if not receptor.exists() or not ligand.exists():
            raise RuntimeError(
                f"Missing prepared receptor/ligand for {system.system_id}: receptor={receptor.exists()} ligand={ligand.exists()}"
            )
        target_boxes = boxes[boxes.target_id == system.target_id]
        if target_boxes.empty:
            raise RuntimeError(f"No boxes for {system.target_id}")
        for box in target_boxes.itertuples(index=False):
            for scoring in ("vina", "vinardo"):
                for replicate in range(args.seeds):
                    seed = stable_seed(
                        "chembl-competitive-docking-v2",
                        system.system_id,
                        box.pocket_id,
                        scoring,
                        replicate,
                    )
                    output_directory = (
                        root
                        / "runs"
                        / system.system_id
                        / box.pocket_id
                        / scoring
                        / f"seed_{seed}"
                    )
                    contract = {
                        "system_id": system.system_id,
                        "target_id": system.target_id,
                        "ligand_key": system.ligand_key,
                        "pocket_id": box.pocket_id,
                        "pocket_class": box.pocket_class,
                        "scoring": scoring,
                        "replicate": replicate,
                        "seed": seed,
                        "center": [box.center_x, box.center_y, box.center_z],
                        "size": [box.size_x, box.size_y, box.size_z],
                        "exhaustiveness": args.exhaustiveness,
                        "cpu": args.cpu_per_job,
                        "num_modes": args.num_modes,
                    }
                    rows.append(
                        {
                            "job_id": f"JOB_{len(rows):06d}",
                            "profile": args.profile,
                            "system_id": system.system_id,
                            "system_role": system.system_role,
                            "target_id": system.target_id,
                            "ligand_key": system.ligand_key,
                            "ligand_chembl_id": system.ligand_chembl_id,
                            "pocket_id": box.pocket_id,
                            "pocket_class": box.pocket_class,
                            "claim_role": box.claim_role,
                            "scoring": scoring,
                            "replicate": replicate,
                            "seed": seed,
                            "center_x": box.center_x,
                            "center_y": box.center_y,
                            "center_z": box.center_z,
                            "size_x": box.size_x,
                            "size_y": box.size_y,
                            "size_z": box.size_z,
                            "exhaustiveness": args.exhaustiveness,
                            "cpu": args.cpu_per_job,
                            "num_modes": args.num_modes,
                            "receptor": str(receptor),
                            "ligand": str(ligand),
                            "output_dir": str(output_directory),
                            "output_pose": str(output_directory / "poses.pdbqt"),
                            "output_log": str(output_directory / "vina.log"),
                            "job_contract_sha256": contract_hash(contract),
                        }
                    )
    jobs = pd.DataFrame(rows)
    manifest_path = root / "inputs" / "docking_jobs.tsv"
    jobs.to_csv(manifest_path, sep="\t", index=False)
    summary = {
        "status": "validated",
        "profile": args.profile,
        "n_jobs": int(len(jobs)),
        "n_systems": int(jobs.system_id.nunique()),
        "n_targets": int(jobs.target_id.nunique()),
        "n_ligands": int(jobs.ligand_key.nunique()),
        "n_system_pocket_pairs": int(jobs[["system_id", "pocket_id"]].drop_duplicates().shape[0]),
        "scoring_functions": sorted(jobs.scoring.unique().tolist()),
        "seeds_per_system_pocket_scoring": args.seeds,
        "exhaustiveness": args.exhaustiveness,
        "cpu_per_job": args.cpu_per_job,
        "num_modes": args.num_modes,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (root / "inputs" / "JOB_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
