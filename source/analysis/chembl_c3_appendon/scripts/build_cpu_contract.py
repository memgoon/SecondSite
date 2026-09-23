#!/usr/bin/env python3
"""Freeze the local inputs for the append-only ChEMBL C3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PACKAGE = Path("analysis/chembl_c3_appendon")
CONTRACT_ID = "chembl_c3_appendon_v1"
SEEDS = (20260817, 20260818, 20260819)
ARMS = ("general_every_pair", "general_protein_anchored")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery")
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_json(path: Path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_paths():
    paths = [
        PACKAGE / "README.md",
        PACKAGE / "EXPERIMENT_CONTRACT.md",
        PACKAGE / "methods/MATERIALS_AND_METHODS.md",
        PACKAGE / "scripts/build_cpu_contract.py",
        PACKAGE / "scripts/validate_cpu_contract.py",
        PACKAGE / "scripts/infer_c3_appendon.py",
        PACKAGE / "scripts/aggregate_c3_appendon.py",
        PACKAGE / "scripts/send_gpu_input.sh",
        PACKAGE / "scripts/run_gpu.sh",
        PACKAGE / "scripts/fetch_gpu_return.sh",
        Path("analysis/role_complete_pair_matrix/scripts/infer_chembl.py"),
        Path("analysis/role_complete_pair_matrix/scripts/model_definitions.py"),
        Path("analysis/role_complete_pair_matrix/scripts/gpu_preflight.py"),
        Path("analysis/role_complete_pair_matrix/validation/CPU_CONTRACT.json"),
        Path("analysis/role_complete_pair_matrix/validation/CHEMBL_SOURCE_FINGERPRINT.json"),
        Path("analysis/role_complete_pair_matrix/data/EVERY_PAIR.tsv.gz"),
        Path("analysis/role_complete_pair_matrix/data/PROTEIN_ANCHORED.tsv.gz"),
        Path("analysis/property_balanced_chembl/data/UNIPROT_PFAM_CACHE.json"),
        Path("analysis/property_balanced_chembl/data/chembl_pocket_extension/TARGET_CHAIN_SEQUENCES.tsv"),
        Path("analysis/property_balanced_chembl/data/chembl_pocket_extension/POCKET_INDICES.json"),
        Path("analysis/property_balanced_chembl/data/chembl_pocket_extension/POCKET_ALIGNMENT_AUDIT.tsv"),
        Path("analysis/property_balanced_chembl/data/chembl_pocket_extension/POCKET_COVERAGE.json"),
        Path("analysis/property_balanced_chembl/gpu_output/pocket_extension/C3_TARGET_CHAIN_EMBEDDING_VALIDATION.json"),
    ]
    for shard in range(4):
        paths.extend(
            [
                Path(
                    "analysis/role_complete_pair_matrix/gpu_output/chembl/full/"
                    f"full_predictions_shard{shard:02d}of04.tsv.gz"
                ),
                Path(
                    "analysis/role_complete_pair_matrix/gpu_output/chembl/full/"
                    f"FULL_SHARD{shard:02d}.json"
                ),
            ]
        )
    for arm in ARMS:
        for seed in SEEDS:
            base = Path(
                "analysis/role_complete_pair_matrix/gpu_output/deploy"
            ) / arm / "c3" / f"seed_{seed}"
            paths.extend([base / "deploy.pt", base / "DEPLOY_REPORT.json"])
    return paths


def main():
    args = parse_args()
    root = args.project_root.resolve()
    paths = source_paths()
    records = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records[str(relative)] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }

    upstream_reports = []
    total_rows = 0
    selected_rows = 0
    for shard in range(4):
        relative = Path(
            "analysis/role_complete_pair_matrix/gpu_output/chembl/full/"
            f"FULL_SHARD{shard:02d}.json"
        )
        report = json.loads((root / relative).read_text(encoding="utf-8"))
        if (
            report.get("status") != "validated"
            or report.get("contract_id") != "role_complete_pair_matrix_v2"
            or "c3" in {m for values in report.get("models_per_arm", {}).values() for m in values}
            or int(report.get("shard", -1)) != shard
        ):
            raise RuntimeError(f"invalid accepted upstream shard report: {relative}")
        total_rows += int(report["rows"])
        selected_rows += int(report["selected_chain_available_rows"])
        upstream_reports.append(
            {
                "shard": shard,
                "rows": int(report["rows"]),
                "selected_chain_available_rows": int(
                    report["selected_chain_available_rows"]
                ),
                "run_fingerprint": report["run_fingerprint"],
                "output_sha256": report["output_sha256"],
            }
        )
    if total_rows != 799873 or selected_rows != 785182:
        raise RuntimeError(
            f"accepted full membership changed: rows={total_rows}, selected={selected_rows}"
        )

    checkpoint_records = []
    for arm in ARMS:
        for seed in SEEDS:
            base = Path("analysis/role_complete_pair_matrix/gpu_output/deploy") / arm / "c3" / f"seed_{seed}"
            report_path = root / base / "DEPLOY_REPORT.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            checkpoint_path = root / base / "deploy.pt"
            actual = sha256(checkpoint_path)
            if (
                report.get("status") != "validated"
                or report.get("contract_id") != "role_complete_pair_matrix_v2"
                or report.get("deploy_arm") != arm
                or report.get("model") != "c3"
                or int(report.get("seed", -1)) != seed
                or report.get("checkpoint_sha256") != actual
            ):
                raise RuntimeError(f"invalid C3 checkpoint provenance: {base}")
            checkpoint_records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "epochs": int(report["epochs"]),
                    "checkpoint_sha256": actual,
                    "epoch_source_fingerprint": report["epoch_source_fingerprint"],
                }
            )

    contract_path = root / PACKAGE / "validation/CPU_CONTRACT.json"
    contract = {
        "status": "validated",
        "contract_id": CONTRACT_ID,
        "purpose": "append_only_missing_C3_full_ChEMBL_inference",
        "existing_full_outputs_modified": False,
        "network_required": False,
        "expected": {
            "source_shards": 4,
            "rows": total_rows,
            "selected_chain_available_rows": selected_rows,
            "selected_chain_unavailable_rows": total_rows - selected_rows,
            "C3_score_columns": [
                "p_every_pair_c3_mean",
                "p_every_pair_c3_sd",
                "p_general_c3_mean",
                "p_general_c3_sd",
            ],
            "deployment_arms": list(ARMS),
            "seeds": list(SEEDS),
        },
        "production_defaults": {
            "chunk_rows": 10000,
            "max_batch_rows": 8,
            "max_batch_residues": 4096,
            "pilot_rows_per_shard": 2500,
        },
        "upstream_full_shards": upstream_reports,
        "C3_checkpoints": checkpoint_records,
        "files": records,
    }
    atomic_json(contract_path, contract)

    manifest_targets = paths + [PACKAGE / "validation/CPU_CONTRACT.json"]
    lines = [f"{sha256(root / path)}  {path}" for path in sorted(manifest_targets, key=str)]
    atomic_text(root / PACKAGE / "manifests/CPU_INPUTS.sha256", "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": "validated",
                "contract_id": CONTRACT_ID,
                "input_files": len(manifest_targets),
                "rows": total_rows,
                "selected_chain_available_rows": selected_rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

