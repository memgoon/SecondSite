#!/usr/bin/env python3
"""Aggregate and independently validate the append-only C3 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACT_ID = "chembl_c3_appendon_v1"
PACKAGE_REL = Path("analysis/chembl_c3_appendon")
UPSTREAM_REL = Path("analysis/role_complete_pair_matrix")
SCORE_COLUMNS = [
    "p_every_pair_c3_mean",
    "p_every_pair_c3_sd",
    "p_general_c3_mean",
    "p_general_c3_sd",
]
IDENTITY_COLUMNS = [
    "OOD_Row_ID",
    "stable_pair_key",
    "uniprot",
    "connectivity_key",
    "selected_chain_available",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_id_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_json(path: Path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


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


def verify_manifest(root: Path, path: Path):
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        if not match:
            failures.append(f"malformed:{line}")
            continue
        expected, relative = match.groups()
        target = root / relative
        if not target.is_file():
            failures.append(f"missing:{relative}")
        elif sha256(target) != expected:
            failures.append(f"hash:{relative}")
    return failures


def aggregate_pilot(root: Path, package: Path, write: bool, validate_only: bool):
    reports = []
    for shard in range(4):
        path = package / "gpu_output/timing" / f"C3_TIMING_SHARD{shard:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        output = Path(report.get("output", ""))
        if not output.is_absolute():
            output = root / output
        if (
            report.get("status") != "validated"
            or report.get("contract_id") != CONTRACT_ID
            or int(report.get("shard", -1)) != shard
            or not output.is_file()
            or report.get("output_sha256") != sha256(output)
        ):
            raise RuntimeError(f"invalid pilot report for shard {shard}")
        reports.append(report)
    estimates = [float(report["estimated_full_shard_seconds"]) for report in reports]
    two_gpu_wall = max(estimates[0], estimates[1]) + max(estimates[2], estimates[3])
    result = {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "pilot_rows": int(sum(int(x["rows"]) for x in reports)),
        "pilot_inference_seconds_sum": float(sum(float(x["inference_seconds"]) for x in reports)),
        "estimated_shard_seconds": estimates,
        "estimated_two_GPU_wall_seconds": float(two_gpu_wall),
        "estimated_two_GPU_wall_hours": float(two_gpu_wall / 3600.0),
        "maximum_peak_cuda_bytes": int(max(int(x["peak_cuda_bytes"]) for x in reports)),
        "batch_contract_identical_to_production": True,
        "network_requests_performed": 0,
        "interpretation": "timing estimate excludes one-time preflight and final aggregation",
    }
    output_path = package / "gpu_output/timing/C3_TIMING_SUMMARY.json"
    if validate_only:
        observed = json.loads(output_path.read_text(encoding="utf-8"))
        if observed != result:
            raise RuntimeError("stored timing summary differs from recomputation")
    elif write:
        atomic_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def validate_shards(root: Path, package: Path, upstream: Path):
    parts = []
    shard_records = []
    for shard in range(4):
        output_path = package / "gpu_output/shards" / f"full_c3_predictions_shard{shard:02d}of04.tsv.gz"
        report_path = package / "gpu_output/shards" / f"C3_SHARD{shard:02d}.json"
        old_path = upstream / "gpu_output/chembl/full" / f"full_predictions_shard{shard:02d}of04.tsv.gz"
        old_report_path = upstream / "gpu_output/chembl/full" / f"FULL_SHARD{shard:02d}.json"
        for path in [output_path, report_path, old_path, old_report_path]:
            if not path.is_file():
                raise FileNotFoundError(path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "validated"
            or report.get("contract_id") != CONTRACT_ID
            or int(report.get("shard", -1)) != shard
            or report.get("output_sha256") != sha256(output_path)
            or old_report.get("status") != "validated"
            or old_report.get("output_sha256") != sha256(old_path)
        ):
            raise RuntimeError(f"invalid output/report hash for shard {shard}")
        value = pd.read_csv(output_path, sep="\t", dtype={"OOD_Row_ID": str})
        old = pd.read_csv(
            old_path,
            sep="\t",
            usecols=IDENTITY_COLUMNS,
            dtype={
                "OOD_Row_ID": str,
                "stable_pair_key": str,
                "uniprot": str,
                "connectivity_key": str,
            },
        )
        expected_columns = IDENTITY_COLUMNS + ["c3_status"] + SCORE_COLUMNS
        if value.columns.tolist() != expected_columns:
            raise RuntimeError(f"C3 shard schema mismatch: {shard}")
        if len(value) != len(old) or int(report.get("rows", -1)) != len(value):
            raise RuntimeError(f"C3 shard row count mismatch: {shard}")
        for column in IDENTITY_COLUMNS:
            left = value[column].fillna("").astype(str).to_numpy()
            right = old[column].fillna("").astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise RuntimeError(f"C3/upstream {column} mismatch: shard {shard}")
        available = pd.to_numeric(value["selected_chain_available"], errors="raise").astype(int).eq(1)
        if not value.loc[available, "c3_status"].eq("scored").all():
            raise RuntimeError(f"eligible C3 status mismatch: shard {shard}")
        if not value.loc[~available, "c3_status"].eq("selected_chain_unavailable").all():
            raise RuntimeError(f"unavailable C3 status mismatch: shard {shard}")
        for column in SCORE_COLUMNS:
            score = pd.to_numeric(value[column], errors="coerce")
            if score.loc[available].isna().any() or not score.loc[available].between(0, 1).all():
                raise RuntimeError(f"invalid eligible scores: {column} shard {shard}")
            if score.loc[~available].notna().any():
                raise RuntimeError(f"unexpected unavailable scores: {column} shard {shard}")
        if value["OOD_Row_ID"].duplicated().any():
            raise RuntimeError(f"duplicate row IDs within shard {shard}")
        if report.get("row_id_sha256") != row_id_digest(value["OOD_Row_ID"]):
            raise RuntimeError(f"row digest mismatch: shard {shard}")
        parts.append(value)
        shard_records.append(
            {
                "shard": shard,
                "rows": int(len(value)),
                "C3_scored_rows": int(available.sum()),
                "C3_unavailable_rows": int((~available).sum()),
                "output_sha256": sha256(output_path),
                "report_sha256": sha256(report_path),
                "run_fingerprint": report["run_fingerprint"],
            }
        )
    union = pd.concat(parts, ignore_index=True)
    if union["OOD_Row_ID"].duplicated().any():
        raise RuntimeError("duplicate OOD_Row_ID across C3 shards")
    return union, shard_records


def derived_outputs(root: Path, package: Path, union: pd.DataFrame, shard_records):
    available = union["selected_chain_available"].astype(int).eq(1)
    validation = {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "source_shards": 4,
        "rows": int(len(union)),
        "unique_OOD_Row_IDs": int(union["OOD_Row_ID"].nunique()),
        "unique_pairs": int(union["stable_pair_key"].nunique()),
        "unique_targets": int(union["uniprot"].nunique()),
        "C3_scored_rows": int(available.sum()),
        "C3_unavailable_rows": int((~available).sum()),
        "C3_score_columns": SCORE_COLUMNS,
        "all_eligible_probabilities_finite_and_bounded": True,
        "all_unavailable_probabilities_NA": True,
        "row_membership_exactly_matches_accepted_full_screen": True,
        "existing_seven_model_outputs_modified": False,
        "model_retraining_performed": False,
        "network_requests_performed": 0,
        "shards": shard_records,
        "interpretation_limit": "append-only C3 external ranking sensitivity; not binding-site or mechanism ground truth",
    }
    if validation["rows"] != 799873 or validation["C3_scored_rows"] != 785182:
        raise RuntimeError("frozen union count mismatch")
    if validation["unique_OOD_Row_IDs"] != validation["rows"]:
        raise RuntimeError("union row identifiers are not unique")
    status = {
        "status": "complete",
        "contract_id": CONTRACT_ID,
        "validation_status": "PASS",
        "rows": validation["rows"],
        "C3_scored_rows": validation["C3_scored_rows"],
        "C3_unavailable_rows": validation["C3_unavailable_rows"],
        "existing_full_screen_preserved": True,
        "web_handoff_updated": False,
        "next_step": "join four C3 score columns to the web handoff by unique OOD_Row_ID",
    }
    report = f"""# ChEMBL C3 append-on report

The missing C3 full-screen predictions were computed without retraining and
without modifying the accepted seven-model ChEMBL outputs.

- source rows: {validation['rows']:,}
- C3-scored rows: {validation['C3_scored_rows']:,}
- selected-chain-unavailable rows retained as NA: {validation['C3_unavailable_rows']:,}
- unique targets: {validation['unique_targets']:,}
- source shards: 4/4 PASS
- network requests: 0

The join-ready output contains only immutable row identity, selected-chain
availability, and the every-pair/protein-anchored C3 ensemble mean and standard
deviation.  It must be joined to the existing rankings by unique `OOD_Row_ID`.
This package does not claim binding, mechanism, competitive/noncompetitive
action, or allostery ground truth.
"""
    return validation, status, report


def portable_targets(root: Path, package: Path):
    targets = [
        package / "README.md",
        package / "EXPERIMENT_CONTRACT.md",
        package / "methods/MATERIALS_AND_METHODS.md",
        package / "validation/CPU_CONTRACT.json",
        package / "validation/CPU_VALIDATION.json",
        package / "manifests/CPU_INPUTS.sha256",
        package / "gpu_output/aggregate/C3_APPENDON_PREDICTIONS.tsv.gz",
        package / "gpu_output/aggregate/C3_APPENDON_VALIDATION.json",
        package / "reports/C3_APPENDON_REPORT.md",
        package / "work/status/C3_APPENDON_STATUS.json",
    ]
    targets.extend(sorted((package / "scripts").glob("*.py")))
    targets.extend(sorted((package / "scripts").glob("*.sh")))
    for shard in range(4):
        targets.extend(
            [
                package / "gpu_output/shards" / f"full_c3_predictions_shard{shard:02d}of04.tsv.gz",
                package / "gpu_output/shards" / f"C3_SHARD{shard:02d}.json",
            ]
        )
    timing = package / "gpu_output/timing/C3_TIMING_SUMMARY.json"
    if timing.is_file():
        targets.append(timing)
        for shard in range(4):
            targets.extend(
                [
                    package / "gpu_output/timing" / f"C3_TIMING_PREDICTIONS_SHARD{shard:02d}.tsv.gz",
                    package / "gpu_output/timing" / f"C3_TIMING_SHARD{shard:02d}.json",
                ]
            )
    missing = [path for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return targets


def main():
    args = parse_args()
    if args.write and args.validate_only:
        raise SystemExit("choose --write or --validate-only")
    root = args.project_root.resolve()
    package = root / PACKAGE_REL
    upstream = root / UPSTREAM_REL
    if args.pilot_only:
        aggregate_pilot(root, package, args.write, args.validate_only)
        return
    cpu_failures = verify_manifest(root, package / "manifests/CPU_INPUTS.sha256")
    if cpu_failures:
        raise RuntimeError(f"CPU input checksum failures: {cpu_failures[:3]}")
    union, shard_records = validate_shards(root, package, upstream)
    validation, status, report = derived_outputs(root, package, union, shard_records)
    output_path = package / "gpu_output/aggregate/C3_APPENDON_PREDICTIONS.tsv.gz"
    validation_path = package / "gpu_output/aggregate/C3_APPENDON_VALIDATION.json"
    report_path = package / "reports/C3_APPENDON_REPORT.md"
    status_path = package / "work/status/C3_APPENDON_STATUS.json"
    manifest_path = package / "manifests/GPU_RESULTS_PORTABLE.sha256"
    if args.validate_only:
        if not output_path.is_file() or sha256(output_path) != json.loads(
            validation_path.read_text(encoding="utf-8")
        ).get("aggregate_output_sha256"):
            raise RuntimeError("aggregate output hash mismatch")
        expected_validation = dict(validation)
        expected_validation["aggregate_output_sha256"] = sha256(output_path)
        if json.loads(validation_path.read_text(encoding="utf-8")) != expected_validation:
            raise RuntimeError("stored validation differs from independent recomputation")
        if json.loads(status_path.read_text(encoding="utf-8")) != status:
            raise RuntimeError("stored status differs from independent recomputation")
        if report_path.read_text(encoding="utf-8") != report:
            raise RuntimeError("stored report differs from independent recomputation")
        failures = verify_manifest(root, manifest_path)
        if failures:
            raise RuntimeError(f"portable checksum failures: {failures[:3]}")
        print(json.dumps(expected_validation, indent=2, sort_keys=True))
        return
    if not args.write:
        raise SystemExit("use --write to build outputs or --validate-only to verify them")
    atomic_tsv_gzip(output_path, union)
    validation["aggregate_output_sha256"] = sha256(output_path)
    atomic_json(validation_path, validation)
    atomic_text(report_path, report)
    atomic_json(status_path, status)
    targets = portable_targets(root, package)
    lines = []
    for target in sorted(targets, key=lambda x: str(x.relative_to(root))):
        lines.append(f"{sha256(target)}  {target.relative_to(root)}")
    atomic_text(manifest_path, "\n".join(lines) + "\n")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

