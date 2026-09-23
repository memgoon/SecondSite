#!/usr/bin/env python3
"""Independent validation for checkpoint-6 exact-reference geometry."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

BUILD = VALIDATION / "CHECKPOINT6_BUILD_SUMMARY.json"
REFERENCE = DATA / "BIOLIP_EXACT_SITE_REFERENCE_OBSERVATIONS.tsv.gz"
REGISTRY = DATA / "CHECKPOINT5_ALLOBENCH_SOURCE_REGISTRY.tsv.gz"
GEOMETRY = DATA / "CHECKPOINT6_EXACT_REFERENCE_GEOMETRY.tsv.gz"
COORDINATES = DATA / "CHECKPOINT6_COORDINATE_MANIFEST.tsv.gz"
COMPARISON = DATA / "CHECKPOINT6_DISTANCE_DEFINITION_COMPARISON.tsv"
OVERLAP_AUDIT = DATA / "CHECKPOINT6_REPORTED_DISTINCT_RESIDUE_OVERLAP_AUDIT.tsv.gz"
FETCH_LOG = DATA / "CHECKPOINT6_COORDINATE_FETCH_LOG.tsv"
SPEC = MANIFESTS / "CHECKPOINT6_DISTANCE_SPEC.json"
INPUT_HASHES = MANIFESTS / "CHECKPOINT6_INPUT_HASHES.tsv"
OUTPUT = VALIDATION / "CHECKPOINT6_VALIDATION.json"

METRICS = [
    "ligand_to_orthosteric_site_min_heavy_A",
    "ligand_to_orthosteric_site_min_CA_A",
    "candidate_site_to_orthosteric_site_min_heavy_A",
    "candidate_site_to_orthosteric_site_min_CA_A",
    "ligand_centroid_to_orthosteric_site_heavy_centroid_A",
    "ligand_centroid_to_orthosteric_site_CA_centroid_A",
    "candidate_site_CA_centroid_to_orthosteric_site_CA_centroid_A",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def close(first: float, second: float, tolerance: float = 1e-12) -> bool:
    if pd.isna(first) and pd.isna(second):
        return True
    return math.isclose(float(first), float(second), rel_tol=tolerance, abs_tol=tolerance)


def protein_macro_auc(frame: pd.DataFrame, metric: str):
    values = []
    rows = 0
    for _protein, group in frame.groupby("uniprot"):
        subset = group[["reference_label", metric]].dropna()
        if subset.reference_label.nunique() != 2:
            continue
        values.append(roc_auc_score(subset.reference_label.eq("allosteric"), subset[metric]))
        rows += len(subset)
    return float(np.mean(values)), len(values), rows


def union_active_positions(source_ids: str, active_by_source: dict) -> str:
    values = set()
    for source_id in filter(None, str(source_ids).split(";")):
        for token in str(active_by_source.get(source_id, "")).split(";"):
            if token and token.lstrip("-").isdigit():
                values.add(int(token))
    return ";".join(str(value) for value in sorted(values))


def main() -> None:
    build = json.loads(BUILD.read_text())
    if build.get("checkpoint") != 6 or build.get("status") != "complete_pending_independent_validation":
        raise RuntimeError("checkpoint-6 build state is invalid")
    for name, record in build["outputs"].items():
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint-6 output hash mismatch: {name}")

    input_hashes = pd.read_csv(INPUT_HASHES, sep="\t", dtype=str, keep_default_na=False)
    if input_hashes.asset_id.duplicated().any():
        raise RuntimeError("checkpoint-6 input asset IDs are duplicated")
    for row in input_hashes.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or str(path.stat().st_size) != row.size_bytes or sha256(path) != row.sha256:
            raise RuntimeError(f"checkpoint-6 input hash mismatch: {row.asset_id}")

    reference = pd.read_csv(REFERENCE, sep="\t", dtype=str, keep_default_na=False)
    geometry = pd.read_csv(GEOMETRY, sep="\t", dtype=str, keep_default_na=False)
    numeric_geometry_columns = METRICS + [
        "exact_nonpolymer_ligand_matches", "polymer_name_collisions_excluded",
        "candidate_orthosteric_site_residue_overlap_count", "candidate_residues_recovered",
    ]
    for column in numeric_geometry_columns:
        geometry[column] = pd.to_numeric(geometry[column], errors="coerce")
    coordinates = pd.read_csv(COORDINATES, sep="\t", dtype=str, keep_default_na=False)
    comparison = pd.read_csv(COMPARISON, sep="\t")
    overlap_audit = pd.read_csv(OVERLAP_AUDIT, sep="\t", dtype=str, keep_default_na=False)
    registry = pd.read_csv(REGISTRY, sep="\t", dtype=str, keep_default_na=False)
    fetch_log = pd.read_csv(FETCH_LOG, sep="\t", dtype=str, keep_default_na=False)
    spec = json.loads(SPEC.read_text())

    if len(reference) != 6117 or len(geometry) != len(reference):
        raise RuntimeError("checkpoint-5 reference universe row count changed")
    if geometry.observation_id.duplicated().any() or reference.observation_id.duplicated().any():
        raise RuntimeError("duplicate exact reference observation")
    if set(geometry.observation_id) != set(reference.observation_id):
        raise RuntimeError("geometry does not preserve every checkpoint-5 observation")
    identity_columns = [
        "source_ordinal", "reference_label", "pdb_id", "receptor_chain", "uniprot",
        "ligand_ccd", "ligand_chain", "ligand_auth_seq_id", "full_inchikey",
    ]
    identity = reference[["observation_id"] + identity_columns].merge(
        geometry[["observation_id"] + identity_columns], on="observation_id",
        suffixes=("_reference", "_geometry"), validate="one_to_one",
    )
    mismatches = 0
    for column in identity_columns:
        left = identity[f"{column}_reference"].astype(str).fillna("")
        right = identity[f"{column}_geometry"].astype(str).fillna("")
        mismatches += int(left.ne(right).sum())
    if mismatches:
        raise RuntimeError(f"checkpoint-5/6 observation identity mismatches: {mismatches}")

    if coordinates.pdb_id.duplicated().any() or len(coordinates) != reference.pdb_id.nunique():
        raise RuntimeError("coordinate manifest is not one row per exact-reference PDB")
    # Do not open/stat the 2,868 coordinate files a second time: the geometry
    # build already parsed every file and froze path/size/mtime.  A second pass
    # would add substantial shared-filesystem I/O without validating the numeric
    # result independently.  Here we validate manifest completeness and the
    # successful per-observation parse/geometry accounting below.
    if coordinates.coordinate_path.eq("").any():
        raise RuntimeError("coordinate manifest contains a blank resolved path")
    if (pd.to_numeric(coordinates.size_bytes, errors="coerce") <= 0).any():
        raise RuntimeError("coordinate manifest contains a non-positive file size")
    if (pd.to_numeric(coordinates.mtime_epoch, errors="coerce") <= 0).any():
        raise RuntimeError("coordinate manifest contains an invalid mtime")

    status_counts = geometry.geometry_status.value_counts().to_dict()
    expected_status = {
        "ok": 6054,
        "orthosteric_site_unmapped": 48,
        "active_site_mapping_failed_checkpoint4_crosscheck": 15,
    }
    if status_counts != expected_status or status_counts != build["geometry_status_counts"]:
        raise RuntimeError(f"unexpected checkpoint-6 geometry status counts: {status_counts}")
    orthosteric = geometry.loc[geometry.reference_label.eq("orthosteric")]
    allosteric = geometry.loc[geometry.reference_label.eq("allosteric")]
    if len(orthosteric) != 4090 or not orthosteric.geometry_status.eq("ok").all():
        raise RuntimeError("all 4,090 exact orthosteric controls must have valid geometry")
    if len(allosteric) != 2027 or int(allosteric.geometry_status.eq("ok").sum()) != 1964:
        raise RuntimeError("allosteric geometry accounting changed")

    ok = geometry.loc[geometry.geometry_status.eq("ok")].copy()
    numeric = ok[METRICS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric.to_numpy() < -1e-10).any():
        raise RuntimeError("valid geometry contains missing, infinite, or negative distance")
    if not ok.exact_nonpolymer_ligand_matches.eq(1).all():
        raise RuntimeError("valid geometry lacks one exact non-polymer ligand instance")
    if int(ok.polymer_name_collisions_excluded.fillna(0).sum()) != 0:
        raise RuntimeError("a polymer residue name collision was unexpectedly accepted")

    orth_ok = ok.loc[ok.reference_label.eq("orthosteric")]
    for metric in (
        "candidate_site_to_orthosteric_site_min_heavy_A",
        "candidate_site_to_orthosteric_site_min_CA_A",
        "candidate_site_CA_centroid_to_orthosteric_site_CA_centroid_A",
    ):
        if not np.allclose(orth_ok[metric].astype(float), 0.0, atol=1e-10):
            raise RuntimeError(f"exact orthosteric self-site invariant failed: {metric}")
    if not np.array_equal(
        orth_ok.candidate_orthosteric_site_residue_overlap_count.astype(int).to_numpy(),
        orth_ok.candidate_residues_recovered.astype(int).to_numpy(),
    ):
        raise RuntimeError("orthosteric self-site residue overlap invariant failed")

    active_by_source = registry.set_index("allobench_source_id").active_site_uniprot_positions.to_dict()
    expected_orthosteric_site = allosteric.reference_source_ids.map(
        lambda value: union_active_positions(value, active_by_source)
    )
    if not expected_orthosteric_site.eq(
        allosteric.orthosteric_site_uniprot_positions.fillna("")
    ).all():
        raise RuntimeError("allosteric orthosteric-site union differs from AlloBench registry")

    comparison_universe = ok.loc[
        ok.reference_label.eq("orthosteric")
        | ok.source_site_relation.eq("distinct_from_orthosteric_site")
    ].copy()
    comparison_universe["binary_label"] = (
        comparison_universe.reference_label.eq("allosteric").astype(int)
    )
    if len(comparison_universe) != 5798 or int(comparison_universe.binary_label.sum()) != 1708:
        raise RuntimeError("distance-comparison universe changed")
    strict = comparison_universe.loc[
        comparison_universe.reference_label.eq("orthosteric")
        | (
            comparison_universe.reference_label.eq("allosteric")
            & comparison_universe.candidate_orthosteric_site_residue_overlap_count.eq(0)
        )
    ].copy()
    if len(strict) != 5129 or int(strict.binary_label.sum()) != 1039:
        raise RuntimeError("no-residue-overlap sensitivity universe changed")

    comparison_by_metric = comparison.set_index("metric")
    if set(comparison_by_metric.index) != set(METRICS):
        raise RuntimeError("distance-comparison metric set changed")
    for metric in METRICS:
        row = comparison_by_metric.loc[metric]
        auc = roc_auc_score(comparison_universe.binary_label, comparison_universe[metric])
        macro, groups, rows = protein_macro_auc(comparison_universe, metric)
        strict_auc = roc_auc_score(strict.binary_label, strict[metric])
        strict_macro, strict_groups, strict_rows = protein_macro_auc(strict, metric)
        expected_values = {
            "comparison_rows_available": len(comparison_universe),
            "comparison_allosteric_rows_available": int(comparison_universe.binary_label.sum()),
            "comparison_orthosteric_rows_available": int(
                (comparison_universe.binary_label == 0).sum()
            ),
            "observation_AUROC_descriptive": auc,
            "within_protein_macro_AUROC_descriptive": macro,
            "within_protein_groups": groups,
            "within_protein_rows": rows,
            "no_residue_overlap_sensitivity_rows": len(strict),
            "no_residue_overlap_sensitivity_allosteric_rows": int(strict.binary_label.sum()),
            "no_residue_overlap_observation_AUROC_descriptive": strict_auc,
            "no_residue_overlap_within_protein_macro_AUROC_descriptive": strict_macro,
            "no_residue_overlap_within_protein_groups": strict_groups,
            "no_residue_overlap_within_protein_rows": strict_rows,
            "allosteric_median_A": comparison_universe.loc[
                comparison_universe.binary_label.eq(1), metric
            ].median(),
            "orthosteric_median_A": comparison_universe.loc[
                comparison_universe.binary_label.eq(0), metric
            ].median(),
        }
        for column, value in expected_values.items():
            if not close(row[column], value):
                raise RuntimeError(f"distance comparison mismatch: {metric}/{column}")

    overlap_expected = comparison_universe.loc[
        comparison_universe.reference_label.eq("allosteric")
        & comparison_universe.candidate_orthosteric_site_residue_overlap_count.gt(0),
        "observation_id",
    ]
    if len(overlap_expected) != 669 or set(overlap_expected) != set(overlap_audit.observation_id):
        raise RuntimeError("reported-distinct residue-overlap audit is incomplete")

    if len(fetch_log) != 804 or set(fetch_log.status) - {"downloaded", "reused"}:
        raise RuntimeError("targeted coordinate fetch did not complete cleanly")
    if spec.get("status") != "candidates_validated_awaiting_user_selection":
        raise RuntimeError("distance-candidate selection status changed")
    if spec.get("selected_distance") is not None:
        raise RuntimeError("a distance definition was selected before user review")
    if [item.get("metric") for item in spec.get("candidate_definitions", [])] != METRICS:
        raise RuntimeError("distance-candidate specification changed")
    if spec.get("ranking_direction") != "larger distance is more distal; no hard threshold in the ranking score":
        raise RuntimeError("distance threshold was introduced into the frozen ranking feature")
    if not build.get("self_tests", {}).get("passed") or build.get("recursive_coordinate_scan_performed"):
        raise RuntimeError("geometry self-test or bounded-I/O contract failed")
    if build.get("selected_distance") is not None or build.get("selection_status") != "awaiting_user_selection":
        raise RuntimeError("checkpoint-6 build selected a distance prematurely")

    result = {
        "checkpoint": 6,
        "status": "validated_candidates_awaiting_user_selection",
        "reference_rows": len(reference),
        "coordinate_pdbs": len(coordinates),
        "coordinate_sources": coordinates.coordinate_source.value_counts().to_dict(),
        "geometry_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "geometry_ok_fraction": float(len(ok) / len(geometry)),
        "exact_orthosteric_geometry_ok": len(orth_ok),
        "exact_allosteric_geometry_ok": int(allosteric.geometry_status.eq("ok").sum()),
        "comparison_rows": len(comparison_universe),
        "reported_distinct_allosteric_rows": int(comparison_universe.binary_label.sum()),
        "reported_distinct_with_direct_residue_overlap": len(overlap_audit),
        "no_residue_overlap_sensitivity_allosteric_rows": int(strict.binary_label.sum()),
        "selected_distance": None,
        "candidate_metrics": METRICS,
        "candidate_descriptive_results": comparison.to_dict("records"),
        "metric_selection_pending_user_review": True,
        "hard_distance_threshold_frozen_for_ranking": False,
        "coordinate_directories_recursively_scanned": False,
        "coordinate_files_content_rehashed": False,
        "coordinate_integrity_check": "manifest path/size/mtime completeness plus successful mmCIF parse during geometry build; no second file pass",
        "checkpoint5_identity_mismatches": mismatches,
        "validated_output_hashes": {
            name: record["sha256"] for name, record in build["outputs"].items()
        },
        "build_summary_sha256": sha256(BUILD),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
