#!/usr/bin/env python3
"""Prepare deterministic protein- and Pfam-family-held-out checkpoint-8 folds."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"
SCRIPTS = PACKAGE / "scripts"

FEATURES = DATA / "CHECKPOINT7_REFERENCE_FEATURES.tsv.gz"
METHOD_VALIDATION = VALIDATION / "CHECKPOINT7_METHOD_VALIDATION.json"
METHOD_SPEC = MANIFESTS / "CHECKPOINT7_METHOD_SPEC.json"
METHOD_SCRIPT = SCRIPTS / "07b_compare_likelihood_estimators.py"

PFAM_CACHE = DATA / "CHECKPOINT8_UNIPROT_PFAM_CACHE.json"
PROTEIN_ASSIGNMENTS = DATA / "CHECKPOINT8_PROTEIN_FAMILY_ASSIGNMENTS.tsv"
OBSERVATION_FOLDS = DATA / "CHECKPOINT8_OBSERVATION_FOLDS.tsv.gz"
FOLD_AUDIT = DATA / "CHECKPOINT8_FOLD_AUDIT.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT8_SPLIT_INPUT_HASHES.tsv"
SPLIT_SPEC = MANIFESTS / "CHECKPOINT8_SPLIT_SPEC.json"
BUILD = VALIDATION / "CHECKPOINT8_SPLIT_BUILD_SUMMARY.json"

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
ACCESSION_RE = re.compile(r"^[A-Z0-9]+$")
PFAM_RE = re.compile(r"^PF[0-9]{5}$")
N_FOLDS = 5
BATCH_SIZE = 50
TIMEOUT_SECONDS = 60
RETRIES = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, values: Sequence[str]) -> str:
    payload = ";".join(values).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:20]


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_tsv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def canonical_accession(value: object) -> str:
    accession = str(value).strip().upper().split("-")[0]
    if not ACCESSION_RE.fullmatch(accession):
        raise ValueError(f"invalid UniProt accession: {value!r}")
    return accession


def parse_pfam(value: str) -> List[str]:
    return sorted({item.strip() for item in value.split(";") if item.strip()})


def request_query(session: requests.Session, query: str):
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,xref_pfam",
        "size": "500",
    }
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(
                UNIPROT_URL,
                params=params,
                timeout=TIMEOUT_SECONDS,
                verify=True,
            )
            response.raise_for_status()
            lines = response.text.splitlines()
            if not lines or lines[0].split("\t")[:2] != ["Entry", "Pfam"]:
                raise RuntimeError("unexpected UniProt TSV schema")
            rows = []
            for line in lines[1:]:
                fields = line.split("\t")
                entry = canonical_accession(fields[0])
                pfam_ids = parse_pfam(fields[1] if len(fields) > 1 else "")
                if any(not PFAM_RE.fullmatch(value) for value in pfam_ids):
                    raise RuntimeError(f"invalid Pfam identifier returned for {entry}")
                rows.append((entry, pfam_ids))
            headers = (
                response.headers.get("X-UniProt-Release", "unknown"),
                response.headers.get("X-UniProt-Release-Date", "unknown"),
            )
            return rows, headers
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"UniProt request failed after {RETRIES} attempts: {last_error}")


def validate_cache(cache: dict, accessions: Sequence[str]) -> None:
    if cache.get("schema_version") != "1.0":
        raise RuntimeError("checkpoint-8 Pfam cache schema changed")
    if cache.get("requested_accessions") != list(accessions):
        raise RuntimeError("checkpoint-8 Pfam cache accession universe changed")
    records = cache.get("records", {})
    if set(records) != set(accessions):
        raise RuntimeError("checkpoint-8 Pfam cache coverage changed")
    for accession, record in records.items():
        pfam_ids = record.get("pfam_ids", [])
        if pfam_ids != sorted(set(pfam_ids)):
            raise RuntimeError(f"duplicate or unsorted Pfam IDs: {accession}")
        if any(not PFAM_RE.fullmatch(value) for value in pfam_ids):
            raise RuntimeError(f"invalid Pfam ID in cache: {accession}")
        status = record.get("annotation_status")
        if status == "annotated" and not pfam_ids:
            raise RuntimeError(f"annotated accession has no Pfam ID: {accession}")
        if status != "annotated" and pfam_ids:
            raise RuntimeError(f"unavailable accession has Pfam IDs: {accession}")
    if cache.get("retrieval", {}).get("verified_tls") is not True:
        raise RuntimeError("checkpoint-8 Pfam cache was not retrieved with TLS verification")


def build_cache(accessions: Sequence[str]) -> dict:
    if PFAM_CACHE.is_file():
        cache = json.loads(PFAM_CACHE.read_text(encoding="utf-8"))
        validate_cache(cache, accessions)
        return cache

    session = requests.Session()
    session.headers.update({"User-Agent": "secondsite-biolip-checkpoint8/1.0"})
    exact_records: Dict[str, List[str]] = {}
    release_headers = set()
    batch_requests = 0
    for start in range(0, len(accessions), BATCH_SIZE):
        batch = list(accessions[start:start + BATCH_SIZE])
        query = " OR ".join(f"accession:{accession}" for accession in batch)
        rows, headers = request_query(session, query)
        release_headers.add(headers)
        for entry, pfam_ids in rows:
            if entry in exact_records:
                raise RuntimeError(f"duplicate UniProt batch response: {entry}")
            exact_records[entry] = pfam_ids
        batch_requests += 1
        print(f"UniProt Pfam batch {min(start + len(batch), len(accessions))}/{len(accessions)}", flush=True)

    records = {}
    individual_requests = 0
    for accession in accessions:
        if accession in exact_records:
            resolved = accession
            pfam_ids = exact_records[accession]
            resolution = "exact_primary_accession"
        else:
            rows, headers = request_query(session, f"accession:{accession}")
            release_headers.add(headers)
            individual_requests += 1
            if len(rows) == 1:
                resolved, pfam_ids = rows[0]
                resolution = "resolved_from_secondary_or_obsolete_accession"
            elif len(rows) == 0:
                resolved, pfam_ids = "", []
                resolution = "unresolved_accession"
            else:
                resolved, pfam_ids = "", []
                resolution = "ambiguous_accession_response"
        if pfam_ids:
            annotation_status = "annotated"
        elif resolution in {"exact_primary_accession", "resolved_from_secondary_or_obsolete_accession"}:
            annotation_status = "no_pfam_annotation"
        else:
            annotation_status = resolution
        records[accession] = {
            "resolved_entry_accession": resolved,
            "accession_resolution": resolution,
            "annotation_status": annotation_status,
            "pfam_ids": pfam_ids,
        }

    if len(release_headers) != 1:
        raise RuntimeError(f"UniProt release changed during checkpoint-8 retrieval: {release_headers}")
    release, release_date = next(iter(release_headers))
    cache = {
        "schema_version": "1.0",
        "created_date": date.today().isoformat(),
        "requested_accessions": list(accessions),
        "retrieval": {
            "endpoint": UNIPROT_URL,
            "fields": ["accession", "xref_pfam"],
            "batch_size": BATCH_SIZE,
            "batch_requests": batch_requests,
            "individual_resolution_requests": individual_requests,
            "verified_tls": True,
            "uniprot_release": release,
            "uniprot_release_date": release_date,
        },
        "records": records,
    }
    validate_cache(cache, accessions)
    atomic_json(PFAM_CACHE, cache)
    return cache


class DisjointSet:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first == root_second:
            return
        if root_first < root_second:
            self.parent[root_second] = root_first
        else:
            self.parent[root_first] = root_second


def build_family_components(cache: dict):
    annotated = sorted(
        accession
        for accession, record in cache["records"].items()
        if record["annotation_status"] == "annotated"
    )
    disjoint = DisjointSet(annotated)
    by_pfam: Dict[str, List[str]] = {}
    for accession in annotated:
        for pfam_id in cache["records"][accession]["pfam_ids"]:
            by_pfam.setdefault(pfam_id, []).append(accession)
    for members in by_pfam.values():
        first = members[0]
        for other in members[1:]:
            disjoint.union(first, other)
    groups: Dict[str, List[str]] = {}
    for accession in annotated:
        groups.setdefault(disjoint.find(accession), []).append(accession)
    protein_to_component = {}
    component_rows = []
    for proteins in sorted((sorted(value) for value in groups.values()), key=lambda x: tuple(x)):
        pfam_ids = sorted({
            pfam_id
            for accession in proteins
            for pfam_id in cache["records"][accession]["pfam_ids"]
        })
        component_id = stable_id("BIOPFAM_", proteins + ["|"] + pfam_ids)
        for accession in proteins:
            protein_to_component[accession] = component_id
        component_rows.append({
            "family_component_id": component_id,
            "protein_ids": ";".join(proteins),
            "pfam_ids": ";".join(pfam_ids),
            "n_proteins": len(proteins),
            "n_pfam_ids": len(pfam_ids),
        })
    return protein_to_component, pd.DataFrame(component_rows)


def assignment_objective(counts: np.ndarray, group_counts: np.ndarray, targets: np.ndarray) -> float:
    safe_targets = np.maximum(targets, 1.0)
    normalized = (counts - targets[None, :]) / safe_targets[None, :]
    group_target = max(float(group_counts.sum()) / N_FOLDS, 1.0)
    group_term = (group_counts - group_target) / group_target
    return float(np.sum(normalized ** 2) + 0.02 * np.sum(group_term ** 2))


def assign_groups(stats: pd.DataFrame, identifier: str) -> Dict[str, int]:
    required = {identifier, "rows", "allosteric_rows", "orthosteric_rows"}
    if not required <= set(stats.columns) or stats[identifier].duplicated().any():
        raise RuntimeError(f"invalid fold-group statistics: {identifier}")
    ordered = stats.copy()
    ordered["largest_class"] = ordered[["allosteric_rows", "orthosteric_rows"]].max(axis=1)
    ordered["tie_hash"] = ordered[identifier].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    )
    ordered = ordered.sort_values(
        ["rows", "largest_class", "tie_hash"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    totals = stats[["rows", "allosteric_rows", "orthosteric_rows"]].sum().to_numpy(dtype=float)
    targets = totals / N_FOLDS
    fold_counts = np.zeros((N_FOLDS, 3), dtype=float)
    fold_group_counts = np.zeros(N_FOLDS, dtype=float)
    assignments: Dict[str, int] = {}
    for row in ordered.itertuples(index=False):
        vector = np.asarray([row.rows, row.allosteric_rows, row.orthosteric_rows], dtype=float)
        candidates = []
        for fold in range(N_FOLDS):
            trial_counts = fold_counts.copy()
            trial_groups = fold_group_counts.copy()
            trial_counts[fold] += vector
            trial_groups[fold] += 1
            candidates.append((
                assignment_objective(trial_counts, trial_groups, targets),
                fold_counts[fold, 0],
                fold_group_counts[fold],
                fold,
            ))
        chosen = min(candidates)[-1]
        assignments[str(getattr(row, identifier))] = int(chosen)
        fold_counts[chosen] += vector
        fold_group_counts[chosen] += 1
    if set(assignments) != set(stats[identifier].astype(str)):
        raise RuntimeError(f"incomplete deterministic assignment: {identifier}")
    return assignments


def main() -> None:
    start = time.time()
    method_validation = json.loads(METHOD_VALIDATION.read_text(encoding="utf-8"))
    if method_validation.get("status") != "validated_likelihood_estimators_awaiting_user_review":
        raise RuntimeError("checkpoint-7 method comparison is not validated")
    frame = pd.read_csv(FEATURES, sep="\t", low_memory=False)
    if len(frame) != 5798 or frame.observation_id.duplicated().any():
        raise RuntimeError("checkpoint-8 reference universe changed")
    accessions = sorted({canonical_accession(value) for value in frame.uniprot})
    if len(accessions) != 370:
        raise RuntimeError("checkpoint-8 protein universe changed")

    cache = build_cache(accessions)
    protein_to_component, components = build_family_components(cache)
    annotated = set(protein_to_component)

    protein_stats = frame.groupby("uniprot", sort=True).binary_label.agg(
        rows="size", allosteric_rows="sum"
    ).reset_index()
    protein_stats["orthosteric_rows"] = protein_stats.rows - protein_stats.allosteric_rows
    protein_folds = assign_groups(protein_stats, "uniprot")

    family_frame = frame[frame.uniprot.isin(annotated)].copy()
    family_frame["family_component_id"] = family_frame.uniprot.map(protein_to_component)
    family_stats = family_frame.groupby("family_component_id", sort=True).binary_label.agg(
        rows="size", allosteric_rows="sum"
    ).reset_index()
    family_stats["orthosteric_rows"] = family_stats.rows - family_stats.allosteric_rows
    family_folds = assign_groups(family_stats, "family_component_id")

    protein_rows = []
    protein_counts = frame.groupby("uniprot").binary_label.agg(
        reference_rows="size", allosteric_rows="sum"
    )
    protein_counts["orthosteric_rows"] = protein_counts.reference_rows - protein_counts.allosteric_rows
    for accession in accessions:
        record = cache["records"][accession]
        component_id = protein_to_component.get(accession, "")
        counts = protein_counts.loc[accession]
        protein_rows.append({
            "uniprot": accession,
            "resolved_entry_accession": record["resolved_entry_accession"],
            "accession_resolution": record["accession_resolution"],
            "pfam_annotation_status": record["annotation_status"],
            "pfam_ids": ";".join(record["pfam_ids"]),
            "family_component_id": component_id,
            "protein_fold": protein_folds[accession],
            "family_fold": family_folds.get(component_id, pd.NA),
            "reference_rows": int(counts.reference_rows),
            "allosteric_rows": int(counts.allosteric_rows),
            "orthosteric_rows": int(counts.orthosteric_rows),
        })
    protein_assignments = pd.DataFrame(protein_rows)
    protein_assignments["family_fold"] = protein_assignments.family_fold.astype("Int64")

    observation_folds = frame[[
        "observation_id", "site_ligand_signature_id", "uniprot", "binary_label"
    ]].merge(
        protein_assignments[[
            "uniprot", "pfam_annotation_status", "family_component_id", "protein_fold", "family_fold"
        ]],
        on="uniprot",
        how="left",
        validate="many_to_one",
    )
    if observation_folds.protein_fold.isna().any():
        raise RuntimeError("protein fold assignment is incomplete")

    audit_rows = []
    for regime, fold_column, eligible in (
        ("protein_held_out", "protein_fold", observation_folds.protein_fold.notna()),
        ("family_held_out", "family_fold", observation_folds.family_fold.notna()),
    ):
        universe = observation_folds[eligible].copy()
        for fold in range(N_FOLDS):
            test = universe[universe[fold_column].astype(int) == fold]
            train = universe[universe[fold_column].astype(int) != fold]
            audit_rows.append({
                "regime": regime,
                "fold": fold,
                "universe_rows": len(universe),
                "train_rows": len(train),
                "test_rows": len(test),
                "test_allosteric_rows": int(test.binary_label.sum()),
                "test_orthosteric_rows": int((test.binary_label == 0).sum()),
                "test_proteins": int(test.uniprot.nunique()),
                "test_both_label_proteins": int(sum(
                    group.binary_label.nunique() == 2
                    for _key, group in test.groupby("uniprot")
                )),
                "test_family_components": int(test.family_component_id.replace("", np.nan).nunique()),
                "train_test_protein_overlap": len(set(train.uniprot) & set(test.uniprot)),
                "train_test_family_component_overlap": len(
                    set(train.family_component_id.dropna()) & set(test.family_component_id.dropna())
                ) if regime == "family_held_out" else pd.NA,
            })
    audit = pd.DataFrame(audit_rows)
    if (audit.test_allosteric_rows == 0).any() or (audit.test_orthosteric_rows == 0).any():
        raise RuntimeError("a checkpoint-8 test fold has one label")
    if (audit.train_test_protein_overlap != 0).any():
        raise RuntimeError("protein leakage detected in checkpoint-8 folds")
    family_audit = audit[audit.regime == "family_held_out"]
    if (family_audit.train_test_family_component_overlap.fillna(0).astype(int) != 0).any():
        raise RuntimeError("Pfam-family leakage detected in checkpoint-8 folds")

    atomic_tsv(protein_assignments, PROTEIN_ASSIGNMENTS)
    atomic_tsv(observation_folds, OBSERVATION_FOLDS, compression="gzip")
    atomic_tsv(audit, FOLD_AUDIT)

    input_rows = []
    for asset_id, path in (
        ("checkpoint7_reference_features", FEATURES),
        ("checkpoint7_method_validation", METHOD_VALIDATION),
        ("checkpoint7_method_spec", METHOD_SPEC),
        ("checkpoint7_method_script", METHOD_SCRIPT),
    ):
        input_rows.append({
            "asset_id": asset_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    atomic_tsv(pd.DataFrame(input_rows), INPUT_HASHES)

    status_counts = protein_assignments.pfam_annotation_status.value_counts().to_dict()
    largest_component = int(components.n_proteins.max()) if len(components) else 0
    spec = {
        "checkpoint": 8,
        "stage": "heldout_split_preparation",
        "status": "splits_frozen_pending_independent_validation",
        "reference_rows": len(frame),
        "reference_proteins": len(accessions),
        "folds": N_FOLDS,
        "protein_held_out": {
            "unit": "exact UniProt accession",
            "eligible_rows": len(frame),
            "eligible_proteins": len(accessions),
        },
        "family_held_out": {
            "unit": "connected component formed by exact shared Pfam accessions",
            "annotation_unavailable_policy": "excluded from family-held-out claims; retained in protein-held-out evaluation",
            "eligible_rows": len(family_frame),
            "eligible_proteins": len(annotated),
            "components": len(components),
            "largest_component_proteins": largest_component,
        },
        "fold_assignment": {
            "algorithm": "deterministic descending-size greedy assignment",
            "balanced_fields": ["rows", "allosteric_rows", "orthosteric_rows"],
            "model_scores_used": False,
        },
        "pfam_annotation_status_counts": status_counts,
        "recursive_directory_scan_performed": False,
    }
    atomic_json(SPLIT_SPEC, spec)

    outputs = {
        "pfam_cache": PFAM_CACHE,
        "protein_assignments": PROTEIN_ASSIGNMENTS,
        "observation_folds": OBSERVATION_FOLDS,
        "fold_audit": FOLD_AUDIT,
        "input_hashes": INPUT_HASHES,
        "split_spec": SPLIT_SPEC,
    }
    build = {
        "checkpoint": 8,
        "stage": "heldout_split_preparation",
        "status": "complete_pending_independent_validation",
        "elapsed_seconds": round(time.time() - start, 3),
        "reference_rows": len(frame),
        "reference_proteins": len(accessions),
        "pfam_annotation_status_counts": status_counts,
        "family_eligible_rows": len(family_frame),
        "family_eligible_proteins": len(annotated),
        "family_components": len(components),
        "largest_family_component_proteins": largest_component,
        "recursive_directory_scan_performed": False,
        "outputs": {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in outputs.items()
        },
    }
    atomic_json(BUILD, build)
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
