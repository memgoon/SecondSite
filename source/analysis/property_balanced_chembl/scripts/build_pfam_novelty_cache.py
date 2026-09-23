#!/usr/bin/env python3
"""Freeze and validate UniProt Pfam annotations used for novelty strata.

The cache covers the union of the property-balanced training proteins and the
ChEMBL reference proteins.  Full-screen proteins absent from this frozen cache
must be reported as annotation-unavailable; they must never be relabelled as
family-unseen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests


UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
ACCESSION_RE = re.compile(r"^[A-Z0-9]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def canonical_accession(value: object) -> str:
    accession = str(value).strip().upper().split("-")[0]
    if not ACCESSION_RE.fullmatch(accession):
        raise ValueError("invalid UniProt accession: {!r}".format(value))
    return accession


def load_input_accessions(root: Path) -> tuple[set[str], set[str], dict]:
    property_path = root / (
        "analysis/ligand_chemistry_balancing/"
        "PROTEIN_ANCHORED_CHEMISTRY_BALANCED.tsv.gz"
    )
    reference_path = root / (
        "analysis/chembl_external_prioritization/gpu_cache/"
        "REFERENCE_MODEL_READY.tsv.gz"
    )
    property_frame = pd.read_csv(
        property_path, sep="\t", usecols=["uniprot"], low_memory=False
    )
    reference_frame = pd.read_csv(
        reference_path, sep="\t", usecols=["uniprot"], low_memory=False
    )
    property_accessions = {
        canonical_accession(value) for value in property_frame["uniprot"]
    }
    reference_accessions = {
        canonical_accession(value) for value in reference_frame["uniprot"]
    }
    provenance = {
        "property_path": str(property_path.relative_to(root)),
        "property_sha256": sha256_file(property_path),
        "property_proteins": len(property_accessions),
        "reference_path": str(reference_path.relative_to(root)),
        "reference_sha256": sha256_file(reference_path),
        "reference_proteins": len(reference_accessions),
        "intersection_proteins": len(property_accessions & reference_accessions),
        "union_proteins": len(property_accessions | reference_accessions),
    }
    return property_accessions, reference_accessions, provenance


def parse_pfam_cell(value: str) -> list[str]:
    return sorted({item.strip() for item in value.split(";") if item.strip()})


def retrieve_batch(
    session: requests.Session,
    accessions: list[str],
    timeout: int,
    retries: int,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    query = " OR ".join("accession:{}".format(item) for item in accessions)
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,xref_pfam",
        "size": str(len(accessions)),
    }
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                UNIPROT_URL, params=params, timeout=timeout, verify=True
            )
            response.raise_for_status()
            lines = response.text.splitlines()
            if not lines or lines[0].split("\t")[:2] != ["Entry", "Pfam"]:
                raise RuntimeError("unexpected UniProt TSV schema")
            records: dict[str, list[str]] = {}
            for line in lines[1:]:
                fields = line.split("\t")
                accession = canonical_accession(fields[0])
                pfam = parse_pfam_cell(fields[1] if len(fields) > 1 else "")
                if accession in records:
                    raise RuntimeError("duplicate UniProt response: {}".format(accession))
                records[accession] = pfam
            headers = {
                "release": response.headers.get("X-UniProt-Release", "unknown"),
                "release_date": response.headers.get(
                    "X-UniProt-Release-Date", "unknown"
                ),
            }
            return records, headers
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        "UniProt retrieval failed after {} attempts: {}".format(retries, last_error)
    )


def validate_payload(
    payload: dict,
    property_accessions: set[str],
    reference_accessions: set[str],
    provenance: dict,
) -> dict:
    records = payload.get("records", {})
    expected = property_accessions | reference_accessions
    observed = set(records)
    statuses = {}
    invalid_pfam = []
    for accession, record in records.items():
        status = record.get("annotation_status")
        statuses[status] = statuses.get(status, 0) + 1
        pfam_ids = record.get("pfam_ids", [])
        if pfam_ids != sorted(set(pfam_ids)):
            invalid_pfam.append(accession)
        if any(not re.fullmatch(r"PF[0-9]{5}", item) for item in pfam_ids):
            invalid_pfam.append(accession)
        if status == "annotated" and not pfam_ids:
            invalid_pfam.append(accession)
        if status != "annotated" and pfam_ids:
            invalid_pfam.append(accession)
    training_pfams = sorted({
        pfam
        for accession in property_accessions
        for pfam in records.get(accession, {}).get("pfam_ids", [])
    })
    reference_annotated = {
        accession
        for accession in reference_accessions
        if records.get(accession, {}).get("annotation_status") == "annotated"
    }
    reference_seen = {
        accession
        for accession in reference_annotated
        if set(records[accession]["pfam_ids"]) & set(training_pfams)
    }
    checks = {
        "schema_version": payload.get("schema_version") == "1.0",
        "verified_tls": payload.get("retrieval", {}).get("verified_tls") is True,
        "input_provenance": payload.get("inputs") == provenance,
        "exact_accession_set": observed == expected,
        "no_invalid_pfam_rows": not invalid_pfam,
        "property_coverage_complete": property_accessions <= observed,
        "reference_coverage_complete": reference_accessions <= observed,
        "training_pfam_nonempty": bool(training_pfams),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "property_proteins": len(property_accessions),
            "reference_proteins": len(reference_accessions),
            "union_proteins": len(expected),
            "cache_records": len(records),
            "training_pfam_ids": len(training_pfams),
            "reference_pfam_annotated_proteins": len(reference_annotated),
            "reference_pfam_seen_property_proteins": len(reference_seen),
            "reference_pfam_unseen_property_proteins": len(
                reference_annotated - reference_seen
            ),
        },
        "annotation_status_counts": statuses,
        "invalid_pfam_accessions": sorted(set(invalid_pfam)),
        "missing_accessions": sorted(expected - observed),
        "extra_accessions": sorted(observed - expected),
    }


def build(root: Path, output: Path, batch_size: int, timeout: int, retries: int) -> dict:
    property_accessions, reference_accessions, provenance = load_input_accessions(root)
    union = sorted(property_accessions | reference_accessions)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "property-balanced-chembl-pfam-cache/1.0"
    })
    retrieved: dict[str, list[str]] = {}
    release_headers = set()
    request_count = 0
    for start in range(0, len(union), batch_size):
        batch = union[start:start + batch_size]
        rows, headers = retrieve_batch(session, batch, timeout, retries)
        duplicate = set(retrieved) & set(rows)
        if duplicate:
            raise RuntimeError("duplicate cross-batch responses: {}".format(sorted(duplicate)))
        retrieved.update(rows)
        release_headers.add((headers["release"], headers["release_date"]))
        request_count += 1
        print(
            "UniProt Pfam {}/{} accessions".format(min(start + len(batch), len(union)), len(union)),
            flush=True,
        )
    if len(release_headers) != 1:
        raise RuntimeError("UniProt release changed during retrieval: {}".format(release_headers))
    release, release_date = next(iter(release_headers))
    records = {}
    for accession in union:
        pfam_ids = retrieved.get(accession)
        if pfam_ids is None:
            status = "unresolved_accession"
            pfam_ids = []
        elif pfam_ids:
            status = "annotated"
        else:
            status = "no_pfam_annotation"
        scopes = []
        if accession in property_accessions:
            scopes.append("property_training")
        if accession in reference_accessions:
            scopes.append("chembl_reference")
        records[accession] = {
            "annotation_status": status,
            "input_scopes": scopes,
            "pfam_ids": pfam_ids,
        }
    payload = {
        "schema_version": "1.0",
        "created_date": date.today().isoformat(),
        "definition": (
            "family_seen_property means at least one exact UniProt Pfam accession "
            "overlaps the Pfam union of the 426 property-training proteins"
        ),
        "inputs": provenance,
        "retrieval": {
            "endpoint": UNIPROT_URL,
            "fields": ["accession", "xref_pfam"],
            "query_batches": request_count,
            "batch_size": batch_size,
            "verified_tls": True,
            "uniprot_release": release,
            "uniprot_release_date": release_date,
        },
        "records": records,
    }
    validation = validate_payload(
        payload, property_accessions, reference_accessions, provenance
    )
    payload["validation"] = validation
    if validation["status"] != "PASS":
        raise RuntimeError("Pfam cache validation failed: {}".format(validation))
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/property_balanced_chembl/data/UNIPROT_PFAM_CACHE.json"
        ),
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if args.build == args.validate_only:
        raise SystemExit("choose exactly one of --build or --validate-only")
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.build:
        payload = build(root, output, args.batch_size, args.timeout, args.retries)
        print(json.dumps(payload["validation"], indent=2, sort_keys=True))
        print("cache_sha256={}".format(sha256_file(output)))
        return
    property_accessions, reference_accessions, provenance = load_input_accessions(root)
    payload = json.loads(output.read_text(encoding="utf-8"))
    validation = validate_payload(
        payload, property_accessions, reference_accessions, provenance
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    print("cache_sha256={}".format(sha256_file(output)))
    if validation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
