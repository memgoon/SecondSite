#!/usr/bin/env python3
"""Independently validate the frozen Pfam-overlap novelty amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd


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


def canonical_accessions(series: pd.Series) -> set[str]:
    return set(series.astype(str).str.strip().str.upper().str.split("-").str[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery")
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    package = root / "analysis/property_balanced_chembl"
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    cache_path = root / config["family_novelty"]["cache_path"]
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    records = cache.get("records", {})

    property_path = root / config["property_cohort"]["path"]
    reference_path = root / (
        "analysis/chembl_external_prioritization/gpu_cache/"
        "REFERENCE_MODEL_READY.tsv.gz"
    )
    property_frame = pd.read_csv(
        property_path, sep="\t", usecols=["uniprot"], low_memory=False
    )
    reference = pd.read_csv(
        reference_path,
        sep="\t",
        usecols=["uniprot", "weak2020_label"],
        low_memory=False,
    )
    property_proteins = canonical_accessions(property_frame["uniprot"])
    reference_proteins = canonical_accessions(reference["uniprot"])
    expected_union = property_proteins | reference_proteins
    training_pfams = {
        pfam
        for accession in property_proteins
        for pfam in records.get(accession, {}).get("pfam_ids", [])
    }
    invalid_records = []
    for accession, record in records.items():
        pfams = record.get("pfam_ids", [])
        status = record.get("annotation_status")
        if pfams != sorted(set(pfams)):
            invalid_records.append(accession)
        if any(re.fullmatch(r"PF[0-9]{5}", item) is None for item in pfams):
            invalid_records.append(accession)
        if bool(pfams) != (status == "annotated"):
            invalid_records.append(accession)

    def classify(accession: str) -> str:
        record = records.get(accession)
        if not record or record.get("annotation_status") != "annotated":
            return "annotation_unavailable"
        if set(record.get("pfam_ids", [])) & training_pfams:
            return "seen"
        return "unseen"

    reference = reference.copy()
    reference["canonical_uniprot"] = (
        reference["uniprot"].astype(str).str.upper().str.split("-").str[0]
    )
    reference["pfam_overlap_status"] = reference["canonical_uniprot"].map(classify)
    reference["target_identity_seen"] = reference["canonical_uniprot"].isin(
        property_proteins
    )
    primary = reference[reference["weak2020_label"].isin([0, 1])].copy()

    def summarize(frame: pd.DataFrame) -> dict:
        row_counts = frame["pfam_overlap_status"].value_counts().to_dict()
        target_counts = (
            frame.groupby("pfam_overlap_status")["canonical_uniprot"]
            .nunique()
            .to_dict()
        )
        return {
            "rows": int(len(frame)),
            "proteins": int(frame["canonical_uniprot"].nunique()),
            "target_identity_seen_rows": int(frame["target_identity_seen"].sum()),
            "target_identity_seen_fraction": float(frame["target_identity_seen"].mean()),
            "pfam_seen_rows": int(row_counts.get("seen", 0)),
            "pfam_unseen_rows": int(row_counts.get("unseen", 0)),
            "pfam_annotation_unavailable_rows": int(
                row_counts.get("annotation_unavailable", 0)
            ),
            "pfam_seen_proteins": int(target_counts.get("seen", 0)),
            "pfam_unseen_proteins": int(target_counts.get("unseen", 0)),
            "pfam_annotation_unavailable_proteins": int(
                target_counts.get("annotation_unavailable", 0)
            ),
        }

    checks = {
        "cache_sha256_matches_config": (
            sha256_file(cache_path) == config["family_novelty"]["cache_sha256"]
        ),
        "cache_self_validation_pass": cache.get("validation", {}).get("status") == "PASS",
        "verified_tls_recorded": cache.get("retrieval", {}).get("verified_tls") is True,
        "uniprot_release_matches_config": (
            cache.get("retrieval", {}).get("uniprot_release")
            == config["family_novelty"]["uniprot_release"]
        ),
        "cache_exactly_covers_frozen_union": set(records) == expected_union,
        "property_count_matches_contract": (
            len(property_frame) == config["property_cohort"]["rows"]
            and len(property_proteins) == config["property_cohort"]["proteins"]
        ),
        "property_cohort_sha256_matches": (
            sha256_file(property_path) == config["property_cohort"]["sha256"]
        ),
        "reference_accessions_all_cached": reference_proteins <= set(records),
        "training_pfam_union_nonempty": bool(training_pfams),
        "cache_records_semantically_valid": not invalid_records,
        "unavailable_rows_not_classified_unseen": not bool(
            reference.loc[
                reference["pfam_overlap_status"].eq("annotation_unavailable"),
                "canonical_uniprot",
            ].map(
                lambda accession: bool(
                    set(records.get(accession, {}).get("pfam_ids", []))
                    & training_pfams
                )
            ).any()
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "amendment_id": config["family_novelty"]["amendment_id"],
        "checks": checks,
        "cache": {
            "path": str(cache_path.relative_to(root)),
            "sha256": sha256_file(cache_path),
            "records": len(records),
            "annotated_records": sum(
                record.get("annotation_status") == "annotated"
                for record in records.values()
            ),
            "annotation_unavailable_records": sum(
                record.get("annotation_status") != "annotated"
                for record in records.values()
            ),
            "training_pfam_ids": len(training_pfams),
            "uniprot_release": cache.get("retrieval", {}).get("uniprot_release"),
        },
        "reference_all": summarize(reference),
        "reference_two_label_primary": summarize(primary),
        "interpretation_limits": [
            "exact Pfam overlap is not general sequence-family novelty",
            "full-screen targets absent from the frozen cache are annotation-unavailable",
            "no model scores or ChEMBL labels were used to build the Pfam cache",
        ],
        "invalid_cache_records": sorted(set(invalid_records)),
    }
    if not args.no_write:
        atomic_json(package / "data/FAMILY_NOVELTY_AUDIT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
