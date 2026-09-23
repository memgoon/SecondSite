#!/usr/bin/env python3
"""Independent invariants and checksum validation for K03 BRENDA outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "analysis/kinetic_allostery_model_v1"
DATA = PROJECT / "data/k03_brenda"
MANIFEST = PROJECT / "manifests/K03_MANIFEST.tsv"
CHECKSUMS = PROJECT / "checksums/K03_CHECKSUMS.sha256"
STATUS = PROJECT / "work/status/K03__BRENDA_PARAMETERS.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_checksums():
    checked = 0
    with CHECKSUMS.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            expected, shown_path = line.split("  ", 1)
            path = full_path(shown_path)
            require(path.exists(), "checksum path missing: {}".format(path))
            observed = sha256_file(path)
            require(observed == expected, "checksum mismatch: {}".format(path))
            checked += 1
    return checked


def finalize_manifest_with_validator():
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    validator = Path(__file__).resolve()
    shown = str(validator.relative_to(ROOT))
    validator_row = {
        "role": "validator",
        "path": shown,
        "bytes": str(validator.stat().st_size),
        "sha256": sha256_file(validator),
        "row_count": "NA",
        "note": "independent K03 output validator",
    }
    replaced = False
    for index, row in enumerate(rows):
        path = full_path(row["path"])
        if row["path"] == shown or row["role"] == "validator":
            rows[index] = validator_row
            replaced = True
        elif path.resolve() == (PROJECT / "scripts/extract_k03_brenda_parameters.py").resolve():
            row["bytes"] = str(path.stat().st_size)
            row["sha256"] = sha256_file(path)
    if not replaced:
        rows.append(validator_row)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with CHECKSUMS.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write("{}  {}\n".format(row["sha256"], row["path"]))
        handle.write("{}  {}\n".format(sha256_file(MANIFEST), MANIFEST.relative_to(ROOT)))


def validate_content():
    with STATUS.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    require(status["status"] == "complete", "K03 status is not complete")
    require(status["source_release"] == "2026.1", "wrong source release")
    expected_counts = status["counts"]

    protein_ids = set()
    protein_count = 0
    for row in read_rows(DATA / "K03_BRENDA_PROTEIN_MAPPINGS.tsv"):
        protein_count += 1
        key = row["protein_mapping_id"]
        require(key not in protein_ids, "duplicate protein_mapping_id: {}".format(key))
        protein_ids.add(key)
        require(row["mapping_status"], "blank protein mapping status")
    require(protein_count == expected_counts["proteins"], "protein row count mismatch")

    compound_ids = set()
    compound_count = 0
    for row in read_rows(DATA / "K03_BRENDA_COMPOUND_MAPPINGS.tsv"):
        compound_count += 1
        key = row["compound_mapping_id"]
        require(key not in compound_ids, "duplicate compound_mapping_id: {}".format(key))
        compound_ids.add(key)
        require(row["mapping_reason"], "blank compound mapping reason")
    require(compound_count == expected_counts["compounds"], "compound row count mismatch")

    publication_ids = set()
    reference_count = 0
    for row in read_rows(DATA / "K03_BRENDA_REFERENCES.tsv"):
        reference_count += 1
        key = row["source_publication_id"]
        require(key not in publication_ids, "duplicate publication id: {}".format(key))
        publication_ids.add(key)
    require(reference_count == expected_counts["references"], "reference row count mismatch")

    source_ids = set()
    source_fields = Counter()
    ledger_count = 0
    for row in read_rows(DATA / "K03_BRENDA_SOURCE_RECORD_LEDGER.tsv"):
        ledger_count += 1
        key = row["source_record_id"]
        require(key not in source_ids, "duplicate source_record_id: {}".format(key))
        source_ids.add(key)
        source_fields[row["source_value_field"]] += 1
        require(row["reason_codes"], "blank ledger reason codes")
        require(row["normalization_disposition"] == "retained_with_quality_tier", "lost source record")
    require(ledger_count == expected_counts["ledger"], "ledger row count mismatch")
    require(dict(source_fields) == status["source_field_counts"], "source field accounting mismatch")

    experiment_ids = set()
    experiment_condition = {}
    experiment_count = 0
    protein_mapping_counts = Counter()
    for row in read_rows(DATA / "K03_BRENDA_EXPERIMENTS.tsv"):
        experiment_count += 1
        key = row["experiment_id"]
        require(key not in experiment_ids, "duplicate experiment_id: {}".format(key))
        experiment_ids.add(key)
        require(row["source_record_id"] in source_ids, "experiment source-record FK failure")
        require(row["protein_mapping_id"] in protein_ids, "experiment protein FK failure")
        experiment_condition[key] = row["condition_id"]
        protein_mapping_counts[row["protein_mapping_status"]] += 1
        for publication_id in row["source_publication_ids"].split(";"):
            if not publication_id.startswith("NA_"):
                require(publication_id in publication_ids, "experiment publication FK failure")
    require(experiment_count == expected_counts["experiments"], "experiment count mismatch")
    require(
        dict(protein_mapping_counts) == status["protein_mapping_status_counts"],
        "protein mapping status accounting mismatch",
    )

    condition_count = 0
    seen_condition_experiments = set()
    condition_completeness = Counter()
    for row in read_rows(DATA / "K03_BRENDA_CONDITIONS.tsv"):
        condition_count += 1
        experiment_id = row["experiment_id"]
        require(experiment_id in experiment_ids, "condition experiment FK failure")
        require(experiment_id not in seen_condition_experiments, "multiple condition rows per experiment")
        seen_condition_experiments.add(experiment_id)
        require(experiment_condition[experiment_id] == row["condition_id"], "condition_id mismatch")
        require(row["raw_condition_text"], "blank raw condition text")
        condition_completeness[row["condition_completeness"]] += 1
    require(condition_count == expected_counts["conditions"], "condition count mismatch")
    require(seen_condition_experiments == experiment_ids, "experiment/condition coverage mismatch")
    require(
        dict(condition_completeness) == status["condition_completeness_counts"],
        "condition completeness accounting mismatch",
    )

    participant_ids = set()
    participant_count = 0
    compound_mapping_counts = Counter()
    for row in read_rows(DATA / "K03_BRENDA_PARTICIPANTS.tsv"):
        participant_count += 1
        key = row["participant_id"]
        require(key not in participant_ids, "duplicate participant_id: {}".format(key))
        participant_ids.add(key)
        require(row["source_record_id"] in source_ids, "participant source-record FK failure")
        require(row["compound_mapping_id"] in compound_ids, "participant compound FK failure")
        require(row["mapping_reason"], "blank participant mapping reason")
        compound_mapping_counts[row["mapping_status"]] += 1
    require(participant_count == expected_counts["participants"], "participant count mismatch")
    require(
        dict(compound_mapping_counts) == status["compound_mapping_status_counts"],
        "compound mapping status accounting mismatch",
    )

    parameter_ids = set()
    parameter_count = 0
    parameter_types = Counter()
    quality_tiers = Counter()
    high_quality_mapping_violations = 0
    secondary_quality_violations = 0
    participant_fk_failures = 0
    for row in read_rows(DATA / "K03_BRENDA_PARAMETERS.tsv"):
        parameter_count += 1
        key = row["parameter_id"]
        require(key not in parameter_ids, "duplicate parameter_id: {}".format(key))
        parameter_ids.add(key)
        require(row["source_record_id"] in source_ids, "parameter source-record FK failure")
        require(row["experiment_id"] in experiment_ids, "parameter experiment FK failure")
        require(row["protein_mapping_id"] in protein_ids, "parameter protein FK failure")
        require(row["quality_reason_codes"], "blank parameter quality reason")
        require(row["raw_source_locator"], "blank parameter source locator")
        require(row["mechanism_label"] == "unresolved", "mechanism inferred from parameter alone")
        for participant_id in row["participant_ids"].split(";"):
            if participant_id and participant_id not in participant_ids:
                participant_fk_failures += 1
        parameter_types[row["parameter_type"]] += 1
        quality_tiers[row["quality_tier"]] += 1
        if row["quality_tier"] in {"Q1", "Q2"}:
            if (
                row["protein_mapping_status"] != "exact_single_uniprot"
                or row["compound_mapping_status"] != "exact_unique_full_inchikey"
                or row["normalized_unit"].startswith("NA_")
                or row["numeric_parse_status"]
                not in {"parsed_single_value", "parsed_value_with_uncertainty", "parsed_range"}
            ):
                high_quality_mapping_violations += 1
        if row["parameter_type"] in {"S0.5", "nH"}:
            if (
                row["parameter_origin"] != "comment_explicit_secondary"
                or row["quality_tier"] in {"Q1", "Q2"}
            ):
                secondary_quality_violations += 1
    require(parameter_count == expected_counts["parameters"], "parameter count mismatch")
    require(dict(parameter_types) == status["parameter_type_counts"], "parameter type count mismatch")
    require(dict(quality_tiers) == status["quality_tier_counts"], "quality count mismatch")
    require(high_quality_mapping_violations == 0, "ambiguous mappings entered Q1/Q2")
    require(secondary_quality_violations == 0, "S0.5/nH quality cap violation")
    require(participant_fk_failures == 0, "parameter participant FK failures")

    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    manifest_counts = {
        Path(row["path"]).name: int(row["row_count"])
        for row in manifest_rows
        if row["row_count"] != "NA"
    }
    actual_by_name = {
        "K03_BRENDA_SOURCE_RECORD_LEDGER.tsv": ledger_count,
        "K03_BRENDA_EXPERIMENTS.tsv": experiment_count,
        "K03_BRENDA_PARAMETERS.tsv": parameter_count,
        "K03_BRENDA_PARTICIPANTS.tsv": participant_count,
        "K03_BRENDA_CONDITIONS.tsv": condition_count,
        "K03_BRENDA_REFERENCES.tsv": reference_count,
        "K03_BRENDA_PROTEIN_MAPPINGS.tsv": protein_count,
        "K03_BRENDA_COMPOUND_MAPPINGS.tsv": compound_count,
    }
    for name, count in actual_by_name.items():
        require(manifest_counts.get(name) == count, "manifest row count mismatch: {}".format(name))

    return {
        "status": "passed",
        "source_records": ledger_count,
        "experiments": experiment_count,
        "parameters": parameter_count,
        "participants": participant_count,
        "conditions": condition_count,
        "proteins": protein_count,
        "compounds": compound_count,
        "references": reference_count,
        "high_quality_mapping_violations": high_quality_mapping_violations,
        "secondary_quality_violations": secondary_quality_violations,
        "participant_fk_failures": participant_fk_failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finalize-manifest",
        action="store_true",
        help="Add/update this validator in the K03 manifest and regenerate checksums.",
    )
    args = parser.parse_args()
    result = validate_content()
    if args.finalize_manifest:
        finalize_manifest_with_validator()
    result["checksums_verified"] = validate_checksums()
    result["manifest_finalized"] = bool(args.finalize_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
