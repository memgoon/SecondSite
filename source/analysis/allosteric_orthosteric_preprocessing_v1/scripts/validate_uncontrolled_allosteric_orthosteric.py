#!/usr/bin/env python3
"""Independent validation for the two-class Uncontrolled preprocessing package.

This validator does not import the production builder.  It reconstructs source
pair sets from the frozen inputs and checks generated tables, lineage,
conflicts, structure dispositions, label vocabulary, and optional checksums.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASD = REPO / "analysis/asd_as_clean.csv"
WWPDB = Path("/shared_data/11.HS_allostery/Data/wwpdb_compounds.pickle")
HARD = Path("/shared_data/11.HS_allostery/18.Structure_known/4.Evaluate_Orthosteric_LowScore_Enrichment/hard_orthosteric_reference_pairs_loaded.tsv")
GTOP_DIR = Path("/shared_data/11.HS_allostery/Data/Orthosteric_References/GtoP")
K03_PARAMETERS = REPO / "analysis/kinetic_allostery_model_v1/data/k03_brenda/K03_BRENDA_PARAMETERS.tsv"
K03_MANIFEST = REPO / "analysis/kinetic_allostery_model_v1/manifests/K03_MANIFEST.tsv"
IK_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
UP_RE = re.compile(r"^[A-Z0-9]{6,10}(?:-[0-9]+)?$")
NA = "NA"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_up(value: str) -> str:
    value = (value or "").strip().upper()
    return value.split("-")[0] if UP_RE.fullmatch(value) else ""


def valid_ik(value: str) -> bool:
    return bool(IK_RE.fullmatch((value or "").strip().upper()))


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def recomputed_key(smiles: str) -> str:
    if not smiles or smiles == NA:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol else ""


def independent_asd_pairs() -> tuple[set[tuple[str, str]], Counter]:
    wwpdb = pickle.load(WWPDB.open("rb"))
    pairs: set[tuple[str, str]] = set()
    counts = Counter()
    with ASD.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts["raw"] += 1
            up = canonical_up(row.get("uniprot", ""))
            if not up:
                counts["excluded_uniprot"] += 1
                continue
            ccd = row.get("ccd", "").strip().upper()
            rec = wwpdb.get(ccd, {}) if ccd else {}
            desc = rec.get("descriptors", {}) if isinstance(rec, dict) else {}
            ik = str(desc.get("INCHIKEY") or "").strip().upper()
            if not valid_ik(ik):
                counts["excluded_inchikey"] += 1
                continue
            smiles = str(desc.get("SMILES_CANONICAL") or desc.get("SMILES") or "").strip()
            got = recomputed_key(smiles)
            if got and got[:14] != ik[:14]:
                counts["excluded_connectivity_conflict"] += 1
                continue
            if got == ik:
                counts["full_match"] += 1
            elif got:
                counts["stereo_only"] += 1
            else:
                counts["unparseable"] += 1
            pairs.add((up, ik))
    return pairs, counts


def read_gtop_source(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        next(handle)
        return list(csv.DictReader(handle, delimiter="\t"))


def split_tokens(value: str) -> list[str]:
    return [x.strip() for x in re.split(r"[;,| ]+", value or "") if x.strip()]


def independent_gtop_pairs() -> tuple[set[tuple[str, str]], dict]:
    chemistry: dict[str, str] = {}
    with HARD.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["Source"].startswith("GtoP") and valid_ik(row["InChIKey"]):
                lid, ik = row["Ligand_ID"].strip(), row["InChIKey"].strip().upper()
                if lid in chemistry and chemistry[lid] != ik:
                    raise AssertionError(f"nonunique GtoP chemistry for {lid}")
                chemistry[lid] = ik
    pairs: set[tuple[str, str]] = set()
    evidence_keys: set[tuple[str, int, str, str]] = set()
    scope_rows = 0
    pairings = read_gtop_source(GTOP_DIR / "endogenous_ligand_pairings_all.tsv")
    for source_row, row in enumerate(pairings, start=3):
        ik = chemistry.get(row.get("Ligand ID", "").strip())
        if not ik:
            continue
        direct = split_tokens(row.get("Target UniProt ID", ""))
        if not direct:
            scope_rows += 1
            continue
        for raw in direct:
            up = canonical_up(raw)
            if not up:
                raise AssertionError(f"invalid direct GtoP pairing accession: {raw}")
            pairs.add((up, ik))
            evidence_keys.add(("GtoP_endogenous_pairing", source_row, up, row["Ligand ID"].strip()))
    interactions = read_gtop_source(GTOP_DIR / "interactions_all.tsv")
    for source_row, row in enumerate(interactions, start=3):
        if row.get("Endogenous", "").strip().strip('"').lower() not in {"true", "1", "yes", "y", "t"}:
            continue
        ik = chemistry.get(row.get("Ligand ID", "").strip())
        if not ik:
            continue
        direct = split_tokens(row.get("Target UniProt ID", ""))
        if not direct:
            scope_rows += 1
            continue
        for raw in direct:
            up = canonical_up(raw)
            if not up:
                raise AssertionError(f"invalid direct GtoP interaction accession: {raw}")
            pairs.add((up, ik))
            evidence_keys.add(("GtoP_endogenous_interaction", source_row, up, row["Ligand ID"].strip()))
    return pairs, {"evidence_keys": evidence_keys, "scope_rows": scope_rows}


def independent_klifs_pairs() -> tuple[set[tuple[str, str]], Counter]:
    pairs: set[tuple[str, str]] = set()
    counts = Counter()
    with HARD.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["Source"] != "KLIFS_ATPsite_anchor":
                continue
            counts["raw"] += 1
            up = canonical_up(row["UniProt"])
            source_ik = row["InChIKey"].strip().upper()
            got = recomputed_key(row["SMILES"].strip())
            if valid_ik(source_ik):
                if got and got[:14] != source_ik[:14]:
                    counts["excluded_connectivity_conflict"] += 1
                    continue
                pairs.add((up, source_ik))
                counts["retained_reported_inchikey"] += 1
            elif valid_ik(got):
                pairs.add((up, got))
                counts["derived_from_source_smiles"] += 1
            else:
                counts["excluded_unresolved"] += 1
    return pairs, counts


def independent_brenda_pairs() -> tuple[set[tuple[str, str]], Counter]:
    pairs: set[tuple[str, str]] = set()
    counts = Counter()
    with K03_PARAMETERS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            counts["raw"] += 1
            if row["protein_mapping_status"] != "exact_single_uniprot":
                continue
            counts["exact_protein"] += 1
            if row["participant_role"] != "substrate":
                continue
            counts["substrate"] += 1
            ik = row["full_inchikeys"].strip().upper()
            if row["compound_mapping_status"] != "exact_unique_full_inchikey" or not valid_ik(ik):
                continue
            up = canonical_up(row["uniprot_accessions"])
            if not up:
                raise AssertionError(f"invalid exact K03 accession {row['parameter_id']}")
            counts["eligible"] += 1
            counts[f"parameter_type:{row['parameter_type']}"] += 1
            counts[f"quality_tier:{row['quality_tier']}"] += 1
            pairs.add((up, ik))
    return pairs, counts


def verify_sha256_manifest(path: Path, base: Path) -> tuple[bool, int, list[str]]:
    if not path.exists():
        return False, 0, [f"missing manifest: {path}"]
    failures = []
    n = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        target = base / rel
        n += 1
        if not target.exists() or sha256_file(target) != expected:
            failures.append(rel)
    return not failures, n, failures


def verify_k03_manifest() -> tuple[bool, int]:
    rows = read_tsv(K03_MANIFEST)
    good = 0
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = REPO / path
        if path.exists() and sha256_file(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]):
            good += 1
    return good == len(rows), good


def run_checks(check_package_manifest: bool) -> tuple[dict, dict]:
    asd_expected, asd_counts = independent_asd_pairs()
    gtop_expected, gtop_meta = independent_gtop_pairs()
    klifs_expected, klifs_counts = independent_klifs_pairs()
    brenda_expected, brenda_counts = independent_brenda_pairs()

    asd_rows = read_tsv(DATA / "ASD_ALLOSTERIC_PAIRS.tsv")
    gtop_rows = read_tsv(DATA / "GTP_ORTHOSTERIC_SOURCE_PAIRS.tsv")
    klifs_rows = read_tsv(DATA / "KLIFS_ORTHOSTERIC_SOURCE_PAIRS.tsv")
    brenda_rows = read_tsv(DATA / "BRENDA_SUBSTRATE_SOURCE_PAIRS.tsv.gz")
    source_rows = read_tsv(DATA / "ORTHOSTERIC_SOURCE_PAIR_LEDGER.tsv.gz")
    union_rows = read_tsv(DATA / "ORTHOSTERIC_UNION_PAIRS.tsv.gz")
    conflict_rows = read_tsv(DATA / "CLASS_LABEL_CONFLICTS.tsv")
    uncontrolled = read_tsv(DATA / "UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC.tsv.gz")
    proteins = read_tsv(DATA / "UNCONTROLLED_PROTEIN_SUMMARY.tsv")
    gtop_evidence = read_tsv(DATA / "GTP_ORTHOSTERIC_EVIDENCE.tsv.gz")
    gtop_scope = read_tsv(DATA / "GTP_ASSEMBLY_OR_UNRESOLVED_TARGET_LEDGER.tsv")
    klifs_exclusions = read_tsv(DATA / "KLIFS_STRUCTURE_CONFLICTS_AND_EXCLUSIONS.tsv")
    asd_exclusions = read_tsv(DATA / "ASD_STRUCTURE_CONFLICTS_AND_EXCLUSIONS.tsv")

    asd_actual = {(r["uniprot"], r["full_inchikey"]) for r in asd_rows}
    gtop_actual = {(r["uniprot"], r["full_inchikey"]) for r in gtop_rows}
    klifs_actual = {(r["uniprot"], r["full_inchikey"]) for r in klifs_rows}
    brenda_actual = {(r["uniprot"], r["full_inchikey"]) for r in brenda_rows}
    orth_expected = gtop_expected | klifs_expected | brenda_expected
    orth_actual = {(r["uniprot"], r["full_inchikey"]) for r in union_rows}
    conflict_expected = asd_expected & orth_expected
    conflict_actual = {(r["uniprot"], r["full_inchikey"]) for r in conflict_rows}
    expected_labeled = {(up, ik, "allosteric") for up, ik in asd_expected - conflict_expected} | {
        (up, ik, "orthosteric") for up, ik in orth_expected - conflict_expected
    }
    actual_labeled = {(r["uniprot"], r["full_inchikey"], r["class_label"]) for r in uncontrolled}

    fp_ok = True
    for row in uncontrolled:
        if row["fingerprint_eligible"] == "true":
            if recomputed_key(row["canonical_smiles"]) != row["full_inchikey"]:
                fp_ok = False
                break
        elif row["fingerprint_eligible"] != "false":
            fp_ok = False
            break

    snapshot_ok = True
    for row in read_tsv(DATA / "INPUT_SNAPSHOT_HASHES.tsv"):
        path = Path(row["path"])
        snapshot_ok &= path.exists() and path.stat().st_size == int(row["bytes"]) and sha256_file(path) == row["sha256"]

    k03_ok, k03_targets = verify_k03_manifest()
    cache_events = read_tsv(DATA / "NETWORK_REQUEST_ACCOUNTING.tsv")
    with gzip.open(ROOT / "cache/TUNA_EXACT_INCHIKEY_QUERY_EVENTS.jsonl.gz", "rt", encoding="utf-8") as handle:
        tuna_events = [json.loads(line) for line in handle if line.strip()]

    checks = {
        "input_snapshot_hashes_match": bool(snapshot_ok),
        "k03_manifest_all_targets_pass": k03_ok and k03_targets == 15,
        "asd_source_pair_set_recomputed": asd_actual == asd_expected,
        "gtop_source_pair_set_recomputed": gtop_actual == gtop_expected,
        "klifs_source_pair_set_recomputed": klifs_actual == klifs_expected,
        "brenda_source_pair_set_recomputed": brenda_actual == brenda_expected,
        "orthosteric_union_recomputed": orth_actual == orth_expected,
        "class_conflict_intersection_recomputed": conflict_actual == conflict_expected,
        "uncontrolled_labeled_set_recomputed": actual_labeled == expected_labeled,
        "uncontrolled_no_duplicate_labeled_key": len(actual_labeled) == len(uncontrolled),
        "allowed_labels_only": {r["class_label"] for r in uncontrolled} == {"allosteric", "orthosteric"},
        "binary_label_mapping_exact": all((r["class_label"] == "allosteric" and r["binary_label"] == "1") or (r["class_label"] == "orthosteric" and r["binary_label"] == "0") for r in uncontrolled),
        "zero_decoy_rows": all(r["decoy_included"] == "false" and "decoy" not in r["class_label"].lower() for r in uncontrolled),
        "no_protein_or_ligand_control_applied": all(r["protein_control_status"] == "uncontrolled_all_proteins" and r["ligand_control_status"] == "uncontrolled_no_property_grouping" for r in uncontrolled),
        "fingerprint_flags_reproduce_full_inchikey": fp_ok,
        "all_uniprots_and_inchikeys_valid": all(canonical_up(r["uniprot"]) == r["uniprot"] and valid_ik(r["full_inchikey"]) for r in uncontrolled),
        "gtop_no_component_prefix_promoted": all(":" not in r["uniprot"] for r in gtop_evidence),
        "gtop_exact_evidence_recomputed": {(r["source_lane"], int(r["source_row_number"]), r["uniprot"], r["ligand_id"]) for r in gtop_evidence} == gtop_meta["evidence_keys"],
        "gtop_scope_not_promoted": len(gtop_scope) == gtop_meta["scope_rows"] and all(r["exact_protein_pair_emitted"] == "false" for r in gtop_scope),
        "klifs_conflicts_and_unresolved_accounted": len(klifs_exclusions) == klifs_counts["excluded_connectivity_conflict"] + klifs_counts["excluded_unresolved"],
        "asd_exclusions_accounted": len(asd_exclusions) == asd_counts["excluded_uniprot"] + asd_counts["excluded_inchikey"] + asd_counts["excluded_connectivity_conflict"],
        "all_conflicts_absent_from_uncontrolled": not any((r["uniprot"], r["full_inchikey"]) in conflict_expected for r in uncontrolled),
        "all_dataset_ids_unique": len({r["dataset_row_id"] for r in uncontrolled}) == len(uncontrolled),
        "protein_summary_partitions_rows": {r["uniprot"] for r in proteins} == {r["uniprot"] for r in uncontrolled},
        "all_bren_da_parameter_types_retained": {k.split(":", 1)[1] for k in brenda_counts if k.startswith("parameter_type:")} == {"Km", "kcat", "kcat_over_Km", "nH"},
        "brenda_exclude_quality_rows_not_filtered": brenda_counts["quality_tier:EXCLUDE"] == 212,
        "tuna_event_count_matches_accounting": len(tuna_events) == 17 and any(r["phase"] == "production_exact_structure_resolution" and int(r["request_count"]) == len(tuna_events) for r in cache_events),
        "zero_fuzzy_network_requests": all(int(r["fuzzy_requests"]) == 0 for r in cache_events) and all("fuzzy" not in json.dumps(e).lower() for e in tuna_events),
        "methods_and_code_availability_present": all((ROOT / p).exists() for p in ["README.md", "CODE_AVAILABILITY.md", "reports/00_PREPROCESSING_CONTRACT.md", "reports/MATERIALS_AND_METHODS.md"]),
        "protein_controlled_not_built": not (DATA / "PROTEIN_CONTROLLED_ALLOSTERIC_ORTHOSTERIC.tsv.gz").exists(),
        "fully_controlled_not_built": not (DATA / "FULLY_CONTROLLED_ALLOSTERIC_ORTHOSTERIC.tsv.gz").exists(),
    }

    manifest_runtime = {"checked": False, "pass": None, "targets": 0, "failures": []}
    if check_package_manifest:
        ok, n, failures = verify_sha256_manifest(ROOT / "manifests/CHECKSUMS.sha256", REPO)
        manifest_runtime = {"checked": True, "pass": ok, "targets": n, "failures": failures}
        checks["package_checksum_manifest_pass"] = ok

    summary = {
        "asd_raw_rows": asd_counts["raw"],
        "asd_pairs": len(asd_expected),
        "gtop_pairs": len(gtop_expected),
        "klifs_pairs": len(klifs_expected),
        "brenda_evidence_rows": brenda_counts["eligible"],
        "brenda_pairs": len(brenda_expected),
        "orthosteric_union_pairs": len(orth_expected),
        "class_conflicts": len(conflict_expected),
        "uncontrolled_rows": len(uncontrolled),
        "uncontrolled_allosteric": sum(r["class_label"] == "allosteric" for r in uncontrolled),
        "uncontrolled_orthosteric": sum(r["class_label"] == "orthosteric" for r in uncontrolled),
        "uncontrolled_unique_proteins": len({r["uniprot"] for r in uncontrolled}),
        "future_protein_control_eligible_proteins": sum(r["has_both_classes"] == "true" for r in proteins),
        "fingerprint_eligible": sum(r["fingerprint_eligible"] == "true" for r in uncontrolled),
        "fingerprint_unresolved": sum(r["fingerprint_eligible"] == "false" for r in uncontrolled),
        "decoy_rows": 0,
        "package_manifest": manifest_runtime,
    }
    return checks, summary


def write_results(checks: dict, summary: dict) -> None:
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    validation = {
        "validation_version": "ALLOSTERIC_ORTHOSTERIC_UNCONTROLLED_VALIDATION_1.0",
        "status": "PASS",
        "checks": checks,
        "summary": summary,
        "status_scope": "preprocessing/accounting completeness only; not biological mechanism validation",
    }
    (ROOT / "validation").mkdir(parents=True, exist_ok=True)
    (ROOT / "validation/VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    status = {
        "task": "allosteric_orthosteric_uncontrolled_preprocessing_v1",
        "status": "complete",
        "validation_status": "PASS",
        "uncontrolled_complete": True,
        "protein_controlled_complete": False,
        "fully_controlled_complete": False,
        "decoy_rows": 0,
        "summary": summary,
        "claim_limit": "Complete means two-class Uncontrolled preprocessing and accounting passed; it does not prove binding mechanism or allostery.",
    }
    (ROOT / "work/status").mkdir(parents=True, exist_ok=True)
    (ROOT / "work/status/UNCONTROLLED_PREPROCESSING.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    report = f"""# Uncontrolled allosteric–orthosteric preprocessing

## Result

The two-class Uncontrolled dataset passes independent source reconstruction. It contains {summary['uncontrolled_rows']:,} exact, nonconflicting protein–ligand rows: {summary['uncontrolled_allosteric']:,} allosteric and {summary['uncontrolled_orthosteric']:,} orthosteric. No decoy rows were included.

## Source accounting

- ASD: {summary['asd_pairs']:,} exact pairs before class-conflict removal.
- GtoPdb: {summary['gtop_pairs']:,} unique exact protein–ligand pairs. Assembly-only targets were not assigned to subunits.
- KLIFS: {summary['klifs_pairs']:,} ATP-site pairs after deterministic structure correction and conflict quarantine.
- BRENDA: {summary['brenda_evidence_rows']:,} exact substrate evidence rows, representing {summary['brenda_pairs']:,} pairs.
- Orthosteric union: {summary['orthosteric_union_pairs']:,} pairs.
- Exact cross-class conflicts: {summary['class_conflicts']:,}, excluded from both labels.

## Future controls

The final Uncontrolled table spans {summary['uncontrolled_unique_proteins']:,} proteins; {summary['future_protein_control_eligible_proteins']:,} have at least one pair in each class and are candidates for a later protein-controlled derivation. Ligand-property grouping has not been performed. Of the final rows, {summary['fingerprint_eligible']:,} have an exact full-InChIKey-reproducing SMILES and {summary['fingerprint_unresolved']:,} retain exact source identity but are not fingerprint-ready.

## Limits

BRENDA substrates are functional substrate references and do not by themselves establish a crystallographic binding pose. GtoP assembly membership was not interpreted as subunit binding. No competitive/noncompetitive, mechanism, binding-site, or new allostery conclusion was made.
"""
    (ROOT / "reports/01_UNCONTROLLED_DATASET_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-results", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    checks, summary = run_checks(check_package_manifest=args.validate_only)
    if args.write_results:
        write_results(checks, summary)
    if not all(checks.values()):
        print(json.dumps({"status": "FAIL", "failed": {k: v for k, v in checks.items() if not v}, "summary": summary}, indent=2, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps({"status": "PASS", "checks": checks, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
