#!/usr/bin/env python3
"""Build the two-class allosteric/orthosteric Uncontrolled dataset.

The builder consumes only named frozen inputs.  It includes no decoy records,
does not filter BRENDA by Km/kcat/quality tier, and performs no protein or
ligand-property matching.  Exact UniProt + full InChIKey conflicts between ASD
and an orthosteric source are quarantined rather than forced into either class.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"

ASD = REPO / "analysis/asd_as_clean.csv"
WWPDB = Path("/shared_data/11.HS_allostery/Data/wwpdb_compounds.pickle")
HARD_ORTHO = Path(
    "/shared_data/11.HS_allostery/18.Structure_known/"
    "4.Evaluate_Orthosteric_LowScore_Enrichment/"
    "hard_orthosteric_reference_pairs_loaded.tsv"
)
OLD_ORTHO_PARSER = Path(
    "/shared_data/11.HS_allostery/18.Structure_known/"
    "4.Evaluate_Orthosteric_LowScore_Enrichment.py"
)
GTOP_PAIRINGS = Path("/shared_data/11.HS_allostery/Data/Orthosteric_References/GtoP/endogenous_ligand_pairings_all.tsv")
GTOP_INTERACTIONS = Path("/shared_data/11.HS_allostery/Data/Orthosteric_References/GtoP/interactions_all.tsv")
GTOP_EVIDENCE = DATA / "GTP_ORTHOSTERIC_EVIDENCE.tsv.gz"
GTOP_PAIRS = DATA / "GTP_ORTHOSTERIC_SOURCE_PAIRS.tsv"
GTOP_SCOPE = DATA / "GTP_ASSEMBLY_OR_UNRESOLVED_TARGET_LEDGER.tsv"
GTOP_SUMMARY = DATA / "GTP_PARSER_SUMMARY.json"

K03_DIR = REPO / "analysis/kinetic_allostery_model_v1/data/k03_brenda"
K03_PARAMETERS = K03_DIR / "K03_BRENDA_PARAMETERS.tsv"
K03_MANIFEST = REPO / "analysis/kinetic_allostery_model_v1/manifests/K03_MANIFEST.tsv"
K03_EXTRACTOR = REPO / "analysis/kinetic_allostery_model_v1/scripts/extract_k03_brenda_parameters.py"
BRENDA_STRUCTURES = DATA / "BRENDA_EXACT_STRUCTURE_SUPPLEMENT.tsv"
BRENDA_STRUCTURE_SUMMARY = DATA / "BRENDA_EXACT_STRUCTURE_RESOLUTION_SUMMARY.json"

IK_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
UP_RE = re.compile(r"^[A-Z0-9]{6,10}(?:-[0-9]+)?$")
NA = "NA"


def valid_ik(value: str) -> bool:
    return bool(IK_RE.fullmatch((value or "").strip().upper()))


def canonical_uniprot(value: str) -> str:
    value = (value or "").strip().upper()
    return value.split("-")[0] if UP_RE.fullmatch(value) else ""


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(x) for x in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def joined(values) -> str:
    vals = sorted({str(x).strip() for x in values if str(x).strip() not in {"", NA} and not str(x).startswith("NA_")})
    return ";".join(vals) if vals else NA


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_smiles_matching(smiles: str, inchikey: str) -> tuple[str, str, str]:
    if not smiles or smiles == NA:
        return NA, "smiles_unavailable", NA
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return NA, "smiles_unparseable", NA
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    got = Chem.MolToInchiKey(mol)
    if got == inchikey:
        return canonical, "smiles_reproduces_full_inchikey", got
    if got[:14] == inchikey[:14]:
        return NA, "smiles_connectivity_only_not_full_inchikey", got
    return NA, "smiles_inchikey_connectivity_conflict", got


def select_exact_smiles(values, inchikey: str) -> tuple[str, str]:
    """Select one deterministic exact SMILES while retaining all alternatives."""
    candidates: set[str] = set()
    for value in values:
        value = str(value).strip()
        if not value or value == NA:
            continue
        mol = Chem.MolFromSmiles(value)
        if mol is None or Chem.MolToInchiKey(mol) != inchikey:
            continue
        candidates.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    if not candidates:
        return NA, NA
    ordered = sorted(candidates, key=lambda x: (len(x), x))
    return ordered[0], ";".join(sorted(candidates))


def write_tsv(path: Path, rows: list[dict], fields: list[str], gz: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as z:
                with io.TextIOWrapper(z, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_asd() -> tuple[list[dict], list[dict], dict]:
    wwpdb = pickle.load(WWPDB.open("rb"))
    lineage: list[dict] = []
    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    status = Counter()
    with ASD.open(encoding="utf-8", newline="") as handle:
        for source_row, row in enumerate(csv.DictReader(handle), start=2):
            up_raw = row.get("uniprot", "").strip()
            up = canonical_uniprot(up_raw)
            ccd = row.get("ccd", "").strip().upper()
            entry = wwpdb.get(ccd, {}) if ccd else {}
            desc = entry.get("descriptors", {}) if isinstance(entry, dict) else {}
            ik = str(desc.get("INCHIKEY") or "").strip().upper()
            raw_smiles = str(desc.get("SMILES_CANONICAL") or desc.get("SMILES") or "").strip()
            canonical = NA
            structure_status = "not_evaluated"
            recomputed_ik = NA
            if not up:
                mapping_status = "excluded_missing_or_invalid_uniprot"
                reason = "ASD UniProt is missing or invalid"
            elif not valid_ik(ik):
                mapping_status = "excluded_wwPDB_full_inchikey_unavailable"
                reason = "ASD CCD has no valid full InChIKey in frozen wwPDB dictionary"
            else:
                canonical, structure_status, recomputed_ik = canonical_smiles_matching(raw_smiles, ik)
                if structure_status == "smiles_inchikey_connectivity_conflict":
                    mapping_status = "excluded_source_smiles_inchikey_connectivity_conflict"
                    reason = "wwPDB source full InChIKey and source SMILES have different connectivity"
                else:
                    mapping_status = "included_exact_protein_full_inchikey"
                    reason = NA
            evidence_id = stable_id("ASDEV", row.get("asd_row_id", source_row), up_raw, row.get("pdb", ""), ccd)
            out = {
                "evidence_id": evidence_id,
                "source_database": "ASD",
                "source_row_number": source_row,
                "asd_row_id": row.get("asd_row_id", "") or NA,
                "pdb": row.get("pdb", "").strip() or NA,
                "ccd": ccd or NA,
                "uniprot_raw": up_raw or NA,
                "uniprot": up or NA,
                "target_id": row.get("target_id", "").strip() or NA,
                "modulator_class": row.get("modulator_class", "").strip() or NA,
                "modulator_feature": row.get("modulator_feature", "").strip() or NA,
                "site_overlap": row.get("site_overlap", "").strip() or NA,
                "n_allo_res": row.get("n_allo_res", "").strip() or NA,
                "full_inchikey": ik if valid_ik(ik) else NA,
                "connectivity_key": ik[:14] if valid_ik(ik) else NA,
                "canonical_smiles": canonical,
                "source_smiles": raw_smiles or NA,
                "structure_status": structure_status,
                "rdkit_full_inchikey_from_source_smiles": recomputed_ik,
                "mapping_status": mapping_status,
                "exclusion_reason": reason,
                "class_label": "allosteric" if mapping_status.startswith("included") else NA,
                "evidence_subtype": "ASD_curated_allosteric_record_no_exact_site_filter" if mapping_status.startswith("included") else NA,
            }
            lineage.append(out)
            status[mapping_status] += 1
            if mapping_status.startswith("included"):
                agg[(up, ik)].append(out)

    pairs = []
    for (up, ik), rows in sorted(agg.items()):
        selected_smiles, all_smiles = select_exact_smiles((r["canonical_smiles"] for r in rows), ik)
        pairs.append(
            {
                "source_pair_id": stable_id("ASDPAIR", up, ik),
                "source_database": "ASD",
                "source_lane": "ASD_all_records",
                "uniprot": up,
                "full_inchikey": ik,
                "connectivity_key": ik[:14],
                "canonical_smiles": selected_smiles,
                "structure_status": joined(r["structure_status"] for r in rows),
                "fingerprint_eligible": str(selected_smiles != NA).lower(),
                "ligand_names": NA,
                "source_ligand_ids": joined(r["ccd"] for r in rows),
                "source_target_ids": joined(r["target_id"] for r in rows),
                "source_structure_ids": joined(r["pdb"] for r in rows),
                "evidence_ids": joined(r["evidence_id"] for r in rows),
                "n_evidence_rows": len(rows),
                "class_label": "allosteric",
                "evidence_subtype": "ASD_curated_allosteric_record_no_exact_site_filter",
                "claim_limit": "ASD allosteric annotation retained without imposing the unrelated BioLiP exact-site audit filter.",
            }
        )
    summary = {
        "raw_rows": len(lineage),
        "included_evidence_rows": status["included_exact_protein_full_inchikey"],
        "excluded_missing_or_invalid_uniprot": status["excluded_missing_or_invalid_uniprot"],
        "excluded_wwPDB_full_inchikey_unavailable": status["excluded_wwPDB_full_inchikey_unavailable"],
        "excluded_source_smiles_inchikey_connectivity_conflict": status["excluded_source_smiles_inchikey_connectivity_conflict"],
        "included_structure_status_counts": dict(sorted(Counter(r["structure_status"] for r in lineage if r["mapping_status"].startswith("included")).items())),
        "unique_pairs": len(pairs),
        "unique_proteins": len({r["uniprot"] for r in pairs}),
    }
    return lineage, pairs, summary


def load_gtop_pairs() -> tuple[list[dict], dict]:
    summary = json.loads(GTOP_SUMMARY.read_text())
    with GTOP_PAIRS.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    out = []
    for row in source_rows:
        canonical, structure_status, got = canonical_smiles_matching(row["canonical_smiles"], row["full_inchikey"])
        if canonical == NA:
            raise ValueError(f"GtoP source structure conflict: {row['source_pair_id']} {structure_status} {got}")
        out.append(
            {
                "source_pair_id": row["source_pair_id"],
                "source_database": "GtoPdb",
                "source_lane": row["source_lane"],
                "uniprot": row["uniprot"],
                "full_inchikey": row["full_inchikey"],
                "connectivity_key": row["connectivity_key"],
                "canonical_smiles": canonical,
                "structure_status": structure_status,
                "fingerprint_eligible": "true",
                "ligand_names": row["ligand_names"],
                "source_ligand_ids": row["ligand_ids"],
                "source_target_ids": row["target_ids"],
                "source_structure_ids": NA,
                "evidence_ids": row["evidence_ids"],
                "n_evidence_rows": int(row["n_evidence_rows"]),
                "class_label": "orthosteric",
                "evidence_subtype": "GtoP_endogenous_ligand_reference",
                "claim_limit": "GtoP endogenous target-ligand reference; complex-only targets are not assigned to subunits.",
            }
        )
    return out, summary


def build_klifs() -> tuple[list[dict], list[dict], list[dict], dict]:
    evidence: list[dict] = []
    exclusions: list[dict] = []
    statuses = Counter()
    with HARD_ORTHO.open(encoding="utf-8", newline="") as handle:
        for source_row, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=2):
            if row["Source"] != "KLIFS_ATPsite_anchor":
                continue
            up = canonical_uniprot(row["UniProt"])
            source_ik = row["InChIKey"].strip().upper()
            raw_smiles = row["SMILES"].strip()
            mol = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
            rdkit_ik = Chem.MolToInchiKey(mol) if mol else ""
            canonical_raw = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else NA
            if valid_ik(source_ik):
                final_ik = source_ik
                identity_method = "reported_full_inchikey"
                if rdkit_ik == source_ik:
                    canonical = canonical_raw
                    structure_status = "smiles_reproduces_full_inchikey"
                    include = True
                elif rdkit_ik and rdkit_ik[:14] == source_ik[:14]:
                    canonical = NA
                    structure_status = "source_smiles_connectivity_only_stereochemistry_conflict"
                    include = True
                elif not rdkit_ik:
                    canonical = NA
                    structure_status = "source_smiles_unparseable_full_inchikey_retained"
                    include = True
                else:
                    canonical = NA
                    structure_status = "source_smiles_full_inchikey_connectivity_conflict"
                    include = False
            elif valid_ik(rdkit_ik):
                final_ik = rdkit_ik
                identity_method = "RDKit_full_inchikey_from_source_smiles"
                canonical = canonical_raw
                structure_status = "derived_smiles_reproduces_full_inchikey"
                include = True
            else:
                final_ik = NA
                identity_method = "unresolved"
                canonical = NA
                structure_status = "source_full_inchikey_invalid_and_smiles_unparseable"
                include = False
            if not up:
                include = False
                structure_status = "invalid_uniprot"
            statuses[structure_status] += 1
            evidence_id = stable_id("KLIFSEV", source_row, row["UniProt"], row["Ligand_ID"], row["CCD"])
            base = {
                "evidence_id": evidence_id,
                "source_database": "KLIFS",
                "source_lane": "KLIFS_ATPsite_anchor",
                "source_row_number": source_row,
                "uniprot": up or NA,
                "target_id": row["Target_ID"].strip() or NA,
                "target_name": row["Target_Name"].strip() or NA,
                "ligand_id": row["Ligand_ID"].strip() or NA,
                "ligand_name": row["Ligand_Name"].strip() or NA,
                "ccd": row["CCD"].strip() or NA,
                "source_smiles": raw_smiles or NA,
                "source_full_inchikey": source_ik or NA,
                "rdkit_full_inchikey_from_source_smiles": rdkit_ik or NA,
                "full_inchikey": final_ik,
                "connectivity_key": final_ik[:14] if valid_ik(final_ik) else NA,
                "canonical_smiles": canonical,
                "identity_resolution_method": identity_method,
                "structure_status": structure_status,
                "included_exact_pair": str(include).lower(),
                "class_label": "orthosteric" if include else NA,
                "evidence_subtype": "KLIFS_ATP_site_anchor" if include else NA,
            }
            if include:
                evidence.append(base)
            else:
                base["exclusion_reason"] = "contradictory_or_unresolved_exact_ligand_identity"
                exclusions.append(base)

    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in evidence:
        agg[(row["uniprot"], row["full_inchikey"])].append(row)
    pairs = []
    for (up, ik), rows in sorted(agg.items()):
        smiles, all_smiles = select_exact_smiles((r["canonical_smiles"] for r in rows), ik)
        statuses_here = joined(r["structure_status"] for r in rows)
        pairs.append(
            {
                "source_pair_id": stable_id("KLIFSPAIR", up, ik),
                "source_database": "KLIFS",
                "source_lane": "KLIFS_ATPsite_anchor",
                "uniprot": up,
                "full_inchikey": ik,
                "connectivity_key": ik[:14],
                "canonical_smiles": smiles,
                "structure_status": statuses_here,
                "fingerprint_eligible": str(smiles != NA).lower(),
                "ligand_names": joined(r["ligand_name"] for r in rows),
                "source_ligand_ids": joined(r["ligand_id"] for r in rows),
                "source_target_ids": joined(r["target_id"] for r in rows),
                "source_structure_ids": joined(r["ccd"] for r in rows),
                "evidence_ids": joined(r["evidence_id"] for r in rows),
                "n_evidence_rows": len(rows),
                "class_label": "orthosteric",
                "evidence_subtype": "KLIFS_ATP_site_anchor",
                "claim_limit": "KLIFS ATP-site anchor; no allostery or activity-mechanism inference.",
            }
        )
    summary = {
        "raw_rows": len(evidence) + len(exclusions),
        "included_evidence_rows": len(evidence),
        "excluded_rows": len(exclusions),
        "unique_pairs": len(pairs),
        "unique_proteins": len({r["uniprot"] for r in pairs}),
        "structure_status_counts": dict(sorted(statuses.items())),
        "fingerprint_eligible_pairs": sum(r["fingerprint_eligible"] == "true" for r in pairs),
    }
    return evidence, pairs, exclusions, summary


def load_brenda_structures() -> dict[str, dict[str, str]]:
    with BRENDA_STRUCTURES.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    out = {r["full_inchikey"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError("Duplicate full InChIKey in BRENDA structure supplement")
    return out


def build_brenda() -> tuple[list[dict], list[dict], dict]:
    structures = load_brenda_structures()
    evidence: list[dict] = []
    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sequential = Counter()
    with K03_PARAMETERS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sequential["all_parameter_rows"] += 1
            if row["protein_mapping_status"] != "exact_single_uniprot":
                continue
            sequential["exact_single_uniprot"] += 1
            if row["participant_role"] != "substrate":
                continue
            sequential["exact_single_uniprot_and_substrate"] += 1
            ik = row["full_inchikeys"].strip().upper()
            if row["compound_mapping_status"] != "exact_unique_full_inchikey" or not valid_ik(ik):
                continue
            sequential["eligible_exact_evidence_rows"] += 1
            up = canonical_uniprot(row["uniprot_accessions"])
            if not up:
                raise ValueError(f"K03 exact_single_uniprot row has invalid accession: {row['parameter_id']}")
            struct = structures.get(ik)
            if struct is None:
                raise ValueError(f"BRENDA structure supplement missing {ik}")
            evidence_id = stable_id("BRENDAEV", row["parameter_id"], up, ik)
            out = {
                "evidence_id": evidence_id,
                "source_database": "BRENDA",
                "source_release": row["source_release"],
                "source_record_id": row["source_record_id"],
                "parameter_id": row["parameter_id"],
                "experiment_id": row["experiment_id"],
                "uniprot": up,
                "brenda_ec_number": row["brenda_ec_number"],
                "participant_name": row["participant_names"],
                "participant_role": row["participant_role"],
                "full_inchikey": ik,
                "connectivity_key": ik[:14],
                "canonical_smiles": struct["canonical_smiles"],
                "structure_resolution_source": struct["resolution_source"],
                "structure_resolution_status": struct["resolution_status"],
                "pubchem_cids": struct["pubchem_cids"],
                "parameter_type": row["parameter_type"],
                "quality_tier": row["quality_tier"],
                "source_publication_ids": row["source_publication_ids"],
                "raw_source_locator": row["raw_source_locator"],
                "class_label": "orthosteric",
                "evidence_subtype": "BRENDA_exact_functional_substrate",
                "parameter_used_as_inclusion_filter": "false",
                "claim_limit": "Exact BRENDA functional substrate reference; no binding pose, mechanism, or allostery inference.",
            }
            evidence.append(out)
            agg[(up, ik)].append(out)

    pairs = []
    for (up, ik), rows in sorted(agg.items()):
        smiles, all_smiles = select_exact_smiles((r["canonical_smiles"] for r in rows), ik)
        pairs.append(
            {
                "source_pair_id": stable_id("BRENDAPAIR", up, ik),
                "source_database": "BRENDA",
                "source_lane": "BRENDA_exact_functional_substrate",
                "uniprot": up,
                "full_inchikey": ik,
                "connectivity_key": ik[:14],
                "canonical_smiles": smiles,
                "structure_status": joined(r["structure_resolution_status"] for r in rows),
                "fingerprint_eligible": str(smiles != NA).lower(),
                "ligand_names": joined(r["participant_name"] for r in rows),
                "source_ligand_ids": NA,
                "source_target_ids": joined(r["brenda_ec_number"] for r in rows),
                "source_structure_ids": joined(r["pubchem_cids"] for r in rows),
                "evidence_ids": joined(r["evidence_id"] for r in rows),
                "n_evidence_rows": len(rows),
                "class_label": "orthosteric",
                "evidence_subtype": "BRENDA_exact_functional_substrate",
                "claim_limit": "Exact BRENDA functional substrate reference; parameter type and quality tier are provenance only.",
                "parameter_types": joined(r["parameter_type"] for r in rows),
                "quality_tiers": joined(r["quality_tier"] for r in rows),
            }
        )
    summary = {
        **dict(sequential),
        "unique_pairs": len(pairs),
        "unique_proteins": len({r["uniprot"] for r in pairs}),
        "unique_full_inchikeys": len({r["full_inchikey"] for r in pairs}),
        "fingerprint_eligible_pairs": sum(r["fingerprint_eligible"] == "true" for r in pairs),
        "fingerprint_unresolved_pairs": sum(r["fingerprint_eligible"] == "false" for r in pairs),
        "parameter_types": dict(sorted(Counter(r["parameter_type"] for r in evidence).items())),
        "quality_tiers": dict(sorted(Counter(r["quality_tier"] for r in evidence).items())),
    }
    return evidence, pairs, summary


SOURCE_PAIR_FIELDS = [
    "source_pair_id", "source_database", "source_lane", "uniprot", "full_inchikey",
    "connectivity_key", "canonical_smiles", "structure_status", "fingerprint_eligible",
    "ligand_names", "source_ligand_ids", "source_target_ids", "source_structure_ids",
    "evidence_ids", "n_evidence_rows", "class_label", "evidence_subtype", "claim_limit",
]


def aggregate_orthosteric(source_pairs: list[dict]) -> list[dict]:
    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in source_pairs:
        agg[(row["uniprot"], row["full_inchikey"])].append(row)
    rows = []
    for (up, ik), evidence in sorted(agg.items()):
        smiles, all_smiles = select_exact_smiles((r["canonical_smiles"] for r in evidence), ik)
        rows.append(
            {
                "orthosteric_pair_id": stable_id("ORTHOPAIR", up, ik),
                "uniprot": up,
                "full_inchikey": ik,
                "connectivity_key": ik[:14],
                "canonical_smiles": smiles,
                "canonical_smiles_candidates": all_smiles,
                "fingerprint_eligible": str(smiles != NA).lower(),
                "structure_statuses": joined(r["structure_status"] for r in evidence),
                "source_databases": joined(r["source_database"] for r in evidence),
                "source_lanes": joined(r["source_lane"] for r in evidence),
                "source_pair_ids": joined(r["source_pair_id"] for r in evidence),
                "evidence_subtypes": joined(r["evidence_subtype"] for r in evidence),
                "n_source_pair_rows": len(evidence),
                "n_evidence_rows": sum(int(r["n_evidence_rows"]) for r in evidence),
                "class_label": "orthosteric",
                "claim_limit": "Orthosteric class requested for preprocessing; source-specific evidence scope remains explicit.",
            }
        )
    return rows


def build_uncontrolled(asd_pairs: list[dict], orth_pairs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    a = {(r["uniprot"], r["full_inchikey"]): r for r in asd_pairs}
    o = {(r["uniprot"], r["full_inchikey"]): r for r in orth_pairs}
    conflict_keys = sorted(set(a) & set(o))
    conflicts = []
    for up, ik in conflict_keys:
        conflicts.append(
            {
                "conflict_id": stable_id("CLASSCONFLICT", up, ik),
                "uniprot": up,
                "full_inchikey": ik,
                "connectivity_key": ik[:14],
                "allosteric_pair_id": a[(up, ik)]["source_pair_id"],
                "orthosteric_pair_id": o[(up, ik)]["orthosteric_pair_id"],
                "orthosteric_source_databases": o[(up, ik)]["source_databases"],
                "orthosteric_source_lanes": o[(up, ik)]["source_lanes"],
                "resolution": "excluded_from_both_classes",
                "reason": "identical_exact_UniProt_full_InChIKey_present_in_allosteric_and_orthosteric_sources",
            }
        )

    rows = []
    for key, source in sorted(a.items()):
        if key in o:
            continue
        rows.append(
            {
                "dataset_row_id": stable_id("UNCONTROLLED", "allosteric", *key),
                "uniprot": key[0],
                "full_inchikey": key[1],
                "connectivity_key": source["connectivity_key"],
                "canonical_smiles": source["canonical_smiles"],
                "fingerprint_eligible": source["fingerprint_eligible"],
                "class_label": "allosteric",
                "binary_label": 1,
                "source_databases": source["source_database"],
                "source_lanes": source["source_lane"],
                "source_pair_ids": source["source_pair_id"],
                "evidence_subtypes": source["evidence_subtype"],
                "n_source_pair_rows": 1,
                "n_evidence_rows": source["n_evidence_rows"],
                "protein_control_status": "uncontrolled_all_proteins",
                "ligand_control_status": "uncontrolled_no_property_grouping",
                "class_conflict_status": "nonconflicting",
                "decoy_included": "false",
                "claim_limit": source["claim_limit"],
            }
        )
    for key, source in sorted(o.items()):
        if key in a:
            continue
        rows.append(
            {
                "dataset_row_id": stable_id("UNCONTROLLED", "orthosteric", *key),
                "uniprot": key[0],
                "full_inchikey": key[1],
                "connectivity_key": source["connectivity_key"],
                "canonical_smiles": source["canonical_smiles"],
                "fingerprint_eligible": source["fingerprint_eligible"],
                "class_label": "orthosteric",
                "binary_label": 0,
                "source_databases": source["source_databases"],
                "source_lanes": source["source_lanes"],
                "source_pair_ids": source["source_pair_ids"],
                "evidence_subtypes": source["evidence_subtypes"],
                "n_source_pair_rows": source["n_source_pair_rows"],
                "n_evidence_rows": source["n_evidence_rows"],
                "protein_control_status": "uncontrolled_all_proteins",
                "ligand_control_status": "uncontrolled_no_property_grouping",
                "class_conflict_status": "nonconflicting",
                "decoy_included": "false",
                "claim_limit": source["claim_limit"],
            }
        )
    rows.sort(key=lambda r: (r["class_label"], r["uniprot"], r["full_inchikey"]))

    by_protein: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_protein[row["uniprot"]].append(row)
    protein_summary = []
    for up, values in sorted(by_protein.items()):
        counts = Counter(r["class_label"] for r in values)
        protein_summary.append(
            {
                "uniprot": up,
                "n_allosteric_pairs": counts["allosteric"],
                "n_orthosteric_pairs": counts["orthosteric"],
                "has_both_classes": str(bool(counts["allosteric"] and counts["orthosteric"])).lower(),
                "future_protein_control_eligible": str(bool(counts["allosteric"] and counts["orthosteric"])).lower(),
            }
        )
    return rows, conflicts, protein_summary


def input_snapshots() -> list[dict]:
    inputs = [
        ("ASD source", ASD), ("wwPDB CCD dictionary", WWPDB),
        ("GtoP pairings", GTOP_PAIRINGS), ("GtoP interactions", GTOP_INTERACTIONS),
        ("historical GtoP/KLIFS exact chemistry table", HARD_ORTHO),
        ("historical orthosteric parser", OLD_ORTHO_PARSER),
        ("K03 parameters", K03_PARAMETERS), ("K03 manifest", K03_MANIFEST),
        ("K03 extractor", K03_EXTRACTOR),
        ("corrected GtoP evidence", GTOP_EVIDENCE), ("corrected GtoP pairs", GTOP_PAIRS),
        ("corrected GtoP target-scope ledger", GTOP_SCOPE),
        ("BRENDA exact structure supplement", BRENDA_STRUCTURES),
        ("BRENDA structure resolution summary", BRENDA_STRUCTURE_SUMMARY),
    ]
    return [
        {
            "role": role,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for role, path in inputs
    ]


def write_documents(summary: dict) -> None:
    CONFIG.mkdir(parents=True, exist_ok=True)
    contract = {
        "contract_version": "ALLOSTERIC_ORTHOSTERIC_UNCONTROLLED_1.0",
        "scope": "Two-class exact protein-ligand preprocessing through Uncontrolled only",
        "pair_key": ["canonical UniProt accession", "full InChIKey"],
        "class_labels": {"allosteric": 1, "orthosteric": 0},
        "allosteric_source": "all ASD rows with exact UniProt and wwPDB CCD full InChIKey; no exact-site filter",
        "orthosteric_sources": {
            "GtoPdb": "direct exact-protein endogenous ligand references only; assemblies quarantined",
            "KLIFS": "ATP-site anchors with valid reported full InChIKey or deterministic RDKit derivation from source SMILES",
            "BRENDA": "exact_single_uniprot + substrate + exact_unique_full_inchikey; parameter type and quality are provenance only",
        },
        "excluded_classes": ["decoy", "weak-binding", "generic negative", "proxy-only label"],
        "conflict_policy": "Exact UniProt/full-InChIKey keys in both classes are excluded from both and retained in CLASS_LABEL_CONFLICTS.tsv",
        "structure_conflict_policy": {
            "full_inchikey_smiles_exact": "retain pair and fingerprint",
            "same_connectivity_stereochemistry_only_conflict": "retain source full-InChIKey pair; set fingerprint unavailable",
            "valid_source_full_inchikey_smiles_unparseable": "retain source full-InChIKey pair; set fingerprint unavailable",
            "different_connectivity": "exclude exact pair and retain in source-conflict ledger",
            "missing_full_inchikey_with_valid_source_smiles": "derive full InChIKey deterministically with pinned RDKit and record provenance",
        },
        "control_scope": {
            "protein_control": "not applied",
            "ligand_property_grouping": "not applied",
            "future_use": "Protein-controlled and Fully-controlled datasets may be derived without rebuilding source evidence",
        },
        "network": {
            "BRENDA_query": "none; K03 reused",
            "Tuna_ES": "exact full InChIKey and CID lookups for missing SMILES only; no fuzzy/name search",
            "production_resolver_requests": summary["brenda_structure_resolution"]["live_network_requests"],
            "interactive_preflight_requests": 6,
        },
        "scientific_claim_limits": [
            "BRENDA substrate does not prove a crystallographic active-site pose",
            "GtoP assembly membership is not an exact subunit binding assignment",
            "No mechanism, competitive/noncompetitive, binding-site, or allostery inference is made from orthosteric sources",
        ],
    }
    write_json(CONFIG / "DATASET_CONTRACT.json", contract)

    (ROOT / "README.md").write_text(
        "# Allosteric–orthosteric preprocessing v1\n\n"
        "This package constructs a two-class, exact UniProt/full-InChIKey Uncontrolled dataset from ASD, GtoPdb, KLIFS, and the frozen K03 BRENDA extraction. It contains no decoys and performs no protein matching or ligand-property grouping.\n\n"
        "## Reproduction order\n\n"
        "1. `parse_gtop_orthosteric_references.py`\n"
        "2. `resolve_brenda_exact_structures.py --verify-existing` (or `--run` only when exact Tuna resolution is intentionally repeated)\n"
        "3. `build_uncontrolled_allosteric_orthosteric.py`\n"
        "4. `validate_uncontrolled_allosteric_orthosteric.py --write-results`\n"
        "5. `finalize_uncontrolled_package.py`\n"
        "6. `validate_uncontrolled_allosteric_orthosteric.py --validate-only`\n",
        encoding="utf-8",
    )
    (ROOT / "CODE_AVAILABILITY.md").write_text(
        "# Code availability\n\n"
        "All preprocessing code created for this dataset is contained in `scripts/`. The scripts use named, checksum-frozen inputs and never recursively scan the repository. The historical parser is not modified; its multi-subunit target defect is corrected append-only by `parse_gtop_orthosteric_references.py`. All generated tables, validation files, reports, and query caches are covered by the package checksum manifest.\n",
        encoding="utf-8",
    )
    (REPORTS / "00_PREPROCESSING_CONTRACT.md").write_text(
        "# Preprocessing contract\n\n"
        "The dataset unit is one nonconflicting exact `(canonical UniProt accession, full InChIKey, class label)` pair. ASD supplies the allosteric class without applying the unrelated BioLiP exact-site filter. GtoPdb, KLIFS, and BRENDA supply the orthosteric class with source-specific evidence scopes retained. Decoys, generic negatives, and proxy-only labels are absent. BRENDA Km, kcat, kcat/Km, nH, and quality tier are metadata rather than selection criteria.\n\n"
        "GtoP targets with only subunit membership are retained at assembly scope and are not flattened into arbitrary protein pairs. KLIFS source-field conflicts are retained in a separate ledger. An exact key occurring in both classes is excluded from both classes and retained in the class-conflict ledger. Protein matching and ligand-property grouping are explicitly deferred.\n",
        encoding="utf-8",
    )
    s = summary
    (REPORTS / "MATERIALS_AND_METHODS.md").write_text(
        f"# Materials and Methods: allosteric–orthosteric preprocessing\n\n"
        f"We constructed an Uncontrolled two-class protein–ligand dataset using a canonical UniProt accession and full InChIKey as the pair key. All {s['asd']['raw_rows']:,} ASD rows were considered without applying an exact-site filter. CCD identifiers were mapped through the frozen wwPDB chemical-component dictionary; {s['asd']['included_evidence_rows']:,} source rows yielded {s['asd']['unique_pairs']:,} unique allosteric pairs. Across sources, a valid source full InChIKey was retained when the associated SMILES differed only in stereochemistry or was unparseable, but the row was marked fingerprint-unavailable. A direct connectivity contradiction caused exact-pair exclusion and separate conflict retention.\n\n"
        f"GtoPdb 2026.1 endogenous pairings and endogenous interactions were reparsed with explicit target scope. Direct target accessions yielded {s['gtop']['exact_source_pairs']:,} source-specific pairs ({s['gtop']['exact_unique_protein_ligand_pairs']:,} unique protein–ligand pairs). Subunit-only or accession-unresolved records ({s['gtop']['assembly_or_unresolved_source_rows']:,} rows) were retained separately and never assigned to an arbitrary component.\n\n"
        f"KLIFS ATP-site anchors were accepted when a valid source full InChIKey was available, except for direct source SMILES/connectivity contradictions. Missing InChIKeys were derived only from the source SMILES using RDKit {rdBase.rdkitVersion}. This produced {s['klifs']['unique_pairs']:,} exact pairs and retained {s['klifs']['excluded_rows']:,} contradictory or unresolved rows separately.\n\n"
        f"BRENDA 2026.1 was not re-downloaded or reparsed. We reused K03 and selected only rows with exact_single_uniprot protein mapping, participant_role=substrate, and exact_unique_full_inchikey compound mapping. Parameter type and quality tier did not affect inclusion. The {s['brenda']['eligible_exact_evidence_rows']:,} selected parameter records represented {s['brenda']['unique_pairs']:,} pairs across {s['brenda']['unique_proteins']:,} proteins. Missing SMILES were queried from Tuna only by exact full InChIKey and CID and were accepted only when RDKit reproduced the same full InChIKey; no fuzzy or name query was used.\n\n"
        f"We unioned the three orthosteric sources, preserved their individual evidence provenance, and quarantined {s['conflicts']:,} exact keys present in both classes. The final Uncontrolled dataset contains {s['uncontrolled']['rows']:,} rows: {s['uncontrolled']['allosteric']:,} allosteric and {s['uncontrolled']['orthosteric']:,} orthosteric. No decoy, protein matching, ligand-property grouping, mechanism label, or allostery inference was introduced.\n",
        encoding="utf-8",
    )


def main() -> None:
    for path in (DATA, REPORTS, CONFIG):
        path.mkdir(parents=True, exist_ok=True)
    asd_lineage, asd_pairs, asd_summary = build_asd()
    gtop_pairs, gtop_summary = load_gtop_pairs()
    klifs_evidence, klifs_pairs, klifs_exclusions, klifs_summary = build_klifs()
    brenda_evidence, brenda_pairs, brenda_summary = build_brenda()

    orth_source_pairs = gtop_pairs + klifs_pairs + brenda_pairs
    orth_source_pairs.sort(key=lambda r: (r["source_database"], r["source_lane"], r["uniprot"], r["full_inchikey"]))
    orth_union = aggregate_orthosteric(orth_source_pairs)
    uncontrolled, conflicts, protein_summary = build_uncontrolled(asd_pairs, orth_union)

    # Core tables
    write_tsv(DATA / "ASD_ALLOSTERIC_SOURCE_LINEAGE.tsv.gz", asd_lineage, list(asd_lineage[0]), True)
    asd_exclusions = [r for r in asd_lineage if not r["mapping_status"].startswith("included")]
    write_tsv(DATA / "ASD_STRUCTURE_CONFLICTS_AND_EXCLUSIONS.tsv", asd_exclusions, list(asd_lineage[0]))
    write_tsv(DATA / "ASD_ALLOSTERIC_PAIRS.tsv", asd_pairs, SOURCE_PAIR_FIELDS)
    write_tsv(DATA / "KLIFS_ORTHOSTERIC_EVIDENCE.tsv.gz", klifs_evidence, list(klifs_evidence[0]), True)
    write_tsv(DATA / "KLIFS_ORTHOSTERIC_SOURCE_PAIRS.tsv", klifs_pairs, SOURCE_PAIR_FIELDS)
    write_tsv(DATA / "KLIFS_STRUCTURE_CONFLICTS_AND_EXCLUSIONS.tsv", klifs_exclusions, list(klifs_exclusions[0]))
    write_tsv(DATA / "BRENDA_SUBSTRATE_EVIDENCE.tsv.gz", brenda_evidence, list(brenda_evidence[0]), True)
    brenda_pair_fields = SOURCE_PAIR_FIELDS + ["parameter_types", "quality_tiers"]
    write_tsv(DATA / "BRENDA_SUBSTRATE_SOURCE_PAIRS.tsv.gz", brenda_pairs, brenda_pair_fields, True)
    write_tsv(DATA / "ORTHOSTERIC_SOURCE_PAIR_LEDGER.tsv.gz", orth_source_pairs, SOURCE_PAIR_FIELDS, True)
    write_tsv(DATA / "ORTHOSTERIC_UNION_PAIRS.tsv.gz", orth_union, list(orth_union[0]), True)
    write_tsv(DATA / "CLASS_LABEL_CONFLICTS.tsv", conflicts, list(conflicts[0]))
    write_tsv(DATA / "UNCONTROLLED_ALLOSTERIC_ORTHOSTERIC.tsv.gz", uncontrolled, list(uncontrolled[0]), True)
    write_tsv(DATA / "UNCONTROLLED_PROTEIN_SUMMARY.tsv", protein_summary, list(protein_summary[0]))

    source_audit = [
        {"source": "ASD", "input_rows": asd_summary["raw_rows"], "selected_evidence_rows": asd_summary["included_evidence_rows"], "unique_pairs": asd_summary["unique_pairs"], "unique_proteins": asd_summary["unique_proteins"], "excluded_or_unresolved_rows": asd_summary["raw_rows"] - asd_summary["included_evidence_rows"], "selection_rule": "all rows; exact UniProt and wwPDB CCD full InChIKey; no exact-site filter"},
        {"source": "GtoPdb", "input_rows": gtop_summary["exact_protein_evidence_rows"] + gtop_summary["assembly_or_unresolved_source_rows"], "selected_evidence_rows": gtop_summary["exact_protein_evidence_rows"], "unique_pairs": gtop_summary["exact_unique_protein_ligand_pairs"], "unique_proteins": gtop_summary["exact_unique_proteins"], "excluded_or_unresolved_rows": gtop_summary["assembly_or_unresolved_source_rows"], "selection_rule": "endogenous ligand with exact chemistry and direct target UniProt; assemblies retained separately"},
        {"source": "KLIFS", "input_rows": klifs_summary["raw_rows"], "selected_evidence_rows": klifs_summary["included_evidence_rows"], "unique_pairs": klifs_summary["unique_pairs"], "unique_proteins": klifs_summary["unique_proteins"], "excluded_or_unresolved_rows": klifs_summary["excluded_rows"], "selection_rule": "ATP-site anchor; valid source full InChIKey or exact RDKit derivation from source SMILES; connectivity contradictions excluded"},
        {"source": "BRENDA", "input_rows": brenda_summary["all_parameter_rows"], "selected_evidence_rows": brenda_summary["eligible_exact_evidence_rows"], "unique_pairs": brenda_summary["unique_pairs"], "unique_proteins": brenda_summary["unique_proteins"], "excluded_or_unresolved_rows": brenda_summary["all_parameter_rows"] - brenda_summary["eligible_exact_evidence_rows"], "selection_rule": "exact_single_uniprot + substrate + exact_unique_full_inchikey; no parameter/quality filter"},
    ]
    write_tsv(DATA / "SOURCE_SELECTION_AUDIT.tsv", source_audit, list(source_audit[0]))

    counts = []
    def add(metric: str, value: int, unit: str, note: str) -> None:
        counts.append({"metric": metric, "value": value, "unit": unit, "note": note})
    add("asd_raw_rows", asd_summary["raw_rows"], "source_rows", "all ASD rows")
    add("asd_unique_allosteric_pairs", asd_summary["unique_pairs"], "pairs", "before class conflict removal")
    add("gtop_unique_orthosteric_pairs", gtop_summary["exact_unique_protein_ligand_pairs"], "pairs", "union of GtoP pairing and interaction lanes")
    add("klifs_unique_orthosteric_pairs", klifs_summary["unique_pairs"], "pairs", "after source conflict policy")
    add("brenda_unique_orthosteric_pairs", brenda_summary["unique_pairs"], "pairs", "all exact substrate parameter types")
    add("orthosteric_source_pair_rows", len(orth_source_pairs), "source_pairs", "source-specific rows before union")
    add("orthosteric_union_pairs", len(orth_union), "pairs", "unique UniProt/full-InChIKey union")
    add("class_label_conflicts", len(conflicts), "pairs", "excluded from both classes")
    add("uncontrolled_allosteric_pairs", sum(r["class_label"] == "allosteric" for r in uncontrolled), "pairs", "nonconflicting")
    add("uncontrolled_orthosteric_pairs", sum(r["class_label"] == "orthosteric" for r in uncontrolled), "pairs", "nonconflicting")
    add("uncontrolled_total_pairs", len(uncontrolled), "pairs", "two-class dataset; no decoys")
    add("uncontrolled_unique_proteins", len(protein_summary), "proteins", "union across both classes")
    add("future_protein_control_eligible_proteins", sum(r["has_both_classes"] == "true" for r in protein_summary), "proteins", "both labels after conflicts")
    add("uncontrolled_fingerprint_eligible_pairs", sum(r["fingerprint_eligible"] == "true" for r in uncontrolled), "pairs", "SMILES reproduces full InChIKey")
    add("uncontrolled_fingerprint_unresolved_pairs", sum(r["fingerprint_eligible"] == "false" for r in uncontrolled), "pairs", "pair retained by full InChIKey; exact SMILES unavailable")
    write_tsv(DATA / "COUNT_SUMMARY.tsv", counts, ["metric", "value", "unit", "note"])
    write_tsv(DATA / "INPUT_SNAPSHOT_HASHES.tsv", input_snapshots(), ["role", "path", "bytes", "sha256"])
    network_rows = [
        {"phase": "interactive_preflight", "request_count": 6, "cached": "false", "query_type": "server health, index mappings, and one exact InChIKey/CID test", "fuzzy_requests": 0, "note": "Pre-production schema verification; no data rows derived directly from these responses."},
        {"phase": "production_exact_structure_resolution", "request_count": json.loads(BRENDA_STRUCTURE_SUMMARY.read_text())["live_network_requests"], "cached": "true", "query_type": "exact InChIKey terms and exact CID terms", "fuzzy_requests": 0, "note": "Full responses cached in one gzip JSONL event ledger."},
    ]
    write_tsv(DATA / "NETWORK_REQUEST_ACCOUNTING.tsv", network_rows, list(network_rows[0]))

    structure_summary = json.loads(BRENDA_STRUCTURE_SUMMARY.read_text())
    summary = {
        "build_version": "ALLOSTERIC_ORTHOSTERIC_UNCONTROLLED_1.0",
        "rdkit_version": rdBase.rdkitVersion,
        "asd": asd_summary,
        "gtop": gtop_summary,
        "klifs": klifs_summary,
        "brenda": brenda_summary,
        "brenda_structure_resolution": structure_summary,
        "orthosteric_source_pair_rows": len(orth_source_pairs),
        "orthosteric_union_pairs": len(orth_union),
        "conflicts": len(conflicts),
        "uncontrolled": {
            "rows": len(uncontrolled),
            "allosteric": sum(r["class_label"] == "allosteric" for r in uncontrolled),
            "orthosteric": sum(r["class_label"] == "orthosteric" for r in uncontrolled),
            "unique_proteins": len(protein_summary),
            "future_protein_control_eligible_proteins": sum(r["has_both_classes"] == "true" for r in protein_summary),
            "fingerprint_eligible": sum(r["fingerprint_eligible"] == "true" for r in uncontrolled),
            "fingerprint_unresolved": sum(r["fingerprint_eligible"] == "false" for r in uncontrolled),
            "decoy_rows": 0,
        },
        "protein_controlled_built": False,
        "fully_controlled_built": False,
    }
    write_json(DATA / "BUILD_SUMMARY.json", summary)
    write_documents(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
