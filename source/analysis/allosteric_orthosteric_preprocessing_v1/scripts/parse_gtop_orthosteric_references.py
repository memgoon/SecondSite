#!/usr/bin/env python3
"""Parse GtoP endogenous ligand references without flattening assemblies.

This is an append-only correction to the historical parser used to create
``hard_orthosteric_reference_pairs_loaded.tsv``.  The historical chemistry
table remains a frozen exact ligand-chemistry crosswalk, but target membership
is reconstructed from the GtoP source tables.  A row is emitted as an exact
protein--ligand reference only when GtoP supplies a direct target UniProt
accession.  Subunit-only targets are retained in a separate scope ledger and
are never assigned to arbitrary components.

No network service is contacted by this script.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
GTOP_DIR = Path("/shared_data/11.HS_allostery/Data/Orthosteric_References/GtoP")
HARD_REFERENCE = Path(
    "/shared_data/11.HS_allostery/18.Structure_known/"
    "4.Evaluate_Orthosteric_LowScore_Enrichment/"
    "hard_orthosteric_reference_pairs_loaded.tsv"
)

IK_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
UP_RE = re.compile(r"^[A-Z0-9]{6,10}(?:-[0-9]+)?$")
TRUE = {"true", "1", "yes", "y", "t"}
NA = "NA"


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def canonical_uniprot(value: str) -> str:
    value = (value or "").strip().upper()
    if not UP_RE.fullmatch(value) or value.lower() == "no_uniprot_id":
        return ""
    return value.split("-")[0]


def valid_ik(value: str) -> bool:
    return bool(IK_RE.fullmatch((value or "").strip().upper()))


def split_tokens(value: str) -> list[str]:
    return [x.strip() for x in re.split(r"[;,| ]+", value or "") if x.strip()]


def parse_component_accessions(value: str) -> list[dict[str, str]]:
    """Parse GtoP ``component_id:accession`` tokens without promoting them."""
    out: list[dict[str, str]] = []
    for token in split_tokens(value):
        if ":" in token:
            component_id, accession_raw = token.split(":", 1)
        else:
            component_id, accession_raw = "", token
        accession = canonical_uniprot(accession_raw)
        out.append(
            {
                "component_id": component_id.strip() or NA,
                "accession_raw": accession_raw.strip() or NA,
                "uniprot": accession or NA,
            }
        )
    return out


def read_gtop(path: Path) -> tuple[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        version = next(handle).strip().strip('"')
        return version, list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str], gzip_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                import io
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def load_exact_chemistry() -> dict[str, dict[str, str]]:
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    with HARD_REFERENCE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row["Source"].startswith("GtoP"):
                continue
            ik = row["InChIKey"].strip().upper()
            ligand_id = row["Ligand_ID"].strip()
            if ligand_id and valid_ik(ik):
                candidates[ligand_id].append(row)

    out: dict[str, dict[str, str]] = {}
    for ligand_id, rows in candidates.items():
        inchikeys = {r["InChIKey"].strip().upper() for r in rows}
        if len(inchikeys) != 1:
            raise ValueError(f"GtoP ligand {ligand_id} has non-unique full InChIKeys: {inchikeys}")
        def joined(field: str) -> str:
            values = sorted({r.get(field, "").strip() for r in rows if r.get(field, "").strip()})
            return ";".join(values) if values else NA
        out[ligand_id] = {
            "ligand_name": joined("Ligand_Name"),
            "canonical_smiles": joined("SMILES"),
            "full_inchikey": next(iter(inchikeys)),
            "pubchem_cids": joined("PubChem_CID"),
            "pubchem_sids": joined("PubChem_SID"),
        }
    return out


def build() -> dict:
    chemistry = load_exact_chemistry()
    pair_version, pair_rows = read_gtop(GTOP_DIR / "endogenous_ligand_pairings_all.tsv")
    int_version, int_rows = read_gtop(GTOP_DIR / "interactions_all.tsv")

    evidence: list[dict] = []
    scope_ledger: list[dict] = []
    missing_chemistry = {"pairing": 0, "interaction": 0}

    def emit_exact(source: str, source_row: int, row: dict, up_raw: str, chem: dict) -> None:
        up = canonical_uniprot(up_raw)
        if not up:
            raise ValueError(f"Invalid direct GtoP UniProt {up_raw!r} at {source}:{source_row}")
        evidence.append(
            {
                "evidence_id": stable_id("GTPEV", source, source_row, up, row.get("Ligand ID", "")),
                "source_database": "GtoPdb",
                "source_release": pair_version.removeprefix("# "),
                "source_lane": source,
                "source_row_number": source_row,
                "target_id": row.get("Target ID", "").strip() or NA,
                "target_name": (row.get("Target Name") or row.get("Target") or "").strip() or NA,
                "target_species": row.get("Target Species", "").strip() or NA,
                "target_scope": "exact_single_protein",
                "uniprot_raw": up_raw.strip(),
                "uniprot": up,
                "ligand_id": row.get("Ligand ID", "").strip(),
                "ligand_name": chem["ligand_name"],
                "canonical_smiles": chem["canonical_smiles"],
                "full_inchikey": chem["full_inchikey"],
                "connectivity_key": chem["full_inchikey"][:14],
                "pubchem_cids": chem["pubchem_cids"],
                "pubchem_sids": chem["pubchem_sids"],
                "class_label": "orthosteric",
                "evidence_subtype": "GtoP_endogenous_ligand_reference",
                "claim_limit": "GtoP endogenous target-ligand reference; no allostery or mechanism inference.",
            }
        )

    for source_row, row in enumerate(pair_rows, start=3):
        ligand_id = row.get("Ligand ID", "").strip()
        chem = chemistry.get(ligand_id)
        if not chem:
            missing_chemistry["pairing"] += 1
            continue
        direct = split_tokens(row.get("Target UniProt ID", ""))
        if direct:
            for up_raw in direct:
                emit_exact("GtoP_endogenous_pairing", source_row, row, up_raw, chem)
            continue
        components = parse_component_accessions(row.get("Target Subunit UniProt ID", ""))
        scope = "assembly_only" if components else "unresolved_target_accession"
        scope_ledger.append(
            {
                "scope_record_id": stable_id("GTPSCOPE", "pairing", source_row, row.get("Target ID", ""), ligand_id),
                "source_database": "GtoPdb",
                "source_release": pair_version.removeprefix("# "),
                "source_lane": "GtoP_endogenous_pairing",
                "source_row_number": source_row,
                "target_id": row.get("Target ID", "").strip() or NA,
                "target_name": row.get("Target Name", "").strip() or NA,
                "target_species": row.get("Target Species", "").strip() or NA,
                "ligand_id": ligand_id,
                "ligand_name": chem["ligand_name"],
                "full_inchikey": chem["full_inchikey"],
                "target_scope": scope,
                "component_ids": ";".join(x["component_id"] for x in components) or NA,
                "component_accessions_raw": ";".join(x["accession_raw"] for x in components) or NA,
                "component_uniprots": ";".join(x["uniprot"] for x in components) or NA,
                "exact_protein_pair_emitted": "false",
                "exclusion_reason": "assembly_relationship_not_assigned_to_arbitrary_subunit" if components else "target_accession_unavailable",
            }
        )

    for source_row, row in enumerate(int_rows, start=3):
        if row.get("Endogenous", "").strip().strip('"').lower() not in TRUE:
            continue
        ligand_id = row.get("Ligand ID", "").strip()
        chem = chemistry.get(ligand_id)
        if not chem:
            missing_chemistry["interaction"] += 1
            continue
        direct = split_tokens(row.get("Target UniProt ID", ""))
        if direct:
            for up_raw in direct:
                emit_exact("GtoP_endogenous_interaction", source_row, row, up_raw, chem)
            continue
        subunit_ids = split_tokens(row.get("Target Subunit IDs", ""))
        scope_ledger.append(
            {
                "scope_record_id": stable_id("GTPSCOPE", "interaction", source_row, row.get("Target ID", ""), ligand_id),
                "source_database": "GtoPdb",
                "source_release": int_version.removeprefix("# "),
                "source_lane": "GtoP_endogenous_interaction",
                "source_row_number": source_row,
                "target_id": row.get("Target ID", "").strip() or NA,
                "target_name": row.get("Target", "").strip() or NA,
                "target_species": row.get("Target Species", "").strip() or NA,
                "ligand_id": ligand_id,
                "ligand_name": chem["ligand_name"],
                "full_inchikey": chem["full_inchikey"],
                "target_scope": "assembly_only_accessions_unavailable" if subunit_ids else "unresolved_target_accession",
                "component_ids": ";".join(subunit_ids) or NA,
                "component_accessions_raw": NA,
                "component_uniprots": NA,
                "exact_protein_pair_emitted": "false",
                "exclusion_reason": "assembly_relationship_not_assigned_to_arbitrary_subunit" if subunit_ids else "target_accession_unavailable",
            }
        )

    evidence.sort(key=lambda r: (r["source_lane"], r["uniprot"], r["full_inchikey"], int(r["source_row_number"])))
    scope_ledger.sort(key=lambda r: (r["source_lane"], int(r["source_row_number"])))
    pair_agg: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in evidence:
        pair_agg[(row["source_lane"], row["uniprot"], row["full_inchikey"])].append(row)
    pairs = []
    for (source, up, ik), rows in sorted(pair_agg.items()):
        def joined(field: str) -> str:
            values = sorted({str(r[field]) for r in rows if str(r[field]) not in {"", NA}})
            return ";".join(values) if values else NA
        pairs.append(
            {
                "source_pair_id": stable_id("GTPPAIR", source, up, ik),
                "source_database": "GtoPdb",
                "source_lane": source,
                "uniprot": up,
                "full_inchikey": ik,
                "connectivity_key": ik[:14],
                "canonical_smiles": joined("canonical_smiles"),
                "ligand_ids": joined("ligand_id"),
                "ligand_names": joined("ligand_name"),
                "target_ids": joined("target_id"),
                "target_names": joined("target_name"),
                "target_species": joined("target_species"),
                "pubchem_cids": joined("pubchem_cids"),
                "pubchem_sids": joined("pubchem_sids"),
                "evidence_ids": joined("evidence_id"),
                "n_evidence_rows": len(rows),
                "class_label": "orthosteric",
                "evidence_subtype": "GtoP_endogenous_ligand_reference",
            }
        )

    evidence_fields = [
        "evidence_id", "source_database", "source_release", "source_lane", "source_row_number",
        "target_id", "target_name", "target_species", "target_scope", "uniprot_raw", "uniprot",
        "ligand_id", "ligand_name", "canonical_smiles", "full_inchikey", "connectivity_key",
        "pubchem_cids", "pubchem_sids", "class_label", "evidence_subtype", "claim_limit",
    ]
    pair_fields = [
        "source_pair_id", "source_database", "source_lane", "uniprot", "full_inchikey",
        "connectivity_key", "canonical_smiles", "ligand_ids", "ligand_names", "target_ids",
        "target_names", "target_species", "pubchem_cids", "pubchem_sids", "evidence_ids",
        "n_evidence_rows", "class_label", "evidence_subtype",
    ]
    scope_fields = [
        "scope_record_id", "source_database", "source_release", "source_lane", "source_row_number",
        "target_id", "target_name", "target_species", "ligand_id", "ligand_name", "full_inchikey",
        "target_scope", "component_ids", "component_accessions_raw", "component_uniprots",
        "exact_protein_pair_emitted", "exclusion_reason",
    ]
    write_tsv(ROOT / "data/GTP_ORTHOSTERIC_EVIDENCE.tsv.gz", evidence, evidence_fields, True)
    write_tsv(ROOT / "data/GTP_ORTHOSTERIC_SOURCE_PAIRS.tsv", pairs, pair_fields)
    write_tsv(ROOT / "data/GTP_ASSEMBLY_OR_UNRESOLVED_TARGET_LEDGER.tsv", scope_ledger, scope_fields)

    summary = {
        "parser_version": "gtop_exact_protein_parser_v1.0",
        "network_requests": 0,
        "source_versions": {"pairings": pair_version, "interactions": int_version},
        "exact_chemistry_ligands": len(chemistry),
        "exact_protein_evidence_rows": len(evidence),
        "exact_source_pairs": len(pairs),
        "exact_unique_protein_ligand_pairs": len({(r["uniprot"], r["full_inchikey"]) for r in pairs}),
        "exact_unique_proteins": len({r["uniprot"] for r in pairs}),
        "assembly_or_unresolved_source_rows": len(scope_ledger),
        "assembly_source_rows": sum(r["target_scope"].startswith("assembly") for r in scope_ledger),
        "unresolved_target_source_rows": sum(r["target_scope"] == "unresolved_target_accession" for r in scope_ledger),
        "source_rows_without_exact_ligand_chemistry": missing_chemistry,
        "rules": {
            "direct_target_uniprot": "emit exact protein-level pair",
            "subunit_only_target": "retain assembly scope; do not emit protein-level pair",
            "component_token": "parse component_id:accession but never promote solely from membership",
        },
    }
    (ROOT / "data/GTP_PARSER_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def verify_existing() -> dict:
    summary_path = ROOT / "data/GTP_PARSER_SUMMARY.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    with gzip.open(ROOT / "data/GTP_ORTHOSTERIC_EVIDENCE.tsv.gz", "rt", encoding="utf-8", newline="") as h:
        evidence = list(csv.DictReader(h, delimiter="\t"))
    with (ROOT / "data/GTP_ORTHOSTERIC_SOURCE_PAIRS.tsv").open(encoding="utf-8", newline="") as h:
        pairs = list(csv.DictReader(h, delimiter="\t"))
    with (ROOT / "data/GTP_ASSEMBLY_OR_UNRESOLVED_TARGET_LEDGER.tsv").open(encoding="utf-8", newline="") as h:
        scope = list(csv.DictReader(h, delimiter="\t"))
    checks = {
        "evidence_count": len(evidence) == summary["exact_protein_evidence_rows"],
        "source_pair_count": len(pairs) == summary["exact_source_pairs"],
        "scope_count": len(scope) == summary["assembly_or_unresolved_source_rows"],
        "all_uniprots_valid": all(canonical_uniprot(r["uniprot"]) == r["uniprot"] for r in evidence),
        "no_component_prefix_in_uniprot": all(":" not in r["uniprot"] for r in evidence),
        "scope_never_emitted": all(r["exact_protein_pair_emitted"] == "false" for r in scope),
        "labels_orthosteric": all(r["class_label"] == "orthosteric" for r in evidence + pairs),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"status": "PASS", "checks": checks, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    result = verify_existing() if args.verify_existing else build()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
