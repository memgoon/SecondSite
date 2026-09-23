#!/usr/bin/env python3
"""Link exact allosteric and orthosteric reference observations.

This checkpoint does not repeat checkpoint-4 residue mapping.  It links frozen
source records to checkpoint-4 observation IDs and keeps broad pair-role
annotations separate from exact structural-site labels.

Exact allosteric rule
---------------------
AlloBench UniProt + PDB + CCD + deposited ligand chain + ligand author residue
must match, and at least one chain-qualified AlloBench allosteric-site author
residue must overlap the BioLiP binding residues for that receptor chain.

Exact orthosteric rule
----------------------
An observation must belong to one of the 247 manually retained BioLiP primary-
ligand pairs.  Exact observation record IDs are resolved to frozen source
ordinals through the prior 989,058-row manifest and then checked against the
new checkpoint-4 master.  Broad GtoP/KLIFS/BRENDA/ChEBI/KEGG/UniProt role
annotations remain pair-level metadata and are not promoted to exact sites.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import duckdb
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
REPORTS = PACKAGE / "reports"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

CP2 = VALIDATION / "CHECKPOINT2_VALIDATION.json"
CP4 = VALIDATION / "CHECKPOINT4_VALIDATION.json"
MASTER = DATA / "BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz"
ELIGIBILITY = DATA / "BIOLIP_SITE_MAPPING_ELIGIBILITY.tsv.gz"
ACTIVE_ORTHO = DATA / "ORTHOSTERIC_PAIR_MASTER_AUDITED.tsv.gz"

ALLOBENCH = ROOT / "data/allobench/AlloBench.csv"
STRICT_PAIRS = ROOT / (
    "analysis/allosteric_orthosteric_preprocessing_v1_strict_pair_multisite_audit/"
    "data/FILTERED_STRICT_ORTHOSTERIC_PAIRS.tsv"
)
AUDIT_LEDGER = ROOT / (
    "analysis/allosteric_orthosteric_preprocessing_v1_strict_pair_multisite_audit/"
    "data/STRICT_PAIR_MULTISITE_AUDIT_LEDGER.tsv"
)
OLD_MANIFEST = ROOT / "analysis/structure_known_candidate_rebuild/data/exact_biolip_manifest.parquet"
OLD_BRIDGE = ROOT / "analysis/structure_known_candidate_rebuild/data/exact_anchor_label_bridge.parquet"

PILOT_INPUTS = {
    "pilot1_v1_1": (
        ROOT / (
            "analysis/allosteric_orthosteric_preprocessing_v1_biolip_primary_ligand_pilot/"
            "data/PILOT_20_BIOLIP_EXACT_OBSERVATIONS.tsv"
        ),
        "pilot_id",
    ),
    "pilot2": (
        ROOT / (
            "analysis/allosteric_orthosteric_preprocessing_v1_biolip_primary_ligand_pilot2_100/"
            "data/PILOT2_100_BIOLIP_EXACT_OBSERVATIONS.tsv"
        ),
        "pilot2_id",
    ),
    "pilot3": (
        ROOT / (
            "analysis/allosteric_orthosteric_preprocessing_v1_biolip_primary_ligand_pilot3_200/"
            "data/PILOT3_200_BIOLIP_EXACT_OBSERVATIONS.tsv"
        ),
        "pilot3_id",
    ),
    "pilot4": (
        ROOT / (
            "analysis/allosteric_orthosteric_preprocessing_v1_biolip_primary_ligand_pilot4_completion_121/"
            "data/PILOT4_121_BIOLIP_EXACT_OBSERVATIONS.tsv"
        ),
        "pilot4_id",
    ),
}

ALLOBENCH_REGISTRY = DATA / "CHECKPOINT5_ALLOBENCH_SOURCE_REGISTRY.tsv.gz"
ALLO_LINKS = DATA / "CHECKPOINT5_ALLOSTERIC_EXACT_LINKS.tsv.gz"
ALLO_OBSERVATIONS = DATA / "CHECKPOINT5_EXACT_ALLOSTERIC_OBSERVATIONS.tsv.gz"
ALLO_RECONCILIATION = DATA / "CHECKPOINT5_LEGACY_ALLOSTERIC_RECONCILIATION.tsv.gz"
ORTHO_LINKS = DATA / "CHECKPOINT5_ORTHOSTERIC_EXACT_LINKS.tsv.gz"
ORTHO_PAIR_COVERAGE = DATA / "CHECKPOINT5_ORTHOSTERIC_PAIR_COVERAGE.tsv.gz"
REFERENCE_CONFLICTS = DATA / "CHECKPOINT5_EXACT_REFERENCE_CONFLICTS.tsv.gz"
REFERENCE_OBSERVATIONS = DATA / "BIOLIP_EXACT_SITE_REFERENCE_OBSERVATIONS.tsv.gz"
COUNT_SUMMARY = DATA / "CHECKPOINT5_COUNT_SUMMARY.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT5_INPUT_HASHES.tsv"
REPORT = REPORTS / "CHECKPOINT5_EXACT_SITE_REFERENCE_LINKAGE.md"
BUILD_SUMMARY = VALIDATION / "CHECKPOINT5_BUILD_SUMMARY.json"

ALLO_TOKEN = re.compile(r"^([^-]+)-([A-Za-z0-9]+)-(-?\d+)([A-Za-z]?)$")
AUTH_TOKEN = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)$")
ATOMIC_RESNUM = re.compile(r"^-?\d+[A-Za-z]?$", re.IGNORECASE)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, value: str, length: int = 20) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def base_accession(value: object) -> str:
    return str(value or "").strip().upper().split("-", 1)[0]


def truth(value: object) -> bool:
    return str(value or "").strip().lower() == "true"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_frame(frame: pd.DataFrame, path: Path, compressed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if compressed:
        with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=6, mtime=0) as raw:
            frame.to_csv(raw, sep="\t", index=False, lineterminator="\n")
    else:
        frame.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
    os.replace(temporary, path)


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def join_values(values: Iterable[object]) -> str:
    return ";".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def parse_allosteric_site(raw: object) -> Tuple[List[Tuple[str, int, str, str]], List[str], str]:
    try:
        parsed = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        return [], [], "unparseable_literal"
    if not isinstance(parsed, (list, tuple)):
        return [], [], "not_a_list"
    residues: List[Tuple[str, int, str, str]] = []
    invalid: List[str] = []
    for item in parsed:
        token = str(item).strip()
        match = ALLO_TOKEN.fullmatch(token)
        if not match:
            invalid.append(token)
            continue
        residues.append(
            (match.group(1).strip(), int(match.group(3)), match.group(4).upper(), match.group(2).upper())
        )
    if not residues:
        status = "no_parseable_residue"
    elif invalid:
        status = f"partially_parsed_{len(residues)}_valid_{len(invalid)}_invalid"
    else:
        status = f"parsed_{len(residues)}"
    return residues, invalid, status


def parse_active_site(raw: object) -> Tuple[List[int], str]:
    try:
        parsed = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        return [], "unparseable_literal"
    if not isinstance(parsed, (list, tuple)):
        return [], "not_a_list"
    positions = sorted({int(value) for value in parsed if isinstance(value, int) and int(value) > 0})
    return positions, f"parsed_{len(positions)}" if positions else "no_positive_integer_position"


def parse_biolip_author_site(raw: object) -> Dict[Tuple[int, str], str]:
    result: Dict[Tuple[int, str], str] = {}
    for token in str(raw or "").split():
        match = AUTH_TOKEN.fullmatch(token)
        if match:
            result[(int(match.group(2)), match.group(3).upper())] = match.group(1).upper()
    return result


def normalize_site_overlap(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "no":
        return "distinct_from_orthosteric_site"
    if text == "yes":
        return "overlaps_orthosteric_site"
    return "source_overlap_unspecified"


def load_checkpoint(path: Path, checkpoint: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if value.get("checkpoint") != checkpoint or value.get("status") != "validated":
        raise RuntimeError(f"checkpoint {checkpoint} is not validated")
    for record in value.get("outputs", {}).values():
        output = Path(record["path"])
        if not output.is_file() or sha256(output) != record["sha256"]:
            raise RuntimeError(f"checkpoint-{checkpoint} output hash mismatch: {output}")
    return value


def build_allobench_registry() -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, int, str, str]]]]:
    source = pd.read_csv(ALLOBENCH, dtype=str, keep_default_na=False)
    rows: List[dict] = []
    parsed_lookup: Dict[str, List[Tuple[str, int, str, str]]] = {}
    for index, row in source.iterrows():
        uniprot = base_accession(row.get("pdb_uniprot"))
        pdb = str(row.get("allosteric_pdb") or "").strip().lower()
        ccd = str(row.get("modulator_alias") or "").strip().upper()
        ligand_chain = str(row.get("modulator_chain") or "").strip()
        ligand_resnum = str(row.get("modulator_resi") or "").strip()
        source_id = stable_id(
            "ABREF_",
            "|".join([str(index), uniprot, pdb, ccd, ligand_chain, ligand_resnum]),
        )
        site, invalid, site_status = parse_allosteric_site(row.get("allosteric_site_residue"))
        active, active_status = parse_active_site(row.get("active_site_residue"))
        parsed_lookup[source_id] = site
        atomic = bool(
            uniprot
            and re.fullmatch(r"[0-9a-z]{4}", pdb)
            and ccd
            and ligand_chain
            and ATOMIC_RESNUM.fullmatch(ligand_resnum)
            and ";" not in ccd
            and "," not in ccd
            and ";" not in ligand_chain
            and "," not in ligand_chain
        )
        rows.append(
            {
                "allobench_source_id": source_id,
                "allobench_row_index": int(index),
                "target_id": str(row.get("target_id") or "").strip(),
                "uniprot": uniprot,
                "pdb_id": pdb,
                "ligand_ccd": ccd,
                "ligand_chain": ligand_chain,
                "ligand_auth_seq_id": ligand_resnum,
                "ligand_metadata_status": "atomic_exact_instance_metadata" if atomic else "non_atomic_or_incomplete_instance_metadata",
                "allosteric_site_residues_raw": str(row.get("allosteric_site_residue") or ""),
                "allosteric_site_residues_normalized": ";".join(
                    f"{chain}:{name}:{number}{icode}" for chain, number, icode, name in site
                ),
                "allosteric_site_chains": join_values(chain for chain, *_ in site),
                "allosteric_site_parse_status": site_status,
                "invalid_allosteric_site_tokens": ";".join(invalid),
                "active_site_residues_raw": str(row.get("active_site_residue") or ""),
                "active_site_uniprot_positions": ";".join(map(str, active)),
                "active_site_parse_status": active_status,
                "source_site_relation": normalize_site_overlap(row.get("site_overlap")),
                "source_site_overlap_raw": str(row.get("site_overlap") or "").strip(),
                "modulator_class": str(row.get("modulator_class") or "").strip(),
                "modulator_feature": str(row.get("modulator_feature") or "").strip(),
                "pubmed_id": str(row.get("pubmed_id") or "").strip(),
                "source_link_status": "pending",
                "candidate_observation_links": 0,
                "site_supported_observation_links": 0,
                "eligible_exact_observation_links": 0,
            }
        )
    return pd.DataFrame(rows), parsed_lookup


def build_allosteric_links(
    registry: pd.DataFrame,
    parsed_lookup: Mapping[str, Sequence[Tuple[str, int, str, str]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    atomic = registry.loc[
        registry["ligand_metadata_status"].eq("atomic_exact_instance_metadata"),
        [
            "allobench_source_id", "allobench_row_index", "target_id", "uniprot", "pdb_id",
            "ligand_ccd", "ligand_chain", "ligand_auth_seq_id", "source_site_relation",
            "source_site_overlap_raw", "pubmed_id",
        ],
    ].copy()
    con = duckdb.connect(database=":memory:")
    con.register("allobench_atomic", atomic)
    master = (
        "read_csv_auto(?, delim='\\t', header=true, compression='gzip', "
        "all_varchar=true, sample_size=-1)"
    )
    eligible = master
    query = f"""
        WITH master AS (
          SELECT CAST(source_ordinal AS BIGINT) AS source_ordinal, observation_id,
                 lower(pdb_id) AS pdb_id, receptor_chain, resolution,
                 upper(ligand_ccd) AS ligand_ccd, ligand_chain, ligand_auth_seq_id,
                 split_part(upper(uniprot_resolved), '-', 1) AS uniprot_resolved,
                 full_inchikey, connectivity_key, binding_residues_auth_raw,
                 binding_uniprot_positions, active_orthosteric_pair_reference,
                 pubmed_id AS biolip_pubmed_id
          FROM {master}
        ), eligibility AS (
          SELECT observation_id, final_site_mapping_usable, final_eligibility_reason
          FROM {eligible}
        )
        SELECT a.*, m.*, e.final_site_mapping_usable, e.final_eligibility_reason
        FROM allobench_atomic a
        INNER JOIN master m
          ON a.pdb_id = m.pdb_id
         AND a.ligand_ccd = m.ligand_ccd
         AND a.ligand_chain = m.ligand_chain
         AND a.ligand_auth_seq_id = m.ligand_auth_seq_id
         AND a.uniprot = m.uniprot_resolved
        LEFT JOIN eligibility e USING (observation_id)
        ORDER BY a.allobench_row_index, m.source_ordinal
    """
    candidates = con.execute(query, [str(MASTER), str(ELIGIBILITY)]).fetchdf()
    con.close()

    link_rows: List[dict] = []
    for row in candidates.itertuples(index=False):
        site = parsed_lookup[str(row.allobench_source_id)]
        same_chain = {
            (number, icode): THREE_TO_ONE.get(name, "")
            for chain, number, icode, name in site
            if chain == str(row.receptor_chain)
        }
        biolip = parse_biolip_author_site(row.binding_residues_auth_raw)
        overlap_positions = sorted(set(same_chain) & set(biolip))
        union_positions = set(same_chain) | set(biolip)
        letter_matches = sum(
            bool(same_chain[position]) and same_chain[position] == biolip.get(position)
            for position in overlap_positions
        )
        if not site:
            status = "allobench_site_unparseable"
        elif not same_chain:
            status = "exact_instance_other_receptor_chain"
        elif overlap_positions:
            status = "exact_instance_site_residue_supported"
        else:
            status = "exact_instance_same_chain_no_site_residue_overlap"
        usable = truth(row.final_site_mapping_usable)
        exact = status == "exact_instance_site_residue_supported"
        link_rows.append(
            {
                "allosteric_link_id": stable_id(
                    "ALLINK_", f"{row.allobench_source_id}|{row.observation_id}"
                ),
                "allobench_source_id": row.allobench_source_id,
                "allobench_row_index": int(row.allobench_row_index),
                "target_id": row.target_id,
                "source_ordinal": int(row.source_ordinal),
                "observation_id": row.observation_id,
                "pdb_id": row.pdb_id,
                "receptor_chain": row.receptor_chain,
                "uniprot": row.uniprot_resolved,
                "ligand_ccd": row.ligand_ccd,
                "ligand_chain": row.ligand_chain,
                "ligand_auth_seq_id": row.ligand_auth_seq_id,
                "full_inchikey": row.full_inchikey,
                "connectivity_key": row.connectivity_key,
                "binding_residues_auth_raw": row.binding_residues_auth_raw,
                "binding_uniprot_positions": row.binding_uniprot_positions,
                "allobench_same_chain_site_positions": ";".join(
                    f"{number}{icode}" for number, icode in sorted(same_chain)
                ),
                "author_position_overlap": ";".join(
                    f"{number}{icode}" for number, icode in overlap_positions
                ),
                "author_position_overlap_count": len(overlap_positions),
                "author_position_jaccard": (
                    len(overlap_positions) / len(union_positions) if union_positions else 0.0
                ),
                "overlap_residue_letter_matches": int(letter_matches),
                "link_status": status,
                "checkpoint4_mapping_usable": usable,
                "checkpoint4_eligibility_reason": row.final_eligibility_reason,
                "eligible_exact_allosteric": bool(exact and usable),
                "source_site_relation": row.source_site_relation,
                "source_site_overlap_raw": row.source_site_overlap_raw,
                "source_pubmed_id": row.pubmed_id,
                "biolip_pubmed_id": row.biolip_pubmed_id,
                "active_broad_orthosteric_pair_reference": truth(
                    row.active_orthosteric_pair_reference
                ),
            }
        )
    links = pd.DataFrame(link_rows)

    by_source = {key: group for key, group in links.groupby("allobench_source_id")}
    for index, row in registry.iterrows():
        source_id = row["allobench_source_id"]
        if row["ligand_metadata_status"] != "atomic_exact_instance_metadata":
            status = "non_atomic_or_incomplete_instance_metadata"
            counts = (0, 0, 0)
        elif not parsed_lookup[source_id]:
            status = "allosteric_site_unparseable"
            counts = (0, 0, 0)
        elif source_id not in by_source:
            status = "exact_instance_not_found_for_resolved_uniprot"
            counts = (0, 0, 0)
        else:
            group = by_source[source_id]
            supported = group["link_status"].eq("exact_instance_site_residue_supported")
            eligible_supported = group["eligible_exact_allosteric"].astype(bool)
            counts = (len(group), int(supported.sum()), int(eligible_supported.sum()))
            if eligible_supported.any():
                status = "eligible_exact_allosteric_source"
            elif supported.any():
                status = "site_supported_but_checkpoint4_mapping_ineligible"
            else:
                status = "instance_found_but_site_not_supported"
        registry.loc[index, "source_link_status"] = status
        registry.loc[index, "candidate_observation_links"] = counts[0]
        registry.loc[index, "site_supported_observation_links"] = counts[1]
        registry.loc[index, "eligible_exact_observation_links"] = counts[2]

    supported = links.loc[
        links["link_status"].eq("exact_instance_site_residue_supported")
    ].copy()
    observation_rows: List[dict] = []
    for observation_id, group in supported.groupby("observation_id", sort=True):
        first = group.sort_values("allobench_row_index").iloc[0]
        relations = sorted(set(group["source_site_relation"]))
        relation = relations[0] if len(relations) == 1 else "mixed_source_site_relation"
        observation_rows.append(
            {
                "observation_id": observation_id,
                "source_ordinal": int(first.source_ordinal),
                "pdb_id": first.pdb_id,
                "receptor_chain": first.receptor_chain,
                "uniprot": first.uniprot,
                "ligand_ccd": first.ligand_ccd,
                "ligand_chain": first.ligand_chain,
                "ligand_auth_seq_id": first.ligand_auth_seq_id,
                "full_inchikey": first.full_inchikey,
                "connectivity_key": first.connectivity_key,
                "binding_residues_auth_raw": first.binding_residues_auth_raw,
                "binding_uniprot_positions": first.binding_uniprot_positions,
                "allobench_source_ids": join_values(group["allobench_source_id"]),
                "allobench_target_ids": join_values(group["target_id"]),
                "source_pubmed_ids": join_values(group["source_pubmed_id"]),
                "source_site_relation": relation,
                "source_site_relation_values": ";".join(relations),
                "n_allobench_source_rows": int(group["allobench_source_id"].nunique()),
                "max_author_position_overlap_count": int(group["author_position_overlap_count"].max()),
                "max_author_position_jaccard": float(group["author_position_jaccard"].max()),
                "checkpoint4_mapping_usable": bool(group["checkpoint4_mapping_usable"].all()),
                "checkpoint4_eligibility_reasons": join_values(group["checkpoint4_eligibility_reason"]),
                "eligible_exact_allosteric": bool(group["eligible_exact_allosteric"].any()),
                "active_broad_orthosteric_pair_reference": bool(
                    group["active_broad_orthosteric_pair_reference"].any()
                ),
            }
        )
    observations = pd.DataFrame(observation_rows)
    return links, observations


def load_pilot_observations() -> pd.DataFrame:
    frames = []
    for stage, (path, id_column) in PILOT_INPUTS.items():
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        frame = frame.rename(columns={id_column: "source_population_id"})
        frame["source_stage"] = stage
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_orthosteric_links() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(STRICT_PAIRS, sep="\t", dtype=str, keep_default_na=False)
    audit = pd.read_csv(AUDIT_LEDGER, sep="\t", dtype=str, keep_default_na=False)
    audit = audit[
        ["audit_id", "manual_site_interpretation", "manual_disposition", "manual_rationale", "claim_limit"]
    ].drop_duplicates("audit_id")
    pairs = pairs.merge(
        audit, left_on="multisite_audit_id", right_on="audit_id", how="left", validate="one_to_one"
    )
    if len(pairs) != 247 or pairs["cumulative_pair_id"].nunique() != 247:
        raise RuntimeError("strict orthosteric pair contract changed")
    observations = load_pilot_observations()
    links = pairs.merge(
        observations,
        on=["source_stage", "source_population_id", "uniprot"],
        how="inner",
        validate="many_to_many",
        suffixes=("_pair", "_observation"),
    )
    links = links.loc[
        links["biolip_ccd"].str.upper().eq(links["ligand_ccd"].str.upper())
    ].copy()
    if links["cumulative_pair_id"].nunique() != 247:
        raise RuntimeError("not all retained orthosteric pairs have exact observations")
    if links["record_id"].duplicated().any():
        raise RuntimeError("an exact orthosteric observation maps to multiple retained pairs")

    con = duckdb.connect(database=":memory:")
    record_keys = links[["record_id"]].drop_duplicates()
    con.register("record_keys", record_keys)
    old_map = con.execute(
        """
        SELECT m.record_id, CAST(m.source_ordinal AS BIGINT) AS source_ordinal,
               m.pdb AS old_pdb, m.receptor_chain AS old_receptor_chain,
               m.uniprot AS old_uniprot, m.ligand_ccd AS old_ligand_ccd,
               m.ligand_chain AS old_ligand_chain, m.ligand_resnum AS old_ligand_resnum
        FROM read_parquet(?) m INNER JOIN record_keys USING(record_id)
        """,
        [str(OLD_MANIFEST)],
    ).fetchdf()
    if len(old_map) != len(record_keys) or old_map["source_ordinal"].duplicated().any():
        raise RuntimeError("old exact manifest does not provide a one-to-one source ordinal map")
    links = links.merge(old_map, on="record_id", how="left", validate="one_to_one")

    identity_checks = {
        "old_pdb_identity": links["pdb"].str.lower().eq(links["old_pdb"].str.lower()),
        "old_receptor_chain_identity": links["receptor_chain"].eq(links["old_receptor_chain"]),
        "old_uniprot_identity": links["uniprot"].map(base_accession).eq(
            links["old_uniprot"].map(base_accession)
        ),
        "old_ligand_ccd_identity": links["ligand_ccd"].str.upper().eq(
            links["old_ligand_ccd"].str.upper()
        ),
        "old_ligand_chain_identity": links["ligand_chain"].eq(links["old_ligand_chain"]),
        "old_ligand_resnum_identity": links["ligand_resnum"].eq(links["old_ligand_resnum"]),
    }
    if not all(series.all() for series in identity_checks.values()):
        raise RuntimeError("pilot exact observations disagree with the frozen source manifest")

    query_keys = links[
        [
            "cumulative_pair_id", "source_stage", "source_population_id", "source_pair_id",
            "uniprot", "biolip_ccd", "full_inchikey", "connectivity_key", "record_id",
            "source_ordinal", "participant_role", "site_role", "representative_pdb",
            "representative_pmid", "multisite_audit_id", "multisite_audit_disposition",
            "multisite_audit_claim_limit", "manual_site_interpretation", "manual_rationale",
            "pdb", "receptor_chain", "binding_site_id", "ligand_ccd", "ligand_chain",
            "ligand_resnum", "binding_res_author", "binding_res_renum",
        ]
    ].copy()
    con.register("orthosteric_source", query_keys)
    scan = (
        "read_csv_auto(?, delim='\\t', header=true, compression='gzip', "
        "all_varchar=true, sample_size=-1)"
    )
    query = f"""
        WITH master AS (
          SELECT CAST(source_ordinal AS BIGINT) AS source_ordinal, observation_id,
                 lower(pdb_id) AS master_pdb, receptor_chain AS master_receptor_chain,
                 split_part(upper(uniprot_resolved), '-', 1) AS master_uniprot,
                 upper(ligand_ccd) AS master_ligand_ccd, ligand_chain AS master_ligand_chain,
                 ligand_auth_seq_id AS master_ligand_resnum, full_inchikey AS master_full_inchikey,
                 connectivity_key AS master_connectivity_key, binding_residues_auth_raw,
                 binding_uniprot_positions, active_orthosteric_pair_reference, pubmed_id AS biolip_pubmed_id
          FROM {scan}
        ), eligibility AS (
          SELECT observation_id, final_site_mapping_usable, final_eligibility_reason FROM {scan}
        )
        SELECT o.*, m.*, e.final_site_mapping_usable, e.final_eligibility_reason
        FROM orthosteric_source o
        INNER JOIN master m USING(source_ordinal)
        LEFT JOIN eligibility e USING(observation_id)
        ORDER BY o.cumulative_pair_id, o.source_ordinal
    """
    linked = con.execute(query, [str(MASTER), str(ELIGIBILITY)]).fetchdf()
    con.close()
    if len(linked) != len(links):
        raise RuntimeError("checkpoint-4 orthosteric source-ordinal join is incomplete")

    linked["checkpoint4_identity_match"] = (
        linked["pdb"].str.lower().eq(linked["master_pdb"])
        & linked["receptor_chain"].eq(linked["master_receptor_chain"])
        & linked["uniprot"].map(base_accession).eq(linked["master_uniprot"])
        & linked["ligand_ccd"].str.upper().eq(linked["master_ligand_ccd"])
        & linked["ligand_chain"].eq(linked["master_ligand_chain"])
        & linked["ligand_resnum"].eq(linked["master_ligand_resnum"])
        & linked["full_inchikey"].str.upper().eq(linked["master_full_inchikey"].str.upper())
    )
    linked["checkpoint4_mapping_usable"] = linked["final_site_mapping_usable"].map(truth)
    linked["active_broad_orthosteric_pair_reference"] = linked[
        "active_orthosteric_pair_reference"
    ].map(truth)
    linked["eligible_exact_orthosteric"] = (
        linked["checkpoint4_identity_match"]
        & linked["checkpoint4_mapping_usable"]
        & linked["active_broad_orthosteric_pair_reference"]
    )
    linked.insert(
        0,
        "orthosteric_link_id",
        [stable_id("ORLINK_", f"{pair}|{obs}") for pair, obs in zip(linked.cumulative_pair_id, linked.observation_id)],
    )
    keep = [
        "orthosteric_link_id", "cumulative_pair_id", "source_stage", "source_population_id",
        "source_pair_id", "record_id", "source_ordinal", "observation_id", "master_pdb",
        "master_receptor_chain", "master_uniprot", "master_ligand_ccd", "master_ligand_chain",
        "master_ligand_resnum", "master_full_inchikey", "master_connectivity_key",
        "binding_residues_auth_raw", "binding_uniprot_positions", "participant_role", "site_role",
        "representative_pdb", "representative_pmid", "multisite_audit_id",
        "multisite_audit_disposition", "multisite_audit_claim_limit", "manual_site_interpretation",
        "manual_rationale", "checkpoint4_identity_match", "checkpoint4_mapping_usable",
        "final_eligibility_reason", "active_broad_orthosteric_pair_reference",
        "eligible_exact_orthosteric", "biolip_pubmed_id",
    ]
    linked = linked[keep].rename(
        columns={
            "master_pdb": "pdb_id", "master_receptor_chain": "receptor_chain",
            "master_uniprot": "uniprot", "master_ligand_ccd": "ligand_ccd",
            "master_ligand_chain": "ligand_chain", "master_ligand_resnum": "ligand_auth_seq_id",
            "master_full_inchikey": "full_inchikey", "master_connectivity_key": "connectivity_key",
            "final_eligibility_reason": "checkpoint4_eligibility_reason",
        }
    )
    coverage = (
        linked.groupby("cumulative_pair_id", as_index=False)
        .agg(
            source_stage=("source_stage", "first"),
            source_population_id=("source_population_id", "first"),
            uniprot=("uniprot", "first"),
            ligand_ccd=("ligand_ccd", "first"),
            full_inchikey=("full_inchikey", "first"),
            exact_observations=("observation_id", "nunique"),
            eligible_exact_observations=("eligible_exact_orthosteric", "sum"),
            distinct_pdbs=("pdb_id", "nunique"),
        )
        .sort_values("cumulative_pair_id")
    )
    return linked, coverage


def build_legacy_reconciliation(links: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    old = con.execute(
        "SELECT CAST(source_ordinal AS BIGINT) AS source_ordinal, record_id FROM read_parquet(?) WHERE allo_exact",
        [str(OLD_BRIDGE)],
    ).fetchdf()
    master_scan = (
        "read_csv_auto(?, delim='\\t', header=true, compression='gzip', "
        "all_varchar=true, sample_size=-1)"
    )
    source_to_observation = con.execute(
        f"SELECT CAST(source_ordinal AS BIGINT) source_ordinal, observation_id FROM {master_scan}",
        [str(MASTER)],
    ).fetchdf()
    con.close()
    old = old.merge(source_to_observation, on="source_ordinal", how="left", validate="one_to_one")
    link_status = (
        links.groupby("observation_id")["link_status"].agg(join_values).rename("checkpoint5_candidate_statuses")
    )
    new = observations[["observation_id", "eligible_exact_allosteric"]].copy()
    union = pd.DataFrame(
        {"observation_id": sorted(set(old.observation_id) | set(new.observation_id))}
    )
    union = union.merge(
        old[["observation_id", "record_id", "source_ordinal"]].assign(legacy_allo_exact=True),
        on="observation_id", how="left",
    )
    union = union.merge(
        new.assign(checkpoint5_site_supported=True), on="observation_id", how="left"
    )
    union = union.merge(link_status, on="observation_id", how="left")
    union["legacy_allo_exact"] = union["legacy_allo_exact"].fillna(False).astype(bool)
    union["checkpoint5_site_supported"] = union["checkpoint5_site_supported"].fillna(False).astype(bool)
    union["eligible_exact_allosteric"] = union["eligible_exact_allosteric"].fillna(False).astype(bool)
    union["reconciliation_state"] = [
        "retained_with_site_residue_support" if old_value and new_value
        else "removed_by_stricter_exact_site_rule" if old_value
        else "newly_added_by_stricter_exact_site_rule"
        for old_value, new_value in zip(union.legacy_allo_exact, union.checkpoint5_site_supported)
    ]
    union["checkpoint5_candidate_statuses"] = union["checkpoint5_candidate_statuses"].fillna(
        "no_resolved_uniprot_exact_instance_link"
    )
    return union.sort_values("observation_id")


def build_unified(
    allosteric: pd.DataFrame,
    orthosteric: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    allo = allosteric.loc[allosteric["eligible_exact_allosteric"].astype(bool)].copy()
    ortho = orthosteric.loc[orthosteric["eligible_exact_orthosteric"].astype(bool)].copy()
    exact_overlap = set(allo.observation_id) & set(ortho.observation_id)
    broad_conflict = set(
        allo.loc[allo["active_broad_orthosteric_pair_reference"].astype(bool), "observation_id"]
    )
    conflict_ids = exact_overlap | broad_conflict
    conflicts = []
    for observation_id in sorted(conflict_ids):
        reasons = []
        if observation_id in exact_overlap:
            reasons.append("exact_allosteric_and_exact_orthosteric")
        if observation_id in broad_conflict:
            reasons.append("exact_allosteric_and_active_broad_orthosteric_pair")
        conflicts.append({"observation_id": observation_id, "conflict_reasons": ";".join(reasons)})
    conflict_frame = pd.DataFrame(conflicts, columns=["observation_id", "conflict_reasons"])

    rows: List[dict] = []
    for row in allo.loc[~allo.observation_id.isin(conflict_ids)].itertuples(index=False):
        rows.append(
            {
                "observation_id": row.observation_id,
                "source_ordinal": int(row.source_ordinal),
                "reference_label": "allosteric",
                "reference_source": "AlloBench_exact_structure_instance_and_site_residue",
                "reference_source_ids": row.allobench_source_ids,
                "pdb_id": row.pdb_id,
                "receptor_chain": row.receptor_chain,
                "uniprot": row.uniprot,
                "ligand_ccd": row.ligand_ccd,
                "ligand_chain": row.ligand_chain,
                "ligand_auth_seq_id": row.ligand_auth_seq_id,
                "full_inchikey": row.full_inchikey,
                "connectivity_key": row.connectivity_key,
                "binding_residues_auth_raw": row.binding_residues_auth_raw,
                "binding_uniprot_positions": row.binding_uniprot_positions,
                "source_site_relation": row.source_site_relation,
                "source_site_relation_values": row.source_site_relation_values,
                "reference_pair_id": "",
                "n_reference_source_rows": int(row.n_allobench_source_rows),
                "site_author_overlap_count": int(row.max_author_position_overlap_count),
                "site_author_overlap_jaccard": float(row.max_author_position_jaccard),
                "active_broad_orthosteric_pair_reference": bool(
                    row.active_broad_orthosteric_pair_reference
                ),
            }
        )
    for row in ortho.loc[~ortho.observation_id.isin(conflict_ids)].itertuples(index=False):
        rows.append(
            {
                "observation_id": row.observation_id,
                "source_ordinal": int(row.source_ordinal),
                "reference_label": "orthosteric",
                "reference_source": "BioLiP_manual_primary_ligand_audit_retained_pair",
                "reference_source_ids": row.orthosteric_link_id,
                "pdb_id": row.pdb_id,
                "receptor_chain": row.receptor_chain,
                "uniprot": row.uniprot,
                "ligand_ccd": row.ligand_ccd,
                "ligand_chain": row.ligand_chain,
                "ligand_auth_seq_id": row.ligand_auth_seq_id,
                "full_inchikey": row.full_inchikey,
                "connectivity_key": row.connectivity_key,
                "binding_residues_auth_raw": row.binding_residues_auth_raw,
                "binding_uniprot_positions": row.binding_uniprot_positions,
                "source_site_relation": "canonical_primary_site",
                "source_site_relation_values": row.site_role,
                "reference_pair_id": row.cumulative_pair_id,
                "n_reference_source_rows": 1,
                "site_author_overlap_count": "",
                "site_author_overlap_jaccard": "",
                "active_broad_orthosteric_pair_reference": bool(
                    row.active_broad_orthosteric_pair_reference
                ),
            }
        )
    unified = pd.DataFrame(rows).sort_values(["reference_label", "observation_id"])
    if unified["observation_id"].duplicated().any():
        raise RuntimeError("unified exact reference contains duplicate observations")
    return unified, conflict_frame


def main() -> None:
    cp2 = load_checkpoint(CP2, 2)
    cp4 = load_checkpoint(CP4, 4)
    required = [
        MASTER, ELIGIBILITY, ACTIVE_ORTHO, ALLOBENCH, STRICT_PAIRS, AUDIT_LEDGER,
        OLD_MANIFEST, OLD_BRIDGE,
    ] + [path for path, _ in PILOT_INPUTS.values()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    registry, parsed_lookup = build_allobench_registry()
    allosteric_links, allosteric_observations = build_allosteric_links(registry, parsed_lookup)
    orthosteric_links, orthosteric_coverage = build_orthosteric_links()
    reconciliation = build_legacy_reconciliation(allosteric_links, allosteric_observations)
    unified, conflicts = build_unified(allosteric_observations, orthosteric_links)

    write_frame(registry.sort_values("allobench_row_index"), ALLOBENCH_REGISTRY)
    write_frame(allosteric_links.sort_values(["allobench_row_index", "source_ordinal"]), ALLO_LINKS)
    write_frame(allosteric_observations.sort_values("observation_id"), ALLO_OBSERVATIONS)
    write_frame(reconciliation, ALLO_RECONCILIATION)
    write_frame(orthosteric_links.sort_values(["cumulative_pair_id", "source_ordinal"]), ORTHO_LINKS)
    write_frame(orthosteric_coverage, ORTHO_PAIR_COVERAGE)
    write_frame(conflicts, REFERENCE_CONFLICTS)
    write_frame(unified, REFERENCE_OBSERVATIONS)

    count_rows = []
    def add(section: str, metric: str, value: int) -> None:
        count_rows.append({"section": section, "metric": metric, "value": int(value)})

    add("allobench", "source_rows", len(registry))
    for key, value in registry["source_link_status"].value_counts().items():
        add("allobench_source_link_status", str(key), int(value))
    add("allosteric", "candidate_exact_instance_uniprot_links", len(allosteric_links))
    add(
        "allosteric", "site_residue_supported_links",
        int(allosteric_links["link_status"].eq("exact_instance_site_residue_supported").sum()),
    )
    add("allosteric", "site_residue_supported_observations", len(allosteric_observations))
    add(
        "allosteric", "eligible_exact_observations",
        int(allosteric_observations["eligible_exact_allosteric"].sum()),
    )
    add("orthosteric", "audited_retained_pairs", len(orthosteric_coverage))
    add("orthosteric", "exact_observations", len(orthosteric_links))
    add(
        "orthosteric", "eligible_exact_observations",
        int(orthosteric_links["eligible_exact_orthosteric"].sum()),
    )
    add("unified", "reference_conflicts_excluded", len(conflicts))
    for label, count in unified["reference_label"].value_counts().items():
        add("unified", f"{label}_observations", int(count))
    add("unified", "total_observations", len(unified))
    count_frame = pd.DataFrame(count_rows)
    write_frame(count_frame, COUNT_SUMMARY, compressed=False)

    input_records = [
        {"asset_id": "checkpoint2_validation", **file_record(CP2)},
        {"asset_id": "checkpoint4_validation", **file_record(CP4)},
        {"asset_id": "checkpoint4_master", **file_record(MASTER)},
        {"asset_id": "checkpoint4_eligibility", **file_record(ELIGIBILITY)},
        {"asset_id": "active_orthosteric_pair_master", **file_record(ACTIVE_ORTHO)},
        {"asset_id": "allobench", **file_record(ALLOBENCH)},
        {"asset_id": "strict_orthosteric_pairs", **file_record(STRICT_PAIRS)},
        {"asset_id": "strict_pair_audit_ledger", **file_record(AUDIT_LEDGER)},
        {"asset_id": "old_exact_manifest_source_ordinal_bridge", **file_record(OLD_MANIFEST)},
        {"asset_id": "old_label_bridge_reconciliation_only", **file_record(OLD_BRIDGE)},
    ]
    for stage, (path, _) in PILOT_INPUTS.items():
        input_records.append({"asset_id": f"{stage}_exact_observations", **file_record(path)})
    write_frame(pd.DataFrame(input_records), INPUT_HASHES, compressed=False)

    outputs = {
        "allobench_registry": file_record(ALLOBENCH_REGISTRY),
        "allosteric_links": file_record(ALLO_LINKS),
        "allosteric_observations": file_record(ALLO_OBSERVATIONS),
        "legacy_allosteric_reconciliation": file_record(ALLO_RECONCILIATION),
        "orthosteric_links": file_record(ORTHO_LINKS),
        "orthosteric_pair_coverage": file_record(ORTHO_PAIR_COVERAGE),
        "reference_conflicts": file_record(REFERENCE_CONFLICTS),
        "reference_observations": file_record(REFERENCE_OBSERVATIONS),
        "count_summary": file_record(COUNT_SUMMARY),
        "input_hashes": file_record(INPUT_HASHES),
    }

    legacy_counts = reconciliation["reconciliation_state"].value_counts().to_dict()
    label_summary = (
        unified.groupby("reference_label")
        .agg(
            observations=("observation_id", "nunique"),
            proteins=("uniprot", "nunique"),
            pdbs=("pdb_id", "nunique"),
            ligands=("full_inchikey", "nunique"),
        )
        .reset_index()
    )
    label_summary_text = "\n".join(
        f"- `{row.reference_label}`: {int(row.observations):,} observations, "
        f"{int(row.proteins):,} proteins, {int(row.pdbs):,} PDBs, "
        f"{int(row.ligands):,} full InChIKeys"
        for row in label_summary.itertuples(index=False)
    )
    report = f"""# Checkpoint 5: exact allosteric and orthosteric site-reference linkage

Status: **complete; independent validation required**

## Allosteric source and exact-link rule

All {len(registry):,} AlloBench source rows are retained in the source registry.
An exact allosteric observation requires exact UniProt, PDB, CCD, deposited
ligand chain and ligand author-residue identity, followed by overlap between a
chain-qualified AlloBench allosteric-site author residue and the BioLiP binding
residues on that receptor chain.

- Candidate exact instance + UniProt links: **{len(allosteric_links):,}**
- Site-residue-supported links: **{int(allosteric_links['link_status'].eq('exact_instance_site_residue_supported').sum()):,}**
- Distinct site-residue-supported observations: **{len(allosteric_observations):,}**
- Mapping-eligible exact allosteric observations: **{int(allosteric_observations['eligible_exact_allosteric'].sum()):,}**

Legacy reconciliation: {legacy_counts}.  The prior exact flag used ligand
instance identity without requiring resolved UniProt and receptor-chain site
residue overlap; it is retained only as a reconciliation column.

## Orthosteric exact controls

The 247 manually retained BioLiP primary-ligand pairs link to
**{len(orthosteric_links):,}** exact observations.  Mapping-eligible exact
orthosteric controls: **{int(orthosteric_links['eligible_exact_orthosteric'].sum()):,}**.
All pair/CCD/InChIKey/source-ordinal identities were checked against the new
checkpoint-4 master.

GtoP, KLIFS, BRENDA, ChEBI, KEGG and UniProt-derived orthosteric roles remain
present through the checkpoint-2 broad pair reference and the observation-level
`active_broad_orthosteric_pair_reference` field.  They are not called exact
structural sites unless they are also among the manually audited BioLiP
observations above.

## Unified reference observations

    {label_summary_text}

- Exact label conflicts excluded: **{len(conflicts):,}**
- Unified exact reference observations: **{len(unified):,}**

AlloBench `site_overlap` is preserved as distinct, overlapping, or unspecified;
overlapping-site annotations are not silently discarded.  Checkpoint 6 can
therefore compare distance definitions on the prespecified distinct-site stratum
and report the overlapping stratum separately.

## Scope boundary

This checkpoint links labels to exact observations.  It does not calculate a
distance, fit a Bayesian ranking model, or rank the BioLiP universe.
"""
    atomic_text(REPORT, report)
    outputs["report"] = file_record(REPORT)

    summary = {
        "checkpoint": 5,
        "status": "complete_pending_independent_validation",
        "scope": "exact allosteric and exact audited BioLiP orthosteric observation linkage",
        "checkpoint2_status": cp2["status"],
        "checkpoint4_status": cp4["status"],
        "allobench_source_rows": len(registry),
        "allobench_source_link_status_counts": {
            str(key): int(value) for key, value in registry["source_link_status"].value_counts().items()
        },
        "allosteric_candidate_links": len(allosteric_links),
        "allosteric_site_supported_links": int(
            allosteric_links["link_status"].eq("exact_instance_site_residue_supported").sum()
        ),
        "allosteric_site_supported_observations": len(allosteric_observations),
        "allosteric_eligible_exact_observations": int(
            allosteric_observations["eligible_exact_allosteric"].sum()
        ),
        "legacy_allosteric_reconciliation_counts": {
            str(key): int(value) for key, value in legacy_counts.items()
        },
        "orthosteric_retained_pairs": len(orthosteric_coverage),
        "orthosteric_exact_observations": len(orthosteric_links),
        "orthosteric_eligible_exact_observations": int(
            orthosteric_links["eligible_exact_orthosteric"].sum()
        ),
        "reference_conflicts_excluded": len(conflicts),
        "unified_reference_label_counts": {
            str(key): int(value) for key, value in unified["reference_label"].value_counts().items()
        },
        "unified_reference_observations": len(unified),
        "large_structure_directory_recursively_scanned": False,
        "outputs": outputs,
    }
    atomic_json(BUILD_SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
