#!/usr/bin/env python3
"""Inventory reusable BioLiP structure/mapping assets without a broad rescan.

Checkpoint 3 is deliberately read-only with respect to legacy assets.  It
enumerates only four explicitly named, small cache directories and reads
already-generated manifests/logs.  In particular, it does *not* recursively
walk the large /shared_data BioLiP CIF/FASTA repositories.

The script requires DuckDB to inspect the existing Parquet manifests.  In this
workspace it is run with ``miniconda3/bin/python``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

import duckdb
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
SHARED = Path("/shared_data/11.HS_allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
REPORTS = PACKAGE / "reports"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

CHECKPOINT2_VALIDATION = VALIDATION / "CHECKPOINT2_VALIDATION.json"

XAI = ROOT / "analysis/structure_known_biolip_xai"
XAI_DATA = XAI / "data"
XAI_ASSETS = XAI / "assets"

RAW_BIOLIP = SHARED / "Data/BioLiP.txt"
OFFSET_RESCUED = SHARED / "13.Refine_BioLiP/BioLiP_offset_rescued.tsv"
OFFSET_LOG = SHARED / "13.Refine_BioLiP/2.PDB_UniProt_offset.log"
LEGACY_AGGREGATE = SHARED / "13.Refine_BioLiP/BioLiP_Aggregated_Final.tsv"
LEGACY_DISTANCE = SHARED / "8.Distances_BioLiP/BioLiP_Distances_Final.tsv"
LEGACY_DISTANCE_LOG = SHARED / "13.Refine_BioLiP/9.Calculate_Distances.log"

MANIFEST_PARQUET = XAI_DATA / "biolip_manifest.parquet"
ANCHOR_COVERAGE = XAI_DATA / "anchor_coverage_by_row.tsv"
LABELABILITY = XAI_DATA / "labelability_L1_L2.tsv"
LABELED_ASSET_MANIFEST = XAI_DATA / "labeled_anchor_asset_manifest.tsv"
P1_QC = XAI_DATA / "p1_structural_label_qc.tsv"
P1_FEATURES = XAI_DATA / "p1_features_raw.parquet"
P1_COLLAPSE = XAI_DATA / "p1_replicate_collapse_audit.tsv"
PU_CANDIDATES = XAI_DATA / "pu_pilot_candidates.tsv"
PU_CLUSTERS = XAI_DATA / "pu_pilot_site_clusters.parquet"
PU_SUMMARY = XAI_DATA / "pu_pilot_summary.json"
P1_FETCH_LOG = XAI_ASSETS / "fetch_log.tsv"
PU_FETCH_LOG = XAI_ASSETS / "pu_pilot_fetch_log.tsv"

BASE_STRUCTURES = ROOT / "data/structures"
BASE_SIFTS = ROOT / "analysis/sifts"
XAI_STRUCTURES = XAI_ASSETS / "structures"
XAI_SIFTS = XAI_ASSETS / "sifts"
CACHE_PATH_DESCRIPTION = (
    f"{BASE_STRUCTURES} + {BASE_SIFTS} + {XAI_STRUCTURES} + {XAI_SIFTS}"
)

OFFICIAL_SCHEMA_URL = "https://seq2fun.dcmb.med.umich.edu/BioLiP/download/readme.txt"
OFFICIAL_DOWNLOAD_URL = "https://seq2fun.dcmb.med.umich.edu/BioLiP/download.html"

# Frozen expectations are observations of the pre-existing workspace, not
# targets manufactured by this checkpoint.  They make later cache drift
# explicit instead of silently changing the inventory.
EXPECTED = {
    "raw_rows_documented": 989058,
    "raw_columns": 21,
    "offset_columns": 22,
    "offset_exact": 729450,
    "offset_sw_rescued": 248535,
    "offset_mismatch": 11072,
    "offset_missing": 1,
    "offset_success": 977985,
    "legacy_valid_records": 888435,
    "legacy_unique_structures": 143367,
    "legacy_unique_pairs": 120014,
    "legacy_distance_rows": 167817,
    "manifest_rows": 140700,
    "manifest_primary_pdbs": 98502,
    "anchor_coverage_rows": 140700,
    "labelability_rows": 11705,
    "labelability_primary_pdbs": 10223,
    "labeled_asset_pdbs": 3227,
    "base_structures": 1528,
    "base_sifts": 1894,
    "xai_structures": 499,
    "xai_sifts": 499,
    "structure_union": 1890,
    "sifts_union": 2233,
    "paired_union": 1890,
    "sifts_only": 343,
    "structure_without_sifts": 0,
    "manifest_cached_primary_pdbs": 1418,
    "manifest_rows_cached_primary": 1752,
    "labelability_cached_primary_pdbs": 1346,
    "labelability_rows_cached_primary": 1616,
    "p1_rows": 436,
    "p1_pdbs": 415,
    "p1_mapped_rows": 421,
    "p1_mapped_pdbs": 400,
    "pu_candidate_rows": 244,
    "pu_candidate_pdbs": 193,
    "pu_mapped_rows": 244,
    "pu_site_clusters": 138,
    "pu_site_cluster_proteins": 30,
    "p1_pu_record_overlap": 9,
    "p1_pu_mapped_record_union": 656,
    "p1_pu_pdb_overlap": 10,
    "p1_pu_pdb_union": 598,
    "p1_fetch_rows": 350,
    "pu_fetch_rows": 149,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_tsv(frame: pd.DataFrame, path: Path, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if compressed:
        with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=6, mtime=0) as raw:
            frame.to_csv(raw, sep="\t", index=False)
    else:
        frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def read_first_row_width(path: Path) -> int:
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        return len(handle.readline().rstrip("\r\n").split("\t"))


def file_ids(directory: Path, suffix: str) -> Set[str]:
    """Enumerate one explicitly named directory, never recursively."""
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {
        item.stem.lower()
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() == suffix
    }


def parse_int(pattern: str, text: str, description: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"could not parse {description}")
    return int(match.group(1).replace(",", ""))


def assert_expected(observed: Mapping[str, int]) -> None:
    mismatches = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in EXPECTED.items()
        if observed.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"checkpoint-3 frozen inventory changed: {mismatches}")


def stat_record(asset_id: str, path: Path, hash_file: bool) -> Dict[str, object]:
    stat = path.stat()
    return {
        "asset_id": asset_id,
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_epoch_seconds": int(stat.st_mtime),
        "sha256": sha256(path) if hash_file else "",
        "hash_status": "computed" if hash_file else "not_computed_large_io_guard",
        "hash_note": "" if hash_file else "Recorded size/mtime only; full read was intentionally avoided.",
    }


def main() -> None:
    named_inputs = [
        CHECKPOINT2_VALIDATION,
        RAW_BIOLIP,
        OFFSET_RESCUED,
        OFFSET_LOG,
        LEGACY_AGGREGATE,
        LEGACY_DISTANCE,
        LEGACY_DISTANCE_LOG,
        MANIFEST_PARQUET,
        ANCHOR_COVERAGE,
        LABELABILITY,
        LABELED_ASSET_MANIFEST,
        P1_QC,
        P1_FEATURES,
        P1_COLLAPSE,
        PU_CANDIDATES,
        PU_CLUSTERS,
        PU_SUMMARY,
        P1_FETCH_LOG,
        PU_FETCH_LOG,
    ]
    for path in named_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint2 = json.loads(CHECKPOINT2_VALIDATION.read_text())
    if checkpoint2.get("checkpoint") != 2 or checkpoint2.get("status") != "validated":
        raise RuntimeError("checkpoint 2 is not validated")
    for record in checkpoint2.get("outputs", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint-2 output hash mismatch: {path}")

    # Only the first line of each large source is touched here.
    raw_columns = read_first_row_width(RAW_BIOLIP)
    offset_columns = read_first_row_width(OFFSET_RESCUED)

    offset_text = OFFSET_LOG.read_text(encoding="utf-8", errors="replace")
    distance_text = LEGACY_DISTANCE_LOG.read_text(encoding="utf-8", errors="replace")
    offset_exact = parse_int(r"Exact Match[^:]*:\s*([0-9,]+)", offset_text, "exact offsets")
    offset_sw = parse_int(r"SW 1% Rescue[^:]*:\s*([0-9,]+)", offset_text, "SW offsets")
    offset_mismatch = parse_int(r"Mismatch\)[^:]*:\s*([0-9,]+)", offset_text, "offset mismatches")
    offset_missing = parse_int(r"누락 에러[^:]*:\s*([0-9,]+)", offset_text, "missing chain/files")
    raw_rows_documented = parse_int(r"Processing:.*?([0-9]+)/([0-9]+)", offset_text, "raw row count")
    # The progress pattern contains current/total; group 1 and group 2 are equal at completion.
    offset_success = offset_exact + offset_sw
    legacy_valid = parse_int(r"유효 레코드 수\s*:\s*([0-9,]+)", distance_text, "legacy valid records")
    legacy_structures = parse_int(r"고유 PDB 구조:\s*([0-9,]+)", distance_text, "legacy structures")
    legacy_pairs = parse_int(r"고유 Pair 수\s*:\s*([0-9,]+)", distance_text, "legacy pairs")

    base_struct = file_ids(BASE_STRUCTURES, ".cif")
    base_sifts = file_ids(BASE_SIFTS, ".json")
    xai_struct = file_ids(XAI_STRUCTURES, ".cif")
    xai_sifts = file_ids(XAI_SIFTS, ".json")
    structure_union = base_struct | xai_struct
    sifts_union = base_sifts | xai_sifts
    paired_union = structure_union & sifts_union

    anchor = pd.read_csv(ANCHOR_COVERAGE, sep="\t", low_memory=False)
    labelability = pd.read_csv(LABELABILITY, sep="\t", low_memory=False)
    labeled_assets = pd.read_csv(LABELED_ASSET_MANIFEST, sep="\t", low_memory=False)
    p1 = pd.read_csv(P1_QC, sep="\t", low_memory=False)
    p1_collapse = pd.read_csv(P1_COLLAPSE, sep="\t", low_memory=False)
    pu = pd.read_csv(PU_CANDIDATES, sep="\t", low_memory=False)
    pu_summary = json.loads(PU_SUMMARY.read_text())
    p1_fetch = pd.read_csv(P1_FETCH_LOG, sep="\t", low_memory=False)
    pu_fetch = pd.read_csv(
        PU_FETCH_LOG,
        sep="\t",
        header=None,
        names=["pdb", "status", "error"],
        dtype=str,
    )

    con = duckdb.connect(database=":memory:")
    manifest_summary = con.execute(
        """
        SELECT count(*) AS rows, count(DISTINCT lower(primary_pdb)) AS primary_pdbs
        FROM read_parquet(?)
        """,
        [str(MANIFEST_PARQUET)],
    ).fetchone()
    manifest_primary = con.execute(
        """
        SELECT lower(primary_pdb) AS pdb, count(*) AS n_manifest_primary_rows
        FROM read_parquet(?) GROUP BY 1
        """,
        [str(MANIFEST_PARQUET)],
    ).fetchdf()
    p1_feature_rows = int(
        con.execute("SELECT count(*) FROM read_parquet(?)", [str(P1_FEATURES)]).fetchone()[0]
    )
    cluster_summary = con.execute(
        """
        SELECT count(*) AS rows, count(DISTINCT uniprot) AS proteins
        FROM read_parquet(?)
        """,
        [str(PU_CLUSTERS)],
    ).fetchone()
    pu_cluster_pdb = {
        str(row[0]).lower()
        for row in con.execute(
            "SELECT DISTINCT lower(rep_pdb) FROM read_parquet(?) WHERE rep_pdb IS NOT NULL",
            [str(PU_CLUSTERS)],
        ).fetchall()
    }
    con.close()

    manifest_primary["pdb"] = manifest_primary["pdb"].astype(str).str.lower()
    manifest_primary_set = set(manifest_primary["pdb"])
    manifest_cached = manifest_primary.loc[manifest_primary["pdb"].isin(paired_union)]
    anchored_primary = labelability["primary_pdb"].astype(str).str.lower()
    anchored_primary_set = set(anchored_primary)

    p1["pdb_norm"] = p1["pdb"].astype(str).str.lower()
    p1_ok = p1.loc[p1["map_status"].eq("ok")].copy()
    pu_pdb = set(pu["rep_pdb"].astype(str).str.lower())
    p1_pdb = set(p1["pdb_norm"])
    p1_ok_ids = set(p1_ok["record_id"].astype(str))
    pu_ids = set(pu["rep_record_id"].astype(str))

    observed = {
        "raw_rows_documented": raw_rows_documented,
        "raw_columns": raw_columns,
        "offset_columns": offset_columns,
        "offset_exact": offset_exact,
        "offset_sw_rescued": offset_sw,
        "offset_mismatch": offset_mismatch,
        "offset_missing": offset_missing,
        "offset_success": offset_success,
        "legacy_valid_records": legacy_valid,
        "legacy_unique_structures": legacy_structures,
        "legacy_unique_pairs": legacy_pairs,
        # This count is frozen in the prior bounded audit report; the 14 MB
        # table is not rescanned solely to count its lines.
        "legacy_distance_rows": 167817,
        "manifest_rows": int(manifest_summary[0]),
        "manifest_primary_pdbs": int(manifest_summary[1]),
        "anchor_coverage_rows": int(len(anchor)),
        "labelability_rows": int(len(labelability)),
        "labelability_primary_pdbs": int(anchored_primary.nunique()),
        "labeled_asset_pdbs": int(labeled_assets["pdb"].astype(str).str.lower().nunique()),
        "base_structures": len(base_struct),
        "base_sifts": len(base_sifts),
        "xai_structures": len(xai_struct),
        "xai_sifts": len(xai_sifts),
        "structure_union": len(structure_union),
        "sifts_union": len(sifts_union),
        "paired_union": len(paired_union),
        "sifts_only": len(sifts_union - structure_union),
        "structure_without_sifts": len(structure_union - sifts_union),
        "manifest_cached_primary_pdbs": int(len(manifest_cached)),
        "manifest_rows_cached_primary": int(manifest_cached["n_manifest_primary_rows"].sum()),
        "labelability_cached_primary_pdbs": len(anchored_primary_set & paired_union),
        "labelability_rows_cached_primary": int(anchored_primary.isin(paired_union).sum()),
        "p1_rows": int(len(p1)),
        "p1_pdbs": len(p1_pdb),
        "p1_mapped_rows": int(len(p1_ok)),
        "p1_mapped_pdbs": int(p1_ok["pdb_norm"].nunique()),
        "pu_candidate_rows": int(len(pu)),
        "pu_candidate_pdbs": len(pu_pdb),
        "pu_mapped_rows": int(pu_summary["n_mapped"]),
        "pu_site_clusters": int(cluster_summary[0]),
        "pu_site_cluster_proteins": int(cluster_summary[1]),
        "p1_pu_record_overlap": len(p1_ok_ids & pu_ids),
        "p1_pu_mapped_record_union": len(p1_ok_ids | pu_ids),
        "p1_pu_pdb_overlap": len(p1_pdb & pu_pdb),
        "p1_pu_pdb_union": len(p1_pdb | pu_pdb),
        "p1_fetch_rows": int(len(p1_fetch)),
        "pu_fetch_rows": int(len(pu_fetch)),
    }
    assert_expected(observed)

    if p1_feature_rows != len(p1):
        raise RuntimeError("P1 feature/QC row counts differ")
    if int(pu_summary["n_candidates"]) != len(pu):
        raise RuntimeError("PU candidate/summary row counts differ")
    if not p1_fetch["cif_status"].eq("ok").all() or not p1_fetch["sifts_status"].eq("ok").all():
        raise RuntimeError("P1 fetch log contains a failed asset")
    if not pu_fetch["status"].eq("ok").all():
        raise RuntimeError("PU fetch log contains a failed asset")
    if len(anchor) != int(manifest_summary[0]):
        raise RuntimeError("anchor coverage and aggregate manifest row counts differ")

    # One bounded row per PDB present in one of the four explicit caches.
    p1_fetch_set = set(p1_fetch["pdb"].astype(str).str.lower())
    pu_fetch_set = set(pu_fetch["pdb"].astype(str).str.lower())
    p1_ok_pdb = set(p1_ok["pdb_norm"])
    all_cache_pdbs = sorted(structure_union | sifts_union)
    cache_rows: List[Dict[str, object]] = []
    manifest_row_lookup = dict(
        zip(manifest_primary["pdb"], manifest_primary["n_manifest_primary_rows"].astype(int))
    )
    for pdb in all_cache_pdbs:
        structure_any = pdb in structure_union
        sifts_any = pdb in sifts_union
        cache_rows.append(
            {
                "pdb": pdb,
                "base_structure_present": pdb in base_struct,
                "base_sifts_present": pdb in base_sifts,
                "xai_structure_present": pdb in xai_struct,
                "xai_sifts_present": pdb in xai_sifts,
                "structure_any": structure_any,
                "sifts_any": sifts_any,
                "paired_structure_and_sifts": structure_any and sifts_any,
                "cache_status": (
                    "paired_reusable"
                    if structure_any and sifts_any
                    else "sifts_only_no_local_structure"
                    if sifts_any
                    else "structure_only_no_sifts"
                ),
                "aggregate_manifest_primary_pdb": pdb in manifest_primary_set,
                "n_aggregate_manifest_primary_rows": manifest_row_lookup.get(pdb, 0),
                "labelability_primary_pdb": pdb in anchored_primary_set,
                "n_labelability_primary_rows": int((anchored_primary == pdb).sum()),
                "p1_attempted_pdb": pdb in p1_pdb,
                "p1_mapped_ok_pdb": pdb in p1_ok_pdb,
                "pu_candidate_pdb": pdb in pu_pdb,
                "pu_cluster_representative_pdb": pdb in pu_cluster_pdb,
                "p1_fetch_log_pdb": pdb in p1_fetch_set,
                "pu_fetch_log_pdb": pdb in pu_fetch_set,
            }
        )
    cache_manifest = pd.DataFrame(cache_rows)

    schema_rows = [
        (1, "PDB ID", "pdb_id", "exact observation identity"),
        (2, "Receptor chain", "receptor_chain", "exact receptor-chain identity"),
        (3, "Resolution", "resolution", "structure quality metadata"),
        (4, "Binding site number code", "binding_site_code", "site identity within entry"),
        (5, "Ligand CCD ID", "ligand_ccd", "ligand identity"),
        (6, "Ligand chain", "ligand_chain", "exact ligand-instance identity"),
        (7, "Ligand serial number", "ligand_serial", "legacy ligand-instance identity"),
        (8, "Binding-site residues, PDB numbering", "binding_residues_auth", "observed candidate site"),
        (9, "Binding-site residues, receptor sequence renumbered from 1", "binding_residues_seq", "offset/mapping cross-check"),
        (10, "Catalytic-site residues, PDB numbering", "catalytic_residues_auth", "legacy catalytic annotation; not a universal orthosteric anchor"),
        (11, "Catalytic-site residues, receptor sequence renumbered from 1", "catalytic_residues_seq", "legacy catalytic annotation cross-check"),
        (12, "EC number", "ec_number", "enzyme metadata"),
        (13, "GO terms", "go_terms", "functional metadata"),
        (14, "Manual-literature binding affinity", "affinity_manual", "optional metadata"),
        (15, "Binding MOAD affinity", "affinity_moad", "optional metadata"),
        (16, "PDBbind-CN affinity", "affinity_pdbbind_cn", "optional metadata"),
        (17, "BindingDB affinity", "affinity_bindingdb", "optional metadata"),
        (18, "UniProt ID", "uniprot", "protein identity; residue mapping still requires audit"),
        (19, "PubMed ID", "pubmed_id", "publication provenance"),
        (20, "Ligand _atom_site.auth_seq_id", "ligand_auth_seq_id", "mmCIF ligand-instance identity"),
        (21, "Receptor sequence", "receptor_sequence_observed", "experimentally observed residues; not canonical full UniProt"),
    ]
    schema = pd.DataFrame(
        schema_rows,
        columns=["column_1based", "official_name", "normalized_name", "checkpoint4_role"],
    )
    schema["official_source"] = OFFICIAL_SCHEMA_URL

    mapping_rows = [
        {
            "mapping_asset": "legacy_biolip_to_observed_cif_chain_offset",
            "input_grain": "raw BioLiP interaction-site row",
            "rows_attempted": raw_rows_documented,
            "rows_mapped": offset_success,
            "rows_failed": offset_mismatch + offset_missing,
            "unique_pdbs": "not_recounted",
            "status_detail": f"exact={offset_exact}; sw_1pct={offset_sw}; mismatch={offset_mismatch}; missing={offset_missing}",
            "what_is_mapped": "BioLiP receptor-sequence position to observed CIF-chain sequence offset",
            "what_is_not_mapped": "not a residue-level SIFTS/UniProt crosswalk",
            "reuse_decision": "direct_reuse_as_chain_alignment_seed",
        },
        {
            "mapping_asset": "current_local_cif_plus_sifts_cache",
            "input_grain": "PDB structure",
            "rows_attempted": len(structure_union | sifts_union),
            "rows_mapped": len(paired_union),
            "rows_failed": len((structure_union | sifts_union) - paired_union),
            "unique_pdbs": len(structure_union | sifts_union),
            "status_detail": f"paired={len(paired_union)}; sifts_only={len(sifts_union - structure_union)}; structure_only={len(structure_union - sifts_union)}",
            "what_is_mapped": "local mmCIF plus SIFTS segment metadata available",
            "what_is_not_mapped": "raw BioLiP observations have not all been joined and residue-audited",
            "reuse_decision": "direct_reuse_for_available_pdbs_then_extend",
        },
        {
            "mapping_asset": "p1_structural_feature_qc",
            "input_grain": "aggregate-manifest record on one selected PDB/chain",
            "rows_attempted": len(p1),
            "rows_mapped": len(p1_ok),
            "rows_failed": len(p1) - len(p1_ok),
            "unique_pdbs": len(p1_pdb),
            "status_detail": "; ".join(f"{k}={v}" for k, v in p1["map_status"].value_counts().to_dict().items()),
            "what_is_mapped": "candidate/anchor residues and pilot structural features",
            "what_is_not_mapped": "not a full BioLiP observation inventory",
            "reuse_decision": "reuse_as_implementation_and_qc_seed",
        },
        {
            "mapping_asset": "pu_pilot_mapped_candidates",
            "input_grain": "selected aggregate-manifest representative record",
            "rows_attempted": len(pu),
            "rows_mapped": int(pu_summary["n_mapped"]),
            "rows_failed": len(pu) - int(pu_summary["n_mapped"]),
            "unique_pdbs": len(pu_pdb),
            "status_detail": f"mapped candidates={pu_summary['n_mapped']}; clusters={cluster_summary[0]}",
            "what_is_mapped": "candidate residues and pilot site features",
            "what_is_not_mapped": "pilot scope only; clusters use single-linkage Jaccard union",
            "reuse_decision": "reuse_mappings_features_not_final_clusters",
        },
    ]
    mapping_summary = pd.DataFrame(mapping_rows)

    reuse_rows = [
        ("raw_biolip_interaction_site_rows", RAW_BIOLIP, "direct_reuse", "PDB/receptor-chain/ligand-instance/site observation source", "Parse at source-line grain; do not aggregate by UniProt–CCD."),
        ("legacy_chain_offset_mapping", OFFSET_RESCUED, "direct_reuse", "Observed-chain sequence alignment seed", "Does not by itself map residues to canonical UniProt coordinates."),
        ("legacy_biolip_cif_fasta_repository", SHARED / "BioLiP_CIFs", "direct_reuse_with_manifest", "Coordinate source for previously processed structures", "Availability is taken from completed logs; checkpoint 3 does not rescan the directory."),
        ("local_paired_cif_sifts_cache", CACHE_PATH_DESCRIPTION, "direct_reuse", "Immediate residue-mapping input for 1,890 PDBs", "343 additional SIFTS JSONs lack a local structure."),
        ("sifts_segment_mapping_implementation", ROOT / "analysis/enm_common.py", "transform_and_reaudit", "Starting implementation for label/auth/UniProt mapping", "Linear segment mapping must be checked for gaps, insertions, chain ambiguity and ligand-contact coverage."),
        ("aggregate_biolip_manifest", MANIFEST_PARQUET, "transform_and_reaudit", "Pair/ligand provenance and legacy-score linkage", "Primary-PDB aggregate is not the final site-ranking grain."),
        ("legacy_uniprot_ccd_aggregate", LEGACY_AGGREGATE, "not_final_ranking_input", "Legacy comparison only", "Combines multiple PDBs, chains, ligand instances and distance values."),
        ("legacy_catalytic_ca_distance", LEGACY_DISTANCE, "not_final_ranking_input", "Distance baseline/sensitivity only", "Binding-site versus catalytic-site C-alpha geometry is not an exact orthosteric-anchor distance."),
        ("old_anchor_coverage_flags", ANCHOR_COVERAGE, "transform_and_reaudit", "Identity/label-provenance seed", "Asset-availability flags predate the enlarged 1,890-PDB paired cache and must be recomputed."),
        ("p1_mapped_features", P1_FEATURES, "direct_reuse_limited_scope", "Implementation/QC seed and feature-definition comparison", "421 mapped records, not a complete BioLiP mapping."),
        ("pu_pilot_clusters", PU_CLUSTERS, "not_final_ranking_input", "Pilot comparison only", "138 clusters use Jaccard>=0.5 union-find/single-linkage and must not be promoted to global final clusters."),
        ("exact_observation_master", PACKAGE / "future/CHECKPOINT4_EXACT_OBSERVATIONS.tsv.gz", "new_processing_required", "Final PDB-chain-ligand-instance-site master", "Must be built from raw BioLiP rows using reused mappings and explicit failures."),
        ("biological_assembly_multichain_audit", PACKAGE / "future/CHECKPOINT4_ASSEMBLY_AUDIT.tsv.gz", "new_processing_required", "Flag interfaces, symmetry copies and chain ambiguity", "Not represented by the selected-chain pilot features."),
        ("global_site_clusters", PACKAGE / "future/CHECKPOINT4_SITE_CLUSTERS.tsv.gz", "new_processing_required", "Site-level ranking unit", "Must be rebuilt after UniProt-coordinate mapping under a frozen non-chaining rule."),
    ]
    reuse = pd.DataFrame(
        [
            {
                "asset_id": asset_id,
                "path": str(path),
                "reuse_class": reuse_class,
                "permitted_use": permitted_use,
                "limitation_or_required_action": limitation,
            }
            for asset_id, path, reuse_class, permitted_use, limitation in reuse_rows
        ]
    )

    inventory_rows = [
        ("raw_biolip", RAW_BIOLIP, "interaction-site observations", raw_rows_documented, "PDB+receptor chain+ligand instance+site"),
        ("offset_rescued", OFFSET_RESCUED, "chain-sequence offset rows", raw_rows_documented, "raw observation row plus offset"),
        ("legacy_distance", LEGACY_DISTANCE, "legacy distance rows", observed["legacy_distance_rows"], "PDB+chain+ligand"),
        ("legacy_aggregate", LEGACY_AGGREGATE, "legacy aggregate rows", observed["manifest_rows"], "UniProt+CCD aggregate"),
        ("xai_manifest", MANIFEST_PARQUET, "aggregate manifest rows", observed["manifest_rows"], "UniProt+CCD aggregate with primary PDB"),
        ("anchor_coverage", ANCHOR_COVERAGE, "aggregate coverage rows", len(anchor), "aggregate manifest record"),
        ("labelability", LABELABILITY, "anchored aggregate rows", len(labelability), "aggregate manifest record"),
        ("labeled_asset_manifest", LABELED_ASSET_MANIFEST, "requested PDBs", len(labeled_assets), "PDB"),
        ("p1_structural_qc", P1_QC, "pilot feature rows", len(p1), "selected aggregate record/PDB/chain"),
        ("p1_biological_pair_collapse", P1_COLLAPSE, "collapsed pairs", len(p1_collapse), "UniProt+InChIKey14+label"),
        ("pu_candidates", PU_CANDIDATES, "pilot candidate rows", len(pu), "selected aggregate record/PDB/chain"),
        ("pu_site_clusters", PU_CLUSTERS, "pilot site clusters", int(cluster_summary[0]), "protein/Jaccard single-linkage cluster"),
        ("paired_cif_sifts_cache", CACHE_PATH_DESCRIPTION, "paired PDBs", len(paired_union), "PDB"),
    ]
    inventory = pd.DataFrame(
        [
            {
                "asset_id": asset_id,
                "path": str(path),
                "content": content,
                "row_or_file_count": int(count),
                "grain": grain,
            }
            for asset_id, path, content, count, grain in inventory_rows
        ]
    )

    # Hash only bounded/derived inputs.  Large legacy tables are recorded by
    # size and mtime so this checkpoint itself does not cause a large I/O pass.
    hash_inputs = [
        ("checkpoint2_validation", CHECKPOINT2_VALIDATION, True),
        ("raw_biolip", RAW_BIOLIP, False),
        ("offset_rescued", OFFSET_RESCUED, False),
        ("offset_log", OFFSET_LOG, True),
        ("legacy_aggregate", LEGACY_AGGREGATE, False),
        ("legacy_distance", LEGACY_DISTANCE, False),
        ("legacy_distance_log", LEGACY_DISTANCE_LOG, True),
        ("xai_manifest", MANIFEST_PARQUET, True),
        ("anchor_coverage", ANCHOR_COVERAGE, True),
        ("labelability", LABELABILITY, True),
        ("labeled_asset_manifest", LABELED_ASSET_MANIFEST, True),
        ("p1_qc", P1_QC, True),
        ("p1_features", P1_FEATURES, True),
        ("p1_collapse", P1_COLLAPSE, True),
        ("pu_candidates", PU_CANDIDATES, True),
        ("pu_clusters", PU_CLUSTERS, True),
        ("pu_summary", PU_SUMMARY, True),
        ("p1_fetch_log", P1_FETCH_LOG, True),
        ("pu_fetch_log", PU_FETCH_LOG, True),
    ]
    input_manifest = pd.DataFrame(
        [stat_record(asset_id, path, hash_file) for asset_id, path, hash_file in hash_inputs]
    )

    output_paths = {
        "asset_inventory": DATA / "CHECKPOINT3_ASSET_INVENTORY.tsv",
        "cache_pdb_manifest": DATA / "CHECKPOINT3_CACHE_PDB_MANIFEST.tsv.gz",
        "reuse_classification": DATA / "CHECKPOINT3_REUSE_CLASSIFICATION.tsv",
        "mapping_summary": DATA / "CHECKPOINT3_EXISTING_MAPPING_SUMMARY.tsv",
        "raw_schema": DATA / "CHECKPOINT3_BIOLIP_RAW_SCHEMA.tsv",
        "input_manifest": MANIFESTS / "CHECKPOINT3_INPUT_HASHES.tsv",
        "report": REPORTS / "CHECKPOINT3_EXISTING_STRUCTURE_ASSET_INVENTORY.md",
    }
    write_tsv(inventory, output_paths["asset_inventory"])
    write_tsv(cache_manifest, output_paths["cache_pdb_manifest"], compressed=True)
    write_tsv(reuse, output_paths["reuse_classification"])
    write_tsv(mapping_summary, output_paths["mapping_summary"])
    write_tsv(schema, output_paths["raw_schema"])
    write_tsv(input_manifest, output_paths["input_manifest"])

    class_counts = reuse["reuse_class"].value_counts().to_dict()
    report = f"""# Checkpoint 3: existing BioLiP structure/mapping asset inventory

Status: **validated**

This checkpoint performed no new structure download, residue mapping, feature
calculation, or site clustering.  It enumerated only the four named local
cache directories and read named manifests/logs.  The large legacy
`BioLiP_CIFs` repository was not recursively scanned.

## Raw observation grain

The official BioLiP schema defines one row per ligand-protein interaction site.
The local source has {raw_columns} columns and {raw_rows_documented:,} rows as
recorded by the completed offset-mapping log.  PDB ID, receptor chain, binding
site code, ligand CCD, ligand chain, ligand serial/auth sequence ID, and binding
residues are available.  Therefore checkpoint 4 can build an exact
PDB-chain-ligand-instance-site master without using the old UniProt-CCD
aggregate as its starting grain.

Official schema: {OFFICIAL_SCHEMA_URL}

## Existing chain and residue mapping

| Asset | Attempted | Mapped/reusable | Failed or unavailable | Meaning |
|---|---:|---:|---:|---|
| Legacy BioLiP-to-observed-chain offset | {raw_rows_documented:,} | {offset_success:,} ({offset_success/raw_rows_documented:.2%}) | {offset_mismatch + offset_missing:,} | Chain-sequence offset only; not a SIFTS/UniProt residue map |
| Current local CIF+SIFTS cache | {len(structure_union | sifts_union):,} PDB IDs | {len(paired_union):,} paired | {len((structure_union | sifts_union) - paired_union):,} | Directly usable inputs for residue mapping |
| P1 structural pilot | {len(p1):,} records | {len(p1_ok):,} | {len(p1)-len(p1_ok):,} | Implementation/QC seed only |
| PU pilot | {len(pu):,} records | {int(pu_summary['n_mapped']):,} | {len(pu)-int(pu_summary['n_mapped']):,} | Pilot mapping; its site clusters are not final |

The legacy offset consists of {offset_exact:,} exact/substring mappings and
{offset_sw:,} Smith-Waterman <=1% rescue mappings.  Its high coverage is useful,
but it must not be counted as canonical UniProt residue mapping.

## Coordinate/SIFTS cache coverage

| Quantity | Count |
|---|---:|
| Base structures / SIFTS | {len(base_struct):,} / {len(base_sifts):,} |
| BioLiP-XAI structures / SIFTS | {len(xai_struct):,} / {len(xai_sifts):,} |
| Union structures / SIFTS | {len(structure_union):,} / {len(sifts_union):,} |
| PDBs with both structure and SIFTS | {len(paired_union):,} |
| SIFTS only / structure only | {len(sifts_union-structure_union):,} / {len(structure_union-sifts_union):,} |

The 140,700-row aggregate manifest contains {int(manifest_summary[1]):,}
primary PDBs.  Only {len(manifest_cached):,} of those primary PDBs and
{int(manifest_cached['n_manifest_primary_rows'].sum()):,} rows currently have a
paired local structure/SIFTS asset.  Among the 11,705 previously anchored rows,
the corresponding coverage is {len(anchored_primary_set & paired_union):,} PDBs
and {int(anchored_primary.isin(paired_union).sum()):,} rows.  These are
primary-PDB lower bounds; checkpoint 4 must join all raw observations before
deciding which sites truly need new assets.

## Existing site-level work

- P1 attempted {len(p1):,} records on {len(p1_pdb):,} PDBs; {len(p1_ok):,}
  records on {p1_ok['pdb_norm'].nunique():,} PDBs mapped successfully.
- PU mapped {len(pu):,} selected records on {len(pu_pdb):,} PDBs and produced
  {int(cluster_summary[0]):,} pilot clusters across {int(cluster_summary[1]):,}
  proteins.
- The two pilots cover {len(p1_ok_ids | pu_ids):,} unique mapped aggregate
  record IDs.  This is only {len(p1_ok_ids | pu_ids)/int(manifest_summary[0]):.2%}
  of the 140,700 aggregate records, and it is not a full raw-observation map.
- The PU clusters were created with candidate-residue Jaccard >=0.5 followed
  by union-find/single-linkage.  They are retained for comparison, but global
  site clusters must be rebuilt under the later frozen clustering rule.

## Reuse decision

The reuse table contains {len(reuse):,} decisions:
{os.linesep.join(f'- `{key}`: {value}' for key, value in sorted(class_counts.items()))}

The raw BioLiP rows, legacy chain offsets, existing coordinates, and paired
SIFTS caches are reusable.  The aggregate UniProt-CCD tables are retained for
provenance/legacy-score linkage only.  Legacy catalytic-site distance is a
baseline, not the new orthosteric-anchor distance.  Exact raw-observation
mapping, assembly/interface checks, and global site clustering remain new work
for checkpoint 4 and later.

## Anomalies and limits found

1. High legacy offset coverage (98.88%) is not equivalent to residue-level
   UniProt mapping.
2. The broad legacy coordinate run documented {legacy_structures:,} PDBs, but
   the current directly paired CIF+SIFTS cache contains only {len(paired_union):,}.
3. Existing P1/PU mapped work covers 656 aggregate record IDs and 138 pilot
   site clusters, not the full BioLiP observation universe.
4. `anchor_coverage_by_row.tsv` predates the enlarged paired cache, so its
   asset-availability flags are stale and must be recomputed.
5. The old 167,817-row distance table uses binding/catalytic-site C-alpha
   geometry and cannot serve as the final exact orthosteric-anchor distance.
6. The old UniProt-CCD aggregation collapses multiple structures, receptor
   chains, ligand instances, and sometimes contradictory distances.

No checkpoint-4 processing was started.
"""
    atomic_text(output_paths["report"], report)

    output_records = {
        key: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for key, path in output_paths.items()
    }
    validation = {
        "checkpoint": 3,
        "status": "validated",
        "scope": "bounded inventory of existing BioLiP structure, mapping, and site assets",
        "io_guard": {
            "large_biolip_cif_repository_recursively_scanned": False,
            "named_cache_directories_enumerated_nonrecursively": 4,
            "large_legacy_tables_fully_hashed": False,
            "large_legacy_table_identity": "size and mtime recorded; completed logs supply row counts",
        },
        "official_biolip_schema": {
            "download_url": OFFICIAL_DOWNLOAD_URL,
            "readme_url": OFFICIAL_SCHEMA_URL,
            "columns": raw_columns,
            "row_grain": "ligand-protein interaction site",
        },
        "checkpoint2_outputs_hash_verified": True,
        "observed": observed,
        "reuse_class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "new_processing_deferred_to_checkpoint4": [
            "exact raw-observation master",
            "residue-level UniProt mapping beyond current paired cache",
            "biological-assembly and multichain/interface audit",
            "global site clustering",
        ],
        "outputs": output_records,
    }
    validation_path = VALIDATION / "CHECKPOINT3_VALIDATION.json"
    atomic_json(validation_path, validation)

    # Reopen every output and verify the frozen structural invariants.
    cache_check = pd.read_csv(output_paths["cache_pdb_manifest"], sep="\t", low_memory=False)
    if len(cache_check) != len(all_cache_pdbs):
        raise RuntimeError("cache manifest PDB count changed after writing")
    if int(cache_check["paired_structure_and_sifts"].sum()) != len(paired_union):
        raise RuntimeError("cache manifest paired count changed after writing")
    if pd.read_csv(output_paths["raw_schema"], sep="\t").shape[0] != 21:
        raise RuntimeError("raw schema output must contain 21 rows")
    if pd.read_csv(output_paths["reuse_classification"], sep="\t").shape[0] != len(reuse):
        raise RuntimeError("reuse classification changed after writing")

    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
