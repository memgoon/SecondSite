#!/usr/bin/env python3
"""Validate checkpoint-4 sequence mapping against direct author-residue mapping.

Only PDBs already listed as paired structure+SIFTS assets by checkpoint 3 are
opened.  This is an independent coordinate-path check:

raw BioLiP author residue -> mmCIF label_seq -> local SIFTS JSON -> UniProt.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import duckdb
import gemmi
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
REPORTS = PACKAGE / "reports"
VALIDATION = PACKAGE / "validation"

BUILD_SUMMARY = VALIDATION / "CHECKPOINT4A_BUILD_SUMMARY.json"
MASTER = DATA / "BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz"
CACHE_MANIFEST = DATA / "CHECKPOINT3_CACHE_PDB_MANIFEST.tsv.gz"

BASE_STRUCTURES = ROOT / "data/structures"
BASE_SIFTS = ROOT / "analysis/sifts"
XAI_STRUCTURES = ROOT / "analysis/structure_known_biolip_xai/assets/structures"
XAI_SIFTS = ROOT / "analysis/structure_known_biolip_xai/assets/sifts"

AUDIT_OUT = DATA / "CHECKPOINT4_LOCAL_AUTHOR_MAPPING_VALIDATION.tsv.gz"
SUMMARY_OUT = VALIDATION / "CHECKPOINT4B_LOCAL_AUTHOR_VALIDATION.json"

TOKEN = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=6, mtime=0) as raw:
        frame.to_csv(raw, sep="\t", index=False)
    os.replace(temporary, path)


def base_accession(value: str) -> str:
    return (value or "").strip().upper().split("-", 1)[0]


def parse_auth_tokens(value: str) -> Tuple[List[Tuple[int, str]], List[str]]:
    parsed: List[Tuple[int, str]] = []
    invalid: List[str] = []
    for token in (value or "").split():
        match = TOKEN.fullmatch(token)
        if not match:
            invalid.append(token)
            continue
        parsed.append((int(match.group(2)), match.group(3).upper()))
    return parsed, invalid


def parse_positions(value: str) -> List[Optional[int]]:
    return [int(token) if token else None for token in str(value or "").split(";")]


def choose_paths(row: Mapping[str, object]) -> Tuple[Path, Path, str]:
    pdb = str(row["pdb"])
    if bool(row["base_structure_present"]) and bool(row["base_sifts_present"]):
        return BASE_STRUCTURES / f"{pdb}.cif", BASE_SIFTS / f"{pdb}.json", "base_cache"
    if bool(row["xai_structure_present"]) and bool(row["xai_sifts_present"]):
        return XAI_STRUCTURES / f"{pdb}.cif", XAI_SIFTS / f"{pdb}.json", "xai_cache"
    raise RuntimeError(f"paired cache row has no co-located structure/SIFTS pair: {pdb}")


def label_to_uniprot_map(sifts_json: Mapping[str, object], accession: str, chain: str) -> Dict[int, int]:
    desired_base = base_accession(accession)
    candidates: Dict[int, Set[int]] = defaultdict(set)
    for key, raw_segments in sifts_json.items():
        if base_accession(str(key)) != desired_base or not isinstance(raw_segments, list):
            continue
        for segment in raw_segments:
            if not isinstance(segment, dict) or str(segment.get("chain") or "") != chain:
                continue
            try:
                label_start = int(segment["label_start"])
                label_end = int(segment["label_end"])
                unp_start = int(segment["unp_start"])
                unp_end = int(segment["unp_end"])
            except (KeyError, TypeError, ValueError):
                continue
            if (label_end - label_start) != (unp_end - unp_start):
                continue
            for label in range(label_start, label_end + 1):
                candidates[label].add(unp_start + (label - label_start))
    return {label: next(iter(values)) for label, values in candidates.items() if len(values) == 1}


def build_author_map(structure_path: Path, chain_name: str, label_map: Mapping[int, int]) -> Tuple[Dict[Tuple[int, str], int], str]:
    try:
        structure = gemmi.read_structure(str(structure_path))
        structure.setup_entities()
    except Exception as exc:
        return {}, f"structure_parse_failed:{type(exc).__name__}"
    if len(structure) == 0:
        return {}, "structure_has_no_model"
    chain = structure[0].find_chain(chain_name)
    if chain is None:
        return {}, "receptor_chain_not_found"
    candidates: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    for residue in chain:
        if residue.label_seq is None:
            continue
        uniprot = label_map.get(int(residue.label_seq))
        if uniprot is None:
            continue
        icode = str(residue.seqid.icode or "").strip().upper()
        candidates[(int(residue.seqid.num), icode)].add(int(uniprot))
    result = {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}
    return result, "ok" if result else "no_author_residues_mapped"


def map_author_tokens(tokens: Sequence[Tuple[int, str]], author_map: Mapping[Tuple[int, str], int]) -> Tuple[List[Optional[int]], int]:
    mapped: List[Optional[int]] = []
    ambiguous = 0
    by_number: Dict[int, Set[int]] = defaultdict(set)
    for (number, _icode), uniprot in author_map.items():
        by_number[number].add(uniprot)
    for number, icode in tokens:
        exact = author_map.get((number, icode))
        if exact is not None:
            mapped.append(exact)
        elif not icode and len(by_number.get(number, set())) == 1:
            mapped.append(next(iter(by_number[number])))
        else:
            if len(by_number.get(number, set())) > 1:
                ambiguous += 1
            mapped.append(None)
    return mapped, ambiguous


def main() -> None:
    for path in (BUILD_SUMMARY, MASTER, CACHE_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    build = json.loads(BUILD_SUMMARY.read_text())
    if build.get("status") != "validated" or build.get("stage") != "checkpoint4a_exact_observation_master":
        raise RuntimeError("checkpoint 4a is not validated")
    for record in build.get("outputs", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint-4a output hash mismatch: {path}")

    cache = pd.read_csv(CACHE_MANIFEST, sep="\t", low_memory=False)
    cache = cache.loc[cache["paired_structure_and_sifts"].astype(bool)].copy()
    cache["pdb"] = cache["pdb"].astype(str).str.lower()
    cache_lookup = cache.set_index("pdb").to_dict("index")

    con = duckdb.connect(database=":memory:")
    con.register("paired_cache", cache[["pdb"]])
    observations = con.execute(
        """
        SELECT
          CAST(source_ordinal AS BIGINT) AS source_ordinal,
          observation_id,
          lower(pdb_id) AS pdb_id,
          receptor_chain,
          uniprot_resolved,
          binding_residues_auth_raw,
          binding_uniprot_positions,
          site_mapping_status
        FROM read_csv_auto(?, delim='\t', header=true, compression='gzip', all_varchar=true, sample_size=-1)
        INNER JOIN paired_cache ON lower(pdb_id) = paired_cache.pdb
        WHERE lower(mapping_usable_for_site_clustering) = 'true'
        ORDER BY pdb_id, source_ordinal
        """,
        [str(MASTER)],
    ).fetchdf()
    con.close()

    counters = Counter()
    output_rows: List[Dict[str, object]] = []
    for pdb, group in observations.groupby("pdb_id", sort=True):
        cache_row = cache_lookup[pdb]
        structure_path, sifts_path, cache_source = choose_paths({"pdb": pdb, **cache_row})
        try:
            sifts_json = json.loads(sifts_path.read_text())
        except Exception as exc:
            for row in group.itertuples(index=False):
                output_rows.append(
                    {
                        "source_ordinal": row.source_ordinal,
                        "observation_id": row.observation_id,
                        "pdb_id": pdb,
                        "receptor_chain": row.receptor_chain,
                        "uniprot_resolved": row.uniprot_resolved,
                        "sequence_path_positions": row.binding_uniprot_positions,
                        "direct_author_positions": "",
                        "direct_mapping_fraction": 0.0,
                        "validation_status": f"sifts_json_failed:{type(exc).__name__}",
                        "ambiguous_author_tokens": 0,
                        "cache_source": cache_source,
                    }
                )
                counters[f"sifts_json_failed:{type(exc).__name__}"] += 1
            continue

        chain_cache: Dict[Tuple[str, str], Tuple[Dict[Tuple[int, str], int], str]] = {}
        for row in group.itertuples(index=False):
            key = (str(row.receptor_chain), str(row.uniprot_resolved))
            if key not in chain_cache:
                label_map = label_to_uniprot_map(sifts_json, key[1], key[0])
                if not label_map:
                    chain_cache[key] = ({}, "no_matching_sifts_json_segment")
                else:
                    chain_cache[key] = build_author_map(structure_path, key[0], label_map)
            author_map, map_status = chain_cache[key]
            auth_tokens, invalid_tokens = parse_auth_tokens(str(row.binding_residues_auth_raw or ""))
            direct, ambiguous = map_author_tokens(auth_tokens, author_map) if author_map else ([None] * len(auth_tokens), 0)
            sequence_positions = parse_positions(str(row.binding_uniprot_positions or ""))
            direct_count = sum(value is not None for value in direct)
            fraction = direct_count / len(auth_tokens) if auth_tokens else 0.0

            if invalid_tokens:
                status = "invalid_author_residue_token"
            elif map_status != "ok":
                status = map_status
            elif len(sequence_positions) != len(direct):
                status = "token_count_mismatch"
            elif all(value is not None for value in direct) and direct == sequence_positions:
                status = "exact_full_agreement"
            elif all(value is not None for value in direct):
                status = "complete_position_disagreement"
            elif direct_count:
                shared = [
                    a == b
                    for a, b in zip(direct, sequence_positions)
                    if a is not None and b is not None
                ]
                status = "partial_agreement" if shared and all(shared) else "partial_disagreement"
            else:
                status = "direct_author_mapping_unavailable"

            output_rows.append(
                {
                    "source_ordinal": int(row.source_ordinal),
                    "observation_id": row.observation_id,
                    "pdb_id": pdb,
                    "receptor_chain": row.receptor_chain,
                    "uniprot_resolved": row.uniprot_resolved,
                    "sequence_path_positions": row.binding_uniprot_positions,
                    "direct_author_positions": ";".join("" if value is None else str(value) for value in direct),
                    "direct_mapping_fraction": round(fraction, 6),
                    "validation_status": status,
                    "ambiguous_author_tokens": ambiguous,
                    "cache_source": cache_source,
                }
            )
            counters[status] += 1

    output = pd.DataFrame(output_rows)
    write_gzip(output, AUDIT_OUT)
    exact = counters["exact_full_agreement"]
    complete_disagree = counters["complete_position_disagreement"]
    directly_complete = exact + complete_disagree
    summary = {
        "stage": "checkpoint4b_cached_author_mapping_validation",
        "status": "validated",
        "paired_cache_pdbs": int(len(cache)),
        "eligible_observations_on_paired_cache": int(len(observations)),
        "validation_status_counts": dict(counters),
        "directly_complete_rows": int(directly_complete),
        "exact_full_agreement_rows": int(exact),
        "complete_position_disagreement_rows": int(complete_disagree),
        "exact_agreement_fraction_among_directly_complete": (
            exact / directly_complete if directly_complete else None
        ),
        "output": {
            "path": str(AUDIT_OUT),
            "bytes": AUDIT_OUT.stat().st_size,
            "sha256": sha256(AUDIT_OUT),
        },
    }
    atomic_json(SUMMARY_OUT, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
