#!/usr/bin/env python3
"""Build an exact BioLiP observation/site master at source-row resolution.

The source is read in one synchronized pass with its legacy offset derivative.
No structure directory is traversed.  CIF-chain FASTA files are opened only by
the PDB IDs encountered in the sorted BioLiP source.

Primary residue mapping path
----------------------------
BioLiP observed-sequence position -> CIF polymer label_seq position -> bulk
SIFTS segment -> UniProt residue.

An independent alignment of the BioLiP observed sequence to the cached
canonical UniProt sequence is retained as a consistency check/rescue.  The old
single integer offset is audited but is not blindly applied to SW-rescued rows.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd
from Bio import Align


ROOT = Path("/disk9/13.Heesu_Allostery")
SHARED = Path("/shared_data/11.HS_allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
REPORTS = PACKAGE / "reports"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

CHECKPOINT3_VALIDATION = VALIDATION / "CHECKPOINT3_VALIDATION.json"
RAW_BIOLIP = SHARED / "Data/BioLiP.txt"
OFFSET_BIOLIP = SHARED / "13.Refine_BioLiP/BioLiP_offset_rescued.tsv"
FASTA_DIR = SHARED / "BioLiP_CIFs/FASTA"
BULK_SIFTS = SHARED / "Data/SIFTS_pdb_chain_uniprot.tsv"
UNIPROT_CACHE = SHARED / "Data/uniprot_cache.pickle"
WWPDB_COMPOUNDS = SHARED / "Data/wwpdb_compounds.pickle"

ACTIVE_ORTHO = DATA / "ORTHOSTERIC_PAIR_MASTER_AUDITED.tsv.gz"
QUARANTINE_ORTHO = DATA / "ORTHOSTERIC_PAIR_AUDIT_QUARANTINE.tsv.gz"

MASTER_OUT = DATA / "BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz"
SEQUENCE_OUT = DATA / "BIOLIP_RECEPTOR_SEQUENCE_MANIFEST.tsv.gz"
FASTA_INPUT_OUT = MANIFESTS / "CHECKPOINT4_FASTA_INPUT_MANIFEST.tsv.gz"
DUPLICATE_OUT = DATA / "CHECKPOINT4_DUPLICATE_NATURAL_OBSERVATIONS.tsv.gz"
STATUS_OUT = DATA / "CHECKPOINT4_MAPPING_STATUS_COUNTS.tsv"
UNIPROT_OUT = DATA / "CHECKPOINT4_UNIPROT_RESOLUTION_COUNTS.tsv"
ALIGNMENT_OUT = DATA / "CHECKPOINT4_ALIGNMENT_AUDIT.tsv"
INPUT_HASHES_OUT = MANIFESTS / "CHECKPOINT4A_INPUT_HASHES.tsv"
SUMMARY_OUT = VALIDATION / "CHECKPOINT4A_BUILD_SUMMARY.json"

RAW_COLUMNS = [
    "pdb_id",
    "receptor_chain",
    "resolution",
    "binding_site_code",
    "ligand_ccd",
    "ligand_chain",
    "ligand_serial",
    "binding_residues_auth_raw",
    "binding_residues_seq_raw",
    "catalytic_residues_auth_raw",
    "catalytic_residues_seq_raw",
    "ec_number",
    "go_terms",
    "affinity_manual",
    "affinity_moad",
    "affinity_pdbbind_cn",
    "affinity_bindingdb",
    "uniprot_raw",
    "pubmed_id",
    "ligand_auth_seq_id",
    "receptor_sequence",
]

MASTER_COLUMNS = [
    "source_ordinal",
    "observation_id",
    "natural_observation_key",
    "pdb_id",
    "receptor_chain",
    "resolution",
    "binding_site_code",
    "ligand_ccd",
    "ligand_chain",
    "ligand_serial",
    "ligand_auth_seq_id",
    "uniprot_raw",
    "uniprot_raw_candidates",
    "uniprot_resolved",
    "uniprot_resolution_status",
    "pubmed_id",
    "ec_number",
    "go_terms",
    "affinity_manual",
    "affinity_moad",
    "affinity_pdbbind_cn",
    "affinity_bindingdb",
    "full_inchikey",
    "connectivity_key",
    "canonical_smiles",
    "chemical_mapping_status",
    "active_orthosteric_pair_reference",
    "quarantined_orthosteric_pair_reference",
    "binding_residues_auth_raw",
    "binding_residues_seq_raw",
    "catalytic_residues_auth_raw",
    "catalytic_residues_seq_raw",
    "binding_residue_count",
    "binding_seq_positions",
    "binding_label_seq_positions",
    "binding_uniprot_positions",
    "binding_residue_mapping_fraction",
    "site_mapping_status",
    "mapping_usable_for_site_clustering",
    "mapping_agreement_status",
    "cif_fasta_status",
    "biolip_to_cif_alignment_mode",
    "biolip_to_cif_match_fraction",
    "biolip_to_cif_query_coverage",
    "biolip_to_cif_mapping_trusted",
    "stored_legacy_offset",
    "recomputed_alignment_offset",
    "legacy_offset_agreement",
    "sifts_segment_status",
    "sifts_candidate_accessions",
    "canonical_alignment_status",
    "canonical_alignment_match_fraction",
    "canonical_alignment_query_coverage",
    "canonical_alignment_trusted",
    "uniprot_residue_letter_matches",
    "uniprot_residue_letter_mismatches",
    "receptor_sequence_id",
    "receptor_sequence_length",
]

RESIDUE_TOKEN = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)$")
ACCESSION_SPLIT = re.compile(r"[,;|\s]+")
VALID_ACCESSION = re.compile(r"^[A-Z0-9]+(?:-[0-9]+)?$")
INVALID_VALUES = {"", "-", "?", ".", "none", "nan"}

EXPECTED_SOURCE_ROWS = 989058


@dataclass(frozen=True)
class Segment:
    accession: str
    res_beg: int
    res_end: int
    sp_beg: int
    sp_end: int


@dataclass
class AlignmentResult:
    mode: str
    position_map: Dict[int, int]
    match_fraction: float
    query_coverage: float
    aligned_identity: float
    offset: Optional[int]
    trusted_cif_mapping: bool


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


class AtomicGzipTSV:
    def __init__(self, path: Path, columns: Sequence[str]):
        self.path = path
        self.columns = list(columns)
        self.temporary = path.with_name(path.name + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = gzip.GzipFile(filename=str(self.temporary), mode="wb", compresslevel=6, mtime=0)
        self.text = io.TextIOWrapper(self.raw, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.text,
            fieldnames=self.columns,
            delimiter="\t",
            extrasaction="raise",
            lineterminator="\n",
        )
        self.writer.writeheader()

    def write(self, row: Mapping[str, object]) -> None:
        self.writer.writerow({key: row.get(key, "") for key in self.columns})

    def close(self, commit: bool = True) -> None:
        try:
            self.text.flush()
            self.text.close()
        finally:
            if commit:
                os.replace(self.temporary, self.path)
            elif self.temporary.exists():
                self.temporary.unlink()


def write_tsv(frame: pd.DataFrame, path: Path, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if compressed:
        with gzip.GzipFile(filename=str(temporary), mode="wb", compresslevel=6, mtime=0) as raw:
            frame.to_csv(raw, sep="\t", index=False)
    else:
        frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def normalize_accession(value: str) -> str:
    return value.strip().upper()


def accession_base(value: str) -> str:
    return normalize_accession(value).split("-", 1)[0]


def parse_accessions(value: str) -> List[str]:
    tokens: List[str] = []
    seen: Set[str] = set()
    for token in ACCESSION_SPLIT.split((value or "").strip().upper()):
        if token.lower() in INVALID_VALUES or not VALID_ACCESSION.fullmatch(token):
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def parse_residue_tokens(value: str) -> Tuple[List[str], List[str], List[int], List[str]]:
    raw_tokens = [token.strip() for token in (value or "").split() if token.strip()]
    letters: List[str] = []
    positions: List[int] = []
    insertion_codes: List[str] = []
    invalid: List[str] = []
    for token in raw_tokens:
        match = RESIDUE_TOKEN.fullmatch(token)
        if not match:
            invalid.append(token)
            continue
        letters.append(match.group(1).upper())
        positions.append(int(match.group(2)))
        insertion_codes.append(match.group(3).upper())
    return raw_tokens, letters, positions, invalid


def stable_id(prefix: str, value: str, length: int = 20) -> str:
    return prefix + hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def load_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                identifier = line[1:].split()[0]
                current = identifier.rsplit(":", 1)[-1]
                records.setdefault(current, [])
            elif current is not None:
                records[current].append(line.upper())
    return {chain: "".join(parts) for chain, parts in records.items()}


def make_aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    return aligner


ALIGNER = make_aligner()


def build_alignment_map(target: str, query: str, purpose: str) -> AlignmentResult:
    target = (target or "").upper()
    query = (query or "").upper()
    if not target or not query:
        return AlignmentResult("missing_sequence", {}, 0.0, 0.0, 0.0, None, False)

    if target == query:
        mapping = {index: index for index in range(1, len(query) + 1)}
        return AlignmentResult("exact_equal", mapping, 1.0, 1.0, 1.0, 0, True)

    start = target.find(query)
    if start >= 0:
        mapping = {index: start + index for index in range(1, len(query) + 1)}
        return AlignmentResult("query_substring_of_target", mapping, 1.0, 1.0, 1.0, start, True)

    start = query.find(target)
    if start >= 0:
        mapping = {start + index: index for index in range(1, len(target) + 1)}
        coverage = len(mapping) / len(query)
        return AlignmentResult("target_substring_of_query", mapping, coverage, coverage, 1.0, -start, True)

    try:
        alignment = ALIGNER.align(target, query)[0]
    except Exception:
        return AlignmentResult("alignment_failed", {}, 0.0, 0.0, 0.0, None, False)

    mapping: Dict[int, int] = {}
    matches = 0
    aligned_pairs = 0
    for target_block, query_block in zip(alignment.aligned[0], alignment.aligned[1]):
        t_start, t_end = int(target_block[0]), int(target_block[1])
        q_start, q_end = int(query_block[0]), int(query_block[1])
        length = min(t_end - t_start, q_end - q_start)
        for offset in range(length):
            target_index = t_start + offset
            query_index = q_start + offset
            mapping[query_index + 1] = target_index + 1
            aligned_pairs += 1
            if target[target_index] == query[query_index]:
                matches += 1

    match_fraction = matches / len(query) if query else 0.0
    coverage = len(mapping) / len(query) if query else 0.0
    identity = matches / aligned_pairs if aligned_pairs else 0.0
    offset: Optional[int] = None
    if len(alignment.aligned[0]) and len(alignment.aligned[1]):
        offset = int(alignment.aligned[0][0][0] - alignment.aligned[1][0][0])

    # This reproduces the legacy <=1% mutation gate for CIF mapping.  For
    # canonical alignment, the status is reported separately and binding-site
    # agreement decides whether it may act as a rescue.
    trusted = match_fraction >= 0.99 if purpose == "cif" else identity >= 0.95
    return AlignmentResult("smith_waterman", mapping, match_fraction, coverage, identity, offset, trusted)


def load_bulk_sifts(path: Path) -> Tuple[Dict[Tuple[str, str], List[Segment]], Dict[str, int]]:
    mapping: Dict[Tuple[str, str], List[Segment]] = defaultdict(list)
    stats = Counter()
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        # Skip provenance comment(s), then let DictReader consume the header.
        lines = (line for line in handle if not line.startswith("#"))
        reader = csv.DictReader(lines, delimiter="\t")
        for row in reader:
            stats["rows"] += 1
            try:
                res_beg = int(row["RES_BEG"])
                res_end = int(row["RES_END"])
                sp_beg = int(row["SP_BEG"])
                sp_end = int(row["SP_END"])
            except (TypeError, ValueError):
                stats["invalid_numeric_segment"] += 1
                continue
            if (res_end - res_beg) != (sp_end - sp_beg):
                stats["nonlinear_segment"] += 1
                continue
            pdb = row["PDB"].strip().lower()
            chain = row["CHAIN"].strip()
            accession = normalize_accession(row["SP_PRIMARY"])
            mapping[(pdb, chain)].append(Segment(accession, res_beg, res_end, sp_beg, sp_end))
            stats["usable_segments"] += 1
    stats["pdb_chain_keys"] = len(mapping)
    stats["pdbs"] = len({key[0] for key in mapping})
    return mapping, dict(stats)


def map_label_positions(
    positions: Sequence[Optional[int]],
    segments: Sequence[Segment],
    raw_accessions: Sequence[str],
) -> Dict[str, object]:
    accessions = sorted({segment.accession for segment in segments})
    raw_bases = {accession_base(value) for value in raw_accessions}
    candidates: Dict[str, List[Optional[int]]] = {}
    ambiguous_within_accession: Set[str] = set()

    for accession in accessions:
        mapped: List[Optional[int]] = []
        for position in positions:
            if position is None:
                mapped.append(None)
                continue
            hits = {
                segment.sp_beg + (position - segment.res_beg)
                for segment in segments
                if segment.accession == accession and segment.res_beg <= position <= segment.res_end
            }
            if len(hits) == 1:
                mapped.append(next(iter(hits)))
            else:
                mapped.append(None)
                if len(hits) > 1:
                    ambiguous_within_accession.add(accession)
        candidates[accession] = mapped

    if not candidates:
        return {
            "resolved_accession": "",
            "mapped_positions": [None] * len(positions),
            "status": "no_sifts_segments",
            "candidate_accessions": [],
        }

    coverage = {accession: sum(value is not None for value in values) for accession, values in candidates.items()}
    complete = [accession for accession, count in coverage.items() if count == len(positions) and len(positions) > 0]
    preferred_complete = [accession for accession in complete if accession_base(accession) in raw_bases]

    if len(preferred_complete) == 1:
        chosen = preferred_complete[0]
        status = "complete_raw_accession_supported"
    elif len(complete) == 1:
        chosen = complete[0]
        status = "complete_unique_sifts_accession"
    elif len(preferred_complete) > 1 or len(complete) > 1:
        return {
            "resolved_accession": "",
            "mapped_positions": [None] * len(positions),
            "status": "ambiguous_multiple_complete_sifts_accessions",
            "candidate_accessions": accessions,
        }
    else:
        maximum = max(coverage.values()) if coverage else 0
        best = [accession for accession, count in coverage.items() if count == maximum]
        preferred = [accession for accession in best if accession_base(accession) in raw_bases]
        choice_pool = preferred if preferred else best
        if maximum > 0 and len(choice_pool) == 1:
            chosen = choice_pool[0]
            status = "partial_unique_sifts_accession"
        else:
            return {
                "resolved_accession": "",
                "mapped_positions": [None] * len(positions),
                "status": "no_unique_sifts_accession",
                "candidate_accessions": accessions,
            }

    if chosen in ambiguous_within_accession:
        status += "_with_overlapping_segment_ambiguity"
    return {
        "resolved_accession": chosen,
        "mapped_positions": candidates[chosen],
        "status": status,
        "candidate_accessions": accessions,
    }


def cache_sequence(uniprot_cache: Mapping[str, object], accession: str) -> Tuple[str, str]:
    for key in (normalize_accession(accession), accession_base(accession)):
        value = uniprot_cache.get(key)
        if isinstance(value, dict) and value.get("sequence"):
            return key, str(value["sequence"]).upper()
        if isinstance(value, str) and value:
            return key, value.upper()
    return "", ""


def map_to_canonical(
    query_sequence: str,
    binding_positions: Sequence[int],
    binding_letters: Sequence[str],
    candidate_accessions: Sequence[str],
    preferred_accession: str,
    uniprot_cache: Mapping[str, object],
    alignment_cache: Optional[MutableMapping[Tuple[str, str], AlignmentResult]] = None,
) -> Dict[str, object]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for accession in ([preferred_accession] if preferred_accession else []) + list(candidate_accessions):
        accession = normalize_accession(accession)
        if accession and accession not in seen:
            seen.add(accession)
            ordered.append(accession)

    results: List[Dict[str, object]] = []
    for accession in ordered:
        cache_key, sequence = cache_sequence(uniprot_cache, accession)
        if not sequence:
            continue
        cache_id = (accession, hashlib.sha256(query_sequence.encode("utf-8")).hexdigest())
        if alignment_cache is not None and cache_id in alignment_cache:
            alignment = alignment_cache[cache_id]
        else:
            alignment = build_alignment_map(sequence, query_sequence, purpose="canonical")
            if alignment_cache is not None:
                alignment_cache[cache_id] = alignment
        mapped = [alignment.position_map.get(position) for position in binding_positions]
        letter_matches = 0
        letter_mismatches = 0
        for letter, position in zip(binding_letters, mapped):
            if position is None or position < 1 or position > len(sequence):
                continue
            if sequence[position - 1] == letter:
                letter_matches += 1
            else:
                letter_mismatches += 1
        results.append(
            {
                "accession": accession,
                "cache_key": cache_key,
                "alignment": alignment,
                "mapped_positions": mapped,
                "mapped_count": sum(position is not None for position in mapped),
                "letter_matches": letter_matches,
                "letter_mismatches": letter_mismatches,
            }
        )

    if not results:
        return {
            "accession": "",
            "mapped_positions": [None] * len(binding_positions),
            "status": "canonical_sequence_unavailable",
            "alignment": AlignmentResult("unavailable", {}, 0.0, 0.0, 0.0, None, False),
            "letter_matches": 0,
            "letter_mismatches": 0,
        }

    if preferred_accession:
        preferred_base = accession_base(preferred_accession)
        preferred_results = [r for r in results if accession_base(str(r["accession"])) == preferred_base]
        if len(preferred_results) == 1:
            chosen = preferred_results[0]
            status = "canonical_preferred_accession"
        else:
            chosen = None
    else:
        chosen = None

    if chosen is None:
        def score(result: Mapping[str, object]) -> Tuple[int, int, int, float, float]:
            alignment = result["alignment"]
            assert isinstance(alignment, AlignmentResult)
            return (
                int(result["mapped_count"] == len(binding_positions) and len(binding_positions) > 0),
                int(result["letter_mismatches"] == 0),
                int(result["letter_matches"]),
                alignment.aligned_identity,
                alignment.match_fraction,
            )

        ranked = sorted(results, key=score, reverse=True)
        if len(ranked) > 1 and score(ranked[0]) == score(ranked[1]):
            return {
                "accession": "",
                "mapped_positions": [None] * len(binding_positions),
                "status": "canonical_alignment_ambiguous_tie",
                "alignment": ranked[0]["alignment"],
                "letter_matches": 0,
                "letter_mismatches": 0,
            }
        chosen = ranked[0]
        status = "canonical_unique_best_alignment"

    alignment = chosen["alignment"]
    assert isinstance(alignment, AlignmentResult)
    mapped_positions = list(chosen["mapped_positions"])
    complete = len(binding_positions) > 0 and all(value is not None for value in mapped_positions)
    no_letter_conflict = int(chosen["letter_mismatches"]) == 0
    if not complete:
        status += "_partial"
    elif not no_letter_conflict:
        status += "_residue_letter_conflict"
    elif alignment.aligned_identity < 0.95:
        status += "_low_identity"
    else:
        status += "_complete"

    return {
        "accession": str(chosen["accession"]),
        "mapped_positions": mapped_positions,
        "status": status,
        "alignment": alignment,
        "letter_matches": int(chosen["letter_matches"]),
        "letter_mismatches": int(chosen["letter_mismatches"]),
    }


def join_positions(values: Sequence[Optional[int]]) -> str:
    return ";".join("" if value is None else str(int(value)) for value in values)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def compound_fields(compounds: Mapping[str, object], ccd: str) -> Tuple[str, str, str, str]:
    record = compounds.get((ccd or "").upper())
    if not isinstance(record, dict):
        return "", "", "", "ccd_not_in_wwpdb_cache"
    descriptors = record.get("descriptors")
    if not isinstance(descriptors, dict):
        return "", "", "", "wwpdb_descriptors_missing"
    inchikey = str(descriptors.get("INCHIKEY") or "").replace("InChIKey=", "").strip()
    smiles = str(descriptors.get("SMILES_CANONICAL") or descriptors.get("SMILES") or "").strip()
    if not inchikey:
        return "", "", smiles, "wwpdb_inchikey_missing"
    return inchikey, inchikey[:14], smiles, "mapped_wwpdb_ccd"


def parse_stored_offset(value: str) -> Optional[int]:
    try:
        return int((value or "").strip())
    except (TypeError, ValueError):
        return None


def file_record(path: Path, digest: str = "") -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_epoch_seconds": int(stat.st_mtime),
        "sha256": digest or sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-every", type=int, default=50000)
    args = parser.parse_args()

    required = [
        CHECKPOINT3_VALIDATION,
        RAW_BIOLIP,
        OFFSET_BIOLIP,
        BULK_SIFTS,
        UNIPROT_CACHE,
        WWPDB_COMPOUNDS,
        ACTIVE_ORTHO,
        QUARANTINE_ORTHO,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not FASTA_DIR.is_dir():
        raise FileNotFoundError(FASTA_DIR)

    checkpoint3 = json.loads(CHECKPOINT3_VALIDATION.read_text())
    if checkpoint3.get("checkpoint") != 3 or checkpoint3.get("status") != "validated":
        raise RuntimeError("checkpoint 3 is not validated")
    for record in checkpoint3.get("outputs", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint-3 output hash mismatch: {path}")

    start_time = time.time()
    print("loading bulk SIFTS segments...", flush=True)
    sifts, sifts_stats = load_bulk_sifts(BULK_SIFTS)
    print(f"bulk SIFTS: {sifts_stats}", flush=True)
    with UNIPROT_CACHE.open("rb") as handle:
        uniprot_cache = pickle.load(handle)
    with WWPDB_COMPOUNDS.open("rb") as handle:
        compounds = pickle.load(handle)

    active = pd.read_csv(ACTIVE_ORTHO, sep="\t", usecols=["orthosteric_pair_key"])
    quarantine = pd.read_csv(QUARANTINE_ORTHO, sep="\t", usecols=["orthosteric_pair_key"])
    active_keys = set(active["orthosteric_pair_key"].astype(str))
    quarantine_keys = set(quarantine["orthosteric_pair_key"].astype(str))

    counters = Counter()
    alignment_modes = Counter()
    site_statuses = Counter()
    uniprot_statuses = Counter()
    sifts_statuses = Counter()
    agreement_statuses = Counter()
    chemical_statuses = Counter()
    natural_key_counts: Counter[str] = Counter()
    invalid_binding_token_examples: List[Dict[str, object]] = []

    master_writer = AtomicGzipTSV(MASTER_OUT, MASTER_COLUMNS)
    sequence_writer = AtomicGzipTSV(
        SEQUENCE_OUT,
        ["receptor_sequence_id", "pdb_id", "receptor_chain", "sequence_sha256", "sequence_length", "receptor_sequence"],
    )
    fasta_writer = AtomicGzipTSV(
        FASTA_INPUT_OUT,
        ["pdb_id", "path", "status", "bytes", "mtime_epoch_seconds", "sha256", "chains"],
    )

    sequence_ids_seen: Set[str] = set()
    pdbs_seen: Set[str] = set()
    previous_pdb = ""
    current_fasta: Dict[str, str] = {}
    current_alignment_cache: Dict[Tuple[str, str], AlignmentResult] = {}
    current_canonical_alignment_cache: Dict[Tuple[str, str], AlignmentResult] = {}
    raw_digest = hashlib.sha256()
    offset_digest = hashlib.sha256()
    completed = False

    try:
        with RAW_BIOLIP.open("rb") as raw_handle, OFFSET_BIOLIP.open("rb") as offset_handle:
            for ordinal, offset_line in enumerate(offset_handle):
                offset_digest.update(offset_line)
                raw_line = raw_handle.readline()
                if not raw_line:
                    raise RuntimeError(f"raw BioLiP ended before offset derivative at row {ordinal}")
                raw_digest.update(raw_line)

                offset_no_newline = offset_line.rstrip(b"\r\n")
                try:
                    embedded_raw, stored_offset_bytes = offset_no_newline.rsplit(b"\t", 1)
                except ValueError as exc:
                    raise RuntimeError(f"offset row {ordinal} has no appended offset column") from exc
                if raw_line.rstrip(b"\r\n") != embedded_raw:
                    raise RuntimeError(f"raw/offset lockstep identity mismatch at row {ordinal}")
                fields = embedded_raw.decode("utf-8", errors="strict").split("\t")
                if len(fields) != len(RAW_COLUMNS):
                    raise RuntimeError(f"row {ordinal} has {len(fields)} columns, expected 21")
                row = dict(zip(RAW_COLUMNS, fields))
                stored_offset_text = stored_offset_bytes.decode("utf-8", errors="replace").strip()

                pdb = row["pdb_id"].strip().lower()
                chain = row["receptor_chain"].strip()
                query_sequence = row["receptor_sequence"].strip().upper()

                if pdb != previous_pdb:
                    counters["pdb_runs"] += 1
                    if pdb in pdbs_seen:
                        counters["noncontiguous_pdb_reappearance"] += 1
                    pdbs_seen.add(pdb)
                    previous_pdb = pdb
                    current_alignment_cache = {}
                    current_canonical_alignment_cache = {}
                    fasta_path = FASTA_DIR / f"{pdb}.faa"
                    if fasta_path.is_file():
                        try:
                            current_fasta = load_fasta(fasta_path)
                            fasta_status = "loaded"
                            fasta_hash = sha256(fasta_path)
                            stat = fasta_path.stat()
                            fasta_writer.write(
                                {
                                    "pdb_id": pdb,
                                    "path": str(fasta_path),
                                    "status": fasta_status,
                                    "bytes": stat.st_size,
                                    "mtime_epoch_seconds": int(stat.st_mtime),
                                    "sha256": fasta_hash,
                                    "chains": ";".join(sorted(current_fasta)),
                                }
                            )
                            counters["fasta_pdb_loaded"] += 1
                        except Exception as exc:
                            current_fasta = {}
                            fasta_writer.write(
                                {
                                    "pdb_id": pdb,
                                    "path": str(fasta_path),
                                    "status": f"parse_failed:{type(exc).__name__}",
                                    "bytes": fasta_path.stat().st_size,
                                    "mtime_epoch_seconds": int(fasta_path.stat().st_mtime),
                                    "sha256": "",
                                    "chains": "",
                                }
                            )
                            counters["fasta_pdb_parse_failed"] += 1
                    else:
                        current_fasta = {}
                        fasta_writer.write(
                            {
                                "pdb_id": pdb,
                                "path": str(fasta_path),
                                "status": "file_missing",
                                "bytes": "",
                                "mtime_epoch_seconds": "",
                                "sha256": "",
                                "chains": "",
                            }
                        )
                        counters["fasta_pdb_missing"] += 1

                sequence_sha = hashlib.sha256(query_sequence.encode("utf-8")).hexdigest()
                sequence_id = stable_id("BLSEQ_", f"{pdb}|{chain}|{sequence_sha}")
                if sequence_id not in sequence_ids_seen:
                    sequence_ids_seen.add(sequence_id)
                    sequence_writer.write(
                        {
                            "receptor_sequence_id": sequence_id,
                            "pdb_id": pdb,
                            "receptor_chain": chain,
                            "sequence_sha256": sequence_sha,
                            "sequence_length": len(query_sequence),
                            "receptor_sequence": query_sequence,
                        }
                    )

                natural_key = "|".join(
                    [
                        pdb,
                        chain,
                        row["binding_site_code"].strip(),
                        row["ligand_ccd"].strip().upper(),
                        row["ligand_chain"].strip(),
                        row["ligand_serial"].strip(),
                        row["ligand_auth_seq_id"].strip(),
                    ]
                )
                natural_key_counts[natural_key] += 1
                observation_id = stable_id("BLOBS_", f"{ordinal}|{natural_key}")

                _, binding_letters, binding_positions, invalid_tokens = parse_residue_tokens(
                    row["binding_residues_seq_raw"]
                )
                if invalid_tokens:
                    counters["rows_with_invalid_binding_seq_tokens"] += 1
                    counters["invalid_binding_seq_tokens"] += len(invalid_tokens)
                    if len(invalid_binding_token_examples) < 25:
                        invalid_binding_token_examples.append(
                            {"source_ordinal": ordinal, "pdb_id": pdb, "tokens": invalid_tokens}
                        )

                target_sequence = current_fasta.get(chain, "")
                if not current_fasta:
                    cif_fasta_status = "pdb_fasta_unavailable"
                elif chain not in current_fasta:
                    cif_fasta_status = "receptor_chain_unavailable"
                else:
                    cif_fasta_status = "available"

                alignment_key = (chain, sequence_sha)
                if alignment_key not in current_alignment_cache:
                    current_alignment_cache[alignment_key] = build_alignment_map(
                        target_sequence, query_sequence, purpose="cif"
                    )
                cif_alignment = current_alignment_cache[alignment_key]
                alignment_modes[cif_alignment.mode] += 1
                label_positions = (
                    [cif_alignment.position_map.get(position) for position in binding_positions]
                    if cif_alignment.trusted_cif_mapping
                    else [None] * len(binding_positions)
                )

                stored_offset = parse_stored_offset(stored_offset_text)
                if stored_offset is None:
                    legacy_offset_agreement = "legacy_offset_unavailable"
                elif cif_alignment.offset is None:
                    legacy_offset_agreement = "recomputed_offset_unavailable"
                elif stored_offset == cif_alignment.offset:
                    legacy_offset_agreement = "agree"
                else:
                    legacy_offset_agreement = "disagree"
                counters[f"legacy_offset_{legacy_offset_agreement}"] += 1

                raw_accessions = parse_accessions(row["uniprot_raw"])
                segments = sifts.get((pdb, chain), [])
                sifts_result = map_label_positions(label_positions, segments, raw_accessions)
                sifts_status = str(sifts_result["status"])
                sifts_statuses[sifts_status] += 1
                sifts_accessions = list(sifts_result["candidate_accessions"])
                sifts_resolved = str(sifts_result["resolved_accession"])
                sifts_positions = list(sifts_result["mapped_positions"])

                canonical_candidates: List[str] = []
                seen_candidates: Set[str] = set()
                for accession in ([sifts_resolved] if sifts_resolved else []) + raw_accessions + sifts_accessions:
                    if accession and accession not in seen_candidates:
                        seen_candidates.add(accession)
                        canonical_candidates.append(accession)
                canonical_result = map_to_canonical(
                    query_sequence,
                    binding_positions,
                    binding_letters,
                    canonical_candidates,
                    sifts_resolved,
                    uniprot_cache,
                    current_canonical_alignment_cache,
                )
                canonical_accession = str(canonical_result["accession"])
                canonical_positions = list(canonical_result["mapped_positions"])
                canonical_alignment = canonical_result["alignment"]
                assert isinstance(canonical_alignment, AlignmentResult)
                canonical_status = str(canonical_result["status"])

                n_binding = len(binding_positions)
                sifts_complete = n_binding > 0 and all(value is not None for value in sifts_positions)
                canonical_complete = n_binding > 0 and all(value is not None for value in canonical_positions)
                canonical_trusted_complete = (
                    canonical_complete
                    and canonical_alignment.trusted_cif_mapping
                    and int(canonical_result["letter_mismatches"]) == 0
                )
                same_accession = (
                    bool(sifts_resolved)
                    and bool(canonical_accession)
                    and accession_base(sifts_resolved) == accession_base(canonical_accession)
                )
                same_positions = sifts_positions == canonical_positions

                if sifts_complete and canonical_trusted_complete and same_accession and same_positions:
                    final_accession = sifts_resolved
                    final_positions = sifts_positions
                    site_status = "mapped_consensus_complete"
                    agreement_status = "complete_exact_agreement"
                    usable = True
                elif sifts_complete and not canonical_trusted_complete:
                    final_accession = sifts_resolved
                    final_positions = sifts_positions
                    site_status = "mapped_sifts_complete_canonical_unavailable_or_partial"
                    agreement_status = "sifts_only_complete"
                    usable = True
                elif sifts_complete and canonical_trusted_complete:
                    final_accession = sifts_resolved
                    final_positions = sifts_positions
                    site_status = "mapped_complete_methods_conflict"
                    agreement_status = "complete_conflict"
                    usable = False
                elif canonical_trusted_complete:
                    final_accession = canonical_accession
                    final_positions = canonical_positions
                    site_status = "mapped_canonical_complete_sifts_unavailable_or_partial"
                    agreement_status = "canonical_only_complete"
                    usable = True
                else:
                    # Preserve the most informative partial mapping, but do not
                    # declare it site-clustering eligible.
                    if sum(value is not None for value in sifts_positions) >= sum(
                        value is not None for value in canonical_positions
                    ):
                        final_accession = sifts_resolved
                        final_positions = sifts_positions
                    else:
                        final_accession = canonical_accession
                        final_positions = canonical_positions
                    mapped_count = sum(value is not None for value in final_positions)
                    site_status = "mapped_partial" if mapped_count else "unmapped"
                    agreement_status = "no_complete_mapping"
                    usable = False

                if not raw_accessions:
                    if final_accession:
                        uniprot_status = "inferred_from_sifts_or_sequence"
                    else:
                        uniprot_status = "raw_missing_unresolved"
                elif final_accession and accession_base(final_accession) in {
                    accession_base(value) for value in raw_accessions
                }:
                    uniprot_status = "resolved_agrees_with_raw"
                elif final_accession:
                    uniprot_status = "resolved_conflicts_with_raw"
                else:
                    uniprot_status = "raw_present_mapping_unresolved"

                full_inchikey, connectivity, smiles, chemical_status = compound_fields(
                    compounds, row["ligand_ccd"]
                )
                pair_key = f"{accession_base(final_accession)}|{full_inchikey}" if final_accession and full_inchikey else ""

                mapped_count = sum(value is not None for value in final_positions)
                mapping_fraction = mapped_count / n_binding if n_binding else 0.0

                output = {
                    "source_ordinal": ordinal,
                    "observation_id": observation_id,
                    "natural_observation_key": natural_key,
                    "pdb_id": pdb,
                    "receptor_chain": chain,
                    "resolution": row["resolution"].strip(),
                    "binding_site_code": row["binding_site_code"].strip(),
                    "ligand_ccd": row["ligand_ccd"].strip().upper(),
                    "ligand_chain": row["ligand_chain"].strip(),
                    "ligand_serial": row["ligand_serial"].strip(),
                    "ligand_auth_seq_id": row["ligand_auth_seq_id"].strip(),
                    "uniprot_raw": row["uniprot_raw"].strip(),
                    "uniprot_raw_candidates": ";".join(raw_accessions),
                    "uniprot_resolved": final_accession,
                    "uniprot_resolution_status": uniprot_status,
                    "pubmed_id": row["pubmed_id"].strip(),
                    "ec_number": row["ec_number"].strip(),
                    "go_terms": row["go_terms"].strip(),
                    "affinity_manual": row["affinity_manual"].strip(),
                    "affinity_moad": row["affinity_moad"].strip(),
                    "affinity_pdbbind_cn": row["affinity_pdbbind_cn"].strip(),
                    "affinity_bindingdb": row["affinity_bindingdb"].strip(),
                    "full_inchikey": full_inchikey,
                    "connectivity_key": connectivity,
                    "canonical_smiles": smiles,
                    "chemical_mapping_status": chemical_status,
                    "active_orthosteric_pair_reference": bool_text(bool(pair_key and pair_key in active_keys)),
                    "quarantined_orthosteric_pair_reference": bool_text(bool(pair_key and pair_key in quarantine_keys)),
                    "binding_residues_auth_raw": row["binding_residues_auth_raw"].strip(),
                    "binding_residues_seq_raw": row["binding_residues_seq_raw"].strip(),
                    "catalytic_residues_auth_raw": row["catalytic_residues_auth_raw"].strip(),
                    "catalytic_residues_seq_raw": row["catalytic_residues_seq_raw"].strip(),
                    "binding_residue_count": n_binding,
                    "binding_seq_positions": ";".join(str(value) for value in binding_positions),
                    "binding_label_seq_positions": join_positions(label_positions),
                    "binding_uniprot_positions": join_positions(final_positions),
                    "binding_residue_mapping_fraction": f"{mapping_fraction:.6f}",
                    "site_mapping_status": site_status,
                    "mapping_usable_for_site_clustering": bool_text(usable),
                    "mapping_agreement_status": agreement_status,
                    "cif_fasta_status": cif_fasta_status,
                    "biolip_to_cif_alignment_mode": cif_alignment.mode,
                    "biolip_to_cif_match_fraction": f"{cif_alignment.match_fraction:.6f}",
                    "biolip_to_cif_query_coverage": f"{cif_alignment.query_coverage:.6f}",
                    "biolip_to_cif_mapping_trusted": bool_text(cif_alignment.trusted_cif_mapping),
                    "stored_legacy_offset": stored_offset_text,
                    "recomputed_alignment_offset": "" if cif_alignment.offset is None else cif_alignment.offset,
                    "legacy_offset_agreement": legacy_offset_agreement,
                    "sifts_segment_status": sifts_status,
                    "sifts_candidate_accessions": ";".join(sifts_accessions),
                    "canonical_alignment_status": canonical_status,
                    "canonical_alignment_match_fraction": f"{canonical_alignment.match_fraction:.6f}",
                    "canonical_alignment_query_coverage": f"{canonical_alignment.query_coverage:.6f}",
                    "canonical_alignment_trusted": bool_text(canonical_alignment.trusted_cif_mapping),
                    "uniprot_residue_letter_matches": int(canonical_result["letter_matches"]),
                    "uniprot_residue_letter_mismatches": int(canonical_result["letter_mismatches"]),
                    "receptor_sequence_id": sequence_id,
                    "receptor_sequence_length": len(query_sequence),
                }
                master_writer.write(output)

                counters["rows"] += 1
                counters["binding_residues"] += n_binding
                counters["mapped_binding_residues"] += mapped_count
                counters["site_clustering_usable_rows"] += int(usable)
                counters["active_orthosteric_reference_rows"] += int(bool(pair_key and pair_key in active_keys))
                counters["quarantined_orthosteric_reference_rows"] += int(bool(pair_key and pair_key in quarantine_keys))
                counters["rows_with_raw_uniprot"] += int(bool(raw_accessions))
                counters["rows_with_full_inchikey"] += int(bool(full_inchikey))
                site_statuses[site_status] += 1
                uniprot_statuses[uniprot_status] += 1
                agreement_statuses[agreement_status] += 1
                chemical_statuses[chemical_status] += 1

                if args.progress_every and (ordinal + 1) % args.progress_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"rows={ordinal+1:,} pdbs={len(pdbs_seen):,} "
                        f"usable={counters['site_clustering_usable_rows']:,} elapsed={elapsed/60:.1f}m",
                        flush=True,
                    )

            trailing_raw = raw_handle.readline()
            if trailing_raw:
                raise RuntimeError("raw BioLiP contains rows after the offset derivative ended")

        if counters["rows"] != EXPECTED_SOURCE_ROWS:
            raise RuntimeError(f"expected {EXPECTED_SOURCE_ROWS} rows, observed {counters['rows']}")
        if counters["noncontiguous_pdb_reappearance"]:
            raise RuntimeError("BioLiP PDB blocks are not contiguous; bounded FASTA cache assumption failed")
        completed = True
    finally:
        master_writer.close(commit=completed)
        sequence_writer.close(commit=completed)
        fasta_writer.close(commit=completed)

    duplicate_rows = [
        {"natural_observation_key": key, "source_row_count": count, "excess_rows": count - 1}
        for key, count in natural_key_counts.items()
        if count > 1
    ]
    duplicate_frame = pd.DataFrame(
        duplicate_rows,
        columns=["natural_observation_key", "source_row_count", "excess_rows"],
    ).sort_values(["source_row_count", "natural_observation_key"], ascending=[False, True])
    write_tsv(duplicate_frame, DUPLICATE_OUT, compressed=True)

    def counter_frame(counter: Counter, key_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            [{key_name: key, "rows": value} for key, value in counter.most_common()],
            columns=[key_name, "rows"],
        )

    status_frame = counter_frame(site_statuses, "site_mapping_status")
    uniprot_frame = counter_frame(uniprot_statuses, "uniprot_resolution_status")
    alignment_rows: List[Dict[str, object]] = []
    for category, counter in [
        ("biolip_to_cif_alignment_mode", alignment_modes),
        ("mapping_agreement_status", agreement_statuses),
        ("sifts_segment_status", sifts_statuses),
        ("chemical_mapping_status", chemical_statuses),
    ]:
        for value, count in counter.most_common():
            alignment_rows.append({"category": category, "value": value, "rows": count})
    alignment_frame = pd.DataFrame(alignment_rows)
    write_tsv(status_frame, STATUS_OUT)
    write_tsv(uniprot_frame, UNIPROT_OUT)
    write_tsv(alignment_frame, ALIGNMENT_OUT)

    input_rows = [
        {"asset_id": "checkpoint3_validation", **file_record(CHECKPOINT3_VALIDATION)},
        {"asset_id": "raw_biolip", **file_record(RAW_BIOLIP, raw_digest.hexdigest())},
        {"asset_id": "offset_biolip", **file_record(OFFSET_BIOLIP, offset_digest.hexdigest())},
        {"asset_id": "bulk_sifts", **file_record(BULK_SIFTS)},
        {"asset_id": "uniprot_cache", **file_record(UNIPROT_CACHE)},
        {"asset_id": "wwpdb_compounds", **file_record(WWPDB_COMPOUNDS)},
        {"asset_id": "active_orthosteric_pairs", **file_record(ACTIVE_ORTHO)},
        {"asset_id": "quarantine_orthosteric_pairs", **file_record(QUARANTINE_ORTHO)},
    ]
    write_tsv(pd.DataFrame(input_rows), INPUT_HASHES_OUT)

    summary = {
        "stage": "checkpoint4a_exact_observation_master",
        "status": "validated",
        "source_rows": counters["rows"],
        "unique_natural_observations": len(natural_key_counts),
        "duplicate_natural_keys": len(duplicate_frame),
        "duplicate_excess_rows": int(duplicate_frame["excess_rows"].sum()) if len(duplicate_frame) else 0,
        "unique_pdbs": len(pdbs_seen),
        "unique_receptor_sequence_records": len(sequence_ids_seen),
        "binding_residues": counters["binding_residues"],
        "mapped_binding_residues": counters["mapped_binding_residues"],
        "site_clustering_usable_rows": counters["site_clustering_usable_rows"],
        "rows_with_raw_uniprot": counters["rows_with_raw_uniprot"],
        "rows_with_full_inchikey": counters["rows_with_full_inchikey"],
        "active_orthosteric_reference_rows": counters["active_orthosteric_reference_rows"],
        "quarantined_orthosteric_reference_rows": counters["quarantined_orthosteric_reference_rows"],
        "fasta_pdb_loaded": counters["fasta_pdb_loaded"],
        "fasta_pdb_missing": counters["fasta_pdb_missing"],
        "fasta_pdb_parse_failed": counters["fasta_pdb_parse_failed"],
        "legacy_offset_agreement_counts": {
            key.replace("legacy_offset_", ""): value
            for key, value in counters.items()
            if key.startswith("legacy_offset_")
        },
        "bulk_sifts": sifts_stats,
        "site_mapping_status_counts": dict(site_statuses),
        "uniprot_resolution_status_counts": dict(uniprot_statuses),
        "mapping_agreement_counts": dict(agreement_statuses),
        "alignment_mode_counts": dict(alignment_modes),
        "chemical_mapping_status_counts": dict(chemical_statuses),
        "invalid_binding_token_examples": invalid_binding_token_examples,
        "raw_offset_lockstep_identity_verified": True,
        "large_structure_directory_recursively_scanned": False,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "outputs": {},
    }
    for name, path in {
        "observation_master": MASTER_OUT,
        "sequence_manifest": SEQUENCE_OUT,
        "fasta_input_manifest": FASTA_INPUT_OUT,
        "duplicate_natural_observations": DUPLICATE_OUT,
        "mapping_status_counts": STATUS_OUT,
        "uniprot_resolution_counts": UNIPROT_OUT,
        "alignment_audit": ALIGNMENT_OUT,
        "input_hashes": INPUT_HASHES_OUT,
    }.items():
        summary["outputs"][name] = file_record(path)
    atomic_json(SUMMARY_OUT, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
