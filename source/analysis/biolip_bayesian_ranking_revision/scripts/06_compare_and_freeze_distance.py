#!/usr/bin/env python3
"""Checkpoint 6: compare exact-site distance definitions before choosing one.

The computation is restricted to the checkpoint-5 exact reference observations.
It resolves only explicitly named PDB coordinate paths and never enumerates the
large coordinate repositories. Distance definitions are compared descriptively,
and no definition is selected until the user reviews the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
REPORTS = PACKAGE / "reports"
VALIDATION = PACKAGE / "validation"
MANIFESTS = PACKAGE / "manifests"

REFERENCE = DATA / "BIOLIP_EXACT_SITE_REFERENCE_OBSERVATIONS.tsv.gz"
REGISTRY = DATA / "CHECKPOINT5_ALLOBENCH_SOURCE_REGISTRY.tsv.gz"
CHECKPOINT5_VALIDATION = VALIDATION / "CHECKPOINT5_VALIDATION.json"
UNIPROT_CACHE = Path("/shared_data/11.HS_allostery/Data/uniprot_cache.pickle")

GEOMETRY = DATA / "CHECKPOINT6_EXACT_REFERENCE_GEOMETRY.tsv.gz"
COORDINATES = DATA / "CHECKPOINT6_COORDINATE_MANIFEST.tsv.gz"
COMPARISON = DATA / "CHECKPOINT6_DISTANCE_DEFINITION_COMPARISON.tsv"
SUMMARIES = DATA / "CHECKPOINT6_DISTANCE_DISTRIBUTIONS.tsv"
CORRELATIONS = DATA / "CHECKPOINT6_DISTANCE_CORRELATIONS.tsv"
STATUS_COUNTS = DATA / "CHECKPOINT6_GEOMETRY_STATUS_COUNTS.tsv"
FETCH_LOG = DATA / "CHECKPOINT6_COORDINATE_FETCH_LOG.tsv"
DISTINCT_OVERLAP_AUDIT = DATA / "CHECKPOINT6_REPORTED_DISTINCT_RESIDUE_OVERLAP_AUDIT.tsv.gz"
SPEC = MANIFESTS / "CHECKPOINT6_DISTANCE_SPEC.json"
INPUT_HASHES = MANIFESTS / "CHECKPOINT6_INPUT_HASHES.tsv"
REPORT = REPORTS / "CHECKPOINT6_DISTANCE_DEFINITION.md"
BUILD = VALIDATION / "CHECKPOINT6_BUILD_SUMMARY.json"

COORDINATE_ROOTS = (
    ("checkpoint6_fetched", PACKAGE / "assets/checkpoint6_cif"),
    ("candidate_rebuild", ROOT / "analysis/structure_known_candidate_rebuild/assets/cif"),
    ("shared_legacy", Path("/shared_data/11.HS_allostery/8.Distances_BioLiP/cif_files")),
    ("base_structures", ROOT / "data/structures"),
    ("biolip_xai", ROOT / "analysis/structure_known_biolip_xai/assets/structures"),
)
SIFTS_ROOT = ROOT / "analysis/structure_known_candidate_rebuild/assets/sifts"

REQUIRED_GEOMETRY_METRIC = "ligand_to_orthosteric_site_min_heavy_A"
DISTANCE_METRICS = (
    "ligand_to_orthosteric_site_min_heavy_A",
    "ligand_to_orthosteric_site_min_CA_A",
    "candidate_site_to_orthosteric_site_min_heavy_A",
    "candidate_site_to_orthosteric_site_min_CA_A",
    "ligand_centroid_to_orthosteric_site_heavy_centroid_A",
    "ligand_centroid_to_orthosteric_site_CA_centroid_A",
    "candidate_site_CA_centroid_to_orthosteric_site_CA_centroid_A",
)

METRIC_DEFINITIONS = {
    "ligand_to_orthosteric_site_min_heavy_A": (
        "minimum Euclidean distance between candidate-ligand heavy atoms and "
        "orthosteric-site protein heavy atoms"
    ),
    "ligand_to_orthosteric_site_min_CA_A": (
        "minimum Euclidean distance between candidate-ligand heavy atoms and "
        "orthosteric-site residue C-alpha atoms"
    ),
    "candidate_site_to_orthosteric_site_min_heavy_A": (
        "minimum Euclidean distance between candidate binding-residue heavy atoms "
        "and orthosteric-site residue heavy atoms"
    ),
    "candidate_site_to_orthosteric_site_min_CA_A": (
        "minimum C-alpha distance between candidate binding residues and "
        "orthosteric-site residues"
    ),
    "ligand_centroid_to_orthosteric_site_heavy_centroid_A": (
        "distance between the candidate-ligand heavy-atom centroid and the "
        "orthosteric-site heavy-atom centroid"
    ),
    "ligand_centroid_to_orthosteric_site_CA_centroid_A": (
        "distance between the candidate-ligand heavy-atom centroid and the "
        "orthosteric-site residue C-alpha centroid"
    ),
    "candidate_site_CA_centroid_to_orthosteric_site_CA_centroid_A": (
        "distance between candidate-site and orthosteric-site C-alpha centroids"
    ),
}

RESIDUE_TOKEN = re.compile(r"^[A-Za-z](-?\d+)([A-Za-z]?)$")
_WORKER_SEQUENCES: Dict[str, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CHECKPOINT6_WORKERS", "16")))
    parser.add_argument("--max-pdb", type=int, default=0, help="bounded implementation pilot; zero means all")
    parser.add_argument("--pilot-output", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_tsv(frame: pd.DataFrame, path: Path, compression: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def parse_positions(value: object) -> List[int]:
    positions = []
    for token in str(value or "").split(";"):
        token = token.strip()
        if token and token.lstrip("-").isdigit():
            positions.append(int(token))
    return sorted(set(positions))


def parse_author_residues(value: object) -> List[Tuple[int, str]]:
    residues = []
    for token in str(value or "").split():
        match = RESIDUE_TOKEN.match(token.strip())
        if match:
            residues.append((int(match.group(1)), match.group(2) or ""))
    return sorted(set(residues))


def three_to_one(name: str) -> str:
    info = gemmi.find_tabulated_residue(name)
    if info is None or not info.is_amino_acid():
        return ""
    code = (info.one_letter_code or "").upper()
    return code if len(code) == 1 else "X"


def best_heavy_atoms(residue: gemmi.Residue) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    best = {}
    for atom in residue:
        if atom.element.is_hydrogen:
            continue
        name = str(atom.name)
        xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
        if name not in best or float(atom.occ) > best[name][0]:
            best[name] = (float(atom.occ), xyz)
    heavy = np.asarray([item[1] for item in best.values()], dtype=float)
    ca = best.get("CA", (None, None))[1]
    if heavy.size == 0:
        heavy = np.empty((0, 3), dtype=float)
    return heavy, ca


def chain_index(model: gemmi.Model, chain_name: str):
    index = {}
    sequence = []
    by_number = defaultdict(list)
    for chain in model:
        if chain.name != chain_name:
            continue
        for residue in chain:
            if residue.entity_type != gemmi.EntityType.Polymer:
                continue
            amino_acid = three_to_one(residue.name)
            if not amino_acid:
                continue
            key = (int(residue.seqid.num), str(residue.seqid.icode).strip())
            heavy, ca = best_heavy_atoms(residue)
            index[key] = {"heavy": heavy, "ca": ca, "aa": amino_acid}
            sequence.append((amino_acid, key))
            by_number[key[0]].append(key)
    return index, sequence, by_number


def resolve_key(index, by_number, key: Tuple[int, str]):
    if key in index:
        return key
    no_insertion = (key[0], "")
    if no_insertion in index:
        return no_insertion
    candidates = by_number.get(key[0], [])
    return candidates[0] if len(candidates) == 1 else None


def new_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    return aligner


def alignment_maps(reference_sequence: str, chain_sequence, aligner: PairwiseAligner):
    observed = "".join(item[0] for item in chain_sequence)
    if not reference_sequence or not observed:
        return {}, {}, np.nan, np.nan
    alignment = aligner.align(reference_sequence, observed)[0]
    uni_to_key, key_to_uni = {}, {}
    matches = mapped = 0
    for (u0, u1), (c0, c1) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(min(u1 - u0, c1 - c0)):
            chain_index_position = c0 + offset
            key = chain_sequence[chain_index_position][1]
            uniprot_position = u0 + offset + 1
            uni_to_key[uniprot_position] = key
            key_to_uni[key] = uniprot_position
            mapped += 1
            if reference_sequence[u0 + offset] == chain_sequence[chain_index_position][0]:
                matches += 1
    identity = matches / mapped if mapped else np.nan
    chain_coverage = mapped / len(observed) if observed else np.nan
    return uni_to_key, key_to_uni, identity, chain_coverage


def sifts_maps(pdb_id: str, chain_name: str, uniprot: str):
    path = SIFTS_ROOT / f"{pdb_id.lower()}.json"
    if not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text())
    except Exception:
        return {}, "sifts_read_error"
    output = {}
    base = uniprot.upper().split("-", 1)[0]
    for segment in value.get(chain_name, []):
        if len(segment) < 7:
            continue
        u0, u1, author_start, insertion, _author_end, _end_insertion, accession = segment[:7]
        if str(accession).upper().split("-", 1)[0] != base or None in (u0, u1, author_start):
            continue
        for position in range(int(u0), int(u1) + 1):
            output[position] = (int(author_start) + position - int(u0), str(insertion or "").strip())
    return output, "sifts" if output else "sifts_no_matching_segment"


def resolve_coordinate(pdb_id: str):
    for source, root in COORDINATE_ROOTS:
        for extension in (".cif.gz", ".cif"):
            path = root / f"{pdb_id.lower()}{extension}"
            if path.is_file():
                stat = path.stat()
                return path, source, int(stat.st_size), int(stat.st_mtime)
    return None, "missing", 0, 0


def find_ligand(model: gemmi.Model, ccd: str, chain_name: str, residue_number: str):
    try:
        number = int(str(residue_number))
    except ValueError:
        return None, 0, 0
    nonpolymer_hits, polymer_name_collisions = [], 0
    for chain in model:
        if chain.name != chain_name:
            continue
        for residue in chain:
            if residue.name.upper() != ccd.upper() or int(residue.seqid.num) != number:
                continue
            if residue.entity_type == gemmi.EntityType.Polymer:
                polymer_name_collisions += 1
                continue
            if residue.entity_type == gemmi.EntityType.Water:
                continue
            heavy, _ca = best_heavy_atoms(residue)
            if len(heavy):
                nonpolymer_hits.append(heavy)
    return (nonpolymer_hits[0] if len(nonpolymer_hits) == 1 else None,
            len(nonpolymer_hits), polymer_name_collisions)


def stack_heavy(keys, index):
    arrays = [index[key]["heavy"] for key in keys if key in index and len(index[key]["heavy"])]
    return np.vstack(arrays) if arrays else None


def stack_ca(keys, index):
    arrays = [index[key]["ca"] for key in keys if key in index and index[key]["ca"] is not None]
    return np.vstack(arrays) if arrays else None


def min_distance(first, second):
    if first is None or second is None or not len(first) or not len(second):
        return np.nan
    return float(np.min(np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)))


def centroid_distance(first, second):
    if first is None or second is None or not len(first) or not len(second):
        return np.nan
    return float(np.linalg.norm(np.mean(first, axis=0) - np.mean(second, axis=0)))


def set_jaccard(first: set, second: set) -> float:
    union = first | second
    return len(first & second) / len(union) if union else np.nan


def initialize_worker(sequences: Dict[str, str]) -> None:
    global _WORKER_SEQUENCES
    _WORKER_SEQUENCES = sequences


def failure_rows(pdb_id: str, rows: Sequence[dict], status: str, source: str = "missing"):
    return [{
        **row, "coordinate_source": source, "coordinate_size_bytes": 0,
        "coordinate_mtime_epoch": 0, "geometry_status": status,
    } for row in rows]


def process_pdb(task):
    pdb_id, rows = task
    coordinate, source, size_bytes, mtime_epoch = resolve_coordinate(pdb_id)
    coordinate_record = {
        "pdb_id": pdb_id, "coordinate_source": source,
        "coordinate_path": str(coordinate or ""), "size_bytes": size_bytes,
        "mtime_epoch": mtime_epoch, "reference_observations": len(rows),
    }
    if coordinate is None:
        return failure_rows(pdb_id, rows, "coordinate_missing", source), coordinate_record
    try:
        structure = gemmi.read_structure(str(coordinate))
        model = structure[0]
    except Exception as error:
        return failure_rows(pdb_id, rows, f"coordinate_read_error:{type(error).__name__}", source), coordinate_record

    aligner = new_aligner()
    chain_cache = {}
    mapping_cache = {}
    output = []
    for task_row in rows:
        row = dict(task_row)
        row.update({
            "coordinate_source": source, "coordinate_size_bytes": size_bytes,
            "coordinate_mtime_epoch": mtime_epoch,
        })
        chain_name = row["receptor_chain"]
        if chain_name not in chain_cache:
            chain_cache[chain_name] = chain_index(model, chain_name)
        index, chain_sequence, by_number = chain_cache[chain_name]
        if not index:
            row["geometry_status"] = "receptor_chain_absent"
            output.append(row)
            continue

        candidate_requested = parse_author_residues(row["binding_residues_auth_raw"])
        candidate_keys = {
            resolved for requested in candidate_requested
            for resolved in [resolve_key(index, by_number, requested)] if resolved is not None
        }
        candidate_heavy = stack_heavy(candidate_keys, index)
        candidate_ca = stack_ca(candidate_keys, index)
        row["candidate_residues_requested"] = len(candidate_requested)
        row["candidate_residues_recovered"] = len(candidate_keys)
        row["candidate_residue_recovery_fraction"] = (
            len(candidate_keys) / len(candidate_requested) if candidate_requested else np.nan
        )

        ligand, ligand_matches, polymer_collisions = find_ligand(
            model, row["ligand_ccd"], row["ligand_chain"], row["ligand_auth_seq_id"]
        )
        row["exact_nonpolymer_ligand_matches"] = ligand_matches
        row["polymer_name_collisions_excluded"] = polymer_collisions
        if ligand_matches == 0:
            row["geometry_status"] = "exact_nonpolymer_ligand_absent"
            output.append(row)
            continue
        if ligand_matches > 1:
            row["geometry_status"] = "exact_nonpolymer_ligand_ambiguous"
            output.append(row)
            continue
        if not candidate_keys:
            row["geometry_status"] = "candidate_binding_residues_absent"
            output.append(row)
            continue

        if row["reference_label"] == "orthosteric":
            orthosteric_site_keys = set(candidate_keys)
            orthosteric_site_requested_count = len(candidate_requested)
            mapping_method = "exact_observed_orthosteric_binding_residues"
            identity = chain_coverage = np.nan
            frozen_mapping_jaccard = 1.0
        else:
            orthosteric_site_positions = [
                int(value)
                for value in row["orthosteric_site_uniprot_positions"].split(";")
                if value
            ]
            orthosteric_site_requested_count = len(orthosteric_site_positions)
            mapping_key = (chain_name, row["uniprot"])
            if mapping_key not in mapping_cache:
                sifts, sifts_status = sifts_maps(pdb_id, chain_name, row["uniprot"])
                if sifts:
                    uni_to_key = {
                        position: resolve_key(index, by_number, author_key)
                        for position, author_key in sifts.items()
                    }
                    uni_to_key = {position: key for position, key in uni_to_key.items() if key is not None}
                    key_to_uni = {key: position for position, key in uni_to_key.items()}
                    mapping_cache[mapping_key] = (uni_to_key, key_to_uni, "sifts", 1.0, np.nan)
                else:
                    sequence = _WORKER_SEQUENCES.get(row["uniprot"], "")
                    uni_to_key, key_to_uni, identity, chain_coverage = alignment_maps(
                        sequence, chain_sequence, aligner
                    )
                    method = "canonical_global_alignment" if uni_to_key else (sifts_status or "mapping_unavailable")
                    mapping_cache[mapping_key] = (
                        uni_to_key, key_to_uni, method, identity, chain_coverage
                    )
            uni_to_key, key_to_uni, mapping_method, identity, chain_coverage = mapping_cache[mapping_key]
            orthosteric_site_keys = {
                uni_to_key[position]
                for position in orthosteric_site_positions
                if position in uni_to_key
            }
            frozen_binding_positions = set(parse_positions(row["binding_uniprot_positions"]))
            independently_mapped_binding = {key_to_uni[key] for key in candidate_keys if key in key_to_uni}
            frozen_mapping_jaccard = set_jaccard(frozen_binding_positions, independently_mapped_binding)

        row["orthosteric_site_mapping_method"] = mapping_method
        row["orthosteric_site_mapping_identity"] = identity
        row["orthosteric_site_mapping_chain_coverage"] = chain_coverage
        row["orthosteric_site_residues_requested"] = orthosteric_site_requested_count
        row["orthosteric_site_residues_recovered"] = len(orthosteric_site_keys)
        row["orthosteric_site_residue_recovery_fraction"] = (
            len(orthosteric_site_keys) / orthosteric_site_requested_count
            if orthosteric_site_requested_count else np.nan
        )
        row["checkpoint4_binding_map_jaccard"] = frozen_mapping_jaccard
        if not orthosteric_site_keys:
            row["geometry_status"] = "orthosteric_site_unmapped"
            output.append(row)
            continue
        if row["reference_label"] == "allosteric" and (
            not np.isfinite(frozen_mapping_jaccard) or frozen_mapping_jaccard < 0.80
        ):
            row["geometry_status"] = "active_site_mapping_failed_checkpoint4_crosscheck"
            output.append(row)
            continue

        orthosteric_site_heavy = stack_heavy(orthosteric_site_keys, index)
        orthosteric_site_ca = stack_ca(orthosteric_site_keys, index)
        overlap = candidate_keys & orthosteric_site_keys
        row["candidate_orthosteric_site_residue_overlap_count"] = len(overlap)
        row["candidate_orthosteric_site_residue_jaccard"] = set_jaccard(
            candidate_keys, orthosteric_site_keys
        )
        row["ligand_heavy_atom_count"] = len(ligand)
        row["candidate_site_heavy_atom_count"] = len(candidate_heavy) if candidate_heavy is not None else 0
        row["orthosteric_site_heavy_atom_count"] = (
            len(orthosteric_site_heavy) if orthosteric_site_heavy is not None else 0
        )
        row["ligand_to_orthosteric_site_min_heavy_A"] = min_distance(
            ligand, orthosteric_site_heavy
        )
        row["ligand_to_orthosteric_site_min_CA_A"] = min_distance(
            ligand, orthosteric_site_ca
        )
        row["candidate_site_to_orthosteric_site_min_heavy_A"] = min_distance(
            candidate_heavy, orthosteric_site_heavy
        )
        row["candidate_site_to_orthosteric_site_min_CA_A"] = min_distance(
            candidate_ca, orthosteric_site_ca
        )
        row["ligand_centroid_to_orthosteric_site_heavy_centroid_A"] = centroid_distance(
            ligand, orthosteric_site_heavy
        )
        row["ligand_centroid_to_orthosteric_site_CA_centroid_A"] = centroid_distance(
            ligand, orthosteric_site_ca
        )
        row["candidate_site_CA_centroid_to_orthosteric_site_CA_centroid_A"] = centroid_distance(
            candidate_ca, orthosteric_site_ca
        )
        row["geometry_status"] = (
            "ok"
            if np.isfinite(row[REQUIRED_GEOMETRY_METRIC])
            else "required_geometry_unavailable"
        )
        output.append(row)
    return output, coordinate_record


def load_worklist():
    reference = pd.read_csv(REFERENCE, sep="\t", dtype=str, keep_default_na=False)
    registry = pd.read_csv(REGISTRY, sep="\t", dtype=str, keep_default_na=False)
    active_by_source = registry.set_index("allobench_source_id")["active_site_uniprot_positions"].to_dict()
    work = []
    for record in reference.to_dict("records"):
        if record["reference_label"] == "allosteric":
            positions = set()
            for source_id in filter(None, record["reference_source_ids"].split(";")):
                positions.update(parse_positions(active_by_source.get(source_id, "")))
            record["orthosteric_site_uniprot_positions"] = ";".join(
                str(value) for value in sorted(positions)
            )
            record["orthosteric_site_definition"] = "AlloBench_active_site_UniProt_positions"
        else:
            record["orthosteric_site_uniprot_positions"] = ""
            record["orthosteric_site_definition"] = "same_exact_observed_orthosteric_binding_site"
        work.append(record)
    by_pdb = defaultdict(list)
    for record in work:
        by_pdb[record["pdb_id"].lower()].append(record)
    with UNIPROT_CACHE.open("rb") as handle:
        cache = pickle.load(handle)
    proteins = set(reference["uniprot"])
    sequences = {
        protein: str(cache.get(protein, {}).get("sequence", "")).upper()
        for protein in proteins
    }
    return reference, sorted(by_pdb.items()), sequences


def quantile(value: pd.Series, probability: float) -> float:
    clean = pd.to_numeric(value, errors="coerce").dropna()
    return float(clean.quantile(probability)) if len(clean) else np.nan


def protein_macro_auc(frame: pd.DataFrame, metric: str):
    values = []
    rows_used = 0
    for _protein, group in frame.groupby("uniprot"):
        subset = group[["reference_label", metric]].copy()
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        subset = subset.dropna()
        if subset["reference_label"].nunique() != 2:
            continue
        values.append(roc_auc_score(subset["reference_label"].eq("allosteric"), subset[metric]))
        rows_used += len(subset)
    return (float(np.mean(values)) if values else np.nan, len(values), rows_used)


def evaluate_definitions(geometry: pd.DataFrame):
    usable = geometry.loc[geometry["geometry_status"].eq("ok")].copy()
    comparison_universe = usable.loc[
        usable["reference_label"].eq("orthosteric")
        | usable["source_site_relation"].eq("distinct_from_orthosteric_site")
    ].copy()
    comparison_universe["binary_label"] = (
        comparison_universe["reference_label"].eq("allosteric").astype(int)
    )
    common = comparison_universe.dropna(subset=list(DISTANCE_METRICS)).copy()
    strict_no_overlap = comparison_universe.loc[
        comparison_universe["reference_label"].eq("orthosteric")
        | (
            comparison_universe["reference_label"].eq("allosteric")
            & pd.to_numeric(
                comparison_universe["candidate_orthosteric_site_residue_overlap_count"],
                errors="coerce",
            ).eq(0)
        )
    ].copy()
    comparison_rows = []
    distribution_rows = []
    for metric in DISTANCE_METRICS:
        available = comparison_universe.dropna(subset=[metric]).copy()
        observation_auc = (
            roc_auc_score(available["binary_label"], available[metric])
            if available["binary_label"].nunique() == 2 else np.nan
        )
        common_auc = (
            roc_auc_score(common["binary_label"], common[metric])
            if common["binary_label"].nunique() == 2 else np.nan
        )
        macro_auc, macro_groups, macro_rows = protein_macro_auc(available, metric)
        strict_available = strict_no_overlap.dropna(subset=[metric]).copy()
        strict_auc = (
            roc_auc_score(strict_available["binary_label"], strict_available[metric])
            if strict_available["binary_label"].nunique() == 2 else np.nan
        )
        strict_macro_auc, strict_macro_groups, strict_macro_rows = protein_macro_auc(
            strict_available, metric
        )
        allosteric = available.loc[available["binary_label"].eq(1), metric]
        orthosteric = available.loc[available["binary_label"].eq(0), metric]
        comparison_rows.append({
            "metric": metric,
            "definition": METRIC_DEFINITIONS[metric],
            "selection": "candidate_awaiting_user_selection",
            "comparison_rows_available": len(available),
            "comparison_allosteric_rows_available": int(available["binary_label"].sum()),
            "comparison_orthosteric_rows_available": int((available["binary_label"] == 0).sum()),
            "common_complete_rows": len(common),
            "observation_AUROC_descriptive": observation_auc,
            "common_rows_AUROC_descriptive": common_auc,
            "within_protein_macro_AUROC_descriptive": macro_auc,
            "within_protein_groups": macro_groups,
            "within_protein_rows": macro_rows,
            "no_residue_overlap_sensitivity_rows": len(strict_available),
            "no_residue_overlap_sensitivity_allosteric_rows": int(strict_available["binary_label"].sum()),
            "no_residue_overlap_observation_AUROC_descriptive": strict_auc,
            "no_residue_overlap_within_protein_macro_AUROC_descriptive": strict_macro_auc,
            "no_residue_overlap_within_protein_groups": strict_macro_groups,
            "no_residue_overlap_within_protein_rows": strict_macro_rows,
            "allosteric_median_A": quantile(allosteric, 0.5),
            "orthosteric_median_A": quantile(orthosteric, 0.5),
            "orthosteric_exact_zero_fraction": float((pd.to_numeric(orthosteric) == 0).mean()),
        })
        for label, group in available.groupby("reference_label"):
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            distribution_rows.append({
                "metric": metric, "reference_label": label,
                "source_site_relation": "reported_distinct_allosteric_or_orthosteric_control",
                "rows": len(series), "median_A": quantile(series, 0.5),
                "q25_A": quantile(series, 0.25), "q75_A": quantile(series, 0.75),
                "minimum_A": float(series.min()), "maximum_A": float(series.max()),
                "fraction_ge_8A": float((series >= 8).mean()),
                "fraction_ge_10A": float((series >= 10).mean()),
                "fraction_ge_12A": float((series >= 12).mean()),
            })
        overlap = usable.loc[
            usable["reference_label"].eq("allosteric")
            & usable["source_site_relation"].ne("distinct_from_orthosteric_site"), metric
        ].dropna()
        if len(overlap):
            distribution_rows.append({
                "metric": metric, "reference_label": "allosteric",
                "source_site_relation": "overlapping_mixed_or_unspecified",
                "rows": len(overlap), "median_A": quantile(overlap, 0.5),
                "q25_A": quantile(overlap, 0.25), "q75_A": quantile(overlap, 0.75),
                "minimum_A": float(overlap.min()), "maximum_A": float(overlap.max()),
                "fraction_ge_8A": float((overlap >= 8).mean()),
                "fraction_ge_10A": float((overlap >= 10).mean()),
                "fraction_ge_12A": float((overlap >= 12).mean()),
            })

    correlation_rows = []
    distinct_allosteric = comparison_universe.loc[comparison_universe["binary_label"].eq(1)]
    for left_index, left in enumerate(DISTANCE_METRICS):
        for right in DISTANCE_METRICS[left_index + 1:]:
            paired = distinct_allosteric[[left, right]].dropna()
            correlation = spearmanr(paired[left], paired[right]).statistic if len(paired) >= 3 else np.nan
            correlation_rows.append({
                "population": "exact_distinct_allosteric",
                "metric_1": left, "metric_2": right,
                "paired_rows": len(paired), "spearman_rho": correlation,
            })
    return (
        pd.DataFrame(comparison_rows), pd.DataFrame(distribution_rows),
        pd.DataFrame(correlation_rows), len(comparison_universe), len(common),
    )


def self_tests() -> dict:
    # A side-chain contact can be close while the corresponding C-alpha is far.
    ligand = np.array([[5.8, 0.0, 0.0]])
    side_chain = np.array([[5.9, 0.0, 0.0]])
    ca = np.array([[0.0, 0.0, 0.0]])
    sidechain_sensitive = min_distance(ligand, side_chain) < 0.2 and min_distance(ligand, ca) > 5.0
    # A residue set compared with itself must have zero site distance.
    site = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    self_zero = min_distance(site, site) == 0.0 and centroid_distance(site, site) == 0.0
    return {
        "sidechain_contact_detected_by_heavy_atoms_but_not_CA": bool(sidechain_sensitive),
        "identical_site_minimum_and_centroid_distances_zero": bool(self_zero),
        "passed": bool(sidechain_sensitive and self_zero),
    }


def main() -> None:
    args = parse_args()
    start = time.time()
    validation = json.loads(CHECKPOINT5_VALIDATION.read_text())
    if validation.get("status") != "validated":
        raise RuntimeError("checkpoint 5 is not independently validated")
    tests = self_tests()
    if not tests["passed"]:
        raise RuntimeError("checkpoint-6 geometry self-tests failed")

    reference, tasks, sequences = load_worklist()
    if args.max_pdb:
        tasks = tasks[:args.max_pdb]
    results, coordinate_rows = [], []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_worker, initargs=(sequences,)
    ) as executor:
        future_map = {executor.submit(process_pdb, task): task[0] for task in tasks}
        for completed, future in enumerate(as_completed(future_map), 1):
            rows, coordinate = future.result()
            results.extend(rows)
            coordinate_rows.append(coordinate)
            if completed % 100 == 0 or completed == len(tasks):
                print(f"checkpoint6 coordinates: {completed:,}/{len(tasks):,} PDB", flush=True)

    geometry = pd.DataFrame(results).sort_values(["source_ordinal", "observation_id"], kind="mergesort")
    coordinates = pd.DataFrame(coordinate_rows).sort_values("pdb_id", kind="mergesort")
    if args.max_pdb:
        destination = args.pilot_output or Path("/tmp/checkpoint6_pilot.tsv.gz")
        atomic_tsv(geometry, destination, compression="gzip")
        print(json.dumps({
            "pilot": True, "pdb": len(tasks), "rows": len(geometry),
            "status_counts": geometry["geometry_status"].value_counts().to_dict(),
            "output": str(destination),
        }, indent=2))
        return

    if len(geometry) != len(reference) or set(geometry.observation_id) != set(reference.observation_id):
        raise RuntimeError("checkpoint-6 geometry does not preserve the checkpoint-5 reference universe")
    comparison, summaries, correlations, comparison_rows, common_rows = evaluate_definitions(geometry)
    status_counts = (
        geometry.groupby(["reference_label", "geometry_status"], dropna=False)
        .size().rename("rows").reset_index()
    )
    distinct_overlap_audit = geometry.loc[
        geometry["geometry_status"].eq("ok")
        & geometry["reference_label"].eq("allosteric")
        & geometry["source_site_relation"].eq("distinct_from_orthosteric_site")
        & pd.to_numeric(
            geometry["candidate_orthosteric_site_residue_overlap_count"], errors="coerce"
        ).gt(0),
        [
            "observation_id", "source_ordinal", "uniprot", "pdb_id", "receptor_chain",
            "ligand_ccd", "ligand_chain", "ligand_auth_seq_id", "reference_source_ids",
            "binding_uniprot_positions", "orthosteric_site_uniprot_positions",
            "candidate_orthosteric_site_residue_overlap_count",
            "candidate_orthosteric_site_residue_jaccard",
            "ligand_to_orthosteric_site_min_heavy_A", "orthosteric_site_mapping_method",
            "checkpoint4_binding_map_jaccard",
        ],
    ].copy()

    atomic_tsv(geometry, GEOMETRY, compression="gzip")
    atomic_tsv(coordinates, COORDINATES, compression="gzip")
    atomic_tsv(comparison, COMPARISON)
    atomic_tsv(summaries, SUMMARIES)
    atomic_tsv(correlations, CORRELATIONS)
    atomic_tsv(status_counts, STATUS_COUNTS)
    atomic_tsv(distinct_overlap_audit, DISTINCT_OVERLAP_AUDIT, compression="gzip")

    input_rows = []
    for asset_id, path in (
        ("checkpoint5_reference", REFERENCE), ("allobench_registry", REGISTRY),
        ("checkpoint5_validation", CHECKPOINT5_VALIDATION), ("uniprot_cache", UNIPROT_CACHE),
        ("checkpoint6_coordinate_fetch_log", FETCH_LOG),
    ):
        input_rows.append({
            "asset_id": asset_id, "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    atomic_tsv(pd.DataFrame(input_rows), INPUT_HASHES)

    spec = {
        "checkpoint": 6,
        "status": "candidates_validated_awaiting_user_selection",
        "selected_distance": None,
        "units": "angstrom",
        "candidate_ligand_atom_scope": "exact deposited non-polymer ligand instance; heavy atoms; hydrogens excluded",
        "orthosteric_site_scope_allosteric_reference": "AlloBench active-site UniProt residues mapped onto the exact receptor chain",
        "orthosteric_site_scope_orthosteric_reference": "the exact observed BioLiP orthosteric binding residues in the same structure",
        "altloc_policy": "highest occupancy coordinate per atom name",
        "polymer_name_collision_policy": "polymer residues are never accepted as ligand instances",
        "ranking_direction": "larger distance is more distal; no hard threshold in the ranking score",
        "residue_overlap": "separate structural QC flag; never hidden inside the continuous distance",
        "descriptive_thresholds_A": [8, 10, 12],
        "candidate_definitions": [
            {"metric": metric, "definition": METRIC_DEFINITIONS[metric]}
            for metric in DISTANCE_METRICS
        ],
        "selection_policy": "no distance selected until user review; checkpoint 7 must not start beforehand",
        "label_metrics_role": "descriptive comparison only; checkpoint 8 supplies held-out evaluation",
        "descriptive_results": comparison.to_dict("records"),
    }
    atomic_json(SPEC, spec)

    coordinate_counts = coordinates["coordinate_source"].value_counts().to_dict()
    relation_counts = (
        geometry.loc[geometry.reference_label.eq("allosteric"), "source_site_relation"]
        .value_counts().to_dict()
    )
    reported_distinct_ok = geometry.loc[
        geometry.geometry_status.eq("ok")
        & geometry.reference_label.eq("allosteric")
        & geometry.source_site_relation.eq("distinct_from_orthosteric_site")
    ]
    report = f"""# Checkpoint 6: structural distance definition comparison

Status: **built; independent validation required**

All {len(reference):,} checkpoint-5 exact reference observations were retained.
Only deterministic coordinate paths for the {len(tasks):,} named PDB IDs were
queried; no coordinate directory was recursively enumerated.

## Coordinate and geometry completion

- Coordinate source counts: `{coordinate_counts}`
- Geometry status by label is recorded in `CHECKPOINT6_GEOMETRY_STATUS_COUNTS.tsv`.
- Allosteric source-site relation counts before geometry filtering: `{relation_counts}`
- Reported-distinct allosteric rows with direct candidate/orthosteric-site residue overlap:
  **{len(distinct_overlap_audit):,}/{len(reported_distinct_ok):,}**.  This is retained
  as a source-annotation-versus-geometry audit, not automatically relabeled.
- Comparison population before per-metric missingness: **{comparison_rows:,}** rows
- Rows complete for all seven distance definitions: **{common_rows:,}**

## Compared definitions

Seven continuous definitions were computed on the same exact observation grain:

1. ligand heavy atom to orthosteric-site heavy atom minimum;
2. ligand heavy atom to orthosteric-site C-alpha minimum;
3. candidate binding-site heavy atom to orthosteric-site heavy atom minimum;
4. candidate binding-site C-alpha to orthosteric-site C-alpha minimum;
5. ligand heavy-atom centroid to orthosteric-site heavy-atom centroid;
6. ligand heavy-atom centroid to orthosteric-site C-alpha centroid;
7. candidate-site C-alpha centroid to orthosteric-site C-alpha centroid.

Observation AUROC and within-protein macro AUROC in the comparison table are
descriptive diagnostics, not model selection or independent validation.

## Selection status

No distance definition is frozen in this checkpoint. The comparison table reports
all seven candidates on the same rows so that empirical separation and geometric
meaning can be considered together. No distance cutoff was optimized; the 8/10/12 A
columns are descriptive summaries only.

Candidate/orthosteric-site residue overlap remains a separate QC field. Metrics on
the no-residue-overlap subset are reported as a circularity-aware sensitivity, not
as an automatic selection rule.

## Scope boundary

This checkpoint computes and validates geometry only. It does not fit the Bayesian
scorer or rank the unlabeled BioLiP universe. Checkpoint 7 remains blocked until
the distance definition is selected.
"""
    atomic_text(REPORT, report)

    output_paths = {
        "geometry": GEOMETRY, "coordinate_manifest": COORDINATES,
        "comparison": COMPARISON, "distributions": SUMMARIES,
        "correlations": CORRELATIONS, "status_counts": STATUS_COUNTS,
        "reported_distinct_overlap_audit": DISTINCT_OVERLAP_AUDIT,
        "distance_spec": SPEC, "input_hashes": INPUT_HASHES, "report": REPORT,
    }
    build = {
        "checkpoint": 6,
        "status": "complete_pending_independent_validation",
        "elapsed_seconds": round(time.time() - start, 3),
        "workers": args.workers,
        "reference_rows": len(reference), "unique_pdb": len(tasks),
        "geometry_status_counts": geometry.geometry_status.value_counts().to_dict(),
        "coordinate_source_counts": coordinate_counts,
        "comparison_rows": comparison_rows,
        "all_seven_metrics_complete_rows": common_rows,
        "reported_distinct_residue_overlap_rows": len(distinct_overlap_audit),
        "selected_distance": None,
        "selection_status": "awaiting_user_selection",
        "self_tests": tests,
        "recursive_coordinate_scan_performed": False,
        "coordinate_content_hashing_performed": False,
        "coordinate_hash_reason": "avoid a second high-I/O pass; exact resolved paths, sizes and mtimes are frozen",
        "outputs": {
            key: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for key, path in output_paths.items()
        },
    }
    atomic_json(BUILD, build)
    print(json.dumps(build, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
