#!/usr/bin/env python3
"""Checkpoint 9c: compute candidate distance to known orthosteric sites.

For every frozen BioLiP candidate observation, exact ligand coordinates and the
already mapped receptor chain are retained. All exact orthosteric-site residue
definitions for the same UniProt protein are transferred to that chain. The
selected distance is the smallest ligand-centroid-to-site-C-alpha-centroid
distance among definitions with at least 80% residue recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
VALIDATION = PACKAGE / "validation"
MANIFESTS = PACKAGE / "manifests"

CANDIDATES = DATA / "CHECKPOINT9_CANDIDATE_UNIVERSE.tsv.gz"
SITE_DEFINITIONS = DATA / "CHECKPOINT9_ORTHOSTERIC_SITE_DEFINITIONS.tsv.gz"
COORDINATES = DATA / "CHECKPOINT9_COORDINATE_MANIFEST.tsv.gz"
SEQUENCES = DATA / "CHECKPOINT9_UNIPROT_SEQUENCES.tsv.gz"
PREPARATION = VALIDATION / "CHECKPOINT9_PREPARATION.json"
ASSET_BUILD = VALIDATION / "CHECKPOINT9_ASSET_BUILD.json"

GEOMETRY = DATA / "CHECKPOINT9_CANDIDATE_GEOMETRY.tsv.gz"
STATUS_COUNTS = DATA / "CHECKPOINT9_GEOMETRY_STATUS_COUNTS.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT9_GEOMETRY_INPUT_HASHES.tsv"
BUILD = VALIDATION / "CHECKPOINT9_GEOMETRY_BUILD.json"

SIFTS_ROOTS = (
    ROOT / "analysis/structure_known_candidate_rebuild/assets/sifts",
    ROOT / "analysis/sifts",
    ROOT / "analysis/structure_known_biolip_xai/assets/sifts",
)
SELECTED_DISTANCE = "ligand_centroid_to_nearest_orthosteric_site_CA_centroid_A"
MIN_SITE_RECOVERY = 0.80
MIN_BINDING_MAP_JACCARD = 0.80
RESIDUE_TOKEN = re.compile(r"^[A-Za-z](-?\d+)([A-Za-z]?)$")

_SEQUENCES: Dict[str, str] = {}
_SITE_DEFINITIONS: Dict[str, List[dict]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CHECKPOINT9_WORKERS", "32")))
    parser.add_argument("--max-pdb", type=int, default=0)
    parser.add_argument("--pilot-output", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_tsv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_positions(value: object) -> List[int]:
    return sorted({int(token) for token in str(value or "").split(";") if token})


def parse_author_residues(value: object) -> List[Tuple[int, str]]:
    residues = []
    for token in str(value or "").split():
        match = RESIDUE_TOKEN.match(token.strip())
        if match:
            residues.append((int(match.group(1)), match.group(2) or ""))
    return sorted(set(residues))


def set_jaccard(first: set, second: set) -> float:
    union = first | second
    return len(first & second) / len(union) if union else np.nan


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
    if heavy.size == 0:
        heavy = np.empty((0, 3), dtype=float)
    return heavy, best.get("CA", (None, None))[1]


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
    plain = (key[0], "")
    if plain in index:
        return plain
    matches = by_number.get(key[0], [])
    return matches[0] if len(matches) == 1 else None


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
        return {}, {}, np.nan, np.nan, np.nan
    alignment = aligner.align(reference_sequence, observed)[0]
    uni_to_key, key_to_uni = {}, {}
    matches = mapped = 0
    for (u0, u1), (c0, c1) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(min(u1 - u0, c1 - c0)):
            key = chain_sequence[c0 + offset][1]
            uniprot_position = u0 + offset + 1
            uni_to_key[uniprot_position] = key
            key_to_uni[key] = uniprot_position
            mapped += 1
            if reference_sequence[u0 + offset] == chain_sequence[c0 + offset][0]:
                matches += 1
    identity = matches / mapped if mapped else np.nan
    chain_coverage = mapped / len(observed) if observed else np.nan
    uniprot_coverage = mapped / len(reference_sequence) if reference_sequence else np.nan
    return uni_to_key, key_to_uni, identity, chain_coverage, uniprot_coverage


def find_ligand(model: gemmi.Model, ccd: str, chain_name: str, residue_number: str):
    try:
        number = int(str(residue_number))
    except ValueError:
        return None, 0, 0
    hits, polymer_collisions = [], 0
    for chain in model:
        if chain.name != chain_name:
            continue
        for residue in chain:
            if residue.name.upper() != ccd.upper() or int(residue.seqid.num) != number:
                continue
            if residue.entity_type == gemmi.EntityType.Polymer:
                polymer_collisions += 1
                continue
            if residue.entity_type == gemmi.EntityType.Water:
                continue
            heavy, _ca = best_heavy_atoms(residue)
            if len(heavy):
                hits.append(heavy)
    return (hits[0] if len(hits) == 1 else None, len(hits), polymer_collisions)


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


def initialize_worker(sequences: Dict[str, str], site_definitions: Dict[str, List[dict]]) -> None:
    global _SEQUENCES, _SITE_DEFINITIONS
    _SEQUENCES = sequences
    _SITE_DEFINITIONS = site_definitions


def failure_rows(rows: Sequence[dict], status: str, source: str, path: str):
    return [{
        **row,
        "coordinate_source": source,
        "coordinate_path": path,
        "geometry_status": status,
    } for row in rows]


def map_site_definitions(uniprot: str, uni_to_key: dict, index: dict):
    mapped = []
    for definition in _SITE_DEFINITIONS.get(uniprot, []):
        positions = definition["positions"]
        keys = {uni_to_key[position] for position in positions if position in uni_to_key}
        recovery = len(keys) / len(positions) if positions else np.nan
        ca = stack_ca(keys, index)
        heavy = stack_heavy(keys, index)
        if ca is not None and len(ca):
            mapped.append({
                **definition,
                "keys": keys,
                "recovered": len(keys),
                "recovery": recovery,
                "ca": ca,
                "heavy": heavy,
            })
    return mapped


def process_pdb(task):
    pdb_id, coordinate_path, coordinate_source, rows = task
    if not coordinate_path:
        return failure_rows(rows, "coordinate_missing", coordinate_source, coordinate_path)
    try:
        structure = gemmi.read_structure(coordinate_path)
        model = structure[0]
    except Exception as error:
        return failure_rows(
            rows, f"coordinate_read_error:{type(error).__name__}", coordinate_source,
            coordinate_path,
        )

    aligner = new_aligner()
    chain_cache = {}
    mapping_cache = {}
    site_cache = {}
    output = []
    for task_row in rows:
        row = dict(task_row)
        row.update({"coordinate_source": coordinate_source, "coordinate_path": coordinate_path})
        chain_name, uniprot = row["receptor_chain"], row["uniprot"]
        if chain_name not in chain_cache:
            chain_cache[chain_name] = chain_index(model, chain_name)
        index, chain_sequence, by_number = chain_cache[chain_name]
        if not index:
            row["geometry_status"] = "receptor_chain_absent"
            output.append(row)
            continue

        mapping_key = (chain_name, uniprot)
        if mapping_key not in mapping_cache:
            sequence = _SEQUENCES.get(uniprot, "")
            mapping_cache[mapping_key] = alignment_maps(sequence, chain_sequence, aligner)
        uni_to_key, key_to_uni, identity, chain_coverage, uniprot_coverage = mapping_cache[mapping_key]
        row["mapping_method"] = "canonical_global_alignment"
        row["mapping_identity"] = identity
        row["mapping_chain_coverage"] = chain_coverage
        row["mapping_uniprot_coverage"] = uniprot_coverage
        if not uni_to_key:
            row["geometry_status"] = "protein_mapping_unavailable"
            output.append(row)
            continue

        requested_binding = parse_author_residues(row["binding_residues_auth_raw"])
        candidate_keys = {
            resolved for requested in requested_binding
            for resolved in [resolve_key(index, by_number, requested)] if resolved is not None
        }
        frozen_binding = set(parse_positions(row["binding_uniprot_positions"]))
        independently_mapped_binding = {key_to_uni[key] for key in candidate_keys if key in key_to_uni}
        binding_jaccard = set_jaccard(frozen_binding, independently_mapped_binding)
        row["candidate_residues_requested"] = len(requested_binding)
        row["candidate_residues_recovered"] = len(candidate_keys)
        row["candidate_residue_recovery_fraction"] = (
            len(candidate_keys) / len(requested_binding) if requested_binding else np.nan
        )
        row["checkpoint4_binding_map_jaccard"] = binding_jaccard
        if not np.isfinite(binding_jaccard) or binding_jaccard < MIN_BINDING_MAP_JACCARD:
            row["geometry_status"] = "candidate_site_mapping_failed_checkpoint4_crosscheck"
            output.append(row)
            continue

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

        if mapping_key not in site_cache:
            site_cache[mapping_key] = map_site_definitions(uniprot, uni_to_key, index)
        mapped_sites = site_cache[mapping_key]
        high_recovery_sites = [site for site in mapped_sites if site["recovery"] >= MIN_SITE_RECOVERY]
        row["orthosteric_site_definitions_total"] = len(_SITE_DEFINITIONS.get(uniprot, []))
        row["orthosteric_site_definitions_with_CA"] = len(mapped_sites)
        row["orthosteric_site_definitions_high_recovery"] = len(high_recovery_sites)
        if not high_recovery_sites:
            row["geometry_status"] = "orthosteric_site_insufficient_recovery"
            output.append(row)
            continue

        comparisons = []
        for site in high_recovery_sites:
            comparisons.append({
                "site": site,
                "selected_distance": centroid_distance(ligand, site["ca"]),
                "min_ca_distance": min_distance(ligand, site["ca"]),
                "min_heavy_distance": min_distance(ligand, site["heavy"]),
            })
        comparisons = [item for item in comparisons if np.isfinite(item["selected_distance"])]
        if not comparisons:
            row["geometry_status"] = "required_geometry_unavailable"
            output.append(row)
            continue
        comparisons.sort(key=lambda item: (item["selected_distance"], item["site"]["id"]))
        nearest = comparisons[0]
        nearest_site = nearest["site"]
        candidate_ca = stack_ca(candidate_keys, index)
        candidate_heavy = stack_heavy(candidate_keys, index)
        overlap = candidate_keys & nearest_site["keys"]
        row.update({
            "nearest_orthosteric_site_definition_id": nearest_site["id"],
            "nearest_orthosteric_site_uniprot_positions": nearest_site["position_text"],
            "nearest_orthosteric_site_residues_requested": len(nearest_site["positions"]),
            "nearest_orthosteric_site_residues_recovered": nearest_site["recovered"],
            "nearest_orthosteric_site_residue_recovery_fraction": nearest_site["recovery"],
            SELECTED_DISTANCE: nearest["selected_distance"],
            "second_nearest_orthosteric_site_CA_centroid_A": (
                comparisons[1]["selected_distance"] if len(comparisons) > 1 else np.nan
            ),
            "ligand_to_nearest_orthosteric_site_min_CA_A": nearest["min_ca_distance"],
            "ligand_to_nearest_orthosteric_site_min_heavy_A": nearest["min_heavy_distance"],
            "candidate_site_CA_centroid_to_nearest_orthosteric_site_CA_centroid_A": (
                centroid_distance(candidate_ca, nearest_site["ca"])
            ),
            "candidate_site_to_nearest_orthosteric_site_min_CA_A": (
                min_distance(candidate_ca, nearest_site["ca"])
            ),
            "candidate_site_to_nearest_orthosteric_site_min_heavy_A": (
                min_distance(candidate_heavy, nearest_site["heavy"])
            ),
            "candidate_nearest_orthosteric_site_residue_overlap_count": len(overlap),
            "candidate_nearest_orthosteric_site_residue_jaccard": set_jaccard(
                candidate_keys, nearest_site["keys"]
            ),
            "ligand_heavy_atom_count_from_coordinate": len(ligand),
            "geometry_status": "ok",
        })
        distance = row[SELECTED_DISTANCE]
        row["distance_ge_8A"] = bool(distance >= 8.0)
        row["distance_ge_10A"] = bool(distance >= 10.0)
        row["distance_ge_12A"] = bool(distance >= 12.0)
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    preparation = json.loads(PREPARATION.read_text())
    assets = json.loads(ASSET_BUILD.read_text())
    if preparation.get("status") != "validated_candidate_universe_frozen":
        raise RuntimeError("checkpoint 9a is not validated")
    if assets.get("status") != "validated":
        raise RuntimeError("checkpoint 9b assets are incomplete")

    candidates = pd.read_csv(CANDIDATES, sep="\t", dtype=str, keep_default_na=False)
    sites = pd.read_csv(SITE_DEFINITIONS, sep="\t", dtype=str, keep_default_na=False)
    coordinates = pd.read_csv(COORDINATES, sep="\t", dtype=str, keep_default_na=False)
    sequences_frame = pd.read_csv(SEQUENCES, sep="\t", dtype=str, keep_default_na=False)
    if len(candidates) != 19_920 or len(sites) != 1_960:
        raise RuntimeError("checkpoint-9 frozen input count changed")
    if coordinates.coordinate_path.eq("").any() or sequences_frame.sequence.eq("").any():
        raise RuntimeError("checkpoint-9 asset manifest is incomplete")
    coordinate_lookup = coordinates.set_index("pdb_id").to_dict("index")
    sequences = sequences_frame.set_index("uniprot").sequence.to_dict()
    site_definitions = defaultdict(list)
    for record in sites.to_dict("records"):
        site_definitions[record["uniprot"]].append({
            "id": record["orthosteric_site_definition_id"],
            "position_text": record["orthosteric_site_uniprot_positions"],
            "positions": parse_positions(record["orthosteric_site_uniprot_positions"]),
        })
    site_definitions = {key: sorted(value, key=lambda row: row["id"]) for key, value in site_definitions.items()}

    grouped = []
    for pdb_id, group in candidates.groupby("pdb_id", sort=True):
        coordinate = coordinate_lookup.get(pdb_id)
        if coordinate is None:
            raise RuntimeError(f"candidate PDB absent from coordinate manifest: {pdb_id}")
        grouped.append((
            pdb_id,
            coordinate["coordinate_path"],
            coordinate["coordinate_source"],
            group.to_dict("records"),
        ))
    if args.max_pdb:
        grouped = grouped[: args.max_pdb]

    output_rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(sequences, site_definitions),
    ) as executor:
        futures = {executor.submit(process_pdb, task): task[0] for task in grouped}
        for completed, future in enumerate(as_completed(futures), 1):
            output_rows.extend(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"candidate geometry: {completed:,}/{len(futures):,} PDBs; "
                    f"rows={len(output_rows):,}", flush=True,
                )
    geometry = pd.DataFrame(output_rows).sort_values(
        ["uniprot", "connectivity_key", "pdb_id", "source_ordinal"], kind="mergesort",
    )
    output_path = args.pilot_output or GEOMETRY
    atomic_tsv(geometry, output_path, compression="gzip")
    if args.max_pdb:
        print(f"pilot output: {output_path}; rows={len(geometry):,}")
        return

    if len(geometry) != len(candidates) or geometry.observation_id.nunique() != len(candidates):
        raise RuntimeError("geometry output does not preserve the frozen observation universe")
    statuses = geometry.geometry_status.value_counts(dropna=False).rename_axis(
        "geometry_status"
    ).reset_index(name="rows")
    atomic_tsv(statuses, STATUS_COUNTS)
    ok = geometry.loc[geometry.geometry_status.eq("ok")].copy()
    if ok.empty or not np.isfinite(pd.to_numeric(ok[SELECTED_DISTANCE], errors="coerce")).all():
        raise RuntimeError("no complete finite candidate geometry was produced")
    if (pd.to_numeric(ok.nearest_orthosteric_site_residue_recovery_fraction) < MIN_SITE_RECOVERY).any():
        raise RuntimeError("a ranked geometry row violates minimum orthosteric-site recovery")
    if (pd.to_numeric(ok.checkpoint4_binding_map_jaccard) < MIN_BINDING_MAP_JACCARD).any():
        raise RuntimeError("a ranked geometry row violates the checkpoint-4 mapping crosscheck")

    input_rows = []
    for asset_id, path in (
        ("candidate_universe", CANDIDATES),
        ("orthosteric_site_definitions", SITE_DEFINITIONS),
        ("coordinate_manifest", COORDINATES),
        ("uniprot_sequences", SEQUENCES),
        ("preparation_validation", PREPARATION),
        ("asset_validation", ASSET_BUILD),
    ):
        input_rows.append({
            "asset_id": asset_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    atomic_tsv(pd.DataFrame(input_rows), INPUT_HASHES)
    build = {
        "checkpoint": "9c",
        "status": "validated_candidate_geometry",
        "candidate_rows": len(geometry),
        "geometry_ok_rows": len(ok),
        "geometry_ok_fraction": len(ok) / len(geometry),
        "geometry_status_counts": geometry.geometry_status.value_counts().to_dict(),
        "geometry_ok_proteins": ok.uniprot.nunique(),
        "geometry_ok_pairs": ok[["uniprot", "full_inchikey"]].drop_duplicates().shape[0],
        "geometry_ok_exact_site_signatures": ok.exact_site_ligand_signature_id.nunique(),
        "distance_definition": (
            "minimum across exact same-protein orthosteric-site definitions of the "
            "Euclidean distance between candidate ligand heavy-atom centroid and "
            "transferred orthosteric-site residue C-alpha centroid"
        ),
        "minimum_orthosteric_site_residue_recovery": MIN_SITE_RECOVERY,
        "minimum_checkpoint4_binding_map_jaccard": MIN_BINDING_MAP_JACCARD,
        "distance_ge_8A_rows": int(ok.distance_ge_8A.astype(str).str.lower().eq("true").sum()),
        "distance_ge_10A_rows": int(ok.distance_ge_10A.astype(str).str.lower().eq("true").sum()),
        "distance_ge_12A_rows": int(ok.distance_ge_12A.astype(str).str.lower().eq("true").sum()),
        "geometry_sha256": sha256(GEOMETRY),
        "directory_enumeration_performed": False,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(build, BUILD)
    print(json.dumps(build, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
