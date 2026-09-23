#!/usr/bin/env python3
"""Build targeted pocket-geometry graphs for a reduced LABind-style pilot.

Only coordinate files named in Protein_Embedding_Map.tsv (plus already frozen
C9 structures) are opened.  No directory traversal is performed.  Canonical
UniProt residue indices are obtained by local sequence alignment, making them
directly indexable into the full-sequence ESM3 tensors.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.PDB import MMCIFParser, PDBParser


ROOT = Path("/disk9/13.Heesu_Allostery")
SHARED = Path("/shared_data/11.HS_allostery")
REMOTE = Path("/disk1/11.HS_allostery")
PACKAGE = ROOT / "analysis/new_pairing_d1_pilot"
SOURCE = PACKAGE / "data/NEW_PAIRING_CLEAN_MATCHED_PFAM_C9_COMPONENT_DISJOINT_PILOT.tsv.gz"
MASKS = PACKAGE / "data/COHORT_POCKET_MASKS.json"
PROTEIN_MAP = SHARED / "14.Organized_input/Protein_Embedding_Map.tsv"
SEQUENCE_CACHE = SHARED / "14.Organized_input/uniprot_protein_cache.json"
STRUCTURE_ROOT = SHARED / "BioLiP_CIFs"
OLD_STRUCTURE_ROOT = ROOT / "analysis/c9_site_auprc/cpu_structures"
GRAPH_ROOT = PACKAGE / "data/structure_graphs"
OUTPUT = PACKAGE / "data/NEW_PAIRING_CLEAN_MATCHED_FAMILY_STRUCTURE_READY_PILOT.tsv.gz"
AUDIT_OUTPUT = PACKAGE / "data/STRUCTURE_GRAPH_AUDIT.tsv"
VALIDATION_OUTPUT = PACKAGE / "validation/STRUCTURE_GRAPH_VALIDATION.json"
MAX_CANDIDATES = 3
MAX_GRAPH_RESIDUES = 768
K_NEIGHBORS = 8
MIN_GRAPH_RESIDUES = 10
MIN_ALIGNMENT_IDENTITY = 0.70


AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def remote_graph_path(local):
    relative = Path(local).relative_to(ROOT)
    return str(REMOTE / relative)


def candidate_coordinate_path(pdb_id):
    pdb_id = str(pdb_id).strip().lower()[:4]
    if len(pdb_id) != 4:
        return None
    for suffix in [".cif", ".pdb", ".cif.gz", ".pdb.gz"]:
        path = STRUCTURE_ROOT / (pdb_id + suffix)
        if path.is_file():
            return path
    return None


def candidate_paths(uid, map_rows):
    paths = []
    old = OLD_STRUCTURE_ROOT / (uid + ".pdb")
    if old.is_file():
        paths.append(old)
    pdb_ids = []
    for column in ["Mapped_PDB", "Original_Target_PDB"]:
        for value in map_rows[column].astype(str):
            value = value.strip().lower()
            if len(value) >= 4:
                pdb_ids.append(value[:4])
    for value in map_rows["Embedding_Path"].astype(str):
        stem = Path(value).stem.lower()
        if len(stem) == 4 and stem[0].isdigit():
            pdb_ids.append(stem)
    for pdb_id in dict.fromkeys(pdb_ids):
        path = candidate_coordinate_path(pdb_id)
        if path is not None and path not in paths:
            paths.append(path)
    return paths[:MAX_CANDIDATES]


def open_structure(path):
    text = str(path).lower()
    if text.endswith(".gz"):
        handle = gzip.open(path, "rt")
        structure_id = Path(path).name.split(".")[0]
        parser = MMCIFParser(QUIET=True) if ".cif." in text else PDBParser(QUIET=True)
        try:
            return parser.get_structure(structure_id, handle)
        finally:
            handle.close()
    parser = MMCIFParser(QUIET=True) if text.endswith(".cif") else PDBParser(QUIET=True)
    return parser.get_structure(Path(path).stem, str(path))


def residue_coordinate(residue):
    if "CA" not in residue:
        return None
    try:
        return np.asarray(residue["CA"].coord, dtype=np.float32)
    except Exception:
        return None


def extract_chains(path):
    structure = open_structure(path)
    model = next(structure.get_models())
    chains = []
    for chain in model:
        sequence = []
        coordinates = []
        residue_ids = []
        for residue in chain:
            aa = AA3.get(str(residue.get_resname()).strip().upper())
            coordinate = residue_coordinate(residue)
            if aa is None or coordinate is None or not np.isfinite(coordinate).all():
                continue
            sequence.append(aa)
            coordinates.append(coordinate)
            residue_ids.append("{}:{}{}".format(chain.id, residue.id[1], str(residue.id[2]).strip()))
        if sequence:
            chains.append({
                "chain": str(chain.id),
                "sequence": "".join(sequence),
                "coordinates": np.asarray(coordinates, dtype=np.float32),
                "residue_ids": residue_ids,
            })
    return chains


def align_chain(canonical, chain):
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(canonical, chain["sequence"])
    if len(alignments) == 0:
        return {}, 0.0
    alignment = alignments[0]
    mapping = {}
    matches = 0
    for (canonical_start, canonical_end), (chain_start, chain_end) in zip(*alignment.aligned):
        length = min(canonical_end - canonical_start, chain_end - chain_start)
        for offset in range(length):
            canonical_index = int(canonical_start + offset)
            chain_index = int(chain_start + offset)
            mapping[canonical_index] = chain_index
            matches += int(canonical[canonical_index] == chain["sequence"][chain_index])
    identity = float(matches) / max(len(mapping), 1)
    return mapping, identity


def choose_graph(uid, canonical, mask_one_based, paths):
    requested = sorted({int(value) - 1 for value in mask_one_based if 0 <= int(value) - 1 < len(canonical)})
    best = None
    errors = []
    for path in paths:
        try:
            chains = extract_chains(path)
        except Exception as error:
            errors.append("{}:{}:{}".format(Path(path).name, type(error).__name__, error))
            continue
        for chain in chains:
            mapping, identity = align_chain(canonical, chain)
            mapped_pocket = [index for index in requested if index in mapping]
            candidate = {
                "uid": uid,
                "path": str(path),
                "chain": chain["chain"],
                "identity": identity,
                "mapping": mapping,
                "mapped_pocket": mapped_pocket,
                "chain_coordinates": chain["coordinates"],
                "chain_length": len(chain["sequence"]),
                "requested": requested,
            }
            score = (len(mapped_pocket), identity, len(mapping), -len(chain["sequence"]))
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None:
            candidate = best[1]
            if candidate["identity"] >= 0.95 and len(candidate["mapped_pocket"]) >= max(MIN_GRAPH_RESIDUES, int(0.70 * len(requested))):
                break
    if best is None:
        return None, errors
    selected = best[1]
    if selected["identity"] < MIN_ALIGNMENT_IDENTITY or len(selected["mapped_pocket"]) < MIN_GRAPH_RESIDUES:
        return None, errors + ["best candidate failed identity/residue threshold"]
    indices = np.asarray(selected["mapped_pocket"], dtype=np.int64)
    if len(indices) > MAX_GRAPH_RESIDUES:
        take = np.linspace(0, len(indices) - 1, MAX_GRAPH_RESIDUES).round().astype(np.int64)
        indices = indices[np.unique(take)]
    xyz = np.asarray([
        selected["chain_coordinates"][selected["mapping"][int(index)]] for index in indices
    ], dtype=np.float32)
    distance = np.sqrt(np.maximum(
        ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=-1), 0.0
    )).astype(np.float32)
    k = min(K_NEIGHBORS, len(indices))
    neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k].astype(np.int64)
    neighbor_distance = np.take_along_axis(distance, neighbors, axis=1).astype(np.float32)
    if k < K_NEIGHBORS:
        pad = np.repeat(neighbors[:, -1:], K_NEIGHBORS - k, axis=1)
        pad_distance = np.repeat(neighbor_distance[:, -1:], K_NEIGHBORS - k, axis=1)
        neighbors = np.concatenate([neighbors, pad], axis=1)
        neighbor_distance = np.concatenate([neighbor_distance, pad_distance], axis=1)
    selected.update({
        "embedding_index": indices,
        "ca_xyz": xyz,
        "neighbor_index": neighbors,
        "neighbor_distance": neighbor_distance,
    })
    return selected, errors


def main():
    frame = pd.read_csv(SOURCE, sep="\t", low_memory=False)
    with open(MASKS, encoding="utf-8") as handle:
        masks = json.load(handle)
    with open(SEQUENCE_CACHE, encoding="utf-8") as handle:
        sequence_cache = json.load(handle)
    protein_map = pd.read_csv(PROTEIN_MAP, sep="\t", dtype=str).fillna("")
    protein_map["uid"] = protein_map["UniProt_ID"].str.split("-").str[0]
    protein_map = protein_map[protein_map["uid"].isin(set(frame["uniprot"].astype(str)))]
    selected_embedding = frame.drop_duplicates("uniprot").set_index("uniprot")["protein_embedding_path"].astype(str).to_dict()

    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    audit = []
    graph_paths = {}
    for position, uid in enumerate(sorted(set(frame["uniprot"].astype(str))), 1):
        embedding_path = selected_embedding[uid]
        expected_suffix = "/14.Organized_input/Protein_Embeddings/{}.pt".format(uid)
        canonical = str(sequence_cache.get(uid, {}).get("seq", "")).strip()
        paths = candidate_paths(uid, protein_map[protein_map["uid"].eq(uid)])
        status = "ready"
        reason = ""
        graph = None
        errors = []
        if expected_suffix not in embedding_path:
            status, reason = "excluded", "embedding_is_not_canonical_uniprot_sequence"
        elif not canonical:
            status, reason = "excluded", "canonical_sequence_missing"
        elif not paths:
            status, reason = "excluded", "coordinate_file_missing"
        else:
            graph, errors = choose_graph(uid, canonical, masks.get(uid, []), paths)
            if graph is None:
                status, reason = "excluded", "coordinate_alignment_or_pocket_coverage_failed"
        record = {
            "uniprot": uid,
            "status": status,
            "reason": reason,
            "protein_embedding_path": embedding_path,
            "n_candidate_coordinate_files": len(paths),
            "candidate_coordinate_files": ";".join(map(str, paths)),
            "errors": " | ".join(errors),
        }
        if graph is not None:
            output_path = GRAPH_ROOT / (uid + ".npz")
            temporary = output_path.with_suffix(".npz.tmp")
            with open(temporary, "wb") as handle:
                np.savez_compressed(
                    handle,
                    embedding_index=graph["embedding_index"],
                    ca_xyz=graph["ca_xyz"],
                    neighbor_index=graph["neighbor_index"],
                    neighbor_distance=graph["neighbor_distance"],
                )
            os.replace(str(temporary), str(output_path))
            graph_paths[uid] = remote_graph_path(output_path)
            record.update({
                "coordinate_file": graph["path"],
                "coordinate_chain": graph["chain"],
                "alignment_identity": graph["identity"],
                "canonical_length": len(canonical),
                "coordinate_chain_length": graph["chain_length"],
                "requested_pocket_residues": len(graph["requested"]),
                "mapped_pocket_residues_before_cap": len(graph["mapped_pocket"]),
                "graph_residues": len(graph["embedding_index"]),
                "pocket_coordinate_coverage": len(graph["mapped_pocket"]) / max(len(graph["requested"]), 1),
                "graph_path": str(output_path),
            })
        audit.append(record)
        if position % 25 == 0 or position == frame["uniprot"].nunique():
            print("structure graphs {}/{} ready={}".format(position, frame["uniprot"].nunique(), len(graph_paths)), flush=True)

    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(AUDIT_OUTPUT, sep="\t", index=False)
    ready_proteins = set(graph_paths)
    ready = frame[frame["uniprot"].astype(str).isin(ready_proteins)].copy().reset_index(drop=True)
    ready["structure_graph_path"] = ready["uniprot"].map(graph_paths)
    ready.to_csv(OUTPUT, sep="\t", index=False, compression="gzip")

    split_counts = []
    for split in ["train", "val", "test"]:
        subset = ready[ready["split"].eq(split)]
        split_counts.append({
            "split": split,
            "n_rows": int(len(subset)),
            "n_proteins": int(subset["uniprot"].nunique()),
            "n_components": int(subset["family_component_id"].nunique()),
            "n_allosteric": int(subset["binary_label"].eq(1).sum()),
            "n_orthosteric": int(subset["binary_label"].eq(0).sum()),
        })
    split_uids = {split: set(ready.loc[ready["split"].eq(split), "uniprot"]) for split in ["train", "val", "test"]}
    split_components = {split: set(ready.loc[ready["split"].eq(split), "family_component_id"]) for split in ["train", "val", "test"]}
    pairings = [("train", "val"), ("train", "test"), ("val", "test")]
    ready_audit = audit_frame[audit_frame["status"].eq("ready")]
    report = {
        "status": "validated",
        "model_name_limit": "reduced LABind-style pair adaptation; not an official LABind reproduction",
        "omissions_relative_to_labind": [
            "Ankh and MolFormer encoders are replaced by frozen ESM3 and UniMol2 embeddings",
            "DSSP and MSMS surface-point features are not used",
            "C-alpha k-nearest-neighbor distance features replace LABind's full six-point residue geometry",
            "pair-level allosteric/orthosteric output replaces per-residue binding-site supervision",
        ],
        "source_rows": int(len(frame)),
        "source_proteins": int(frame["uniprot"].nunique()),
        "structure_ready_rows": int(len(ready)),
        "structure_ready_proteins": int(ready["uniprot"].nunique()),
        "excluded_proteins": int(frame["uniprot"].nunique() - ready["uniprot"].nunique()),
        "exclusion_reasons": audit_frame.loc[audit_frame["status"].ne("ready"), "reason"].value_counts().astype(int).to_dict(),
        "min_alignment_identity": float(ready_audit["alignment_identity"].min()) if len(ready_audit) else None,
        "median_alignment_identity": float(ready_audit["alignment_identity"].median()) if len(ready_audit) else None,
        "median_pocket_coordinate_coverage": float(ready_audit["pocket_coordinate_coverage"].median()) if len(ready_audit) else None,
        "min_graph_residues": int(ready_audit["graph_residues"].min()) if len(ready_audit) else None,
        "max_graph_residues": int(ready_audit["graph_residues"].max()) if len(ready_audit) else None,
        "protein_disjoint": all(not (split_uids[left] & split_uids[right]) for left, right in pairings),
        "family_component_disjoint": all(not (split_components[left] & split_components[right]) for left, right in pairings),
        "all_ready_proteins_have_both_labels": bool(ready.groupby("uniprot")["binary_label"].nunique().eq(2).all()),
        "split_counts": split_counts,
        "inputs": {
            "dataset": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
            "masks": {"path": str(MASKS), "sha256": sha256(MASKS)},
            "protein_map": {"path": str(PROTEIN_MAP), "sha256": sha256(PROTEIN_MAP)},
            "sequence_cache": {"path": str(SEQUENCE_CACHE), "sha256": sha256(SEQUENCE_CACHE)},
        },
        "outputs": {
            "dataset": {"path": str(OUTPUT), "sha256": sha256(OUTPUT)},
            "audit": {"path": str(AUDIT_OUTPUT), "sha256": sha256(AUDIT_OUTPUT)},
        },
    }
    if not all([
        len(ready) > 0, ready["split"].nunique() == 3,
        report["protein_disjoint"], report["family_component_disjoint"],
        report["all_ready_proteins_have_both_labels"],
    ]):
        report["status"] = "failed"
    atomic_json(VALIDATION_OUTPUT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("structure graph validation failed")


if __name__ == "__main__":
    main()
