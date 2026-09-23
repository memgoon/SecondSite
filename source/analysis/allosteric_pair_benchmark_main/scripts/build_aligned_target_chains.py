#!/usr/bin/env python3
"""Build target-chain sequences and fpocket masks in one exact coordinate system.

The legacy pocket_masks.json stores raw structure residue numbers without a
chain identifier.  Those values must not be used as row indices into ESM3
tensors.  This script reads only the fpocket/PDB paths explicitly named for the
448 benchmark proteins, selects the structure chain aligned to each UniProt
target, and writes a zero-based pocket mask for that exact chain sequence.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from Bio import Align


AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/disk9/13.Heesu_Allostery"))
    parser.add_argument("--shared-root", type=Path, default=Path("/shared_data/11.HS_allostery"))
    parser.add_argument("--minimum-identity", type=float, default=0.70)
    parser.add_argument("--minimum-chain-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-aligned-residues", type=int, default=30)
    parser.add_argument("--minimum-pocket-residues", type=int, default=5)
    parser.add_argument("--max-pockets", type=int, default=5)
    parser.add_argument("--api-delay", type=float, default=0.05)
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_text(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def normalize_sequence(value):
    return "".join(character for character in str(value).upper() if character.isalpha())


def fetch_uniprot_sequence(uid, session):
    response = session.get(
        "https://rest.uniprot.org/uniprotkb/{}.fasta".format(uid), timeout=30
    )
    if response.status_code != 200:
        return "", "HTTP {}".format(response.status_code)
    lines = [line.strip() for line in response.text.splitlines() if line and not line.startswith(">")]
    sequence = normalize_sequence("".join(lines))
    return (sequence, "") if sequence else ("", "empty FASTA response")


def atom_fields(line):
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    parts = line.split()
    if len(parts) < 8:
        return None
    residue_name = parts[5].upper()
    if residue_name not in AA3:
        return None
    try:
        label_sequence_id = int(parts[7])
    except ValueError:
        return None
    return str(parts[6]), label_sequence_id, AA3[residue_name]


def parse_structure_chains(path):
    residues = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = atom_fields(line)
            if fields is None:
                continue
            chain, label_sequence_id, amino_acid = fields
            residues.setdefault(chain, {})
            previous = residues[chain].get(label_sequence_id)
            if previous is not None and previous != amino_acid:
                raise ValueError("conflicting residue identity {}:{}".format(chain, label_sequence_id))
            residues[chain][label_sequence_id] = amino_acid
    chains = {}
    for chain, mapping in residues.items():
        ordered = sorted(mapping)
        chains[chain] = {
            "label_sequence_ids": ordered,
            "sequence": "".join(mapping[value] for value in ordered),
        }
    return chains


def pocket_score(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Drug Score" in line or "Druggability Score" in line:
                try:
                    return float(line.rsplit(":", 1)[1].strip())
                except (IndexError, ValueError):
                    return None
            if line.startswith("_atom_site.group_PDB"):
                break
    return None


def parse_pocket_residues(directory, max_pockets):
    records = []
    files = sorted(glob.glob(str(directory / "pockets/pocket*_atm.cif")))
    for path in files:
        score = pocket_score(path)
        if score is not None and score < 0.0001:
            continue
        match = re.search(r"pocket(\d+)_atm\.cif$", str(path))
        pocket_id = int(match.group(1)) if match else -1
        residues = set()
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = atom_fields(line)
                if fields is not None:
                    residues.add((fields[0], fields[1]))
        records.append({"path": path, "pocket_id": pocket_id, "score": score, "residues": residues})
    # ``None`` is ordered below every valid fpocket score.  Ties are resolved
    # by pocket identifier, so the selected union is deterministic.
    ranked = sorted(
        records,
        key=lambda value: (value["score"] is not None, value["score"] if value["score"] is not None else -1.0, -value["pocket_id"]),
        reverse=True,
    )
    selected = ranked[:max(int(max_pockets), 1)]
    residues = set().union(*(value["residues"] for value in selected)) if selected else set()
    return residues, len(files), len(records), selected


def alignment_summary(canonical, chain_sequence):
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(canonical, chain_sequence)
    if len(alignments) == 0:
        return {"matches": 0, "aligned": 0, "identity": 0.0, "chain_coverage": 0.0}
    alignment = alignments[0]
    matches = 0
    aligned = 0
    for (left_start, left_end), (right_start, right_end) in zip(*alignment.aligned):
        length = min(left_end - left_start, right_end - right_start)
        for offset in range(length):
            aligned += 1
            matches += int(canonical[left_start + offset] == chain_sequence[right_start + offset])
    return {
        "matches": int(matches),
        "aligned": int(aligned),
        "identity": float(matches) / max(int(aligned), 1),
        "chain_coverage": float(aligned) / max(len(chain_sequence), 1),
    }


def choose_chain(canonical, chains, pocket_residues):
    candidates = []
    for chain, info in sorted(chains.items()):
        pocket_ids = {residue_id for value_chain, residue_id in pocket_residues if value_chain == chain}
        pocket_index = [
            position for position, residue_id in enumerate(info["label_sequence_ids"])
            if residue_id in pocket_ids
        ]
        alignment = alignment_summary(canonical, info["sequence"])
        candidate = dict(info)
        candidate.update(alignment)
        candidate.update({
            "chain": chain,
            "pocket_index": pocket_index,
            "n_pocket_residues": len(pocket_index),
        })
        # Matching residues identify the target chain; pocket count resolves
        # equivalent biological copies without favoring an unrelated long chain.
        candidate["score"] = (
            int(candidate["identity"] >= 0.70),
            candidate["matches"],
            candidate["n_pocket_residues"],
            candidate["chain_coverage"],
            candidate["identity"],
        )
        candidates.append(candidate)
    return max(candidates, key=lambda value: value["score"]) if candidates else None


def write_fasta(path, uid, pdb_id, chain, sequence):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        ">{}\n{}\n".format(uid, sequence), encoding="ascii"
    )
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    package = args.project_root / "analysis/allosteric_pair_benchmark_main"
    # Always rebuild from the immutable 448-protein source, never from the
    # filtered release cohort produced by prepare_cpu_release.py.
    cohort_path = (
        args.project_root
        / "analysis/new_pairing_pfam_only_pilot/data/PFAM_ONLY_POCKET_READY.tsv.gz"
    )
    fpocket_map_path = args.shared_root / "14.Organized_input/fpocket_Map.tsv"
    local_sequence_path = args.shared_root / "14.Organized_input/uniprot_protein_cache.json"
    output_cif_root = args.shared_root / "BioLiP_CIFs/fpocket"
    output_table = package / "data/TARGET_CHAIN_SEQUENCES.tsv"
    output_masks = package / "data/POCKET_INDICES.json"
    output_cache = package / "data/UNIPROT_SEQUENCE_CACHE.json"
    output_audit = package / "validation/POCKET_ALIGNMENT_AUDIT.tsv"
    output_validation = package / "validation/POCKET_ALIGNMENT_VALIDATION.json"
    # Versioned output prevents an earlier, less strict run from leaving extra
    # FASTA files in the directory consumed by the GPU embedder.
    fasta_root = package / "data/target_chain_fasta_v2"
    for path in [cohort_path, fpocket_map_path, local_sequence_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = pd.read_csv(cohort_path, sep="\t", low_memory=False)
    proteins = sorted(frame["uniprot"].astype(str).unique())
    fpocket_map = pd.read_csv(fpocket_map_path, sep="\t", dtype=str).fillna("")
    fpocket_map["uid"] = fpocket_map["UniProt_ID"].str.split("-").str[0]
    fpocket_map = fpocket_map[fpocket_map["uid"].isin(proteins)].copy()
    if fpocket_map["uid"].duplicated().any():
        raise RuntimeError("fpocket map is not unique for benchmark proteins")
    fpocket_map = fpocket_map.set_index("uid")
    local_sequences = json.loads(local_sequence_path.read_text(encoding="utf-8"))
    frozen_sequences = {}
    if output_cache.is_file():
        frozen_sequences = json.loads(output_cache.read_text(encoding="utf-8"))

    session = requests.Session()
    audit = []
    selected_rows = []
    aligned_masks = {}
    for position, uid in enumerate(proteins, 1):
        status = "ready"
        reason = ""
        sequence_source = ""
        canonical = normalize_sequence(frozen_sequences.get(uid, {}).get("sequence", ""))
        if canonical:
            sequence_source = frozen_sequences[uid].get("source", "frozen_cache")
        if not canonical:
            canonical = normalize_sequence(local_sequences.get(uid, {}).get("seq", ""))
            if canonical:
                sequence_source = "local_uniprot_cache"
            else:
                canonical, error = fetch_uniprot_sequence(uid, session)
                sequence_source = "uniprot_rest" if canonical else ""
                if not canonical:
                    status, reason = "excluded", "canonical_sequence_missing:{}".format(error)
                time.sleep(max(args.api_delay, 0.0))
        if canonical:
            frozen_sequences[uid] = {
                "sequence": canonical,
                "sequence_sha256": sha256_text(canonical),
                "source": sequence_source,
            }

        mapped = fpocket_map.loc[uid] if uid in fpocket_map.index else None
        pdb_id = str(mapped["Mapped_PDB"]).strip().lower()[:4] if mapped is not None else ""
        directory = output_cif_root / (pdb_id + "_out") if pdb_id else Path("/")
        coordinate_path = directory / (pdb_id + "_out.cif") if pdb_id else Path("/")
        selected = None
        n_pocket_files = 0
        n_retained_pocket_files = 0
        n_chains = 0
        if status == "ready" and mapped is None:
            status, reason = "excluded", "fpocket_mapping_missing"
        elif status == "ready" and not coordinate_path.is_file():
            status, reason = "excluded", "fpocket_coordinate_file_missing"
        elif status == "ready":
            try:
                chains = parse_structure_chains(coordinate_path)
                pockets, n_pocket_files, n_retained_pocket_files, selected_pockets = parse_pocket_residues(
                    directory, args.max_pockets
                )
                n_chains = len(chains)
                selected = choose_chain(canonical, chains, pockets)
                if selected is None:
                    status, reason = "excluded", "no_protein_chain"
                elif selected["identity"] < args.minimum_identity:
                    status, reason = "excluded", "target_chain_alignment_identity_below_threshold"
                elif selected["aligned"] < args.minimum_aligned_residues:
                    status, reason = "excluded", "target_chain_alignment_too_short"
                elif selected["chain_coverage"] < args.minimum_chain_coverage:
                    status, reason = "excluded", "target_chain_alignment_coverage_below_threshold"
                elif selected["n_pocket_residues"] < args.minimum_pocket_residues:
                    status, reason = "excluded", "too_few_aligned_pocket_residues"
            except Exception as error:
                status, reason = "excluded", "{}:{}".format(type(error).__name__, error)

        record = {
            "uniprot": uid,
            "status": status,
            "reason": reason,
            "sequence_source": sequence_source,
            "canonical_length": len(canonical),
            "mapped_pdb": pdb_id,
            "fpocket_coordinate_file": str(coordinate_path) if pdb_id else "",
            "n_structure_chains": n_chains,
            "n_pocket_files": n_pocket_files,
            "n_retained_pocket_files": n_retained_pocket_files,
            "max_selected_pockets": int(args.max_pockets),
        }
        if selected is not None:
            record.update({
                "selected_chain": selected["chain"],
                "target_chain_length": len(selected["sequence"]),
                "alignment_matches": selected["matches"],
                "alignment_residues": selected["aligned"],
                "alignment_identity": selected["identity"],
                "target_chain_coverage": selected["chain_coverage"],
                "pocket_residues": selected["n_pocket_residues"],
                "pocket_fraction": selected["n_pocket_residues"] / max(len(selected["sequence"]), 1),
                "selected_fpocket_ids": ";".join(str(value["pocket_id"]) for value in selected_pockets),
                "selected_fpocket_scores": ";".join(
                    "" if value["score"] is None else "{:.6g}".format(value["score"])
                    for value in selected_pockets
                ),
            })
        audit.append(record)
        if status == "ready":
            sequence = selected["sequence"]
            indices = [int(value) for value in selected["pocket_index"]]
            fasta_path = fasta_root / (uid + ".fasta")
            write_fasta(fasta_path, uid, pdb_id, selected["chain"], sequence)
            aligned_masks[uid] = indices
            selected_rows.append({
                "uniprot": uid,
                "mapped_pdb": pdb_id,
                "selected_chain": selected["chain"],
                "target_chain_sequence": sequence,
                "target_chain_length": len(sequence),
                "sequence_sha256": sha256_text(sequence),
                "pocket_residues": len(indices),
                "pocket_fraction": len(indices) / max(len(sequence), 1),
                "fasta_path": str(fasta_path),
                "gpu_embedding_path": str(
                    Path("/disk1/11.HS_allostery")
                    / "analysis/allosteric_pair_benchmark_main/gpu_cache/target_chain_embeddings_v2"
                    / (uid + ".pt")
                ),
            })
        if position % 25 == 0 or position == len(proteins):
            print("aligned target chains {}/{} ready={}".format(position, len(proteins), len(selected_rows)), flush=True)

    audit_frame = pd.DataFrame(audit)
    selected_frame = pd.DataFrame(selected_rows).sort_values("uniprot").reset_index(drop=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    selected_frame.to_csv(output_table, sep="\t", index=False)
    audit_frame.to_csv(output_audit, sep="\t", index=False)
    atomic_json(output_masks, aligned_masks)
    atomic_json(output_cache, frozen_sequences)

    ready = audit_frame[audit_frame["status"].eq("ready")]
    report = {
        "status": "validated",
        "coordinate_contract": (
            "Each ESM3 tensor is generated from the selected target-chain sequence; each pocket mask is a "
            "zero-based index into that exact sequence. Raw PDB/mmCIF residue numbers are never tensor indices."
        ),
        "pocket_definition": (
            "The union of the {} highest fpocket druggability-score pockets on the selected target chain."
        ).format(args.max_pockets),
        "alignment_thresholds": {
            "minimum_identity": float(args.minimum_identity),
            "minimum_chain_coverage": float(args.minimum_chain_coverage),
            "minimum_aligned_residues": int(args.minimum_aligned_residues),
            "minimum_pocket_residues": int(args.minimum_pocket_residues),
        },
        "source_proteins": len(proteins),
        "ready_proteins": int(len(ready)),
        "excluded_proteins": int(len(proteins) - len(ready)),
        "exclusion_reasons": audit_frame.loc[
            audit_frame["status"].ne("ready"), "reason"
        ].value_counts().astype(int).to_dict(),
        "minimum_alignment_identity": float(ready["alignment_identity"].min()) if len(ready) else None,
        "median_alignment_identity": float(ready["alignment_identity"].median()) if len(ready) else None,
        "minimum_target_chain_coverage": float(ready["target_chain_coverage"].min()) if len(ready) else None,
        "median_target_chain_coverage": float(ready["target_chain_coverage"].median()) if len(ready) else None,
        "minimum_pocket_residues": int(ready["pocket_residues"].min()) if len(ready) else None,
        "median_pocket_fraction": float(ready["pocket_fraction"].median()) if len(ready) else None,
        "maximum_pocket_fraction": float(ready["pocket_fraction"].max()) if len(ready) else None,
        "maximum_target_chain_length": int(ready["target_chain_length"].max()) if len(ready) else None,
        "outputs": {
            "target_chain_table": str(output_table),
            "aligned_pocket_indices": str(output_masks),
            "alignment_audit": str(output_audit),
            "fasta_directory": str(fasta_root),
        },
    }
    # Missing coordinate files are reported explicitly.  The common universe
    # is frozen downstream only if enough target chains survive to preserve all
    # Pfam folds and a meaningful multi-family benchmark.
    if report["ready_proteins"] < 430:
        report["status"] = "failed"
    atomic_json(output_validation, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "validated":
        raise SystemExit("target-chain alignment contract failed")


if __name__ == "__main__":
    main()
