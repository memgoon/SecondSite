#!/usr/bin/env python3
"""Checkpoint 9b: resolve and fetch only named candidate structures/sequences.

No coordinate directory is enumerated. Every lookup is an exact candidate PDB
filename, and network retrieval is restricted to the frozen checkpoint-9a IDs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gemmi
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
ASSETS = PACKAGE / "assets/checkpoint9_cif"
VALIDATION = PACKAGE / "validation"

CANDIDATES = DATA / "CHECKPOINT9_CANDIDATE_UNIVERSE.tsv.gz"
PREPARATION = VALIDATION / "CHECKPOINT9_PREPARATION.json"
MANIFEST = DATA / "CHECKPOINT9_COORDINATE_MANIFEST.tsv.gz"
FETCH_LOG = DATA / "CHECKPOINT9_COORDINATE_FETCH_LOG.tsv"
SEQUENCES = DATA / "CHECKPOINT9_UNIPROT_SEQUENCES.tsv.gz"
BUILD = VALIDATION / "CHECKPOINT9_ASSET_BUILD.json"

COORDINATE_ROOTS = (
    ("checkpoint9_fetched", ASSETS),
    ("checkpoint6_fetched", PACKAGE / "assets/checkpoint6_cif"),
    ("candidate_rebuild", ROOT / "analysis/structure_known_candidate_rebuild/assets/cif"),
    ("shared_legacy", Path("/shared_data/11.HS_allostery/BioLiP_CIFs")),
    ("base_structures", ROOT / "data/structures"),
    ("biolip_xai", ROOT / "analysis/structure_known_biolip_xai/assets/structures"),
)
PDB_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
HEADERS = {"User-Agent": "BioLiP-ranking-revision/2.0 (targeted candidate retrieval)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="perform targeted downloads")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-pdb", type=int, default=0)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_tsv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_coordinate(pdb_id: str):
    for source, root in COORDINATE_ROOTS:
        for extension in (".cif.gz", ".cif"):
            path = root / f"{pdb_id.lower()}{extension}"
            if path.is_file():
                return path, source
    return None, "missing"


def request_bytes(url: str, attempts: int = 3) -> bytes:
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except Exception as error:  # recorded after bounded retries
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def fetch_structure(pdb_id: str) -> dict:
    existing, source = resolve_coordinate(pdb_id)
    if existing is not None:
        return {
            "pdb_id": pdb_id,
            "fetch_status": "reused",
            "coordinate_source": source,
            "coordinate_path": str(existing),
            "size_bytes": existing.stat().st_size,
            "sha256": sha256(existing),
        }
    output = ASSETS / f"{pdb_id.lower()}.cif.gz"
    try:
        raw = request_bytes(PDB_URL.format(pdb=pdb_id.lower()))
        text = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        document = gemmi.cif.read_string(text.decode("utf-8", "replace"))
        structure = gemmi.make_structure_from_block(document.sole_block())
        if len(structure) < 1:
            raise ValueError("empty_structure")
        compressed = gzip.compress(text, mtime=0)
        atomic_bytes(output, compressed)
        return {
            "pdb_id": pdb_id,
            "fetch_status": "downloaded",
            "coordinate_source": "checkpoint9_fetched",
            "coordinate_path": str(output),
            "size_bytes": len(compressed),
            "sha256": sha256_bytes(compressed),
        }
    except Exception as error:
        return {
            "pdb_id": pdb_id,
            "fetch_status": f"error:{type(error).__name__}:{str(error)[:160]}",
            "coordinate_source": "missing",
            "coordinate_path": "",
            "size_bytes": 0,
            "sha256": "",
        }


def parse_fasta(raw: bytes) -> str:
    lines = raw.decode("utf-8", "replace").splitlines()
    return "".join(line.strip() for line in lines if line and not line.startswith(">") ).upper()


def fetch_sequence(accession: str) -> dict:
    try:
        raw = request_bytes(UNIPROT_URL.format(accession=accession))
        sequence = parse_fasta(raw)
        if not sequence or any(letter not in "ABCDEFGHIKLMNPQRSTVWXYZUO" for letter in sequence):
            raise ValueError("invalid_or_empty_sequence")
        return {
            "uniprot": accession,
            "sequence_status": "downloaded",
            "sequence": sequence,
            "sequence_length": len(sequence),
            "sequence_sha256": sha256_bytes(sequence.encode("ascii")),
        }
    except Exception as error:
        return {
            "uniprot": accession,
            "sequence_status": f"error:{type(error).__name__}:{str(error)[:160]}",
            "sequence": "",
            "sequence_length": 0,
            "sequence_sha256": "",
        }


def main() -> None:
    args = parse_args()
    started = time.time()
    preparation = json.loads(PREPARATION.read_text())
    if preparation.get("status") != "validated_candidate_universe_frozen":
        raise RuntimeError("checkpoint 9a preparation is not validated")
    candidates = pd.read_csv(
        CANDIDATES, sep="\t", usecols=["pdb_id", "uniprot", "observation_id"],
        dtype=str, keep_default_na=False,
    )
    if len(candidates) != 19_920:
        raise RuntimeError("candidate universe row count changed")
    pdb_counts = candidates.groupby("pdb_id").observation_id.size().to_dict()
    pdb_ids = sorted(pdb_counts)
    if args.max_pdb:
        pdb_ids = pdb_ids[: args.max_pdb]

    initial = []
    missing = []
    for pdb_id in pdb_ids:
        path, source = resolve_coordinate(pdb_id)
        if path is None:
            missing.append(pdb_id)
        else:
            initial.append({
                "pdb_id": pdb_id,
                "fetch_status": "reused",
                "coordinate_source": source,
                "coordinate_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": "pending",
            })
    print(
        f"candidate PDBs={len(pdb_ids):,}; locally resolved={len(initial):,}; "
        f"targeted missing={len(missing):,}", flush=True,
    )
    if missing and not args.run:
        print("dry-run: pass --run to fetch only the listed missing PDB IDs")
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_structure, pdb_id): pdb_id for pdb_id in pdb_ids}
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(f"coordinate resolution: {completed:,}/{len(futures):,}", flush=True)
    fetch_log = pd.DataFrame(results).sort_values("pdb_id", kind="mergesort")
    fetch_log["candidate_observations"] = fetch_log.pdb_id.map(pdb_counts).astype(int)
    atomic_tsv(fetch_log, FETCH_LOG)

    # Re-resolve exact paths after downloads so the manifest reflects usable assets.
    manifest_rows = []
    for pdb_id in pdb_ids:
        path, source = resolve_coordinate(pdb_id)
        manifest_rows.append({
            "pdb_id": pdb_id,
            "candidate_observations": int(pdb_counts[pdb_id]),
            "coordinate_source": source,
            "coordinate_path": str(path or ""),
            "size_bytes": path.stat().st_size if path else 0,
            "sha256": sha256(path) if path else "",
        })
    manifest = pd.DataFrame(manifest_rows)
    atomic_tsv(manifest, MANIFEST, compression="gzip")

    proteins = sorted(set(candidates.uniprot))
    sequence_rows = []
    with ThreadPoolExecutor(max_workers=min(args.workers, 12)) as executor:
        futures = {executor.submit(fetch_sequence, protein): protein for protein in proteins}
        for future in as_completed(futures):
            sequence_rows.append(future.result())
    sequences = pd.DataFrame(sequence_rows).sort_values("uniprot", kind="mergesort")
    atomic_tsv(sequences, SEQUENCES, compression="gzip")

    missing_coordinates = int(manifest.coordinate_path.eq("").sum())
    missing_sequences = int(sequences.sequence.eq("").sum())
    build = {
        "checkpoint": "9b",
        "status": "validated" if not missing_coordinates and not missing_sequences else "incomplete_assets",
        "candidate_pdbs": len(pdb_ids),
        "coordinate_assets_available": int(manifest.coordinate_path.ne("").sum()),
        "coordinate_assets_missing": missing_coordinates,
        "coordinate_source_counts": manifest.coordinate_source.value_counts().to_dict(),
        "candidate_proteins": len(proteins),
        "uniprot_sequences_available": int(sequences.sequence.ne("").sum()),
        "uniprot_sequences_missing": missing_sequences,
        "downloaded_coordinate_bytes": int(fetch_log.loc[
            fetch_log.fetch_status.eq("downloaded"), "size_bytes"
        ].sum()),
        "coordinate_manifest_sha256": sha256(MANIFEST),
        "sequence_table_sha256": sha256(SEQUENCES),
        "directory_enumeration_performed": False,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(build, BUILD)
    print(json.dumps(build, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
