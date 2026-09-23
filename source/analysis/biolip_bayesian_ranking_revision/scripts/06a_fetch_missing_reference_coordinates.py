#!/usr/bin/env python3
"""Fetch only checkpoint-6 exact-reference coordinates known to be missing."""

from __future__ import annotations

import argparse
import gzip
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gemmi
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
MANIFEST = PACKAGE / "data/CHECKPOINT6_COORDINATE_MANIFEST.tsv.gz"
OUTPUT = PACKAGE / "assets/checkpoint6_cif"
LOG = PACKAGE / "data/CHECKPOINT6_COORDINATE_FETCH_LOG.tsv"
URL = "https://files.rcsb.org/download/{pdb}.cif.gz"
HEADERS = {"User-Agent": "BioLiP-ranking-revision/1.0 (targeted exact-reference fetch)"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pdb", type=int, default=0)
    return parser.parse_args()


def fetch(pdb_id: str) -> dict:
    output = OUTPUT / f"{pdb_id.lower()}.cif.gz"
    if output.is_file():
        return {"pdb_id": pdb_id, "status": "reused", "size_bytes": output.stat().st_size}
    try:
        request = urllib.request.Request(URL.format(pdb=pdb_id.lower()), headers=HEADERS)
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
        text = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        document = gemmi.cif.read_string(text.decode("utf-8", "replace"))
        structure = gemmi.make_structure_from_block(document.sole_block())
        if len(structure) < 1:
            raise ValueError("empty_structure")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp{os.getpid()}")
        temporary.write_bytes(gzip.compress(text, mtime=0))
        os.replace(temporary, output)
        return {"pdb_id": pdb_id, "status": "downloaded", "size_bytes": output.stat().st_size}
    except Exception as error:
        return {"pdb_id": pdb_id, "status": f"error:{type(error).__name__}", "size_bytes": 0}


def main() -> None:
    args = parse_args()
    coordinate_manifest = pd.read_csv(MANIFEST, sep="\t", dtype=str, keep_default_na=False)
    missing = sorted(set(coordinate_manifest.loc[
        coordinate_manifest["coordinate_source"].eq("missing"), "pdb_id"
    ]))
    if args.max_pdb:
        missing = missing[:args.max_pdb]
    print(f"targeted missing reference PDBs: {len(missing):,}", flush=True)
    if not args.run:
        print("dry-run: pass --run to download; no directory enumeration was performed")
        return
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, pdb_id): pdb_id for pdb_id in missing}
        for completed, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                print(f"checkpoint6 fetch: {completed:,}/{len(futures):,}", flush=True)
    frame = pd.DataFrame(results).sort_values("pdb_id", kind="mergesort")
    temporary = LOG.with_name(LOG.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, LOG)
    print(frame.status.value_counts().to_string())
    print(f"downloaded_bytes={int(frame.size_bytes.sum()):,}; wall_seconds={time.time()-started:.1f}")


if __name__ == "__main__":
    main()
