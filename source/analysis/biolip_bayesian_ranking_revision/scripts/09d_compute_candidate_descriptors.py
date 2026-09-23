#!/usr/bin/env python3
"""Checkpoint 9d: compute the frozen ligand descriptors for BioLiP candidates."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
VALIDATION = PACKAGE / "validation"

CANDIDATES = DATA / "CHECKPOINT9_CANDIDATE_UNIVERSE.tsv.gz"
PREPARATION = VALIDATION / "CHECKPOINT9_PREPARATION.json"
OUTPUT = DATA / "CHECKPOINT9_CANDIDATE_LIGAND_DESCRIPTORS.tsv.gz"
BUILD = VALIDATION / "CHECKPOINT9_DESCRIPTOR_BUILD.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(smiles: str) -> str:
    return "BLDESC_" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:20]


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression="gzip")
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    started = time.time()
    preparation = json.loads(PREPARATION.read_text())
    if preparation.get("status") != "validated_candidate_universe_frozen":
        raise RuntimeError("checkpoint 9a is not validated")
    candidates = pd.read_csv(
        CANDIDATES, sep="\t",
        usecols=["canonical_smiles", "full_inchikey", "connectivity_key"],
        dtype=str, keep_default_na=False,
    )
    smiles_values = sorted(set(candidates.canonical_smiles) - {""})
    rows = []
    for smiles in smiles_values:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            rows.append({
                "canonical_smiles": smiles,
                "descriptor_id": stable_id(smiles),
                "descriptor_status": "rdkit_parse_failed",
            })
            continue
        rows.append({
            "canonical_smiles": smiles,
            "descriptor_id": stable_id(smiles),
            "descriptor_status": "ok",
            "molecular_weight": float(Descriptors.MolWt(molecule)),
            "clogp": float(Crippen.MolLogP(molecule)),
            "aromatic_ring_count": int(Lipinski.NumAromaticRings(molecule)),
            "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        })
    descriptors = pd.DataFrame(rows).sort_values("canonical_smiles", kind="mergesort")
    atomic_tsv(descriptors, OUTPUT)
    failed = int(descriptors.descriptor_status.ne("ok").sum())
    covered_rows = int(candidates.canonical_smiles.isin(
        set(descriptors.loc[descriptors.descriptor_status.eq("ok"), "canonical_smiles"])
    ).sum())
    build = {
        "checkpoint": "9d",
        "status": (
            "validated_complete" if failed == 0 and covered_rows == len(candidates)
            else "validated_with_explicit_parse_failures"
        ),
        "candidate_rows": len(candidates),
        "unique_canonical_smiles": len(smiles_values),
        "descriptor_rows": len(descriptors),
        "descriptor_ok": int(descriptors.descriptor_status.eq("ok").sum()),
        "descriptor_failed": failed,
        "candidate_rows_with_descriptors": covered_rows,
        "descriptor_definitions": {
            "molecular_weight": "RDKit Descriptors.MolWt",
            "clogp": "RDKit Crippen.MolLogP",
            "aromatic_ring_count": "RDKit Lipinski.NumAromaticRings",
            "heavy_atom_count": "RDKit molecule heavy atom count",
        },
        "candidate_universe_sha256": sha256(CANDIDATES),
        "descriptor_table_sha256": sha256(OUTPUT),
        "rdkit_version": getattr(Chem, "__version__", "recorded_by_environment"),
        "parse_failure_policy": (
            "retain candidates for raw-distance ranking; chemistry-containing rankings are NA"
        ),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(build, BUILD)
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
