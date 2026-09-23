#!/usr/bin/env python3
"""Compute checkpoint-7 ligand descriptors with the frozen RDKit environment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

MASTER = DATA / "BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz"
GEOMETRY = DATA / "CHECKPOINT6_EXACT_REFERENCE_GEOMETRY.tsv.gz"
SELECTION = MANIFESTS / "CHECKPOINT6_DISTANCE_SELECTION.json"
CHECKPOINT6_VALIDATION = VALIDATION / "CHECKPOINT6_VALIDATION.json"
OUTPUT = DATA / "CHECKPOINT7_LIGAND_DESCRIPTORS.tsv.gz"
REPORT = VALIDATION / "CHECKPOINT7_DESCRIPTOR_BUILD.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression="gzip")
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    validation = json.loads(CHECKPOINT6_VALIDATION.read_text())
    selection = json.loads(SELECTION.read_text())
    if validation.get("status") != "validated_candidates_awaiting_user_selection":
        raise RuntimeError("checkpoint 6 has not passed independent validation")
    if selection.get("decision_status") != "selected_by_user_before_checkpoint_7":
        raise RuntimeError("checkpoint-6 distance selection is not frozen")

    geometry = pd.read_csv(
        GEOMETRY,
        sep="\t",
        usecols=["observation_id", "reference_label", "geometry_status", "source_site_relation"],
        dtype=str,
        keep_default_na=False,
    )
    universe = geometry.loc[
        geometry.geometry_status.eq("ok")
        & (
            geometry.reference_label.eq("orthosteric")
            | geometry.source_site_relation.eq("distinct_from_orthosteric_site")
        ),
        ["observation_id"],
    ]
    if len(universe) != 5798 or universe.observation_id.duplicated().any():
        raise RuntimeError("checkpoint-7 descriptor universe changed")

    master = pd.read_csv(
        MASTER,
        sep="\t",
        usecols=[
            "observation_id", "ligand_ccd", "full_inchikey", "connectivity_key",
            "canonical_smiles",
        ],
        dtype=str,
        keep_default_na=False,
    )
    selected = universe.merge(master, on="observation_id", how="left", validate="one_to_one")
    if selected[["full_inchikey", "connectivity_key", "canonical_smiles"]].eq("").any().any():
        raise RuntimeError("a checkpoint-7 ligand lacks a chemical identity or SMILES")

    unique = (
        selected[["canonical_smiles"]]
        .drop_duplicates()
        .sort_values("canonical_smiles", kind="mergesort")
        .reset_index(drop=True)
    )
    rows = []
    for smiles in unique.canonical_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            rows.append({
                "canonical_smiles": smiles,
                "descriptor_status": "rdkit_parse_failed",
                "molecular_weight": np.nan,
                "clogp": np.nan,
                "aromatic_ring_count": np.nan,
                "heavy_atom_count": np.nan,
            })
            continue
        rows.append({
            "canonical_smiles": smiles,
            "descriptor_status": "ok",
            "molecular_weight": float(Descriptors.MolWt(molecule)),
            "clogp": float(Descriptors.MolLogP(molecule)),
            "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
            "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        })
    descriptors = pd.DataFrame(rows)
    descriptors.insert(
        0,
        "descriptor_id",
        descriptors.canonical_smiles.map(
            lambda value: "BLDESC_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
        ),
    )
    if descriptors.descriptor_id.duplicated().any():
        raise RuntimeError("descriptor IDs are not unique")
    if not descriptors.descriptor_status.eq("ok").all():
        raise RuntimeError("not every checkpoint-7 ligand parsed with RDKit")

    merged = selected.merge(descriptors, on="canonical_smiles", validate="many_to_one")
    numeric = merged[["molecular_weight", "clogp", "aromatic_ring_count", "heavy_atom_count"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("descriptor output contains missing or non-finite values")

    atomic_tsv(descriptors, OUTPUT)
    report = {
        "checkpoint": 7,
        "stage": "ligand_descriptor_build",
        "status": "validated",
        "rdkit_version": rdBase.rdkitVersion,
        "reference_observations": len(selected),
        "unique_canonical_smiles": len(descriptors),
        "unique_full_inchikey": int(selected.full_inchikey.nunique()),
        "unique_connectivity_key": int(selected.connectivity_key.nunique()),
        "rdkit_parse_failures": 0,
        "selected_distance": selection["selected_distance"],
        "inputs": {
            "master_sha256": sha256(MASTER),
            "geometry_sha256": sha256(GEOMETRY),
            "checkpoint6_validation_sha256": sha256(CHECKPOINT6_VALIDATION),
            "distance_selection_sha256": sha256(SELECTION),
        },
        "output": {
            "path": str(OUTPUT),
            "size_bytes": OUTPUT.stat().st_size,
            "sha256": sha256(OUTPUT),
        },
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
