#!/usr/bin/env python3
"""Checkpoint 9a: freeze the BioLiP observations eligible for ranking.

Only observations on proteins with at least one exact, manually audited
orthosteric-site observation can receive the selected structural distance.
Known broad orthosteric pairs and exact allosteric protein-ligand pairs are
removed before any score is calculated.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
VALIDATION = PACKAGE / "validation"

MASTER = DATA / "BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz"
ELIGIBILITY = DATA / "BIOLIP_SITE_MAPPING_ELIGIBILITY.tsv.gz"
REFERENCE = DATA / "BIOLIP_EXACT_SITE_REFERENCE_OBSERVATIONS.tsv.gz"
CHECKPOINT8 = VALIDATION / "CHECKPOINT8_VALIDATION.json"

CANDIDATES = DATA / "CHECKPOINT9_CANDIDATE_UNIVERSE.tsv.gz"
SITE_DEFINITIONS = DATA / "CHECKPOINT9_ORTHOSTERIC_SITE_DEFINITIONS.tsv.gz"
COUNTS = DATA / "CHECKPOINT9_CANDIDATE_FILTER_COUNTS.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT9_PREPARATION_INPUT_HASHES.tsv"
BUILD = VALIDATION / "CHECKPOINT9_PREPARATION.json"

EXPECTED = {
    "master_rows": 989_058,
    "mapping_eligible_rows": 927_697,
    "reference_rows": 6_117,
    "orthosteric_reference_rows": 4_090,
    "allosteric_reference_rows": 2_027,
    "orthosteric_reference_proteins": 164,
    "orthosteric_unique_site_definitions": 1_960,
    "candidate_rows": 19_920,
    "candidate_pairs": 3_109,
    "candidate_proteins": 153,
    "candidate_pdbs": 4_985,
    "candidate_exact_site_signatures": 9_804,
}

MASTER_COLUMNS = [
    "source_ordinal", "observation_id", "natural_observation_key", "pdb_id",
    "receptor_chain", "resolution", "binding_site_code", "ligand_ccd",
    "ligand_chain", "ligand_serial", "ligand_auth_seq_id", "uniprot_raw",
    "uniprot_resolved", "uniprot_resolution_status", "pubmed_id", "ec_number",
    "go_terms", "affinity_manual", "affinity_moad", "affinity_pdbbind_cn",
    "affinity_bindingdb", "full_inchikey", "connectivity_key", "canonical_smiles",
    "chemical_mapping_status", "active_orthosteric_pair_reference",
    "quarantined_orthosteric_pair_reference", "binding_residues_auth_raw",
    "binding_residues_seq_raw", "binding_uniprot_positions", "binding_residue_count",
    "site_mapping_status", "mapping_agreement_status", "receptor_sequence_id",
    "receptor_sequence_length",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_tsv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, compression=compression)
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def normalized_positions(value: object) -> str:
    positions = sorted({int(token) for token in str(value or "").split(";") if token})
    return ";".join(str(position) for position in positions)


def stable_id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def assert_count(name: str, observed: int) -> None:
    expected = EXPECTED[name]
    if int(observed) != int(expected):
        raise RuntimeError(f"{name} changed: observed={observed:,}, expected={expected:,}")


def main() -> None:
    started = time.time()
    checkpoint8 = json.loads(CHECKPOINT8.read_text())
    if checkpoint8.get("status") != "validated_heldout_ranking_matrix_awaiting_user_review":
        raise RuntimeError("checkpoint 8 validation is not in the expected frozen state")

    master = pd.read_csv(
        MASTER, sep="\t", usecols=MASTER_COLUMNS, dtype=str, keep_default_na=False,
    )
    eligibility = pd.read_csv(
        ELIGIBILITY, sep="\t",
        usecols=["observation_id", "final_site_mapping_usable", "final_eligibility_reason"],
        dtype=str, keep_default_na=False,
    )
    reference = pd.read_csv(REFERENCE, sep="\t", dtype=str, keep_default_na=False)

    assert_count("master_rows", len(master))
    if eligibility.observation_id.duplicated().any():
        raise RuntimeError("checkpoint-4 eligibility contains duplicate observation IDs")
    assert_count("mapping_eligible_rows", int(truthy(eligibility.final_site_mapping_usable).sum()))
    assert_count("reference_rows", len(reference))
    assert_count("orthosteric_reference_rows", int(reference.reference_label.eq("orthosteric").sum()))
    assert_count("allosteric_reference_rows", int(reference.reference_label.eq("allosteric").sum()))

    orthosteric = reference.loc[reference.reference_label.eq("orthosteric")].copy()
    orthosteric["orthosteric_site_uniprot_positions"] = (
        orthosteric.binding_uniprot_positions.map(normalized_positions)
    )
    if orthosteric.orthosteric_site_uniprot_positions.eq("").any():
        raise RuntimeError("an exact orthosteric observation has no mapped site residues")
    assert_count("orthosteric_reference_proteins", orthosteric.uniprot.nunique())

    site_rows = []
    for (uniprot, positions), group in orthosteric.groupby(
        ["uniprot", "orthosteric_site_uniprot_positions"], sort=True,
    ):
        site_rows.append({
            "orthosteric_site_definition_id": stable_id("BLOS_", f"{uniprot}|{positions}"),
            "uniprot": uniprot,
            "orthosteric_site_uniprot_positions": positions,
            "orthosteric_site_residue_count": len(positions.split(";")),
            "source_observations": len(group),
            "source_pdbs": group.pdb_id.nunique(),
            "source_ligand_connectivities": group.connectivity_key.nunique(),
            "source_ligand_ccds": ";".join(sorted(set(group.ligand_ccd))),
            "representative_observation_id": sorted(group.observation_id)[0],
            "reference_source_ids": ";".join(sorted(set(filter(None, group.reference_source_ids)))),
        })
    site_definitions = pd.DataFrame(site_rows).sort_values(
        ["uniprot", "orthosteric_site_definition_id"], kind="mergesort",
    )
    assert_count("orthosteric_unique_site_definitions", len(site_definitions))
    if site_definitions.orthosteric_site_definition_id.duplicated().any():
        raise RuntimeError("orthosteric-site definition IDs are not unique")

    merged = master.merge(eligibility, on="observation_id", validate="one_to_one")
    eligible = merged.loc[truthy(merged.final_site_mapping_usable)].copy()
    chemically_mapped = eligible.loc[
        eligible.full_inchikey.ne("") & eligible.canonical_smiles.ne("")
    ].copy()
    orthosteric_proteins = set(site_definitions.uniprot)
    site_supported = chemically_mapped.loc[
        chemically_mapped.uniprot_resolved.isin(orthosteric_proteins)
    ].copy()
    nonorthosteric = site_supported.loc[
        ~truthy(site_supported.active_orthosteric_pair_reference)
    ].copy()
    exact_allosteric_pairs = set(zip(
        reference.loc[reference.reference_label.eq("allosteric"), "uniprot"],
        reference.loc[reference.reference_label.eq("allosteric"), "full_inchikey"],
    ))
    known_allosteric_pair = np.fromiter(
        (
            (uniprot, ligand) in exact_allosteric_pairs
            for uniprot, ligand in zip(nonorthosteric.uniprot_resolved, nonorthosteric.full_inchikey)
        ),
        dtype=bool,
        count=len(nonorthosteric),
    )
    candidates = nonorthosteric.loc[~known_allosteric_pair].copy()
    candidates["uniprot"] = candidates.uniprot_resolved
    candidates["binding_uniprot_positions"] = (
        candidates.binding_uniprot_positions.map(normalized_positions)
    )
    candidates["candidate_pair_id"] = [
        stable_id("BLPAIR_", f"{protein}|{ligand}")
        for protein, ligand in zip(candidates.uniprot, candidates.full_inchikey)
    ]
    candidates["exact_site_ligand_signature_id"] = [
        stable_id("BLSITE_", f"{protein}|{ligand}|{positions}")
        for protein, ligand, positions in zip(
            candidates.uniprot, candidates.full_inchikey,
            candidates.binding_uniprot_positions,
        )
    ]
    candidates["candidate_selection_rule"] = (
        "mapping_and_chemistry_available; exact_orthosteric_site_available_for_protein; "
        "not_broad_known_orthosteric_pair; not_exact_known_allosteric_pair"
    )
    candidates = candidates.sort_values(
        ["uniprot", "connectivity_key", "pdb_id", "receptor_chain", "source_ordinal"],
        kind="mergesort",
    )

    assert_count("candidate_rows", len(candidates))
    assert_count("candidate_pairs", candidates[["uniprot", "full_inchikey"]].drop_duplicates().shape[0])
    assert_count("candidate_proteins", candidates.uniprot.nunique())
    assert_count("candidate_pdbs", candidates.pdb_id.nunique())
    assert_count("candidate_exact_site_signatures", candidates.exact_site_ligand_signature_id.nunique())
    if candidates.observation_id.duplicated().any():
        raise RuntimeError("candidate observation IDs are duplicated")
    if truthy(candidates.active_orthosteric_pair_reference).any():
        raise RuntimeError("a broad known orthosteric pair survived the candidate filter")
    if any(pair in exact_allosteric_pairs for pair in zip(candidates.uniprot, candidates.full_inchikey)):
        raise RuntimeError("an exact known allosteric pair survived the candidate filter")

    output_columns = [
        "source_ordinal", "observation_id", "natural_observation_key", "candidate_pair_id",
        "exact_site_ligand_signature_id", "pdb_id", "receptor_chain", "resolution",
        "binding_site_code", "ligand_ccd", "ligand_chain", "ligand_serial",
        "ligand_auth_seq_id", "uniprot", "uniprot_raw", "uniprot_resolution_status",
        "pubmed_id", "ec_number", "go_terms", "affinity_manual", "affinity_moad",
        "affinity_pdbbind_cn", "affinity_bindingdb", "full_inchikey", "connectivity_key",
        "canonical_smiles", "chemical_mapping_status", "binding_residues_auth_raw",
        "binding_residues_seq_raw", "binding_uniprot_positions", "binding_residue_count",
        "site_mapping_status", "mapping_agreement_status", "receptor_sequence_id",
        "receptor_sequence_length", "candidate_selection_rule",
    ]
    candidates = candidates[output_columns]

    count_rows = [
        {"filter_step": "raw_BioLiP_observations", "rows": len(master)},
        {"filter_step": "site_mapping_eligible", "rows": len(eligible)},
        {"filter_step": "site_and_chemical_mapping_available", "rows": len(chemically_mapped)},
        {"filter_step": "protein_has_exact_orthosteric_site", "rows": len(site_supported)},
        {"filter_step": "remove_broad_known_orthosteric_pairs", "rows": len(nonorthosteric)},
        {"filter_step": "remove_exact_known_allosteric_pairs", "rows": len(candidates)},
    ]
    counts = pd.DataFrame(count_rows)

    atomic_tsv(site_definitions, SITE_DEFINITIONS, compression="gzip")
    atomic_tsv(candidates, CANDIDATES, compression="gzip")
    atomic_tsv(counts, COUNTS)

    input_rows = []
    for asset_id, path in (
        ("observation_master", MASTER),
        ("site_mapping_eligibility", ELIGIBILITY),
        ("exact_site_reference", REFERENCE),
        ("checkpoint8_validation", CHECKPOINT8),
    ):
        input_rows.append({
            "asset_id": asset_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    atomic_tsv(pd.DataFrame(input_rows), INPUT_HASHES)

    build = {
        "checkpoint": "9a",
        "status": "validated_candidate_universe_frozen",
        "elapsed_seconds": time.time() - started,
        **EXPECTED,
        "candidate_universe_sha256": sha256(CANDIDATES),
        "orthosteric_site_definitions_sha256": sha256(SITE_DEFINITIONS),
        "distance_scope": (
            "distance to the closest successfully transferred exact orthosteric-site "
            "definition for the same UniProt protein"
        ),
        "directory_enumeration_performed": False,
        "probability_claim": False,
    }
    atomic_json(build, BUILD)
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
