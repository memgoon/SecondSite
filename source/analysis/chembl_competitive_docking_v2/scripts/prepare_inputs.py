#!/usr/bin/env python3
"""Prepare exact-site receptors, pH-aware ligands, and docking boxes.

The important contracts are deliberately fail-closed:

* site residues are selected by chain + residue name + residue number, never
  by residue name alone;
* polymer residues cannot be mistaken for identically named crystallographic
  ligands (the failure that contaminated the legacy TDO2 boxes);
* ligand protonation is generated at pH 7.4 before 3-D embedding and audited;
* indispensable metals/cofactors must survive receptor preparation; and
* de-novo fpocket cavities must be spatially distinct from every named site.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
import pandas as pd
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


@dataclass(frozen=True)
class ResidueSelector:
    chain: str
    residue_name: str
    residue_number: int
    insertion_code: str = ""

    @property
    def key(self) -> str:
        suffix = self.insertion_code if self.insertion_code else ""
        return f"{self.chain}:{self.residue_name}:{self.residue_number}{suffix}"

    def matches(self, chain_name: str, residue: gemmi.Residue) -> bool:
        return (
            chain_name == self.chain
            and residue.name.strip().upper() == self.residue_name
            and residue.seqid.num == self.residue_number
            and residue.seqid.icode.strip() == self.insertion_code
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(path)


def parse_chain_map(value: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in str(value).split(";"):
        if not item.strip():
            continue
        source, separator, output = item.strip().partition("=")
        if not separator or not source or len(output) != 1:
            raise ValueError(f"Invalid chain mapping {item!r}; expected source=single_output_character")
        if source in mapping or output in mapping.values():
            raise ValueError(f"Duplicate source/output chain in mapping {value!r}")
        mapping[source] = output
    if not mapping:
        raise ValueError("chain_map cannot be empty")
    return mapping


def parse_selectors(value: object) -> List[ResidueSelector]:
    if pd.isna(value) or not str(value).strip():
        return []
    selectors: List[ResidueSelector] = []
    for item in str(value).split(";"):
        fields = item.strip().split(":")
        if len(fields) != 3:
            raise ValueError(f"Invalid residue selector {item!r}; expected chain:RES:number")
        chain, residue_name, number_text = fields
        match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", number_text)
        if not match:
            raise ValueError(f"Invalid residue number in selector {item!r}")
        selectors.append(
            ResidueSelector(
                chain=chain,
                residue_name=residue_name.upper(),
                residue_number=int(match.group(1)),
                insertion_code=match.group(2),
            )
        )
    if len({selector.key for selector in selectors}) != len(selectors):
        raise ValueError(f"Duplicate residue selectors in {value!r}")
    return selectors


def heavy_coordinates(residue: gemmi.Residue) -> List[List[float]]:
    return [
        [atom.pos.x, atom.pos.y, atom.pos.z]
        for atom in residue
        if atom.element.name.upper() != "H"
    ]


def add_residue(structure_model: gemmi.Model, chain_name: str, residue: gemmi.Residue) -> None:
    output_chain = None
    for chain in structure_model:
        if chain.name == chain_name:
            output_chain = chain
            break
    if output_chain is None:
        output_chain = gemmi.Chain(chain_name)
        structure_model.add_chain(output_chain)
        output_chain = structure_model[-1]
    output_chain.add_residue(residue.clone())


def select_structure_and_sites(
    source_path: Path,
    selected_path: Path,
    site_directory: Path,
    chain_map: Dict[str, str],
    sites: pd.DataFrame,
    retain_selectors: Sequence[ResidueSelector],
) -> Tuple[Dict[str, np.ndarray], List[str], int, Dict[str, str]]:
    structure = gemmi.read_structure(str(source_path))
    structure.remove_alternative_conformations()
    if len(structure) != 1:
        raise RuntimeError(f"Expected one model in {source_path}, found {len(structure)}")
    model = structure[0]
    source_chains = {chain.name for chain in model}
    missing_chains = set(chain_map) - source_chains
    if missing_chains:
        raise RuntimeError(f"Missing requested source chains {sorted(missing_chains)} in {source_path}")

    site_selectors = {
        row.pocket_id: parse_selectors(row.selectors) for row in sites.itertuples(index=False)
    }
    all_site_selectors = [selector for values in site_selectors.values() for selector in values]
    # A cofactor can legitimately serve both as a site-definition residue and
    # as a retained receptor component (for example TDO2 HEM402).  Count the
    # physical residue once per unique exact selector, rather than once per
    # role in the contract.
    unique_selectors = {
        selector.key: selector for selector in all_site_selectors + list(retain_selectors)
    }
    selector_hits = {key: 0 for key in unique_selectors}
    site_coordinates: Dict[str, List[List[float]]] = {key: [] for key in site_selectors}
    site_flags: Dict[str, List[str]] = {key: [] for key in site_selectors}

    selected = gemmi.Structure()
    selected.name = f"{structure.name}_selected"
    selected_model = gemmi.Model("1")
    site_structures: Dict[str, gemmi.Structure] = {}
    site_models: Dict[str, gemmi.Model] = {}
    for pocket_id in site_selectors:
        current = gemmi.Structure()
        current.name = f"{structure.name}_{pocket_id}"
        current_model = gemmi.Model("1")
        site_structures[pocket_id] = current
        site_models[pocket_id] = current_model

    retained_labels: List[str] = []
    for source_chain in model:
        if source_chain.name not in chain_map:
            continue
        output_chain_name = chain_map[source_chain.name]
        output_chain = gemmi.Chain(output_chain_name)
        for residue in source_chain:
            for selector in unique_selectors.values():
                if selector.matches(source_chain.name, residue):
                    selector_hits[selector.key] += 1
            matching_sites: List[str] = []
            for pocket_id, selectors in site_selectors.items():
                for selector in selectors:
                    if selector.matches(source_chain.name, residue):
                        matching_sites.append(pocket_id)
                        site_coordinates[pocket_id].extend(heavy_coordinates(residue))
                        site_flags[pocket_id].append(str(residue.het_flag))
                        add_residue(site_models[pocket_id], output_chain_name, residue)
            retained = False
            for selector in retain_selectors:
                if selector.matches(source_chain.name, residue):
                    retained = True
                    retained_labels.append(selector.key)
            # Gemmi marks polymer residues with het_flag == "A".  Every other
            # residue is removed unless selected explicitly as an indispensable
            # cofactor/metal.  Being a named site does not itself retain a
            # crystallographic ligand in the docking receptor.
            if residue.het_flag == "A" or retained:
                output_chain.add_residue(residue.clone())
        if len(output_chain):
            selected_model.add_chain(output_chain)
    selected.add_model(selected_model)
    selected.write_pdb(str(selected_path))

    problems = {key: count for key, count in selector_hits.items() if count != 1}
    if problems:
        raise RuntimeError(
            f"Every exact selector must match exactly one residue in {source_path.name}; observed {problems}"
        )
    site_directory.mkdir(parents=True, exist_ok=True)
    site_arrays: Dict[str, np.ndarray] = {}
    site_flag_text: Dict[str, str] = {}
    for pocket_id, coordinates in site_coordinates.items():
        site_structures[pocket_id].add_model(site_models[pocket_id])
        site_path = site_directory / f"{pocket_id}.pdb"
        site_structures[pocket_id].write_pdb(str(site_path))
        if not coordinates:
            raise RuntimeError(f"No coordinates extracted for {pocket_id} in {source_path.name}")
        site_arrays[pocket_id] = np.asarray(coordinates, dtype=float)
        site_flag_text[pocket_id] = ";".join(sorted(set(site_flags[pocket_id])))
    return site_arrays, retained_labels, len(selected_model), site_flag_text


def load_legacy_receptor_preparer(workspace: Path):
    path = workspace / "analysis" / "dual_role_metabolite_docking_v1" / "scripts" / "prepare_inputs.py"
    if not path.exists():
        raise RuntimeError(f"Missing shared receptor-preparation implementation: {path}")
    spec = importlib.util.spec_from_file_location("legacy_docking_prepare_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_receptor, module.parse_fpocket_info, module.parse_pqr_coordinates


def make_box(
    coordinates: np.ndarray, padding: float, minimum: float, maximum: float
) -> Tuple[np.ndarray, np.ndarray]:
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    center = (lower + upper) / 2.0
    size = np.clip((upper - lower) + 2.0 * padding, minimum, maximum)
    return center, np.ceil(size / 0.375) * 0.375


def nearest_distance(a: np.ndarray, b: np.ndarray) -> float:
    best = math.inf
    for start in range(0, len(a), 256):
        distances = np.linalg.norm(a[start : start + 256, None, :] - b[None, :, :], axis=2)
        best = min(best, float(distances.min()))
    return best


def run_fpocket(fpocket_bin: Path, protein_pdb: Path, target_directory: Path) -> Path:
    expected = protein_pdb.with_name(protein_pdb.stem + "_out")
    if expected.exists():
        shutil.rmtree(expected)
    result = subprocess.run(
        [str(fpocket_bin), "-f", str(protein_pdb)],
        cwd=target_directory,
        text=True,
        capture_output=True,
    )
    (target_directory / "fpocket.log").write_text(
        "COMMAND\n{}\n\nSTDOUT\n{}\nSTDERR\n{}".format(
            " ".join([str(fpocket_bin), "-f", str(protein_pdb)]), result.stdout, result.stderr
        )
    )
    if result.returncode != 0 or not expected.exists() or not list(expected.glob("*_info.txt")):
        raise RuntimeError(f"fpocket failed for {protein_pdb}; see {target_directory / 'fpocket.log'}")
    return expected


def named_boxes(target_sites: pd.DataFrame, coordinates: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for site in target_sites.itertuples(index=False):
        values = coordinates[site.pocket_id]
        if len(values) != int(site.expected_heavy_atoms):
            raise RuntimeError(
                f"{site.target_id}/{site.pocket_id} expected {site.expected_heavy_atoms} heavy atoms, found {len(values)}"
            )
        center, size = make_box(
            values,
            padding=float(site.padding_A),
            minimum=float(site.minimum_size_A),
            maximum=float(site.maximum_size_A),
        )
        rows.append(
            {
                "target_id": site.target_id,
                "pocket_id": site.pocket_id,
                "pocket_class": site.pocket_class,
                "claim_role": site.claim_role,
                "fpocket_rank": 0,
                "fpocket_score": np.nan,
                "druggability_score": np.nan,
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "size_x": size[0],
                "size_y": size[1],
                "size_z": size[2],
                "min_distance_to_any_named_site_A": 0.0,
            }
        )
    return pd.DataFrame(rows)


def de_novo_boxes(
    target: pd.Series,
    fpocket_directory: Path,
    named_coordinates: Dict[str, np.ndarray],
    parse_fpocket_info,
    parse_pqr_coordinates,
) -> pd.DataFrame:
    info_paths = list(fpocket_directory.glob("*_info.txt"))
    if len(info_paths) != 1:
        raise RuntimeError(f"Expected one fpocket info file in {fpocket_directory}")
    info = parse_fpocket_info(info_paths[0])
    all_named = np.concatenate(list(named_coordinates.values()), axis=0)
    candidates: List[Dict[str, object]] = []
    for rank, values in info.items():
        vertices_path = fpocket_directory / "pockets" / f"pocket{rank}_vert.pqr"
        if not vertices_path.exists():
            continue
        vertices = parse_pqr_coordinates(vertices_path)
        center, size = make_box(vertices, padding=5.0, minimum=22.5, maximum=31.5)
        candidates.append(
            {
                "rank": int(rank),
                "score": values.get("Score", np.nan),
                "druggability": values.get("Druggability Score", np.nan),
                "center": center,
                "size": size,
                "distance": nearest_distance(vertices, all_named),
            }
        )
    minimum_distance = float(target.min_de_novo_distance_A)
    eligible = [item for item in candidates if float(item["distance"]) > minimum_distance]
    # Prespecified union of the most druggable and highest-scoring cavities.
    best_score = sorted(
        eligible,
        key=lambda item: (-np.nan_to_num(item["score"], nan=-1.0), int(item["rank"])),
    )[:16]
    best_drug = sorted(
        eligible,
        key=lambda item: (-np.nan_to_num(item["druggability"], nan=-1.0), int(item["rank"])),
    )[:16]
    union = {int(item["rank"]): item for item in best_score + best_drug}
    ordered = sorted(
        union.values(),
        key=lambda item: (
            -np.nan_to_num(item["druggability"], nan=-1.0),
            -np.nan_to_num(item["score"], nan=-1.0),
            int(item["rank"]),
        ),
    )
    retained: List[Dict[str, object]] = []
    for item in ordered:
        if any(np.linalg.norm(np.asarray(item["center"]) - np.asarray(old["center"])) < 7.0 for old in retained):
            continue
        retained.append(item)
        if len(retained) >= int(target.max_de_novo_pockets):
            break
    if not retained:
        raise RuntimeError(f"No de-novo pocket survived for {target.target_id}")
    rows: List[Dict[str, object]] = []
    for item in retained:
        center, size = np.asarray(item["center"]), np.asarray(item["size"])
        rows.append(
            {
                "target_id": target.target_id,
                "pocket_id": f"fpocket_{item['rank']}",
                "pocket_class": "de_novo",
                "claim_role": "exploratory_best_of_n",
                "fpocket_rank": int(item["rank"]),
                "fpocket_score": item["score"],
                "druggability_score": item["druggability"],
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "size_x": size[0],
                "size_y": size[1],
                "size_z": size[2],
                "min_distance_to_any_named_site_A": item["distance"],
            }
        )
    return pd.DataFrame(rows)


def protonate_smiles(obabel: Path, smiles: str, pH: float) -> str:
    command = [str(obabel), f"-:{smiles}", "-osmi", "-p", str(pH)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"Open Babel protonation failed: {' '.join(command)}\n{result.stderr}")
    return result.stdout.split()[0]


def prepare_ligand(
    ligand: pd.Series,
    output_directory: Path,
    env_bin: Path,
    pH: float,
) -> Dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    protonated_smiles = protonate_smiles(env_bin / "obabel", str(ligand.canonical_smiles), pH)
    molecule = Chem.MolFromSmiles(protonated_smiles)
    if molecule is None:
        raise RuntimeError(f"RDKit could not parse pH-protonated SMILES for {ligand.ligand_key}")
    net_charge = int(Chem.GetFormalCharge(molecule))
    positive_atoms = sum(atom.GetFormalCharge() > 0 for atom in molecule.GetAtoms())
    negative_atoms = sum(atom.GetFormalCharge() < 0 for atom in molecule.GetAtoms())
    if net_charge != int(ligand.expected_net_charge_pH7_4):
        raise RuntimeError(
            f"{ligand.ligand_key}: expected net charge {ligand.expected_net_charge_pH7_4} at pH {pH}, got {net_charge}: {protonated_smiles}"
        )
    if positive_atoms < int(ligand.minimum_positive_atoms) or negative_atoms < int(ligand.minimum_negative_atoms):
        raise RuntimeError(
            f"{ligand.ligand_key}: charge-pattern contract failed (+ atoms={positive_atoms}, - atoms={negative_atoms})"
        )

    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 20260825
    parameters.numThreads = 0
    conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=20, params=parameters))
    if not conformer_ids:
        raise RuntimeError(f"No conformer generated for {ligand.ligand_key}")
    energies: List[Tuple[float, int]] = []
    mmff_properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
    for conformer_id in conformer_ids:
        try:
            forcefield = (
                AllChem.MMFFGetMoleculeForceField(molecule, mmff_properties, confId=conformer_id)
                if mmff_properties is not None
                else AllChem.UFFGetMoleculeForceField(molecule, confId=conformer_id)
            )
            forcefield.Minimize(maxIts=1000)
            energies.append((float(forcefield.CalcEnergy()), int(conformer_id)))
        except Exception:
            continue
    if not energies:
        raise RuntimeError(f"No conformer optimized for {ligand.ligand_key}")
    energy, selected_id = min(energies)
    selected = Chem.Mol(molecule)
    selected.RemoveAllConformers()
    selected.AddConformer(molecule.GetConformer(selected_id), assignId=True)
    selected.SetProp("_Name", str(ligand.ligand_chembl_id))

    sdf_path = output_directory / "ligand_3d_pH7_4.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(selected)
    writer.close()
    setups = list(MoleculePreparation(charge_model="gasteiger").prepare(selected))
    if len(setups) != 1:
        raise RuntimeError(f"Expected one Meeko setup for {ligand.ligand_key}, got {len(setups)}")
    pdbqt, success, error = PDBQTWriterLegacy.write_string(setups[0])
    if not success:
        raise RuntimeError(f"Meeko preparation failed for {ligand.ligand_key}: {error}")
    pdbqt_path = output_directory / "ligand.pdbqt"
    pdbqt_path.write_text(pdbqt)
    metadata = {
        "ligand_key": ligand.ligand_key,
        "ligand_chembl_id": ligand.ligand_chembl_id,
        "ligand_role": ligand.ligand_role,
        "input_canonical_smiles": ligand.canonical_smiles,
        "protonated_smiles_pH7_4": protonated_smiles,
        "pH": pH,
        "net_formal_charge": net_charge,
        "positive_formal_charge_atoms": positive_atoms,
        "negative_formal_charge_atoms": negative_atoms,
        "heavy_atoms": int(selected.GetNumHeavyAtoms()),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(selected)),
        "molecular_weight_after_protonation": float(Descriptors.MolWt(selected)),
        "selected_conformer_energy": energy,
        "n_embedded_conformers": len(conformer_ids),
        "sdf_sha256": sha256(sdf_path),
        "pdbqt_sha256": sha256(pdbqt_path),
    }
    (output_directory / "ligand_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--env-bin", required=True, type=Path)
    parser.add_argument("--fpocket-bin", required=True, type=Path)
    parser.add_argument("--pH", type=float, default=7.4)
    args = parser.parse_args()
    root = args.root.resolve()
    workspace = root.parents[1]
    inputs = root / "inputs"
    raw_directory = inputs / "raw"
    receptor_directory = inputs / "receptors"
    ligand_directory = inputs / "ligands"
    for directory in (raw_directory, receptor_directory, ligand_directory):
        directory.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(root / "targets.tsv", sep="\t")
    sites = pd.read_csv(root / "sites.tsv", sep="\t")
    ligands = pd.read_csv(root / "ligands.tsv", sep="\t")
    systems = pd.read_csv(root / "systems.tsv", sep="\t")
    comparisons = pd.read_csv(root / "comparisons.tsv", sep="\t")
    if set(sites.target_id) - set(targets.target_id):
        raise RuntimeError("sites.tsv contains unknown target IDs")
    if set(systems.target_id) - set(targets.target_id) or set(systems.ligand_key) - set(ligands.ligand_key):
        raise RuntimeError("systems.tsv contains unknown target or ligand IDs")
    if set(comparisons.target_id) - set(targets.target_id):
        raise RuntimeError("comparisons.tsv contains unknown target IDs")

    prepare_receptor, parse_fpocket_info, parse_pqr_coordinates = load_legacy_receptor_preparer(workspace)
    box_tables: List[pd.DataFrame] = []
    target_audit: List[Dict[str, object]] = []
    site_audit: List[Dict[str, object]] = []
    for target_tuple in targets.itertuples(index=False):
        target = pd.Series(target_tuple._asdict())
        target_sites = sites[sites.target_id == target.target_id].copy()
        source_suffix = ".cif" if str(target.structure_url).lower().endswith(".cif") else ".pdb"
        source_path = raw_directory / f"{target.structure_id}{source_suffix}"
        download(str(target.structure_url), source_path)
        current_directory = receptor_directory / target.target_id
        current_directory.mkdir(parents=True, exist_ok=True)
        selected_path = current_directory / "protein_selected.pdb"
        retain_selectors = parse_selectors(target.retain_selectors)
        coordinates, retained_labels, n_chains, site_flags = select_structure_and_sites(
            source_path=source_path,
            selected_path=selected_path,
            site_directory=current_directory / "site_atoms",
            chain_map=parse_chain_map(target.chain_map),
            sites=target_sites,
            retain_selectors=retain_selectors,
        )
        receptor_path = current_directory / "receptor.pdbqt"
        retain_names = sorted({selector.residue_name for selector in retain_selectors})
        preparation_method = prepare_receptor(args.env_bin, selected_path, receptor_path, retain_names)
        fpocket_directory = run_fpocket(args.fpocket_bin, selected_path, current_directory)
        current_boxes = pd.concat(
            [
                named_boxes(target_sites, coordinates),
                de_novo_boxes(
                    target,
                    fpocket_directory,
                    coordinates,
                    parse_fpocket_info,
                    parse_pqr_coordinates,
                ),
            ],
            ignore_index=True,
        )
        box_tables.append(current_boxes)

        for site in target_sites.itertuples(index=False):
            other_distances = {
                other: nearest_distance(coordinates[site.pocket_id], values)
                for other, values in coordinates.items()
                if other != site.pocket_id
            }
            site_audit.append(
                {
                    "target_id": target.target_id,
                    "pocket_id": site.pocket_id,
                    "pocket_class": site.pocket_class,
                    "selectors": site.selectors,
                    "n_heavy_atoms": len(coordinates[site.pocket_id]),
                    "expected_heavy_atoms": int(site.expected_heavy_atoms),
                    "matched_het_flags": site_flags[site.pocket_id],
                    "nearest_other_named_site_A": min(other_distances.values()) if other_distances else np.nan,
                    "nearest_other_named_site": min(other_distances, key=other_distances.get) if other_distances else "",
                    "site_pdb_sha256": sha256(current_directory / "site_atoms" / f"{site.pocket_id}.pdb"),
                }
            )
        target_audit.append(
            {
                "target_id": target.target_id,
                "structure_id": target.structure_id,
                "source_sha256": sha256(source_path),
                "selected_chain_count": n_chains,
                "selected_pdb_sha256": sha256(selected_path),
                "receptor_pdbqt_sha256": sha256(receptor_path),
                "receptor_preparation_method": preparation_method,
                "retained_exact_residues": ";".join(retained_labels),
                "n_named_sites": len(target_sites),
                "n_de_novo_sites": int((current_boxes.pocket_class == "de_novo").sum()),
            }
        )
        print(f"prepared target {target.target_id}: {len(current_boxes)} boxes", flush=True)

    ligand_audit: List[Dict[str, object]] = []
    for ligand_tuple in ligands.itertuples(index=False):
        ligand = pd.Series(ligand_tuple._asdict())
        metadata = prepare_ligand(ligand, ligand_directory / ligand.ligand_key, args.env_bin, args.pH)
        ligand_audit.append(metadata)
        print(
            f"prepared ligand {ligand.ligand_key}: charge={metadata['net_formal_charge']} "
            f"(+{metadata['positive_formal_charge_atoms']}/-{metadata['negative_formal_charge_atoms']})",
            flush=True,
        )

    boxes = pd.concat(box_tables, ignore_index=True)
    boxes.to_csv(inputs / "docking_boxes.tsv", sep="\t", index=False)
    pd.DataFrame(target_audit).to_csv(inputs / "target_preparation_audit.tsv", sep="\t", index=False)
    pd.DataFrame(site_audit).to_csv(inputs / "site_definition_audit.tsv", sep="\t", index=False)
    pd.DataFrame(ligand_audit).to_csv(inputs / "ligand_preparation_audit.tsv", sep="\t", index=False)

    contract_files = [
        root / "targets.tsv",
        root / "sites.tsv",
        root / "comparisons.tsv",
        root / "ligands.tsv",
        root / "systems.tsv",
        inputs / "docking_boxes.tsv",
        inputs / "target_preparation_audit.tsv",
        inputs / "site_definition_audit.tsv",
        inputs / "ligand_preparation_audit.tsv",
    ]
    validation = {
        "status": "validated",
        "pH": args.pH,
        "exact_chain_resname_resnum_site_selection": True,
        "polymer_hetero_name_collision_blocked": True,
        "ligand_protonation_before_3d_embedding": True,
        "required_cofactors_fail_closed": True,
        "n_targets": int(len(targets)),
        "n_ligands": int(len(ligands)),
        "n_systems": int(len(systems)),
        "n_named_boxes": int((boxes.pocket_class != "de_novo").sum()),
        "n_de_novo_boxes": int((boxes.pocket_class == "de_novo").sum()),
        "file_sha256": {str(path.relative_to(root)): sha256(path) for path in contract_files},
    }
    (inputs / "PREPARATION_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
