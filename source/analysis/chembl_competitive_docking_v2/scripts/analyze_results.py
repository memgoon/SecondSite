#!/usr/bin/env python3
"""Aggregate direct named-site comparisons and exploratory de-novo docking."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCORE = re.compile(r"^REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)")


def first_score(path: Path) -> float:
    for line in path.read_text().splitlines():
        match = SCORE.match(line)
        if match:
            return float(match.group(1))
    raise RuntimeError(f"No Vina score in {path}")


def first_pose_atoms(path: Path) -> np.ndarray:
    coordinates: List[Tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line.split()[-1].upper() if line.split() else ""
        if element.startswith("H"):
            continue
        coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not coordinates:
        raise RuntimeError(f"No first-pose heavy atoms in {path}")
    return np.asarray(coordinates, dtype=float)


def pdb_atoms(path: Path) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    coordinates: List[Tuple[float, float, float]] = []
    metadata: List[Dict[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip().upper() or line[12:16].strip()[:1].upper()
        if element == "H":
            continue
        coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        metadata.append(
            {
                "chain": line[21].strip() or "_",
                "resname": line[17:20].strip(),
                "resseq": line[22:26].strip(),
                "icode": line[26].strip(),
            }
        )
    if not coordinates:
        raise RuntimeError(f"No heavy atoms in {path}")
    return np.asarray(coordinates, dtype=float), metadata


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.min(cKDTree(b).query(a, k=1)[0]))


def contacts(
    pose: np.ndarray,
    protein: np.ndarray,
    metadata: List[Dict[str, str]],
    cutoff: float = 4.0,
) -> Tuple[str, int]:
    tree = cKDTree(protein)
    atom_indices = sorted({index for hits in tree.query_ball_point(pose, cutoff) for index in hits})
    residues = sorted(
        {
            (metadata[index]["chain"], metadata[index]["resname"], metadata[index]["resseq"], metadata[index]["icode"])
            for index in atom_indices
        },
        key=lambda item: (item[0], int(item[2]) if item[2].lstrip("-").isdigit() else 10**9, item[2], item[3]),
    )
    return ",".join(f"{chain}:{name}{number}{icode}" for chain, name, number, icode in residues), len(residues)


def read_complete_jobs(root: Path, jobs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    incomplete: List[Dict[str, object]] = []
    for job in jobs.itertuples(index=False):
        status_path = Path(job.output_dir) / "status.json"
        pose_path = Path(job.output_pose)
        try:
            status = json.loads(status_path.read_text())
            if status.get("status") != "complete":
                raise RuntimeError(f"status={status.get('status')}")
            if status.get("job_contract_sha256") != job.job_contract_sha256:
                raise RuntimeError("job contract mismatch")
            if not pose_path.exists() or pose_path.stat().st_size == 0:
                raise RuntimeError("pose missing")
            score = first_score(pose_path)
        except Exception as error:
            incomplete.append({"job_id": job.job_id, "reason": str(error)})
            continue
        rows.append(
            {
                "job_id": job.job_id,
                "system_id": job.system_id,
                "system_role": job.system_role,
                "target_id": job.target_id,
                "ligand_key": job.ligand_key,
                "ligand_chembl_id": job.ligand_chembl_id,
                "pocket_id": job.pocket_id,
                "pocket_class": job.pocket_class,
                "claim_role": job.claim_role,
                "scoring": job.scoring,
                "replicate": int(job.replicate),
                "seed": int(job.seed),
                "best_score": score,
                "elapsed_seconds": status.get("elapsed_seconds", math.nan),
                "pose_path": str(pose_path),
            }
        )
    if incomplete or len(rows) != len(jobs):
        (root / "results").mkdir(parents=True, exist_ok=True)
        (root / "results" / "incomplete_jobs.json").write_text(json.dumps(incomplete, indent=2) + "\n")
        raise RuntimeError(f"Docking incomplete ({len(rows)}/{len(jobs)})")
    return pd.DataFrame(rows)


def named_comparisons(
    table: pd.DataFrame,
    systems: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    replicate_rows: List[Dict[str, object]] = []
    for comparison in comparisons.itertuples(index=False):
        target_systems = systems[systems.target_id == comparison.target_id]
        for system in target_systems.itertuples(index=False):
            for scoring in sorted(table.scoring.unique()):
                group = table[(table.system_id == system.system_id) & (table.scoring == scoring)]
                reference = group[group.pocket_id == comparison.reference_pocket].set_index("replicate")
                test = group[group.pocket_id == comparison.test_pocket].set_index("replicate")
                if reference.empty or test.empty or set(reference.index) != set(test.index):
                    raise RuntimeError(
                        f"Missing/misaligned named sites for {comparison.comparison_id}/{system.system_id}/{scoring}"
                    )
                for replicate in sorted(reference.index):
                    reference_score = float(reference.loc[replicate, "best_score"])
                    test_score = float(test.loc[replicate, "best_score"])
                    replicate_rows.append(
                        {
                            "comparison_id": comparison.comparison_id,
                            "comparison_role": comparison.comparison_role,
                            "target_id": comparison.target_id,
                            "system_id": system.system_id,
                            "system_role": system.system_role,
                            "ligand_key": system.ligand_key,
                            "scoring": scoring,
                            "replicate": int(replicate),
                            "reference_pocket": comparison.reference_pocket,
                            "test_pocket": comparison.test_pocket,
                            "reference_score": reference_score,
                            "test_score": test_score,
                            "test_minus_reference_kcal_mol": test_score - reference_score,
                        }
                    )
    replicates = pd.DataFrame(replicate_rows)
    if replicates.empty:
        return replicates, pd.DataFrame()
    summary = (
        replicates.groupby(
            [
                "comparison_id",
                "comparison_role",
                "target_id",
                "system_id",
                "system_role",
                "ligand_key",
                "scoring",
                "reference_pocket",
                "test_pocket",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=("replicate", "size"),
            reference_median_score=("reference_score", "median"),
            test_median_score=("test_score", "median"),
            median_test_minus_reference_kcal_mol=("test_minus_reference_kcal_mol", "median"),
            min_test_minus_reference_kcal_mol=("test_minus_reference_kcal_mol", "min"),
            max_test_minus_reference_kcal_mol=("test_minus_reference_kcal_mol", "max"),
            test_site_win_fraction=("test_minus_reference_kcal_mol", lambda values: float((values < 0).mean())),
        )
    )
    return replicates, summary


def cd73_background_null(named_summary: pd.DataFrame) -> pd.DataFrame:
    current = named_summary[named_summary.comparison_id == "CD73_INTERFACE_VS_ACTIVE_A"]
    rows: List[Dict[str, object]] = []
    for scoring in sorted(current.scoring.unique()):
        group = current[current.scoring == scoring]
        candidate = group[group.system_role == "primary_candidate"]
        background = group[group.system_role == "matched_background"]
        if len(candidate) != 1:
            raise RuntimeError(f"Expected one CD73 candidate for {scoring}, found {len(candidate)}")
        candidate_gap = float(candidate.iloc[0].median_test_minus_reference_kcal_mol)
        values = background.median_test_minus_reference_kcal_mol.astype(float).to_numpy()
        if len(values):
            # More negative means stronger apparent dimer-interface preference.
            at_least_as_interface_favoring = int((values <= candidate_gap).sum())
            empirical_p = (1 + at_least_as_interface_favoring) / (1 + len(values))
            rank = 1 + int((values < candidate_gap).sum())
            percentile = 100.0 * rank / (len(values) + 1)
        else:
            at_least_as_interface_favoring = 0
            empirical_p = percentile = np.nan
            rank = np.nan
        rows.append(
            {
                "scoring": scoring,
                "candidate_system_id": candidate.iloc[0].system_id,
                "candidate_interface_minus_active_median_kcal_mol": candidate_gap,
                "n_matched_background_ligands": int(len(values)),
                "background_gap_median": float(np.median(values)) if len(values) else np.nan,
                "background_gap_min": float(np.min(values)) if len(values) else np.nan,
                "background_gap_max": float(np.max(values)) if len(values) else np.nan,
                "backgrounds_at_least_as_interface_favoring": at_least_as_interface_favoring,
                "candidate_rank_more_interface_favoring_is_better": rank,
                "candidate_rank_percentile": percentile,
                "one_sided_empirical_p_plus_one": empirical_p,
                "null_interpretation": "Physicochemical matching controls generic pocket scoring, not biological inactivity against CD73.",
            }
        )
    return pd.DataFrame(rows)


def de_novo_comparisons(table: pd.DataFrame, targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    replicate_rows: List[Dict[str, object]] = []
    for target in targets.itertuples(index=False):
        target_table = table[table.target_id == target.target_id]
        for (system_id, scoring, replicate), group in target_table.groupby(["system_id", "scoring", "replicate"]):
            reference = group[group.pocket_id == target.primary_reference_pocket]
            de_novo = group[group.pocket_class == "de_novo"]
            if len(reference) != 1 or de_novo.empty:
                raise RuntimeError(f"Missing reference/de-novo sites for {system_id}/{scoring}/{replicate}")
            best = de_novo.sort_values(["best_score", "pocket_id"], kind="mergesort").iloc[0]
            replicate_rows.append(
                {
                    "target_id": target.target_id,
                    "system_id": system_id,
                    "system_role": group.iloc[0].system_role,
                    "scoring": scoring,
                    "replicate": int(replicate),
                    "reference_pocket": target.primary_reference_pocket,
                    "reference_score": float(reference.iloc[0].best_score),
                    "best_de_novo_pocket": best.pocket_id,
                    "best_de_novo_score": float(best.best_score),
                    "best_de_novo_minus_reference_kcal_mol": float(best.best_score - reference.iloc[0].best_score),
                    "n_de_novo_pockets": int(de_novo.pocket_id.nunique()),
                    "best_of_n_caveat": True,
                }
            )
    replicates = pd.DataFrame(replicate_rows)
    summary = (
        replicates.groupby(
            ["target_id", "system_id", "system_role", "scoring", "reference_pocket", "n_de_novo_pockets"],
            as_index=False,
        )
        .agg(
            n_seeds=("replicate", "size"),
            median_best_de_novo_minus_reference_kcal_mol=("best_de_novo_minus_reference_kcal_mol", "median"),
            de_novo_win_fraction=("best_de_novo_minus_reference_kcal_mol", lambda values: float((values < 0).mean())),
            dominant_de_novo_pocket=("best_de_novo_pocket", lambda values: values.value_counts().index[0]),
            dominant_de_novo_fraction=("best_de_novo_pocket", lambda values: float(values.value_counts().iloc[0] / len(values))),
        )
    )
    summary["best_of_n_caveat"] = True
    return replicates, summary


def representative_contacts(root: Path, table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    representatives = table.loc[
        table.groupby(["system_id", "pocket_id", "scoring"]).best_score.idxmin()
    ]
    for target_id, target_rows in representatives.groupby("target_id"):
        receptor_directory = root / "inputs" / "receptors" / target_id
        protein, metadata = pdb_atoms(receptor_directory / "protein_selected.pdb")
        site_paths = sorted((receptor_directory / "site_atoms").glob("*.pdb"))
        site_atoms = {path.stem: pdb_atoms(path)[0] for path in site_paths}
        for row in target_rows.itertuples(index=False):
            pose = first_pose_atoms(Path(row.pose_path))
            distances = {site: min_distance(pose, coordinates) for site, coordinates in site_atoms.items()}
            contact_text, n_contacts = contacts(pose, protein, metadata)
            nearest = min(distances, key=distances.get)
            rows.append(
                {
                    "target_id": target_id,
                    "system_id": row.system_id,
                    "system_role": row.system_role,
                    "pocket_id": row.pocket_id,
                    "pocket_class": row.pocket_class,
                    "scoring": row.scoring,
                    "representative_seed": row.seed,
                    "representative_score": row.best_score,
                    "nearest_named_site": nearest,
                    "min_pose_to_nearest_named_site_A": distances[nearest],
                    "all_named_site_distances_A": ";".join(f"{key}={distances[key]:.3f}" for key in sorted(distances)),
                    "n_contact_residues_4A": n_contacts,
                    "contact_residues_4A": contact_text,
                    "pose_path": row.pose_path,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    jobs = pd.read_csv(root / "inputs" / "docking_jobs.tsv", sep="\t")
    systems = pd.read_csv(root / "systems.tsv", sep="\t")
    systems = systems[systems.system_id.isin(jobs.system_id.unique())]
    targets = pd.read_csv(root / "targets.tsv", sep="\t")
    targets = targets[targets.target_id.isin(jobs.target_id.unique())]
    comparisons = pd.read_csv(root / "comparisons.tsv", sep="\t")
    comparisons = comparisons[comparisons.target_id.isin(jobs.target_id.unique())]
    boxes = pd.read_csv(root / "inputs" / "docking_boxes.tsv", sep="\t")

    table = read_complete_jobs(root, jobs)
    table.to_csv(results / "job_results.tsv", sep="\t", index=False)
    pocket_summary = (
        table.groupby(
            ["target_id", "system_id", "system_role", "pocket_id", "pocket_class", "claim_role", "scoring"],
            as_index=False,
        )
        .agg(
            n_seeds=("best_score", "size"),
            median_score=("best_score", "median"),
            mean_score=("best_score", "mean"),
            sd_score=("best_score", "std"),
            min_score=("best_score", "min"),
            max_score=("best_score", "max"),
            median_elapsed_seconds=("elapsed_seconds", "median"),
        )
        .merge(
            boxes[
                [
                    "target_id", "pocket_id", "fpocket_rank", "fpocket_score", "druggability_score",
                    "min_distance_to_any_named_site_A",
                ]
            ],
            on=["target_id", "pocket_id"],
            how="left",
            validate="many_to_one",
        )
    )
    pocket_summary.to_csv(results / "pocket_summary.tsv", sep="\t", index=False)

    named_replicates, named_summary = named_comparisons(table, systems, comparisons)
    named_replicates.to_csv(results / "named_site_comparison_replicates.tsv", sep="\t", index=False)
    named_summary.to_csv(results / "named_site_comparison_summary.tsv", sep="\t", index=False)
    null = cd73_background_null(named_summary) if "CD73" in set(targets.target_id) else pd.DataFrame()
    null.to_csv(results / "cd73_matched_background_null.tsv", sep="\t", index=False)

    de_novo_replicates, de_novo_summary = de_novo_comparisons(table, targets)
    de_novo_replicates.to_csv(results / "de_novo_best_of_n_replicates.tsv", sep="\t", index=False)
    de_novo_summary.to_csv(results / "de_novo_best_of_n_summary.tsv", sep="\t", index=False)
    contact_table = representative_contacts(root, table)
    contact_table.to_csv(results / "representative_pose_contacts.tsv", sep="\t", index=False)

    primary = named_summary[
        (named_summary.system_role.isin(["primary_candidate", "corrected_reanalysis"]))
        & (named_summary.comparison_role.isin(["primary_named_site", "corrected_reanalysis"]))
    ].copy()
    primary.to_csv(results / "primary_named_site_results.tsv", sep="\t", index=False)
    profile = str(jobs.profile.iloc[0])
    if jobs.profile.nunique() != 1:
        raise RuntimeError("One manifest cannot mix run profiles")
    expected_backgrounds = 12 if profile in {"full", "cd73"} else 0
    observed_backgrounds = int(systems[systems.system_role == "matched_background"].system_id.nunique())
    if observed_backgrounds != expected_backgrounds:
        raise RuntimeError(
            f"Profile {profile} expected {expected_backgrounds} CD73 matched backgrounds, found {observed_backgrounds}"
        )
    validation = {
        "status": "validated",
        "profile": profile,
        "n_jobs": int(len(table)),
        "n_targets": int(table.target_id.nunique()),
        "n_systems": int(table.system_id.nunique()),
        "n_scoring_functions": int(table.scoring.nunique()),
        "seeds_per_system_pocket_scoring": sorted(
            table.groupby(["system_id", "pocket_id", "scoring"]).size().unique().tolist()
        ),
        "cd73_matched_background_ligands": observed_backgrounds,
        "primary_cd73_comparison": "known_dimer_interface minus active_site_A, directly matched one box versus one box",
        "cd73_symmetry_sensitivity": "known_dimer_interface minus active_site_B is reported separately",
        "tdo2_annotation_fix": "catalytic HEM402+hetero TRP403 and exo hetero TRP404 use exact chain/resname/resnum selectors",
        "ligand_charge_contract": "pH 7.4 protonation precedes 3-D embedding; L-tryptophan zwitterion and ATP -4 are fail-closed checks",
        "de_novo_role": "secondary exploratory best-of-N analysis only",
        "interpretation_limit": "Docking is structural triage. It does not establish binding, inhibition mechanism, or allostery.",
        "matched_background_limit": "Background compounds are physicochemical controls, not experimentally confirmed CD73 inactives.",
    }
    (results / "VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(primary.to_string(index=False), flush=True)
    if not null.empty:
        print("\nCD73 matched-background null", flush=True)
        print(null.to_string(index=False), flush=True)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
