#!/usr/bin/env python3
"""Checkpoint 10: filter and prioritize the frozen BioLiP site rankings.

This step does not fit or change a ranking model.  It adds chemical-role
checks, removes obvious inorganic/small-fragment cases from the conservative
review set, and measures agreement among four prespecified ranking views.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path("/disk9/13.Heesu_Allostery")
PACKAGE = ROOT / "analysis/biolip_bayesian_ranking_revision"
DATA = PACKAGE / "data"
MANIFESTS = PACKAGE / "manifests"
REPORTS = PACKAGE / "reports"
VALIDATION = PACKAGE / "validation"

SITE_FEATURES = DATA / "CHECKPOINT9_SITE_FEATURES.tsv.gz"
CANDIDATE_FEATURES = DATA / "CHECKPOINT9_CANDIDATE_FEATURES.tsv.gz"
OBSERVATION_MASTER = DATA / "BIOLIP_EXACT_OBSERVATION_MASTER.tsv.gz"
ORTHOSTERIC_PAIRS = DATA / "ORTHOSTERIC_PAIR_MASTER_AUDITED.tsv.gz"
ALLOSTERIC_LINKS = DATA / "CHECKPOINT5_ALLOSTERIC_EXACT_LINKS.tsv.gz"
ALLOBENCH_REGISTRY = DATA / "CHECKPOINT5_ALLOBENCH_SOURCE_REGISTRY.tsv.gz"
CHECKPOINT9 = VALIDATION / "CHECKPOINT9_VALIDATION.json"
MANUAL_REVIEW = MANIFESTS / "CHECKPOINT10_MANUAL_REVIEW.tsv"

SITE_AUDIT = DATA / "CHECKPOINT10_SITE_CANDIDATE_AUDIT.tsv.gz"
PAIR_SHORTLIST = DATA / "CHECKPOINT10_PAIR_SHORTLIST.tsv.gz"
FILTER_COUNTS = DATA / "CHECKPOINT10_FILTER_COUNTS.tsv"
SENSITIVITY = DATA / "CHECKPOINT10_FILTER_SENSITIVITY.tsv"
INPUT_HASHES = MANIFESTS / "CHECKPOINT10_INPUT_HASHES.tsv"
BUILD = VALIDATION / "CHECKPOINT10_BUILD.json"
REPORT = REPORTS / "CHECKPOINT10_FILTERED_CANDIDATES.md"

DISTANCE = "ligand_centroid_to_nearest_orthosteric_site_CA_centroid_A"
MIN_HEAVY_DISTANCE = "median_ligand_to_site_min_heavy_A"

# These four views were fixed before filtering: direct structural distance,
# two capped-KDE variants that performed well for conditional/top-rank
# retrieval, and the corresponding quantile-bin alternative.
CONSENSUS_SCORES = {
    "raw_distance": "score_RAW_D",
    "kde_distance_aromatic": "score_KDE_LEGACY_D_AR",
    "kde_distance_weight": "score_KDE_LEGACY_D_MW",
    "quantile_distance_aromatic": "score_QNB_D_AR",
}

EXPECTED = {
    "site_rows": 5_150,
    "candidate_pairs": 2_890,
    "candidate_proteins": 153,
    "manual_review_pairs": 6,
}


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


def atomic_text(value: str, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def split_tokens(value: object) -> set[str]:
    return {token for token in str(value or "").split(";") if token}


def structure_properties(smiles_values: list[str]) -> pd.DataFrame:
    # Elements in the conventional metal blocks. B, Si, As, Se, halogens and
    # other common covalent nonmetals are intentionally not called metals.
    metals = (
        set(range(3, 5)) | set(range(11, 14)) | set(range(19, 33))
        | set(range(37, 52)) | set(range(55, 85)) | set(range(87, 119))
    )
    rows = []
    for smiles in sorted(set(smiles_values) - {""}):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            rows.append({
                "canonical_smiles": smiles,
                "rdkit_structure_status": "parse_failed",
                "carbon_atom_count": np.nan,
                "metal_atom_count": np.nan,
                "molecular_fragment_count": np.nan,
            })
            continue
        atomic_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
        rows.append({
            "canonical_smiles": smiles,
            "rdkit_structure_status": "ok",
            "carbon_atom_count": sum(value == 6 for value in atomic_numbers),
            "metal_atom_count": sum(value in metals for value in atomic_numbers),
            "molecular_fragment_count": len(Chem.GetMolFrags(molecule)),
        })
    return pd.DataFrame(rows)


def orthosteric_similarity(site: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    pair = site[[
        "candidate_pair_id", "uniprot", "full_inchikey", "connectivity_key",
        "canonical_smiles",
    ]].drop_duplicates("candidate_pair_id")

    relevant = reference.loc[
        reference.uniprot.isin(set(pair.uniprot)) & reference.canonical_smiles.ne("")
    ].drop_duplicates(["uniprot", "connectivity_key", "canonical_smiles"])
    reference_by_protein: dict[str, list[dict[str, object]]] = {}
    for protein, group in relevant.groupby("uniprot", sort=False):
        records = []
        for row in group.itertuples(index=False):
            molecule = Chem.MolFromSmiles(row.canonical_smiles)
            if molecule is None:
                continue
            records.append({
                "fingerprint": generator.GetFingerprint(molecule),
                "checkpoint1_pair_id": row.checkpoint1_pair_id,
                "full_inchikey": row.full_inchikey,
                "connectivity_key": row.connectivity_key,
                "source_databases": row.source_databases,
                "evidence_subtypes": row.evidence_subtypes,
            })
        reference_by_protein[protein] = records

    rows = []
    for row in pair.itertuples(index=False):
        molecule = Chem.MolFromSmiles(str(row.canonical_smiles))
        references = reference_by_protein.get(row.uniprot, [])
        if molecule is None:
            rows.append({
                "candidate_pair_id": row.candidate_pair_id,
                "orthosteric_similarity_status": "candidate_parse_failed",
                "same_protein_orthosteric_compounds_compared": len(references),
                "max_same_protein_orthosteric_tanimoto": np.nan,
                "closest_orthosteric_pair_id": "",
                "closest_orthosteric_full_inchikey": "",
                "closest_orthosteric_connectivity_key": "",
                "closest_orthosteric_sources": "",
                "closest_orthosteric_role_descriptions": "",
            })
            continue
        if not references:
            rows.append({
                "candidate_pair_id": row.candidate_pair_id,
                "orthosteric_similarity_status": "no_parseable_same_protein_reference",
                "same_protein_orthosteric_compounds_compared": 0,
                "max_same_protein_orthosteric_tanimoto": np.nan,
                "closest_orthosteric_pair_id": "",
                "closest_orthosteric_full_inchikey": "",
                "closest_orthosteric_connectivity_key": "",
                "closest_orthosteric_sources": "",
                "closest_orthosteric_role_descriptions": "",
            })
            continue
        fingerprint = generator.GetFingerprint(molecule)
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, [record["fingerprint"] for record in references]
        )
        index = int(np.argmax(similarities))
        closest = references[index]
        rows.append({
            "candidate_pair_id": row.candidate_pair_id,
            "orthosteric_similarity_status": "ok",
            "same_protein_orthosteric_compounds_compared": len(references),
            "max_same_protein_orthosteric_tanimoto": float(similarities[index]),
            "closest_orthosteric_pair_id": closest["checkpoint1_pair_id"],
            "closest_orthosteric_full_inchikey": closest["full_inchikey"],
            "closest_orthosteric_connectivity_key": closest["connectivity_key"],
            "closest_orthosteric_sources": closest["source_databases"],
            "closest_orthosteric_role_descriptions": closest["evidence_subtypes"],
        })
    return pd.DataFrame(rows)


def add_average_rank_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    global_columns = []
    protein_columns = []
    for short_name, score_column in CONSENSUS_SCORES.items():
        valid_count = int(output[score_column].notna().sum())
        global_column = f"{short_name}_global_average_rank_percentile"
        protein_column = f"{short_name}_within_protein_average_rank_percentile"
        output[global_column] = (
            output[score_column].rank(method="average", ascending=False) - 0.5
        ) / valid_count
        group_count = output.groupby("uniprot")[score_column].transform("count")
        output[protein_column] = (
            output.groupby("uniprot")[score_column].rank(method="average", ascending=False) - 0.5
        ) / group_count
        global_columns.append(global_column)
        protein_columns.append(protein_column)
    output["consensus_global_percentile_median"] = output[global_columns].median(axis=1)
    output["consensus_within_protein_percentile_median"] = output[protein_columns].median(axis=1)
    output["consensus_models_in_global_top_10pct"] = sum(
        output[column].le(0.10) for column in global_columns
    )
    output["consensus_models_in_within_protein_top_10pct"] = sum(
        output[column].le(0.10) for column in protein_columns
    )
    return output


def count_row(name: str, frame: pd.DataFrame, detail: str) -> dict[str, object]:
    return {
        "filter_step": name,
        "site_rows": len(frame),
        "protein_ligand_pairs": frame.candidate_pair_id.nunique(),
        "proteins": frame.uniprot.nunique(),
        "detail": detail,
    }


def main() -> None:
    started = time.time()
    RDLogger.DisableLog("rdApp.*")
    checkpoint9 = json.loads(CHECKPOINT9.read_text())
    if checkpoint9.get("status") != "validated_unlabelled_biolip_rankings":
        raise RuntimeError("checkpoint 9 is not in its validated state")

    site = pd.read_csv(SITE_FEATURES, sep="\t", low_memory=False)
    candidate_features = pd.read_csv(CANDIDATE_FEATURES, sep="\t", low_memory=False)
    orthosteric = pd.read_csv(
        ORTHOSTERIC_PAIRS, sep="\t", dtype=str, keep_default_na=False, low_memory=False
    )
    allosteric_links = pd.read_csv(
        ALLOSTERIC_LINKS, sep="\t", dtype=str, keep_default_na=False, low_memory=False
    )
    allobench = pd.read_csv(
        ALLOBENCH_REGISTRY, sep="\t", dtype=str, keep_default_na=False, low_memory=False
    )
    manual = pd.read_csv(MANUAL_REVIEW, sep="\t", dtype=str, keep_default_na=False)

    if len(site) != EXPECTED["site_rows"]:
        raise RuntimeError("checkpoint-9 site universe changed")
    if site.candidate_pair_id.nunique() != EXPECTED["candidate_pairs"]:
        raise RuntimeError("checkpoint-9 pair universe changed")
    if site.uniprot.nunique() != EXPECTED["candidate_proteins"]:
        raise RuntimeError("checkpoint-9 protein universe changed")
    if len(manual) != EXPECTED["manual_review_pairs"] or manual[["uniprot", "ligand_ccd"]].duplicated().any():
        raise RuntimeError("manual-review table does not contain six unique pairs")

    for column in [DISTANCE, MIN_HEAVY_DISTANCE, "molecular_weight", "heavy_atom_count"]:
        site[column] = pd.to_numeric(site[column], errors="coerce")
    site["any_candidate_site_residue_overlap"] = truthy(
        site["any_candidate_site_residue_overlap"]
    )

    best_resolution = (
        candidate_features.assign(
            resolution_numeric=pd.to_numeric(candidate_features.resolution, errors="coerce")
        )
        .groupby(["uniprot", "full_inchikey", "candidate_site_cluster_id"], sort=False)
        .resolution_numeric.min()
        .rename("best_observation_resolution_A")
        .reset_index()
    )
    site = site.merge(
        best_resolution,
        on=["uniprot", "full_inchikey", "candidate_site_cluster_id"],
        validate="one_to_one",
    )
    site = site.merge(
        structure_properties(site.canonical_smiles.fillna("").tolist()),
        on="canonical_smiles", how="left", validate="many_to_one",
    )
    site = site.merge(
        orthosteric_similarity(site, orthosteric),
        on="candidate_pair_id", validate="many_to_one",
    )

    # Broad allosteric novelty checks deliberately include AlloBench links
    # that could not enter the exact-site validation set used in checkpoint 9.
    known_allosteric_exact = set(zip(allosteric_links.uniprot, allosteric_links.full_inchikey))
    known_allosteric_connectivity = set(zip(
        allosteric_links.uniprot, allosteric_links.connectivity_key
    ))
    known_allosteric_ccd = set(zip(allobench.uniprot, allobench.ligand_ccd))
    site["known_allosteric_exact_pair"] = [
        (protein, ligand) in known_allosteric_exact
        for protein, ligand in zip(site.uniprot, site.full_inchikey)
    ]
    site["known_allosteric_connectivity_pair"] = [
        (protein, ligand) in known_allosteric_connectivity
        for protein, ligand in zip(site.uniprot, site.connectivity_key)
    ]
    site["known_allosteric_protein_ccd_pair"] = [
        any((protein, ccd) in known_allosteric_ccd for ccd in split_tokens(ccds))
        for protein, ccds in zip(site.uniprot, site.ligand_ccds)
    ]
    site["known_allosteric_by_any_broad_link"] = site[[
        "known_allosteric_exact_pair", "known_allosteric_connectivity_pair",
        "known_allosteric_protein_ccd_pair",
    ]].any(axis=1)
    site["same_protein_exact_orthosteric_connectivity"] = (
        site.connectivity_key.eq(site.closest_orthosteric_connectivity_key)
    )

    # Global BioLiP occurrence is a review flag, not an automatic exclusion:
    # a widely observed compound may be a cofactor/drug or an experimental additive.
    master = pd.read_csv(
        OBSERVATION_MASTER, sep="\t",
        usecols=["pdb_id", "uniprot_resolved", "ligand_ccd", "full_inchikey"],
        dtype=str, keep_default_na=False,
    )
    ccd_occurrence = master.groupby("ligand_ccd", sort=False).agg(
        ccd_biolip_observations=("pdb_id", "size"),
        ccd_biolip_pdbs=("pdb_id", "nunique"),
        ccd_biolip_proteins=("uniprot_resolved", "nunique"),
    ).reset_index()
    ccd_lookup = ccd_occurrence.set_index("ligand_ccd").to_dict("index")
    site["max_ccd_biolip_observations"] = [
        max((ccd_lookup.get(ccd, {}).get("ccd_biolip_observations", 0) for ccd in split_tokens(value)), default=0)
        for value in site.ligand_ccds
    ]
    site["max_ccd_biolip_pdbs"] = [
        max((ccd_lookup.get(ccd, {}).get("ccd_biolip_pdbs", 0) for ccd in split_tokens(value)), default=0)
        for value in site.ligand_ccds
    ]
    site["max_ccd_biolip_proteins"] = [
        max((ccd_lookup.get(ccd, {}).get("ccd_biolip_proteins", 0) for ccd in split_tokens(value)), default=0)
        for value in site.ligand_ccds
    ]
    site["widely_observed_compound_review_flag"] = site.max_ccd_biolip_proteins.ge(20)

    site = add_average_rank_percentiles(site)

    site["passes_spatial_separation"] = (
        site[DISTANCE].ge(10.0)
        & site[MIN_HEAVY_DISTANCE].ge(6.0)
        & ~site.any_candidate_site_residue_overlap
    )
    site["passes_substantial_organic_compound"] = (
        site.descriptor_status.eq("ok")
        & site.rdkit_structure_status.eq("ok")
        & site.carbon_atom_count.ge(4)
        & site.heavy_atom_count.ge(8)
        & site.molecular_weight.between(100.0, 1200.0, inclusive="both")
        & site.metal_atom_count.eq(0)
        & site.molecular_fragment_count.eq(1)
    )
    site["passes_known_role_novelty"] = (
        ~site.known_allosteric_by_any_broad_link
        & ~site.same_protein_exact_orthosteric_connectivity
        & site.max_same_protein_orthosteric_tanimoto.lt(0.70)
    )
    site["passes_review_filter"] = site[[
        "passes_spatial_separation", "passes_substantial_organic_compound",
        "passes_known_role_novelty",
    ]].all(axis=1)
    site["passes_ranking_consensus"] = (
        site.passes_review_filter
        & site.consensus_models_in_global_top_10pct.ge(2)
        & site.consensus_models_in_within_protein_top_10pct.ge(3)
    )
    site["passes_multi_structure_shortlist"] = (
        site.passes_ranking_consensus
        & site.source_pdbs.ge(2)
        & site.fraction_observations_ge_10A.ge(0.75)
    )

    # Similarity categories are deliberately asymmetric: high similarity is a
    # risk flag, whereas low similarity is not positive evidence for allostery.
    similarity = site.max_same_protein_orthosteric_tanimoto
    site["orthosteric_similarity_review_category"] = np.select(
        [
            site.same_protein_exact_orthosteric_connectivity,
            similarity.ge(0.80),
            similarity.ge(0.70),
            similarity.notna(),
        ],
        [
            "exact_connectivity_match",
            "strong_analogue_ge_0.80",
            "moderate_analogue_0.70_to_0.80",
            "below_0.70_neutral",
        ],
        default="not_computable",
    )

    counts = [
        count_row("all_ranked_sites", site, "All checkpoint-9 site rows."),
        count_row(
            "spatially_separate",
            site.loc[site.passes_spatial_separation],
            "Site-centroid distance >=10 A, minimum heavy-atom distance >=6 A, and no binding-residue overlap.",
        ),
        count_row(
            "spatial_and_compound_filter",
            site.loc[site.passes_spatial_separation & site.passes_substantial_organic_compound],
            "Also requires one carbon-containing component, 4+ carbon atoms, 8+ heavy atoms, 100-1200 Da, and no metal atom.",
        ),
        count_row(
            "review_filter",
            site.loc[site.passes_review_filter],
            "Also removes broad known AlloBench links, exact same-protein orthosteric connectivity, and same-protein orthosteric similarity >=0.70.",
        ),
        count_row(
            "ranking_consensus",
            site.loc[site.passes_ranking_consensus],
            "Global top 10% in >=2/4 and within-protein top 10% in >=3/4 prespecified ranking views.",
        ),
        count_row(
            "multi_structure_shortlist",
            site.loc[site.passes_multi_structure_shortlist],
            "Consensus site repeated in >=2 PDB entries with >=75% of observations at least 10 A away.",
        ),
    ]

    sensitivity_rows = []
    chemistry = site.passes_substantial_organic_compound
    not_known = ~site.known_allosteric_by_any_broad_link & ~site.same_protein_exact_orthosteric_connectivity
    for distance_cutoff in (8.0, 10.0, 12.0):
        for minimum_heavy_cutoff in (4.0, 6.0, 8.0):
            for similarity_cutoff in (0.60, 0.70, 0.80):
                base = (
                    site[DISTANCE].ge(distance_cutoff)
                    & site[MIN_HEAVY_DISTANCE].ge(minimum_heavy_cutoff)
                    & ~site.any_candidate_site_residue_overlap
                    & chemistry & not_known
                    & site.max_same_protein_orthosteric_tanimoto.lt(similarity_cutoff)
                )
                consensus = (
                    base
                    & site.consensus_models_in_global_top_10pct.ge(2)
                    & site.consensus_models_in_within_protein_top_10pct.ge(3)
                )
                repeated = (
                    consensus & site.source_pdbs.ge(2)
                    & site.fraction_observations_ge_10A.ge(0.75)
                )
                for name, mask in (("review_filter", base), ("ranking_consensus", consensus), ("multi_structure", repeated)):
                    subset = site.loc[mask]
                    sensitivity_rows.append({
                        "distance_cutoff_A": distance_cutoff,
                        "minimum_heavy_atom_distance_cutoff_A": minimum_heavy_cutoff,
                        "orthosteric_similarity_cutoff": similarity_cutoff,
                        "summary_level": name,
                        "site_rows": len(subset),
                        "protein_ligand_pairs": subset.candidate_pair_id.nunique(),
                        "proteins": subset.uniprot.nunique(),
                    })

    repeated = site.loc[site.passes_multi_structure_shortlist].copy()
    pair_rows = []
    for pair_id, group in repeated.groupby("candidate_pair_id", sort=True):
        ordered = group.sort_values(
            ["consensus_global_percentile_median", "consensus_within_protein_percentile_median", "candidate_site_ligand_id"],
            kind="mergesort",
        )
        representative = ordered.iloc[0]
        pair_rows.append({
            "candidate_pair_id": pair_id,
            "uniprot": representative.uniprot,
            "full_inchikey": representative.full_inchikey,
            "connectivity_key": representative.connectivity_key,
            "canonical_smiles": representative.canonical_smiles,
            "ligand_ccds": representative.ligand_ccds,
            "shortlisted_site_locations": len(group),
            "shortlisted_site_ids": ";".join(sorted(group.candidate_site_ligand_id)),
            "representative_site_id": representative.candidate_site_ligand_id,
            "representative_pdb_id": representative.representative_pdb_id,
            "best_observation_resolution_A": group.best_observation_resolution_A.min(),
            "maximum_source_pdbs_for_one_site": group.source_pdbs.max(),
            "maximum_site_distance_A": group[DISTANCE].max(),
            "minimum_site_heavy_atom_distance_A": group[MIN_HEAVY_DISTANCE].min(),
            "max_same_protein_orthosteric_tanimoto": representative.max_same_protein_orthosteric_tanimoto,
            "closest_orthosteric_sources": representative.closest_orthosteric_sources,
            "consensus_global_percentile_best": group.consensus_global_percentile_median.min(),
            "consensus_within_protein_percentile_best": group.consensus_within_protein_percentile_median.min(),
            "max_ccd_biolip_pdbs": group.max_ccd_biolip_pdbs.max(),
            "max_ccd_biolip_proteins": group.max_ccd_biolip_proteins.max(),
            "widely_observed_compound_review_flag": group.widely_observed_compound_review_flag.any(),
        })
    pairs = pd.DataFrame(pair_rows)
    if len(pairs) != EXPECTED["manual_review_pairs"]:
        raise RuntimeError(
            f"multi-structure pair shortlist changed: observed={len(pairs)}, expected=6"
        )
    pairs = pairs.merge(
        manual,
        left_on=["uniprot", "ligand_ccds"], right_on=["uniprot", "ligand_ccd"],
        how="left", validate="one_to_one",
    )
    if pairs.manual_disposition.eq("").any() or pairs.manual_disposition.isna().any():
        raise RuntimeError("a shortlisted pair lacks manual review")
    pairs["retained_after_manual_review"] = ~pairs.manual_disposition.str.startswith("exclude_")
    pairs = pairs.sort_values(
        ["retained_after_manual_review", "consensus_global_percentile_best", "candidate_pair_id"],
        ascending=[False, True, True], kind="mergesort",
    )

    atomic_tsv(site, SITE_AUDIT, compression="gzip")
    atomic_tsv(pairs, PAIR_SHORTLIST, compression="gzip")
    atomic_tsv(pd.DataFrame(counts), FILTER_COUNTS)
    atomic_tsv(pd.DataFrame(sensitivity_rows), SENSITIVITY)

    input_paths = [
        SITE_FEATURES, CANDIDATE_FEATURES, OBSERVATION_MASTER, ORTHOSTERIC_PAIRS,
        ALLOSTERIC_LINKS, ALLOBENCH_REGISTRY, CHECKPOINT9, MANUAL_REVIEW,
    ]
    input_hashes = pd.DataFrame([
        {"asset": path.name, "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in input_paths
    ])
    atomic_tsv(input_hashes, INPUT_HASHES)

    retained = pairs.loc[pairs.retained_after_manual_review]
    lines = [
        "# Checkpoint 10: filtered BioLiP candidates",
        "",
        "This step does not alter or refit any of the 46 ranking definitions.",
        "It applies transparent post-ranking filters and records every removed class.",
        "",
        "## Candidate funnel",
        "",
        "| Step | Site locations | Protein-ligand pairs | Proteins |",
        "|---|---:|---:|---:|",
    ]
    for row in counts:
        lines.append(
            f"| {row['filter_step']} | {row['site_rows']:,} | {row['protein_ligand_pairs']:,} | {row['proteins']:,} |"
        )
    lines.extend([
        "",
        "## Multi-structure shortlist after manual structure-context review",
        "",
        "| UniProt | Protein | CCD | Ligand | Site locations | PDB support | Distance (A) | Same-protein orthosteric Tanimoto | Decision |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in pairs.itertuples(index=False):
        lines.append(
            f"| {row.uniprot} | {row.protein_name} | {row.ligand_ccds} | {row.ligand_name} | "
            f"{row.shortlisted_site_locations} | {row.maximum_source_pdbs_for_one_site} | "
            f"{row.maximum_site_distance_A:.2f} | {row.max_same_protein_orthosteric_tanimoto:.3f} | "
            f"{row.manual_disposition} |"
        )
    lines.extend([
        "",
        f"Automatic multi-structure shortlist: **{len(pairs)} pairs**.  "
        f"Retained after checking the actual structure context: **{len(retained)} pairs**.",
        "",
        "The retained list is a review queue for database expansion, not a claim that five new allosteric mechanisms were discovered. Several entries are established non-active-site ligands that were missing from the frozen AlloBench source table.",
        "",
        "Low fingerprint similarity is treated only as absence of an orthosteric-analogue warning; it is not counted as positive evidence for allostery.",
    ])
    atomic_text("\n".join(lines) + "\n", REPORT)

    output_paths = [SITE_AUDIT, PAIR_SHORTLIST, FILTER_COUNTS, SENSITIVITY, INPUT_HASHES, REPORT]
    build = {
        "status": "built_awaiting_independent_validation",
        "runtime_seconds": time.time() - started,
        "ranking_models_refit": False,
        "ranking_views_used_for_consensus": CONSENSUS_SCORES,
        "site_rows": len(site),
        "review_filter_sites": int(site.passes_review_filter.sum()),
        "ranking_consensus_sites": int(site.passes_ranking_consensus.sum()),
        "multi_structure_sites": int(site.passes_multi_structure_shortlist.sum()),
        "multi_structure_pairs": len(pairs),
        "manually_retained_pairs": int(pairs.retained_after_manual_review.sum()),
        "known_allosteric_sites_removed_by_broad_link_check": int(site.known_allosteric_by_any_broad_link.sum()),
        "output_hashes": {path.name: sha256(path) for path in output_paths},
    }
    atomic_json(build, BUILD)
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
