"""Reconstruct exact BioLiP pairs and fit distance/ligand likelihood rankings."""

from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
import argparse
import hashlib
import itertools
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

FEATURE_CODES = ("D", "MW", "LP", "AR")
CONTINUOUS = {"D", "MW", "LP"}
ALPHA = 0.5
DENSITY_FLOOR = 1e-12
CAPS = {
    "KDE_LEGACY": {
        "D": math.log(100),
        "MW": math.log(10),
        "LP": math.log(10),
        "AR": math.log(10),
    },
    "KDE100": {code: math.log(100) for code in FEATURE_CODES},
}
FEATURES = {
    "D": "distance",
    "MW": "molecular_weight",
    "LP": "clogp",
    "AR": "aromatic_ring_count",
}
METHODS = [("RAW:RAW_D", ("D",))] + [
    (e + ":" + "+".join(s), s)
    for e in ("QNB", "KDE_LEGACY", "KDE100")
    for n in range(1, 5)
    for s in itertools.combinations(FEATURE_CODES, n)
]


def smoothed_bin_model(bin_ids: np.ndarray, labels: np.ndarray, n_bins: int):
    negative = np.bincount(bin_ids[labels == 0], minlength=n_bins).astype(int)
    positive = np.bincount(bin_ids[labels == 1], minlength=n_bins).astype(int)
    negative_probability = (negative + ALPHA) / (
        int(np.sum(labels == 0)) + ALPHA * n_bins
    )
    positive_probability = (positive + ALPHA) / (
        int(np.sum(labels == 1)) + ALPHA * n_bins
    )
    return negative, positive, np.log(positive_probability / negative_probability)


def fit_qnb(train_values: np.ndarray, train_labels: np.ndarray, code: str):
    if code in CONTINUOUS:
        requested = int(math.ceil(len(train_values) ** (1.0 / 3.0)))
        internal_edges = np.unique(
            np.quantile(
                train_values,
                np.linspace(0.0, 1.0, requested + 1),
                method="linear",
            )[1:-1]
        )
        train_bins = np.searchsorted(internal_edges, train_values, side="right")
        n_bins = len(internal_edges) + 1
        categories = []
        edges = [
            float("-inf"),
            *[float(value) for value in internal_edges],
            float("inf"),
        ]
        estimator = "pooled_equal_frequency_bins_cube_root_rule"
    else:
        categories = sorted(np.unique(train_values.astype(int)).tolist())
        lookup = {category: index for index, category in enumerate(categories)}
        train_bins = np.asarray(
            [lookup[int(value)] for value in train_values], dtype=int
        )
        n_bins = len(categories) + 1
        requested = n_bins
        edges = []
        estimator = "observed_categories_plus_unseen_category_bin"
    negative, positive, log_lr = smoothed_bin_model(train_bins, train_labels, n_bins)
    return {
        "estimator": estimator,
        "requested_bins": requested,
        "n_bins": n_bins,
        "internal_edges": (
            internal_edges if code in CONTINUOUS else np.asarray([], dtype=float)
        ),
        "edges": edges,
        "categories": categories,
        "negative_counts": negative,
        "positive_counts": positive,
        "bin_log_lr": log_lr,
    }


def transform_qnb(model: dict, test_values: np.ndarray, code: str):
    if code in CONTINUOUS:
        bin_ids = np.searchsorted(model["internal_edges"], test_values, side="right")
        unseen_rows = 0
    else:
        lookup = {category: index for index, category in enumerate(model["categories"])}
        unseen_bin = model["n_bins"] - 1
        bin_ids = np.asarray(
            [lookup.get(int(value), unseen_bin) for value in test_values], dtype=int
        )
        unseen_rows = int(np.sum(bin_ids == unseen_bin))
    return model["bin_log_lr"][bin_ids], unseen_rows


def fit_kde_component(train_values: np.ndarray, train_labels: np.ndarray, code: str):
    if code in CONTINUOUS:
        negative_fit = gaussian_kde(train_values[train_labels == 0], bw_method="scott")
        positive_fit = gaussian_kde(train_values[train_labels == 1], bw_method="scott")
        return {
            "estimator": "gaussian_kde_scott_training_fold",
            "minimum": float(np.min(train_values)),
            "maximum": float(np.max(train_values)),
            "negative_fit": negative_fit,
            "positive_fit": positive_fit,
            "negative_kde_factor": float(negative_fit.factor),
            "positive_kde_factor": float(positive_fit.factor),
            "categories": [],
            "n_bins": np.nan,
            "negative_counts": np.asarray([], dtype=int),
            "positive_counts": np.asarray([], dtype=int),
            "bin_log_lr": np.asarray([], dtype=float),
        }
    categories = sorted(np.unique(train_values.astype(int)).tolist())
    lookup = {category: index for index, category in enumerate(categories)}
    train_bins = np.asarray([lookup[int(value)] for value in train_values], dtype=int)
    n_bins = len(categories) + 1
    negative, positive, log_lr = smoothed_bin_model(train_bins, train_labels, n_bins)
    return {
        "estimator": "categorical_frequency_with_unseen_category_bin",
        "minimum": np.nan,
        "maximum": np.nan,
        "negative_fit": None,
        "positive_fit": None,
        "negative_kde_factor": np.nan,
        "positive_kde_factor": np.nan,
        "categories": categories,
        "n_bins": n_bins,
        "negative_counts": negative,
        "positive_counts": positive,
        "bin_log_lr": log_lr,
    }


def transform_kde(model: dict, test_values: np.ndarray, code: str):
    if code in CONTINUOUS:
        evaluated = np.clip(test_values, model["minimum"], model["maximum"])
        negative_density = model["negative_fit"](evaluated) + DENSITY_FLOOR
        positive_density = model["positive_fit"](evaluated) + DENSITY_FLOOR
        return np.log(positive_density / negative_density), int(
            np.sum(evaluated != test_values)
        )
    lookup = {category: index for index, category in enumerate(model["categories"])}
    unseen_bin = int(model["n_bins"]) - 1
    bin_ids = np.asarray(
        [lookup.get(int(value), unseen_bin) for value in test_values], dtype=int
    )
    return model["bin_log_lr"][bin_ids], int(np.sum(bin_ids == unseen_bin))


def contact_consensus(group):
    """Average contact frequency within PDB, then equally across PDBs; retain ≥60%."""
    by_pdb = defaultdict(list)
    for row in group.to_dict("records"):
        value = row["binding_uniprot_positions"]
        positions = (
            set()
            if pd.isna(value) or str(value) in {"", "NA"}
            else {int(x) for x in str(value).split(";")}
        )
        by_pdb[row["pdb_id"]].append(positions)
    frequency = defaultdict(Fraction)
    for sets in by_pdb.values():
        counts = Counter(residue for positions in sets for residue in positions)
        for residue, n in counts.items():
            frequency[residue] += Fraction(n, len(sets) * len(by_pdb))
    return sorted(r for r, f in frequency.items() if f >= Fraction(3, 5)), frequency


def aggregate_observations(frame):
    if frame.observation_id.duplicated().any():
        raise ValueError("Duplicate structural observation")
    pairs, frequencies = [], []
    for (protein, ligand), group in frame.groupby(
        ["uniprot", "full_inchikey"], sort=True
    ):
        pid = (
            "PAIR_"
            + hashlib.sha256(
                json.dumps(
                    [protein, ligand],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()[:20]
        )
        core, freq = contact_consensus(group)
        row = dict(
            pair_id=pid,
            uniprot=protein,
            full_inchikey=ligand,
            observations=len(group),
            pdbs=group.pdb_id.nunique(),
            consensus60=";".join(map(str, core)),
            observation_ids=";".join(sorted(group.observation_id)),
        )
        for code, col in FEATURES.items():
            row[code] = (
                np.nan
                if group[col].isna().any()
                else float(group.groupby("pdb_id")[col].median().median())
            )
        for col in [
            "binary_label",
            "family_component_id",
            "protein_fold",
            "family_fold",
        ]:
            if col in group:
                if group[col].nunique(dropna=False) != 1 or group[col].isna().any():
                    raise ValueError(
                        f"Inconsistent reference assignment for {pid}: {col}"
                    )
                row[col] = group[col].iloc[0]
        row["feature_complete"] = all(np.isfinite(row[c]) for c in FEATURE_CODES)
        if {
            "min_heavy_distance",
            "residue_overlap",
            "compound_pass",
            "known_role_pass",
        }.issubset(group):
            row["min_heavy_distance"] = (
                np.nan
                if group.min_heavy_distance.isna().any()
                else group.groupby("pdb_id").min_heavy_distance.median().median()
            )
            row["no_residue_overlap"] = bool(group.residue_overlap.eq(0).all())
            row["compound_pass"] = bool(group.compound_pass.eq(1).all())
            row["known_role_pass"] = bool(group.known_role_pass.eq(1).all())
            row["fraction_ge10"] = (
                group.assign(ge10=group.distance.ge(10))
                .groupby("pdb_id")
                .ge10.mean()
                .mean()
            )
        pairs.append(row)
        frequencies.extend(
            dict(
                pair_id=pid, residue=residue, frequency=float(f), exact_fraction=str(f)
            )
            for residue, f in sorted(freq.items())
        )
    return pd.DataFrame(pairs), pd.DataFrame(frequencies)


def score_methods(train, test):
    if set(train.binary_label) != {0, 1}:
        raise ValueError("One-class reference fit")
    if not np.isfinite(train[list(FEATURE_CODES)].to_numpy(float)).all():
        raise ValueError("Incomplete reference features")
    valid = np.isfinite(test[list(FEATURE_CODES)].to_numpy(float)).all(axis=1)
    result = pd.DataFrame(np.nan, index=test.index, columns=[m for m, _ in METHODS])
    if not valid.any():
        return result
    components = {}
    y = train.binary_label.to_numpy(int)
    for code in FEATURE_CODES:
        x = train[code].to_numpy(float)
        z = test.loc[valid, code].to_numpy(float)
        q = fit_qnb(x, y, code)
        k = fit_kde_component(x, y, code)
        qscore, _ = transform_qnb(q, z, code)
        kscore, _ = transform_kde(k, z, code)
        components["QNB", code] = np.round(qscore, 12)
        for estimator in CAPS:
            components[estimator, code] = np.round(
                np.clip(kscore, -CAPS[estimator][code], CAPS[estimator][code]), 12
            )
    for method, codes in METHODS:
        values = (
            test.loc[valid, "D"].to_numpy(float)
            if method == "RAW:RAW_D"
            else sum(components[method.split(":")[0], c] for c in codes)
        )
        result.loc[valid, method] = np.round(values, 12)
    return result


def held_out_scores(pairs, fold_column):
    output = []
    group = "family_component_id" if fold_column == "family_fold" else "uniprot"
    for fold in sorted(pairs[fold_column].unique()):
        train = pairs[pairs[fold_column] != fold]
        test = pairs[pairs[fold_column] == fold]
        if set(train[group]) & set(test[group]) or set(train.uniprot) & set(
            test.uniprot
        ):
            raise ValueError("Held-out structural reference leakage")
        scores = score_methods(train, test)
        for method, _ in METHODS:
            part = test[
                [
                    "pair_id",
                    "uniprot",
                    "full_inchikey",
                    "binary_label",
                    "family_component_id",
                ]
            ].copy()
            part["method"] = method
            part["score"] = scores[method]
            part["fold"] = fold
            part["regime"] = fold_column
            output.append(part)
    return pd.concat(output, ignore_index=True)


def rank_candidates(reference, candidates):
    scores = score_methods(reference, candidates)
    output = []
    for method, _ in METHODS:
        part = candidates[
            ["pair_id", "uniprot", "full_inchikey", "feature_complete"]
        ].copy()
        part["method"] = method
        part["score"] = scores[method]
        part["global_rank"] = part.score.rank(ascending=False, method="average")
        part["global_fraction"] = (part.global_rank - 0.5) / part.score.notna().sum()
        part["within_protein_rank"] = part.groupby("uniprot").score.rank(
            ascending=False, method="average"
        )
        part["within_protein_fraction"] = (
            part.within_protein_rank - 0.5
        ) / part.groupby("uniprot").score.transform("count")
        output.append(part)
    return pd.concat(output, ignore_index=True)


def candidate_funnel(pairs, rankings):
    views = ["RAW:RAW_D", "KDE_LEGACY:D+AR", "KDE_LEGACY:D+MW", "QNB:D+AR"]
    scored = rankings[rankings.method.isin(views)].copy()
    scored["global_hit"] = scored.global_fraction <= 0.1
    scored["protein_hit"] = scored.within_protein_fraction <= 0.1
    votes = scored.groupby("pair_id")[["global_hit", "protein_hit"]].sum()
    out = pairs.merge(votes, on="pair_id", validate="one_to_one")
    out["spatial_pass"] = (
        (out.D >= 10) & (out.min_heavy_distance >= 6) & out.no_residue_overlap
    )
    out["role_and_compound_pass"] = (
        out.spatial_pass & out.compound_pass & out.known_role_pass
    )
    out["agreement_pass"] = (
        out.role_and_compound_pass & (out.global_hit >= 2) & (out.protein_hit >= 3)
    )
    out["repeated_structure_pass"] = (
        out.agreement_pass & (out.pdbs >= 2) & (out.fraction_ge10 >= 0.75)
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-observations", type=Path, required=True)
    parser.add_argument("--candidate-observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    references, rf = aggregate_observations(
        pd.read_csv(args.reference_observations, sep="\t", float_precision="round_trip")
    )
    candidates, cf = aggregate_observations(
        pd.read_csv(args.candidate_observations, sep="\t", float_precision="round_trip")
    )
    reference = references[references.feature_complete].copy()
    oof = pd.concat(
        [held_out_scores(reference, col) for col in ["protein_fold", "family_fold"]],
        ignore_index=True,
    )
    ranked = rank_candidates(reference, candidates)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, table in [
        ("reference_pairs", references),
        ("candidate_pairs", candidates),
        (
            "contact_frequencies",
            pd.concat([rf.assign(source="reference"), cf.assign(source="candidate")]),
        ),
        ("reference_scores", oof),
        ("candidate_rankings", ranked),
    ]:
        table.to_csv(args.output / f"{name}.tsv.gz", sep="\t", index=False, na_rep="NA")
    if {
        "min_heavy_distance",
        "no_residue_overlap",
        "compound_pass",
        "known_role_pass",
        "fraction_ge10",
    }.issubset(candidates):
        candidate_funnel(candidates, ranked).to_csv(
            args.output / "review_funnel.tsv", sep="\t", index=False, na_rep="NA"
        )


if __name__ == "__main__":
    main()
