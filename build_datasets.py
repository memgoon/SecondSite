"""Construct role-based datasets and audit supplied evaluation partitions."""

from pathlib import Path
import argparse
import itertools
import json

import pandas as pd


FOLD_COLUMNS = {
    "random": "matrix_row_fold",
    "pfam_held_out": "matrix_family_fold",
    "ligand_held_out": "matrix_ligand_fold",
    "double_held_out": "matrix_family_fold",
}

# Dataset and evaluation names used in the manuscript.
DATASET_NAMES = {
    "every_pair": "every_pair",
    "both_class_proteins": "both_class_proteins",
    "both_class_ligands": "both_class_ligands",
    "both_class_proteins_ligands": "both_class_proteins_ligands",
}
REGIME_NAMES = {
    "random": "random",
    "pfam_held_out": "pfam_held_out",
    "ligand_held_out": "ligand_held_out",
    "double_held_out": "double_held_out",
}


def both_roles(frame, identity):
    return frame[frame.groupby(identity).binary_label.transform("nunique") == 2].copy()


def construct_datasets(frame):
    if (
        frame.main_row_id.duplicated().any()
        or frame.duplicated(["uniprot", "full_inchikey"]).any()
    ):
        raise ValueError("The input must contain one row per exact pair")
    if not set(frame.binary_label).issubset({0, 1}):
        raise ValueError("Invalid binding-role label")
    if "matrix_evaluation_eligible" not in frame:
        raise ValueError("Supply the frozen Both-class proteins evaluation membership")
    protein = frame[frame.matrix_evaluation_eligible.eq(1)].copy()
    if (
        protein.empty
        or not protein.groupby("uniprot").binary_label.nunique().eq(2).all()
    ):
        raise ValueError(
            "Frozen Both-class proteins membership must retain both class labels"
        )
    double = protein.copy()
    while True:
        reduced = both_roles(both_roles(double, "uniprot"), "full_inchikey")
        if len(reduced) == len(double):
            break
        double = reduced
    return {
        "every_pair": frame.copy(),
        "both_class_proteins": protein,
        "both_class_ligands": both_roles(frame, "full_inchikey"),
        "both_class_proteins_ligands": double,
    }


def disjoint(parts, columns):
    for col in columns:
        for a, b in itertools.combinations(parts, 2):
            if set(a[col]) & set(b[col]):
                raise ValueError(f"Partition overlap: {col}")


def scaffold_set(frame):
    return {s for text in frame.murcko_scaffolds_json for s in json.loads(text) if s}


def partition(frame, regime, fold, scaffold_exclusion=False):
    """Use frozen fold assignments; purge double-held-out identities before fitting."""
    col = FOLD_COLUMNS[regime]
    # Dataset-specific completed assignments supersede the shared initial folds.
    if col == "matrix_family_fold" and "run_family_fold" in frame:
        col = "run_family_fold"
    elif col == "matrix_row_fold" and "run_row_fold" in frame:
        col = "run_row_fold"
    eligible = (
        pd.Series(True, index=frame.index)
        if "run_family_fold" in frame
        else frame.matrix_evaluation_eligible.eq(1)
    )
    test = frame[frame[col].eq(fold) & eligible].copy()
    validation = frame[frame[col].eq((fold + 1) % 5) & eligible].copy()
    train = frame[~frame[col].isin([fold, (fold + 1) % 5])].copy()
    if regime == "double_held_out":
        validation = validation[
            ~validation.connectivity_key.isin(test.connectivity_key)
        ].copy()
        blocked = set(test.connectivity_key) | set(validation.connectivity_key)
        train = train[~train.connectivity_key.isin(blocked)].copy()
    if scaffold_exclusion:
        if regime != "double_held_out":
            raise ValueError("Scaffold sensitivity uses double-held-out partitions")
        test_scaffolds = scaffold_set(test)
        validation = validation[
            ~validation.murcko_scaffolds_json.map(
                lambda s: bool(set(json.loads(s)) & test_scaffolds)
            )
        ].copy()
        blocked = test_scaffolds | scaffold_set(validation)
        train = train[
            ~train.murcko_scaffolds_json.map(
                lambda s: bool(set(json.loads(s)) & blocked)
            )
        ].copy()
    columns = ["main_row_id"]
    if regime in {"pfam_held_out", "double_held_out"}:
        columns += ["uniprot", "family_component_id"]
    if regime in {"ligand_held_out", "double_held_out"}:
        columns += ["connectivity_key"]
    disjoint([train, validation, test], columns)
    return train, validation, test


def leave_one_family_out(frame, family):
    test = frame[frame.family_component_id.eq(family)].copy()
    train = frame[
        ~frame.family_component_id.eq(family)
        & ~frame.connectivity_key.isin(test.connectivity_key)
    ].copy()
    disjoint(
        [train, test],
        ["main_row_id", "uniprot", "family_component_id", "connectivity_key"],
    )
    return train, frame.iloc[:0].copy(), test


def add_scaffolds(frame):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    frame = frame.copy()
    variants = {}
    for key, group in frame.groupby("connectivity_key"):
        values = set()
        for smiles in group.canonical_smiles.unique():
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Invalid structure for {key}")
            values.add(
                Chem.MolToSmiles(
                    MurckoScaffold.GetScaffoldForMol(molecule), isomericSmiles=False
                )
            )
        variants[key] = json.dumps(sorted(values - {""}))
    frame["murcko_scaffolds_json"] = frame.connectivity_key.map(variants)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ready-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-scaffolds", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.model_ready_pairs, sep="\t")
    if args.include_scaffolds:
        frame = add_scaffolds(frame)
    sets = construct_datasets(frame)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for name, data in sets.items():
        # The extra every-pair proteins augment training; other datasets use all their rows for evaluation.
        data["matrix_evaluation_eligible"] = (
            data.main_row_id.isin(sets["both_class_proteins"].main_row_id).astype(int)
            if name == "every_pair"
            else 1
        )
        output_name = name
        data.to_csv(args.output / f"{output_name}.tsv.gz", sep="\t", index=False)
        summary.append(
            dict(
                dataset=output_name,
                pairs=len(data),
                proteins=data.uniprot.nunique(),
                ligands=data.full_inchikey.nunique(),
                allosteric=int(data.binary_label.sum()),
                orthosteric=int(data.binary_label.eq(0).sum()),
            )
        )
    pd.DataFrame(summary).to_csv(
        args.output / "dataset_counts.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
