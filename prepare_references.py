"""Integrate identity-resolved source annotations into exact protein–ligand pairs."""

from pathlib import Path
import argparse
import hashlib
import json
import re

import pandas as pd
from rdkit import Chem


SOURCES = {"ASD", "GtoPdb", "KLIFS", "BRENDA", "UniProt", "ChEBI", "KEGG"}
KEY = ["uniprot", "full_inchikey"]


def integrate_sources(frame):
    """Validate previously resolved identities; retain conflicts for review.

    Source-specific role assessment and salt/fragment decisions precede this
    function. A compound name or enzyme-family match cannot supply exact identity.
    """
    required = KEY + ["source", "source_record_id", "canonical_smiles", "binary_label"]
    if not set(required).issubset(frame):
        raise ValueError(f"Required columns: {required}")
    if frame[required].isna().any().any():
        raise ValueError("Missing source identity, structure or label")
    if not set(frame.source).issubset(SOURCES) or not set(frame.binary_label).issubset(
        {0, 1}
    ):
        raise ValueError("Unknown source or binding-role label")
    accepted, unresolved = [], []
    for row in frame.to_dict("records"):
        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        exact = mol is not None and Chem.MolToInchiKey(mol) == row["full_inchikey"]
        if (
            not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", str(row["full_inchikey"]))
            or not exact
        ):
            unresolved.append(
                {**row, "reason": "structure_does_not_reproduce_exact_identity"}
            )
            continue
        row["canonical_smiles"] = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        )
        accepted.append(row)
    valid = pd.DataFrame(accepted, columns=required)
    if valid.empty:
        raise ValueError("No identity-resolved source records")
    conflict = valid.groupby(KEY).binary_label.transform("nunique") > 1
    conflicts = valid[conflict].copy()
    pairs = []
    for (protein, ligand), group in valid[~conflict].groupby(KEY, sort=True):
        pair_id = (
            "PAIR_" + hashlib.sha256(f"{protein}|{ligand}".encode()).hexdigest()[:20]
        )
        pairs.append(
            dict(
                main_row_id=pair_id,
                uniprot=protein,
                full_inchikey=ligand,
                connectivity_key=ligand[:14],
                binary_label=int(group.binary_label.iloc[0]),
                canonical_smiles=sorted(
                    set(group.canonical_smiles), key=lambda x: (len(x), x)
                )[0],
                sources=";".join(sorted(set(group.source))),
                source_records=len(group),
            )
        )
    return (
        pd.DataFrame(pairs),
        valid,
        conflicts,
        pd.DataFrame(unresolved, columns=required + ["reason"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat(
        [pd.read_csv(p, sep="\t") for p in args.sources], ignore_index=True
    )
    pairs, lineage, conflicts, unresolved = integrate_sources(frame)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [
        ("pairs", pairs),
        ("source_records", lineage),
        ("conflicts", conflicts),
        ("unresolved", unresolved),
    ]:
        data.to_csv(args.output / f"{name}.tsv", sep="\t", index=False, na_rep="NA")
    summary = dict(
        source_records=len(frame),
        pairs=len(pairs),
        proteins=pairs.uniprot.nunique(),
        conflicting_pairs=len(conflicts[KEY].drop_duplicates()),
        unresolved_records=len(unresolved),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
