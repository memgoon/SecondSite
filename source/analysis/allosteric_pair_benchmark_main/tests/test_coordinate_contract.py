#!/usr/bin/env python3
"""Lightweight CPU checks for the target-chain coordinate contract."""

from pathlib import Path
import hashlib
import json

import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def read_fasta(path):
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    assert lines and lines[0].startswith(">")
    return lines[0][1:], "".join(lines[1:])


def main():
    cohort = pd.read_csv(PACKAGE / "data/MAIN_COHORT.tsv.gz", sep="\t", low_memory=False)
    chains = pd.read_csv(PACKAGE / "data/TARGET_CHAIN_SEQUENCES.tsv", sep="\t", low_memory=False)
    masks = json.loads((PACKAGE / "data/POCKET_INDICES.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (PACKAGE / "validation/POCKET_ALIGNMENT_VALIDATION.json").read_text(encoding="utf-8")
    )
    assert validation["status"] == "validated"
    assert len(chains) == chains["uniprot"].nunique() == cohort["uniprot"].nunique()
    assert set(chains["uniprot"].astype(str)) == set(masks) == set(cohort["uniprot"].astype(str))
    assert set(cohort["family_fold"]) == set(range(5))
    assert cohort.groupby("uniprot")["binary_label"].nunique().eq(2).all()
    assert cohort["representation_unit"].eq("structure_matched_target_chain").all()
    for row in chains.itertuples(index=False):
        uid = str(row.uniprot)
        header, sequence = read_fasta(PACKAGE / "data/target_chain_fasta_v2" / (uid + ".fasta"))
        assert header == uid
        assert len(sequence) == int(row.target_chain_length)
        assert sequence_hash(sequence) == str(row.sequence_sha256)
        values = [int(value) for value in masks[uid]]
        assert values and min(values) >= 0 and max(values) < len(sequence)
        assert len(values) == len(set(values))
    print("target-chain coordinate contract checks passed")


if __name__ == "__main__":
    main()
