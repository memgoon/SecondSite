#!/usr/bin/env python3
"""Pure-pandas locked split contract shared by CPU audit and GPU training."""

from __future__ import annotations

import pandas as pd


def pfam_set(frame):
    values = set()
    for value in frame["pfam_accessions"].fillna("").astype(str):
        values.update(token for token in value.split(";") if token.startswith("PF"))
    return values


def row_pfams(value):
    return {token for token in str(value).split(";") if token.startswith("PF")}


def class_and_protein_balanced_weights(train):
    size = train.groupby(["uniprot", "binary_label"])["main_row_id"].transform("size").astype(float)
    weight = 1.0 / size
    class_total = weight.groupby(train["binary_label"]).transform("sum")
    weight = weight / class_total.clip(lower=1e-12)
    return weight * (len(weight) / weight.sum())


def build_split(frame, regime, test_fold):
    if regime not in {"row_random", "unseen_family"}:
        raise ValueError("unknown regime: {}".format(regime))
    core = frame[frame["pool_membership"].eq("frozen_arm_a_core")].copy()
    extra = frame[frame["pool_membership"].eq("additional_input_ready_pair")].copy()
    fold_column = "row_fold" if regime == "row_random" else "family_fold"
    validation_fold = (int(test_fold) + 1) % 5
    core_train = core[~core[fold_column].isin([test_fold, validation_fold])].copy()
    validation = core[core[fold_column].eq(validation_fold)].copy()
    test = core[core[fold_column].eq(test_fold)].copy()
    heldout_pair_keys = set(validation["pair_key"].astype(str)) | set(test["pair_key"].astype(str))
    eligible = extra[~extra["pair_key"].astype(str).isin(heldout_pair_keys)].copy()
    exclusion_counts = {
        "exact_locked_pair": int(len(extra) - len(eligible)),
        "heldout_protein": 0,
        "heldout_pfam": 0,
        "missing_pfam": 0,
    }
    if regime == "unseen_family":
        heldout_proteins = set(validation["uniprot"].astype(str)) | set(test["uniprot"].astype(str))
        heldout_pfams = pfam_set(pd.concat([validation, test], ignore_index=True))
        same_protein = eligible["uniprot"].astype(str).isin(heldout_proteins)
        missing_pfam = eligible["pfam_accessions"].fillna("").astype(str).map(
            lambda value: not row_pfams(value)
        )
        shared_pfam = eligible["pfam_accessions"].fillna("").astype(str).map(
            lambda value: bool(row_pfams(value) & heldout_pfams)
        )
        exclusion_counts["heldout_protein"] = int(same_protein.sum())
        exclusion_counts["missing_pfam"] = int((missing_pfam & ~same_protein).sum())
        exclusion_counts["heldout_pfam"] = int((shared_pfam & ~same_protein).sum())
        eligible = eligible[~same_protein & ~missing_pfam & ~shared_pfam].copy()

    train = pd.concat([core_train, eligible], ignore_index=True)
    if set(core_train["main_row_id"].astype(str)) - set(train["main_row_id"].astype(str)):
        raise RuntimeError("Arm B train lost an Arm A training row")
    if set(train["pair_key"].astype(str)) & heldout_pair_keys:
        raise RuntimeError("Arm B train contains a locked validation/test exact pair")
    if regime == "unseen_family":
        heldout = pd.concat([validation, test], ignore_index=True)
        if set(train["uniprot"].astype(str)) & set(heldout["uniprot"].astype(str)):
            raise RuntimeError("protein overlap in Arm B unseen-family split")
        if pfam_set(train) & pfam_set(heldout):
            raise RuntimeError("Pfam overlap in Arm B unseen-family split")

    development_compounds = set(train["connectivity_key"].astype(str)) | set(
        validation["connectivity_key"].astype(str)
    )
    test["unseen_compound"] = (
        ~test["connectivity_key"].astype(str).isin(development_compounds)
    ).astype(int)
    validation["unseen_compound"] = (
        ~validation["connectivity_key"].astype(str).isin(set(train["connectivity_key"].astype(str)))
    ).astype(int)
    train["unseen_compound"] = 0
    train["training_weight"] = class_and_protein_balanced_weights(train)
    validation["training_weight"] = 1.0
    test["training_weight"] = 1.0
    heldout = pd.concat([validation, test], ignore_index=True)
    audit = {
        "regime": regime,
        "test_fold": int(test_fold),
        "validation_fold": int(validation_fold),
        "arm_a_train_rows": int(len(core_train)),
        "eligible_additional_rows": int(len(eligible)),
        "eligible_additional_proteins": int(eligible["uniprot"].nunique()),
        "arm_b_train_rows": int(len(train)),
        "locked_validation_rows": int(len(validation)),
        "locked_test_rows": int(len(test)),
        "common_unseen_compound_test_rows": int(test["unseen_compound"].eq(1).sum()),
        "excluded_additional_rows": exclusion_counts,
        "train_locked_pair_overlap": int(len(set(train["pair_key"].astype(str)) & heldout_pair_keys)),
        "train_heldout_protein_overlap": int(
            len(set(train["uniprot"].astype(str)) & set(heldout["uniprot"].astype(str)))
        ) if regime == "unseen_family" else None,
        "train_heldout_pfam_overlap": int(
            len(pfam_set(train) & pfam_set(heldout))
        ) if regime == "unseen_family" else None,
        "development_common_unseen_compound_overlap": int(
            len(
                development_compounds
                & set(test.loc[test["unseen_compound"].eq(1), "connectivity_key"].astype(str))
            )
        ),
    }
    return train, validation, test, validation_fold, audit
