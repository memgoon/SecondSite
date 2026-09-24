"""Evaluate fixed predictions with conditional metrics and paired cluster uncertainty."""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(frame, score="score", weights=None):
    y, s = frame.binary_label.to_numpy(int), frame[score].to_numpy(float)
    if not np.isfinite(s).all():
        raise ValueError("Nonfinite prediction")
    if weights is not None:
        keep = np.asarray(weights) > 0
        y, s, weights = y[keep], s[keep], np.asarray(weights)[keep]
    if len(set(y)) != 2:
        return dict(
            auroc=np.nan,
            allosteric_ap=np.nan,
            orthosteric_ap=np.nan,
            symmetric_ap=np.nan,
        )
    pos = average_precision_score(y, s, sample_weight=weights)
    neg = average_precision_score(1 - y, -s, sample_weight=weights)
    return dict(
        auroc=float(roc_auc_score(y, s, sample_weight=weights)),
        allosteric_ap=float(pos),
        orthosteric_ap=float(neg),
        symmetric_ap=float((pos + neg) / 2),
    )


def conditional_metrics(frame, identity, score="score"):
    columns = (["outer_fold"] if "outer_fold" in frame else []) + [identity]
    values, used_rows = [], 0
    groups = list(frame.groupby(columns))
    for _, group in groups:
        if group.binary_label.nunique() == 2:
            values.append(binary_metrics(group, score)["auroc"])
            used_rows += len(group)
    return dict(
        auroc=float(np.mean(values)) if values else np.nan,
        n_rows_used=used_rows,
        n_groups_used=len(values),
        n_groups_skipped=len(groups) - len(values),
        row_coverage=used_rows / len(frame) if len(frame) else 0,
    )


def checkpoint_selection(frame, model, regime, ligand_identity="connectivity_key"):
    if model == "ligand" or (
        model not in {"ligand", "protein"} and regime == "unseen_ligand"
    ):
        groups = ["uniprot"]
    elif model == "protein" or regime in {"unseen_family", "double_unseen"}:
        groups = [ligand_identity]
    elif regime == "row_random":
        groups = ["uniprot", ligand_identity]
    else:
        raise ValueError("Unknown checkpoint selection regime")
    estimates = {key: conditional_metrics(frame, key) for key in groups}
    supported = all(
        v["n_groups_used"] >= 8 and np.isfinite(v["auroc"]) for v in estimates.values()
    )
    value = (
        np.mean([v["auroc"] for v in estimates.values()])
        if supported
        else binary_metrics(frame)["symmetric_ap"]
    )
    if not np.isfinite(value):
        raise ValueError("Validation has no usable checkpoint-selection statistic")
    return float(value), dict(
        metric="conditional_auroc_mean" if supported else "pooled_symmetric_ap",
        fallback=not supported,
        support=estimates,
    )


def ensemble_predictions(frame):
    keys = [k for k in ["dataset", "regime", "model", "main_row_id"] if k in frame]
    if "model" not in keys or "main_row_id" not in keys or "seed" not in frame:
        raise ValueError("Predictions require model, main_row_id and seed")
    if frame.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate prediction for a seed")
    metadata = [
        c
        for c in [
            "uniprot",
            "full_inchikey",
            "connectivity_key",
            "family_component_id",
            "binary_label",
            "outer_fold",
        ]
        if c in frame
    ]
    for _, group in frame.groupby(keys):
        if (
            len(group) != 3
            or group.seed.nunique() != 3
            or any(group[c].nunique(dropna=False) != 1 for c in metadata)
        ):
            raise ValueError(
                "Ensemble requires three seeds with consistent identity, label and fold"
            )
    result = frame.groupby(keys, as_index=False).agg(
        **{c: (c, "first") for c in metadata},
        score=("score", "mean"),
        seed_sd=("score", "std")
    )
    return result


def paired_bootstrap(
    frame,
    score_a,
    score_b,
    cluster="family_component_id",
    replicates=10000,
    seed=20260902,
    identity=None,
):
    """Resample clusters with multiplicity; return A−B on identical rows."""
    if not np.isfinite(frame[[score_a, score_b]].to_numpy(float)).all():
        raise ValueError("Paired bootstrap requires common finite rows")
    codes, levels = pd.factorize(frame[cluster], sort=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        len(levels), np.ones(len(levels)) / len(levels), size=replicates
    )
    if identity is None:
        statistic = lambda col, w: binary_metrics(frame, col, w)["auroc"]
    else:
        columns = (["outer_fold"] if "outer_fold" in frame else []) + [identity]
        blocks = [
            g.index
            for _, g in frame.reset_index(drop=True).groupby(columns)
            if g.binary_label.nunique() == 2
        ]
        nested = all(frame.iloc[idx][cluster].nunique() == 1 for idx in blocks)
        aucs = {
            col: np.array(
                [binary_metrics(frame.iloc[idx], col)["auroc"] for idx in blocks]
            )
            for col in [score_a, score_b]
        }

        def statistic(col, w):
            if nested:
                weights = np.array([w[idx[0]] for idx in blocks])
                return (
                    np.average(aucs[col], weights=weights) if weights.sum() else np.nan
                )
            # A ligand can span multiple held-out families. Recompute its
            # weighted AUROC after resampling, without changing the cluster unit.
            values = [
                binary_metrics(frame.iloc[idx], col, w[idx])["auroc"] for idx in blocks
            ]
            valid = np.asarray(values)[np.isfinite(values)]
            return float(valid.mean()) if len(valid) else np.nan

    point = statistic(score_a, np.ones(len(frame))) - statistic(
        score_b, np.ones(len(frame))
    )
    values = np.array(
        [statistic(score_a, w[codes]) - statistic(score_b, w[codes]) for w in draws]
    )
    valid = values[np.isfinite(values)]
    if len(valid) < 0.95 * replicates:
        raise ValueError("Fewer than 95% of bootstrap replicates are defined")
    return dict(
        delta=float(point),
        ci_low=float(np.quantile(valid, 0.025)),
        ci_high=float(np.quantile(valid, 0.975)),
        replicates=replicates,
        valid_replicates=len(valid),
        cluster=cluster,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-reference",
        choices=["ligand", "protein"],
        help="Optional paired comparisons against this baseline",
    )
    parser.add_argument("--cluster", default="family_component_id")
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    frame = pd.concat(
        [pd.read_csv(p, sep="\t") for p in args.predictions], ignore_index=True
    )
    ensemble = ensemble_predictions(frame)
    rows = []
    keys = [k for k in ["dataset", "regime", "model"] if k in ensemble]
    for identifiers, group in ensemble.groupby(keys):
        identifiers = identifiers if isinstance(identifiers, tuple) else (identifiers,)
        base = dict(zip(keys, identifiers))
        rows.append(
            dict(
                **base,
                metric_scope="pooled",
                n_rows_used=len(group),
                **binary_metrics(group)
            )
        )
        for identity in ["uniprot", "full_inchikey"]:
            rows.append(
                dict(
                    **base,
                    metric_scope="within_" + identity,
                    **conditional_metrics(group, identity)
                )
            )
    args.output.mkdir(parents=True, exist_ok=False)
    ensemble.to_csv(args.output / "ensemble_predictions.tsv.gz", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(
        args.output / "metrics.tsv", sep="\t", index=False, na_rep="NA"
    )
    if args.bootstrap_reference:
        differences = []
        group_keys = [k for k in ["dataset", "regime"] if k in ensemble]
        grouped = ensemble.groupby(group_keys) if group_keys else [((), ensemble)]
        for identifiers, part in grouped:
            identifiers = (
                identifiers if isinstance(identifiers, tuple) else (identifiers,)
            )
            baseline = part[part.model.eq(args.bootstrap_reference)]
            if baseline.empty:
                raise ValueError("Requested baseline is absent")
            for model, test in part[~part.model.eq(args.bootstrap_reference)].groupby(
                "model"
            ):
                paired = test.merge(
                    baseline[["main_row_id", "score"]],
                    on="main_row_id",
                    validate="one_to_one",
                    suffixes=("_a", "_b"),
                )
                if paired.empty:
                    raise ValueError("No common rows for paired comparison")
                for identity in [None, "uniprot", "full_inchikey"]:
                    result = paired_bootstrap(
                        paired,
                        "score_a",
                        "score_b",
                        args.cluster,
                        args.replicates,
                        identity=identity,
                    )
                    differences.append(
                        dict(
                            zip(group_keys, identifiers),
                            model=model,
                            reference=args.bootstrap_reference,
                            metric_scope=(
                                "pooled" if identity is None else "within_" + identity
                            ),
                            rows=len(paired),
                            **result
                        )
                    )
        pd.DataFrame(differences).to_csv(
            args.output / "paired_differences.tsv", sep="\t", index=False, na_rep="NA"
        )


if __name__ == "__main__":
    main()
