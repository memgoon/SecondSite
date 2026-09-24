"""Compare neural and structural rankings on exactly matched held-out pairs."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from evaluate_models import binary_metrics


def spearman(a, b):
    a, b = rankdata(a), rankdata(b)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else np.nan


def top_choice_agreement(a, b):
    x, y = np.asarray(a) == max(a), np.asarray(b) == max(b)
    return float((x & y).sum() / (x.sum() * y.sum()))


def joined_family_components(frame):
    """Join the two source family partitions before cluster resampling."""
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in frame.to_dict("records"):
        keys = [
            "protein:" + str(r["uniprot"]),
            "neural:" + str(r["neural_family"]),
            "structural:" + str(r["structural_family"]),
        ]
        for key in keys[1:]:
            a, b = root(keys[0]), root(key)
            parent[max(a, b)] = min(a, b)
    return frame.uniprot.map(lambda u: root("protein:" + str(u)))


def comparison_table(frame):
    records = []
    for (model, method, protein), g in frame.groupby(["model", "method", "uniprot"]):
        if len(g) < 2:
            continue
        rho = spearman(g.neural_score, g.structural_score) if len(g) >= 3 else np.nan
        agreement = top_choice_agreement(g.neural_score, g.structural_score)
        records.append(
            dict(
                model=model,
                method=method,
                uniprot=protein,
                cluster=g.cluster.iloc[0],
                pairs=len(g),
                spearman=rho,
                top_choice_agreement=agreement,
                random_choice_expectation=1 / len(g),
                excess_agreement=agreement - 1 / len(g),
                neural_auroc=binary_metrics(
                    g.rename(columns={"binary_label_neural": "binary_label"}),
                    "neural_score",
                )["auroc"],
                structural_auroc=binary_metrics(
                    g.rename(columns={"binary_label_neural": "binary_label"}),
                    "structural_score",
                )["auroc"],
            )
        )
    return pd.DataFrame(records)


def summarize(table, replicates=10000, seed=20260922, cluster_levels=None):
    levels = sorted(
        table.cluster.unique() if cluster_levels is None else cluster_levels
    )
    draws = np.random.RandomState(seed).multinomial(
        len(levels), np.ones(len(levels)) / len(levels), size=replicates
    )
    rows = []
    samples = {}
    for (model, method), g in table.groupby(["model", "method"]):
        for metric in [
            "spearman",
            "excess_agreement",
            "neural_auroc",
            "structural_auroc",
        ]:
            valid = g[g[metric].notna()]
            grouped = (
                valid.groupby("cluster")[metric]
                .agg(["sum", "count"])
                .reindex(levels, fill_value=0)
            )
            denominator = draws @ grouped["count"].to_numpy()
            bootstrap = np.divide(
                draws @ grouped["sum"].to_numpy(),
                denominator,
                out=np.full(replicates, np.nan),
                where=denominator > 0,
            )
            samples[model, method, metric] = (float(valid[metric].mean()), bootstrap)
            if valid.empty:
                continue
            rows.append(
                dict(
                    model=model,
                    method=method,
                    metric=metric,
                    proteins=len(valid),
                    pairs=int(valid.pairs.sum()),
                    estimate=valid[metric].mean(),
                    ci_low=np.nanquantile(bootstrap, 0.025),
                    ci_high=np.nanquantile(bootstrap, 0.975),
                    valid_replicates=int(np.isfinite(bootstrap).sum()),
                )
            )
    contrasts = []
    # A contrast is valid only on identical eligible proteins; do not pair different supports.
    for (model, method, metric), (value, draw) in samples.items():
        if model in {"ligand", "protein"} or ("ligand", method, metric) not in samples:
            continue
        a = table[
            (table.model == model) & (table.method == method) & table[metric].notna()
        ]
        b = table[
            (table.model == "ligand") & (table.method == method) & table[metric].notna()
        ]
        shared = a[["uniprot", "cluster", metric]].merge(
            b[["uniprot", metric]],
            on="uniprot",
            validate="one_to_one",
            suffixes=("_a", "_b"),
        )
        if shared.empty:
            continue
        shared["delta"] = shared[metric + "_a"] - shared[metric + "_b"]
        stat = (
            shared.groupby("cluster")
            .delta.agg(["sum", "count"])
            .reindex(levels, fill_value=0)
        )
        denom = draws @ stat["count"].to_numpy()
        delta = np.divide(
            draws @ stat["sum"].to_numpy(),
            denom,
            out=np.full(replicates, np.nan),
            where=denom > 0,
        )
        contrasts.append(
            dict(
                model=model,
                method=method,
                metric=metric,
                contrast="model_minus_ligand",
                proteins=len(shared),
                estimate=shared.delta.mean(),
                ci_low=np.nanquantile(delta, 0.025),
                ci_high=np.nanquantile(delta, 0.975),
                valid_replicates=int(np.isfinite(delta).sum()),
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--structural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["RAW:RAW_D", "KDE_LEGACY:D+AR", "KDE_LEGACY:D+MW", "QNB:D+AR"],
    )
    args = parser.parse_args()
    neural = pd.read_csv(args.neural, sep="\t")
    structural = pd.read_csv(args.structural, sep="\t")
    structural = structural[structural.method.isin(args.methods)].copy()
    models = {"ligand", "protein", "c1", "c2", "c3", "d1", "d2", "d3"}
    if set(neural.model) != models or set(structural.method) != set(args.methods):
        raise ValueError(
            "Supply all eight neural models and the requested structural rankings"
        )
    keys = ["uniprot", "full_inchikey"]
    if (
        neural.duplicated(keys + ["model"]).any()
        or structural.duplicated(keys + ["method"]).any()
    ):
        raise ValueError(
            "Select one dataset, held-out regime and seed ensemble before matching"
        )
    matched = neural.merge(structural, on=keys, suffixes=("_neural", "_structural"))
    if not (matched.binary_label_neural == matched.binary_label_structural).all():
        raise ValueError("Reference label disagreement")
    matched = matched.rename(
        columns={
            "score_neural": "neural_score",
            "score_structural": "structural_score",
            "family_component_id_neural": "neural_family",
            "family_component_id_structural": "structural_family",
        }
    )
    complete = matched.groupby(keys).apply(
        lambda g: len(g) == len(models) * len(args.methods)
        and np.isfinite(g[["neural_score", "structural_score"]].to_numpy()).all()
    )
    shared = set(complete[complete].index)
    matched = matched[
        [
            tuple(row) in shared
            for row in matched[keys].itertuples(index=False, name=None)
        ]
    ].copy()
    if matched.empty:
        raise ValueError("No common complete pair set")
    if matched[["neural_family", "structural_family"]].isna().any().any():
        raise ValueError("Missing family membership")
    matched["cluster"] = joined_family_components(matched)
    valid = matched[
        np.isfinite(matched.neural_score) & np.isfinite(matched.structural_score)
    ]
    table = comparison_table(valid)
    summary, contrasts = summarize(table, cluster_levels=matched.cluster.unique())
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [
        ("matched_pairs", matched),
        ("protein_comparisons", table),
        ("summary", summary),
        ("paired_differences", contrasts),
    ]:
        data.to_csv(args.output / f"{name}.tsv.gz", sep="\t", index=False, na_rep="NA")


if __name__ == "__main__":
    main()
