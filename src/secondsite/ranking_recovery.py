"""Recovery of assay annotations and literature-supported pairs in fixed rankings."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def weights(scores, fraction):
    scores = np.asarray(scores, float)
    if not len(scores) or not np.isfinite(scores).all() or not 0 < fraction <= 1:
        raise ValueError("Finite scores and a fraction in (0,1] are required")
    n = len(scores)
    k = max(1, int(np.ceil(n * fraction)))
    threshold = np.partition(scores, n - k)[n - k]
    above, ties = scores > threshold, scores == threshold
    return above.astype(float) + ties * (k - int(above.sum())) / int(ties.sum())


def interval(values):
    valid = np.isfinite(values)
    count = int(valid.sum())
    if count < 0.95 * len(values):
        return np.nan, np.nan, count
    return *np.quantile(values[valid], [0.025, 0.975]), count


def evaluate(
    metadata, predictions, evidence_columns, fractions=(0.001, 0.01, 0.05), reps=10000
):
    """All models share the same rows and protein-cluster resampling draws."""
    metadata = metadata.sort_values("main_row_id").reset_index(drop=True)
    if (
        metadata.main_row_id.duplicated().any()
        or metadata.uniprot.isna().any()
        or metadata.empty
        or reps < 1
    ):
        raise ValueError("Invalid evaluation population")
    for column in evidence_columns:
        if not metadata[column].isin([0, 1]).all():
            raise ValueError("Evidence indicators must be 0/1: " + column)
    proteins, codes = np.unique(metadata.uniprot, return_inverse=True)
    ng = len(proteins)
    sizes = np.bincount(codes, minlength=ng).astype(float)
    rng = np.random.default_rng(20261014)
    draws = rng.integers(0, ng, (reps, ng))
    counts = np.zeros((reps, ng), dtype=np.int32)
    np.add.at(counts, (np.repeat(np.arange(reps), ng), draws.ravel()), 1)
    groups = [g.index.to_numpy() for _, g in metadata.groupby("uniprot", sort=True)]
    tasks, rows = {}, []
    for (cohort, model), frame in predictions.groupby(["cohort", "model"]):
        frame = frame.sort_values("main_row_id").reset_index(drop=True)
        if frame.main_row_id.tolist() != metadata.main_row_id.tolist():
            raise ValueError("Model evaluation rows must match exactly")
        score = frame.score.to_numpy(float)
        for fraction in fractions:
            for mode in ["pooled", "within_protein"]:
                w = (
                    weights(score, fraction)
                    if mode == "pooled"
                    else np.zeros(len(score))
                )
                if mode == "within_protein":
                    for ix in groups:
                        w[ix] = weights(score[ix], fraction)
                selected = np.bincount(codes, weights=w, minlength=ng)
                for label in evidence_columns:
                    tag = metadata[label].to_numpy(float)
                    positives = np.bincount(codes, weights=tag, minlength=ng)
                    hits = np.bincount(codes, weights=w * tag, minlength=ng)
                    expected = selected * positives / sizes
                    stats = np.column_stack(
                        [sizes, positives, selected, hits, expected]
                    )
                    total, boot = stats.sum(axis=0), counts @ stats
                    den = (
                        total[2] * total[1] / total[0] if mode == "pooled" else total[4]
                    )
                    bden = (
                        boot[:, 2] * boot[:, 1] / boot[:, 0]
                        if mode == "pooled"
                        else boot[:, 4]
                    )
                    recall = total[3] / total[1] if total[1] > 0 else np.nan
                    enrichment = total[3] / den if den > 0 else np.nan
                    brecall = np.divide(
                        boot[:, 3],
                        boot[:, 1],
                        out=np.full(reps, np.nan),
                        where=boot[:, 1] > 0,
                    )
                    benrichment = np.divide(
                        boot[:, 3], bden, out=np.full(reps, np.nan), where=bden > 0
                    )
                    lo, hi, n = interval(brecall)
                    elo, ehi, en = interval(benrichment)
                    key = (cohort, model, mode, fraction, label)
                    tasks[key] = recall, enrichment, brecall, benrichment
                    rows.append(
                        dict(
                            cohort=cohort,
                            model=model,
                            ranking=mode,
                            top_fraction=fraction,
                            label_set=label,
                            universe_pairs=len(metadata),
                            proteins=ng,
                            positive_pairs=int(total[1]),
                            selected_pairs=total[2],
                            recovered_positive_pairs=total[3],
                            random_expected_hits=den,
                            recall=recall,
                            recall_ci_low=lo,
                            recall_ci_high=hi,
                            recall_valid_replicates=n,
                            enrichment=enrichment,
                            enrichment_ci_low=elo,
                            enrichment_ci_high=ehi,
                            enrichment_valid_replicates=en,
                        )
                    )
    paired = []
    for (cohort, model, mode, fraction, label), value in tasks.items():
        comparators = [(cohort, b) for b in ["ligand", "protein"] if b != model]
        if cohort != "every_pair":
            comparators.append(("every_pair", model))
        for ref_cohort, baseline in comparators:
            key = ref_cohort, baseline, mode, fraction, label
            if key not in tasks:
                continue
            ref = tasks[key]
            for i, metric in enumerate(["recall", "enrichment"]):
                lo, hi, n = interval(value[i + 2] - ref[i + 2])
                paired.append(
                    dict(
                        cohort=cohort,
                        model=model,
                        reference_cohort=ref_cohort,
                        baseline=baseline,
                        ranking=mode,
                        top_fraction=fraction,
                        label_set=label,
                        metric=metric,
                        difference=value[i] - ref[i],
                        ci_low=lo,
                        ci_high=hi,
                        valid_replicates=n,
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--evidence-columns", nargs="+", required=True)
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.001, 0.01, 0.05]
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata, sep="\t", dtype={"main_row_id": str})
    predictions = pd.concat(
        [
            pd.read_csv(p, sep="\t", dtype={"main_row_id": str})
            for p in args.predictions
        ],
        ignore_index=True,
    )
    results, paired = evaluate(
        metadata, predictions, args.evidence_columns, args.fractions, args.replicates
    )
    args.output.mkdir(parents=True, exist_ok=False)
    results.to_csv(args.output / "recovery.tsv", sep="\t", index=False)
    paired.to_csv(args.output / "paired_differences.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
