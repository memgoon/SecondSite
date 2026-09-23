"""Tie-aware CPU metrics shared with the explicit double-unseen analysis."""
import numpy as np
import pandas as pd

def auc(y, score):
    y = np.asarray(y, dtype=int)
    p = int(y.sum())
    n = len(y) - p
    if not p or not n:
        return np.nan
    rank = pd.Series(np.asarray(score)).rank(method='average').to_numpy()
    return float((rank[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def ap(y, score):
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    if not y.sum() or y.sum() == len(y):
        return np.nan
    idx = np.argsort(-s, kind='mergesort')
    y, s = y[idx], s[idx]
    end = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[end]
    return float(np.sum(np.diff(np.r_[0., tp / y.sum()]) * tp / (end + 1)))


def stats(frame, field=None):
    groups = [(None, frame)] if field is None else frame.groupby(['outer_fold', field], sort=False)
    values, rows, skipped = [], 0, 0
    for _, x in groups:
        if x.binary_label.nunique() != 2:
            skipped += 1
            continue
        y, s = x.binary_label.to_numpy(), x.p_allosteric.to_numpy()
        a, b = ap(y, s), ap(1 - y, 1 - s)
        values.append([auc(y, s), a, (a + b) / 2])
        rows += len(x)
    v = np.mean(values, axis=0) if values else [np.nan] * 3
    return dict(auroc=float(v[0]), allosteric_ap=float(v[1]), symmetric_ap=float(v[2]),
                n_rows_input=len(frame), n_rows_used=rows, n_groups_used=len(values), n_groups_skipped=skipped)


def resample_family_statistics(sums, counts, draw):
    """Preserve every sampled family occurrence for nested protein-macro means."""
    denominator = np.asarray(counts)[draw].sum(axis=1)
    numerator = np.asarray(sums)[draw].sum(axis=1)
    valid = denominator > 0
    return numerator[valid] / denominator[valid], denominator[valid]

