"""CPU-safe frozen split and metric primitives (no PyTorch/sklearn dependency)."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
ARMS = ('every_pair', 'protein_anchored')
FILES = {'every_pair': 'EVERY_PAIR.tsv.gz', 'protein_anchored': 'PROTEIN_ANCHORED.tsv.gz',
         'protein_ligand_role_complete': 'PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz'}
MODELS = ('ligand', 'protein', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3')
SEEDS = (20260817, 20260818, 20260819)
VERSION = 'explicit_double_unseen_v1'
TRAINING = dict(epochs=25, patience=5, min_epochs=1, hidden_dim=256, heads=4,
                dropout=0.30, lr=1e-4, weight_decay=1e-4, batch_size=6,
                eval_batch_size=8, full_bidirectional_batch_size=1,
                full_bidirectional_eval_batch_size=2, max_atoms=120,
                max_protein_residues=4096)
SELECTION = ('Same model-aware rule as original family-held-out baseline: ligand-only '
             'within-protein AUROC; protein-only and joint models within-ligand AUROC; '
             'minimum 8 two-class groups, otherwise pooled symmetric AP. No test selection.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(str(tmp), str(path))


def partition(frame, fold):
    """Test-first purged 3-way split; no label or model-score dependent selection."""
    vf = (int(fold) + 1) % 5
    test = frame[(frame.matrix_family_fold == fold) & (frame.matrix_evaluation_eligible == 1)].copy()
    val0 = frame[(frame.matrix_family_fold == vf) & (frame.matrix_evaluation_eligible == 1)].copy()
    train0 = frame[~frame.matrix_family_fold.isin([fold, vf])].copy()
    val = val0[~val0.connectivity_key.isin(test.connectivity_key)].copy()
    blocked = set(test.connectivity_key) | set(val.connectivity_key)
    train = train0[~train0.connectivity_key.isin(blocked)].copy()
    used = set(train.main_row_id) | set(val.main_row_id) | set(test.main_row_id)
    excluded = frame[~frame.main_row_id.isin(used)].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError('Empty partition')
    for field in ('main_row_id', 'uniprot', 'family_component_id', 'connectivity_key'):
        sets = [set(x[field].astype(str)) for x in (train, val, test)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError('Overlap in ' + field)
    for x in (train, val, test):
        if set(x.binary_label.astype(int)) != {0, 1}:
            raise ValueError('Single-class partition')
    if len(used) + len(excluded) != len(frame):
        raise ValueError('Partition accounting failure')
    return train, val, test, excluded


def read_frame(arm):
    return pd.read_csv(PACKAGE / 'data' / FILES[arm], sep='\t', dtype={'main_row_id': str})


def verify_contract():
    c = json.loads((PACKAGE / 'validation/CPU_CONTRACT.json').read_text())
    if c['status'] != 'validated' or c['version'] != VERSION:
        raise ValueError('Contract not validated')
    for name, expected in c['files'].items():
        if sha(PACKAGE / name) != expected:
            raise ValueError('Contract hash mismatch: ' + name)
    return c


def expected_jobs():
    return [(a, m, s, f) for a in ARMS for m in MODELS for s in SEEDS for f in range(5)]


def fit_dir(arm, model, seed, fold):
    return PACKAGE / 'gpu_output/benchmark/fits' / arm / 'double_unseen' / model / ('seed_%s' % seed) / ('fold_%s' % fold)


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
