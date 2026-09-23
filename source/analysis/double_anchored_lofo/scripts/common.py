"""Frozen double-anchored leave-one-family-out experiment, CPU-safe utilities."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
VERSION = 'double_anchored_lofo_fixed25_v1'
MODELS = ('ligand', 'protein', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3')
REGIMES = ('family_only', 'double_unseen')
SEEDS = (20260817, 20260818, 20260819)
TRAINING = dict(epochs=25, hidden_dim=256, heads=4, dropout=.30, lr=1e-4,
                weight_decay=1e-4, batch_size=6, eval_batch_size=8,
                c3_batch_size=1, c3_eval_batch_size=2, max_atoms=120,
                max_protein_residues=4096, gradient_clip=5.0,
                checkpoint_selection='fixed final epoch 25; no validation or test-based selection',
                training_weight='train-only equal protein-label group totals, then equal label totals',
                inference='FP32 sigmoid after mixed-precision logits; single-input deduplicated by exact input identity')
ATP = 'ZKHQWZAMYRWXGA-KQYNXXCUSA-N'
ADP = 'XTWYTFMLZFPYCI-KQYNXXCUSA-N'


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
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(str(temporary), str(path))


def read_data():
    return pd.read_csv(PACKAGE / 'data/COHORT.tsv.gz', sep='\t', dtype={'main_row_id': str})


def families(frame):
    return sorted(frame.family_component_id.astype(str).unique())


def partition(frame, regime, fold):
    if regime not in REGIMES:
        raise ValueError(regime)
    family = families(frame)[int(fold)]
    test = frame[frame.family_component_id == family].copy()
    training_candidates = frame[frame.family_component_id != family].copy()
    train = training_candidates.copy()
    if regime == 'double_unseen':
        train = train[~train.connectivity_key.isin(test.connectivity_key)].copy()
    excluded = training_candidates[~training_candidates.main_row_id.isin(train.main_row_id)].copy()
    for field in ('main_row_id', 'uniprot', 'family_component_id'):
        if set(train[field]) & set(test[field]):
            raise ValueError('Overlap: ' + field)
    test['unseen_compound'] = (~test.connectivity_key.isin(train.connectivity_key)).astype(int)
    if regime == 'double_unseen' and not test.unseen_compound.eq(1).all():
        raise ValueError('Ligand overlap')
    for part in (train, test):
        if set(part.binary_label) != {0, 1}:
            raise ValueError('Empty or single-class split')
    if len(train) + len(test) + len(excluded) != len(frame):
        raise ValueError('Partition accounting')
    return train, test, excluded


def jobs():
    return [(r, m, s, f) for r in REGIMES for m in MODELS for s in SEEDS for f in range(47)]


def fit_dir(job):
    regime, model, seed, fold = job
    return PACKAGE / 'gpu_output/fits' / regime / model / ('seed_%s' % seed) / ('fold_%02d' % fold)


def identity(job):
    regime, model, seed, fold = job
    return dict(cohort_arm='double_anchored', regime=regime, model=model,
                seed=int(seed), outer_fold=int(fold))


def verify_contract():
    c = json.loads((PACKAGE / 'validation/CPU_CONTRACT.json').read_text())
    if c['status'] != 'validated' or c['version'] != VERSION or c['training'] != TRAINING:
        raise ValueError('Invalid scientific contract')
    for name, expected in c['files'].items():
        if sha(PACKAGE / name) != expected:
            raise ValueError('Contract SHA256 mismatch: ' + name)
    if digest(c['files']) != c['manifest_digest']:
        raise ValueError('Manifest digest mismatch')
    return c


def compatible_report(job, run, hashes=True):
    """Strict completed-fit gate; an incomplete fit restarts from epoch 1."""
    directory = fit_dir(job)
    try:
        r = json.loads((directory / 'FIT_REPORT.json').read_text())
        if (r['status'] != 'validated' or r['run_fingerprint'] != run['run_fingerprint']
                or r['training'] != TRAINING or r['final_epoch'] != 25
                or r['validation_rows'] != 0 or r['model_version'] != 'role_complete_matrix_v2'
                or any(r[k] != v for k, v in identity(job).items())):
            return None
        for name in ('final.pt', 'predictions.tsv.gz', 'history.tsv'):
            path = directory / name
            if not path.is_file() or (hashes and sha(path) != r['sha256'][name]):
                return None
        return r
    except (OSError, ValueError, KeyError, TypeError):
        return None


def local_path(path, root):
    for prefix in ('/disk1/11.HS_allostery/', '/disk9/13.Heesu_Allostery/', '/shared_data/11.HS_allostery/'):
        if str(path).startswith(prefix):
            return str(root / str(path)[len(prefix):])
    return str(path)


def input_identity(model, frame):
    if model == 'protein':
        return frame.uniprot.astype(str)
    if model == 'ligand':
        return frame.ligand_embedding_path.astype(str)
    return frame.main_row_id.astype(str)
