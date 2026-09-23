#!/usr/bin/env python3
"""CPU-only, paired family-cluster bootstrap of frozen 395-row OOF predictions."""
import os
# Parallelism is across bootstrap chunks, not nested BLAS pools.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'double_anchored_lofo'
MODELS = ['ligand', 'protein', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3']
REGIMES = ['family_only', 'double_unseen']


def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cluster_matrix(y, scores, clusters, n):
    """U[i,j] counts positive-family i wins over negative-family j (ties=1/2)."""
    positive, negative = y == 1, y == 0
    a, b = scores[positive, None], scores[None, negative]
    wins = (a > b).astype(float) + 0.5 * (a == b)
    result = np.zeros((n, n), dtype=float)
    np.add.at(result, (clusters[positive, None], clusters[None, negative]), wins)
    return result


def evaluate_counts(counts, matrices, pos, neg):
    denominator = counts.dot(pos) * counts.dot(neg)
    values = np.full((len(counts), len(matrices)), np.nan)
    valid = denominator > 0
    for k, matrix in enumerate(matrices):
        numerator = np.sum(counts.dot(matrix) * counts, axis=1)
        values[valid, k] = numerator[valid] / denominator[valid]
    return values


def brute_auc(y, score):
    a, b = score[y == 1, None], score[None, y == 0]
    return float(np.mean((a > b) + 0.5 * (a == b))) if a.size and b.size else np.nan


def self_test():
    # Pooled AUROC is not an average of family AUROCs. Check the weighted
    # cross-family pair counts against literal duplicated rows, including ties.
    rng = np.random.RandomState(923)
    y = np.array([1, 0, 1, 0, 1, 0, 1])
    c = np.array([0, 0, 1, 1, 2, 2, 3])
    s = np.array([.9, .4, .4, .8, .7, .7, .1])
    u = cluster_matrix(y, s, c, 4)
    pos = np.bincount(c[y == 1], minlength=4)
    neg = np.bincount(c[y == 0], minlength=4)
    weights = np.vstack(([2, 1, 0, 0], [0, 0, 0, 4],
                         rng.multinomial(4, [0.25] * 4, size=100)))
    actual = evaluate_counts(weights, [u], pos, neg)[:, 0]
    expected = []
    for w in weights:
        rows = np.repeat(np.arange(len(y)), w[c])
        expected.append(brute_auc(y[rows], s[rows]))
    require(np.allclose(actual, expected, equal_nan=True, atol=1e-12),
            'Bootstrap multiplicity/tie/degeneracy test failed')
    require(abs(actual[0] - evaluate_counts(np.array([[1, 1, 0, 0]]),
                [u], pos, neg)[0, 0]) > 1e-3, 'Test must detect collapsed duplicates')


def load_inputs(source):
    root = source / 'local_analysis'
    validation = root / 'VALIDATION.json'
    contract = json.loads(validation.read_text())
    require(contract['status'] == 'validated', 'Source analysis not validated')
    paths = [validation]
    for name in ['OOF_SEED_ENSEMBLE.tsv.gz', 'METRICS.tsv']:
        p = root / name
        require(sha(p) == contract['files'][name], 'Source checksum mismatch: ' + name)
        paths.append(p)
    data = pd.read_csv(root / 'OOF_SEED_ENSEMBLE.tsv.gz', sep='\t')
    metrics = pd.read_csv(root / 'METRICS.tsv', sep='\t')
    require(len(data) == 2 * 8 * 395, 'Unexpected source row count')
    require(set(data.regime) == set(REGIMES) and set(data.model) == set(MODELS),
            'Unexpected models/regimes')
    meta = ['main_row_id', 'binary_label', 'uniprot', 'family_component_id',
            'full_inchikey', 'connectivity_key', 'outer_fold']
    reference = data[(data.regime == REGIMES[0]) & (data.model == MODELS[0])]
    reference = reference.sort_values('main_row_id').reset_index(drop=True)
    require(len(reference) == 395 and reference.main_row_id.nunique() == 395,
            'Reference row IDs not unique')
    require(reference.uniprot.nunique() == 98 and reference.family_component_id.nunique() == 47,
            'Unexpected protein/family support')
    require(set(reference.binary_label) == {0, 1} and reference.binary_label.sum() == 143,
            'Unexpected labels')
    require(reference.groupby('uniprot').family_component_id.nunique().max() == 1,
            'Protein spans families')
    require(reference.groupby('family_component_id').outer_fold.nunique().max() == 1
            and reference.outer_fold.nunique() == 47, 'Family/fold mismatch')
    y = reference.binary_label.to_numpy(dtype=int)
    clusters, families = pd.factorize(reference.family_component_id, sort=True)
    pos = np.bincount(clusters[y == 1], minlength=len(families))
    neg = np.bincount(clusters[y == 0], minlength=len(families))
    matrices, keys = [], []
    for regime in REGIMES:
        for model in MODELS:
            z = data[(data.regime == regime) & (data.model == model)]
            z = z.sort_values('main_row_id').reset_index(drop=True)
            require(z[meta].equals(reference[meta]), 'Nonidentical rows/metadata: ' + regime + '/' + model)
            s = z.p_allosteric.to_numpy(dtype=float)
            require(np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all(), 'Invalid probability')
            value = brute_auc(y, s)
            old = metrics[(metrics.kind == 'seed_ensemble') & (metrics.metric == 'pooled')
                          & (metrics.regime == regime) & (metrics.model == model)]
            require(len(old) == 1 and abs(value - float(old.auroc.iloc[0])) < 1e-12,
                    'Pooled point estimate does not reproduce source')
            matrices.append(cluster_matrix(y, s, clusters, len(families)))
            keys.append((regime, model))
    return np.array(matrices), pos, neg, keys, paths, contract


def worker(task):
    counts, matrices, pos, neg = task
    return evaluate_counts(counts, matrices, pos, neg)


def summary(values, estimate, requested):
    finite = np.isfinite(values)
    v = values[finite]
    ok = len(v) >= int(np.ceil(requested * .95))
    return dict(estimate=float(estimate), ci_low=float(np.quantile(v, .025)) if ok else None,
                ci_high=float(np.quantile(v, .975)) if ok else None,
                bootstrap_fraction_gt_zero=float(np.mean(v > 0)) if ok else None,
                replicates_requested=requested, valid_replicates=int(finite.sum()),
                invalid_replicates=int((~finite).sum()),
                status='validated' if ok else 'insufficient_valid_replicates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--workers', type=int, default=int(os.environ.get('BOOTSTRAP_WORKERS', '8')))
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--self-test-only', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test_only:
        print('PASS: exact pooled AUROC vs literal cluster resampling, ties and missing class.')
        return
    require(args.workers > 0 and args.replicates > 0, 'Positive workers/replicates required')
    require(not args.output.exists(), 'Output already exists; choose a new --output (no overwrite)')
    matrices, pos, neg, keys, paths, contract = load_inputs(args.source)
    point = evaluate_counts(np.ones((1, len(pos))), matrices, pos, neg)[0]
    # One frozen RNG stream makes output independent of worker count/chunking.
    draw = np.random.RandomState(args.seed).randint(0, len(pos), size=(args.replicates, len(pos)))
    counts = np.zeros((args.replicates, len(pos)), dtype=float)
    np.add.at(counts, (np.arange(args.replicates)[:, None], draw), 1)
    chunks = [counts[start:start + 256] for start in range(0, len(counts), 256)]
    workers = min(args.workers, len(chunks))
    print('Validated 395 identical rows, 47 families, 16 fitted ensembles. '
          'Bootstrap: {} replicates, {} workers.'.format(args.replicates, workers), flush=True)
    tasks = [(chunk, matrices, pos, neg) for chunk in chunks]
    results = []
    if workers == 1:
        results = [worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, result in enumerate(pool.map(worker, tasks), 1):
                results.append(result)
                if k % 10 == 0 or k == len(tasks):
                    print('Bootstrap chunks: {}/{}'.format(k, len(tasks)), flush=True)
    boot = np.concatenate(results)
    absolute, contrasts, split = [], [], []
    for i, (regime, model) in enumerate(keys):
        row = summary(boot[:, i], point[i], args.replicates)
        row.pop('bootstrap_fraction_gt_zero')
        row['bootstrap_fraction_gt_chance'] = float(np.mean(boot[np.isfinite(boot[:, i]), i] > .5))
        absolute.append(dict(regime=regime, model=model, metric='pooled_auroc',
                             n_rows=395, n_proteins=98, n_families=47, **row))
        if model not in ['ligand', 'protein']:
            for baseline in ['ligand', 'protein']:
                j = keys.index((regime, baseline))
                contrasts.append(dict(regime=regime, model=model, baseline=baseline,
                    metric='pooled_auroc_difference', **summary(boot[:, i] - boot[:, j], point[i] - point[j], args.replicates)))
    for model in MODELS:
        i, j = keys.index(('double_unseen', model)), keys.index(('family_only', model))
        split.append(dict(model=model, contrast='double_unseen_minus_family_only',
            metric='pooled_auroc_difference', **summary(boot[:, i] - boot[:, j], point[i] - point[j], args.replicates)))
    args.output.mkdir(parents=True)
    for name, rows in [('POOLED_AUROC_CI.tsv', absolute), ('PAIR_VS_SINGLE_INPUT_CI.tsv', contrasts),
                       ('SPLIT_DIFFERENCE_CI.tsv', split)]:
        pd.DataFrame(rows).to_csv(args.output / name, sep='\t', index=False)
    np.savez_compressed(str(args.output / 'BOOTSTRAP_REPLICATES.npz'),
                        pooled_auroc=boot, family_counts=counts.astype(np.int16),
                        keys=np.array(['/'.join(k) for k in keys]))
    ok = all(r['status'] == 'validated' for r in absolute + contrasts + split)
    outputs = ['POOLED_AUROC_CI.tsv', 'PAIR_VS_SINGLE_INPUT_CI.tsv',
               'SPLIT_DIFFERENCE_CI.tsv', 'BOOTSTRAP_REPLICATES.npz']
    manifest = dict(status='validated' if ok else 'failed',
        scope='secondary pooled analysis; original within-protein primary remains unchanged',
        no_training_executed=True, source_run_fingerprint=contract['run_fingerprint'],
        rows=395, proteins=98, families=47, absolute_endpoints=16, paired_contrasts=32,
        bootstrap_cluster='family_component_id', paired_across_all_models_and_regimes=True,
        multiplicity='exact weighted cross-family positive-negative pairs, ties count 0.5',
        self_test_passed=True, minimum_valid_fraction=.95, seed=args.seed,
        replicates=args.replicates, workers=workers,
        interval='95% percentile, conditional on frozen mean-score three-seed ensembles',
        limitations=['Not training-size causal evidence; not training-seed uncertainty.',
                     'No multiple-comparison adjustment; bootstrap fractions are not p-values or posterior probabilities.',
                     'Pooled OOF AUROC includes cross-protein and cross-fold score comparisons.'],
        source_hashes={str(p.resolve()): sha(p) for p in paths}, script_sha256=sha(Path(__file__)),
        files={name: sha(args.output / name) for name in outputs})
    (args.output / 'VALIDATION.json').write_text(json.dumps(manifest, indent=2) + '\n')
    require(ok, 'Insufficient valid bootstrap replicates; inspect VALIDATION.json')
    print('Complete, validated: ' + str(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
