"""Local CPU matched-row comparison with multiplicity-preserving paired bootstrap."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from common import (PACKAGE, MODELS, auc, stats, digest, write_json, verify_contract,
                    read_frame, sha, resample_family_statistics)
from validate_fits import collect


def align_pair(left, right):
    left = left.sort_values('main_row_id').reset_index(drop=True)
    right = right.sort_values('main_row_id').reset_index(drop=True)
    assert not left.main_row_id.duplicated().any() and len(left) == len(right)
    for col in ('main_row_id', 'binary_label', 'uniprot', 'outer_fold', 'family_component_id'):
        assert left[col].astype(str).tolist() == right[col].astype(str).tolist(), col
    return left, right


def protein_table(frame):
    rows = []
    for (fold, protein), g in frame.groupby(['outer_fold', 'uniprot'], sort=True):
        if g.binary_label.nunique() != 2:
            continue
        assert g.family_component_id.nunique() == 1
        rows.append(dict(outer_fold=fold, uniprot=protein, family=str(g.family_component_id.iloc[0]),
                         rows=len(g), value=auc(g.binary_label, g.p_allosteric)))
    return pd.DataFrame(rows, columns=['outer_fold', 'uniprot', 'family', 'rows', 'value'])


def prepare_pooled(frame, families):
    ordered = frame.sort_values('p_allosteric', kind='mergesort')
    scores = ordered.p_allosteric.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    family_index = {x: i for i, x in enumerate(families)}
    indices = ordered.family_component_id.astype(str).map(family_index).to_numpy(dtype=int)
    return indices, ordered.binary_label.to_numpy(dtype=float), starts


def weighted_pooled(prepared, multiplicities):
    """Exact weighted AUROC including ties; equivalent to duplicating sampled rows."""
    indices, labels, starts = prepared
    weights = multiplicities[:, indices]
    positive = np.add.reduceat(weights * labels, starts, axis=1)
    negative = np.add.reduceat(weights * (1-labels), starts, axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    numerator = (positive * (np.cumsum(negative, axis=1) - .5*negative)).sum(axis=1)
    values = np.full(len(denominator), np.nan)
    valid = denominator > 0
    values[valid] = numerator[valid] / denominator[valid]
    return values, weights.sum(axis=1)


def contrast_job(job):
    metadata, left, right, replicates = job
    left, right = align_pair(left, right)
    metric = metadata.get('metric', 'within_protein')
    families = sorted(set(left.family_component_id.astype(str)))
    rng = np.random.RandomState(int(digest(metadata)[:8], 16))
    if metric == 'within_protein':
        x, y = protein_table(left), protein_table(right)
        x = x.merge(y, on=['outer_fold', 'uniprot', 'family', 'rows'],
                    suffixes=('_test', '_reference'), validate='one_to_one')
        assert len(x) == len(protein_table(left)) == len(protein_table(right))
        if not len(x):
            raise ValueError('No supported within-protein groups')
        x['delta'] = x.value_test - x.value_reference
        groups = x.groupby('family').agg(delta_sum=('delta', 'sum'), count=('delta', 'size'),
                                        row_count=('rows', 'sum')).reindex(families).fillna(0.)
        sums, counts = groups.delta_sum.to_numpy(), groups['count'].to_numpy()
        point = float(x.delta.mean())
        observed_left, observed_right = float(x.value_test.mean()), float(x.value_reference.mean())
        rows_used, groups_used = int(x.rows.sum()), len(x)
    elif metric == 'pooled':
        prepared_left, prepared_right = prepare_pooled(left, families), prepare_pooled(right, families)
        observed_left, observed_right = auc(left.binary_label, left.p_allosteric), auc(right.binary_label, right.p_allosteric)
        point = observed_left - observed_right
        rows_used, groups_used = len(left), 1
    else:
        raise ValueError(metric)
    values, nused, usedrows = [], [], []
    for start in range(0, replicates, 128):
        draw = rng.randint(0, len(families), size=(min(128, replicates-start), len(families)))
        if metric == 'within_protein':
            denominator = counts[draw].sum(axis=1)
            valid = denominator > 0
            values.extend((sums[draw].sum(axis=1)[valid] / denominator[valid]).tolist())
            nused.extend(denominator[valid].tolist())
            usedrows.extend(groups.row_count.to_numpy()[draw].sum(axis=1)[valid].tolist())
        else:
            mult = np.zeros((len(draw), len(families)), dtype=float)
            for i, d in enumerate(draw):
                mult[i] = np.bincount(d, minlength=len(families))
            a, rows = weighted_pooled(prepared_left, mult)
            b, _ = weighted_pooled(prepared_right, mult)
            valid = np.isfinite(a) & np.isfinite(b)
            values.extend((a-b)[valid].tolist())
            usedrows.extend(rows[valid].tolist())
            nused.extend([1.] * int(valid.sum()))
    if len(values) < .95 * replicates:
        raise RuntimeError('Too few valid bootstrap replicates: ' + str(metadata))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return dict(metadata, metric=metric, delta=point, ci_low=float(lo), ci_high=float(hi),
        test_auroc=observed_left, reference_auroc=observed_right,
        n_rows_input=len(left), n_rows_used=rows_used, n_groups_used=groups_used,
        n_families=len(families), bootstrap_replicates=replicates, valid_replicates=len(values),
        invalid_replicates=replicates-len(values),
        bootstrap_fraction_delta_gt_zero=float(np.mean(np.asarray(values) > 0)),
        bootstrap_groups_used_min=float(min(nused)), bootstrap_groups_used_max=float(max(nused)),
        bootstrap_rows_used_min=float(min(usedrows)), bootstrap_rows_used_max=float(max(usedrows)),
        inference_scope='pointwise 95% CI; 44 comparisons not multiplicity-adjusted; no post hoc winner claim',
        cluster='family_component_id', multiplicity_preserved=True,
        model_seed_handling='mean probability over 3 frozen seeds; CI conditional on fitted models')


def make_jobs(ensemble, baseline, ring_ids, replicates):
    jobs = []
    for universe in ('all_test', 'ring_scaffold_test'):
        e = ensemble if universe == 'all_test' else ensemble[ensemble.main_row_id.isin(ring_ids)]
        b = baseline if universe == 'all_test' else baseline[baseline.main_row_id.isin(ring_ids)]
        for model in MODELS:
            g, reference = e[e.model == model], b[b.model == model]
            for metric in ('pooled', 'within_protein'):
                jobs.append((dict(cohort_arm='every_pair', universe=universe, metric=metric,
                    contrast='murcko_minus_connectivity_double_held_out', model=model, comparator=model),
                    g, reference, replicates))
            if model not in ('ligand', 'protein'):
                jobs.append((dict(cohort_arm='every_pair', universe=universe, metric='within_protein',
                    contrast='joint_minus_ligand_under_murcko_exclusion', model=model, comparator='ligand'),
                    g, e[e.model == 'ligand'], replicates))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--bootstrap-replicates', type=int, default=10000)
    args = p.parse_args()
    if args.workers < 1 or args.bootstrap_replicates != 10000:
        raise ValueError('Positive workers and frozen 10000 replicates required')
    c = verify_contract()
    raw = collect()
    out = PACKAGE / 'cpu_output'
    out.mkdir(exist_ok=True)
    fields = ['cohort_arm', 'regime', 'model', 'outer_fold', 'main_row_id', 'uniprot',
              'family_component_id', 'full_inchikey', 'connectivity_key', 'binary_label']
    assert raw.groupby(fields).seed.nunique().eq(3).all()
    ensemble = raw.groupby(fields, as_index=False).p_allosteric.mean()
    assert len(ensemble) == 8*4637
    base = pd.read_csv(PACKAGE / 'data/BASELINE_ENSEMBLE.tsv.gz', sep='\t', dtype={'main_row_id': str})
    frame = read_frame('every_pair')
    ring_ids = set(frame.loc[frame.scaffold_available == 1, 'main_row_id'])
    ensemble['scaffold_available'] = ensemble.main_row_id.isin(ring_ids).astype(int)
    ensemble.to_csv(out / 'ENSEMBLE_PREDICTIONS.tsv.gz', sep='\t', index=False)
    metrics = []
    for universe in ('all_test', 'ring_scaffold_test', 'acyclic_test'):
        def subset(d):
            if universe == 'all_test':
                return d
            keep = d.main_row_id.isin(ring_ids)
            return d[keep if universe == 'ring_scaffold_test' else ~keep]
        for label, data in [('new_seed_ensemble', ensemble.assign(seed='ensemble')),
                            ('existing_seed_ensemble', base.assign(seed='ensemble')),
                            ('new_individual_seed', raw)]:
            for (arm, regime, model, seed), g in subset(data).groupby(['cohort_arm', 'regime', 'model', 'seed']):
                for metric, field in [('pooled', None), ('within_protein', 'uniprot'),
                    ('within_ligand', 'connectivity_key'), ('family_macro', 'family_component_id')]:
                    metrics.append(dict(source=label, cohort_arm=arm, regime=regime, model=model,
                        seed=seed, metric=metric, universe=universe, **stats(g, field)))
    pd.DataFrame(metrics).to_csv(out / 'METRICS.tsv', sep='\t', index=False)
    jobs = make_jobs(ensemble, base, ring_ids, args.bootstrap_replicates)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        comparisons = []
        for i, result in enumerate(pool.map(contrast_job, jobs), 1):
            comparisons.append(result)
            print('Bootstrap %d/%d: %s %s %s' % (i, len(jobs), result['model'], result['metric'], result['universe']), flush=True)
    pd.DataFrame(comparisons).to_csv(out / 'PAIRED_BOOTSTRAP.tsv', sep='\t', index=False)
    summary = pd.read_csv(PACKAGE / 'gpu_output/FIT_SUMMARY.tsv', sep='\t')
    stability = summary.groupby(['cohort_arm', 'model']).agg(
        n_fits=('best_epoch', 'size'), median_best_epoch=('best_epoch', 'median'),
        epoch1_fraction=('best_epoch', lambda x: float((x == 1).mean())),
        fallback_fraction=('checkpoint_selection_fallback_at_best_epoch', 'mean')).reset_index()
    stability['epoch1_review_flag'] = stability.epoch1_fraction >= 1/3
    stability.to_csv(out / 'TRAINING_STABILITY.tsv', sep='\t', index=False)
    names = ['METRICS.tsv', 'PAIRED_BOOTSTRAP.tsv', 'ENSEMBLE_PREDICTIONS.tsv.gz', 'TRAINING_STABILITY.tsv']
    write_json(out / 'VALIDATION.json', dict(status='validated', fits=120,
        prediction_rows=len(raw), ensemble_rows=len(ensemble), test_rows=4637,
        ring_test_rows=int(((frame.matrix_evaluation_eligible == 1) & (frame.scaffold_available == 1)).sum()),
        primary='paired change of ligand-only pooled AUROC under added scaffold exclusion',
        paired_conditional_lift='joint minus ligand-only within-protein AUROC',
        comparisons=len(comparisons), replicates=10000, cluster='family_component_id',
        expected_fallback_fits=c['expected_fallback_fits'],
        actual_fallback_fits=int(summary.checkpoint_selection_fallback_at_best_epoch.sum()),
        input_manifest_digest=c['manifest_digest'], cpu_only_analysis=True, no_sklearn_required=True,
        output_sha256={x: sha(out/x) for x in names},
        limitation='Training size and validation composition change; no size-matched control; acyclic molecules lack Murcko scaffold'))
    print('PASS: 120 fits; 44 paired bootstrap comparisons; CPU analysis complete.')


if __name__ == '__main__':
    main()
