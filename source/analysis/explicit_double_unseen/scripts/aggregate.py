"""Local CPU metrics and paired family bootstrap on identical test rows."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import pandas as pd
from common import PACKAGE, ARMS, MODELS, auc, stats, digest, write_json, verify_contract, resample_family_statistics
from validate_fits import collect


def protein_table(frame):
    rows = []
    for (fold, protein), g in frame.groupby(['outer_fold', 'uniprot'], sort=True):
        if g.binary_label.nunique() != 2:
            continue
        assert g.family_component_id.nunique() == 1
        rows.append(dict(outer_fold=fold, uniprot=protein, family=g.family_component_id.iloc[0],
                         rows=len(g), value=auc(g.binary_label, g.p_allosteric)))
    return pd.DataFrame(rows)


def contrast_job(job):
    metadata, left, right, replicates = job
    x, y = protein_table(left), protein_table(right)
    x = x.merge(y, on=['outer_fold', 'uniprot', 'family', 'rows'], suffixes=('_test', '_reference'), validate='one_to_one')
    assert len(x) == len(protein_table(left)) == len(protein_table(right))
    x['delta'] = x.value_test - x.value_reference
    groups = x.groupby('family').agg(delta_sum=('delta', 'sum'), count=('delta', 'size'))
    # Sampling all held-out families, including those without a defined statistic.
    families = sorted(set(left.family_component_id.astype(str)))
    groups.index = groups.index.astype(str)
    groups = groups.reindex(families).fillna(0.)
    sums, counts = groups.delta_sum.to_numpy(), groups['count'].to_numpy()
    rng = np.random.RandomState(int(digest(metadata)[:8], 16))
    values, nused = [], []
    for start in range(0, replicates, 256):
        draw = rng.randint(0, len(groups), size=(min(256, replicates-start), len(groups)))
        sampled, denominator = resample_family_statistics(sums, counts, draw)
        values.extend(sampled.tolist())
        nused.extend(denominator.tolist())
    if len(values) < .95 * replicates:
        raise RuntimeError('Too few valid bootstrap replicates: ' + str(metadata))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return dict(metadata, delta=float(x.delta.mean()), ci_low=float(lo), ci_high=float(hi),
                test_auroc=float(x.value_test.mean()), reference_auroc=float(x.value_reference.mean()),
                n_rows_used=int(x.rows.sum()), n_groups_used=len(x), n_families=len(groups),
                bootstrap_replicates=replicates, valid_replicates=len(values),
                invalid_replicates=replicates-len(values),
                bootstrap_fraction_delta_gt_zero=float(np.mean(np.asarray(values) > 0)),
                bootstrap_groups_used_min=float(min(nused)), bootstrap_groups_used_max=float(max(nused)),
                cluster='family_component_id', model_seed_handling='mean probability over 3 frozen seeds; CI conditional on fitted models',
                multiplicity_preserved=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--bootstrap-replicates', type=int, default=10000)
    args = p.parse_args()
    if args.workers < 1 or args.bootstrap_replicates != 10000:
        raise ValueError('Positive workers and frozen 10000 replicates required')
    c = verify_contract()
    # A,A,B with per-protein deltas 1,1,0 must give 2/3, not 1/2.
    assert resample_family_statistics([1., 0.], [1., 1.], np.array([[0, 0, 1]]))[0][0] == 2/3
    raw = collect()
    out = PACKAGE / 'cpu_output'
    out.mkdir(exist_ok=True)
    fields = ['cohort_arm', 'regime', 'model', 'outer_fold', 'main_row_id', 'uniprot',
              'family_component_id', 'full_inchikey', 'connectivity_key', 'binary_label']
    assert raw.groupby(fields).seed.nunique().eq(3).all()
    ensemble = raw.groupby(fields, as_index=False).p_allosteric.mean()
    ensemble.to_csv(out / 'ENSEMBLE_PREDICTIONS.tsv.gz', sep='\t', index=False)
    base = pd.read_csv(PACKAGE / 'data/BASELINE_ENSEMBLE.tsv.gz', sep='\t', dtype={'main_row_id': str})
    metrics = []
    for label, data in [('new_seed_ensemble', ensemble), ('existing_seed_ensemble', base)]:
        for (arm, regime, model), g in data.groupby(['cohort_arm', 'regime', 'model']):
            for metric, field in [('pooled', None), ('within_protein', 'uniprot'),
                                   ('within_ligand', 'connectivity_key'), ('family_macro', 'family_component_id')]:
                metrics.append(dict(source=label, cohort_arm=arm, regime=regime, model=model,
                                    seed='ensemble', metric=metric, **stats(g, field)))
    for (arm, model, seed), g in raw.groupby(['cohort_arm', 'model', 'seed']):
        for metric, field in [('pooled', None), ('within_protein', 'uniprot'),
                              ('within_ligand', 'connectivity_key'), ('family_macro', 'family_component_id')]:
            metrics.append(dict(source='new_individual_seed', cohort_arm=arm, regime='double_unseen',
                                model=model, seed=seed, metric=metric, **stats(g, field)))
    pd.DataFrame(metrics).to_csv(out / 'METRICS.tsv', sep='\t', index=False)
    jobs = []
    for a in ARMS:
        e = ensemble[ensemble.cohort_arm == a]
        b = base[(base.cohort_arm == a) & (base.regime == 'unseen_family')]
        ligand = e[e.model == 'ligand']
        for model in MODELS:
            g = e[e.model == model]
            reference = b[b.model == model]
            assert set(g.main_row_id) == set(reference.main_row_id)
            jobs.append((dict(cohort_arm=a, contrast='explicit_double_unseen_minus_family_held_out',
                              model=model, comparator=model), g, reference, args.bootstrap_replicates))
            if model not in ('ligand', 'protein'):
                jobs.append((dict(cohort_arm=a, contrast='joint_minus_ligand_in_explicit_double_unseen',
                                  model=model, comparator='ligand'), g, ligand, args.bootstrap_replicates))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        comparisons = list(pool.map(contrast_job, jobs))
    pd.DataFrame(comparisons).to_csv(out / 'PAIRED_WITHIN_PROTEIN_BOOTSTRAP.tsv', sep='\t', index=False)
    summary = pd.read_csv(PACKAGE / 'gpu_output/FIT_SUMMARY.tsv', sep='\t')
    stability = summary.groupby(['cohort_arm', 'model']).agg(
        n_fits=('best_epoch', 'size'), median_best_epoch=('best_epoch', 'median'),
        epoch1_fraction=('best_epoch', lambda x: float((x == 1).mean())),
        fallback_fraction=('checkpoint_selection_fallback_at_best_epoch', 'mean')).reset_index()
    stability['epoch1_review_flag'] = stability.epoch1_fraction >= 1/3
    stability.to_csv(out / 'TRAINING_STABILITY.tsv', sep='\t', index=False)
    write_json(out / 'VALIDATION.json', dict(status='validated', fits=240,
        prediction_rows=len(raw), ensemble_rows=len(ensemble), test_rows_per_arm=4637,
        primary='fold-restricted within-protein AUROC, joint minus ligand-only',
        comparisons=len(comparisons), replicates=10000, cluster_multiplicity_test='A,A,B = 2/3 PASS',
        conditional_metric_coverage_reported=True, expected_fallback_fits=c['expected_fallback_fits'],
        actual_fallback_fits=int(summary.checkpoint_selection_fallback_at_best_epoch.sum()),
        cpu_only_analysis=True, no_sklearn_required=True,
        limitation='identity purging reduces training size and changes validation composition; no size-matched retraining control in this extension'))
    print('PASS: 240 fits aggregated; 28 paired family-cluster bootstrap comparisons.')


if __name__ == '__main__':
    main()
