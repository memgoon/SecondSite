"""CPU regression checks, independent of tensor files and GPU dependencies."""
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from common import PACKAGE, ARMS, MODELS, SEEDS, VERSION, TRAINING, SELECTION
from common import verify_contract, partition, read_frame, expected_jobs, auc, ap, write_json, resample_family_statistics, scaffold_set


def main():
    c = verify_contract()
    assert c['training'] == TRAINING and c['checkpoint_selection'] == SELECTION
    manifest = pd.read_csv(PACKAGE / 'data/SPLIT_MEMBERSHIP.tsv.gz', sep='\t', dtype={'main_row_id': str})
    support = pd.read_csv(PACKAGE / 'data/CHECKPOINT_SUPPORT.tsv', sep='\t')
    all_folds = {}
    for arm in ARMS:
        frame = read_frame(arm)
        test_ids = []
        for fold in range(5):
            parts = partition(frame, fold)
            for label, subset in zip(('train', 'validation', 'test', 'excluded'), parts):
                rows = manifest[(manifest.cohort_arm == arm) & (manifest.outer_fold == fold) & (manifest.partition == label)]
                assert set(rows.main_row_id) == set(subset.main_row_id)
            for col in ('family_component_id', 'uniprot', 'connectivity_key'):
                a, b, d = [set(x[col]) for x in parts[:3]]
                assert not (a & b or a & d or b & d)
            a, b, d = [scaffold_set(x) for x in parts[:3]]
            assert not (a & b or a & d or b & d)
            test_ids.extend(parts[2].main_row_id)
        assert len(test_ids) == len(set(test_ids)) == 4637
        all_folds[arm] = [set(partition(frame, f)[2].main_row_id) for f in range(5)]
    assert len(expected_jobs()) == len(set(expected_jobs())) == c['expected_fits'] == 120
    assert int(support.fallback_to_pooled_symmetric_ap.sum()) * 3 == c['expected_fallback_fits']
    assert auc([0, 1], [.2, .8]) == 1.0
    assert auc([0, 1], [.5, .5]) == .5
    assert ap([0, 1], [.5, .5]) == .5
    assert resample_family_statistics([1., 0.], [1., 1.], np.array([[0, 0, 1]]))[0][0] == 2/3
    values, used = resample_family_statistics([0., 0.], [0., 1.], np.array([[0, 0], [0, 1]]))
    assert values.tolist() == [0.] and used.tolist() == [1.]
    # Actual analysis path: identical fixed predictions must give exactly zero paired CI.
    from aggregate import contrast_job, stats, prepare_pooled, weighted_pooled, make_jobs
    example = pd.DataFrame(dict(outer_fold=[0, 0, 1, 1], uniprot=['A', 'A', 'B', 'B'],
        main_row_id=['0', '1', '2', '3'],
        family_component_id=['FA', 'FA', 'FB', 'FB'], binary_label=[0, 1, 0, 1],
        p_allosteric=[.1, .9, .2, .8]))
    same = contrast_job((dict(test='synthetic_identical'), example, example.copy(), 10000))
    assert same['delta'] == same['ci_low'] == same['ci_high'] == 0.
    assert same['valid_replicates'] == 10000
    assert stats(example, 'uniprot')['auroc'] == 1.
    same_pooled = contrast_job((dict(test='synthetic_identical', metric='pooled'), example, example.copy(), 10000))
    assert same_pooled['delta'] == same_pooled['ci_low'] == same_pooled['ci_high'] == 0.
    tied = example.copy()
    tied['p_allosteric'] = [.2, .5, .5, .1]
    prep = prepare_pooled(tied, ['FA', 'FB'])
    for mult in ([2., 1.], [1., 2.], [0., 3.], [3., 0.]):
        sampled = pd.concat([tied[tied.family_component_id == f] for f, n in zip(['FA', 'FB'], mult)
                             for _ in range(int(n))], ignore_index=True)
        result, _ = weighted_pooled(prep, np.array([mult]))
        assert np.isclose(result[0], auc(sampled.binary_label, sampled.p_allosteric))
    baseline = pd.read_csv(PACKAGE / 'data/BASELINE_ENSEMBLE.tsv.gz', sep='\t', dtype={'main_row_id': str})
    actual = read_frame('every_pair')
    ring_ids = set(actual.loc[actual.scaffold_available == 1, 'main_row_id'])
    jobs = make_jobs(baseline, baseline.copy(), ring_ids, 32)
    assert len(jobs) == 44
    for job in jobs:
        result = contrast_job(job)
        assert result['valid_replicates'] == 32
        if result['contrast'] == 'murcko_minus_connectivity_double_held_out':
            assert result['delta'] == result['ci_low'] == result['ci_high'] == 0.
    # Execute the actual vendored checkpoint selector's AST without importing torch.
    tree = ast.parse((PACKAGE / 'scripts/trainer_core.py').read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'checkpoint_selection')
    module = ast.parse('')
    module.body = [function]
    scope = dict(np=np, JOINT_MODELS=set(MODELS)-{'ligand', 'protein'}, MIN_CHECKPOINT_GROUPS=8)
    exec(compile(ast.fix_missing_locations(module), '<selector-test>', 'exec'), scope)
    for model in MODELS:
        for protein_groups, ligand_groups in [(20, 20), (20, 3), (3, 20), (3, 3)]:
            metrics = dict(pooled=dict(symmetric_ap=.6),
                protein_macro=dict(macro_auroc=.7, n_groups_used=protein_groups),
                ligand_macro=dict(macro_auroc=.8, n_groups_used=ligand_groups))
            select = scope['checkpoint_selection']
            assert select(metrics, 'pfam_murcko', model) == select(metrics, 'unseen_family', model)
    write_json(PACKAGE / 'validation/STATIC_VALIDATION.json', dict(status='validated',
        source_hashes_match=True, fits=120, explicit_partitions_match_frozen_manifest=True,
        all_overlap_zero=True, test_coverage_per_arm=4637,
        selector_equivalent_to_original_family_rule=True, tie_aware_metric_tests=True,
        bootstrap_multiplicity_self_test=True, weighted_pooled_matches_explicit_duplicates=True,
        all_44_analysis_paths_exercised=True, gpu_forward_and_training='not run on this CPU host'))
    print('PASS: frozen hashes, 3-way partitions, OOF coverage, checkpoint rule and metric tests.')


if __name__ == '__main__':
    main()
