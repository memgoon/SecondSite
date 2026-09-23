"""Freeze explicit disjoint train/validation/test manifests before GPU training."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from common import PACKAGE, ARMS, FILES, MODELS, SEEDS, VERSION, TRAINING, SELECTION
from common import sha, digest, write_json, partition, read_frame


def main():
    root = PACKAGE.parents[1]
    source = root / 'analysis/role_complete_pair_matrix'
    for directory in ('data', 'validation'):
        (PACKAGE / directory).mkdir(exist_ok=True)
    for name in FILES.values():
        shutil.copyfile(source / 'data' / name, PACKAGE / 'data' / name)
    shutil.copyfile(root / 'analysis/allosteric_pair_benchmark_broad_superset/data/POCKET_INDICES.json',
                    PACKAGE / 'data/POCKET_INDICES.json')
    # Freeze pre-existing baseline predictions for identical-row comparison; no refitting.
    baseline = pd.read_csv(source / 'gpu_output/benchmark/aggregate/ENSEMBLE_OOF_PREDICTIONS.tsv.gz', sep='\t')
    baseline = baseline[baseline.cohort_arm.isin(ARMS) & baseline.regime.isin(['row_random', 'unseen_family'])]
    baseline.to_csv(PACKAGE / 'data/BASELINE_ENSEMBLE.tsv.gz', sep='\t', index=False)
    summary, memberships, eligibility = [], [], []
    expected_trains = {'every_pair': [3474, 3899, 4123, 3914, 3383],
                       'protein_anchored': [1653, 2032, 2195, 2064, 1581],
                       'protein_ligand_role_complete': [46, 19, 16, 67, 91]}
    for arm in FILES:
        d = read_frame(arm)
        assert len(d) == {'every_pair': 6854, 'protein_anchored': 4637, 'protein_ligand_role_complete': 395}[arm]
        assert not d.main_row_id.duplicated().any()
        all_tests = []
        for f in range(5):
            parts = partition(d, f)
            assert len(parts[0]) == expected_trains[arm][f]
            for name, part in zip(('train', 'validation', 'test', 'excluded'), parts):
                summary.append(dict(cohort_arm=arm, outer_fold=f, partition=name, rows=len(part),
                    proteins=part.uniprot.nunique(), families=part.family_component_id.nunique(),
                    ligands=part.connectivity_key.nunique(), allosteric=int(part.binary_label.sum()),
                    orthosteric=int((part.binary_label == 0).sum()),
                    two_class_proteins=int((part.groupby('uniprot').binary_label.nunique() == 2).sum()),
                    two_class_ligands=int((part.groupby('connectivity_key').binary_label.nunique() == 2).sum())))
                members = part[['main_row_id', 'uniprot', 'family_component_id', 'connectivity_key', 'binary_label']].copy()
                members['cohort_arm'], members['outer_fold'], members['partition'] = arm, f, name
                memberships.append(members)
            all_tests.extend(parts[2].main_row_id)
            v = parts[1]
            for model in MODELS:
                field = 'uniprot' if model == 'ligand' else 'connectivity_key'
                support = int((v.groupby(field).binary_label.nunique() == 2).sum())
                eligibility.append(dict(cohort_arm=arm, outer_fold=f, model=model,
                    validation_selection_group=field, valid_groups=support,
                    fallback_to_pooled_symmetric_ap=support < 8, scheduled=arm in ARMS))
        expected = d[d.matrix_evaluation_eligible == 1].main_row_id
        assert len(all_tests) == len(set(all_tests)) == len(expected)
        assert set(all_tests) == set(expected)
    pd.DataFrame(summary).to_csv(PACKAGE / 'data/SPLIT_COUNTS.tsv', sep='\t', index=False)
    pd.concat(memberships, ignore_index=True).to_csv(PACKAGE / 'data/SPLIT_MEMBERSHIP.tsv.gz', sep='\t', index=False)
    pd.DataFrame(eligibility).to_csv(PACKAGE / 'data/CHECKPOINT_SUPPORT.tsv', sep='\t', index=False)
    for f in range(5):
        a, b = [partition(read_frame(arm), f) for arm in ARMS]
        assert set(a[1].main_row_id) == set(b[1].main_row_id)
        assert set(a[2].main_row_id) == set(b[2].main_row_id)
    # Each expected baseline cell is complete and has the frozen row metadata.
    for arm in ARMS:
        ref = read_frame(arm).query('matrix_evaluation_eligible == 1').set_index('main_row_id')
        for regime in ('row_random', 'unseen_family'):
            for model in MODELS:
                b = baseline[(baseline.cohort_arm == arm) & (baseline.regime == regime) & (baseline.model == model)].copy()
                b.main_row_id = b.main_row_id.astype(str)
                assert len(b) == 4637 and not b.main_row_id.duplicated().any()
                assert set(b.main_row_id) == set(ref.index)
                for field in ('binary_label', 'uniprot', 'family_component_id', 'connectivity_key'):
                    assert b[field].astype(str).tolist() == ref.loc[b.main_row_id, field].astype(str).tolist()
                col = 'matrix_row_fold' if regime == 'row_random' else 'matrix_family_fold'
                assert b.outer_fold.astype(int).tolist() == ref.loc[b.main_row_id, col].astype(int).tolist()
    # Only named package files and one-level script enumeration; never a workspace walk.
    files = sorted([p for sub in ('data', 'scripts') for p in (PACKAGE / sub).iterdir() if p.is_file()])
    files += [PACKAGE / 'README.md', PACKAGE / 'requirements-cpu.txt', PACKAGE / 'requirements-gpu.txt']
    hashes = {str(p.relative_to(PACKAGE)): sha(p) for p in files}
    scheduled = [x for x in eligibility if x['scheduled']]
    contract = dict(status='validated', version=VERSION, files=hashes, manifest_digest=digest(hashes),
        expected_fits=240, arms=list(ARMS), models=list(MODELS), seeds=list(SEEDS), folds=list(range(5)),
        expected_test_rows_per_arm=4637, training=TRAINING, checkpoint_selection=SELECTION,
        expected_fallback_fits=sum(x['fallback_to_pooled_symmetric_ap'] for x in scheduled) * 3,
        all_three_partitions_disjoint_fields=['uniprot', 'family_component_id', 'connectivity_key'],
        no_score_or_test_label_based_purging=True, all_test_rows_retained=True,
        role_complete='audit only: 16-91 train rows; not scheduled',
        selection_scope='same model-aware criterion as existing family baseline; validation membership changes',
        representation='unchanged selected target chain / pocket tensors, not the full-UniProt pilot',
        comparison_limit='matched test rows, but training size and validation composition change; not a pure novelty causal effect')
    write_json(PACKAGE / 'validation/CPU_CONTRACT.json', contract)
    print(pd.DataFrame(summary).query("partition == 'train'").to_string(index=False))
    print('PASS: 240 fits; all partitions pairwise disjoint; 4,637 OOF rows per general arm.')


if __name__ == '__main__':
    main()
