"""Local-only preparation; compute scaffolds once, then freeze CPU/GPU inputs."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from common import (PACKAGE, ARMS, MODELS, SEEDS, VERSION, TRAINING, SELECTION,
                    sha, digest, write_json, partition, connectivity_partition,
                    read_frame, scaffold_set, scaffold_overlap_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-scaffolds', action='store_true',
                        help='Re-freeze code hashes without requiring RDKit; verify scaffold provenance.')
    args = parser.parse_args()
    root = PACKAGE.parents[1]
    old = root / 'analysis/explicit_double_unseen'
    source = old / 'data/EVERY_PAIR.tsv.gz'
    old_validation = json.loads((old / 'cpu_output/VALIDATION.json').read_text())
    assert old_validation['status'] == 'validated' and old_validation['fits'] == 240
    (PACKAGE / 'data').mkdir(exist_ok=True)
    (PACKAGE / 'validation').mkdir(exist_ok=True)
    frame = pd.read_csv(source, sep='\t', keep_default_na=False, dtype={'main_row_id': str})
    assert len(frame) == 6854 and frame.full_inchikey.nunique() == 4418
    assert not frame.main_row_id.duplicated().any()
    scaffold_path = PACKAGE / 'data/LIGAND_SCAFFOLDS.tsv'
    provenance_path = PACKAGE / 'data/SCAFFOLD_PROVENANCE.json'
    if args.reuse_scaffolds:
        provenance = json.loads(provenance_path.read_text())
        assert provenance['source_sha256'] == sha(source)
        assert provenance['scaffold_sha256'] == sha(scaffold_path)
        scaffolds = pd.read_csv(scaffold_path, sep='\t', keep_default_na=False)
    else:
        from rdkit import Chem, rdBase
        from rdkit.Chem.Scaffolds import MurckoScaffold
        rows = []
        for key, g in frame.groupby('full_inchikey', sort=True):
            found, canonical = set(), set()
            for smiles in sorted(set(g.canonical_smiles)):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError('Unparseable SMILES: ' + key)
                # Keep atom/bond identities. Discard scaffold stereochemistry only.
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                found.add(Chem.MolToSmiles(scaffold, isomericSmiles=False))
                canonical.add(Chem.MolToSmiles(mol, isomericSmiles=True))
            rows.append(dict(full_inchikey=key, connectivity_key=g.connectivity_key.iloc[0],
                murcko_scaffolds_json=json.dumps(sorted(found - {''})),
                n_observed_scaffold_variants=len(found - {''}),
                empty_scaffold_observed=int('' in found), canonical_smiles_json=json.dumps(sorted(canonical))))
        scaffolds = pd.DataFrame(rows)
        # Same recorded connectivity may have alternate charge/protonation forms.
        # Block EVERY observed scaffold variant; never pick the most convenient one.
        variants = {k: json.dumps(sorted({s for v in g.murcko_scaffolds_json for s in json.loads(v)}))
                    for k, g in scaffolds.groupby('connectivity_key')}
        scaffolds['murcko_scaffolds_json'] = scaffolds.connectivity_key.map(variants)
        scaffolds['scaffold_available'] = scaffolds.murcko_scaffolds_json.map(lambda v: int(bool(json.loads(v))))
        scaffolds.to_csv(scaffold_path, sep='\t', index=False)
        provenance = dict(status='validated', rdkit_version=rdBase.rdkitVersion,
            source_path=str(source.relative_to(root)), source_sha256=sha(source),
            scaffold_sha256=sha(scaffold_path), unique_ligands=len(scaffolds),
            definition='RDKit Bemis-Murcko scaffold, atom/bond types retained, isomericSmiles=False; not generic scaffold',
            empty_scaffold_policy='No common empty-scaffold group; connectivity still excluded; ring-only evaluation separately',
            multiple_representation_policy='Union of all observed Murcko variants per connectivity; purge any shared variant',
            ligands_with_multiple_observed_variants=int((scaffolds.n_observed_scaffold_variants > 1).sum()),
            tautomer_or_analogue_similarity_exclusion=False)
        write_json(provenance_path, provenance)
    assert not scaffolds.full_inchikey.duplicated().any()
    assert set(scaffolds.full_inchikey) == set(frame.full_inchikey)
    frame = frame.merge(scaffolds[['full_inchikey', 'murcko_scaffolds_json', 'scaffold_available']],
                        on='full_inchikey', how='left', validate='many_to_one')
    assert frame.murcko_scaffolds_json.notna().all()
    assert frame.groupby('connectivity_key').murcko_scaffolds_json.nunique().max() == 1
    frame.to_csv(PACKAGE / 'data/EVERY_PAIR.tsv.gz', sep='\t', index=False)
    shutil.copyfile(old / 'data/POCKET_INDICES.json', PACKAGE / 'data/POCKET_INDICES.json')
    baseline_run = json.loads((old / 'gpu_output/RUN_CONTRACT.json').read_text())
    assert baseline_run['status'] == 'validated'
    assert all(baseline_run['training_contract'][k] == v for k, v in TRAINING.items())
    assert baseline_run['training_contract']['checkpoint_selection'] == SELECTION
    shutil.copyfile(old / 'gpu_output/RUN_CONTRACT.json', PACKAGE / 'data/BASELINE_RUN_CONTRACT.json')
    baseline_path = old / 'cpu_output/ENSEMBLE_PREDICTIONS.tsv.gz'
    baseline = pd.read_csv(baseline_path, sep='\t', dtype={'main_row_id': str})
    baseline = baseline[baseline.cohort_arm == 'every_pair'].copy()
    assert set(baseline.regime) == {'double_unseen'}
    for model in MODELS:
        b = baseline[baseline.model == model]
        assert len(b) == 4637 and not b.main_row_id.duplicated().any()
        ref = frame[frame.matrix_evaluation_eligible == 1].set_index('main_row_id')
        assert set(b.main_row_id) == set(ref.index)
        for field in ('binary_label', 'uniprot', 'family_component_id', 'full_inchikey', 'connectivity_key'):
            assert b[field].astype(str).tolist() == ref.loc[b.main_row_id, field].astype(str).tolist()
        assert b.outer_fold.astype(int).tolist() == ref.loc[b.main_row_id, 'matrix_family_fold'].astype(int).tolist()
        assert np.isfinite(b.p_allosteric).all() and b.p_allosteric.between(0, 1).all()
    baseline.to_csv(PACKAGE / 'data/BASELINE_ENSEMBLE.tsv.gz', sep='\t', index=False)
    write_json(PACKAGE / 'data/BASELINE_PROVENANCE.json', dict(
        source=str(baseline_path.relative_to(root)), source_sha256=sha(baseline_path),
        validation_sha256=sha(old / 'cpu_output/VALIDATION.json'),
        split_sha256=sha(old / 'data/SPLIT_MEMBERSHIP.tsv.gz'),
        cpu_contract_sha256=sha(old / 'validation/CPU_CONTRACT.json'),
        run_contract_sha256=sha(old / 'gpu_output/RUN_CONTRACT.json'),
        baseline_training=old_validation, comparison='completed explicit double-held-out retraining, NOT family-held-out subset rescoring'))
    counts, memberships, support, audit, removals = [], [], [], [], []
    old_membership = pd.read_csv(old / 'data/SPLIT_MEMBERSHIP.tsv.gz', sep='\t', dtype={'main_row_id': str})
    for fold in range(5):
        original = connectivity_partition(frame, fold)
        previous = old_membership[(old_membership.cohort_arm == 'every_pair') & (old_membership.outer_fold == fold)]
        for name, part in zip(('train', 'validation', 'test', 'excluded'), original):
            assert set(part.main_row_id) == set(previous.loc[previous.partition == name, 'main_row_id'])
        parts = partition(frame, fold)
        assert set(parts[2].main_row_id) == set(original[2].main_row_id)
        for i, name in [(0, 'train'), (1, 'validation')]:
            removed = original[i][~original[i].main_row_id.isin(parts[i].main_row_id)].copy()
            hits_test = scaffold_overlap_mask(removed, scaffold_set(parts[2]))
            hits_val = scaffold_overlap_mask(removed, scaffold_set(parts[1])) if i == 0 else pd.Series(False, index=removed.index)
            assert (hits_test | hits_val).all()
            r = removed[['main_row_id', 'uniprot', 'connectivity_key', 'binary_label']].copy()
            r['outer_fold'], r['original_partition'] = fold, name
            r['test_scaffold_overlap'], r['retained_validation_scaffold_overlap'] = hits_test, hits_val
            removals.append(r)
        # Feasibility gates are not tuned to performance; fail before expensive training.
        assert len(parts[0]) >= 100 and parts[0].binary_label.value_counts().min() >= 20
        assert len(parts[1]) >= 20 and parts[1].binary_label.value_counts().min() >= 5
        for name, part, before in zip(('train', 'validation', 'test', 'excluded'), parts, original):
            counts.append(dict(cohort_arm='every_pair', outer_fold=fold, partition=name,
                original_rows=len(before), rows=len(part), removed_from_original=len(before)-len(part),
                proteins=part.uniprot.nunique(), families=part.family_component_id.nunique(),
                ligands=part.connectivity_key.nunique(), nonempty_scaffolds=len(scaffold_set(part)),
                ring_rows=int(part.scaffold_available.sum()), acyclic_rows=int((part.scaffold_available == 0).sum()),
                allosteric=int(part.binary_label.sum()), orthosteric=int((part.binary_label == 0).sum())))
            cols = ['main_row_id', 'uniprot', 'family_component_id', 'connectivity_key',
                    'full_inchikey', 'binary_label', 'murcko_scaffolds_json', 'scaffold_available']
            member = part[cols].copy()
            member['cohort_arm'], member['outer_fold'], member['partition'] = 'every_pair', fold, name
            memberships.append(member)
        for model in MODELS:
            field = 'uniprot' if model == 'ligand' else 'connectivity_key'
            n = int((parts[1].groupby(field).binary_label.nunique() == 2).sum())
            support.append(dict(cohort_arm='every_pair', outer_fold=fold, model=model,
                validation_selection_group=field, valid_groups=n, fallback_to_pooled_symmetric_ap=n < 8))
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            audit.append(dict(outer_fold=fold, partition_a=['train', 'validation', 'test'][i],
                partition_b=['train', 'validation', 'test'][j],
                nonempty_scaffold_overlap=len(scaffold_set(parts[i]) & scaffold_set(parts[j])),
                connectivity_overlap=len(set(parts[i].connectivity_key) & set(parts[j].connectivity_key)),
                family_overlap=len(set(parts[i].family_component_id) & set(parts[j].family_component_id))))
    pd.DataFrame(counts).to_csv(PACKAGE / 'data/SPLIT_COUNTS.tsv', sep='\t', index=False)
    pd.concat(memberships, ignore_index=True).to_csv(PACKAGE / 'data/SPLIT_MEMBERSHIP.tsv.gz', sep='\t', index=False)
    pd.DataFrame(support).to_csv(PACKAGE / 'data/CHECKPOINT_SUPPORT.tsv', sep='\t', index=False)
    pd.DataFrame(audit).to_csv(PACKAGE / 'data/OVERLAP_AUDIT.tsv', sep='\t', index=False)
    pd.concat(removals, ignore_index=True).to_csv(PACKAGE / 'data/ADDITIONAL_EXCLUDED_ROWS.tsv.gz', sep='\t', index=False)
    files = sorted(p for folder in ('scripts', 'data') for p in (PACKAGE / folder).iterdir() if p.is_file())
    files += [PACKAGE / x for x in ('README.md', 'requirements-cpu.txt', 'requirements-gpu.txt')]
    hashes = {str(p.relative_to(PACKAGE)): sha(p) for p in files}
    write_json(PACKAGE / 'validation/CPU_CONTRACT.json', dict(status='validated',
        version=VERSION, files=hashes, manifest_digest=digest(hashes), expected_fits=120,
        arms=list(ARMS), models=list(MODELS), seeds=list(SEEDS), folds=list(range(5)),
        training=TRAINING, checkpoint_selection=SELECTION, expected_test_rows_per_arm=4637,
        expected_fallback_fits=sum(x['fallback_to_pooled_symmetric_ap'] for x in support)*len(SEEDS),
        all_test_rows_retained=True, scaffold_definition=provenance,
        partition_rule='Nested purge of original explicit double-held-out training and validation; no reinstated rows',
        interpretation='Additional exact Murcko scaffold exclusion, not elimination of all chemical similarity; train/validation size changes',
        bootstrap_replicates=10000, bootstrap_location='local CPU host after fetch'))
    print(pd.DataFrame(counts).query("partition != 'excluded'").to_string(index=False))
    print('Validated 120 fits; expected fallback fits:', sum(x['fallback_to_pooled_symmetric_ap'] for x in support)*3)


if __name__ == '__main__':
    main()
