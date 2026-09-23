"""Freeze 47 paired family-only/double-unseen train-test splits, no validation."""
import json
import shutil
from pathlib import Path
import pandas as pd
from common import PACKAGE, VERSION, TRAINING, REGIMES, MODELS, SEEDS, ATP, ADP
from common import read_data, partition, families, sha, digest, write_json, jobs


def main():
    root = PACKAGE.parents[1]
    for sub in ('data', 'validation'):
        (PACKAGE / sub).mkdir(parents=True, exist_ok=True)
    source = root / 'analysis/explicit_double_unseen/data/PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz'
    shutil.copyfile(source, PACKAGE / 'data/COHORT.tsv.gz')
    d = read_data()
    assert len(d) == 395 and d.uniprot.nunique() == 98 and d.full_inchikey.nunique() == 40
    assert len(families(d)) == 47 and int(d.binary_label.sum()) == 143
    assert not d.main_row_id.duplicated().any()
    assert d.groupby('uniprot').binary_label.nunique().eq(2).all()
    assert d.groupby('full_inchikey').binary_label.nunique().eq(2).all()
    assert d.groupby('full_inchikey').ligand_embedding_path.nunique().eq(1).all()
    assert d.groupby('uniprot').protein_embedding_path.nunique().eq(1).all()
    pockets = json.loads((root / 'analysis/explicit_double_unseen/data/POCKET_INDICES.json').read_text())
    write_json(PACKAGE / 'data/POCKET_INDICES.json', {u: pockets[u] for u in sorted(d.uniprot.unique())})
    pd.DataFrame(dict(outer_fold=range(47), family_component_id=families(d))).to_csv(
        PACKAGE / 'data/FAMILIES.tsv', sep='\t', index=False)
    rows, membership = [], []
    for regime in REGIMES:
        seen = []
        for fold in range(47):
            train, test, excluded = partition(d, regime, fold)
            seen.extend(test.main_row_id)
            for label, part in [('train', train), ('test', test), ('excluded', excluded)]:
                rows.append(dict(regime=regime, outer_fold=fold, held_out_family=families(d)[fold],
                    partition=label, rows=len(part), proteins=part.uniprot.nunique(),
                    families=part.family_component_id.nunique(), ligands=part.full_inchikey.nunique(),
                    allosteric=int(part.binary_label.sum()), orthosteric=int((part.binary_label == 0).sum()),
                    two_label_proteins=int((part.groupby('uniprot').binary_label.nunique() == 2).sum()),
                    two_label_ligands=int((part.groupby('full_inchikey').binary_label.nunique() == 2).sum())))
                m = part[['main_row_id', 'uniprot', 'family_component_id', 'full_inchikey', 'connectivity_key', 'binary_label']].copy()
                m['regime'], m['outer_fold'], m['partition'] = regime, fold, label
                membership.append(m)
        assert len(seen) == len(set(seen)) == 395 and set(seen) == set(d.main_row_id)
    counts = pd.DataFrame(rows)
    counts.to_csv(PACKAGE / 'data/SPLIT_COUNTS.tsv', sep='\t', index=False)
    pd.concat(membership, ignore_index=True).to_csv(PACKAGE / 'data/SPLIT_MEMBERSHIP.tsv.gz', sep='\t', index=False)
    ligands = d.groupby('full_inchikey').agg(rows=('binary_label', 'size'), allosteric=('binary_label', 'sum'))
    ligands['orthosteric'] = ligands.rows - ligands.allosteric
    ligands['name_if_prespecified'] = ['ATP' if k == ATP else 'ADP' if k == ADP else '' for k in ligands.index]
    ligands.to_csv(PACKAGE / 'data/LIGAND_COUNTS.tsv', sep='\t')
    train = counts[counts.partition == 'train']
    assert train[train.regime == 'double_unseen'].rows.min() == 148
    assert train[train.regime == 'double_unseen'].rows.max() == 389
    # All implementation and dependency files are named/one-level enumerated, no workspace traversal.
    files = [p for sub in ('data', 'scripts') for p in (PACKAGE / sub).iterdir() if p.is_file()]
    files += [PACKAGE / 'README.md', PACKAGE / 'requirements-cpu.txt', PACKAGE / 'requirements-gpu.txt']
    hashes = {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(files)}
    support_rows = support_groups = 0
    for _, g in d.groupby(['family_component_id', 'full_inchikey']):
        if g.binary_label.nunique() == 2:
            support_rows += len(g)
            support_groups += 1
    assert (support_rows, support_groups) == (42, 14)
    contract = dict(status='validated', version=VERSION, source_path=str(source), source_sha256=sha(source),
        files=hashes, manifest_digest=digest(hashes), training=TRAINING,
        source_rows=395, source_proteins=98, source_ligands=40, source_families=47,
        expected_fits=len(jobs()), expected_prediction_rows=2*8*3*395,
        models=list(MODELS), seeds=list(SEEDS), regimes=list(REGIMES),
        validation_rows=0, test_rows_per_model_seed_regime=395,
        train_range={r: dict(min=int(train[train.regime == r].rows.min()),
                            median=float(train[train.regime == r].rows.median()),
                            max=int(train[train.regime == r].rows.max())) for r in REGIMES},
        primary_support=dict(rows=395, proteins=98, family_clusters=47),
        secondary_within_ligand_support=dict(rows=42, groups=14, role='limited-support descriptive sensitivity'),
        test_labels_not_used_for_training_or_checkpoint_selection=True,
        test_structure='same held-out family and rows in both regimes',
        no_external_training_rows=True, no_transferred_head_weights=True,
        purged_training_not_reanchored=True,
        epoch_rationale='25 is the existing maximum training budget, now fixed in advance; not chosen from test performance')
    write_json(PACKAGE / 'validation/CPU_CONTRACT.json', contract)
    print(json.dumps({k: contract[k] for k in ['expected_fits', 'expected_prediction_rows', 'train_range',
                     'primary_support', 'secondary_within_ligand_support']}, indent=2))


if __name__ == '__main__':
    main()
