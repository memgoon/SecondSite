"""Assemble 4x4 descriptive tables without relabeling heterogeneous protocols."""
import pandas as pd
from common import PACKAGE,REGIMES,MODELS


def build_overview(new_metrics,out):
    rows=[]
    old=pd.read_csv(PACKAGE/'data/LEGACY_MATRIX_METRICS.tsv',sep='\t')
    names={'pooled':'pooled','protein_macro_fold_restricted':'within_protein'}
    q=old[(old.evaluation_universe=='arm_evaluation')&old.endpoint.isin(names)&old.regime.isin(REGIMES[:3])]
    for r in q.itertuples():
        rows.append(dict(cohort_arm=r.cohort_arm,regime=r.regime,model=r.model,metric=names[r.endpoint],
            auroc=r.auroc,n_rows_input=r.n_rows_input,n_rows_used=r.n_rows_used,
            protocol='legacy 5-fold validation-selected',source='role_complete_pair_matrix',
            note='Every-pair evaluates 4637 anchored rows, not all 6854 training-source rows'))
    old=pd.read_csv(PACKAGE/'data/LEGACY_EXPLICIT_METRICS.tsv',sep='\t')
    q=old[(old.source=='new_seed_ensemble')&old.metric.isin(['pooled','within_protein'])]
    for r in q.itertuples():
        rows.append(dict(cohort_arm=r.cohort_arm,regime='double_unseen',model=r.model,metric=r.metric,
            auroc=r.auroc,n_rows_input=r.n_rows_input,n_rows_used=r.n_rows_used,
            protocol='explicit purged 5-fold validation-selected',source='explicit_double_unseen',
            note='4637-row test universe; legacy AMP identity-roundoff may affect single-input rank controls'))
    old=pd.read_csv(PACKAGE/'data/LEGACY_DOUBLE_ANCHORED_METRICS.tsv',sep='\t')
    q=old[(old.kind=='seed_ensemble')&(old.regime=='double_unseen')&old.metric.isin(['pooled','within_protein'])]
    for r in q.itertuples():
        rows.append(dict(cohort_arm='protein_ligand_role_complete',regime='double_unseen',model=r.model,metric=r.metric,
            auroc=r.auroc,n_rows_input=r.n_rows_input,n_rows_used=r.n_rows_used,
            protocol='47-fold leave-one-family-out; fixed 25 epochs; no validation',source='double_anchored_lofo',
            note='Not protocol-identical to five-fold cells; purged train derives from double-anchored source'))
    q=new_metrics[(new_metrics.kind=='seed_ensemble')&new_metrics.metric.isin(['pooled','within_protein'])]
    for r in q.itertuples():
        rows.append(dict(cohort_arm='ligand_anchored',regime=r.regime,model=r.model,metric=r.metric,
            auroc=r.auroc,n_rows_input=r.n_rows_input,n_rows_used=r.n_rows_used,
            protocol='5-fold validation-selected; legacy assignments extended for 274 previously unassigned rows',
            source='ligand_anchored_benchmark',note='1365-row universe differs from other cohorts; descriptive, not paired cohort-effect estimate'))
    table=pd.DataFrame(rows)
    if len(table)!=256 or table.duplicated(['cohort_arm','regime','model','metric']).any():raise ValueError('4x4 matrix coverage')
    table.to_csv(out/'MATRIX_4X4_ALL_MODELS.tsv',sep='\t',index=False)
    for metric in ['pooled','within_protein']:
        table[table.metric.eq(metric)].pivot(index=['cohort_arm','model'],columns='regime',values='auroc').reindex(columns=REGIMES).to_csv(out/('MATRIX_4X4_'+metric.upper()+'.tsv'),sep='\t')
    table[(table.model=='ligand')&(table.metric=='pooled')].to_csv(out/'MATRIX_LIGAND_ONLY_POOLED.tsv',sep='\t',index=False)
    return table
