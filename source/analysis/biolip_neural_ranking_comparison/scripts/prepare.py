"""Join exact pairs; freeze the common population before computing statistics."""
import numpy as np
import pandas as pd
from common import *


def main():
    contract=PACKAGE/'validation/CPU_CONTRACT.json'
    if contract.exists():
        check_contract();print('Existing CPU contract verified; no regeneration');return
    if (PACKAGE/'cpu_output/VALIDATION.json').exists():raise RuntimeError('Output without source contract')
    assert read_json(FILES['biolip_validation'])['status']=='PASS_COMPUTATIONAL_VALIDATION'
    assert read_json(FILES['neural_validation'])['status']=='validated'
    ref=pd.read_csv(FILES['reference'],sep='\t')
    nn=pd.read_csv(FILES['neural_oof'],sep='\t')
    nn=nn[(nn.cohort_arm=='every_pair')&(nn.regime=='unseen_family')].copy()
    bio=pd.read_csv(FILES['biolip_oof'],sep='\t')
    bio=bio[(bio.regime=='family_held_out')&bio.ranking_id.isin(METHODS)].copy()
    cohort=pd.read_csv(FILES['cohort'],sep='\t',low_memory=False)
    assert len(ref)==1322 and ref.pair_id.is_unique
    assert not ref.duplicated(['uniprot','full_inchikey']).any()
    assert ref.feature_complete.all()
    assert set(nn.model)==set(MODELS) and set(bio.ranking_id)==set(METHODS)
    assert not nn.duplicated(['model','uniprot','full_inchikey']).any()
    assert not bio.duplicated(['ranking_id','pair_id']).any()
    assert np.isfinite(nn.p_allosteric).all() and nn.p_allosteric.between(0,1).all()
    assert np.isfinite(bio.score).all()
    keys=['uniprot','full_inchikey']
    cols=['pair_id','uniprot','full_inchikey','binary_label','family_component_id','family_fold']
    common=ref[cols].rename(columns={'family_component_id':'biolip_family','family_fold':'biolip_fold'})
    first=None
    for model in MODELS:
        x=nn[nn.model==model]
        x=common[['pair_id']+keys+['binary_label']].merge(x,on=keys,validate='one_to_one',suffixes=('_bio',''))
        assert (x.binary_label_bio==x.binary_label).all()
        ids=set(x.pair_id)
        if first is None:first=ids
        else:assert ids==first
        assert len(x)==987
        add=['pair_id','p_allosteric']
        if model=='ligand':add+=['family_component_id','outer_fold','main_row_id']
        common=common.merge(x[add].rename(columns={'p_allosteric':'nn_'+model,
            'family_component_id':'neural_family','outer_fold':'neural_fold'}),on='pair_id',validate='one_to_one')
    for method in METHODS:
        x=bio[bio.ranking_id==method]
        z=common[['pair_id','uniprot','binary_label','biolip_fold','biolip_family']].merge(x,on='pair_id',validate='one_to_one',suffixes=('','_score'))
        assert len(z)==987 and (z.binary_label==z.binary_label_score).all()
        assert (z.uniprot==z.uniprot_score).all() and (z.biolip_fold==z.fold).all()
        assert (z.biolip_family==z.family_component_id).all()
        common=common.merge(x[['pair_id','score']].rename(columns={'score':method}),on='pair_id',validate='one_to_one')
    assert len(common)==987 and common.uniprot.nunique()==278
    assert (common.groupby('uniprot')[['neural_fold','biolip_fold']].nunique()==1).all().all()
    # Verify exact prediction-row identity and the frozen outer test-fold assignment.
    z=common.merge(cohort[['main_row_id']+keys+['binary_label','matrix_family_fold','matrix_evaluation_eligible']],
        on='main_row_id',validate='one_to_one',suffixes=('','_cohort'))
    for col in keys+['binary_label']:assert (z[col]==z[col+'_cohort']).all()
    assert (z.neural_fold==z.matrix_family_fold).all() and z.matrix_evaluation_eligible.eq(1).all()
    assert cohort.groupby('uniprot').protein_embedding_path.nunique().max()==1
    assert common.groupby('uniprot').nn_protein.nunique().eq(1).all(), 'Nonconstant protein control: audit before use'
    # Conservative cluster union: never separate proteins sharing a component in
    # either source partition. Use full references, not only matched proteins.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for _,r in ref[['uniprot','family_component_id']].drop_duplicates().iterrows():
        union('u:'+r.uniprot,'bio:'+r.family_component_id)
    for _,r in cohort[['uniprot','family_component_id']].drop_duplicates().iterrows():
        union('u:'+r.uniprot,'nn:'+r.family_component_id)
    roots={u:find('u:'+u) for u in common.uniprot.unique()}
    labels={v:'UNION_%04d'%i for i,v in enumerate(sorted(set(roots.values())))}
    common['bootstrap_family']=common.uniprot.map({u:labels[v] for u,v in roots.items()})
    common=common.sort_values(['uniprot','pair_id']).reset_index(drop=True)
    assert common.bootstrap_family.nunique()>=2
    both=common.groupby('uniprot').binary_label.nunique().eq(2)
    assert both.sum()==81 and common.uniprot.isin(both[both].index).sum()==409
    exact=set(zip(cohort.uniprot,cohort.full_inchikey))
    coverage=ref[['pair_id']+keys+['binary_label']].copy()
    coverage['in_every_pair_full_training']=[k in exact for k in zip(ref.uniprot,ref.full_inchikey)]
    coverage['in_common_heldout_comparison']=coverage.pair_id.isin(common.pair_id)
    coverage['exclusion_reason']=np.where(coverage.in_common_heldout_comparison,'included',
        np.where(coverage.in_every_pair_full_training,'training_only_rows_without_existing_OOF','no_existing_neural_OOF_for_pair'))
    assert coverage.in_every_pair_full_training.sum()==1158
    data=PACKAGE/'data';data.mkdir(parents=True,exist_ok=True)
    common.to_csv(data/'MATCHED_SCORES.tsv.gz',sep='\t',index=False,compression='gzip')
    coverage.to_csv(data/'REFERENCE_COVERAGE.tsv',sep='\t',index=False)
    proteins=common.groupby('uniprot').agg(rows=('pair_id','size'),labels=('binary_label','nunique'),
        neural_fold=('neural_fold','first'),biolip_fold=('biolip_fold','first'),
        neural_family=('neural_family','first'),biolip_family=('biolip_family','first'),bootstrap_family=('bootstrap_family','first')).reset_index()
    proteins.to_csv(data/'PROTEIN_SUPPORT.tsv',sep='\t',index=False)
    files=list(FILES.values())+implementation_files()+[data/n for n in ['MATCHED_SCORES.tsv.gz','REFERENCE_COVERAGE.tsv','PROTEIN_SUPPORT.tsv']]
    payload=dict(files={str(p.relative_to(ROOT)):sha(p) for p in files}, models=list(MODELS),biolip_methods=list(METHODS),
        rows=987,proteins=278,label_comparison_proteins=81,label_comparison_rows=409,
        bootstrap_clusters=int(common.bootstrap_family.nunique()),reference_rows=1322,
        min_spearman_rows=MIN_RHO_ROWS,replicates=REPLICATES,random_seed=SEED,
        neural_cohort='every_pair',neural_regime='unseen_family',biolip_regime='family_held_out',
        independent_external_validation=False,new_training=False,new_inference=False,
        bootstrap_unit='union of neural and BioLiP protein-family components',
        min_valid_replicate_fraction=.95,interval_scope='pointwise; fixed scores; no multiple-testing correction')
    write_json(contract,dict(status='validated',version=VERSION,payload=payload,fingerprint=digest(payload)))
    print('CPU inputs frozen:',len(common),'pairs;',len(proteins),'proteins;',payload['bootstrap_clusters'],'union family clusters')


if __name__=='__main__':main()
