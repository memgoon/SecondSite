"""Freeze new cohort and extend only missing folds; no recursive file discovery."""
import json
import shutil
import pandas as pd
from common import *


def main():
    if (PACKAGE/'gpu_output/RUN_CONTRACT.json').exists():raise ValueError('Do not rebuild a started experiment')
    root=PACKAGE.parents[1]
    src=root/'analysis/explicit_double_unseen/data/EVERY_PAIR.tsv.gz'
    e=pd.read_csv(src,sep='\t',dtype={'main_row_id':str})
    assert len(e)==6854
    d=e[e.groupby('full_inchikey').binary_label.transform('nunique').eq(2)].copy()
    assert (len(d),d.uniprot.nunique(),d.full_inchikey.nunique(),d.family_component_id.nunique(),int(d.binary_label.sum()))==(1365,543,122,251,405)
    assert not d.duplicated(['uniprot','full_inchikey']).any() and not d.main_row_id.duplicated().any()
    assert d.groupby('full_inchikey').ligand_embedding_path.nunique().eq(1).all()
    assert d.groupby('family_component_id').matrix_family_fold.nunique().eq(1).all()
    assert d.groupby('uniprot').protein_embedding_path.nunique().eq(1).all()
    family=d.groupby('family_component_id').agg(rows=('main_row_id','size'),fold=('matrix_family_fold','first')).reset_index()
    loads=[int(d.matrix_family_fold.eq(f).sum()) for f in range(5)]
    for row in family[family.fold.eq(-1)].sort_values(['rows','family_component_id'],ascending=[False,True]).itertuples():
        f=min(range(5),key=lambda f:(loads[f],f));family.loc[family.family_component_id.eq(row.family_component_id),'fold']=f;loads[f]+=row.rows
    d['run_family_fold']=d.family_component_id.map(family.set_index('family_component_id').fold)
    d['run_row_fold']=d.matrix_row_fold
    loads=[int(d.run_row_fold.eq(f).sum()) for f in range(5)]
    for ix in sorted(d.index[d.run_row_fold.eq(-1)],key=lambda i:hashlib.sha256(('20260916|'+d.loc[i,'main_row_id']).encode()).hexdigest()):
        f=min(range(5),key=lambda f:(loads[f],f));d.loc[ix,'run_row_fold']=f;loads[f]+=1
    for col in ('run_row_fold','run_family_fold','matrix_ligand_fold'):assert set(d[col])==set(range(5))
    for new,old in [('run_row_fold','matrix_row_fold'),('run_family_fold','matrix_family_fold')]:
        assert d.loc[d[old]>=0,new].eq(d.loc[d[old]>=0,old]).all()
    d['cohort_arm']='ligand_anchored'
    (PACKAGE/'data').mkdir(exist_ok=True);(PACKAGE/'validation').mkdir(exist_ok=True)
    d.to_csv(PACKAGE/'data/COHORT.tsv.gz',sep='\t',index=False)
    pocket_src=root/'analysis/explicit_double_unseen/data/POCKET_INDICES.json'
    pockets=json.loads(pocket_src.read_text());write_json(PACKAGE/'data/POCKET_INDICES.json',{u:pockets[u] for u in sorted(d.uniprot.unique())})
    counts=[];members=[];support=[];selections=[]
    for regime in REGIMES:
        seen=[]
        for fold in range(5):
            tr,va,te,ex=partition(d,regime,fold);seen.extend(te.main_row_id)
            for name,part in [('train',tr),('validation',va),('test',te),('excluded',ex)]:
                counts.append(dict(regime=regime,outer_fold=fold,partition=name,rows=len(part),
                    allosteric=int(part.binary_label.sum()),proteins=part.uniprot.nunique(),ligands=part.full_inchikey.nunique()))
                z=part[['main_row_id','uniprot','family_component_id','full_inchikey','connectivity_key','binary_label']].copy()
                z['regime']=regime;z['outer_fold']=fold;z['partition']=name;members.append(z)
            for model in MODELS:
                v=checkpoint_selection(va.assign(p_allosteric=.5),regime,model)
                selections.append(dict(regime=regime,outer_fold=fold,model=model,**v))
            for metric,field in [('within_ligand','full_inchikey'),('within_protein','uniprot')]:
                support.append(dict(regime=regime,outer_fold=fold,metric=metric,**stats(te.assign(outer_fold=fold,p_allosteric=.5),field)))
        assert len(seen)==len(set(seen))==1365
    pd.DataFrame(counts).to_csv(PACKAGE/'data/SPLIT_COUNTS.tsv',sep='\t',index=False)
    pd.concat(members).to_csv(PACKAGE/'data/SPLIT_MEMBERSHIP.tsv.gz',sep='\t',index=False)
    pd.DataFrame(support).to_csv(PACKAGE/'data/METRIC_SUPPORT.tsv',sep='\t',index=False)
    pd.DataFrame(selections).to_csv(PACKAGE/'data/CHECKPOINT_SUPPORT.tsv',sep='\t',index=False)
    lig=d.groupby('full_inchikey').binary_label.agg(['size','sum']);lig.to_csv(PACKAGE/'data/LIGAND_COUNTS.tsv',sep='\t')
    # Read-only snapshots for the final matrix; never inputs to training or selection.
    legacy={'LEGACY_MATRIX_METRICS.tsv':'analysis/role_complete_pair_matrix/gpu_output/benchmark/aggregate/MATRIX_METRICS.tsv',
            'LEGACY_EXPLICIT_METRICS.tsv':'analysis/explicit_double_unseen/cpu_output/METRICS.tsv',
            'LEGACY_DOUBLE_ANCHORED_METRICS.tsv':'analysis/double_anchored_lofo/local_analysis/METRICS.tsv'}
    for name,rel in legacy.items():shutil.copyfile(str(root/rel),str(PACKAGE/'data'/name))
    files=[p for sub in ('scripts','data') for p in (PACKAGE/sub).iterdir() if p.is_file()]
    files += [PACKAGE/n for n in ('README.md','requirements-cpu.txt','requirements-gpu.txt')]
    hashes={str(p.relative_to(PACKAGE)):sha(p) for p in sorted(files)}
    contract=dict(status='validated',version=VERSION,files=hashes,manifest_digest=digest(hashes),training=TRAINING,
        source=str(src),source_sha256=sha(src),pocket_source_sha256=sha(pocket_src),
        source_rows=1365,source_proteins=543,source_ligands=122,source_families=251,
        expected_fits=480,expected_prediction_rows=131040,seeds=list(SEEDS),models=list(MODELS),regimes=list(REGIMES),
        main_regime='unseen_family',main_comparison='joint minus protein-only within ligand',
        ligand_only_pooled_also_reported=True,no_ligand_ratio_reweighting=True,no_ATP_ADP_exclusion=True,
        expected_fallback_fits=int(pd.DataFrame(selections).selection_fallback.sum()*3),
        protein_unseen_definition='Pfam family components held out; not a separate exact-protein-only split',
        preflight_cuda_tested=False,existing_cohorts_retrained=False)
    write_json(PACKAGE/'validation/CPU_CONTRACT.json',contract)
    print(json.dumps({k:contract[k] for k in ['expected_fits','expected_prediction_rows','expected_fallback_fits']},indent=2))


if __name__=='__main__':main()
