"""Local four-cohort evaluation and normalized web exports; no database mutations."""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from common import *
from assemble import assemble,checked_run
from bootstrap import job as bootstrap_job

DEST=PACKAGE/'local_analysis'
FRAME=None
SCOPE=None

def add_novelty(f):
    c=read_json(OLD/'validation/CPU_CONTRACT.json')
    records=read_json(ROOT/c['external_resources']['pfam_cache']['path'])['records']
    uid=f.uniprot.astype(str).str.split('-').str[0];key=f.connectivity_key.astype(str).str.upper()
    for arm,path in COHORT.items():
        d=pd.read_csv(path,sep='\t',usecols=['uniprot','connectivity_key'])
        proteins=set(d.uniprot.astype(str).str.split('-').str[0]);ligands=set(d.connectivity_key.astype(str).str.upper())
        complete=all(u in records and records[u].get('annotation_status')=='annotated' and records[u].get('pfam_ids') for u in proteins)
        union={p for u in proteins for p in records.get(u,{}).get('pfam_ids',[])}
        status={}
        for u in uid.unique():
            r=records.get(u,{});ids=set(r.get('pfam_ids',[]))
            status[u]='annotation_unavailable' if r.get('annotation_status')!='annotated' or not ids else 'seen' if ids&union else 'unseen' if complete else 'training_reference_incomplete'
        f['target_seen_training_'+arm]=uid.isin(proteins).astype(int)
        f['ligand_seen_training_'+arm]=key.isin(ligands).astype(int)
        f['pfam_status_training_'+arm]=uid.map(status)
    f['_id']=f[row_id('reference' if 'reference_row_id' in f else 'full')].astype(str)
    return f

def strata(f):
    # SAME strata for all four arms, using the broad every-pair training reference.
    yield 'all',f
    p,l='target_seen_training_every_pair','ligand_seen_training_every_pair'
    yield 'exact_target_unseen',f[f[p].eq(0)]
    yield 'ligand_connectivity_unseen',f[f[l].eq(0)]
    yield 'exact_double_novel',f[f[p].eq(0)&f[l].eq(0)]
    for name in ('seen','unseen','annotation_unavailable','training_reference_incomplete'):
        yield 'pfam_'+name,f[f.pfam_status_training_every_pair.eq(name)]

def enrichment_rows(f,col,meta):
    results=[]
    f=f[['_id','uniprot','has_allosteric_text',col]]
    for within in (False,True):
        x=f
        if within:
            good=x.groupby('uniprot').has_allosteric_text.sum();x=x[x.uniprot.isin(good[good>0].index)]
        n=len(x);positives=int(x.has_allosteric_text.sum());kind='within_text_positive_targets' if within else 'pooled'
        if n and positives:
            ranked=x.sort_values([col,'_id'],ascending=[False,True],kind='mergesort')
            if within:
                positions=ranked.groupby('uniprot').cumcount().to_numpy()
                sizes=ranked.groupby('uniprot')['_id'].transform('size').to_numpy()
        for fraction in (.001,.005,.01,.05):
            if not n or not positives:
                results.append(dict(meta,ranking=kind,top_fraction=fraction,universe_rows=n,positive_rows=positives,
                    top_rows=0,top_positive_rows=0,enrichment=np.nan,status='not_evaluated_no_positive_class'));continue
            top=ranked[positions<np.maximum(1,np.ceil(sizes*fraction))] if within else ranked.head(max(1,int(np.ceil(n*fraction))))
            hits=int(top.has_allosteric_text.sum())
            results.append(dict(meta,ranking=kind,top_fraction=fraction,universe_rows=n,positive_rows=positives,
                top_rows=len(top),top_positive_rows=hits,enrichment=(hits/len(top))/(positives/n),status='evaluated'))
    return results

def evaluate_arm(arm):
    f=FRAME;scope=SCOPE;matched=f[f.selected_chain_available.eq(1)&f.pocket_available.eq(1)]
    metrics=[];enrichment=[];candidate=[]
    export=f[[row_id(scope)]].copy()
    for model in MODELS:
        col=score(arm,model)
        for suffix in ['mean','sd']+['seed_%d'%s for s in SEEDS]:
            name=score(arm,model,suffix)
            if name in f:export[name]=f[name]
        export[model+'_global_rank']=f[col].rank(method='min',ascending=False)
        export[model+'_within_target_rank']=f.groupby('uniprot')[col].rank(method='min',ascending=False)
        # Common matched rows primary; model-available coverage remains a separate analysis.
        for universe,base in [('matched_selected_chain_and_pocket',matched),('model_available',f[f[col].notna()])]:
            for label,x in strata(base):
                meta=dict(scope=scope,training_arm=arm,model=model,universe=universe,stratum=label,
                          novelty_reference='every_pair',comparison_scope='identical_rows_across_arms')
                if scope=='reference':
                    z=x[x.weak2020_label.isin([0,1])];vals=[];rows=0
                    for _,g in z.groupby('uniprot'):
                        if g.weak2020_label.nunique()!=2:continue
                        vals.append(auc(g.weak2020_label,g[col]));rows+=len(g)
                    metrics.append(dict(meta,n_rows=len(z),positive_rows=int(z.weak2020_label.sum()),
                        auroc=auc(z.weak2020_label,z[col]),auprc=ap(z.weak2020_label,z[col]),
                        target_macro_auroc=float(np.mean(vals)) if vals else np.nan,
                        n_macro_groups=len(vals),n_macro_rows=rows))
                else:enrichment.extend(enrichment_rows(x,col,meta))
        if scope=='full':
            unknown=matched.loc[matched.has_allosteric_text.eq(0)&matched.has_orthosteric_text.eq(0)&matched[col].notna(),
                ['_id',col,'target_seen_training_every_pair','ligand_seen_training_every_pair']]
            for name,x in [('all_unannotated',unknown),('exact_double_novel',unknown[unknown.target_seen_training_every_pair.eq(0)&unknown.ligand_seen_training_every_pair.eq(0)])]:
                ids=x.sort_values([col,'_id'],ascending=[False,True],kind='mergesort').head(100)[['_id',col]].copy()
                ids.columns=['row_id','ranking_score'];ids['training_arm']=arm;ids['model']=model;ids['candidate_scope']=name;candidate.append(ids)
    write_frame(DEST/('RANKINGS_%s_%s.tsv.gz'%(scope.upper(),arm)),export)
    if candidate:write_frame(DEST/('CANDIDATES_%s.tsv.gz'%arm),pd.concat(candidate,ignore_index=True))
    return metrics,enrichment

def export_metadata(f,scope):
    # Sequence/vector cache paths are unnecessary for Django/PostgreSQL import.
    cols=[x for x in f if not x.startswith('p_') and x not in ('_id','ligand_embedding_path','protein_embedding_path')]
    write_frame(DEST/('PAIRS_%s.tsv.gz'%scope.upper()),f[cols])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
    if args.workers<1:raise ValueError('Positive worker count required')
    c,run=checked_run();DEST.mkdir(exist_ok=True)
    write_frame(DEST/'WEB_COLUMN_MAP.tsv',pd.read_csv(PACKAGE/'data/WEB_COLUMN_MAP.tsv',sep='\t'))
    global FRAME,SCOPE
    all_metrics=[];all_enrichment=[];ci=[]
    biochemical_ids=set()
    for i in range(4):
        path=source('biochemical',i);assert sha(path)==c['payload']['files'][relative(path)]
        biochemical_ids.update(pd.read_csv(path,sep='\t',usecols=['OOD_Row_ID'],dtype={'OOD_Row_ID':str}).OOD_Row_ID)
    assert len(biochemical_ids)==492
    bio=[]
    for scope in ('reference','full'):
        frames=[]
        for _,shard in [(s,i) for s,i in scopes() if s==scope]:
            frames.append(assemble(scope,shard,c,run));print('Verified predictions',scope,shard,flush=True)
        FRAME=add_novelty(pd.concat(frames,ignore_index=True));del frames;SCOPE=scope
        assert not FRAME._id.duplicated().any()
        broad=pd.read_csv(COHORT['every_pair'],sep='\t',usecols=['uniprot','connectivity_key'])
        banned=set(broad.uniprot.astype(str).str.split('-').str[0]+'|'+broad.connectivity_key.str.upper())
        assert not (FRAME.uniprot.astype(str).str.split('-').str[0]+'|'+FRAME.connectivity_key.str.upper()).isin(banned).any()
        export_metadata(FRAME,scope)
        # Four forked arm jobs share read-only input; no repeated full-file reads per head.
        with ProcessPoolExecutor(max_workers=min(4,args.workers),mp_context=mp.get_context('fork')) as pool:
            for metrics,enrichment in pool.map(evaluate_arm,ARMS):all_metrics.extend(metrics);all_enrichment.extend(enrichment)
        if scope=='reference':
            ref=FRAME[FRAME.selected_chain_available.eq(1)&FRAME.pocket_available.eq(1)&FRAME.weak2020_label.isin([0,1])].copy()
            assert len(ref)==10628 and int(ref.weak2020_label.sum())==4941
            assert (ref.groupby('uniprot').weak2020_label.nunique()==2).sum()==38
            jobs=[]
            for a in ARMS:
                for m in MODELS:
                    if a!='every_pair':
                        for metric in ('pooled','target_macro'):
                            meta=dict(training_arm=a,model=m,metric=metric,contrast='minus_every_pair_same_model')
                            jobs.append((meta,ref[['uniprot','weak2020_label',score(a,m),score('every_pair',m)]],score(a,m),score('every_pair',m),10000))
                    if m not in ('ligand','protein'):
                        meta=dict(training_arm=a,model=m,metric='target_macro',contrast='joint_minus_ligand_same_training')
                        jobs.append((meta,ref[['uniprot','weak2020_label',score(a,m),score(a,'ligand')]],score(a,m),score(a,'ligand'),10000))
            with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
                for i,r in enumerate(pool.map(bootstrap_job,jobs),1):
                    ci.append(r);print('Reference bootstrap %d/%d'%(i,len(jobs)),flush=True)
            assert len(ci)==72
            del jobs,ref
        else:bio=FRAME[FRAME._id.isin(biochemical_ids)].copy();assert len(bio)==492
        FRAME=None
    FRAME=bio;SCOPE='biochemical';export_metadata(bio,'biochemical')
    for arm in ARMS:
        metrics,enrichment=evaluate_arm(arm);all_enrichment.extend(enrichment)
    FRAME=None
    write_frame(DEST/'REFERENCE_METRICS.tsv',pd.DataFrame(all_metrics))
    write_frame(DEST/'TEXT_ENRICHMENT.tsv',pd.DataFrame(all_enrichment))
    write_frame(DEST/'REFERENCE_PAIRED_BOOTSTRAP.tsv',pd.DataFrame(ci))
    files={p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.name!='VALIDATION.json'}
    write_json(DEST/'VALIDATION.json',dict(status='validated',run_fingerprint=run['run_fingerprint'],
        arms=list(ARMS),models=list(MODELS),training_fits=0,new_checkpoints_used=33,
        reused_arm_model_combinations=21,new_arm_model_combinations=11,
        reference_rows=29929,matched_reference_labelled_rows=10628,reference_positive_rows=4941,
        target_macro_targets=38,target_macro_rows=3497,full_rows=796165,biochemical_rows=492,
        reference_bootstrap_comparisons=72,bootstrap_replicates=10000,all_old_scores_preserved=True,
        probability_calibrated=False,biochemical_independent_metabolite_validation=False,
        web_import_performed=False,sha256=files))
    print('PASS: four cohorts x eight heads; local CPU statistics and web files complete',flush=True)

if __name__=='__main__':main()
