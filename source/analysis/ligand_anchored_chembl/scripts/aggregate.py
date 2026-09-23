"""CPU-only validation, matched-row comparisons and web-ready ranking export."""
import math
from shared import *

def top(frame,score,fraction):
    k=max(1,int(math.ceil(len(frame)*fraction)))
    return frame.sort_values([score,'_id'],ascending=[False,True],kind='mergesort').head(k)

def enrichment(frame,score,fraction,within=False):
    if within:
        targets=frame.groupby('uniprot').has_allosteric_text.sum()
        frame=frame[frame.uniprot.isin(targets[targets>0].index)]
    n=len(frame);pos=int(frame.has_allosteric_text.sum())
    if not n or not pos:
        return dict(universe_rows=n,positive_rows=pos,top_rows=0,top_positive_rows=0,enrichment=np.nan,
                    status='not_evaluated_no_positive_class')
    chosen=pd.concat([top(x,score,fraction) for _,x in frame.groupby('uniprot',sort=False)]) if within else top(frame,score,fraction)
    hit=int(chosen.has_allosteric_text.sum())
    return dict(universe_rows=n,positive_rows=pos,top_rows=len(chosen),top_positive_rows=hit,
                enrichment=(hit/len(chosen))/(pos/n),status='evaluated')

def strata(frame,reference):
    if reference=='ligand_anchored':p,l,f='protein_seen_ligand_anchored','ligand_seen_ligand_anchored','pfam_family_status_ligand_anchored'
    elif reference=='every_pair':p,l,f='protein_seen_every_pair','ligand_seen_every_pair','pfam_family_status_every_pair'
    else:
        p='protein_seen_protein_anchored' if 'protein_seen_protein_anchored' in frame else 'protein_seen_general'
        l='ligand_seen_protein_anchored' if 'ligand_seen_protein_anchored' in frame else 'ligand_seen_general'
        f='pfam_family_status_general'
    yield 'all',frame
    yield 'exact_target_unseen',frame[frame[p].eq(0)]
    yield 'ligand_connectivity_unseen',frame[frame[l].eq(0)]
    yield 'exact_double_novel',frame[frame[p].eq(0)&frame[l].eq(0)]
    for value in ('seen','unseen','annotation_unavailable','training_reference_incomplete'):
        yield 'pfam_'+value,frame[frame[f].eq(value)]

def validate():
    c=verify_contract();r=read_json(PACKAGE/'gpu_output/RUN_CONTRACT.json')
    assert r['status']=='validated' and r['core']['cpu_fingerprint']==c['fingerprint'] and digest(r['core'])==r['run_fingerprint']
    for m in MODELS:
        for seed in SEEDS:
            p=deploy_dir(m,seed);report=read_json(p/'DEPLOY_REPORT.json');ident=report['identity']
            assert report['status']=='validated' and ident['run_fingerprint']==r['run_fingerprint']
            assert ident['epoch_source']==c['payload']['epochs'][m] and ident['model']==m and ident['seed']==seed
            assert ident['epochs']==c['payload']['epochs'][m]['epochs'] and ident['training_rows']==1365
            assert ident['training']==bench.TRAINING
            assert sha(p/'deploy.pt')==report['checkpoint_sha256'] and sha(p/'history.tsv')==report['history_sha256']
    cohort=bench.read_data();oldc=read_json(OLD/'validation/CPU_CONTRACT.json')
    pfam=read_json(ROOT/oldc['external_resources']['pfam_cache']['path'])
    broad=pd.read_csv(OLD/'data/EVERY_PAIR.tsv.gz',sep='\t',usecols=['uniprot','connectivity_key'])
    banned=set(broad.uniprot.str.split('-').str[0]+'|'+broad.connectivity_key.str.upper())
    frames={k:[] for k in ('reference','full','biochemical')}
    for scope,i in tasks():
        p=output(scope,i);rep=read_json(p.with_suffix('.json'))
        assert rep['status']=='validated' and rep['run_fingerprint']==r['run_fingerprint'] and rep['scope']==scope and rep['shard']==i
        assert rep['sha256']==sha(p) and rep['source_sha256']==sha(prior(scope,i))
        assert rep['checkpoint_hashes']==deployment_hashes(scope)
        old=prior_frame(scope,i)
        new=pd.read_csv(p,sep='\t',dtype={row_id(scope):str},low_memory=False)
        assert len(new)==len(old)==rep['rows']
        # Includes original scores, labels and structural availability: no row-based accidental joins.
        pd.testing.assert_frame_equal(new[old.columns],old,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        recomputed=novelty(old.copy(),cohort,pfam)
        for col in ['protein_seen_ligand_anchored','ligand_seen_ligand_anchored','pfam_family_status_ligand_anchored']:
            assert new[col].tolist()==recomputed[col].tolist()
        assert not (new.uniprot.str.split('-').str[0]+'|'+new.connectivity_key.str.upper()).isin(banned).any()
        validate_scores(new,scope)
        for m in names(scope):
            seedvalues=new[['p_ligand_anchored_%s_seed_%d'%(m,s) for s in SEEDS]].to_numpy(float)
            assert np.allclose(seedvalues.mean(1),new['p_ligand_anchored_%s_mean'%m],equal_nan=True,atol=2e-7)
            assert np.allclose(seedvalues.std(1),new['p_ligand_anchored_%s_sd'%m],equal_nan=True,atol=2e-7)
        new['_id']=new[row_id(scope)].astype(str);frames[scope].append(new)
    frames={k:pd.concat(v,ignore_index=True) for k,v in frames.items()}
    for scope,f in frames.items():
        assert not f._id.duplicated().any(),scope
    assert len(frames['full'])==796165 and len(frames['reference'])==29929
    assert set(frames['biochemical']._id).issubset(set(frames['full']._id))
    return c,r,frames

def main():
    c,r,frames=validate();dest=PACKAGE/'local_analysis';dest.mkdir(exist_ok=True)
    rows=[];refrows=[];candidates=[]
    arms=['every_pair','general','ligand_anchored']
    for scope,frame in frames.items():
        matched=frame[frame.selected_chain_available.eq(1)&frame.pocket_available.eq(1)]
        available_arms=arms+(['role_complete'] if scope=='biochemical' else [])
        for arm in available_arms:
            for m in names(scope):
                score='p_%s_%s_mean'%(arm,m)
                # Identical primary strata across training arms; own novelty reported separately.
                universes=[('matched_selected_chain_and_pocket',matched),('model_available',frame[frame[score].notna()])]
                for universe,baseframe in universes:
                    for ref in ['general','arm_relative']:
                        refarm=arm if ref=='arm_relative' else 'general'
                        # Role-complete own-novelty is not a new claim in this extension.
                        if refarm=='role_complete':continue
                        for label,x in strata(baseframe,refarm):
                            meta=dict(scope=scope,training_arm=arm,model=m,universe=universe,novelty_reference=refarm,
                                      stratum=label,comparison_scope='arm_specific_rows' if ref=='arm_relative' else 'identical_rows_across_arms')
                            if scope=='reference':
                                z=x[x.weak2020_label.isin([0,1])]
                                yy=z.weak2020_label.to_numpy(int);ss=z[score].to_numpy(float)
                                vals=[auc(g.weak2020_label,g[score]) for _,g in z.groupby('uniprot') if g.weak2020_label.nunique()==2]
                                used=sum(len(g) for _,g in z.groupby('uniprot') if g.weak2020_label.nunique()==2)
                                refrows.append(dict(meta,n_rows=len(z),positive_rows=int(yy.sum()),auroc=auc(yy,ss),auprc=ap(yy,ss),
                                    target_macro_auroc=float(np.mean(vals)) if vals else np.nan,n_macro_groups=len(vals),n_macro_rows=used))
                            else:
                                for fraction in [.001,.005,.01,.05]:
                                    for within in [False,True]:
                                        rows.append(dict(meta,top_fraction=fraction,ranking='within_text_positive_targets' if within else 'pooled',
                                                         **enrichment(x,score,fraction,within)))
        # Web export contains the entire universe, not only high scores or positives.
        webcols=[x for x in frame if not x.startswith('p_') and x not in ['_id','ligand_embedding_path','protein_embedding_path']]
        webcols += [x for x in frame if x.startswith('p_ligand_anchored_')]
        web=frame[webcols].copy()
        web['training_cohort']='ligand_anchored';web['training_pairs']=1365
        web['run_fingerprint']=r['run_fingerprint']
        for m in names(scope):
            score='p_ligand_anchored_%s_mean'%m
            web[m+'_pooled_rank']=frame[score].rank(method='min',ascending=False)
            web[m+'_within_target_rank']=frame.groupby('uniprot')[score].rank(method='min',ascending=False)
            if scope!='reference':
                unlabelled=frame[frame.has_allosteric_text.eq(0)&frame.has_orthosteric_text.eq(0)&frame[score].notna()]
                for label,x in [('all',unlabelled),('exact_double_novel',unlabelled[unlabelled.protein_seen_ligand_anchored.eq(0)&unlabelled.ligand_seen_ligand_anchored.eq(0)])]:
                    z=x.sort_values([score,'_id'],ascending=[False,True],kind='mergesort').head(5000).copy()
                    z['ranked_model']=m;z['candidate_scope']=label;z['screen_scope']=scope
                    candidates.append(z)
        atomic_frame(dest/('WEB_%s_RANKINGS.tsv.gz'%scope.upper()),web)
    atomic_frame(dest/'REFERENCE_METRICS.tsv',pd.DataFrame(refrows))
    atomic_frame(dest/'TEXT_ENRICHMENT.tsv',pd.DataFrame(rows))
    atomic_frame(dest/'TOP_UNANNOTATED_CANDIDATES.tsv.gz',pd.concat(candidates,ignore_index=True))
    outputs=['REFERENCE_METRICS.tsv','TEXT_ENRICHMENT.tsv','TOP_UNANNOTATED_CANDIDATES.tsv.gz']+['WEB_%s_RANKINGS.tsv.gz'%s.upper() for s in frames]
    write_json(dest/'VALIDATION.json',dict(status='validated',version=VERSION,run_fingerprint=r['run_fingerprint'],
        rows={k:len(v) for k,v in frames.items()},all_prior_metadata_and_scores_unchanged=True,matched_comparison=True,
        full_models=list(FULL_MODELS),biochemical_models=list(MODELS),score_is_calibrated_probability=False,
        enrichment_tie_rule='descending score then ascending frozen row ID; within-target top-k minimum one',
        identity_constant_ranks='single-input ties have no within-identity biological ranking interpretation',
        bootstrap_performed=False,files={name:sha(dest/name) for name in outputs}))
    print('CPU aggregation validated. Web files: '+str(dest),flush=True)

if __name__=='__main__':main()
