"""CPU tests: frozen scope, real task coverage, synthetic output assembly and bootstrap."""
import ast
import tempfile
from unittest.mock import patch
from common import *
from bootstrap import prepare,weighted_auc,job
from aggregate import enrichment_rows
import assemble as assembly
import common as common_module
import aggregate as analysis

def main():
    c=verify_contract();ts=tasks();assert len(ts)==c['payload']['tasks']
    snapshot=read_json(PACKAGE/'data/PRIOR_LOCAL_VALIDATION.json')
    assert snapshot['validation']['status']=='validated'
    assert snapshot['source_required_on_gpu'] is False
    assert snapshot['source_path'] not in c['payload']['files']
    assert relative(PACKAGE/'data/PRIOR_LOCAL_VALIDATION.json') in c['payload']['files']
    # Missing external files must be reported together before reading any bytes.
    fake_payload=dict(files={'missing_fixture_a.pt':'a','missing_fixture_b.pt':'b'})
    fake_contract=dict(status='validated',version=VERSION,payload=fake_payload,fingerprint=digest(fake_payload))
    with patch.object(common_module,'read_json',return_value=fake_contract),\
         patch.object(Path,'is_file',return_value=False),patch.object(common_module,'sha') as hash_mock:
        try:verify_contract(external=True)
        except FileNotFoundError as exc:
            assert 'missing_fixture_a.pt' in str(exc) and 'missing_fixture_b.pt' in str(exc)
        else:raise AssertionError('Missing external paths accepted')
        hash_mock.assert_not_called()
    assert len(NEW)==11 and c['payload']['training_fits']==0
    counts=pd.read_csv(PACKAGE/'data/UNIVERSES.tsv',sep='\t')
    for scope,i in scopes():
        n=int(counts[(counts.scope==scope)&(counts.shard==i)].rows.iloc[0])
        for a,m in NEW:
            selected=[t for t in ts if (t['scope'],t['shard'],t['arm'],t['model'])==(scope,i,a,m)]
            assert [k for t in selected for k in range(t['start'],t['stop'])]==list(range(n))
    assert len({t['task_id'] for t in ts})==len(ts)
    assert auc([0,1],[.5,.5])==.5
    y=np.array([0,1,0,1]);s=np.array([.2,.5,.5,.1]);target=np.array([0,0,1,1])
    for mult in ([2,1],[1,2],[0,3],[3,0]):
        idx=np.concatenate([np.flatnonzero(target==k) for k,n in enumerate(mult) for _ in range(n)])
        got=weighted_auc(prepare(y,s,target),np.array([mult]))[0]
        assert np.isclose(got,auc(y[idx],s[idx]))
    f=pd.DataFrame(dict(uniprot=['A','A','B','B'],weak2020_label=y,left=s,right=s))
    for metric in ('pooled','target_macro'):
        r=job((dict(metric=metric,test='identical'),f,'left','right',10000))
        assert r['delta']==r['ci_low']==r['ci_high']==0 and r['valid_replicates']==10000
    f=pd.DataFrame(dict(uniprot=['A','A','B','B'],_id=['0','1','2','3'],has_allosteric_text=[1,0,1,0],p=[.9,.1,.8,.2]))
    e=enrichment_rows(f,'p',{});small=[x for x in e if x['ranking']=='within_text_positive_targets' and x['top_fraction']==.001][0]
    assert small['top_rows']==2 and small['top_positive_rows']==2
    f.has_allosteric_text=0
    assert all(x['status']=='not_evaluated_no_positive_class' for x in enrichment_rows(f,'p',{}))
    # Exercise REAL assembler on temporary synthetic predictions, including an NA target.
    with tempfile.TemporaryDirectory(prefix='secondsite_completion_test_') as temp:
        root=Path(temp);old=root/'source.tsv.gz'
        f=pd.DataFrame(dict(reference_row_id=['0','1','2'],uniprot=['A','B','C'],
            selected_chain_available=[1,0,1],pocket_available=[1,0,0]))
        for a in ARMS[:3]:
            for m in MODELS:
                if m=='c3':continue
                ok=available(f,m)
                f[score(a,m)]=np.where(ok,.6,np.nan);f[score(a,m,'sd')]=np.where(ok,.02,np.nan)
        write_frame(old,f)
        records=read_json(PACKAGE/'data/CHECKPOINTS.json');own=[]
        def paths(t):return root/(t['task_id']+'.tsv.gz'),root/(t['task_id']+'.json'),root/(t['task_id']+'.lock')
        for a,m in NEW:
            t=dict(task_id=a+'_'+m,scope='reference',shard=0,start=0,stop=3,arm=a,model=m);own.append(t)
            x=f[['reference_row_id']].copy();ok=available(f,m);v=np.tile([.4,.5,.6],(3,1));v[~ok]=np.nan
            for j,s in enumerate(SEEDS):x[score(a,m,'seed_%d'%s)]=v[:,j]
            x[score(a,m)]=v.mean(1);x[score(a,m,'sd')]=v.std(1)
            p,r,_=paths(t);write_frame(p,x)
            write_json(r,dict(status='validated',task=t,run_fingerprint='synthetic',sha256=sha(p),
                rows=3,available_rows=int(ok.sum()),checkpoint_hashes={str(s):records[a+'/'+m+'/'+str(s)]['sha256'] for s in SEEDS}))
        # Explicit patch points are confined to synthetic test scope, never real output paths.
        fake=dict(payload=dict(files={str(old):sha(old)}))
        with patch.object(assembly,'source',return_value=old),patch.object(assembly,'relative',side_effect=str),\
             patch.object(assembly,'tasks',return_value=own),patch.object(assembly,'task_paths',side_effect=paths),\
             patch.object(common_module,'task_paths',side_effect=paths):
            run={'run_fingerprint':'synthetic'}
            result=assembly.assemble('reference',0,fake,run)
            pd.testing.assert_frame_equal(result[f.columns],f)
            assert score('double_anchored','c3') in result and result[score('double_anchored','c3')].isna().sum()==1
            probe=result.copy();probe['weak2020_label']=[0,1,1];probe['_id']=probe.reference_row_id
            probe['target_seen_training_every_pair']=[1,0,0];probe['ligand_seen_training_every_pair']=[1,0,0]
            probe['pfam_status_training_every_pair']=['seen','annotation_unavailable','unseen']
            with patch.object(analysis,'FRAME',probe),patch.object(analysis,'SCOPE','reference'),patch.object(analysis,'DEST',root):
                for arm in ARMS:
                    metrics,_=analysis.evaluate_arm(arm)
                    assert len(metrics)==8*2*8
                    exported=pd.read_csv(root/('RANKINGS_REFERENCE_%s.tsv.gz'%arm),sep='\t')
                    assert len(exported)==3 and all(score(arm,m) in exported for m in MODELS)
            p,r,_=paths(own[0]);bad=pd.read_csv(p,sep='\t',dtype={'reference_row_id':str});bad.loc[0,'reference_row_id']='wrong';write_frame(p,bad)
            assert not complete(own[0],run)
            meta=read_json(r);meta['sha256']=sha(p);write_json(r,meta)
            try:assembly.assemble('reference',0,fake,run)
            except AssertionError:pass
            else:raise AssertionError('Mismatched row ID accepted')
    for p in (PACKAGE/'scripts').iterdir():
        if p.suffix=='.py':ast.parse(p.read_text())
    write_json(PACKAGE/'validation/STATIC_VALIDATION.json',dict(status='validated',
        inference_tasks=len(ts),training_fits=0,checkpoints=33,complete_task_coverage=True,
        weighted_bootstrap_ties_and_multiplicity=True,minimum_one_topk=True,
        synthetic_assembly_preserves_old_scores_and_NA=True,row_mismatch_rejected=True,
        all_four_cohort_web_export_schema_tested=True,
        local_validation_bundled_without_gpu_source_dependency=True,
        missing_contract_paths_reported_before_hashing=True,
        gpu_execution='not run locally; required preflight is on GPU host'))
    print('PASS: scope, task coverage, bootstrap, top-k, NA and row-join regression tests')

if __name__=='__main__':main()
