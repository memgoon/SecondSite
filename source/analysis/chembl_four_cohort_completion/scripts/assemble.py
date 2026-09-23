"""Independent CPU reassembly. Existing predictions are copied, never replaced."""
from common import *

def checked_run():
    c=verify_contract();r=read_json(PACKAGE/'gpu_output/RUN_CONTRACT.json')
    assert r['status']=='validated' and r['core']['cpu_fingerprint']==c['fingerprint']
    assert digest(r['core'])==r['run_fingerprint']
    return c,r

def assemble(scope,shard,c=None,run=None):
    if c is None:c,run=checked_run()
    path=source(scope,shard)
    assert sha(path)==c['payload']['files'][relative(path)]
    f=pd.read_csv(path,sep='\t',dtype={row_id(scope):str},low_memory=False)
    records=read_json(PACKAGE/'data/CHECKPOINTS.json')
    all_tasks=tasks()
    for a,m in NEW:
        columns=new_columns(a,m);values=[];covered=[]
        selected=[t for t in all_tasks if (t['scope'],t['shard'],t['arm'],t['model'])==(scope,shard,a,m)]
        for t in selected:
            assert complete(t,run)
            p,r,_=task_paths(t);report=read_json(r)
            assert report['checkpoint_hashes']=={str(s):records[a+'/'+m+'/'+str(s)]['sha256'] for s in SEEDS}
            x=pd.read_csv(p,sep='\t',dtype={row_id(scope):str});ref=f.iloc[t['start']:t['stop']]
            assert x.columns.tolist()==[row_id(scope)]+columns
            assert x[row_id(scope)].tolist()==ref[row_id(scope)].tolist()
            assert len(x)==report['rows']==len(ref)
            ok=available(ref,m);assert report['available_rows']==int(ok.sum())
            v=x[columns].to_numpy(float)
            assert np.isfinite(v[ok]).all() and np.isnan(v[~ok]).all()
            assert ((v[ok,:4]>=0)&(v[ok,:4]<=1)).all()
            assert ((v[ok,4]>=0)&(v[ok,4]<=.500001)).all()
            assert np.allclose(v[:,:3].mean(1),v[:,3],equal_nan=True,atol=2e-7,rtol=0)
            assert np.allclose(v[:,:3].std(1),v[:,4],equal_nan=True,atol=2e-7,rtol=0)
            values.append(v);covered.extend(range(t['start'],t['stop']))
        assert covered==list(range(len(f)))
        v=np.concatenate(values)
        for j,col in enumerate(columns):
            assert col not in f
            f[col]=v[:,j]
    f=f.copy()
    for a in ARMS:
        for m in MODELS:
            ok=available(f,m)
            for suffix in ('mean','sd'):
                v=f[score(a,m,suffix)].to_numpy(float)
                assert np.isfinite(v[ok]).all() and np.isnan(v[~ok]).all()
    return f

def validate_parts():
    c,r=checked_run();rows=[]
    for scope,i in scopes():
        f=assemble(scope,i,c,r)
        rows.append(dict(scope=scope,shard=i,rows=len(f),heads_per_cohort=8,cohorts=4))
        print('Assembled/validated',scope,i,len(f),flush=True)
        del f
    result=dict(status='validated',run_fingerprint=r['run_fingerprint'],tasks=len(tasks()),
                sources=rows,old_scores_preserved=True,all_missing_predictions_filled=True)
    write_json(PACKAGE/'gpu_output/VALIDATION.json',result)
    return result

if __name__=='__main__':validate_parts()
