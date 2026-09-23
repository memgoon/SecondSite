"""Freeze 120 existing Pfam CV reports and the external scoring universes."""
from shared import *

OWN=('shared.py','build_contract.py','runtime.py','aggregate.py','transfer.py','status.py','test_pipeline.py',
     'run_gpu.sh','send_gpu_input.sh','fetch_gpu_return.sh')

def main():
    bench.verify_contract()
    run=read_json(BENCH/'gpu_output/RUN_CONTRACT.json')
    cohort=bench.read_data()
    assert len(cohort)==1365 and cohort.uniprot.nunique()==543 and cohort.full_inchikey.nunique()==122
    assert cohort.binary_label.value_counts().to_dict()=={0:960,1:405}
    files={};sources={}
    def freeze(p):files[relative(p)]=sha(p)
    for m in MODELS:
        rows=[]
        for s in SEEDS:
            for f in range(5):
                job=('unseen_family',m,s,f)
                r=bench.compatible_report(job,run,hashes=False)
                if r is None:raise ValueError('Invalid CV report '+str(job))
                p=bench.fit_dir(job)/'FIT_REPORT.json';freeze(p)
                history=bench.fit_dir(job)/'history.tsv';freeze(history)
                best,_=bench.replay_history(pd.read_csv(history,sep='\t'))
                assert best==r['best_epoch'] and sha(history)==r['sha256']['history.tsv']
                rows.append(dict(path=relative(p),sha256=sha(p),model=m,seed=s,fold=f,best_epoch=best,
                                 artifact_hashes=r['sha256']))
        sources[m]=dict(epochs=max(1,int(np.median([r['best_epoch'] for r in rows]))),reports=rows)
    for n in OWN:freeze(PACKAGE/'scripts'/n)
    for n in ['README.md','requirements-cpu.txt','requirements-gpu.txt']:freeze(PACKAGE/n)
    for n in ['common.py','metrics.py','base_trainer.py','model_definitions.py','gpu_runtime.py','queue_jobs.py']:freeze(BENCH/'scripts'/n)
    for n in ['data/COHORT.tsv.gz','data/POCKET_INDICES.json','validation/CPU_CONTRACT.json','gpu_output/RUN_CONTRACT.json']:freeze(BENCH/n)
    for n in ['scripts/infer_chembl.py','scripts/model_definitions.py','validation/CPU_CONTRACT.json','validation/CHEMBL_SOURCE_FINGERPRINT.json','data/EVERY_PAIR.tsv.gz']:freeze(OLD/n)
    oldc=read_json(OLD/'validation/CPU_CONTRACT.json')
    for r in oldc['external_resources'].values():freeze(ROOT/r['path'])
    for scope,i in tasks():freeze(prior(scope,i))
    payload=dict(files=files,epochs=sources,training=bench.TRAINING,cohort_rows=1365,deploy_fits=24,
        full_models=list(FULL_MODELS),biochemical_models=list(MODELS),seeds=list(SEEDS),
        single_input_inference='deduplicate actual input; batch one; broadcast',
        scoring='mean and population SD of three FP32 sigmoid scores; not calibrated allostery probabilities',
        comparison='same frozen external rows; broad 6854-pair blacklist; exclude legacy_ood_label=0 in full/biochemical as in prior aggregation; retain all heads',
        c3_scope='source-linked biochemical subset only; same scope as prior deployment')
    result=dict(status='validated',version=VERSION,payload=payload,fingerprint=digest(payload))
    path=PACKAGE/'validation/CPU_CONTRACT.json'
    if (PACKAGE/'gpu_output/RUN_CONTRACT.json').exists() and path.exists() and read_json(path)!=result:
        raise ValueError('Cannot change a started run contract; preserve output and version package')
    write_json(path,result)
    print(json.dumps(dict(status='validated',files=len(files),epochs={m:x['epochs'] for m,x in sources.items()},fits=24),indent=2))

if __name__=='__main__':main()
