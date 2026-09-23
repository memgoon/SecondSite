"""Freeze 33 existing checkpoints and 11 missing arm/head combinations per scope."""
from common import *

def main():
    if (PACKAGE/'gpu_output/RUN_CONTRACT.json').exists():raise ValueError('Do not rebuild a started experiment')
    files={}
    def freeze(p):files[relative(p)]=sha(p)
    prior_validation=LA/'local_analysis/VALIDATION.json'
    val=read_json(prior_validation);assert val['status']=='validated'
    # CPU aggregation is not run on the GPU host. Ship its verified provenance
    # inside this package instead of requiring the CPU-only original there.
    write_json(PACKAGE/'data/PRIOR_LOCAL_VALIDATION.json',dict(
        source_path=relative(prior_validation),source_sha256=sha(prior_validation),
        source_required_on_gpu=False,validation=val))
    records={}
    for a,m in NEW:
        for s in SEEDS:
            d=deploy(a,m,s);r=read_json(d/'DEPLOY_REPORT.json');assert r['status']=='validated'
            if a=='ligand_anchored':
                i=r['identity'];assert i['model']==m and i['seed']==s and i['training_rows']==1365
            else:
                assert r['model']==m and r['seed']==s
                assert r['training_rows']==dict(every_pair=6854,protein_anchored=4637,double_anchored=395)[a]
                assert r['source_report_count']==15 and r['epoch_source_regimes']==['unseen_family']
            assert sha(d/'deploy.pt')==r['checkpoint_sha256']
            freeze(d/'DEPLOY_REPORT.json');freeze(d/'deploy.pt')
            records[a+'/'+m+'/'+str(s)]=dict(path=relative(d/'deploy.pt'),sha256=r['checkpoint_sha256'],
                report=relative(d/'DEPLOY_REPORT.json'),report_sha256=sha(d/'DEPLOY_REPORT.json'),
                state_key='state_dict' if a=='ligand_anchored' else 'model_state_dict')
    assert len(records)==33
    for p in COHORT.values():freeze(p)
    for n in ['scripts/infer_chembl.py','scripts/model_definitions.py','validation/CPU_CONTRACT.json','validation/CHEMBL_SOURCE_FINGERPRINT.json']:freeze(OLD/n)
    # The ligand-anchored architecture is the same implementation plus trailing whitespace.
    original=(OLD/'scripts/model_definitions.py').read_text().strip()
    other=ROOT/'analysis/ligand_anchored_benchmark/scripts/model_definitions.py'
    assert other.read_text().strip()==original;freeze(other)
    oldc=read_json(OLD/'validation/CPU_CONTRACT.json')
    for x in oldc['external_resources'].values():freeze(ROOT/x['path'])
    tasks_= []; summaries=[]; all_meta=[]
    for scope,i in scopes():
        p=source(scope,i);freeze(p);freeze(p.with_suffix('.json'))
        report=read_json(p.with_suffix('.json'));assert report['status']=='validated' and report['sha256']==sha(p)
        f=pd.read_csv(p,sep='\t',dtype={row_id(scope):str},low_memory=False)
        assert not f[row_id(scope)].duplicated().any()
        if scope=='full':assert not f.legacy_ood_label.eq(0).any()
        for a in ARMS[:3]:
            for m in MODELS:
                if m=='c3':continue
                for suffix in ('mean','sd'):
                    values=f[score(a,m,suffix)].to_numpy(float);ok=available(f,m)
                    assert np.isfinite(values[ok]).all() and np.isnan(values[~ok]).all()
        for a,m in NEW:assert score(a,m) not in f
        for start in range(0,len(f),CHUNK_ROWS):
            stop=min(start+CHUNK_ROWS,len(f))
            for a,m in NEW:
                tid='%s_%02d_%06d_%s_%s'%(scope,i,start,a,m)
                tasks_.append(dict(task_id=tid,scope=scope,shard=i,start=start,stop=stop,arm=a,model=m))
        summaries.append(dict(scope=scope,shard=i,rows=len(f),targets=f.uniprot.nunique()))
        # Compact input-only references, also used to audit local inference assembly.
        for x in ('selected_chain_available','pocket_available'):assert f[x].isin([0,1]).all()
        if scope=='reference':
            matched=f[f.selected_chain_available.eq(1)&f.pocket_available.eq(1)&f.weak2020_label.isin([0,1])]
            assert len(matched)==10628 and int(matched.weak2020_label.sum())==4941
    assert sum(x['rows'] for x in summaries if x['scope']=='full')==796165
    assert summaries[0]['rows']==29929
    # Only reference ligand tensors absent from the original tensor hash manifest need a local hash.
    reference=pd.read_csv(source('reference',0),sep='\t',usecols=['ligand_embedding_path'])
    prior_run=read_json(LA/'gpu_output/RUN_CONTRACT.json')
    assert prior_run['status']=='validated' and digest(prior_run['core'])==prior_run['run_fingerprint']
    freeze(LA/'gpu_output/RUN_CONTRACT.json');freeze(LA/'validation/CPU_CONTRACT.json')
    ligand_resources={};remote_only=0
    for path in sorted(reference.ligand_embedding_path.unique()):
        logical=relative(local(path));expected=prior_run['core']['inputs'][logical]
        if local(path).is_file():assert sha(local(path))==expected
        else:remote_only+=1
        ligand_resources[logical]=expected
    write_json(PACKAGE/'data/REFERENCE_LIGAND_HASHES.json',ligand_resources)
    write_json(PACKAGE/'data/CHECKPOINTS.json',records)
    write_json(PACKAGE/'data/TASKS.json',tasks_)
    write_frame(PACKAGE/'data/UNIVERSES.tsv',pd.DataFrame(summaries))
    mapping=[]
    for a in ARMS:
        for m in MODELS:
            mapping.append(dict(training_arm=a,training_pairs=dict(every_pair=6854,protein_anchored=4637,ligand_anchored=1365,double_anchored=395)[a],
                model=m,mean_column=score(a,m),sd_column=score(a,m,'sd'),
                global_rank_column=m+'_global_rank',within_target_rank_column=m+'_within_target_rank',
                individual_seed_columns_present=(a,m) in NEW or a=='ligand_anchored',
                source='new_inference' if (a,m) in NEW else 'reused_completed_prediction',
                calibrated_probability=False))
    write_frame(PACKAGE/'data/WEB_COLUMN_MAP.tsv',pd.DataFrame(mapping))
    for i in range(4):freeze(source('biochemical',i))
    for folder in ('scripts','data'):
        for p in (PACKAGE/folder).iterdir():
            if p.is_file():freeze(p)
    for n in ['README.md','requirements-cpu.txt','requirements-gpu.txt']:freeze(PACKAGE/n)
    payload=dict(files=files,arms=list(ARMS),models=list(MODELS),seeds=list(SEEDS),
        new_combinations=[list(x) for x in NEW],reused_combinations=21,checkpoints=33,
        training_fits=0,chunk_rows=CHUNK_ROWS,tasks=len(tasks_),test_rows=10628,
        same_external_rows=True,scoring='FP32 sigmoid; seed ensemble; ranking scores, not calibrated probabilities',
        reference_primary='same 10628 labelled rows; target-macro support 38 targets / 3497 rows',
        bootstrap='local CPU, 10000 paired target-cluster replicates, pointwise intervals',
        representation='unchanged selected target chain and pocket; not full-protein-sequence pilot',
        reference_tensors_verified_from_prior_gpu_manifest=len(ligand_resources),
        reference_tensors_not_present_locally=remote_only,
        biochemical='derived from completed full predictions on same 492 source-linked rows; no independent metabolite validation')
    write_json(PACKAGE/'validation/CPU_CONTRACT.json',dict(status='validated',version=VERSION,payload=payload,fingerprint=digest(payload)))
    print('PASS: 0 training fits; 33 existing checkpoints; %d resumable inference chunks'%len(tasks_))

if __name__=='__main__':main()
