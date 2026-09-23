"""GPU inference only. Frozen existing weights; dynamic file-locked chunk queue."""
import argparse
import fcntl
import importlib.util
import time
from common import *
import torch

spec=importlib.util.spec_from_file_location('completion_external_inputs',str(OLD/'scripts/infer_chembl.py'))
external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)

def settings():
    values=dict(batch_size=int(os.environ.get('BATCH_SIZE','16')),
                c3_batch_size=int(os.environ.get('C3_BATCH_SIZE','4')),
                cpu_threads=int(os.environ.get('CPU_THREADS','6')),single_input_batch_size=1)
    if min(values.values())<1:raise ValueError('Thread/batch counts must be positive')
    return values

def reference_arrays(frame):
    f=frame.copy();f['ligand_embedding_path']=f.ligand_embedding_path.map(lambda p:str(local(p)))
    return external.load_reference_arrays(f),f.ligand_embedding_path.astype(str).to_numpy()

def load_inputs(scope,shard):
    print('Loading frozen %s shard %d inputs'%(scope,shard),flush=True)
    f=pd.read_csv(source(scope,shard),sep='\t',dtype={row_id(scope):str},low_memory=False)
    if scope=='reference':arrays,keys=reference_arrays(f)
    else:
        paths=external.cache_paths(ROOT,shard,4)
        raw=pd.read_pickle(paths['frame']);raw['OOD_Row_ID']=raw.OOD_Row_ID.astype(str)
        if raw.OOD_Row_ID.duplicated().any():raise ValueError('Duplicate cache IDs')
        raw=raw.set_index('OOD_Row_ID').loc[f.OOD_Row_ID]
        assert raw.UniProt_ID.fillna('').astype(str).str.split('-').str[0].tolist()==f.uniprot.astype(str).tolist()
        assert raw.InChIKey14.astype(str).str.upper().tolist()==f.connectivity_key.astype(str).tolist()
        mapping=read_json(paths['mapping'])['lig_path_to_idx']
        keys=raw.Ligand_Embedding_Path.astype(str).to_numpy()
        indices=np.array([mapping[p] for p in keys],dtype=int)
        arrays=(external.load_mmap(paths['ligand']),external.load_mmap(paths['ligand_mask']),
                torch.zeros(1,1,1536),torch.ones(1,dtype=torch.long),indices,np.zeros(len(f),dtype=int))
    context=external.load_pocket_context(ROOT,OLD,f.uniprot)
    uids=f.uniprot.astype(str).str.split('-').str[0]
    for col,key in [('selected_chain_available','protein_cache'),('pocket_available','pocket_cache')]:
        assert f[col].astype(int).tolist()==uids.isin(context[key]).astype(int).tolist()
    lengths=np.array([min(4096,len(context['protein_cache'][u])) if u in context['protein_cache'] else 1 for u in uids])
    return f,arrays,keys,context,uids,lengths

def load_ensemble(arm,model,records):
    result=[]
    for seed in SEEDS:
        rec=records[arm+'/'+model+'/'+str(seed)]
        blob=torch.load(str(ROOT/rec['path']),map_location='cpu')
        if arm=='ligand_anchored':
            assert blob['identity']['model']==model and blob['identity']['seed']==seed
        else:
            assert blob['model']==model and blob['seed']==seed and blob['training_rows']==dict(every_pair=6854,protein_anchored=4637,double_anchored=395)[arm]
        m=external.make_model(model,hidden=256,dropout=.30,heads=4).cuda().eval()
        m.load_state_dict(blob[rec['state_key']],strict=True);result.append(m)
    return result

def make_input(arrays,context,uids,positions,model):
    batch=external.make_batch(*arrays[:4],arrays[4][positions],arrays[5][positions])
    if model!='ligand':
        batch,dest=external.attach_selected_chains(batch,uids.iloc[positions].tolist(),context,require_pocket=model.startswith('d'))
        assert len(dest)==len(positions)
    for field in ('ligand','protein','pocket'):
        n=max(1,int(batch[field+'_mask'].sum(1).max().item()))
        batch[field]=batch[field][:,:n];batch[field+'_mask']=batch[field+'_mask'][:,:n]
    return {k:v.cuda(non_blocking=True) for k,v in batch.items()}

def preflight():
    c=verify_contract(external=True);cfg=settings()
    assert torch.cuda.is_available(),'CUDA required'
    previous=read_json(OLD/'validation/CHEMBL_SOURCE_FINGERPRINT.json')
    required={relative(p) for i in range(4) for k,p in external.cache_paths(ROOT,i,4).items() if k in ('frame','mapping','ligand','ligand_mask')}
    inputs={}
    listed=[x for x in previous['ood_cache_artifacts'] if x['path'] in required]+previous['selected_chain_tensors']
    for i,x in enumerate(listed):
        if sha(ROOT/x['path'])!=x['sha256']:raise ValueError('Changed input '+x['path'])
        inputs[x['path']]=x['sha256']
        if i%100==0:print('Input hashes %d/%d'%(i+1,len(listed)),flush=True)
    for p,h in read_json(PACKAGE/'data/REFERENCE_LIGAND_HASHES.json').items():
        if sha(ROOT/p)!=h:raise ValueError('Reference ligand changed: '+p)
        inputs[p]=h
    records=read_json(PACKAGE/'data/CHECKPOINTS.json')
    # Test every requested ensemble, including longest available reference-chain C3 input.
    f,arrays,keys,context,uids,lengths=load_inputs('reference',0)
    for a,m in NEW:
        ok=np.flatnonzero(available(f,m));idx=np.array([ok[np.argmax(lengths[ok])]])
        if m=='c3':idx=np.repeat(idx,cfg['c3_batch_size'])
        batch=make_input(arrays,context,uids,idx,m)
        ensemble=load_ensemble(a,m,records)
        with torch.no_grad():
            for model in ensemble:
                with torch.autocast(device_type='cuda'):logits=model(batch)
                assert torch.isfinite(torch.sigmoid(logits.float())).all()
        del ensemble,batch;torch.cuda.empty_cache()
        print('Forward PASS:',a,m,flush=True)
    core=dict(cpu_fingerprint=c['fingerprint'],input_hashes=inputs,settings=cfg,
              torch_version=str(torch.__version__),numpy_version=np.__version__,pandas_version=pd.__version__)
    result=dict(status='validated',core=core,run_fingerprint=digest(core))
    path=PACKAGE/'gpu_output/RUN_CONTRACT.json'
    if path.exists() and read_json(path)!=result:raise ValueError('Changed inputs/settings/environment; refuse mixed resume')
    write_json(path,result);print('PREFLIGHT PASS: inference only; no model training',flush=True)

def infer(rank):
    c=verify_contract();run=read_json(PACKAGE/'gpu_output/RUN_CONTRACT.json')
    assert run['status']=='validated' and run['core']['cpu_fingerprint']==c['fingerprint']
    assert run['core']['settings']==settings() and run['core']['torch_version']==str(torch.__version__)
    records=read_json(PACKAGE/'data/CHECKPOINTS.json');models={};single_cache={};cached=None;loaded=None;done=0
    for task in tasks():
        p,rep,lock=task_paths(task);p.parent.mkdir(parents=True,exist_ok=True)
        if complete(task,run):continue
        with lock.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if complete(task,run):continue
            start_time=time.monotonic();a,m=task['arm'],task['model']
            scope,shard=task['scope'],task['shard']
            if cached!=(scope,shard):
                loaded=None
                loaded=load_inputs(scope,shard);cached=(scope,shard)
            f,arrays,input_keys,context,uids,lengths=loaded
            sub=f.iloc[task['start']:task['stop']].copy();ok=available(sub,m)
            selected=np.flatnonzero(ok)+task['start']
            single=m in ('ligand','protein')
            keys=uids.to_numpy() if m=='protein' else input_keys
            # Separate ligand cache namespaces: a path in two independently frozen
            # tensor caches is not by itself proof of identical stored values.
            lookup=single_cache.setdefault((a,m,scope,shard) if m=='ligand' else (a,m),{})
            if single:
                selected=selected[~pd.Series(keys[selected]).duplicated().to_numpy()]
                selected=np.array([i for i in selected if str(keys[i]) not in lookup],dtype=int)
            selected=selected[np.argsort(lengths[selected],kind='stable')]
            batch_size=1 if single else settings()['c3_batch_size'] if m=='c3' else settings()['batch_size']
            values=np.full((len(sub),3),np.nan,dtype=np.float32)
            if len(selected) and (a,m) not in models:models[a,m]=load_ensemble(a,m,records)
            with torch.no_grad():
                for start in range(0,len(selected),batch_size):
                    idx=selected[start:start+batch_size];batch=make_input(arrays,context,uids,idx,m)
                    scores=[]
                    for model in models[a,m]:
                        with torch.autocast(device_type='cuda'):logits=model(batch)
                        scores.append(torch.sigmoid(logits.float()).cpu().numpy().reshape(-1))
                    scores=np.stack(scores,axis=1)
                    if single:
                        for i,v in zip(idx,scores):lookup[str(keys[i])]=v.copy()
                    else:values[idx-task['start']]=scores
                    if start%(batch_size*100)==0:print(task['task_id'],'rows %d/%d'%(start+len(idx),len(selected)),flush=True)
                    del batch
            if single:
                for i in np.flatnonzero(ok):values[i]=lookup[str(keys[i+task['start']])]
            assert np.isfinite(values[ok]).all() and np.isnan(values[~ok]).all()
            x=pd.DataFrame({row_id(scope):sub[row_id(scope)].to_numpy()})
            for j,s in enumerate(SEEDS):x[score(a,m,'seed_%d'%s)]=values[:,j]
            x[score(a,m)]=values.mean(axis=1);x[score(a,m,'sd')]=values.std(axis=1)
            write_frame(p,x)
            write_json(rep,dict(status='validated',task=task,run_fingerprint=run['run_fingerprint'],
                rows=len(x),available_rows=int(ok.sum()),sha256=sha(p),elapsed_seconds=time.monotonic()-start_time,
                checkpoint_hashes={str(s):records[a+'/'+m+'/'+str(s)]['sha256'] for s in SEEDS}))
            done+=1;write_json(PACKAGE/('gpu_output/progress/worker_%d.json'%rank),dict(last_task=task['task_id'],completed_this_worker=done,run_fingerprint=run['run_fingerprint']))
            print('DONE',task['task_id'],'%.1f sec'%(time.monotonic()-start_time),flush=True)
    print('Worker %d finished available tasks'%rank,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','infer']);p.add_argument('--rank',type=int,default=0);args=p.parse_args()
    torch.set_num_threads(settings()['cpu_threads'])
    if args.stage=='preflight':preflight()
    else:infer(args.rank)

if __name__=='__main__':main()
