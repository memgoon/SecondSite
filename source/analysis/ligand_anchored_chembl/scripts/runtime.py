"""GPU deployment and inference; old pipelines are read-only dependencies."""
import argparse
import importlib.util
import time
from shared import *
import torch
from torch.utils.data import DataLoader
import gpu_runtime as training
import base_trainer as base
import model_definitions as models

spec=importlib.util.spec_from_file_location('old_chembl_inputs',str(OLD/'scripts/infer_chembl.py'))
external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)

def save_model(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(value,str(tmp));os.replace(str(tmp),str(path))

def preflight():
    c=verify_contract()
    assert torch.cuda.is_available(),'CUDA unavailable'
    snapshot={}
    def check(p,expected=None):
        p=Path(p);actual=sha(p)
        if expected and expected!=actual:raise ValueError('Input hash mismatch: '+str(p))
        snapshot[relative(p)]=actual
    for path,h in read_json(BENCH/'gpu_output/RUN_CONTRACT.json')['input_hashes'].items():check(local(path),h)
    for e in c['payload']['epochs'].values():
        for r in e['reports']:
            parent=(ROOT/r['path']).parent
            for name,h in r['artifact_hashes'].items():check(parent/name,h)
    previous=read_json(OLD/'validation/CHEMBL_SOURCE_FINGERPRINT.json')
    # The mixed legacy protein caches are not used. Hash only explicitly required cache artifacts.
    required={relative(p) for i in range(4) for k,p in external.cache_paths(ROOT,i,4).items() if k in ('frame','mapping','ligand','ligand_mask')}
    for r in previous['ood_cache_artifacts']+previous['selected_chain_tensors']:
        if r['path'] in required or r in previous['selected_chain_tensors']:check(ROOT/r['path'],r['sha256'])
    reference=pd.read_csv(prior('reference'),sep='\t',usecols=['ligand_embedding_path'])
    for path in sorted(reference.ligand_embedding_path.unique()):check(local(path))
    print('Explicit input hashes complete: %d files'%len(snapshot),flush=True)
    frame,samples=training.load_inputs()
    batch=training.gpu_batch(base.collate_pairs([samples[str(frame.iloc[0].main_row_id)]]),torch.device('cuda:0'))
    for m in MODELS:
        model=training.make_model(m).cuda().eval()
        with torch.no_grad(),torch.cuda.amp.autocast():logits=model(batch)
        assert torch.isfinite(torch.sigmoid(logits.float())).all()
        del model
    core=dict(cpu_fingerprint=c['fingerprint'],inputs=snapshot,torch_version=str(torch.__version__),
              numpy_version=np.__version__,pandas_version=pd.__version__,full_batch_size=16,biochemical_batch_size=2)
    result=dict(status='validated',core=core,run_fingerprint=digest(core))
    p=PACKAGE/'gpu_output/RUN_CONTRACT.json'
    if p.exists() and read_json(p)!=result:raise ValueError('Run inputs/environment changed; refuse mixed resume')
    write_json(p,result);print('PREFLIGHT PASS: 24 new deployment fits; 9 inference tasks',flush=True)

def run_contract():
    c=verify_contract();r=read_json(PACKAGE/'gpu_output/RUN_CONTRACT.json')
    assert r['status']=='validated' and r['run_fingerprint']==digest(r['core']) and r['core']['cpu_fingerprint']==c['fingerprint']
    assert r['core']['torch_version']==str(torch.__version__)
    return c,r

def deploy(rank,world):
    c,r=run_contract();frame,samples=training.load_inputs()
    weights=base.class_and_protein_balanced_weights(frame)
    data=[dict(samples[str(k)],weight=float(w)) for k,w in zip(frame.main_row_id,weights)]
    for index,(m,seed) in enumerate((m,s) for m in MODELS for s in SEEDS):
        if index%world!=rank:continue
        path=deploy_dir(m,seed);ep=c['payload']['epochs'][m]
        identity=dict(model=m,seed=seed,epochs=ep['epochs'],training_rows=len(frame),epoch_source=ep,
                      run_fingerprint=r['run_fingerprint'],training=bench.TRAINING)
        report=path/'DEPLOY_REPORT.json';checkpoint=path/'deploy.pt'
        if report.exists():
            old=read_json(report)
            if old.get('identity')!=identity:raise ValueError('Incompatible deploy resume '+str(path))
            if old.get('status')=='validated' and checkpoint.exists() and sha(checkpoint)==old['checkpoint_sha256']:
                print('SKIP deploy %s %d'%(m,seed),flush=True);continue
        base.seed_everything(seed);model=training.make_model(m).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        scaler=torch.cuda.amp.GradScaler()
        loader=DataLoader(data,batch_size=1 if m=='c3' else 6,shuffle=True,
            generator=torch.Generator().manual_seed(seed),collate_fn=base.collate_pairs,num_workers=0,pin_memory=True)
        history=[];start=time.time()
        for epoch in range(1,ep['epochs']+1):
            model.train();losses=[]
            for batch in loader:
                tensors=training.gpu_batch(batch,torch.device('cuda:0'));optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    logits=model(tensors)
                    per=torch.nn.functional.binary_cross_entropy_with_logits(logits,tensors['label'],reduction='none')
                    loss=(per*tensors['weight']).sum()/tensors['weight'].sum().clamp_min(1e-8)
                if not torch.isfinite(loss):raise ValueError('Nonfinite deploy loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                scaler.step(optimizer);scaler.update();losses.append(float(loss.detach().cpu()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),elapsed=time.time()-start))
            print('deploy %s seed=%d epoch=%d/%d loss=%.6f'%(m,seed,epoch,ep['epochs'],np.mean(losses)),flush=True)
        save_model(checkpoint,dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},identity=identity))
        atomic_frame(path/'history.tsv',pd.DataFrame(history))
        write_json(report,dict(status='validated',identity=identity,checkpoint_sha256=sha(checkpoint),history_sha256=sha(path/'history.tsv')))
        del model

def ensemble(m,r):
    result=[];hashes={}
    for seed in SEEDS:
        p=deploy_dir(m,seed);report=read_json(p/'DEPLOY_REPORT.json')
        ident=report['identity']
        assert report['status']=='validated' and ident['run_fingerprint']==r['run_fingerprint'] and ident['model']==m and ident['seed']==seed
        assert sha(p/'deploy.pt')==report['checkpoint_sha256']
        blob=torch.load(str(p/'deploy.pt'),map_location='cpu');assert blob['identity']==ident
        model=training.make_model(m).cuda().eval();model.load_state_dict(blob['state_dict'])
        result.append(model);hashes[str(seed)]=report['checkpoint_sha256']
    return result,hashes

def input_arrays(scope,shard,frame):
    if scope=='reference':
        f=frame.copy();f['ligand_embedding_path']=f.ligand_embedding_path.map(lambda p:str(local(p)))
        arrays=external.load_reference_arrays(f)
        return arrays,f.ligand_embedding_path.astype(str).to_numpy()
    paths=external.cache_paths(ROOT,shard,4)
    raw=pd.read_pickle(paths['frame'])
    raw['OOD_Row_ID']=raw.OOD_Row_ID.astype(str)
    if raw.OOD_Row_ID.duplicated().any():raise ValueError('Duplicate raw row IDs')
    raw=raw.set_index('OOD_Row_ID').loc[frame.OOD_Row_ID.astype(str)]
    assert raw.UniProt_ID.fillna('').astype(str).str.split('-').str[0].tolist()==frame.uniprot.astype(str).tolist()
    assert raw.InChIKey14.fillna('').astype(str).str.upper().tolist()==frame.connectivity_key.astype(str).tolist()
    mapping=read_json(paths['mapping'])['lig_path_to_idx']
    paths_by_row=raw.Ligand_Embedding_Path.astype(str).to_numpy()
    row_lig=np.array([mapping[p] for p in paths_by_row],dtype=int)
    return (external.load_mmap(paths['ligand']),external.load_mmap(paths['ligand_mask']),
            torch.zeros(1,1,1536),torch.ones(1,dtype=torch.long),row_lig,np.zeros(len(frame),dtype=int)),paths_by_row

def infer_one(scope,shard,c,r):
    frame=prior_frame(scope,shard)
    if frame[row_id(scope)].duplicated().any():raise ValueError('Duplicate source IDs')
    arrays,input_keys=input_arrays(scope,shard,frame)
    context=external.load_pocket_context(ROOT,OLD,frame.uniprot)
    uid=frame.uniprot.astype(str).str.split('-').str[0]
    for field,cache in [('selected_chain_available','protein_cache'),('pocket_available','pocket_cache')]:
        assert np.array_equal(frame[field].to_numpy(),uid.isin(context[cache]).astype(int).to_numpy())
    lengths=np.array([len(context['protein_cache'][u]) if u in context['protein_cache'] else 1 for u in uid])
    for m in names(scope):
        models_,hashes=ensemble(m,r)
        identity=dict(run_fingerprint=r['run_fingerprint'],scope=scope,shard=shard,model=m,checkpoint_hashes=hashes,
                      source_sha256=c['payload']['files'][relative(prior(scope,shard))])
        part=PACKAGE/('gpu_output/parts/%s_%02d_%s.tsv.gz'%(scope,shard,m));report=part.with_suffix('.json')
        if report.exists():
            meta=read_json(report)
            if meta.get('identity')!=identity:raise ValueError('Incompatible inference resume')
            if part.exists() and meta.get('status')=='validated' and sha(part)==meta['sha256']:
                x=pd.read_csv(part,sep='\t',dtype={row_id(scope):str})
                assert x[row_id(scope)].tolist()==frame[row_id(scope)].tolist()
                for key in x.columns[1:]:frame[key]=x[key].to_numpy()
                print('SKIP %s %d %s'%(scope,shard,m),flush=True);del models_;continue
        available=np.ones(len(frame),bool) if m=='ligand' else frame['pocket_available' if m.startswith('d') else 'selected_chain_available'].eq(1).to_numpy()
        keys=uid.to_numpy() if m=='protein' else input_keys if m=='ligand' else np.arange(len(frame))
        selected=np.flatnonzero(available)
        single=m in ('ligand','protein')
        if single:selected=selected[~pd.Series(keys[selected]).duplicated().to_numpy()]
        selected=selected[np.argsort(lengths[selected],kind='stable')]
        bs=1 if single else 2 if scope=='biochemical' else 16
        result=np.full((len(frame),len(SEEDS)),np.nan,dtype=np.float32)
        with torch.no_grad():
            for start in range(0,len(selected),bs):
                idx=selected[start:start+bs]
                batch=external.make_batch(*arrays[:4],arrays[4][idx],arrays[5][idx])
                if m!='ligand':
                    batch,dest=external.attach_selected_chains(batch,uid.iloc[idx].tolist(),context,require_pocket=m.startswith('d'))
                    assert len(dest)==len(idx)
                # Trim padding for a deterministic actual-input single-row representation.
                if single:
                    for field in ('ligand','protein','pocket'):
                        n=int(batch[field+'_mask'].sum().item())
                        batch[field]=batch[field][:,:n];batch[field+'_mask']=batch[field+'_mask'][:,:n]
                gpu={k:v.cuda(non_blocking=True) for k,v in batch.items()}
                for j,model in enumerate(models_):
                    with torch.cuda.amp.autocast():logits=model(gpu)
                    result[idx,j]=torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
                if start%(bs*200)==0:print('%s shard=%d %s %d/%d'%(scope,shard,m,start+len(idx),len(selected)),flush=True)
        if single:
            lookup={keys[i]:result[i].copy() for i in selected}
            for i in np.flatnonzero(available):result[i]=lookup[keys[i]]
        if not np.isfinite(result[available]).all():raise ValueError('Nonfinite inference')
        columns=[]
        for j,s in enumerate(SEEDS):
            key='p_ligand_anchored_%s_seed_%d'%(m,s);frame[key]=result[:,j];columns.append(key)
        for suffix,values in [('mean',result.mean(axis=1)),('sd',result.std(axis=1))]:
            key='p_ligand_anchored_%s_%s'%(m,suffix);frame[key]=values;columns.append(key)
        atomic_frame(part,frame[[row_id(scope)]+columns])
        write_json(report,dict(status='validated',identity=identity,rows=len(frame),sha256=sha(part)))
        del models_;torch.cuda.empty_cache()
    cohort=bench.read_data();pfam=read_json(ROOT/read_json(OLD/'validation/CPU_CONTRACT.json')['external_resources']['pfam_cache']['path'])
    frame=novelty(frame,cohort,pfam);validate_scores(frame,scope)
    out=output(scope,shard);atomic_frame(out,frame)
    write_json(out.with_suffix('.json'),dict(status='validated',scope=scope,shard=shard,rows=len(frame),
        run_fingerprint=r['run_fingerprint'],checkpoint_hashes=deployment_hashes(scope),
        source_sha256=c['payload']['files'][relative(prior(scope,shard))],sha256=sha(out)))

def infer(rank,world):
    c,r=run_contract()
    for index,(scope,shard) in enumerate(tasks()):
        if index%world!=rank:continue
        out=output(scope,shard);rep=out.with_suffix('.json')
        if rep.exists():
            report=read_json(rep)
            if report.get('run_fingerprint')!=r['run_fingerprint']:raise ValueError('Changed inference run')
            if out.exists() and report['status']=='validated' and sha(out)==report['sha256'] and report.get('checkpoint_hashes')==deployment_hashes(scope):
                print('SKIP completed %s %d'%(scope,shard),flush=True);continue
        infer_one(scope,shard,c,r)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','deploy','infer']);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    args=p.parse_args();assert 0<=args.rank<args.world
    torch.set_num_threads(int(os.environ.get('CPU_THREADS','6')))
    if args.stage=='preflight':preflight()
    elif args.stage=='deploy':deploy(args.rank,args.world)
    else:infer(args.rank,args.world)

if __name__=='__main__':main()
