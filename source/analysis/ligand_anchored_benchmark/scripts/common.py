"""CPU-safe contract for the ligand-anchored fourth cohort."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import stats

PACKAGE=Path(__file__).resolve().parents[1]
VERSION='ligand_anchored_matrix_v1'
MODELS=('ligand','protein','c1','c2','c3','d1','d2','d3')
REGIMES=('row_random','unseen_family','unseen_ligand','double_unseen')
SEEDS=(20260817,20260818,20260819)
TRAINING=dict(epochs=25,patience=5,min_epochs=1,min_checkpoint_groups=8,
    hidden_dim=256,heads=4,dropout=.30,lr=1e-4,weight_decay=1e-4,batch_size=6,
    eval_batch_size=8,c3_batch_size=1,c3_eval_batch_size=2,max_atoms=120,
    max_protein_residues=4096,gradient_clip=5.0,
    loss='sum(weight*BCE)/sum(weight) per minibatch, matching original main benchmark',
    checkpoint_selection='legacy model/regime-aware validation AUROC; support<8 fallback pooled symmetric AP; improvement >1e-6',
    training_weight='train-only protein-label inverse counts, equal class totals; no ligand reweighting',
    inference='FP32 sigmoid; single-input deduplicated by actual input identity, batch 1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    os.replace(str(temp),str(path))


def read_data():return pd.read_csv(PACKAGE/'data/COHORT.tsv.gz',sep='\t',dtype={'main_row_id':str})


def partition(frame,regime,fold):
    column={'row_random':'run_row_fold','unseen_family':'run_family_fold',
            'unseen_ligand':'matrix_ligand_fold','double_unseen':'run_family_fold'}[regime]
    vf=(int(fold)+1)%5
    test=frame[frame[column].eq(fold)].copy()
    val=frame[frame[column].eq(vf)].copy()
    train=frame[~frame[column].isin([fold,vf])].copy()
    if regime=='double_unseen':
        val=val[~val.connectivity_key.isin(test.connectivity_key)].copy()
        blocked=set(test.connectivity_key)|set(val.connectivity_key)
        train=train[~train.connectivity_key.isin(blocked)].copy()
    fields=['main_row_id']
    if regime in ('unseen_family','double_unseen'):fields+=['uniprot','family_component_id']
    if regime in ('unseen_ligand','double_unseen'):fields+=['connectivity_key']
    for field in fields:
        a,b,c=[set(x[field]) for x in (train,val,test)]
        if a&b or a&c or b&c:raise ValueError('Overlap in '+field)
    for part in (train,val,test):
        if set(part.binary_label)!={0,1}:raise ValueError('Empty/single-class partition')
    dev=set(train.connectivity_key)|set(val.connectivity_key)
    test['unseen_compound']=(~test.connectivity_key.isin(dev)).astype(int)
    val['unseen_compound']=(~val.connectivity_key.isin(train.connectivity_key)).astype(int)
    train['unseen_compound']=0
    used=set(train.main_row_id)|set(val.main_row_id)|set(test.main_row_id)
    excluded=frame[~frame.main_row_id.isin(used)].copy()
    assert len(train)+len(val)+len(test)+len(excluded)==len(frame)
    return train,val,test,excluded


def selection_fields(regime,model):
    if model=='ligand':return ['uniprot']
    if model=='protein':return ['full_inchikey']
    if model not in MODELS:raise ValueError(model)
    if regime in ('unseen_family','double_unseen'):return ['full_inchikey']
    if regime=='unseen_ligand':return ['uniprot']
    if regime=='row_random':return ['full_inchikey','uniprot']
    raise ValueError(regime)


def checkpoint_selection(prediction,regime,model):
    # Validation is from a single fold, so conditional grouping has no cross-model scales.
    p=prediction.assign(outer_fold=0)
    fields=selection_fields(regime,model)
    v={k:stats(p,k) for k in fields}
    support={k:x['n_groups_used'] for k,x in v.items()}
    fallback=any(x['n_groups_used']<8 or not np.isfinite(x['auroc']) for x in v.values())
    score=stats(p)['symmetric_ap'] if fallback else float(np.mean([x['auroc'] for x in v.values()]))
    if not np.isfinite(score):raise ValueError('No valid validation selection statistic')
    return dict(selection_score=float(score),selection_fallback=bool(fallback),
        selection_metric='pooled.symmetric_ap' if fallback else 'mean('+','.join(fields)+').auroc',
        selection_support=json.dumps(support,sort_keys=True))


def replay_history(history):
    best=-np.inf;epoch=-1;stale=0
    for row in history.itertuples():
        if stale>=TRAINING['patience']:raise ValueError('Epochs continued after stopping rule')
        if row.selection_score>best+1e-6:best=float(row.selection_score);epoch=int(row.epoch);stale=0
        else:stale+=1
    if len(history)!=25 and stale<5:raise ValueError('Training stopped too early')
    return epoch,best


def jobs():return [(r,m,s,f) for r in REGIMES for m in MODELS for s in SEEDS for f in range(5)]


def fit_dir(job):
    r,m,s,f=job
    return PACKAGE/'gpu_output/fits'/r/m/('seed_%d'%s)/('fold_%02d'%f)


def identity(job):
    r,m,s,f=job
    return dict(cohort_arm='ligand_anchored',regime=r,model=m,seed=int(s),outer_fold=int(f))


def verify_contract():
    c=json.loads((PACKAGE/'validation/CPU_CONTRACT.json').read_text())
    if c['status']!='validated' or c['version']!=VERSION or c['training']!=TRAINING:raise ValueError('Contract mismatch')
    for name,h in c['files'].items():
        if sha(PACKAGE/name)!=h:raise ValueError('Contract SHA256 mismatch: '+name)
    if digest(c['files'])!=c['manifest_digest']:raise ValueError('Manifest digest mismatch')
    return c


def compatible_report(job,run,hashes=True):
    try:
        p=fit_dir(job);r=json.loads((p/'FIT_REPORT.json').read_text())
        if (r['status']!='validated' or r['run_fingerprint']!=run['run_fingerprint'] or r['training']!=TRAINING
            or r['model_version']!='role_complete_matrix_v2' or not 1<=r['best_epoch']<=r['epochs_run']<=25
            or r['validation_rows']<=0 or any(r[k]!=v for k,v in identity(job).items())):return None
        for name in ('best.pt','predictions.tsv.gz','validation_predictions.tsv.gz','history.tsv'):
            if not (p/name).is_file() or (hashes and sha(p/name)!=r['sha256'][name]):return None
        return r
    except (OSError,ValueError,KeyError,TypeError):return None


def local_path(path,root):
    for prefix in ('/disk1/11.HS_allostery/','/disk9/13.Heesu_Allostery/','/shared_data/11.HS_allostery/'):
        if str(path).startswith(prefix):return str(root/str(path)[len(prefix):])
    return str(path)


def input_identity(model,frame):
    return frame[{'protein':'uniprot','ligand':'ligand_embedding_path'}.get(model,'main_row_id')].astype(str)
