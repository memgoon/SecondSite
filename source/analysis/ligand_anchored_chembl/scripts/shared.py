"""CPU-only definitions. All paths are explicit; no recursive discovery."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parents[1]
OLD=ROOT/'analysis/role_complete_pair_matrix'
BENCH=ROOT/'analysis/ligand_anchored_benchmark'
sys.path.append(str(BENCH/'scripts'))
import common as bench
from metrics import auc,ap
VERSION='ligand_anchored_chembl_v1'
MODELS=bench.MODELS
FULL_MODELS=tuple(m for m in MODELS if m!='c3')
SEEDS=bench.SEEDS
sha=bench.sha
digest=bench.digest
write_json=bench.write_json

def read_json(p):return json.loads(Path(p).read_text())
def local(p):return Path(bench.local_path(p,ROOT))
def relative(p):return str(Path(p).relative_to(ROOT))
def prior(scope,shard=0):
    if scope=='reference':return OLD/'gpu_output/chembl/reference_predictions.tsv.gz'
    return OLD/('gpu_output/chembl/%s/%s_predictions_shard%02dof04.tsv.gz'%(scope,scope,shard))
def tasks():return [('reference',0)]+[('full',i) for i in range(4)]+[('biochemical',i) for i in range(4)]
def output(scope,shard):return PACKAGE/('gpu_output/predictions/%s_%02d.tsv.gz'%(scope,shard))
def deploy_dir(model,seed):return PACKAGE/('gpu_output/deploy/%s/seed_%d'%(model,seed))
def deployment_hashes(scope):
    result={}
    for m in names(scope):
        for s in SEEDS:
            p=deploy_dir(m,s);r=read_json(p/'DEPLOY_REPORT.json')
            h=sha(p/'deploy.pt')
            if r['status']!='validated' or r['checkpoint_sha256']!=h:raise ValueError('Invalid deployment checkpoint')
            result['%s/%d'%(m,s)]=h
    return result
def atomic_frame(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(tmp,sep='\t',index=False,compression='gzip' if path.suffix=='.gz' else None)
    os.replace(str(tmp),str(path))
def names(scope):return MODELS if scope=='biochemical' else FULL_MODELS
def row_id(scope):return 'reference_row_id' if scope=='reference' else 'OOD_Row_ID'
def prior_frame(scope,shard=0):
    f=pd.read_csv(prior(scope,shard),sep='\t',dtype={row_id(scope):str},low_memory=False)
    # Preserve the legacy aggregator's eligibility rule, not merely raw shard membership.
    if scope!='reference':f=f[~f.legacy_ood_label.eq(0)].copy()
    return f.reset_index(drop=True)
def verify_contract():
    c=read_json(PACKAGE/'validation/CPU_CONTRACT.json')
    assert c['version']==VERSION and c['status']=='validated'
    assert digest(c['payload'])==c['fingerprint']
    for name,h in c['payload']['files'].items():
        if sha(ROOT/name)!=h:raise ValueError('Changed input/code: '+name)
    return c
def validate_scores(frame,scope):
    for m in names(scope):
        available=np.ones(len(frame),dtype=bool) if m=='ligand' else frame['pocket_available' if m.startswith('d') else 'selected_chain_available'].eq(1).to_numpy()
        for suffix in ('mean','sd'):
            x=frame['p_ligand_anchored_%s_%s'%(m,suffix)].to_numpy(float)
            if not np.isfinite(x[available]).all() or not np.isnan(x[~available]).all():raise ValueError('Availability/finite mismatch '+m)
            if ((x[available]<0)|(x[available]>(1 if suffix=='mean' else .5+1e-6))).any():raise ValueError('Score range '+m)
    return True
def novelty(frame,cohort,pfam):
    uid=frame.uniprot.astype(str).str.split('-').str[0]
    keys=frame.connectivity_key.astype(str).str.upper()
    train_uid=set(cohort.uniprot.astype(str).str.split('-').str[0])
    train_keys=set(cohort.connectivity_key.astype(str).str.upper())
    frame['protein_seen_ligand_anchored']=uid.isin(train_uid).astype(int)
    frame['ligand_seen_ligand_anchored']=keys.isin(train_keys).astype(int)
    records=pfam['records']
    complete=all(u in records and records[u].get('annotation_status')=='annotated' and records[u].get('pfam_ids') for u in train_uid)
    union={p for u in train_uid for p in records.get(u,{}).get('pfam_ids',[])}
    status=[]
    for u in uid:
        r=records.get(u,{});ids=set(r.get('pfam_ids',[]))
        status.append('annotation_unavailable' if r.get('annotation_status')!='annotated' or not ids else
                      'seen' if ids&union else 'unseen' if complete else 'training_reference_incomplete')
    frame['pfam_family_status_ligand_anchored']=status
    return frame
