"""CPU-only contracts for inference completion. No training entrypoint."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
OLD = ROOT / 'analysis/role_complete_pair_matrix'
LA = ROOT / 'analysis/ligand_anchored_chembl'
VERSION = 'chembl_four_cohort_completion_v1'
ARMS = ('every_pair', 'protein_anchored', 'ligand_anchored', 'double_anchored')
MODELS = ('ligand', 'protein', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3')
SEEDS = (20260817, 20260818, 20260819)
PREFIX = dict(every_pair='every_pair', protein_anchored='general', ligand_anchored='ligand_anchored', double_anchored='role_complete')
NEW = tuple((a,m) for a in ARMS for m in MODELS if a=='double_anchored' or m=='c3')
CHUNK_ROWS = 4096
COHORT = dict(every_pair=OLD/'data/EVERY_PAIR.tsv.gz', protein_anchored=OLD/'data/PROTEIN_ANCHORED.tsv.gz',
    double_anchored=OLD/'data/PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz',
    ligand_anchored=ROOT/'analysis/ligand_anchored_benchmark/data/COHORT.tsv.gz')

def read_json(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def relative(p):return str(Path(p).relative_to(ROOT))
def local(p):
    p=str(p)
    for prefix in ('/disk1/11.HS_allostery/','/disk9/13.Heesu_Allostery/','/shared_data/11.HS_allostery/'):
        if p.startswith(prefix):return ROOT/p[len(prefix):]
    return Path(p)
def write_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');os.replace(str(tmp),str(p))
def write_frame(p,f):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    f.to_csv(tmp,sep='\t',index=False,compression='gzip' if p.suffix=='.gz' else None);os.replace(str(tmp),str(p))
def source(scope,shard):return LA/('gpu_output/predictions/%s_%02d.tsv.gz'%(scope,shard))
def row_id(scope):return 'reference_row_id' if scope=='reference' else 'OOD_Row_ID'
def scopes():return [('reference',0)]+[('full',i) for i in range(4)]
def score(a,m,suffix='mean'):return 'p_%s_%s_%s'%(PREFIX[a],m,suffix)
def new_columns(a,m):return [score(a,m,'seed_%d'%s) for s in SEEDS]+[score(a,m),score(a,m,'sd')]
def available(f,m):
    return np.ones(len(f),bool) if m=='ligand' else f['pocket_available' if m.startswith('d') else 'selected_chain_available'].eq(1).to_numpy()
def deploy(a,m,s):
    if a=='ligand_anchored':return LA/('gpu_output/deploy/%s/seed_%d'%(m,s))
    arm=dict(every_pair='general_every_pair',protein_anchored='general_protein_anchored',double_anchored='biochemical_role_complete')[a]
    return OLD/('gpu_output/deploy/%s/%s/seed_%d'%(arm,m,s))
def task_paths(t):
    p=PACKAGE/'gpu_output/parts'/t['task_id']
    return p.with_suffix('.tsv.gz'),p.with_suffix('.json'),p.with_suffix('.lock')
def tasks():return read_json(PACKAGE/'data/TASKS.json')
def verify_contract(external=False):
    c=read_json(PACKAGE/'validation/CPU_CONTRACT.json')
    if c['status']!='validated' or c['version']!=VERSION or digest(c['payload'])!=c['fingerprint']:raise ValueError('Contract invalid')
    selected={k:h for k,h in c['payload']['files'].items()
              if external or k.startswith(relative(PACKAGE)+'/')}
    missing=[k for k in selected if not (ROOT/k).is_file()]
    if missing:
        raise FileNotFoundError('Missing %d contract files (no inference started):\n%s' %
                                (len(missing),'\n'.join(str(ROOT/k) for k in missing)))
    for k,h in selected.items():
        if sha(ROOT/k)!=h:raise ValueError('Changed file: '+k)
    return c
def complete(t,run,check_hash=True):
    p,r,_=task_paths(t)
    if not p.is_file() or not r.is_file():return False
    meta=read_json(r)
    if meta.get('run_fingerprint')!=run['run_fingerprint'] or meta.get('task')!=t:raise ValueError('Incompatible task '+t['task_id'])
    return meta.get('status')=='validated' and (not check_hash or sha(p)==meta['sha256'])
def auc(y,s):
    y=np.asarray(y,int);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return np.nan
    r=pd.Series(np.asarray(s)).rank(method='average').to_numpy()
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def ap(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float)
    if not y.sum() or y.sum()==len(y):return np.nan
    order=np.argsort(-s,kind='mergesort');y=y[order];s=s[order]
    end=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.cumsum(y)[end]
    return float(np.sum(np.diff(np.r_[0.,tp/y.sum()])*tp/(end+1)))
