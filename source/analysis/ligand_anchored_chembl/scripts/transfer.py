"""Explicit-file archives only, with safe extraction into this new package."""
import argparse
import tarfile
from pathlib import PurePosixPath
from shared import *

def return_names():
    names=['gpu_output/RUN_CONTRACT.json']
    for m in MODELS:
        for s in SEEDS:
            names += [str((deploy_dir(m,s)/n).relative_to(PACKAGE)) for n in ('deploy.pt','DEPLOY_REPORT.json','history.tsv')]
    for scope,i in tasks():
        names += [str(p.relative_to(PACKAGE)) for p in (output(scope,i),output(scope,i).with_suffix('.json'))]
    return names

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['input','return','extract']);p.add_argument('--archive');a=p.parse_args()
    c=verify_contract()
    if a.mode=='extract':
        allowed=set(return_names())
        with tarfile.open(a.archive,'r:gz') as t:
            members=t.getmembers();names=[m.name for m in members]
            if set(names)!=allowed or len(names)!=len(allowed):raise ValueError('Unexpected/duplicate/missing return files')
            for m in members:
                p=PurePosixPath(m.name)
                if not m.isfile() or p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive')
                dest=PACKAGE/m.name
                for parent in [dest]+list(dest.parents):
                    if parent==PACKAGE.parent:break
                    if parent.is_symlink():raise ValueError('Symlink destination')
            t.extractall(str(PACKAGE),members=members)
        print('Extracted explicit return files. CPU independent validation follows.');return
    if a.mode=='input':
        prefix=relative(PACKAGE)+'/'
        names=[k[len(prefix):] for k in c['payload']['files'] if k.startswith(prefix)]+['validation/CPU_CONTRACT.json']
    else:
        names=return_names()
        run=read_json(PACKAGE/'gpu_output/RUN_CONTRACT.json')
        for scope,i in tasks():
            out=output(scope,i);r=read_json(out.with_suffix('.json'))
            assert r['status']=='validated' and r['run_fingerprint']==run['run_fingerprint'] and sha(out)==r['sha256']
            assert r['checkpoint_hashes']==deployment_hashes(scope)
        for m in MODELS:
            for s in SEEDS:
                d=deploy_dir(m,s);r=read_json(d/'DEPLOY_REPORT.json')
                assert r['status']=='validated' and r['identity']['run_fingerprint']==run['run_fingerprint'] and sha(d/'deploy.pt')==r['checkpoint_sha256']
    archive=PACKAGE/('ligand_anchored_chembl_gpu_%s_v1.tar.gz'%a.mode)
    temp=archive.with_suffix('.tmp')
    with tarfile.open(str(temp),'w:gz') as t:
        for name in names:t.add(str(PACKAGE/name),arcname=name,recursive=False)
    os.replace(str(temp),str(archive))
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(str(archive),flush=True)

if __name__=='__main__':main()
