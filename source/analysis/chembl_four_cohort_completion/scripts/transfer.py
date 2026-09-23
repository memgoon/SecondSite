"""Explicit named archives; safe extraction; never search input cache trees."""
import argparse
import tarfile
from pathlib import PurePosixPath
from common import *

def return_names():
    names=['gpu_output/RUN_CONTRACT.json','gpu_output/VALIDATION.json']
    for t in tasks():
        names.extend(str(p.relative_to(PACKAGE)) for p in task_paths(t)[:2])
    return names

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['input','return','extract']);p.add_argument('--archive');args=p.parse_args()
    c=verify_contract()
    if args.mode=='extract':
        allowed=set(return_names())
        with tarfile.open(args.archive,'r:gz') as tar:
            members=tar.getmembers();names=[m.name for m in members]
            if set(names)!=allowed or len(names)!=len(allowed):raise ValueError('Missing, duplicated, or unexpected archive members')
            for m in members:
                q=PurePosixPath(m.name)
                if not m.isfile() or q.is_absolute() or '..' in q.parts:raise ValueError('Unsafe archive entry')
                for parent in [PACKAGE/m.name]+list((PACKAGE/m.name).parents):
                    if parent==PACKAGE.parent:break
                    if parent.is_symlink():raise ValueError('Symlink extraction target')
            tar.extractall(str(PACKAGE),members=members)
        print('Extracted completion outputs only');return
    if args.mode=='input':
        prefix=relative(PACKAGE)+'/'
        names=[x[len(prefix):] for x in c['payload']['files'] if x.startswith(prefix)]+['validation/CPU_CONTRACT.json']
    else:
        from assemble import validate_parts
        validate_parts();names=return_names()
    path=PACKAGE/('chembl_four_cohort_completion_gpu_%s_v1.tar.gz'%args.mode)
    tmp=path.with_suffix('.tmp')
    with tarfile.open(str(tmp),'w:gz') as tar:
        for name in names:tar.add(str(PACKAGE/name),arcname=name,recursive=False)
    os.replace(str(tmp),str(path));path.with_suffix(path.suffix+'.sha256').write_text(sha(path)+'  '+path.name+'\n')
    print(str(path),flush=True)

if __name__=='__main__':main()
