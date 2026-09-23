"""Create explicit input archive, or safely unpack an already checksum-verified return."""
import argparse
import tarfile
from pathlib import PurePosixPath
from common import PACKAGE,verify_contract,jobs,fit_dir


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['input','extract'])
    p.add_argument('--archive')
    args=p.parse_args()
    contract=verify_contract()
    if args.mode=='input':
        archive=PACKAGE/'double_anchored_lofo_gpu_input_v1.tar.gz'
        names=list(contract['files'])+['validation/CPU_CONTRACT.json']
        with tarfile.open(str(archive),'w:gz') as t:
            for name in names:
                t.add(str(PACKAGE/name),arcname=name,recursive=False)
        print(str(archive))
        return
    allowed={'gpu_output/'+p for p in ('RUN_CONTRACT.json','FITS_VALIDATION.json','FIT_SUMMARY.tsv','FIT_INVENTORY.tsv')}
    for job in jobs():
        allowed.update(str((fit_dir(job)/p).relative_to(PACKAGE)) for p in ('FIT_REPORT.json','final.pt','predictions.tsv.gz','history.tsv'))
    with tarfile.open(args.archive,'r:gz') as t:
        members=t.getmembers()
        for member in members:
            path=PurePosixPath(member.name)
            is_log=len(path.parts)==3 and path.parts[:2]==('gpu_output','logs') and path.suffix=='.log'
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or (str(path) not in allowed and not is_log)):
                raise ValueError('Unexpected archive member: '+member.name)
        names=[m.name for m in members]
        if len(names)!=len(set(names)) or not allowed.issubset(set(names)):
            raise ValueError('Duplicate or missing return files')
        for member in members:
            destination=PACKAGE/member.name
            # Reject existing symlink parents rather than follow them during extraction.
            for parent in [destination]+list(destination.parents):
                if parent==PACKAGE.parent:
                    break
                if parent.is_symlink():
                    raise ValueError('Refusing symlink destination '+str(parent))
        t.extractall(str(PACKAGE),members=members)
    print('Return archive extracted; CPU validation follows.')


if __name__=='__main__':
    main()
