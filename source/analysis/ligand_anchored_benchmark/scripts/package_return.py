"""Explicit artifact list; no recursive archive walk."""
import tarfile
from common import PACKAGE,jobs,fit_dir,sha


def main():
    paths=[PACKAGE/'gpu_output'/p for p in ('RUN_CONTRACT.json','FITS_VALIDATION.json','FIT_SUMMARY.tsv','FIT_INVENTORY.tsv')]
    for job in jobs():
        paths.extend(fit_dir(job)/p for p in ('FIT_REPORT.json','best.pt','predictions.tsv.gz','validation_predictions.tsv.gz','history.tsv'))
    logdir=PACKAGE/'gpu_output/logs'
    if logdir.exists():
        paths.extend(p for p in logdir.iterdir() if p.is_file())
    archive=PACKAGE/'ligand_anchored_benchmark_gpu_return_v1.tar.gz'
    temp=archive.with_suffix('.tmp')
    with tarfile.open(str(temp),'w:gz',compresslevel=1) as tar:
        for path in paths:
            if not path.is_file():
                raise ValueError('Missing return artifact: '+str(path))
            tar.add(str(path),arcname=str(path.relative_to(PACKAGE)),recursive=False)
    temp.replace(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print('Complete. Return archive: '+str(archive),flush=True)
    print('size_bytes='+str(archive.stat().st_size),flush=True)


if __name__=='__main__':
    main()
