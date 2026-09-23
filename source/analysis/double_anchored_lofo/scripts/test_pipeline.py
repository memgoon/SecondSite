"""Synthetic end-to-end CPU smoke test, confined to an auto-cleaned /tmp directory.

Creates fake checkpoint bytes and constant predictions, NOT experiment results.
No CUDA test is implied. Exercises all 2,256 paths, metadata checks and aggregation.
"""
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import common


def main():
    source=common.PACKAGE
    contract=common.verify_contract()
    with tempfile.TemporaryDirectory(prefix='double_anchored_lofo_test_') as temporary:
        root=Path(temporary)/'package'
        for name in list(contract['files'])+['validation/CPU_CONTRACT.json']:
            target=root/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(str(source/name),str(target))
        common.PACKAGE=root
        # Imports after redirecting common.PACKAGE keep all writes in the test directory.
        import validate_fits
        import aggregate
        d=common.read_data()
        paths=set(d.ligand_embedding_path)|set(d.protein_embedding_path)
        core=dict(cpu_manifest=contract['manifest_digest'],input_hashes={p:'0'*64 for p in paths},
                  training=common.TRAINING,torch_version='SYNTHETIC_NO_GPU',numpy_version='test',
                  pandas_version='test',version=common.VERSION)
        run=dict(core,run_fingerprint=common.digest(core),status='validated',synthetic_test_only=True)
        common.write_json(root/'gpu_output/RUN_CONTRACT.json',run)
        import queue_jobs
        queue_jobs.initialize(run)
        with ThreadPoolExecutor(max_workers=8) as pool:
            claimed=list(pool.map(queue_jobs.claim,[str(i) for i in range(8)]))
        assert len(set(claimed))==8
        queue_jobs.initialize(run)
        with queue_jobs.connect() as connection:
            assert connection.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]==2256
        partitions={(r,f):common.partition(d,r,f) for r in common.REGIMES for f in range(47)}
        history=pd.DataFrame(dict(epoch=list(range(1,26)),training_loss=[.5]*25,elapsed_seconds=list(range(1,26))))
        for job in common.jobs():
            train,test,excluded=partitions[(job[0],job[3])]
            p=test.copy()
            p['p_allosteric']=.5
            for k,v in common.identity(job).items():
                p[k]=v
            directory=common.fit_dir(job)
            directory.mkdir(parents=True)
            (directory/'final.pt').write_bytes(b'SYNTHETIC CHECKPOINT - CPU PIPELINE TEST ONLY')
            p.to_csv(directory/'predictions.tsv.gz',sep='\t',index=False)
            history.to_csv(directory/'history.tsv',sep='\t',index=False)
            report=dict(common.identity(job),status='validated',training=common.TRAINING,
                final_epoch=25,validation_rows=0,train_rows=len(train),test_rows=len(test),excluded_rows=len(excluded),
                held_out_family=common.families(d)[job[3]],model_version='role_complete_matrix_v2',
                run_fingerprint=run['run_fingerprint'],sha256={n:common.sha(directory/n) for n in ('final.pt','predictions.tsv.gz','history.tsv')})
            common.write_json(directory/'FIT_REPORT.json',report)
        sys.argv=['aggregate.py','--workers','2']
        aggregate.main()
        result=json.loads((root/'local_analysis/VALIDATION.json').read_text())
        assert result['status']=='validated' and result['bootstrap_comparisons']==36
        # Corrupt metadata but refresh its checksum: semantic validator must still reject it.
        directory=common.fit_dir(common.jobs()[0])
        p=pd.read_csv(directory/'predictions.tsv.gz',sep='\t',dtype={'main_row_id':str})
        p.loc[0,'binary_label']=1-int(p.loc[0,'binary_label'])
        p.to_csv(directory/'predictions.tsv.gz',sep='\t',index=False)
        report=json.loads((directory/'FIT_REPORT.json').read_text())
        report['sha256']['predictions.tsv.gz']=common.sha(directory/'predictions.tsv.gz')
        common.write_json(directory/'FIT_REPORT.json',report)
        try:
            validate_fits.validate()
        except ValueError as error:
            assert 'metadata mismatch' in str(error),str(error)
        else:
            raise AssertionError('Corrupt labels accepted')
        print('SYNTHETIC_PIPELINE_PASS: 2256 fits, 18960 rows, 36 bootstrap comparisons; corrupt labels rejected.')
    common.PACKAGE=source
    common.write_json(source/'validation/SYNTHETIC_PIPELINE_TEST.json',dict(status='validated',
        synthetic_only=True,cuda_execution_tested=False,fits=2256,prediction_rows=18960,
        bootstrap_comparisons=36,replicates_per_comparison=10000,
        corrupt_metadata_rejected=True,concurrent_queue_claims_distinct=True,
        manifest_digest=contract['manifest_digest']))


if __name__=='__main__':
    main()
