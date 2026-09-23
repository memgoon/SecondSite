"""Synthetic full CPU pipeline in /tmp; never creates real GPU results."""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import common


def main():
    source=common.PACKAGE;c=common.verify_contract()
    with tempfile.TemporaryDirectory(prefix='ligand_anchored_test_') as temporary:
        root=Path(temporary)/'package'
        for name in list(c['files'])+['validation/CPU_CONTRACT.json']:
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(str(source/name),str(target))
        common.PACKAGE=root
        import validate_fits,aggregate,queue_jobs
        d=common.read_data();paths=set(d.ligand_embedding_path)|set(d.protein_embedding_path)
        core=dict(cpu_manifest=c['manifest_digest'],input_hashes={p:'0'*64 for p in paths},training=common.TRAINING,
            torch_version='SYNTHETIC_NO_GPU',numpy_version='test',pandas_version='test',version=common.VERSION)
        run=dict(core,run_fingerprint=common.digest(core),status='validated',synthetic_test_only=True)
        common.write_json(root/'gpu_output/RUN_CONTRACT.json',run)
        queue_jobs.initialize(run)
        with ThreadPoolExecutor(max_workers=8) as pool:claimed=list(pool.map(queue_jobs.claim,[str(i) for i in range(8)]))
        assert len(set(claimed))==8
        queue_jobs.initialize(run)
        with queue_jobs.connect() as con:assert con.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]==480
        parts={(r,f):common.partition(d,r,f) for r in common.REGIMES for f in range(5)}
        for job in common.jobs():
            tr,va,te,ex=parts[(job[0],job[3])];directory=common.fit_dir(job);directory.mkdir(parents=True)
            tables=[]
            for raw in [te,va]:
                p=raw.copy();p['p_allosteric']=.5;p['validation_fold']=(job[3]+1)%5
                for k,v in common.identity(job).items():p[k]=v
                tables.append(p)
            tables[0].to_csv(directory/'predictions.tsv.gz',sep='\t',index=False)
            tables[1].to_csv(directory/'validation_predictions.tsv.gz',sep='\t',index=False)
            selected=common.checkpoint_selection(tables[1],job[0],job[1])
            pd.DataFrame([dict(epoch=e,training_loss=.5,elapsed_seconds=e,**selected) for e in range(1,7)]).to_csv(directory/'history.tsv',sep='\t',index=False)
            (directory/'best.pt').write_bytes(b'SYNTHETIC CHECKPOINT - NOT FOR INFERENCE')
            report=dict(common.identity(job),status='validated',training=common.TRAINING,best_epoch=1,epochs_run=6,
                best_validation_score=.5,validation_rows=len(va),validation_fold=(job[3]+1)%5,
                train_rows=len(tr),test_rows=len(te),excluded_rows=len(ex),model_version='role_complete_matrix_v2',
                selection_fallback=selected['selection_fallback'],selection_metric=selected['selection_metric'],selection_support=selected['selection_support'],
                run_fingerprint=run['run_fingerprint'],sha256={n:common.sha(directory/n) for n in ('best.pt','predictions.tsv.gz','validation_predictions.tsv.gz','history.tsv')})
            common.write_json(directory/'FIT_REPORT.json',report)
        sys.argv=['aggregate.py','--workers','2'];aggregate.main()
        v=json.loads((root/'local_analysis/VALIDATION.json').read_text());assert v['status']=='validated' and v['bootstrap_comparisons']==56
        matrix=pd.read_csv(root/'local_analysis/MATRIX_4X4_ALL_MODELS.tsv',sep='\t');assert len(matrix)==256
        # Exercise explicit artifact packaging and safe extraction without real weights.
        import package_return,transfer
        package_return.main()
        sys.argv=['transfer.py','extract','--archive',str(root/'ligand_anchored_benchmark_gpu_return_v1.tar.gz')]
        transfer.main()
        directory=common.fit_dir(common.jobs()[0])
        table=pd.read_csv(directory/'predictions.tsv.gz',sep='\t',dtype={'main_row_id':str});table.loc[0,'binary_label']=1-int(table.loc[0,'binary_label'])
        table.to_csv(directory/'predictions.tsv.gz',sep='\t',index=False)
        report=json.loads((directory/'FIT_REPORT.json').read_text());report['sha256']['predictions.tsv.gz']=common.sha(directory/'predictions.tsv.gz')
        common.write_json(directory/'FIT_REPORT.json',report)
        try:validate_fits.validate()
        except ValueError as error:assert 'metadata mismatch' in str(error)
        else:raise AssertionError('Corrupt labels accepted')
    common.PACKAGE=source
    common.write_json(source/'validation/SYNTHETIC_PIPELINE_TEST.json',dict(status='validated',synthetic_only=True,
        cuda_execution_tested=False,fits=480,prediction_rows=131040,bootstrap_comparisons=56,
        matrix_rows=256,metadata_corruption_rejected=True,archive_roundtrip_tested=True,
        concurrent_queue_claims_distinct=True,manifest_digest=c['manifest_digest']))
    print('SYNTHETIC_PIPELINE_PASS: 480 fits, 131040 predictions, 56 bootstrap comparisons, 4x4 matrix.',flush=True)


if __name__=='__main__':main()
