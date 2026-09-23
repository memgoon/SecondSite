import json,os,subprocess,sys
from pathlib import Path
from common import PKG,OLD,read,digest,save,atomic,versions

def main():
    lockfile=PKG/'INPUT_LOCK.json'
    if lockfile.exists():raise RuntimeError('already frozen; do not overwrite')
    needed={
        'reference':OLD/'data/CHECKPOINT7_REFERENCE_FEATURES.tsv.gz',
        'folds':OLD/'data/CHECKPOINT8_OBSERVATION_FOLDS.tsv.gz',
        'candidates':OLD/'data/CHECKPOINT9_CANDIDATE_FEATURES.tsv.gz',
        'audit':OLD/'data/CHECKPOINT10_SITE_CANDIDATE_AUDIT.tsv.gz',
        'old_pairs':OLD/'data/CHECKPOINT9_PAIR_RANKINGS_LONG.tsv.gz',
        'coordinates':OLD/'data/CHECKPOINT9_COORDINATE_MANIFEST.tsv.gz',
        'sequences':OLD/'data/CHECKPOINT9_UNIPROT_SEQUENCES.tsv.gz'}
    originals={}
    for n in ['CHECKPOINT8_EVALUATION_INPUT_HASHES.tsv','CHECKPOINT9_GEOMETRY_INPUT_HASHES.tsv']:
        p=OLD/'manifests'/n
        for r in read(p):originals[r['path']]=r['sha256']
        needed[n]=p
    for name in ['CHECKPOINT9_SCORING_BUILD.json','CHECKPOINT10_BUILD.json']:
        p=OLD/'validation'/name;j=json.loads(p.read_text());needed[name]=p
        for n,h in j['output_hashes'].items():originals[str(OLD/'data'/n)]=h
    for key in ['reference','folds','candidates','audit','old_pairs','coordinates','sequences']:
        p=needed[key]
        if digest(p)!=originals[str(p)]:raise ValueError('source chain mismatch '+key)
    for name,expected in [('CHECKPOINT8_SPLIT_VALIDATION.json','validated_heldout_splits'),('CHECKPOINT9_VALIDATION.json','validated_unlabelled_biolip_rankings'),('CHECKPOINT10_VALIDATION.json','validated_filtered_candidate_shortlist')]:
        p=OLD/'validation'/name;j=json.loads(p.read_text());needed[name]=p
        if j['status']!=expected or j.get('errors'):raise ValueError('source validation '+name)
    scripts=['common.py','analysis_steps.py','spatial.py','run_pipeline.py','validate_pipeline.py','make_report.py','test_pipeline.py','freeze_package.py','run_local.sh']
    paths=[PKG/'scripts'/n for n in scripts]+[PKG/'README.md',PKG/'WORK_LOG.md']
    deps=[OLD/'scripts/08_evaluate_heldout_rankings.py',OLD/'scripts/09c_compute_candidate_geometry.py']
    runtime={str(p):digest(p) for p in paths+deps}
    initial={str(p):digest(p) for p in needed.values()}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    test=subprocess.run([sys.executable,str(PKG/'scripts/test_pipeline.py'),'--results',str(PKG/'validation/TEST_RESULTS.tsv')],capture_output=True,text=True,env=env)
    atomic(PKG/'validation/TEST_LOG.txt',(test.stdout+test.stderr).encode())
    if test.returncode:raise ValueError('synthetic tests failed; inspect TEST_LOG.txt')
    result=json.loads(test.stdout)
    lock=dict(inputs={k:dict(path=str(p),sha256=digest(p)) for k,p in needed.items()},runtime_hashes=runtime,library_versions=versions(),
        expected_rows=[len(read(needed['reference'])),len(read(needed['candidates']))],synthetic=False,threshold='3/5',bootstrap_replicates=10000)
    save(lockfile,lock)
    pre=subprocess.run([sys.executable,str(PKG/'scripts/run_pipeline.py'),'--preflight-only'],capture_output=True,text=True,env=env)
    atomic(PKG/'validation/PREFLIGHT_LOG.txt',(pre.stdout+pre.stderr).encode())
    if pre.returncode:raise ValueError('input preflight failed; preserve lock and logs')
    if initial!={str(p):digest(p) for p in needed.values()}:raise ValueError('source preservation failure')
    save(PKG/'work/status/CODE_READINESS.json',dict(status='READY_FOR_LOCAL_EXECUTION',tests=result['tests'],tests_pass=True,
        preflight=json.loads(pre.stdout),production_executed=False,network_requests_performed=0,upstream_files_preserved=True,
        manual_curation_complete=False,web_deployed=False,resume_command='bash '+str(PKG/'scripts/run_local.sh')+' --workers 16 --io-workers 2'))
    selected=paths+[lockfile,PKG/'validation/TEST_RESULTS.tsv',PKG/'validation/TEST_LOG.txt',PKG/'validation/PREFLIGHT_LOG.txt',PKG/'work/status/CODE_READINESS.json']
    atomic(PKG/'PACKAGE_CHECKSUMS.sha256',''.join('%s  %s\n'%(digest(p),p.relative_to(PKG)) for p in sorted(selected)).encode())
    subprocess.run(['sha256sum','--check','--quiet','PACKAGE_CHECKSUMS.sha256'],cwd=str(PKG),check=True)
    print((PKG/'work/status/CODE_READINESS.json').read_text())

if __name__=='__main__':main()
