"""Independent CPU audit of predictions, selected checkpoint and split metadata."""
import numpy as np
import pandas as pd
from common import *


def validate_table(p,expected,job):
    if p.main_row_id.duplicated().any() or set(p.main_row_id)!=set(expected.main_row_id):raise ValueError('Prediction coverage')
    a=p.set_index('main_row_id').sort_index();b=expected.set_index('main_row_id').sort_index()
    for field in ('uniprot','family_component_id','full_inchikey','connectivity_key','binary_label','unseen_compound','ligand_embedding_path','protein_embedding_path'):
        if not a[field].astype(str).equals(b[field].astype(str)):raise ValueError('Prediction metadata mismatch: '+field)
    for k,v in identity(job).items():
        if not p[k].astype(str).eq(str(v)).all():raise ValueError('Prediction identity '+k)
    if not p.validation_fold.eq((job[3]+1)%5).all():raise ValueError('Validation fold identity')
    if not np.isfinite(p.p_allosteric).all() or not p.p_allosteric.between(0,1).all():raise ValueError('Probability range/finite')
    key={'ligand':'ligand_embedding_path','protein':'uniprot'}.get(job[1])
    if key and not p.groupby(key).p_allosteric.nunique().eq(1).all():raise ValueError('Single-input identity varies')


def validate():
    c=verify_contract();d=read_data()
    run=json.loads((PACKAGE/'gpu_output/RUN_CONTRACT.json').read_text())
    core={k:run[k] for k in ('cpu_manifest','input_hashes','training','torch_version','numpy_version','pandas_version','version')}
    if (digest(core)!=run['run_fingerprint'] or run['cpu_manifest']!=c['manifest_digest'] or run['training']!=TRAINING
        or run['version']!=VERSION or run['status']!='validated'):raise ValueError('Run fingerprint mismatch')
    if set(run['input_hashes'])!=(set(d.ligand_embedding_path)|set(d.protein_embedding_path)):raise ValueError('Incomplete cache inventory')
    predictions=[];reports=[];inventory=[]
    for job in jobs():
        r=compatible_report(job,run)
        if r is None:raise ValueError('Missing/incompatible fit '+str(job))
        tr,va,te,ex=partition(d,job[0],job[3]);path=fit_dir(job)
        for k,v in dict(train_rows=len(tr),validation_rows=len(va),test_rows=len(te),excluded_rows=len(ex),validation_fold=(job[3]+1)%5).items():
            if r[k]!=v:raise ValueError('Split report mismatch '+k)
        p=pd.read_csv(path/'predictions.tsv.gz',sep='\t',dtype={'main_row_id':str})
        v=pd.read_csv(path/'validation_predictions.tsv.gz',sep='\t',dtype={'main_row_id':str})
        validate_table(p,te,job);validate_table(v,va,job)
        h=pd.read_csv(path/'history.tsv',sep='\t')
        if list(h.epoch)!=list(range(1,len(h)+1)) or len(h)!=r['epochs_run']:raise ValueError('Epoch history')
        if not np.isfinite(h[['training_loss','selection_score','elapsed_seconds']].to_numpy()).all():raise ValueError('Nonfinite history')
        if any('test' in k for k in h.columns):raise ValueError('Test metric in selection history')
        ep,sc=replay_history(h)
        if ep!=r['best_epoch'] or abs(sc-r['best_validation_score'])>1e-10:raise ValueError('Checkpoint not chosen by declared rule')
        selected=checkpoint_selection(v,job[0],job[1])
        if abs(selected['selection_score']-sc)>1e-6:raise ValueError('Best validation predictions disagree')
        for k in ('selection_fallback','selection_metric','selection_support'):
            if r[k]!=selected[k] or not h[k].eq(selected[k]).all():raise ValueError('Selection contract '+k)
        predictions.append(p);reports.append({k:x for k,x in r.items() if k not in ('training','sha256')})
        inventory.append(dict(identity(job),report_sha256=sha(path/'FIT_REPORT.json'),**r['sha256']))
    raw=pd.concat(predictions,ignore_index=True)
    if len(raw)!=131040:raise ValueError('Total OOF rows')
    for _,g in raw.groupby(['regime','model','seed']):
        if len(g)!=1365 or g.main_row_id.nunique()!=1365:raise ValueError('OOF coverage')
    summary=pd.DataFrame(reports)
    if int(summary.selection_fallback.sum())!=c['expected_fallback_fits']:raise ValueError('Unexpected fallback count')
    summary.to_csv(PACKAGE/'gpu_output/FIT_SUMMARY.tsv',sep='\t',index=False)
    pd.DataFrame(inventory).to_csv(PACKAGE/'gpu_output/FIT_INVENTORY.tsv',sep='\t',index=False)
    write_json(PACKAGE/'gpu_output/FITS_VALIDATION.json',dict(status='validated',fits=480,prediction_rows=len(raw),
        fallback_fits=int(summary.selection_fallback.sum()),run_fingerprint=run['run_fingerprint'],
        inventory_sha256=sha(PACKAGE/'gpu_output/FIT_INVENTORY.tsv')))
    return raw


if __name__=='__main__':
    x=validate();print('Validated 480 fits and %d test predictions; validation-only selection.'%len(x),flush=True)
