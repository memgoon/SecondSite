"""Read only explicitly contracted fit paths; independently validate returned artifacts."""
import json
import numpy as np
import pandas as pd
from common import *


def validate():
    contract = verify_contract()
    run = json.loads((PACKAGE / 'gpu_output/RUN_CONTRACT.json').read_text())
    core = {k: run[k] for k in ('cpu_manifest','input_hashes','training','torch_version','numpy_version','pandas_version','version')}
    if (digest(core) != run['run_fingerprint'] or run['cpu_manifest'] != contract['manifest_digest']
            or run['training'] != TRAINING or run['status'] != 'validated'):
        raise ValueError('Run contract mismatch')
    expected_paths = set(read_data().ligand_embedding_path) | set(read_data().protein_embedding_path)
    if set(run['input_hashes']) != expected_paths:
        raise ValueError('Input-cache manifest incomplete')
    frame = read_data()
    records, reports, inventory = [], [], []
    for job in jobs():
        report = compatible_report(job, run)
        if report is None:
            raise ValueError('Missing/incompatible/corrupt fit: '+str(job))
        train, test, excluded = partition(frame, job[0], job[3])
        for k,v in dict(train_rows=len(train),test_rows=len(test),excluded_rows=len(excluded),held_out_family=families(frame)[job[3]]).items():
            if report[k] != v:
                raise ValueError('Split report mismatch '+str(job))
        path = fit_dir(job)
        p = pd.read_csv(path/'predictions.tsv.gz', sep='\t', dtype={'main_row_id':str})
        if p.main_row_id.duplicated().any() or set(p.main_row_id) != set(test.main_row_id):
            raise ValueError('Test coverage mismatch')
        a = p.set_index('main_row_id').sort_index()
        b = test.set_index('main_row_id').sort_index()
        for field in ('uniprot','family_component_id','full_inchikey','connectivity_key','binary_label',
                      'unseen_compound','ligand_embedding_path','protein_embedding_path'):
            if not a[field].astype(str).equals(b[field].astype(str)):
                raise ValueError('Prediction metadata mismatch: '+field)
        for k,v in identity(job).items():
            if not p[k].astype(str).eq(str(v)).all():
                raise ValueError('Prediction identity mismatch '+k)
        scores = p.p_allosteric.to_numpy()
        if not (np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()):
            raise ValueError('Invalid probabilities')
        key = {'ligand':'ligand_embedding_path','protein':'uniprot'}.get(job[1])
        if key and not p.groupby(key).p_allosteric.nunique().eq(1).all():
            raise ValueError('Single-input identity not constant')
        history = pd.read_csv(path/'history.tsv',sep='\t')
        if list(history.columns) != ['epoch','training_loss','elapsed_seconds'] or list(history.epoch) != list(range(1,26)):
            raise ValueError('Fixed training history mismatch')
        if not np.isfinite(history[['training_loss','elapsed_seconds']].to_numpy()).all():
            raise ValueError('Nonfinite training history')
        records.append(p)
        reports.append({k:v for k,v in report.items() if k not in ('training','sha256')})
        inventory.append(dict(identity(job), report_sha256=sha(path/'FIT_REPORT.json'), **report['sha256']))
    prediction = pd.concat(records,ignore_index=True)
    if len(prediction) != contract['expected_prediction_rows']:
        raise ValueError('Total prediction coverage')
    for _, g in prediction.groupby(['regime','model','seed']):
        if len(g)!=395 or g.main_row_id.nunique()!=395:
            raise ValueError('OOF coverage')
    pd.DataFrame(reports).to_csv(PACKAGE/'gpu_output/FIT_SUMMARY.tsv',sep='\t',index=False)
    pd.DataFrame(inventory).to_csv(PACKAGE/'gpu_output/FIT_INVENTORY.tsv',sep='\t',index=False)
    write_json(PACKAGE/'gpu_output/FITS_VALIDATION.json',dict(status='validated',fits=len(reports),
        prediction_rows=len(prediction),run_fingerprint=run['run_fingerprint'],
        inventory_sha256=sha(PACKAGE/'gpu_output/FIT_INVENTORY.tsv'), validation_rows=0))
    return prediction


if __name__ == '__main__':
    result = validate()
    print('Validated 2,256 fits; %d prediction rows; no retraining.' % len(result),flush=True)
