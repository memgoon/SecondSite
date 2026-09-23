"""Read only 120 expected fit paths; independently validate returned artifacts."""
import json
import numpy as np
import pandas as pd
from common import PACKAGE, ARMS, expected_jobs, fit_dir, sha, digest, verify_contract, read_frame, partition, write_json


def collect():
    c = verify_contract()
    run = json.loads((PACKAGE / 'gpu_output/RUN_CONTRACT.json').read_text())
    assert run['status'] == 'validated' and run['cpu_manifest_digest'] == c['manifest_digest']
    baseline_run = json.loads((PACKAGE / 'data/BASELINE_RUN_CONTRACT.json').read_text())
    assert run['embedding_files'] == baseline_run['embedding_files']
    assert run['parameter_counts'] == baseline_run['parameter_counts']
    assert run['run_fingerprint'] == digest(dict(cpu=c['manifest_digest'], resources=run['embedding_files'],
        torch=run['versions']['torch'], numpy=run['versions']['numpy'], pandas=run['versions']['pandas'],
        torch_num_threads=run['torch_num_threads']))
    assert all(run['training_contract'][k] == v for k, v in c['training'].items())
    assert run['training_contract']['checkpoint_selection'] == c['checkpoint_selection']
    assert run['split_sha256'] == c['files']['data/SPLIT_MEMBERSHIP.tsv.gz']
    frames = {a: read_frame(a) for a in ARMS}
    all_predictions, reports, hashes = [], [], {}
    expected_contract = dict(run['training_contract'], run_fingerprint=run['run_fingerprint'],
                             split_sha256=run['split_sha256'])
    for a, model, seed, fold in expected_jobs():
        directory = fit_dir(a, model, seed, fold)
        r = json.loads((directory / 'FIT_REPORT.json').read_text())
        expected = dict(expected_contract, cohort_sha256=run['cohort_sha256'][a])
        identity = dict(cohort_arm=a, model=model, seed=seed, outer_fold=fold,
                        validation_fold=(fold + 1) % 5, regime='pfam_murcko')
        assert r['status'] == 'validated' and all(r[k] == v for k, v in identity.items())
        assert r['training_contract'] == expected
        assert r['probability_conversion'] == 'sigmoid applied after FP32 logit cast'
        assert r['model_version'] == 'role_complete_matrix_v2'
        for name, key in [('best.pt', 'checkpoint_sha256'), ('predictions.tsv.gz', 'prediction_sha256'),
                          ('history.tsv', 'history_sha256')]:
            value = sha(directory / name)
            assert value == r[key], str(directory / name)
            hashes[str((directory / name).relative_to(PACKAGE))] = value
        for name in ('FIT_REPORT.json',):
            hashes[str((directory / name).relative_to(PACKAGE))] = sha(directory / name)
        train, val, test, _ = partition(frames[a], fold)
        assert r['split_counts']['train'] == len(train)
        assert r['split_counts']['validation'] == len(val)
        assert r['split_counts']['test'] == len(test)
        p = pd.read_csv(directory / 'predictions.tsv.gz', sep='\t', dtype={'main_row_id': str})
        assert len(p) == len(test) and not p.main_row_id.duplicated().any()
        assert set(p.main_row_id) == set(test.main_row_id)
        test = test.set_index('main_row_id').loc[p.main_row_id]
        for field in ('binary_label', 'uniprot', 'family_component_id', 'full_inchikey', 'connectivity_key'):
            assert p[field].astype(str).tolist() == test[field].astype(str).tolist(), field
        for field, value in identity.items():
            assert p[field].eq(value).all(), field
        assert p.unseen_compound.eq(1).all()
        assert np.isfinite(p.p_allosteric).all() and p.p_allosteric.between(0, 1).all()
        expected_groups = int((val.groupby('uniprot' if model == 'ligand' else 'connectivity_key').binary_label.nunique() == 2).sum())
        assert r['checkpoint_selection_fallback_at_best_epoch'] == (expected_groups < 8)
        history = pd.read_csv(directory / 'history.tsv', sep='\t')
        assert 1 <= r['best_epoch'] <= len(history) <= 25
        assert r['best_epoch'] in set(history.epoch)
        all_predictions.append(p)
        reports.append({k: r[k] for k in ['cohort_arm', 'model', 'seed', 'outer_fold', 'best_epoch',
                        'checkpoint_selection_fallback_at_best_epoch', 'best_validation_score', 'elapsed_seconds']})
    predictions = pd.concat(all_predictions, ignore_index=True)
    assert len(predictions) == len(ARMS) * 8 * 3 * 4637
    assert sum(x['checkpoint_selection_fallback_at_best_epoch'] for x in reports) == c['expected_fallback_fits']
    write_json(PACKAGE / 'gpu_output/FITS_VALIDATION.json', dict(status='validated', observed_fits=len(expected_jobs()),
        prediction_rows=len(predictions), run_fingerprint=run['run_fingerprint'],
        all_prediction_metadata_rejoined=True, all_three_partition_overlap_zero=True,
        sha256=hashes))
    pd.DataFrame(reports).to_csv(PACKAGE / 'gpu_output/FIT_SUMMARY.tsv', sep='\t', index=False)
    return predictions


if __name__ == '__main__':
    collect()
    print('PASS: all 120 fits and prediction/checkpoint hashes validated.')
