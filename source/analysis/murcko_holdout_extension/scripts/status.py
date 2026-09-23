"""Lightweight progress: inspect only 120 expected FIT_REPORT paths."""
import json
from collections import Counter, defaultdict
from common import PACKAGE, ARMS, expected_jobs, fit_dir

path = PACKAGE / 'gpu_output/RUN_CONTRACT.json'
if not path.is_file():
    print('GPU preflight not completed yet: 0/120 fits.')
    raise SystemExit(0)
run = json.loads(path.read_text())
counts, times = Counter(), defaultdict(list)
for arm, model, seed, fold in expected_jobs():
    path = fit_dir(arm, model, seed, fold) / 'FIT_REPORT.json'
    if not path.is_file():
        continue
    try:
        report = json.loads(path.read_text())
        if report['status'] == 'validated' and report['training_contract']['run_fingerprint'] == run['run_fingerprint']:
            counts[arm] += 1
            times[model].append(report.get('elapsed_seconds', 0.))
    except (ValueError, KeyError):
        pass
print('Completed reports: %d/120 (lightweight count, not full hash validation)' % sum(counts.values()))
for arm in ARMS:
    print('%s: %d/120' % (arm, counts[arm]))
for model, values in sorted(times.items()):
    print('%s: %d/15 fits, observed mean %.1f min/fit' % (model, len(values), sum(values)/len(values)/60))
