"""CPU-only, frozen comparison of existing held-out predictions."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
BIO = ROOT/'analysis/biolip_consensus60_pipeline_v2/runs/complete_reanalysis'
NN = ROOT/'analysis/role_complete_pair_matrix'
MODELS = ('ligand','protein','c1','c2','c3','d1','d2','d3')
METHODS = ('RAW:RAW_D','KDE_LEGACY:D+AR','KDE_LEGACY:D+MW','QNB:D+AR')
VERSION = 'biolip_neural_heldout_comparison_v1'
SEED = 20260922
REPLICATES = 10000
MIN_RHO_ROWS = 3
FILES = {
    'reference': BIO/'data/REFERENCE_PAIRS.tsv',
    'biolip_oof': BIO/'data/OOF_PAIR_SCORES.tsv.gz',
    'neural_oof': NN/'gpu_output/benchmark/aggregate/ENSEMBLE_OOF_PREDICTIONS.tsv.gz',
    'cohort': NN/'data/EVERY_PAIR.tsv.gz',
    'biolip_validation': BIO/'VALIDATION.json',
    'neural_validation': NN/'gpu_output/benchmark/aggregate/AGGREGATE_VALIDATION.json',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(str(temp),str(path))


def implementation_files():
    # One named directory only, no recursive filesystem traversal.
    return [PACKAGE/'README.md'] + sorted(p for p in (PACKAGE/'scripts').iterdir()
        if p.is_file() and p.suffix in ('.py','.sh'))


def check_contract():
    c=read_json(PACKAGE/'validation/CPU_CONTRACT.json')
    assert c['status']=='validated' and c['version']==VERSION
    assert digest(c['payload'])==c['fingerprint']
    for name,h in c['payload']['files'].items():
        assert sha(ROOT/name)==h, 'Changed/missing frozen file: '+name
    return c
