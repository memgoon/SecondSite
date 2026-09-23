"""Read-only validation of the source release; never import analysis entrypoints."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CURRENT = [
    'role_complete_pair_matrix/scripts/train_matrix.py',
    'explicit_double_unseen/scripts/train_gpu.py',
    'double_anchored_lofo/scripts/gpu_runtime.py',
    'ligand_anchored_benchmark/scripts/gpu_runtime.py',
    'murcko_holdout_extension/scripts/train_gpu.py',
    'chembl_four_cohort_completion/scripts/runtime.py',
    'biolip_consensus60_pipeline_v2/scripts/run_pipeline.py',
    'biolip_neural_ranking_comparison/scripts/run_comparison.py',
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def safe_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe relative path: ' + relative)
    full = (root / p).resolve()
    full.relative_to(root.resolve())
    return full

def verify_index(root, rows):
    paths = [r['release_path'] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate source-index path')
    for row in rows:
        p = safe_path(root, row['release_path'])
        if not p.is_file() or sha(p) != row['sha256']:
            raise ValueError('Source checksum mismatch: ' + row['release_path'])

def validate(root, require_manifest=True):
    root = root.resolve()
    with (root/'provenance/SOURCE_INDEX.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    verify_index(root, rows)
    stages = json.loads((root/'workflow/STAGES.json').read_text())
    assert [s['stage'] for s in stages] == [f'{i:02d}' for i in range(1, 9)]
    excluded_packages = {'secondsite_web_handoff', 'secondsite_biolip_consensus60_release_v1', 'manuscript_figures'}
    assert not any(r['package'] in excluded_packages for r in rows)
    assert not any(r['source_path'].endswith('/render_main_figure.py') for r in rows)
    for package in excluded_packages:
        assert not (root/'source/analysis'/package).exists(), package
    assert not (root/'workflow/09').exists() and not (root/'workflow/10').exists()
    indexed = {r['release_path'] for r in rows}
    assigned = [p for s in stages for p in s['files']]
    assert len(assigned) == len(set(assigned)) and set(assigned) == indexed
    for p in REQUIRED_CURRENT:
        assert 'source/analysis/' + p in indexed, p
    python_count = shell_count = 0
    patterns = [r'\bgh[pousr]_[A-Za-z0-9]{25,}\b', r'\bhf_[A-Za-z0-9]{25,}\b',
                r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    for row in rows:
        p = safe_path(root, row['release_path'])
        s = p.read_text()
        assert not any(re.search(pattern, s) for pattern in patterns), 'Credential pattern: '+str(p)
        if p.suffix == '.py':
            ast.parse(s, filename=row['release_path'])
            compile(s, row['release_path'], 'exec')
            python_count += 1
        elif p.suffix == '.sh':
            subprocess.run(['bash', '-n', str(p)], check=True, capture_output=True)
            shell_count += 1
    manifest = root/'manifests/CHECKSUMS.sha256'
    manifest_count = 0
    if require_manifest or manifest.exists():
        lines = manifest.read_text().splitlines()
        names = []
        for line in lines:
            digest, name = line.split('  ', 1)
            assert re.fullmatch(r'[a-f0-9]{64}', digest)
            assert sha(safe_path(root, name)) == digest, name
            names.append(name)
        assert len(names) == len(set(names))
        assert indexed.issubset(names)
        manifest_count = len(names)
    return dict(status='PASS_SOURCE_PACKAGE_CHECKS', indexed_source_files=len(rows),
                python_syntax_files=python_count, shell_syntax_files=shell_count,
                workflow_stages=len(stages), manifest_targets_checked=manifest_count,
                read_only=True, network_performed=False, production_analysis_executed=False,
                clean_machine_end_to_end_reproduction_verified=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--before-manifest', action='store_true')
    args = parser.parse_args()
    print(json.dumps(validate(args.root, require_manifest=not args.before_manifest), indent=2))

if __name__ == '__main__':
    main()
