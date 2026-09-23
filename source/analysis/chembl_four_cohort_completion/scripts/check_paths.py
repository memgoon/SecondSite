"""Read-only, manifest-only path check. No recursion, tensor reads or torch."""
import hashlib
import json
from pathlib import Path


def main():
    package=Path(__file__).resolve().parents[1]
    root=package.parents[1]
    c=json.loads((package/'validation/CPU_CONTRACT.json').read_text())
    fingerprint=hashlib.sha256(json.dumps(c['payload'],sort_keys=True).encode()).hexdigest()
    if c.get('status')!='validated' or fingerprint!=c.get('fingerprint'):
        raise ValueError('Invalid CPU contract')
    paths=c['payload']['files']
    missing=[str(root/p) for p in paths if not (root/p).is_file()]
    print(json.dumps(dict(status='missing_files' if missing else 'contract_paths_present',
        contract_files=len(paths),missing_count=len(missing),missing_paths=missing,
        scope='Contract paths only, including all 33 checkpoints and reports; tensor caches and file hashes are checked by GPU preflight',
        recursive_scan=False),indent=2))
    if missing:raise SystemExit(1)


if __name__=='__main__':main()
