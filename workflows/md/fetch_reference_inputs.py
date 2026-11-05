"""Fetch only the pinned AMMo reference inputs listed in REFERENCE_INPUTS.json."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=str(Path(__file__).resolve().parent/'refs'))
    p.add_argument('--local-ammo',help='Use an existing AMMo checkout instead of downloading')
    a=p.parse_args(); root=Path(__file__).resolve().parent
    spec=json.loads((root/'REFERENCE_INPUTS.json').read_text())
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for row in spec['files']:
        dest=out/row['destination']
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest()!=row['sha256']:
                raise SystemExit('Existing reference differs: '+str(dest))
            continue
        if a.local_ammo:
            content=(Path(a.local_ammo)/row['source']).read_bytes()
        else:
            url='https://raw.githubusercontent.com/michellab/AMMo/'+spec['revision']+'/'+row['source']
            with urllib.request.urlopen(url,timeout=60) as response:content=response.read()
        if hashlib.sha256(content).hexdigest()!=row['sha256']:
            raise SystemExit('Reference checksum mismatch: '+row['source'])
        dest.write_bytes(content)
    print('Verified',len(spec['files']),'reference files at',out)


if __name__=='__main__':main()
