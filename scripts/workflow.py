"""Browse ordered scientific entrypoints without launching an analysis."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['list', 'show'])
    p.add_argument('--stage', choices=[f'{i:02d}' for i in range(1, 9)])
    a = p.parse_args()
    stages = json.loads((ROOT/'workflow/STAGES.json').read_text())
    if a.action == 'show' and not a.stage:
        p.error('show requires --stage')
    for stage in stages:
        if a.stage and a.stage != stage['stage']:
            continue
        print(stage['stage'] + '  ' + stage['title'])
        if a.action == 'show':
            print(stage['description'])
            print('\n'.join(stage['files']))

if __name__ == '__main__':
    main()
