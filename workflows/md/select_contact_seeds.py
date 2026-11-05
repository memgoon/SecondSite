"""Write the pocket-contacting H5 seed list from the complete H4 strata table."""
import argparse
import json
import math
from pathlib import Path


def select(data):
    if set(data['legs']) != {'open','closed'}:
        raise ValueError('Require both complete steering directions')
    chosen=[]; excluded=[]
    for leg in ('open','closed'):
        rows=data['legs'][leg]['rows']
        if len(rows)!=100 or {int(r['seed']) for r in rows}!=set(range(1,101)):
            raise ValueError('Require the original 100 distinct seeds per direction')
        for r in rows:
            if r['stratum'] not in ('FRJ_site_contact','other_surface_contact','no_protein_contact'):
                raise ValueError('Unknown or missing contact classification')
            if r.get('wpd') is None or not math.isfinite(float(r['wpd'])):
                raise ValueError('Missing WPD coverage measurement')
            item=dict(leg=leg,seed=int(r['seed']),stratum=r['stratum'])
            (chosen if r['stratum']=='FRJ_site_contact' else excluded).append(item)
    return chosen,excluded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strata',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    chosen,excluded=select(json.loads(Path(a.strata).read_text()))
    out=Path(a.out)
    if out.exists() or Path(str(out)+'.json').exists():
        raise SystemExit('Output already exists; choose a new destination')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(f"{r['leg']}:{r['seed']}\n" for r in chosen))
    Path(str(out)+'.json').write_text(json.dumps(dict(selected=chosen,excluded=excluded,
        rule='At least one FRJ-site residue in contact within 4.5 angstroms under periodic minimum-image distances.'),indent=2)+'\n')
    print('Selected',len(chosen),'Excluded',len(excluded))


if __name__=='__main__':main()
