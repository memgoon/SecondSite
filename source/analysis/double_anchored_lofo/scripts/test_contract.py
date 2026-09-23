"""CPU regression tests without touching prior experiment outputs."""
import ast
import json
import subprocess
import numpy as np
import pandas as pd
from common import *
from metrics import auc,ap,stats,resample_family_statistics


def main():
    contract=verify_contract()
    for path in (PACKAGE/'scripts').iterdir():
        if path.suffix=='.py':
            ast.parse(path.read_text(),filename=str(path))
        elif path.suffix=='.sh':
            subprocess.run(['bash','-n',str(path)],check=True)
    d=read_data()
    assert len(jobs())==2256 and contract['expected_prediction_rows']==18960
    assert len(set(jobs()))==2256
    for fold in range(47):
        a,t,_=partition(d,'family_only',fold)
        b,u,e=partition(d,'double_unseen',fold)
        assert set(t.main_row_id)==set(u.main_row_id)
        assert set(b.main_row_id)<=set(a.main_row_id)
        assert not (set(b.connectivity_key)&set(u.connectivity_key))
        assert not (set(b.family_component_id)&set(u.family_component_id))
        assert len(t)+len(a)==395 and len(u)+len(b)+len(e)==395
    oof=pd.concat([partition(d,'double_unseen',f)[1].assign(outer_fold=f) for f in range(47)])
    oof['p_allosteric']=.5
    for field,rows,groups in [('uniprot',395,98),('full_inchikey',42,14)]:
        s=stats(oof,field)
        assert (s['n_rows_used'],s['n_groups_used'])==(rows,groups)
        assert s['auroc']==.5
    assert auc([0,1],[.1,.9])==1 and auc([0,1],[.9,.1])==0
    assert auc([0,1],[.5,.5])==.5
    assert ap([0,1],[.5,.5])==.5
    sampled,_=resample_family_statistics([1.,0.],[1.,1.],np.array([[0,0,1]]))
    assert abs(sampled[0]-2./3.)<1e-12
    # Exercise actual bootstrap function and paired computation on a small synthetic family table.
    from aggregate import bootstrap
    r=bootstrap(('constant_delta',np.array([.2,.4]),np.array([1.,2.]),10000))
    assert r['valid_replicates']==10000 and abs(r['estimate']-.2)<1e-12
    assert abs(r['ci_low']-.2)<1e-12 and abs(r['ci_high']-.2)<1e-12
    result=dict(status='validated',fits=2256,partitions=94,primary_rows=395,
        primary_groups=98,secondary_rows=42,secondary_groups=14,multiplicity='A,A,B=2/3',
        cuda_execution_tested=False,manifest_digest=contract['manifest_digest'])
    write_json(PACKAGE/'validation/STATIC_TESTS.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
