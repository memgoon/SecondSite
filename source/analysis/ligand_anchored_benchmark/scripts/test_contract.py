"""CPU split, selection, metric, archive and bootstrap regression tests."""
import ast
import subprocess
import numpy as np
import pandas as pd
from common import *
from metrics import auc,ap,resample_family_statistics


def main():
    c=verify_contract();d=read_data()
    for p in (PACKAGE/'scripts').iterdir():
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        elif p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True)
    assert len(jobs())==len(set(jobs()))==480
    expected_ligand={'row_random':(884,151),'unseen_family':(859,122),'unseen_ligand':(1365,122),'double_unseen':(859,122)}
    fallback=0
    for regime in REGIMES:
        seen=[];pieces=[]
        for fold in range(5):
            tr,va,te,ex=partition(d,regime,fold);seen.extend(te.main_row_id)
            pieces.append(te.assign(outer_fold=fold,p_allosteric=.5))
            assert len(tr)+len(va)+len(te)+len(ex)==1365
            for m in MODELS:
                sel=checkpoint_selection(va.assign(p_allosteric=.5),regime,m)
                assert abs(sel['selection_score']-.5)<1e-12
                fallback+=3*int(sel['selection_fallback'])
        assert len(seen)==len(set(seen))==1365
        result=stats(pd.concat(pieces),'full_inchikey')
        assert (result['n_rows_used'],result['n_groups_used'])==expected_ligand[regime]
    assert fallback==c['expected_fallback_fits']
    assert auc([0,1],[.5,.5])==.5 and ap([0,1],[.5,.5])==.5
    value,_=resample_family_statistics([1.,0.],[1.,1.],np.array([[0,0,1]]));assert abs(value[0]-2/3)<1e-12
    h=pd.DataFrame(dict(epoch=range(1,7),selection_score=[.5]*6));assert replay_history(h)==(1,.5)
    h2=pd.DataFrame(dict(epoch=range(1,8),selection_score=[.4,.6,.5,.55,.59,.6,.59]));assert replay_history(h2)==(2,.6)
    try:replay_history(h.iloc[:4])
    except ValueError:pass
    else:raise AssertionError('Premature stop accepted')
    from aggregate import family_deletion_statistics
    a=pd.DataFrame(dict(outer_fold=[0]*6,full_inchikey=['x','x','x','y','y','y'],
        family_component_id=['A','B','C','A','B','C'],binary_label=[1,0,1,0,1,0],
        p_allosteric=[.8,.7,.6,.5,.4,.3]))
    fast=family_deletion_statistics(a).set_index('deleted_family')
    for f in ['A','B','C']:
        slow=stats(a[a.family_component_id.ne(f)],'full_inchikey')
        assert (np.isnan(slow['auroc']) and np.isnan(fast.loc[f,'auroc'])) or abs(slow['auroc']-fast.loc[f,'auroc'])<1e-12
        assert slow['n_rows_used']==fast.loc[f,'n_rows_used']
    result=dict(status='validated',fits=480,test_predictions=131040,partitions=20,
        expected_fallback_fits=fallback,main_family_support_rows=859,main_family_support_blocks=122,
        matrix_cells=16,manifest_digest=c['manifest_digest'],cuda_execution_tested=False)
    write_json(PACKAGE/'validation/STATIC_TESTS.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
