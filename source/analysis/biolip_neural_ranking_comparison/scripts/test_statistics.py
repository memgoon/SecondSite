"""Synthetic tests only: no real-analysis results are produced."""
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from statistics_core import auc,spearman,top1_agreement,bootstrap_values
import run_comparison as runtime


def main():
    assert np.isclose(spearman([1,1,3],[2,2,6]),1)
    assert np.isclose(spearman([1,2,3],[3,2,1]),-1)
    assert np.isnan(spearman([1,1,1],[1,2,3]))
    assert auc([0,1],[.4,.4])==.5
    assert auc([0,1],[.1,.9])==1
    assert np.isnan(auc([1,1],[.1,.9]))
    assert np.isclose(top1_agreement([1,1,1],[1,2,3]),1/3)
    assert np.isclose(top1_agreement([1,1,1],[2,2,2]),1/3)
    assert top1_agreement([2,2,0],[0,2,2])==.25
    draws=np.array([[2,1],[1,2],[0,3]])
    got=bootstrap_values([1,0],[0,1],draws,2)
    assert np.allclose(got,[2/3,1/3,0])
    # Nested family A has two proteins; occurrences must weight BOTH proteins.
    got=bootstrap_values([1,0,0],[0,0,1],draws,2)
    assert np.allclose(got,[2/5,1/4,0])
    assert np.isnan(bootstrap_values([np.nan,0],[0,1],np.array([[3,0]]),2)[0])
    rng=np.random.RandomState(2)
    for _ in range(50):
        order=rng.permutation(4);a=np.array([2,2,0,1]);b=np.array([0,2,2,1])
        assert top1_agreement(a,b)==top1_agreement(a[order],b[order])
    # Exercise the actual report/worker path with synthetic groups only.
    runtime.CODES=np.array([0,0,1]);runtime.NC=2;runtime.SIZES=np.array([3,4,5])
    runtime.MULT=np.array([[2,0],[1,1],[0,2]]*100,float)
    meta=dict(metric='synthetic',model='test',biolip_method='',comparison='absolute')
    item=(meta,np.array([1.,0.,0.]))
    serial=runtime.bootstrap_job(item)
    assert np.isclose(serial['estimate'],1/3) and serial['valid_replicates']==300
    assert serial['n_pairs_used']==12 and serial['n_proteins_used']==3
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('fork')) as pool:
        parallel=list(pool.map(runtime.bootstrap_job,[item,item]))
    assert parallel==[serial,serial]
    empty=runtime.bootstrap_job((meta,np.array([np.nan]*3)))
    assert empty['status']=='not_defined_no_eligible_nonconstant_groups'
    print('PASS: tied ranks, constant controls, top-1 ties, A,A,B multiplicity, undefined replicates')


if __name__=='__main__':main()
