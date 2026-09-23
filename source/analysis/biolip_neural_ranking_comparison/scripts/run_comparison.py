"""Compute CPU summaries/paired confidence intervals, never new model scores."""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from common import *
from statistics_core import auc,spearman,top1_agreement,bootstrap_values

MULT=None
CODES=None
SIZES=None
NC=None


def bootstrap_job(item):
    meta,values=item;v=np.asarray(values,float);ok=np.isfinite(v)
    result=dict(meta,estimate=float(np.mean(v[ok])) if ok.any() else np.nan,
        n_proteins_used=int(ok.sum()),n_proteins_skipped=int((~ok).sum()),
        n_pairs_used=int(SIZES[ok].sum()),n_families_used=int(len(set(CODES[ok]))),
        n_families_resampled=NC,replicates_requested=len(MULT),ci_low=np.nan,ci_high=np.nan,
        valid_replicates=0,invalid_replicates=len(MULT),minimum_valid_fraction=.95,
        bootstrap_unit='union_of_source_family_components',
        interval_scope='pointwise, conditional on fixed scores; not multiplicity-adjusted')
    if not ok.any():result['status']='not_defined_no_eligible_nonconstant_groups';return result
    if len(set(CODES[ok]))<2:result['status']='insufficient_family_support';return result
    draws=bootstrap_values(v,CODES,MULT,NC);valid=draws[np.isfinite(draws)]
    result.update(valid_replicates=len(valid),invalid_replicates=len(draws)-len(valid))
    if len(valid)<.95*len(draws):result['status']='insufficient_valid_replicates';return result
    result.update(ci_low=float(np.percentile(valid,2.5)),ci_high=float(np.percentile(valid,97.5)),status='evaluated')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    c=check_contract();out=PACKAGE/'cpu_output';out.mkdir(exist_ok=True)
    final=out/'VALIDATION.json'
    if final.exists():
        old=read_json(final);assert old['cpu_fingerprint']==c['fingerprint']
        for name,h in old['sha256'].items():assert sha(out/name)==h,name
        assert old['status']=='validated'
        print('Already complete; output hashes match. No bootstrap repeated.');return
    f=pd.read_csv(PACKAGE/'data/MATCHED_SCORES.tsv.gz',sep='\t')
    groups=list(f.groupby('uniprot',sort=True));uids=[u for u,g in groups]
    global MULT,CODES,SIZES,NC
    families=[g.bootstrap_family.iloc[0] for _,g in groups]
    CODES,labels=pd.factorize(families,sort=True);NC=len(labels)
    SIZES=np.array([len(g) for _,g in groups]);rng=np.random.RandomState(SEED)
    MULT=rng.multinomial(NC,np.full(NC,1/NC),size=REPLICATES).astype(np.float64)
    assert NC==c['payload']['bootstrap_clusters']
    jobs=[];per=[];pooled=[];rho={};top={};aucs={}
    for model in MODELS:
        col='nn_'+model;aucs[model]=np.array([auc(g.binary_label,g[col]) for _,g in groups])
        pooled.append(dict(kind='label_auroc',model=model,biolip_method='',value=auc(f.binary_label,f[col]),n_pairs=len(f),role='descriptive_cross_fold_pooled'))
        jobs.append((dict(metric='label_auroc',model=model,biolip_method='',comparison='absolute'),aucs[model]))
        for method in METHODS:
            a=[];b=[]
            for u,g in groups:
                r=spearman(g[col],g[method]) if len(g)>=MIN_RHO_ROWS else np.nan
                t=top1_agreement(g[col],g[method]) if len(g)>=2 else np.nan
                chance=1/len(g) if len(g)>=2 else np.nan
                a.append(r);b.append(t)
                per.append(dict(uniprot=u,bootstrap_family=g.bootstrap_family.iloc[0],rows=len(g),
                    model=model,biolip_method=method,spearman=r,top1_agreement=t,
                    top1_random_expectation=chance,top1_above_random=t-chance,
                    neural_constant=g[col].nunique()==1,biolip_constant=g[method].nunique()==1))
            rho[model,method]=np.array(a);top[model,method]=np.array(b)
            chance=np.where(SIZES>=2,1/SIZES,np.nan)
            for metric,values in [('spearman',np.array(a)),('top1_agreement',np.array(b)),('top1_above_random',np.array(b)-chance)]:
                jobs.append((dict(metric=metric,model=model,biolip_method=method,comparison='absolute'),values))
            pooled.append(dict(kind='spearman',model=model,biolip_method=method,
                value=spearman(f[col],f[method]),n_pairs=len(f),role='descriptive_cross_fold_pooled'))
    for method in METHODS:
        val=np.array([auc(g.binary_label,g[method]) for _,g in groups])
        jobs.append((dict(metric='label_auroc',model='biolip',biolip_method=method,comparison='absolute'),val))
        pooled.append(dict(kind='label_auroc',model='biolip',biolip_method=method,value=auc(f.binary_label,f[method]),n_pairs=len(f),role='descriptive_cross_fold_pooled'))
    for model in MODELS[2:]:
        jobs.append((dict(metric='label_auroc',model=model,biolip_method='',comparison='joint_minus_ligand'),aucs[model]-aucs['ligand']))
        for method in METHODS:
            # NaN subtraction ensures a paired intersection, not two different supports.
            for metric,values in [('spearman',rho[model,method]-rho['ligand',method]),
                                  ('top1_agreement',top[model,method]-top['ligand',method])]:
                jobs.append((dict(metric=metric,model=model,biolip_method=method,comparison='joint_minus_ligand'),values))
    print('Statistics:',len(jobs),'comparisons;',REPLICATES,'paired family draws;',NC,'families',flush=True)
    results=[]
    if args.workers==1:
        for job in jobs:results.append(bootstrap_job(job))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs)),mp_context=mp.get_context('fork')) as pool:
            for i,result in enumerate(pool.map(bootstrap_job,jobs),1):
                results.append(result)
                if i%20==0 or i==len(jobs):print('Bootstrap %d/%d'%(i,len(jobs)),flush=True)
    result=pd.DataFrame(results)
    assert not result.status.eq('insufficient_valid_replicates').any(),'Too few defined replicates; no completion PASS'
    checks=result[(result.model=='protein')&(result.comparison=='absolute')]
    assert checks[checks.metric=='spearman'].estimate.isna().all()
    assert np.allclose(checks[checks.metric=='top1_above_random'].estimate,0)
    assert np.allclose(checks[checks.metric=='label_auroc'].estimate,.5)
    result.to_csv(out/'WITHIN_PROTEIN_METRICS_AND_CI.tsv',sep='\t',index=False)
    pd.DataFrame(per).to_csv(out/'PER_PROTEIN_AGREEMENT.tsv.gz',sep='\t',index=False,compression='gzip')
    pd.DataFrame(pooled).to_csv(out/'POOLED_DESCRIPTIVE.tsv',sep='\t',index=False)
    label_rows=[]
    for u,g in groups:
        for model in MODELS:
            label_rows.append(dict(uniprot=u,bootstrap_family=g.bootstrap_family.iloc[0],model=model,
                rows=len(g),positive_rows=int(g.binary_label.sum()),auroc=auc(g.binary_label,g['nn_'+model])))
    pd.DataFrame(label_rows).to_csv(out/'PER_PROTEIN_LABEL_AUROC.tsv',sep='\t',index=False)
    # Persist bootstrap draw identities compactly so all paired comparisons can be audited.
    np.savez_compressed(str(out/'BOOTSTRAP_DRAWS.npz'),multiplicities=MULT.astype(np.int32),family_labels=np.array(labels,dtype=str))
    (out/'REPORT.md').write_text(
        '# BioLiP / neural ranking comparison\n\n'
        'Completed on 987 exact pairs / 278 proteins. Label-conditioned AUROC support: 409 pairs / 81 proteins.\n\n'
        'No new training or inference. Neural scores: every-pair, Pfam held-out, three-seed mean. '
        'BioLiP scores: its own family-held-out predictions; four prespecified methods. '
        'Source folds differ. This is a cross-method analysis of shared reference cases, NOT independent external validation.\n\n'
        'Primary: mean within-protein Spearman (at least 3 pairs, both scores varying), '
        'tie-aware top-1 agreement (at least 2 pairs), and within-protein label AUROC (both labels). '
        'All metrics give each eligible protein equal weight. Supports vary across metrics; paired differences use intersections.\n\n'
        '95% pointwise intervals use 10,000 common draws of the union of both source family partitions. '
        'Multiplicity is preserved. Intervals condition on fixed predictions, not retraining, and are not adjusted for multiple testing. '
        'No permutation p-values or functional-validation claims are made.\n\n'
        'Constant protein-only scores have undefined within-protein Spearman, expected top-1 agreement 1/n, and label AUROC 0.5. '
        'A high agreement can follow label separation or shared chemical features. Distance-based reference definitions are not independent allostery evidence.\n\n'
        'Inspect WITHIN_PROTEIN_METRICS_AND_CI.tsv including support/status; use joint_minus_ligand rows to ask whether '
        'agreement exceeds the ligand-only baseline. POOLED_DESCRIPTIVE.tsv is secondary because it mixes targets and fitted fold score scales.\n')
    names=['WITHIN_PROTEIN_METRICS_AND_CI.tsv','PER_PROTEIN_AGREEMENT.tsv.gz','POOLED_DESCRIPTIVE.tsv',
           'PER_PROTEIN_LABEL_AUROC.tsv','BOOTSTRAP_DRAWS.npz','REPORT.md']
    write_json(final,dict(status='validated',cpu_fingerprint=c['fingerprint'],rows=987,proteins=278,
        comparisons=len(jobs),bootstrap_replicates=REPLICATES,bootstrap_clusters=NC,workers=args.workers,
        status_counts=result.status.value_counts().to_dict(),paired_common_draws=True,
        structural_constant_controls_passed=True,independent_external_validation=False,
        training_fits=0,inference_jobs=0,sha256={n:sha(out/n) for n in names}))
    print('Complete:',final,flush=True)


if __name__=='__main__':main()
