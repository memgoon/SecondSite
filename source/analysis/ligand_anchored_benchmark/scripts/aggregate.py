"""Local CPU metrics and conditional ligand-cluster bootstrap; no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from common import *
from metrics import auc,stats,resample_family_statistics
from validate_fits import validate


def cluster_bootstrap(task):
    label,sums,counts=task
    seed=int(hashlib.sha256(label.encode()).hexdigest()[:8],16)
    draws=np.random.RandomState(seed).randint(len(counts),size=(10000,len(counts)))
    values,n=resample_family_statistics(sums,counts,draws)
    ok=len(values)>=9500
    return dict(comparison=label,estimate=float(sums.sum()/counts.sum()),
        ci_low=float(np.quantile(values,.025)) if ok else np.nan,ci_high=float(np.quantile(values,.975)) if ok else np.nan,
        bootstrap_fraction_gt_zero=float(np.mean(values>0)) if ok else np.nan,
        valid_replicates=len(values),invalid_replicates=10000-len(values),requested_replicates=10000,
        groups_used_min=int(n.min()) if len(n) else 0,groups_used_max=int(n.max()) if len(n) else 0,
        cluster='full_inchikey',scope='conditional on fixed protein/family panel; residual cross-ligand family dependence not represented',
        status='validated' if ok else 'insufficient_valid_replicates')


def family_deletion_statistics(frame):
    """Exact delete-one-family sensitivity, NOT a bootstrap confidence interval."""
    families=sorted(frame.family_component_id.unique());idx={k:i for i,k in enumerate(families)}
    total=np.zeros(len(families));counts=np.zeros(len(families),dtype=int);rows=np.zeros(len(families),dtype=int)
    for _,g in frame.groupby(['outer_fold','full_inchikey']):
        p=g[g.binary_label.eq(1)];n=g[g.binary_label.eq(0)]
        if not len(p) or not len(n):continue
        pf=np.array([idx[k] for k in p.family_component_id]);nf=np.array([idx[k] for k in n.family_component_id])
        keep_p=(np.arange(len(families))[:,None]!=pf).astype(float)
        keep_n=(np.arange(len(families))[:,None]!=nf).astype(float)
        win=(p.p_allosteric.to_numpy()[:,None]>n.p_allosteric.to_numpy()).astype(float)
        win+=.5*(p.p_allosteric.to_numpy()[:,None]==n.p_allosteric.to_numpy())
        den=keep_p.sum(axis=1)*keep_n.sum(axis=1);valid=den>0
        num=np.einsum('fi,ij,fj->f',keep_p,win,keep_n,optimize=True)
        total[valid]+=num[valid]/den[valid];counts[valid]+=1
        rows[valid]+=(keep_p.sum(axis=1)+keep_n.sum(axis=1))[valid].astype(int)
    return pd.DataFrame(dict(deleted_family=families,auroc=np.divide(total,counts,out=np.full_like(total,np.nan),where=counts>0),
                             n_groups_used=counts,n_rows_used=rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=32);args=p.parse_args()
    if args.workers<1:raise ValueError('Positive workers required')
    selftest,_=resample_family_statistics([1.,0.],[1.,1.],np.array([[0,0,1]]));assert abs(selftest[0]-2/3)<1e-12
    raw=validate();out=PACKAGE/'local_analysis';out.mkdir(exist_ok=True)
    raw.to_csv(out/'OOF_ALL_SEEDS.tsv.gz',sep='\t',index=False)
    columns=['regime','model','outer_fold','main_row_id','uniprot','family_component_id','full_inchikey','connectivity_key','binary_label','unseen_compound']
    ensemble=raw.groupby(columns,as_index=False).p_allosteric.mean()
    assert len(ensemble)==43680
    ensemble.to_csv(out/'OOF_SEED_ENSEMBLE.tsv.gz',sep='\t',index=False)
    expected=pd.read_csv(PACKAGE/'data/METRIC_SUPPORT.tsv',sep='\t').groupby(['regime','metric'])[['n_rows_used','n_groups_used']].sum()
    rows=[]
    for data,keys,kind in [(raw,['regime','model','seed'],'individual_seed'),(ensemble,['regime','model'],'seed_ensemble')]:
        for key,g in data.groupby(keys):
            ident=dict(zip(keys,key))
            for metric,field in [('pooled',None),('within_ligand','full_inchikey'),('within_protein','uniprot'),('family_macro','family_component_id')]:
                v=stats(g,field)
                if metric.startswith('within_'):
                    ex=expected.loc[(ident['regime'],metric)]
                    if (v['n_rows_used'],v['n_groups_used'])!=(ex.n_rows_used,ex.n_groups_used):raise ValueError('Support mismatch')
                if (metric=='within_ligand' and ident['model']=='ligand') or (metric=='within_protein' and ident['model']=='protein'):
                    if abs(v['auroc']-.5)>1e-12:raise ValueError('Identity control failed')
                rows.append(dict(ident,kind=kind,metric=metric,**v))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'METRICS.tsv',sep='\t',index=False)
    from matrix_overview import build_overview
    build_overview(metrics,out)
    metrics[(metrics.model=='ligand')&(metrics.kind=='seed_ensemble')].to_csv(out/'LIGAND_ONLY_ALL_METRICS.tsv',sep='\t',index=False)
    block_rows=[]
    for (r,m,f,k),g in ensemble.groupby(['regime','model','outer_fold','full_inchikey']):
        if g.binary_label.nunique()==2:block_rows.append(dict(regime=r,model=m,outer_fold=f,full_inchikey=k,rows=len(g),auroc=auc(g.binary_label,g.p_allosteric)))
    blocks=pd.DataFrame(block_rows);blocks.to_csv(out/'WITHIN_LIGAND_BLOCKS.tsv',sep='\t',index=False)
    tasks=[]
    for regime in REGIMES:
        pivot=blocks[blocks.regime.eq(regime)].pivot(index=['outer_fold','full_inchikey'],columns='model',values='auroc')
        if pivot.isna().any().any():raise ValueError('Unpaired model support')
        def add(label,delta):
            sums=delta.groupby(level='full_inchikey').sum().to_numpy();counts=delta.groupby(level='full_inchikey').count().to_numpy()
            tasks.append((regime+'/'+label,sums,counts))
        for model in MODELS:add(model+'/minus_chance',pivot[model]-.5)
        for model in MODELS[2:]:add(model+'/minus_protein',pivot[model]-pivot.protein)
    with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks))) as pool:ci=list(pool.map(cluster_bootstrap,tasks))
    pd.DataFrame(ci).to_csv(out/'WITHIN_LIGAND_CONDITIONAL_BOOTSTRAP.tsv',sep='\t',index=False)
    deletion=[]
    for regime in ('unseen_family','double_unseen'):
        for model in MODELS:
            z=family_deletion_statistics(ensemble[(ensemble.regime==regime)&(ensemble.model==model)])
            z['regime']=regime;z['model']=model;deletion.append(z)
    deletion=pd.concat(deletion,ignore_index=True)
    deletion.to_csv(out/'DELETE_ONE_FAMILY_SENSITIVITY.tsv',sep='\t',index=False)
    delta=deletion.pivot(index=['regime','deleted_family'],columns='model',values='auroc')
    for model in MODELS[2:]:delta[model+'_minus_protein']=delta[model]-delta.protein
    delta.to_csv(out/'DELETE_ONE_FAMILY_PAIRED_LIFT.tsv',sep='\t')
    summary=pd.read_csv(PACKAGE/'gpu_output/FIT_SUMMARY.tsv',sep='\t')
    stability=summary.groupby(['regime','model']).agg(fits=('best_epoch','size'),best_epoch_median=('best_epoch','median'),
        epoch1_fraction=('best_epoch',lambda x:float(x.eq(1).mean())),fallback_fits=('selection_fallback','sum'))
    stability['early_selection_flag']=stability.epoch1_fraction>=1/3
    stability.to_csv(out/'TRAINING_STABILITY.tsv',sep='\t')
    if any(x['status']!='validated' for x in ci):raise ValueError('Bootstrap validity gate failed')
    run=json.loads((PACKAGE/'gpu_output/RUN_CONTRACT.json').read_text())
    write_json(out/'VALIDATION.json',dict(status='validated',fits=480,prediction_rows=131040,ensemble_rows=43680,
        bootstrap_comparisons=len(ci),bootstrap_replicates=10000,bootstrap_cluster='full_inchikey',
        bootstrap_limit='conditional on fixed family/protein panel; family deletion sensitivity is not a CI',
        main_regime='unseen_family',protein_unseen_means='family-held-out',main_lift='joint minus protein-only within ligand',
        no_training_executed=True,no_ATP_ADP_exclusion=True,existing_cohorts_retrained=False,
        run_fingerprint=run['run_fingerprint'],files={x.name:sha(x) for x in out.iterdir() if x.is_file() and x.suffix in ('.tsv','.gz')}))
    print('Validated: 480 fits, four regimes. Pooled ligand-only and conditional primary results both reported.',flush=True)


if __name__=='__main__':main()
