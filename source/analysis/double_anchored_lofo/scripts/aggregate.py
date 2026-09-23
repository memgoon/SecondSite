"""Local CPU analysis only. Family-cluster bootstrap preserves multiplicity."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from common import *
from metrics import auc, stats, resample_family_statistics
from validate_fits import validate


def self_test():
    value,_ = resample_family_statistics([1.,0.],[1.,1.],np.array([[0,0,1]]))
    if abs(value[0]-2./3.)>1e-12:
        raise AssertionError('Cluster multiplicity regression')


def bootstrap(task):
    label,sums,counts,replicates = task
    seed = int(hashlib.sha256(label.encode()).hexdigest()[:8],16)
    draw = np.random.RandomState(seed).randint(0,len(counts),size=(replicates,len(counts)))
    values,groups = resample_family_statistics(sums,counts,draw)
    minimum = int(np.ceil(.95*replicates))
    ok = len(values)>=minimum
    return dict(comparison=label,estimate=float(np.sum(sums)/np.sum(counts)),
                ci_low=float(np.quantile(values,.025)) if ok else np.nan,
                ci_high=float(np.quantile(values,.975)) if ok else np.nan,
                bootstrap_fraction_gt_zero=float(np.mean(values>0)) if ok else np.nan,
                replicates_requested=replicates,valid_replicates=len(values),invalid_replicates=replicates-len(values),
                groups_used_min=int(groups.min()) if len(groups) else 0,
                groups_used_median=float(np.median(groups)) if len(groups) else 0,
                groups_used_max=int(groups.max()) if len(groups) else 0,
                status='validated' if ok else 'insufficient_valid_replicates',
                note='percentile CI conditional on fitted seed-ensemble; not a posterior probability or seed-uncertainty CI')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.workers<1:
        raise ValueError('workers must be positive')
    self_test()
    raw=validate()
    out=PACKAGE/'local_analysis'
    out.mkdir(exist_ok=True)
    raw.to_csv(out/'OOF_ALL_SEEDS.tsv.gz',sep='\t',index=False)
    meta=['regime','model','outer_fold','main_row_id','uniprot','family_component_id','full_inchikey','connectivity_key','binary_label','unseen_compound']
    ensemble=raw.groupby(meta,as_index=False).p_allosteric.mean()
    if len(ensemble)!=6320:
        raise ValueError('Seed ensemble coverage')
    ensemble.to_csv(out/'OOF_SEED_ENSEMBLE.tsv.gz',sep='\t',index=False)
    summaries=[]
    for data, fields, kind in [(raw,['regime','model','seed'],'individual_seed'),(ensemble,['regime','model'],'seed_ensemble')]:
        for keys,g in data.groupby(fields):
            ident=dict(zip(fields,keys))
            for metric,field in [('pooled',None),('within_protein','uniprot'),('family_macro','family_component_id'),('within_ligand','full_inchikey')]:
                result=stats(g,field)
                if metric=='within_protein' and (result['n_rows_used'],result['n_groups_used'])!=(395,98):
                    raise ValueError('Primary support mismatch')
                if metric=='within_ligand' and (result['n_rows_used'],result['n_groups_used'])!=(42,14):
                    raise ValueError('Within-ligand support mismatch')
                if (metric=='within_protein' and ident['model']=='protein') or (metric=='within_ligand' and ident['model']=='ligand'):
                    if abs(result['auroc']-.5)>1e-12:
                        raise ValueError('Identity-constant control failed')
                summaries.append(dict(ident,kind=kind,metric=metric,**result))
    pd.DataFrame(summaries).to_csv(out/'METRICS.tsv',sep='\t',index=False)
    # One protein belongs to exactly one held-out family. Precompute its AUROC
    # and resample family sums/counts: A,A,B must count A twice, not merge it.
    group_rows=[]
    for (regime,model,fold,uid),g in ensemble.groupby(['regime','model','outer_fold','uniprot']):
        group_rows.append(dict(regime=regime,model=model,outer_fold=fold,uniprot=uid,rows=len(g),auroc=auc(g.binary_label,g.p_allosteric)))
    groups=pd.DataFrame(group_rows)
    groups.to_csv(out/'PER_PROTEIN_AUROC.tsv',sep='\t',index=False)
    pivot=groups.pivot_table(index=['outer_fold','uniprot'],columns=['regime','model'],values='auroc')
    tasks=[]
    def add(name,series):
        by=series.groupby(level='outer_fold')
        sums=by.sum().reindex(range(47),fill_value=0).to_numpy()
        counts=by.count().reindex(range(47),fill_value=0).to_numpy()
        if counts.sum()!=98:
            raise ValueError('Paired bootstrap support mismatch')
        tasks.append((name,sums,counts,10000))
    for regime in REGIMES:
        for model in MODELS:
            add('%s/%s/minus_chance'%(regime,model),pivot[(regime,model)]-.5)
        for model in MODELS[2:]:
            add('%s/%s/minus_ligand'%(regime,model),pivot[(regime,model)]-pivot[(regime,'ligand')])
    for model in MODELS:
        add('double_unseen_minus_family_only/'+model,pivot[('double_unseen',model)]-pivot[('family_only',model)])
    if len(tasks)!=36:
        raise ValueError('Unexpected bootstrap comparison count')
    with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks))) as pool:
        results=list(pool.map(bootstrap,tasks))
    pd.DataFrame(results).to_csv(out/'PRIMARY_PAIRED_BOOTSTRAP.tsv',sep='\t',index=False)
    sensitivity=[]
    ligand_rows=[]
    for (regime,model),g in ensemble.groupby(['regime','model']):
        for name,keys in [('without_ATP',[ATP]),('without_ADP',[ADP]),('without_ATP_ADP',[ATP,ADP])]:
            z=g[~g.full_inchikey.isin(keys)]
            sensitivity.append(dict(regime=regime,model=model,subset=name,**stats(z,'uniprot')))
        for key,z in g.groupby('full_inchikey'):
            ligand_rows.append(dict(regime=regime,model=model,full_inchikey=key,rows=len(z),
                allosteric=int(z.binary_label.sum()),mean_score=float(z.p_allosteric.mean()),**stats(z,'full_inchikey')))
    pd.DataFrame(sensitivity).to_csv(out/'ATP_ADP_EXCLUSION_SENSITIVITY.tsv',sep='\t',index=False)
    pd.DataFrame(ligand_rows).to_csv(out/'PER_LIGAND_RESULTS.tsv',sep='\t',index=False)
    if any(r['status']!='validated' for r in results):
        write_json(out/'VALIDATION.json',dict(status='failed_insufficient_valid_replicates'))
        raise ValueError('Bootstrap valid-replicate gate failed')
    run=json.loads((PACKAGE/'gpu_output/RUN_CONTRACT.json').read_text())
    write_json(out/'VALIDATION.json',dict(status='validated',fits=2256,prediction_rows=len(raw),
        run_fingerprint=run['run_fingerprint'],primary_rows=395,primary_proteins=98,family_clusters=47,
        bootstrap_comparisons=36,bootstrap_replicates=10000,multiplicity_self_test='A,A,B = 2/3',
        no_training_executed=True,within_ligand_role='42 rows/14 groups; descriptive only',
        files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.tsv','.gz')}))
    print('Local aggregation and 10,000-replicate bootstrap validated.',flush=True)


if __name__=='__main__':
    main()
