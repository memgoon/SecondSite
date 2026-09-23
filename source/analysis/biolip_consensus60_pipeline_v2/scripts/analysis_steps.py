import importlib.util,itertools,math,statistics
from collections import defaultdict,Counter
from fractions import Fraction
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score
from common import OLD,identity,val

spec=importlib.util.spec_from_file_location('frozen_estimators',OLD/'scripts/08_evaluate_heldout_rankings.py')
EST=importlib.util.module_from_spec(spec);spec.loader.exec_module(EST)
CODES=['D','MW','LP','AR']
METHODS=[('RAW:RAW_D',['D'])]+[(e+':'+('+'.join(s)),list(s)) for e in ['QNB','KDE_LEGACY','KDE100'] for n in range(1,5) for s in itertools.combinations(CODES,n)]
assert len({rid for rid,_ in METHODS})==46
VIEWS=['RAW:RAW_D','KDE_LEGACY:D+AR','KDE_LEGACY:D+MW','QNB:D+AR']
REF_D='ligand_centroid_to_orthosteric_site_CA_centroid_A'
CAND_D='ligand_centroid_to_nearest_orthosteric_site_CA_centroid_A'

def med(rs,k):
    if any(val(r.get(k)) is None for r in rs):return None
    pdbs=sorted({r['pdb_id'] for r in rs})
    return statistics.median(statistics.median(float(r[k]) for r in rs if r['pdb_id']==p) for p in pdbs)

def contacts(rs):
    by=defaultdict(list)
    for r in rs:by[r['pdb_id']].append(set(map(int,r['binding_uniprot_positions'].split(';'))))
    f=defaultdict(Fraction);support=Counter()
    for sets in by.values():
        c=Counter(x for s in sets for x in s)
        for x,n in c.items():f[x]+=Fraction(n,len(sets)*len(by));support[x]+=1
    return sorted(x for x,v in f.items() if v>=Fraction(3,5)),[
        dict(residue=x,frequency=float(v),fraction=str(v),pdb_support=support[x]) for x,v in sorted(f.items())]

def rebuild(task):
    source,group=task['source'],task['rows'];first=group[0];pid='PAIR_'+identity([first['uniprot'],first['full_inchikey']])[:20]
    if len({(r['uniprot'],r['full_inchikey']) for r in group})!=1:raise ValueError('pair identity')
    core,freq=contacts(group)
    fields={'D':REF_D if source=='reference' else CAND_D,'MW':'molecular_weight','LP':'clogp','AR':'aromatic_ring_count'}
    row=dict(pair_id=pid,uniprot=first['uniprot'],full_inchikey=first['full_inchikey'],source=source,
        observations=len(group),pdbs=len({r['pdb_id'] for r in group}),consensus60=';'.join(map(str,core)),
        observation_ids=';'.join(sorted(r['observation_id'] for r in group)),**{k:med(group,v) for k,v in fields.items()})
    if source=='reference':
        for k in ['binary_label','family_component_id','protein_fold','family_fold']:
            v={r[k] for r in group}
            if len(v)!=1:raise ValueError('reference pair crosses label/fold identity: '+pid+'/'+k)
            row[k]=int(next(iter(v))) if k!='family_component_id' else next(iter(v))
    else:
        if len({r['candidate_pair_id'] for r in group})!=1:raise ValueError('old pair ID mismatch')
        row['old_pair_id']=first['candidate_pair_id']
        row['min_heavy_distance']=med(group,'ligand_to_nearest_orthosteric_site_min_heavy_A')
        row['no_residue_overlap']=all(val(r['candidate_nearest_orthosteric_site_residue_overlap_count'])==0 for r in group)
        pdbs={r['pdb_id'] for r in group}
        row['fraction_ge10']=sum(sum(float(r[CAND_D])>=10 for r in group if r['pdb_id']==p)/sum(r['pdb_id']==p for r in group) for p in pdbs)/len(pdbs)
    row['feature_complete']=all(row[k] is not None for k in CODES)
    return dict(pair=row,frequencies=freq)

def rebuild_batch(batch):return [rebuild(task) for task in batch]

def fit_score(task):
    train,test,regime,fold=task['train'],task['test'],task['regime'],task['fold']
    if set(r['pair_id'] for r in train)&set(r['pair_id'] for r in test) and regime!='deployment':raise ValueError('pair leakage')
    if regime!='deployment' and set(r['uniprot'] for r in train)&set(r['uniprot'] for r in test):raise ValueError('protein leakage')
    if regime=='family_held_out' and set(r['family_component_id'] for r in train)&set(r['family_component_id'] for r in test):raise ValueError('family leakage')
    y=np.array([r['binary_label'] for r in train],dtype=int)
    if set(y)!={0,1}:raise ValueError('one-class training fold')
    components={};models=[]
    for code in CODES:
        x=np.array([r[code] for r in train],float);z=np.array([r[code] for r in test],float)
        if not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('incomplete fit/test features')
        q=EST.fit_qnb(x,y,code);k=EST.fit_kde_component(x,y,code)
        qscore,_=EST.transform_qnb(q,z,code);kscore,_=EST.transform_kde(k,z,code)
        components['QNB',code]=np.round(qscore,12)
        for e in ['KDE_LEGACY','KDE100']:components[e,code]=np.round(np.clip(kscore,-EST.CAPS[e][code],EST.CAPS[e][code]),12)
        models.append(dict(feature=code,train_min=float(min(x)),train_max=float(max(x)),
            qnb_edges=[float(v) for v in q['internal_edges']],qnb_categories=q['categories'],
            qnb_log_lr=q['bin_log_lr'].tolist(),kde_negative_factor=None if code=='AR' else k['negative_kde_factor'],
            kde_positive_factor=None if code=='AR' else k['positive_kde_factor']))
    predictions=[]
    for i,r in enumerate(test):
        scores={rid:(round(r['D'],12) if rid=='RAW:RAW_D' else round(float(sum(components[rid.split(':')[0],c][i] for c in cs)),12)) for rid,cs in METHODS}
        predictions.append(dict(pair_id=r['pair_id'],regime=regime,fold=fold,scores=scores))
    return dict(regime=regime,fold=fold,train_pair_ids=sorted(r['pair_id'] for r in train),
        test_pair_ids=sorted(r['pair_id'] for r in test),train_hash=identity(train),models=models,predictions=predictions)

def evaluation(task):
    rs=task['rows'];rid=task['ranking_id'];regime=task['regime'];per=[]
    for u in sorted({r['uniprot'] for r in rs}):
        g=[r for r in rs if r['uniprot']==u];y=np.array([r['binary_label'] for r in g]);s=np.array([r['score'] for r in g])
        if len(set(y))<2:continue
        rank=rankdata(-s,method='average');first=float(min(rank[y==1]))
        raw=np.array([r['raw_score'] for r in g]);rawfirst=float(min(rankdata(-raw,method='average')[y==1]))
        per.append(dict(uniprot=u,family_component_id=g[0]['family_component_id'],n_pairs=len(g),
            auc=float(roc_auc_score(y,s)),ap=float(average_precision_score(y,s)),recall1=float(first<=1),mrr=1/first,
            delta_auc_vs_raw=float(roc_auc_score(y,s)-roc_auc_score(y,raw)),delta_mrr_vs_raw=1/first-1/rawfirst))
    if not per:raise ValueError('no both-label proteins in evaluation')
    cluster='family_component_id' if regime=='family_held_out' else 'uniprot'
    keys=sorted({r[cluster] for r in per});rng=np.random.default_rng(20260902 if regime=='family_held_out' else 20260901)
    draws=rng.integers(0,len(keys),size=(task['replicates'],len(keys)),dtype=np.int32)
    boot={}
    for m in ['auc','ap','recall1','mrr','delta_auc_vs_raw','delta_mrr_vs_raw']:
        sums=np.array([sum(r[m] for r in per if r[cluster]==k) for k in keys]);ns=np.array([sum(r[cluster]==k for r in per) for k in keys])
        samples=sums[draws].sum(axis=1)/ns[draws].sum(axis=1)
        boot[m]=dict(value=float(np.mean([r[m] for r in per])),low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)))
    return dict(regime=regime,ranking_id=rid,pairs=len(rs),both_label_proteins=len(per),both_label_pairs=sum(r['n_pairs'] for r in per),
        pooled_auc=float(roc_auc_score([r['binary_label'] for r in rs],[r['score'] for r in rs])),
        per_protein=per,bootstrap=boot,replicates=task['replicates'],cluster_unit=cluster,draws_hash=identity(draws.tolist()))
