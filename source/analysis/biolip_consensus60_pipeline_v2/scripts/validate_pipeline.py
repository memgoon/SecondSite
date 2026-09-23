"""Read-only numerical validation; does not import producer/grouping/model functions."""
import argparse,json,gzip,math,itertools
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction
import numpy as np
from scipy.stats import gaussian_kde,rankdata
from sklearn.metrics import roc_auc_score,average_precision_score
from common import digest,identity,read

def median(xs):
    xs=sorted(xs);n=len(xs);return xs[n//2] if n%2 else (xs[n//2-1]+xs[n//2])/2
def missing(v):return v is None or str(v).lower() in {'','na','nan','none'}
def close(a,b):
    assert missing(a)==missing(b),(a,b)
    if not missing(a):assert abs(float(a)-float(b))<1e-8,(a,b)
def pdbmedian(rs,k):
    if any(missing(r[k]) for r in rs):return None
    return median([median([float(r[k]) for r in rs if r['pdb_id']==p]) for p in sorted({r['pdb_id'] for r in rs})])
def core(rs):
    by=defaultdict(list)
    for r in rs:by[r['pdb_id']].append(set(map(int,r['binding_uniprot_positions'].split(';'))))
    positions=set.union(*(s for sets in by.values() for s in sets));f={}
    for p in positions:f[p]=sum((Fraction(sum(p in s for s in sets),len(sets)) for sets in by.values()),Fraction())/len(by)
    return sorted(p for p,v in f.items() if v>=Fraction(3,5)),f

def independent_scores(train,test):
    codes=['D','MW','LP','AR'];y=np.array([r['binary_label'] for r in train],int);c={}
    for code in codes:
        x=np.array([r[code] for r in train]);z=np.array([r[code] for r in test]);cats=sorted(set(x.astype(int)))
        if code!='AR':
            edges=np.unique(np.quantile(x,np.linspace(0,1,int(math.ceil(len(x)**(1/3)))+1)[1:-1]))
            tb=np.searchsorted(edges,x,side='right');zb=np.searchsorted(edges,z,side='right');n=len(edges)+1
        else:
            lookup={v:i for i,v in enumerate(cats)};tb=np.array([lookup[int(v)] for v in x]);zb=np.array([lookup.get(int(v),len(cats)) for v in z]);n=len(cats)+1
        neg=np.bincount(tb[y==0],minlength=n)+.5;pos=np.bincount(tb[y==1],minlength=n)+.5
        logs=np.log((pos/(sum(y==1)+.5*n))/(neg/(sum(y==0)+.5*n)));q=logs[zb]
        if code=='AR':k=q
        else:
            zz=np.clip(z,min(x),max(x));k=np.log((gaussian_kde(x[y==1],bw_method='scott')(zz)+1e-12)/(gaussian_kde(x[y==0],bw_method='scott')(zz)+1e-12))
        c['QNB',code]=np.round(q,12)
        for e in ['KDE_LEGACY','KDE100']:
            cap=math.log(100 if e=='KDE100' or code=='D' else 10);c[e,code]=np.round(np.clip(k,-cap,cap),12)
    result={'RAW:RAW_D':np.round([r['D'] for r in test],12)}
    for e in ['QNB','KDE_LEGACY','KDE100']:
        for n in range(1,5):
            for s in itertools.combinations(codes,n):result[e+':'+('+'.join(s))]=np.round(sum(c[e,k] for k in s),12)
    return result

def task_values(out,stage):
    proof=json.loads((out/'stages'/(stage+'.json')).read_text());result=[]
    for name,h in sorted(proof['commit_hashes'].items()):
        p=out/'tasks'/stage/name;assert digest(p)==h;commit=json.loads(p.read_text());data=p.with_name(name.replace('.commit.json','.json.gz'))
        assert digest(data)==commit['output'];result.append(json.loads(gzip.decompress(data.read_bytes())))
    return result

def check(out,final=False):
    lock=json.loads((out/'INPUT_LOCK.json').read_text())
    for x in lock['inputs'].values():assert digest(x['path'])==x['sha256']
    for p,h in lock['runtime_hashes'].items():assert digest(p)==h
    refs=read(lock['inputs']['reference']['path']);cands=read(lock['inputs']['candidates']['path']);groups={}
    folds={r['observation_id']:r for r in read(lock['inputs']['folds']['path'])}
    for source,rs in [('reference',refs),('candidate',cands)]:
        g=defaultdict(list)
        for r in rs:g['PAIR_'+identity([r['uniprot'],r['full_inchikey']])[:20]].append(r)
        groups.update({(source,pid):rows for pid,rows in g.items()})
    tables={s:read(out/'data'/name) for s,name in [('reference','REFERENCE_PAIRS.tsv'),('candidate','CANDIDATE_PAIRS.tsv')]}
    normalized={};frequencies=defaultdict(dict)
    for r in read(out/'data/RESIDUE_FREQUENCIES.tsv.gz'):
        k=(r['source'],r['pair_id']);p=int(r['residue']);assert p not in frequencies[k];frequencies[k][p]=r
    for source,ps in tables.items():
        assert len(ps)==len({p['pair_id'] for p in ps})
        assert {p['pair_id'] for p in ps}=={pid for s,pid in groups if s==source}
        normalized[source]=[]
        for p in ps:
            rs=groups[source,p['pair_id']];co,f=core(rs)
            assert p['consensus60']==(';'.join(map(str,co)) or '')
            assert int(p['observations'])==len(rs) and int(p['pdbs'])==len({r['pdb_id'] for r in rs})
            assert p['observation_ids']==';'.join(sorted(r['observation_id'] for r in rs))
            assert set(frequencies[source,p['pair_id']])==set(f)
            for pos,v in f.items():
                assert Fraction(frequencies[source,p['pair_id']][pos]['fraction'])==v
                close(frequencies[source,p['pair_id']][pos]['frequency'],float(v))
            fields={'D':'ligand_centroid_to_orthosteric_site_CA_centroid_A' if source=='reference' else 'ligand_centroid_to_nearest_orthosteric_site_CA_centroid_A','MW':'molecular_weight','LP':'clogp','AR':'aromatic_ring_count'}
            q=dict(p)
            for code,col in fields.items():close(p[code],pdbmedian(rs,col));q[code]=None if missing(p[code]) else float(p[code])
            assert p['feature_complete']==str(all(q[k] is not None for k in fields))
            if source=='reference':
                assert {r['binary_label'] for r in rs}=={p['binary_label']}
                for k in ['protein_fold','family_fold','family_component_id']:assert {folds[r['observation_id']][k] for r in rs}=={p[k]}
                for k in ['protein_fold','family_fold','binary_label']:q[k]=int(p[k])
            normalized[source].append(q)
    lineage=read(out/'data/OBSERVATION_LINEAGE.tsv.gz');assert len(lineage)==len(refs)+len(cands)
    assert len({(r['source'],r['observation_id']) for r in lineage})==len(lineage)
    assert {(r['source'],r['observation_id'],r['pair_id']) for r in lineage}=={(s,r['observation_id'],pid) for (s,pid),rs in groups.items() for r in rs}
    rp=normalized['reference'];cp=normalized['candidate'];fits=task_values(out,'04_refit_46_rankings');assert len(fits)==11
    independently_predicted={};model_checks=0
    for fit in fits:
        regime,fold=fit['regime'],fit['fold']
        if regime=='deployment':train=rp;test=[r for r in cp if r['feature_complete']=='True']
        else:
            k='family_fold' if regime=='family_held_out' else 'protein_fold';train=[r for r in rp if r[k]!=fold];test=[r for r in rp if r[k]==fold]
            assert not set(r['uniprot'] for r in train)&set(r['uniprot'] for r in test)
            if regime=='family_held_out':assert not set(r['family_component_id'] for r in train)&set(r['family_component_id'] for r in test)
        train=sorted(train,key=lambda r:r['pair_id']);test=sorted(test,key=lambda r:r['pair_id'])
        assert fit['train_pair_ids']==sorted(r['pair_id'] for r in train) and fit['test_pair_ids']==sorted(r['pair_id'] for r in test)
        pred=independent_scores(train,test);lookup={p['pair_id']:p for p in fit['predictions']};assert len(lookup)==len(test)
        for i,r in enumerate(test):
            assert set(lookup[r['pair_id']]['scores'])==set(pred)
            for rid,values in pred.items():
                close(lookup[r['pair_id']]['scores'][rid],values[i]);independently_predicted[regime,r['pair_id'],rid]=float(values[i]);model_checks+=1
    # All OOF rows and all 92 macro bootstrap intervals are re-evaluated, not sampled.
    oof=read(out/'data/OOF_PAIR_SCORES.tsv.gz');assert len(oof)==len(rp)*2*46
    seen=set()
    for r in oof:
        k=(r['regime'],r['pair_id'],r['ranking_id']);assert k not in seen;seen.add(k);close(r['score'],independently_predicted[k])
    metricrows=read(out/'data/HELDOUT_PAIR_METRICS_AND_CI.tsv');assert len(metricrows)==92
    for m in metricrows:
        regime,rid=m['regime'],m['ranking_id'];per=[]
        for protein in sorted({r['uniprot'] for r in rp}):
            rs=[r for r in rp if r['uniprot']==protein];y=[r['binary_label'] for r in rs];s=[independently_predicted[regime,r['pair_id'],rid] for r in rs]
            if len(set(y))<2:continue
            ranks=rankdata(-np.asarray(s));first=min(rank for rank,label in zip(ranks,y) if label==1)
            raw=[independently_predicted[regime,r['pair_id'],'RAW:RAW_D'] for r in rs];rawfirst=min(rank for rank,label in zip(rankdata(-np.asarray(raw)),y) if label==1)
            per.append(dict(uniprot=protein,family=rs[0]['family_component_id'],auc=roc_auc_score(y,s),ap=average_precision_score(y,s),recall1=float(first<=1),mrr=1/first,n=len(rs),delta_auc_vs_raw=roc_auc_score(y,s)-roc_auc_score(y,raw),delta_mrr_vs_raw=1/first-1/rawfirst))
        assert int(m['both_label_proteins'])==len(per) and int(m['both_label_pairs'])==sum(r['n'] for r in per)
        close(m['pooled_auc'],roc_auc_score([r['binary_label'] for r in rp],[independently_predicted[regime,r['pair_id'],rid] for r in rp]))
        field='family' if regime=='family_held_out' else 'uniprot';keys=sorted({r[field] for r in per});nrep=int(m['replicates'])
        assert nrep==(lock['test_replicates'] if lock.get('synthetic') else 10000)
        draws=np.random.default_rng(20260902 if field=='family' else 20260901).integers(0,len(keys),size=(nrep,len(keys)),dtype=np.int32)
        assert identity(draws.tolist())==m['draws_hash']
        for metric in ['auc','ap','recall1','mrr','delta_auc_vs_raw','delta_mrr_vs_raw']:
            sums=np.array([sum(r[metric] for r in per if r[field]==k) for k in keys]);ns=np.array([sum(r[field]==k for r in per) for k in keys])
            values=np.take(sums,draws).sum(1)/np.take(ns,draws).sum(1)
            close(m[metric+'_value'],np.mean([r[metric] for r in per]));close(m[metric+'_low'],np.quantile(values,.025));close(m[metric+'_high'],np.quantile(values,.975))
    rankings=read(out/'data/CANDIDATE_RANKINGS.tsv.gz');assert len(rankings)==len(cp)*46
    pm={r['pair_id']:r for r in cp};counts={p:[0,0] for p in pm};indexed={(r['pair_id'],r['ranking_id']):r for r in rankings};assert len(indexed)==len(rankings)
    for rid in sorted({r['ranking_id'] for r in rankings}):
        good=[r for r in cp if r['feature_complete']=='True'];vals=[independently_predicted['deployment',r['pair_id'],rid] for r in good];gr=rankdata(-np.array(vals));gmap=dict(zip([r['pair_id'] for r in good],gr));local={};sizes={}
        for u in {r['uniprot'] for r in good}:
            subset=[r for r in good if r['uniprot']==u];lr=rankdata(-np.array([independently_predicted['deployment',r['pair_id'],rid] for r in subset]))
            local.update(zip([r['pair_id'] for r in subset],lr));sizes.update({r['pair_id']:len(subset) for r in subset})
        for pid,p in pm.items():
            r=indexed[pid,rid]
            if p['feature_complete']!='True':assert r['ranking_score']=='NA';continue
            close(r['ranking_score'],independently_predicted['deployment',pid,rid]);close(r['global_average_rank'],gmap[pid]);close(r['within_protein_average_rank'],local[pid])
            gp=(gmap[pid]-.5)/len(good);lp=(local[pid]-.5)/sizes[pid];close(r['global_percentile'],gp);close(r['within_protein_percentile'],lp)
            if rid in ['RAW:RAW_D','KDE_LEGACY:D+AR','KDE_LEGACY:D+MW','QNB:D+AR']:counts[pid][0]+=gp<=.1;counts[pid][1]+=lp<=.1
    flags=defaultdict(list)
    for r in read(lock['inputs']['audit']['path']):flags[r['candidate_pair_id']].append(r)
    for p in tables['candidate']:
        rs=groups['candidate',p['pair_id']];old=flags[p['old_pair_id']]
        chemistry=all(r['passes_substantial_organic_compound']=='True' for r in old);role=all(r['passes_known_role_novelty']=='True' for r in old)
        heavy=pdbmedian(rs,'ligand_to_nearest_orthosteric_site_min_heavy_A');close(p['min_heavy_distance'],heavy)
        no_overlap=all(float(r['candidate_nearest_orthosteric_site_residue_overlap_count'])==0 for r in rs)
        spatial=float(p['D'])>=10 and heavy is not None and heavy>=6 and no_overlap
        review=spatial and chemistry and role;agree=review and counts[p['pair_id']][0]>=2 and counts[p['pair_id']][1]>=3
        pdbs={r['pdb_id'] for r in rs};fraction=sum(sum(float(r['ligand_centroid_to_nearest_orthosteric_site_CA_centroid_A'])>=10 for r in rs if r['pdb_id']==pdb)/sum(r['pdb_id']==pdb for r in rs) for pdb in pdbs)/len(pdbs)
        close(p['fraction_ge10'],fraction);repeat=agree and len(pdbs)>=2 and fraction>=.75
        assert [p[k] for k in ['chemistry_pass','known_role_pass','distance_screen','review_filter','ranking_agreement','repeated_structure']]==list(map(str,[chemistry,role,spatial,review,agree,repeat]))
        assert p['final_manual_disposition']=='not_recurated'
    short=read(out/'data/PROVISIONAL_PAIR_SHORTLIST.tsv');assert short==[r for r in tables['candidate'] if r['repeated_structure']=='True']
    for name in ['PAIR_RANKINGS.tsv.gz','PAIR_RECORDS.tsv','NESTED_LOCATIONS.tsv']:
        original={'PAIR_RANKINGS.tsv.gz':'CANDIDATE_RANKINGS.tsv.gz','PAIR_RECORDS.tsv':'CANDIDATE_PAIRS.tsv','NESTED_LOCATIONS.tsv':'NESTED_LOCATION_CONSENSUS.tsv'}[name]
        assert digest(out/'secondsite_handoff'/name)==digest(out/'data'/original)
    if not lock.get('synthetic'):
        # Check every spatial comparison classification and location consensus, without trusting a PASS label.
        for r in read(out/'data/SPATIAL_COMPARISONS.tsv.gz'):
            if r['state'] in ['same_location_geometry','distinct_location_geometry']:
                assert int(r['common_CA'])>=20 and float(r['aligned_CA_RMSD'])<=2.5
                if r['state']=='same_location_geometry':assert float(r['ligand_centroid_distance'])<=4 and float(r['ligand_min_heavy_distance'])<=2
                else:assert float(r['ligand_centroid_distance'])>=10 and float(r['ligand_min_heavy_distance'])>=4
        for r in read(out/'data/NESTED_LOCATION_CONSENSUS.tsv'):
            ids=set(r['observation_ids'].split(';'));rs=[x for x in groups['candidate',r['pair_id']] if x['observation_id'] in ids];assert len(rs)==len(ids)
            c,_=core(rs);assert r['consensus60']==';'.join(map(str,c))
        for r in read(lock['inputs']['coordinates']['path']):
            if r['pdb_id'] in {x['pdb_id'] for x in cands}:assert digest(r['coordinate_path'])==r['sha256']
    summary=json.loads((out/'SUMMARY.json').read_text());assert summary['reference_pairs']==len(rp) and summary['candidate_pairs']==len(cp)
    assert summary['candidate_rankings']==len(rankings) and summary['provisional_shortlist']==len(short)
    if final:
        for line in (out/'FINAL_CHECKSUMS.sha256').read_text().splitlines():
            h,p=line.split('  ',1);assert digest(out/p)==h,p
    return dict(status='PASS_COMPUTATIONAL_VALIDATION',reference_pairs=len(rp),candidate_pairs=len(cp),model_scores_independently_refit_checked=model_checks,
        evaluated_methods=92,bootstrap_CIs_recomputed=True,all_candidate_ranks_and_filters_checked=True,
        spatial_numeric_decisions_checked=True,independent_coordinate_mapping_not_reimplemented=True,manual_scientific_validation=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--verify-final',action='store_true');a=p.parse_args()
    print(json.dumps(check(Path(a.run_dir),a.verify_final),sort_keys=True))
