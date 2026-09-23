#!/usr/bin/env python3
import argparse,fcntl,json,os,sys,uuid,subprocess
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from common import PKG,OLD,digest,identity,save,atomic,table,read,tasks,versions
from analysis_steps import rebuild_batch,fit_score,evaluation,METHODS,VIEWS,CODES
from spatial import extract_batch,audit_batch

def batches(seq,n):return [seq[i:i+n] for i in range(0,len(seq),n)]
def flatten(xs):return [r for batch in xs for r in batch]

def inputs(lock):
    if lock.get('library_versions',versions())!=versions():raise ValueError('scientific library versions changed')
    for item in lock['inputs'].values():
        if digest(item['path'])!=item['sha256']:raise ValueError('input checksum failure: '+item['path'])
    for name,h in lock['runtime_hashes'].items():
        if digest(name)!=h:raise ValueError('runtime dependency checksum failure: '+name)
    refs=read(lock['inputs']['reference']['path']);folds={r['observation_id']:r for r in read(lock['inputs']['folds']['path'])}
    if len(folds)!=len(refs):raise ValueError('fold/reference observation counts')
    for r in refs:
        f=folds[r['observation_id']]
        if r['uniprot']!=f['uniprot'] or r['binary_label']!=f['binary_label']:raise ValueError('fold identity')
        r.update({k:f[k] for k in ['family_component_id','protein_fold','family_fold']})
    candidates=read(lock['inputs']['candidates']['path'])
    if [len(refs),len(candidates)]!=lock['expected_rows']:raise ValueError('frozen membership changed')
    for rs in [refs,candidates]:
        if len({r['observation_id'] for r in rs})!=len(rs):raise ValueError('duplicate observation')
    jobs=[];sourcegroups={}
    for source,rs in [('reference',refs),('candidate',candidates)]:
        grouped=defaultdict(list)
        for r in rs:grouped[r['uniprot'],r['full_inchikey']].append(r)
        for key,g in sorted(grouped.items()):
            g=sorted(g,key=lambda r:r['observation_id']);pid='PAIR_'+identity(list(key))[:20]
            sourcegroups[source,pid]=g;jobs.append(dict(source=source,rows=g))
    return refs,candidates,jobs,sourcegroups

def export(out,reference,candidates,fits,metrics,spatial,groups,lock):
    data=out/'data';frames={r['pair_id']:r for r in reference+candidates}
    cframes={r['pair_id']:dict(r) for r in candidates};pred={p['pair_id']:p['scores'] for p in fits[-1]['predictions']}
    olds={(r['candidate_pair_id'],r['ranking_id']):r for r in read(lock['inputs']['old_pairs']['path'])}
    flags=defaultdict(list)
    for r in read(lock['inputs']['audit']['path']):flags[r['candidate_pair_id']].append(r)
    spatialmap={r['pair_id']:r for r in spatial};rankrows=[];viewcounts={p:[0,0] for p in cframes}
    for rid,_ in METHODS:
        good=[pid for pid in sorted(cframes) if pid in pred]
        gr=rankdata([-pred[p][rid] for p in good],method='average');globalrank=dict(zip(good,gr));local={};localn={}
        for protein in sorted({r['uniprot'] for r in candidates}):
            ids=[p for p in good if cframes[p]['uniprot']==protein]
            local.update(zip(ids,rankdata([-pred[p][rid] for p in ids],method='average')));localn.update({p:len(ids) for p in ids})
        for pid,r in sorted(cframes.items()):
            gp=(globalrank[pid]-.5)/len(good) if pid in pred else None;lp=(local[pid]-.5)/localn[pid] if pid in pred else None
            old=olds[r['old_pair_id'],rid]
            rankrows.append(dict(pair_id=pid,uniprot=r['uniprot'],full_inchikey=r['full_inchikey'],ranking_id=rid,
                ranking_score=pred[pid][rid] if pid in pred else None,global_average_rank=float(globalrank[pid]) if pid in pred else None,
                within_protein_average_rank=float(local[pid]) if pid in pred else None,global_percentile=gp,within_protein_percentile=lp,
                old_best_site_score=old['best_site_ranking_score'],old_pair_global_rank=old['pair_global_rank'],
                score_status='scored' if pid in pred else 'incomplete_features',interpretation='relative_rank_not_probability'))
            if rid in VIEWS and gp is not None:viewcounts[pid][0]+=int(gp<=.1);viewcounts[pid][1]+=int(lp<=.1)
    for pid,r in cframes.items():
        audit=flags[r['old_pair_id']]
        r['chemistry_pass']=all(x['passes_substantial_organic_compound']=='True' for x in audit)
        r['known_role_pass']=all(x['passes_known_role_novelty']=='True' for x in audit)
        r['legacy_flag_disagreement']=any(len({x[k] for x in audit})>1 for k in ['passes_substantial_organic_compound','passes_known_role_novelty'])
        r['spatial_status']=spatialmap[pid]['status'];r['location_count']=len(spatialmap[pid]['locations'])
        r['distance_screen']=r['D'] is not None and r['D']>=10 and r['min_heavy_distance'] is not None and r['min_heavy_distance']>=6 and r['no_residue_overlap']
        r['review_filter']=r['distance_screen'] and r['chemistry_pass'] and r['known_role_pass']
        r['global_top10_views'],r['within_protein_top10_views']=viewcounts[pid]
        r['ranking_agreement']=r['review_filter'] and viewcounts[pid][0]>=2 and viewcounts[pid][1]>=3
        r['repeated_structure']=r['ranking_agreement'] and r['pdbs']>=2 and r['fraction_ge10']>=.75
        r['site_consistent_shortlist']=r['repeated_structure'] and r['spatial_status']=='geometry_consistent_single_location'
        r['review_required']=True;r['final_manual_disposition']='not_recurated'
    candidates=[cframes[k] for k in sorted(cframes)]
    table(data/'REFERENCE_PAIRS.tsv',reference);table(data/'CANDIDATE_PAIRS.tsv',candidates)
    table(data/'CANDIDATE_RANKINGS.tsv.gz',rankrows)
    table(data/'PROVISIONAL_PAIR_SHORTLIST.tsv',[r for r in candidates if r['repeated_structure']],sorted(candidates[0]))
    table(data/'SPATIAL_REVIEW_QUEUE.tsv',[r for r in candidates if r['spatial_status']!='geometry_consistent_single_location'],sorted(candidates[0]))
    obs=[];freq=[];loc=[];comparisons=[];consensusgeometry=[]
    for (source,pid),rs in sorted(groups.items()):
        for r in rs:obs.append(dict(source=source,pair_id=pid,**{k:r[k] for k in ['observation_id','pdb_id','receptor_chain','ligand_ccd','ligand_chain','ligand_auth_seq_id','binding_uniprot_positions']}))
    table(data/'OBSERVATION_LINEAGE.tsv.gz',obs)
    for s in spatial:
        for x in s['locations']:
            loc.append(dict(pair_id=s['pair_id'],location_id=x['location_id'],consensus60=x['consensus60'],pdbs=x['pdbs'],observation_ids=';'.join(x['observation_ids']),spatial_status=s['status']))
            for g in x['geometry']:consensusgeometry.append(dict(pair_id=s['pair_id'],location_id=x['location_id'],**g))
        for x in s['comparisons']:comparisons.append(dict(pair_id=s['pair_id'],**x))
    table(data/'NESTED_LOCATION_CONSENSUS.tsv',loc,['pair_id','location_id','consensus60','pdbs','observation_ids','spatial_status'])
    table(data/'SPATIAL_COMPARISONS.tsv.gz',comparisons,['pair_id','first','second','state','common_CA','aligned_CA_RMSD','ligand_centroid_distance','ligand_min_heavy_distance'])
    table(data/'CONSENSUS_GEOMETRY.tsv.gz',consensusgeometry,['pair_id','location_id','observation_id','consensus_CA_recovery','ligand_to_own_consensus_CA_centroid'])
    mr=[];per=[]
    for m in metrics:
        row={k:m[k] for k in ['regime','ranking_id','pairs','both_label_proteins','both_label_pairs','pooled_auc','replicates','cluster_unit','draws_hash']}
        for metric,v in m['bootstrap'].items():
            for k,value in v.items():row[metric+'_'+k]=value
        mr.append(row)
        per.extend(dict(regime=m['regime'],ranking_id=m['ranking_id'],**p) for p in m['per_protein'])
    table(data/'HELDOUT_PAIR_METRICS_AND_CI.tsv',mr);table(data/'HELDOUT_PER_PROTEIN_METRICS.tsv',per)
    oof=[]
    for fit in fits[:-1]:
        for p in fit['predictions']:
            r=frames[p['pair_id']]
            for rid,score in p['scores'].items():oof.append(dict(regime=p['regime'],fold=p['fold'],pair_id=p['pair_id'],uniprot=r['uniprot'],binary_label=r['binary_label'],family_component_id=r['family_component_id'],ranking_id=rid,score=score))
    table(data/'OOF_PAIR_SCORES.tsv.gz',oof)
    funnel=[]
    for key in ['all','distance_screen','review_filter','ranking_agreement','repeated_structure','site_consistent_shortlist']:
        selected=candidates if key=='all' else [r for r in candidates if r[key]]
        funnel.append(dict(step=key,pairs=len(selected),proteins=len({r['uniprot'] for r in selected})))
    table(data/'FILTER_COUNTS.tsv',funnel)
    # Explicit handoff files, not a database migration or a claimed live deployment.
    handoff=out/'secondsite_handoff';table(handoff/'PAIR_RANKINGS.tsv.gz',rankrows);table(handoff/'PAIR_RECORDS.tsv',candidates)
    table(handoff/'NESTED_LOCATIONS.tsv',loc,['pair_id','location_id','consensus60','pdbs','observation_ids','spatial_status'])
    save(handoff/'MANIFEST.json',dict(live_deployment=False,ranking_unit='exact_protein_full_InChIKey_pair',
        location_records_are_nested=True,manual_dispositions_transferred=False,publication_requires_review=True))
    return dict(reference_pairs=len(reference),candidate_pairs=len(candidates),candidate_rankings=len(rankrows),
        reference_observations=sum(r['observations'] for r in reference),candidate_observations=sum(r['observations'] for r in candidates),
        candidate_incomplete_pairs=sum(not r['feature_complete'] for r in candidates),fit_jobs=len(fits),evaluation_methods=len(metrics),
        provisional_shortlist=sum(r['repeated_structure'] for r in candidates),spatial_status_counts=dict(Counter(s['status'] for s in spatial)),
        reference_unit_changed=True,old_CP8_metrics_not_reused=True,live_web_modified=False,manual_curation_complete=False)

def run(args):
    lock=json.loads(Path(args.lock).read_text());refs,cands,jobs,groups=inputs(lock)
    if args.preflight_only:
        print(json.dumps(dict(status='PASS_INPUTS',reference_observations=len(refs),candidate_observations=len(cands),workers=args.workers,io_workers=args.io_workers)));return
    if args.test_fail_after and not lock.get('synthetic'):raise ValueError('test interruption forbidden on production')
    out=Path(args.output).resolve()
    if PKG/'runs' not in out.parents and not lock.get('synthetic'):raise ValueError('use a new v2 runs/ directory')
    out.mkdir(parents=True,exist_ok=True)
    with open(out/'RUN.lock','a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime=dict(python=sys.version,libraries=versions())
        fp=dict(lock_hash=digest(args.lock),runtime=runtime,bootstrap_replicates=10000 if not lock.get('synthetic') else lock['test_replicates'])
        if (out/'RUN_FINGERPRINT.json').exists() and json.loads((out/'RUN_FINGERPRINT.json').read_text())!=fp:raise ValueError('changed run fingerprint')
        if (out/'FINAL_CHECKSUMS.sha256').exists():
            for line in (out/'FINAL_CHECKSUMS.sha256').read_text().splitlines():
                h,n=line.split('  ',1)
                if digest(out/n)!=h:raise ValueError('completed output corruption: '+n)
            # A canonical completed run is read-only on resume; do not redraw/re-export it.
            subprocess.run([sys.executable,str(PKG/'scripts/validate_pipeline.py'),'--run-dir',str(out),'--verify-final'],check=True)
            print('SKIP validated completed pipeline',flush=True)
            return
        save(out/'RUN_FINGERPRINT.json',fp);save(out/'INPUT_LOCK.json',lock)
        try:
            rebuilt=flatten(tasks(out,'01_pair_reconstruction',rebuild_batch,batches(jobs,50),args.workers,args.test_fail_after))
            rp=sorted([r['pair'] for r in rebuilt if r['pair']['source']=='reference'],key=lambda r:r['pair_id'])
            cp=sorted([r['pair'] for r in rebuilt if r['pair']['source']=='candidate'],key=lambda r:r['pair_id'])
            if any(not r['feature_complete'] for r in rp):raise ValueError('incomplete reference feature row; no silent deletion')
            if lock.get('synthetic'):
                spatial=[dict(pair_id=r['pair_id'],status='unresolved',locations=[],comparisons=[]) for r in cp]
            else:
                coordinate={r['pdb_id']:r for r in read(lock['inputs']['coordinates']['path'])}
                sequences={r['uniprot']:r['sequence'] for r in read(lock['inputs']['sequences']['path'])}
                by=defaultdict(list)
                for r in cands:by[r['pdb_id']].append(r)
                extraction=[]
                for pdb,rs in sorted(by.items()):
                    c=coordinate[pdb]
                    # Recheck even when resuming a completed coordinate task.
                    if digest(c['coordinate_path'])!=c['sha256']:raise ValueError('cached coordinate changed: '+pdb)
                    extraction.append(dict(pdb_id=pdb,path=c['coordinate_path'],sha256=c['sha256'],rows=rs,
                        sequences={r['uniprot']:sequences[r['uniprot']] for r in rs}))
                assets=flatten(tasks(out,'02_cached_coordinates',extract_batch,batches(extraction,10),args.io_workers));assetmap={x['pdb_id']:x for x in assets}
                spatial_jobs=[]
                for r in cp:
                    rs=groups['candidate',r['pair_id']]
                    spatial_jobs.append(dict(pair_id=r['pair_id'],rows=rs,assets={p:assetmap[p] for p in {x['pdb_id'] for x in rs}}))
                spatial=flatten(tasks(out,'03_spatial_consistency',audit_batch,batches(spatial_jobs,25),args.workers))
            fitjobs=[]
            for regime,column in [('protein_held_out','protein_fold'),('family_held_out','family_fold')]:
                for fold in range(5):fitjobs.append(dict(regime=regime,fold=fold,train=[r for r in rp if r[column]!=fold],test=[r for r in rp if r[column]==fold]))
            fitjobs.append(dict(regime='deployment',fold=-1,train=rp,test=[r for r in cp if r['feature_complete']]))
            fits=tasks(out,'04_refit_46_rankings',fit_score,fitjobs,args.workers)
            lookup={r['pair_id']:r for r in rp};metricjobs=[]
            for regime in ['protein_held_out','family_held_out']:
                preds=[p for f in fits if f['regime']==regime for p in f['predictions']]
                if len(preds)!=len(rp) or {p['pair_id'] for p in preds}!=set(lookup):raise ValueError('OOF coverage')
                for rid,_ in METHODS:metricjobs.append(dict(regime=regime,ranking_id=rid,replicates=fp['bootstrap_replicates'],rows=[dict(lookup[p['pair_id']],score=p['scores'][rid],raw_score=p['scores']['RAW:RAW_D']) for p in preds]))
            metrics=tasks(out,'05_heldout_and_bootstrap',evaluation,metricjobs,args.workers)
            save(out/'STATUS.json',dict(stage='06_merge_filters_handoff',status='RUNNING'))
            summary=export(out,rp,cp,fits,metrics,spatial,groups,lock)
            frequency_rows=[dict(source=x['pair']['source'],pair_id=x['pair']['pair_id'],**r) for x in rebuilt for r in x['frequencies']]
            table(out/'data/RESIDUE_FREQUENCIES.tsv.gz',frequency_rows)
            save(out/'SUMMARY.json',summary)
            subprocess.run([sys.executable,str(PKG/'scripts/make_report.py'),'--run-dir',str(out)],check=True)
            check=subprocess.run([sys.executable,str(PKG/'scripts/validate_pipeline.py'),'--run-dir',str(out)],check=True,capture_output=True,text=True)
            save(out/'VALIDATION.json',json.loads(check.stdout));inputs(lock)
            save(out/'STATUS.json',dict(status='PASS_COMPUTATIONAL_PIPELINE',manual_review_complete=False,live_deployment=False,network_requests_performed=0,**summary))
            # Bounded, shallow listings in this run only; no recursive filesystem scan.
            directories=[out,out/'data',out/'stages',out/'figure_source',out/'secondsite_handoff']
            directories += [out/'tasks'/s for s in ['01_pair_reconstruction','02_cached_coordinates','03_spatial_consistency','04_refit_46_rankings','05_heldout_and_bootstrap'] if (out/'tasks'/s).exists()]
            files=sorted(p for folder in directories for p in folder.iterdir() if p.is_file() and p.name not in ['RUN.lock','FINAL_CHECKSUMS.sha256'] and '.pending.' not in p.name and '.orphan.' not in p.name and not p.name.startswith('FAILURE_'))
            atomic(out/'FINAL_CHECKSUMS.sha256',''.join('%s  %s\n'%(digest(p),p.relative_to(out)) for p in files).encode())
        except BaseException as e:
            save(out/('FAILURE_'+uuid.uuid4().hex+'.json'),dict(error=repr(e)))
            save(out/'STATUS.json',dict(status='FAILED_RESUMABLE',error=repr(e)));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',default=str(PKG/'INPUT_LOCK.json'));p.add_argument('--output',default=str(PKG/'runs/complete_reanalysis'))
    p.add_argument('--workers',type=int,default=min(8,len(os.sched_getaffinity(0))));p.add_argument('--io-workers',type=int,default=2)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--test-fail-after',type=int,default=0,help=argparse.SUPPRESS)
    a=p.parse_args()
    if min(a.workers,a.io_workers)<1:p.error('worker counts must be positive')
    run(a)

if __name__=='__main__':main()
