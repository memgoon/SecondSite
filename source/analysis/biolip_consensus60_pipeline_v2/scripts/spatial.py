"""Offline spatial consistency audit; geometry support is not a biological label."""
import importlib.util,itertools
from collections import defaultdict
import numpy as np
import gemmi
from common import OLD,digest,identity
from analysis_steps import contacts

spec=importlib.util.spec_from_file_location('frozen_geometry',OLD/'scripts/09c_compute_candidate_geometry.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)

def extract(task):
    path=task['path']
    if digest(path)!=task['sha256']:raise ValueError('coordinate checksum: '+path)
    structure=gemmi.read_structure(path);model=structure[0];chains={};obs=[]
    for r in task['rows']:
        key=r['receptor_chain']+'|'+r['uniprot']
        if key not in chains:
            index,seq,_=G.chain_index(model,r['receptor_chain'])
            mapping,_,identity_,coverage,_=G.alignment_maps(task['sequences'][r['uniprot']],seq,G.new_aligner())
            ca={str(u):index[k]['ca'].tolist() for u,k in mapping.items() if index[k]['ca'] is not None}
            chains[key]=dict(ca=ca,identity=None if not np.isfinite(identity_) else identity_,coverage=None if not np.isfinite(coverage) else coverage)
        ligand,count,_=G.find_ligand(model,r['ligand_ccd'],r['ligand_chain'],r['ligand_auth_seq_id'])
        obs.append(dict(observation_id=r['observation_id'],chain_key=key,
            ligand=None if ligand is None else ligand.tolist(),exact_ligand_matches=count))
    if digest(path)!=task['sha256']:raise ValueError('coordinate changed during read')
    return dict(pdb_id=task['pdb_id'],coordinate_sha256=task['sha256'],chains=chains,observations=obs)

def compare(a,b):
    if a['ligand'] is None or b['ligand'] is None:return dict(state='unresolved_missing_ligand')
    for x in [a,b]:
        if x['chain']['identity'] is None or x['chain']['identity']<.95 or x['chain']['coverage']<.8:return dict(state='unresolved_alignment_quality')
    shared=sorted(set(a['chain']['ca'])&set(b['chain']['ca']),key=int)
    if len(shared)<20:return dict(state='unresolved_fewer_than_20_common_CA')
    A=np.array([a['chain']['ca'][k] for k in shared]);B=np.array([b['chain']['ca'][k] for k in shared])
    ac=A.mean(axis=0);bc=B.mean(axis=0);u,_,vt=np.linalg.svd((B-bc).T@(A-ac))
    corr=np.eye(3);corr[-1,-1]=np.linalg.det(u@vt);R=u@corr@vt
    aligned=(B-bc)@R+ac;rmsd=float(np.sqrt(np.mean(np.sum((aligned-A)**2,axis=1))))
    L=np.array(a['ligand']);M=(np.array(b['ligand'])-bc)@R+ac
    center=float(np.linalg.norm(L.mean(axis=0)-M.mean(axis=0)));minimum=float(np.linalg.norm(L[:,None,:]-M[None,:,:],axis=2).min())
    if rmsd>2.5:state='unresolved_conformational_difference'
    elif center<=4 and minimum<=2:state='same_location_geometry'
    elif center>=10 and minimum>=4:state='distinct_location_geometry'
    else:state='unresolved_geometry_band'
    return dict(state=state,common_CA=len(shared),aligned_CA_RMSD=rmsd,ligand_centroid_distance=center,ligand_min_heavy_distance=minimum)

def audit_pair(task):
    rs=task['rows'];assets=task['assets'];unique={};oid_to_key={}
    for r in rs:
        key='|'.join(r[k] for k in ['pdb_id','receptor_chain','ligand_ccd','ligand_chain','ligand_auth_seq_id'])
        oid_to_key[r['observation_id']]=key
        asset=assets[r['pdb_id']];o=next(x for x in asset['observations'] if x['observation_id']==r['observation_id'])
        unique[key]=dict(ligand=o['ligand'],chain=asset['chains'][o['chain_key']])
    keys=sorted(unique);comparisons=[];relations={};parent={k:k for k in keys}
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    for a,b in itertools.combinations(keys,2):
        v=compare(unique[a],unique[b]);relations[a,b]=v['state'];comparisons.append(dict(first=a,second=b,**v))
        if v['state']=='same_location_geometry':parent[root(b)]=root(a)
    partitions=defaultdict(list)
    for k in keys:partitions[root(k)].append(k)
    # No single-linkage inference: ALL within-group edges must be same and ALL cross-group edges distinct.
    coherent=all(state==('same_location_geometry' if root(a)==root(b) else 'distinct_location_geometry') for (a,b),state in relations.items())
    valid_single=all(v['ligand'] is not None and v['chain']['identity'] is not None and v['chain']['identity']>=.95 for v in unique.values())
    if len(keys)==1:status='single_observation_location' if valid_single else 'unresolved'
    elif not coherent:status='unresolved'
    elif len(partitions)==1:status='geometry_consistent_single_location'
    else:status='geometry_supported_distinct_locations'
    locations=[]
    if status!='unresolved':
        for members in sorted(partitions.values()):
            selected=[r for r in rs if oid_to_key[r['observation_id']] in members];core,_=contacts(selected)
            location='LOC_'+identity([task['pair_id'],sorted(members)])[:20]
            # Real per-observation CA coordinates, never a virtual mixed-coordinate structure.
            geometry=[]
            for r in selected:
                x=unique[oid_to_key[r['observation_id']]];ca=[x['chain']['ca'][str(p)] for p in core if str(p) in x['chain']['ca']]
                recovery=len(ca)/len(core) if core else 0
                d=float(np.linalg.norm(np.mean(x['ligand'],axis=0)-np.mean(ca,axis=0))) if ca and x['ligand'] is not None and recovery>=.8 else None
                geometry.append(dict(observation_id=r['observation_id'],consensus_CA_recovery=recovery,ligand_to_own_consensus_CA_centroid=d))
            locations.append(dict(location_id=location,consensus60=';'.join(map(str,core)),observation_ids=sorted(r['observation_id'] for r in selected),
                pdbs=len({r['pdb_id'] for r in selected}),geometry=geometry))
    return dict(pair_id=task['pair_id'],status=status,unique_structural_instances=len(keys),comparisons=comparisons,locations=locations,
                biological_assembly_audited=False,manual_scientific_review_complete=False)

def extract_batch(batch):return [extract(t) for t in batch]
def audit_batch(batch):return [audit_pair(t) for t in batch]
