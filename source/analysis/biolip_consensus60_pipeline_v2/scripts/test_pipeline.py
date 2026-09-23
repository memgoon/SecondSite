"""Temporary-only end-to-end tests; no real-data production invocation."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from common import PKG,OLD,save,table,read,digest
from spatial import compare,audit_pair,extract

def fixture(root):
    rng=np.random.default_rng(811);ref=[];folds=[];cand=[];old=[];audit=[]
    for i in range(20):
        for label in [0,1]:
            for pdb in [0,1]:
                oid='O_%d_%d_%d'%(i,label,pdb)
                r=dict(observation_id=oid,uniprot='U%d'%i,full_inchikey='KEY%d_%d'%(i,label),pdb_id='P%d_%d'%(i,pdb),receptor_chain='A',ligand_chain='A',ligand_auth_seq_id='500',ligand_ccd='ABC',
                    binding_uniprot_positions='1;2;3' if pdb==0 else '2;3;4',binary_label=str(label),
                    ligand_centroid_to_orthosteric_site_CA_centroid_A=str(2+label*8+rng.random()*4),molecular_weight=str(200+50*rng.random()),clogp=str(rng.normal()),aromatic_ring_count=str((i+label)%4))
                ref.append(r);folds.append(dict(observation_id=oid,uniprot=r['uniprot'],binary_label=str(label),family_component_id='F%d'%i,protein_fold=str(i%5),family_fold=str((i//2)%5)))
    for i in range(10):
        for pdb in range(2):
            r=dict(observation_id='C%d_%d'%(i,pdb),candidate_pair_id='OLD%d'%i,uniprot='U%d'%i,full_inchikey='CANDKEY%d'%i,pdb_id='C%d_%d'%(i,pdb),receptor_chain='A',ligand_chain='A',ligand_auth_seq_id='500',ligand_ccd='XYZ',binding_uniprot_positions='1;2;3' if pdb==0 else '2;3;4',
                ligand_centroid_to_nearest_orthosteric_site_CA_centroid_A=str(12+rng.random()*10),ligand_to_nearest_orthosteric_site_min_heavy_A='7',candidate_nearest_orthosteric_site_residue_overlap_count='0',molecular_weight=str(210+rng.random()*50),clogp=str(rng.normal()),aromatic_ring_count=str(i%4))
            cand.append(r)
        audit.append(dict(candidate_pair_id='OLD%d'%i,passes_substantial_organic_compound='True',passes_known_role_novelty='True'))
    from analysis_steps import METHODS
    for i in range(10):
        for rid,_ in METHODS:old.append(dict(candidate_pair_id='OLD%d'%i,ranking_id=rid,best_site_ranking_score='20',pair_global_rank=str(i+1)))
    files={}
    for name,rs in [('reference',ref),('folds',folds),('candidates',cand),('audit',audit),('old_pairs',old)]:
        p=root/(name+'.tsv');table(p,rs);files[name]=dict(path=str(p),sha256=digest(p))
    runtime={str(PKG/'scripts'/n):digest(PKG/'scripts'/n) for n in ['common.py','analysis_steps.py','spatial.py','run_pipeline.py','validate_pipeline.py','make_report.py']}
    for n in ['08_evaluate_heldout_rankings.py','09c_compute_candidate_geometry.py']:runtime[str(OLD/'scripts'/n)]=digest(OLD/'scripts'/n)
    lock=root/'LOCK.json';save(lock,dict(inputs=files,runtime_hashes=runtime,expected_rows=[80,20],synthetic=True,test_replicates=100))
    return lock

def invoke(lock,out,workers,*extra):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(out.parent/'mplcache'))
    return subprocess.run([sys.executable,str(PKG/'scripts/run_pipeline.py'),'--lock',str(lock),'--output',str(out),'--workers',str(workers),'--io-workers','1',*extra],capture_output=True,text=True,env=env)

def main():
    p=argparse.ArgumentParser();p.add_argument('--results');a=p.parse_args();checks=[]
    def ok(n):checks.append(dict(test=n,status='PASS'))
    rng=np.random.default_rng(2);ca={str(i+1):r.tolist() for i,r in enumerate(rng.normal(size=(30,3)))}
    first=dict(chain=dict(ca=ca,identity=1.,coverage=1.),ligand=[[0.,0.,0.],[1.,0.,0.]])
    second=dict(chain=dict(ca={k:(np.array(v)+[2,3,4]).tolist() for k,v in ca.items()},identity=1.,coverage=1.),ligand=[[2.,3.,4.],[3.,3.,4.]])
    assert compare(first,second)['state']=='same_location_geometry';ok('rigid_transform_same_site')
    second['ligand']=[[22.,3.,4.],[23.,3.,4.]];assert compare(first,second)['state']=='distinct_location_geometry';ok('spatially_distinct_site')
    second['ligand']=[[8.,3.,4.],[9.,3.,4.]];assert compare(first,second)['state']=='unresolved_geometry_band';ok('ambiguous_geometry_not_forced')
    # Same-edge chaining must not create a site: 0/3/6 A positions form an unresolved chain.
    obs=[];asset=dict(pdb_id='S',chains={'A|U':first['chain']},observations=[])
    for i,x in enumerate([0.,3.,6.]):
        obs.append(dict(observation_id='O%d'%i,pdb_id='S',receptor_chain='A',uniprot='U',ligand_ccd='ABC',ligand_chain='A',ligand_auth_seq_id=str(i),binding_uniprot_positions='1;2'))
        asset['observations'].append(dict(observation_id='O%d'%i,chain_key='A|U',ligand=[[x,0,0],[x+1,0,0]]))
    assert audit_pair(dict(pair_id='P',rows=obs,assets={'S':asset}))['status']=='unresolved';ok('no_single_linkage_spatial_expansion')
    with tempfile.TemporaryDirectory(prefix='biolip60_v2_') as td:
        root=Path(td)
        import gemmi
        structure=gemmi.Structure();model=gemmi.Model('1');chain=gemmi.Chain('A')
        for i in range(30):
            res=gemmi.Residue();res.name='ALA';res.seqid=gemmi.SeqId(i+1,' ');res.entity_type=gemmi.EntityType.Polymer
            atom=gemmi.Atom();atom.name='CA';atom.element=gemmi.Element('C');atom.pos=gemmi.Position(i*1.4,np.sin(i),np.cos(i));res.add_atom(atom);chain.add_residue(res)
        model.add_chain(chain);ligchain=gemmi.Chain('B');res=gemmi.Residue();res.name='ABC';res.seqid=gemmi.SeqId(100,' ');res.entity_type=gemmi.EntityType.NonPolymer
        for i in range(3):
            atom=gemmi.Atom();atom.name='C%d'%i;atom.element=gemmi.Element('C');atom.pos=gemmi.Position(i,2,3);res.add_atom(atom)
        ligchain.add_residue(res);model.add_chain(ligchain);structure.add_model(model);structure.setup_entities();structure.assign_label_seq_id()
        cif=root/'synthetic.cif';structure.make_mmcif_document().write_file(str(cif))
        er=dict(observation_id='SYN',receptor_chain='A',uniprot='U',ligand_ccd='ABC',ligand_chain='B',ligand_auth_seq_id='100')
        ex=extract(dict(path=str(cif),sha256=digest(cif),pdb_id='SYN',rows=[er],sequences={'U':'A'*30}))
        assert len(ex['chains']['A|U']['ca'])==30 and len(ex['observations'][0]['ligand'])==3
        ok('actual_synthetic_CIF_read_chain_mapping_and_exact_ligand_extraction')
        lock=fixture(root);clean=root/'clean'
        x=invoke(lock,clean,2)
        if x.returncode:raise RuntimeError(x.stdout+'\n'+x.stderr)
        ok('parallel_complete_reference_refit_evaluation_filter_export_validation')
        resume=root/'resume';x=invoke(lock,resume,2,'--test-fail-after','1');assert x.returncode==86,x.stderr
        marks=list((resume/'tasks/01_pair_reconstruction').glob('*.commit.json'));before={p.name:(digest(p),p.stat().st_mtime_ns) for p in marks};assert before
        x=invoke(lock,resume,1)
        if x.returncode:raise RuntimeError(x.stdout+'\n'+x.stderr)
        assert before=={p.name:(digest(p),p.stat().st_mtime_ns) for p in marks};ok('fresh_process_resume_skips_completed_tasks')
        # Numeric artifacts deterministic across CPU-worker counts, completed task order and resume.
        for name in ['REFERENCE_PAIRS.tsv','CANDIDATE_PAIRS.tsv','CANDIDATE_RANKINGS.tsv.gz','HELDOUT_PAIR_METRICS_AND_CI.tsv','OOF_PAIR_SCORES.tsv.gz']:
            assert digest(clean/'data'/name)==digest(resume/'data'/name),name
        ok('one_vs_two_workers_numeric_byte_determinism')
        names=['STATUS.json','VALIDATION.json','FINAL_CHECKSUMS.sha256'];h={n:digest(clean/n) for n in names}
        x=subprocess.run([sys.executable,str(PKG/'scripts/validate_pipeline.py'),'--run-dir',str(clean),'--verify-final'],capture_output=True,text=True)
        assert x.returncode==0,x.stderr;assert h=={n:digest(clean/n) for n in names};ok('independent_validate_only_preserves_files')
        manifest_before=(clean/'FINAL_CHECKSUMS.sha256').read_text()
        x=invoke(lock,clean,2);assert x.returncode==0,x.stderr
        assert h=={n:digest(clean/n) for n in names},'manifest difference: '+str(set(manifest_before.splitlines())^set((clean/'FINAL_CHECKSUMS.sha256').read_text().splitlines()))
        ok('completed_resume_hash_determinism')
        target=clean/'data/REFERENCE_PAIRS.tsv';rs=read(target);rs[0]['consensus60']='999';table(target,rs)
        x=subprocess.run([sys.executable,str(PKG/'scripts/validate_pipeline.py'),'--run-dir',str(clean)],capture_output=True,text=True);assert x.returncode!=0;ok('semantic_consensus_tamper_rejected')
        x=invoke(lock,clean,2);assert x.returncode!=0 and 'completed output corruption' in x.stderr;ok('corrupt_completed_run_blocks')
        src=Path(json.loads(lock.read_text())['inputs']['audit']['path']);src.write_text('tampered synthetic input')
        x=invoke(lock,root/'tamper',2);assert x.returncode!=0 and 'input checksum failure' in x.stderr;ok('input_tamper_blocks')
    if a.results:table(Path(a.results),checks)
    print(json.dumps(dict(status='PASS',tests=len(checks),checks=checks,synthetic_only=True)))

if __name__=='__main__':main()
