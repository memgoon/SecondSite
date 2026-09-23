"""CPU regression tests; no GPU training or directory-wide scan."""
import ast
import subprocess
from shared import *
from aggregate import top,enrichment,strata
from transfer import return_names

def main():
    for name in ('shared.py','build_contract.py','runtime.py','aggregate.py','transfer.py','status.py','test_pipeline.py'):
        ast.parse((PACKAGE/'scripts'/name).read_text())
    for name in ('run_gpu.sh','send_gpu_input.sh','fetch_gpu_return.sh'):
        subprocess.run(['bash','-n',str(PACKAGE/'scripts'/name)],check=True)
    assert len(MODELS)*len(SEEDS)==24 and len(FULL_MODELS)==7 and len(tasks())==9
    assert len(return_names())==len(set(return_names()))==91
    x=pd.DataFrame(dict(_id=['b','a','c'],uniprot=['U','U','V'],score=[.9,.9,.1],has_allosteric_text=[0,1,1]))
    assert top(x,'score',.001)._id.tolist()==['a']
    assert enrichment(x,'score',.001,True)['top_rows']==2
    assert enrichment(x.assign(has_allosteric_text=0),'score',.001)['status']=='not_evaluated_no_positive_class'
    assert auc([0,1],[.5,.5])==.5 and ap([0,1],[.5,.5])==.5
    co=pd.DataFrame(dict(uniprot=['A','B'],connectivity_key=['K','L']))
    q=pd.DataFrame(dict(uniprot=['A','C','D'],connectivity_key=['K','M','N']))
    pf=dict(records={'A':dict(annotation_status='annotated',pfam_ids=['F']),
                     'C':dict(annotation_status='annotated',pfam_ids=['G'])})
    result=novelty(q,co,pf)
    assert result.pfam_family_status_ligand_anchored.tolist()==['seen','training_reference_incomplete','annotation_unavailable']
    if (PACKAGE/'validation/CPU_CONTRACT.json').exists():
        verify_contract()
    rows={}
    for scope in ('reference','full','biochemical'):
        parts=[]
        for s,i in tasks():
            if s!=scope:continue
            f=prior_frame(s,i)
            assert not f[row_id(s)].duplicated().any();parts.append(f)
        f=pd.concat(parts);assert not f[row_id(scope)].duplicated().any();rows[scope]=len(f)
    assert rows==dict(reference=29929,full=796165,biochemical=492),rows
    cohort=bench.read_data();pfam=read_json(ROOT/read_json(OLD/'validation/CPU_CONTRACT.json')['external_resources']['pfam_cache']['path'])
    q=novelty(cohort[['uniprot','connectivity_key']].copy(),cohort,pfam)
    assert q.protein_seen_ligand_anchored.eq(1).all() and q.ligand_seen_ligand_anchored.eq(1).all()
    result=dict(status='CPU_TESTS_PASS',rows=rows,deploy_fits=24,inference_tasks=9,
                gpu_execution_tested=False,tests=['Python/shell syntax','deterministic ties and minimum top-k',
                    'zero-positive status','tie-aware AUROC/AP','incomplete Pfam safeguard',
                    'exact prior eligible rows','unique row IDs','self-novelty controls','frozen dependency hashes'])
    write_json(PACKAGE/'validation/STATIC_TESTS.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
