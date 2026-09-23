import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();root=Path(a.run_dir)
    metrics=read(root/'data/HELDOUT_PAIR_METRICS_AND_CI.tsv');counts=read(root/'data/FILTER_COUNTS.tsv');summary=json.loads((root/'SUMMARY.json').read_text())
    chosen=['RAW:RAW_D','KDE_LEGACY:D+AR','KDE_LEGACY:D+MW','QNB:D+AR']
    vals=[next(r for r in metrics if r['regime']=='family_held_out' and r['ranking_id']==name) for name in chosen]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6,2.8));scores=[float(r['auc_value']) for r in vals]
    ax.errorbar(scores,list(range(4)),xerr=[[v-float(r['auc_low']) for v,r in zip(scores,vals)],[float(r['auc_high'])-v for v,r in zip(scores,vals)]],fmt='o',color='#397f80',capsize=3)
    ax.set_yticks(range(4),chosen);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Pair-based within-protein macro AUROC (95% cluster-bootstrap CI)');fig.tight_layout()
    dest=root/'figure_source';dest.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(dest/('heldout_pair_recovery.'+ext),dpi=220,metadata={'CreationDate':None} if ext=='pdf' else None)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3));ax.barh([r['step'] for r in counts],[int(r['pairs']) for r in counts],color='#397f80');ax.invert_yaxis();ax.set_xlabel('Exact protein–ligand pairs');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/('pair_filter_counts.'+ext),dpi=220,metadata={'CreationDate':None} if ext=='pdf' else None)
    plt.close(fig)
    report='# BioLiP consensus-60 downstream reanalysis\n\n'+json.dumps(summary,indent=2)+'\n\n'
    report+='Reference and candidate observations are collapsed to exact protein/full-InChIKey pairs before fitting. Raw features use a median of within-PDB medians. The 60% contact consensus uses equal PDB weight and is NOT used to invent a virtual coordinate model. Missing observations remain in lineage; candidate pairs with incomplete model inputs have explicit NA predictions. Original protein/family folds are preserved. All 45 likelihood combinations are refitted, plus the direct-distance ranking. Evaluation uses the new pair unit, including 10,000 cluster-bootstrap replicates; old site-based AUROCs are not a numerical before/after benchmark.\n\n'
    report+='Cached ligand-to-known-orthosteric-site distances are unchanged inputs because that measurement does not depend on the former candidate Jaccard clusters. Source-target geometry is not silently redefined. Cached coordinates are separately checked to align receptor CA atoms, compare ligand positions, and measure ligand-to-60%-consensus distances in each real structure. Spatial decisions are conservative geometry-support categories, not a substitute for biological-assembly/crystal-contact review. A pair remains one pair even with nested distinct locations. Ambiguous alignment/conformation/position cases go to review; no forced assignment. Residue observability is not used to change the frozen 60% denominator.\n\n'
    report+='Pair-level global/within-protein percentiles and CP10-like screens are recomputed. Chemistry and known-role flags are reused only after input hash verification, with disagreement retained. Old six-pair manual dispositions are not automatically transferred. All shortlists require manual structure-context reassessment. Figure-source plots/tables and SecondSite handoff files are generated, but the manuscript, original figures and live database are not modified.\n\n'
    report+='Computational PASS does not mean novel allostery validated, manual review completed, or SecondSite deployed. Independent numerical checks and task hashes are reported in VALIDATION.json. Final next step is to inspect SPATIAL_REVIEW_QUEUE.tsv and the provisional shortlist, then approve curation and publication updates separately.\n'
    atomic(root/'REPORT.md',report.encode())

if __name__=='__main__':main()
