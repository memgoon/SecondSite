"""Paired target-cluster bootstrap, preserving multiplicities and score ties."""
from common import *

def prepare(y,s,target):
    order=np.argsort(s,kind='mergesort');scores=np.asarray(s)[order]
    return np.asarray(y)[order],np.asarray(target)[order],np.r_[0,np.flatnonzero(np.diff(scores))+1]

def weighted_auc(prepared,mult):
    y,t,starts=prepared;w=mult[:,t]
    pos=np.add.reduceat(w*y,starts,axis=1);neg=np.add.reduceat(w*(1-y),starts,axis=1)
    den=pos.sum(1)*neg.sum(1);values=np.full(len(den),np.nan);ok=den>0
    num=(pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(1)
    values[ok]=num[ok]/den[ok]
    return values

def job(item):
    meta,frame,left,right,reps=item
    y=frame.weak2020_label.to_numpy(int);codes,names=pd.factorize(frame.uniprot,sort=True)
    ls,rs=frame[left].to_numpy(float),frame[right].to_numpy(float)
    if meta['metric']=='pooled':
        a,b=prepare(y,ls,codes),prepare(y,rs,codes);point=auc(y,ls)-auc(y,rs)
        lv,rv=auc(y,ls),auc(y,rs);support=len(frame);groups=1
    else:
        delta=np.full(len(names),np.nan);aval=delta.copy();bval=delta.copy();support=0
        for k in range(len(names)):
            sel=codes==k
            if len(set(y[sel]))!=2:continue
            aval[k]=auc(y[sel],ls[sel]);bval[k]=auc(y[sel],rs[sel]);delta[k]=aval[k]-bval[k];support+=int(sel.sum())
        lv,rv=float(np.nanmean(aval)),float(np.nanmean(bval));point=lv-rv;groups=int(np.isfinite(delta).sum())
    rng=np.random.RandomState(int(digest(meta)[:8],16));values=[]
    for start in range(0,reps,64):
        draw=rng.randint(0,len(names),size=(min(64,reps-start),len(names)))
        if meta['metric']=='pooled':
            mult=np.array([np.bincount(d,minlength=len(names)) for d in draw],float)
            v=weighted_auc(a,mult)-weighted_auc(b,mult)
        else:
            selected=delta[draw];count=np.isfinite(selected).sum(1);v=np.full(len(draw),np.nan);ok=count>0
            v[ok]=np.nansum(selected,axis=1)[ok]/count[ok]
        values.extend(v[np.isfinite(v)].tolist())
    if len(values)<.95*reps:raise ValueError('Insufficient defined bootstrap replicates: '+str(meta))
    lo,hi=np.percentile(values,[2.5,97.5])
    return dict(meta,delta=point,ci_low=float(lo),ci_high=float(hi),test_auroc=lv,reference_auroc=rv,
        rows_input=len(frame),rows_used=support,groups_used=groups,targets_resampled=len(names),
        requested_replicates=reps,valid_replicates=len(values),invalid_replicates=reps-len(values),
        bootstrap_fraction_delta_gt_zero=float((np.asarray(values)>0).mean()),
        cluster='uniprot',ci_scope='pointwise, conditional on fixed ensembles; no multiple-comparison correction')
