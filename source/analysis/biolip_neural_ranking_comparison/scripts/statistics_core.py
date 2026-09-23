"""Tie-aware statistics and multiplicity-preserving family bootstrap."""
import numpy as np
import pandas as pd


def auc(y,s):
    y=np.asarray(y,int);n=int(y.sum());z=len(y)-n
    if not n or not z:return np.nan
    ranks=pd.Series(np.asarray(s,float)).rank(method='average').to_numpy()
    return float((ranks[y==1].sum()-n*(n+1)/2)/(n*z))


def spearman(a,b):
    a=pd.Series(np.asarray(a,float)).rank(method='average').to_numpy()
    b=pd.Series(np.asarray(b,float)).rank(method='average').to_numpy()
    a=a-a.mean();b=b-b.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den>0 else np.nan


def top1_agreement(a,b):
    """Probability of matching independent uniform choices from tied top sets."""
    a=np.asarray(a);b=np.asarray(b)
    x=a==a.max();y=b==b.max()
    return float((x&y).sum()/(x.sum()*y.sum()))


def bootstrap_values(values,cluster_codes,multiplicities,n_clusters):
    values=np.asarray(values,float);codes=np.asarray(cluster_codes,int);ok=np.isfinite(values)
    sums=np.bincount(codes[ok],weights=values[ok],minlength=n_clusters)
    counts=np.bincount(codes[ok],minlength=n_clusters)
    denominator=multiplicities.dot(counts)
    numerator=multiplicities.dot(sums)
    result=np.full(len(denominator),np.nan)
    good=denominator>0;result[good]=numerator[good]/denominator[good]
    return result
