from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, adjusted_rand_score


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact two-sided McNemar test without statsmodels dependency."""
    y=np.asarray(y_true); a=np.asarray(pred_a); b=np.asarray(pred_b)
    a_ok=a==y; b_ok=b==y
    n01=int(np.sum(a_ok & ~b_ok)); n10=int(np.sum(~a_ok & b_ok))
    n=n01+n10
    if n==0: return {"n01":0,"n10":0,"statistic":0.0,"pvalue":1.0}
    from scipy.stats import binomtest
    p=float(binomtest(min(n01,n10), n=n, p=0.5, alternative='two-sided').pvalue)
    stat=(abs(n01-n10)-1)**2/max(n,1)
    return {"n01":n01,"n10":n10,"statistic":float(stat),"pvalue":p}


def wilcoxon_paired(values_a, values_b):
    a=np.asarray(values_a,float); b=np.asarray(values_b,float)
    if len(a)!=len(b): raise ValueError('paired arrays must have same length')
    if np.allclose(a,b): return {"statistic":0.0,"pvalue":1.0}
    r=wilcoxon(a,b,zero_method='wilcox',alternative='two-sided',mode='auto')
    return {"statistic":float(r.statistic),"pvalue":float(r.pvalue)}


def art_mapping_stability(assignments_by_fold):
    """Pairwise ARI summary when assignments refer to a common sample subset."""
    keys=sorted(assignments_by_fold)
    vals=[]
    for i,k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            a=np.asarray(assignments_by_fold[k1]); b=np.asarray(assignments_by_fold[k2])
            if len(a)==len(b): vals.append(adjusted_rand_score(a,b))
    return float(np.mean(vals)) if vals else float('nan')
