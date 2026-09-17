#!/usr/bin/env python
from __future__ import annotations
import argparse,json
import numpy as np,pandas as pd
from mchf.analysis import mcnemar_exact,wilcoxon_paired

def load(path):
    d=pd.read_csv(path); pred=(d.score.to_numpy()>=d.threshold.to_numpy()).astype(int) if 'threshold' in d else (d.score.to_numpy()>=.5).astype(int); return d,pred

def main():
    p=argparse.ArgumentParser(); p.add_argument('--proposed',required=True); p.add_argument('--baseline',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    A,pa=load(a.proposed);B,pb=load(a.baseline); A=A.sort_values('index');B=B.sort_values('index')
    if not np.array_equal(A['index'].to_numpy(),B['index'].to_numpy()): raise ValueError('Predictions must refer to identical samples')
    mc=mcnemar_exact(A.y_true,pa,pb)
    # Fold-level accuracy paired Wilcoxon.
    va=[];vb=[]
    for f,g in A.groupby('fold'):
        idx=g.index; va.append(float(np.mean(pa[idx]==g.y_true.to_numpy()))); gb=B[B.fold==f]; pb_f=(gb.score.to_numpy()>=gb.threshold.to_numpy()).astype(int) if 'threshold' in gb else (gb.score.to_numpy()>=.5).astype(int); vb.append(float(np.mean(pb_f==gb.y_true.to_numpy())))
    wi=wilcoxon_paired(va,vb); res={'mcnemar':mc,'wilcoxon_fold_accuracy':wi,'proposed_fold_accuracy':va,'baseline_fold_accuracy':vb}
    open(a.output,'w').write(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__':main()
