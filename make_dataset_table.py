#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from mchf.metrics import bootstrap_ci


def row_from_csv(path, display_name, n_boot=1000, seed=12345):
    df=pd.read_csv(path)
    y=df.y_true.to_numpy(dtype=int); score=df.score.to_numpy(dtype=float)
    thresholds=df.threshold.to_numpy(dtype=float)
    pred=(score>=thresholds).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    auc=roc_auc_score(y,score); lo,hi=bootstrap_ci(y,score,n_boot=n_boot,seed=seed)
    return {
      'Dataset':display_name,
      'Accuracy (%)':100*(tp+tn)/len(y),
      'Sensitivity (%)':100*tp/max(tp+fn,1),
      'Specificity (%)':100*tn/max(tn+fp,1),
      'Precision (%)':100*tp/max(tp+fp,1),
      'F1-score (%)':100*(2*tp)/max(2*tp+fp+fn,1),
      'AUC (95% CI)':f'{auc:.3f} ({lo:.3f}–{hi:.3f})',
      'N':len(y),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ddsm',required=True); p.add_argument('--inbreast',required=True); p.add_argument('--pooled',required=True)
    p.add_argument('--output',default='results/dataset_specific_performance.csv'); p.add_argument('--bootstrap',type=int,default=1000); p.add_argument('--seed',type=int,default=12345)
    args=p.parse_args()
    rows=[row_from_csv(args.ddsm,'DDSM',args.bootstrap,args.seed),row_from_csv(args.inbreast,'INbreast',args.bootstrap,args.seed),row_from_csv(args.pooled,'DDSM + INbreast pooled',args.bootstrap,args.seed)]
    out=pd.DataFrame(rows)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(path,index=False)
    print(out.to_string(index=False,float_format=lambda x:f'{x:.2f}'))
    print('\nSaved:',path)
if __name__=='__main__': main()
