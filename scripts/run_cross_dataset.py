#!/usr/bin/env python
from __future__ import annotations
import argparse,subprocess,sys,json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output-dir',default='results/cross_dataset');p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=50);a=p.parse_args();o=Path(a.output_dir);o.mkdir(parents=True,exist_ok=True);rows=[]
    for tr,te in [('DDSM','INbreast'),('INbreast','DDSM')]:
        pred=o/f'{tr}_to_{te}.csv';cmd=[sys.executable,str(Path(__file__).with_name('train_external.py')),'--features',a.features,'--train-dataset',tr,'--test-dataset',te,'--output',str(pred),'--device',a.device,'--epochs',str(a.epochs)];subprocess.check_call(cmd);s=json.load(open(pred.with_suffix('.summary.json')));rows.append({'Training Dataset':tr,'Testing Dataset':te,'Accuracy (%)':100*s['accuracy'],'Sensitivity (%)':100*s['sensitivity'],'Specificity (%)':100*s['specificity'],'AUC':s['auc'],'95% CI (AUC)':f"{s['auc_ci'][0]:.3f}–{s['auc_ci'][1]:.3f}"})
    t=pd.DataFrame(rows);t.to_csv(o/'table10.csv',index=False);print(t.to_string(index=False))
if __name__=='__main__':main()
