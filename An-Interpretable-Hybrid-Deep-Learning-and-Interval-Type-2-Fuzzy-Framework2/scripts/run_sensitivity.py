#!/usr/bin/env python
from __future__ import annotations
import argparse,subprocess,sys,json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output-dir',default='results/sensitivity');p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=50);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
    for shift in [-.10,-.05,0,.05,.10]:
        tag=('original' if shift==0 else f'{shift:+.0%}');pred=out/f'{tag}.csv'
        cmd=[sys.executable,str(Path(__file__).with_name('train_cv.py')),'--features',a.features,'--output',str(pred),'--dataset','pooled','--device',a.device,'--epochs',str(a.epochs),'--membership-shift',str(shift)]
        subprocess.check_call(cmd);s=json.load(open(pred.with_suffix('.summary.json')));rows.append({'Threshold Perturbation':tag,'Accuracy (%)':100*s['accuracy'],'Sensitivity (%)':100*s['sensitivity'],'Specificity (%)':100*s['specificity'],'AUC':s['auc']})
    tab=pd.DataFrame(rows);tab.to_csv(out/'table8.csv',index=False);print(tab.to_string(index=False))
if __name__=='__main__':main()
