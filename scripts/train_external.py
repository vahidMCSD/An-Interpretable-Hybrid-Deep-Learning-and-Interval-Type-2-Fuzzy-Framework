#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix,roc_auc_score
from mchf.training import train_feature_model
from mchf.metrics import bootstrap_ci

def load(p):z=np.load(p,allow_pickle=True);return {k:z[k] for k in z.files}
def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--train-dataset',required=True);p.add_argument('--test-dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--epochs',type=int,default=50);p.add_argument('--seed',type=int,default=12345);p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args();d=load(a.features);trpool=np.where(d['dataset'].astype(str)==a.train_dataset)[0];te=np.where(d['dataset'].astype(str)==a.test_dataset)[0];y=d['label'].astype(int);g=d['patient_id'].astype(str);sg=StratifiedGroupKFold(5,shuffle=True,random_state=a.seed);reltr,relva=next(sg.split(np.zeros(len(trpool)),y[trpool],g[trpool]));tr,va=trpool[reltr],trpool[relva];r=train_feature_model(d,tr,va,te,a.device,a.seed,epochs=a.epochs);rows=[]
    for j,idx in enumerate(te):rows.append({'index':int(idx),'dataset':str(d['dataset'][idx]),'patient_id':str(d['patient_id'][idx]),'y_true':int(y[idx]),'score':float(r['prob'][j]),'threshold':float(r['threshold']),'art_score':float(r['art_score'][j]),'art_pred':int(r['art_pred'][j])})
    q=pd.DataFrame(rows);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);q.to_csv(out,index=False);ph=(q.score.to_numpy()>=q.threshold.to_numpy()).astype(int);tn,fp,fn,tp=confusion_matrix(q.y_true,ph,labels=[0,1]).ravel();s={'accuracy':(tp+tn)/len(q),'sensitivity':tp/max(tp+fn,1),'specificity':tn/max(tn+fp,1),'precision':tp/max(tp+fp,1),'f1':2*tp/max(2*tp+fp+fn,1),'auc':roc_auc_score(q.y_true,q.score),'auc_ci':bootstrap_ci(q.y_true,q.score,n_boot=a.bootstrap,seed=a.seed)};open(out.with_suffix('.summary.json'),'w').write(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
if __name__=='__main__':main()
