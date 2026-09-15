#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix,roc_auc_score
from mchf.training import train_feature_model
from mchf.metrics import bootstrap_ci
from mchf.utils.seed import seed_everything

def load(path):z=np.load(path,allow_pickle=True);return {k:z[k] for k in z.files}
def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset',choices=['DDSM','INbreast','pooled'],default='pooled');p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--seed',type=int,default=12345);p.add_argument('--outer-folds',type=int,default=5);p.add_argument('--inner-folds',type=int,default=3);p.add_argument('--pca-variance',type=float,default=.98);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--epochs',type=int,default=50);p.add_argument('--patience',type=int,default=5);p.add_argument('--dropout',type=float,default=.1);p.add_argument('--fuzzy-uncertainty',type=float,default=.2);p.add_argument('--membership-shift',type=float,default=0);p.add_argument('--rho',type=float,default=.85);p.add_argument('--beta',type=float,default=.5);p.add_argument('--art-alpha',type=float,default=1e-3);p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args();seed_everything(a.seed);d=load(a.features)
    if a.dataset!='pooled':
        keep=np.where(d['dataset'].astype(str)==a.dataset)[0];n=len(d['label']);d={k:(v[keep] if getattr(v,'shape',()) and len(v)==n else v) for k,v in d.items()}
    y=d['label'].astype(int);groups=d['patient_id'].astype(str);outer=StratifiedGroupKFold(a.outer_folds,shuffle=True,random_state=a.seed);rows=[]
    for fold,(tv,te) in enumerate(outer.split(np.zeros(len(y)),y,groups),1):
        inner=StratifiedGroupKFold(a.inner_folds,shuffle=True,random_state=a.seed+fold);rt,rv=next(inner.split(np.zeros(len(tv)),y[tv],groups[tv]));tr,va=tv[rt],tv[rv];r=train_feature_model(d,tr,va,te,a.device,a.seed+fold,a.pca_variance,a.lr,a.batch_size,a.weight_decay,a.epochs,a.patience,a.dropout,a.fuzzy_uncertainty,a.membership_shift,a.rho,a.beta,a.art_alpha,'km');o=r['outputs'];fuzzy=o['fuzzy_score'].detach().cpu().numpy();flo=o['fuzzy_lower'].detach().cpu().numpy();fup=o['fuzzy_upper'].detach().cpu().numpy()
        for j,idx in enumerate(te):rows.append({'index':int(idx),'fold':fold,'dataset':str(d['dataset'][idx]),'patient_id':str(d['patient_id'][idx]),'y_true':int(y[idx]),'score':float(r['prob'][j]),'threshold':float(r['threshold']),'fuzzy_score':float(fuzzy[j]),'fuzzy_lower':float(flo[j]),'fuzzy_upper':float(fup[j]),'art_score':float(r['art_score'][j]),'art_pred':int(r['art_pred'][j]),'pca_dim':int(r['pca_dim']),'n_art_categories':int(r['n_art_categories']),'val_auc':float(r['val_auc'])})
    pred=pd.DataFrame(rows).sort_values('index');out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);pred.to_csv(out,index=False);yv=pred.y_true.to_numpy();s=pred.score.to_numpy();ph=(s>=pred.threshold.to_numpy()).astype(int);tn,fp,fn,tp=confusion_matrix(yv,ph,labels=[0,1]).ravel();summary={'dataset':a.dataset,'n':len(pred),'accuracy':(tp+tn)/len(pred),'sensitivity':tp/max(tp+fn,1),'specificity':tn/max(tn+fp,1),'precision':tp/max(tp+fp,1),'f1':2*tp/max(2*tp+fp+fn,1),'auc':roc_auc_score(yv,s),'auc_ci':bootstrap_ci(yv,s,n_boot=a.bootstrap,seed=a.seed)};open(out.with_suffix('.summary.json'),'w').write(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
