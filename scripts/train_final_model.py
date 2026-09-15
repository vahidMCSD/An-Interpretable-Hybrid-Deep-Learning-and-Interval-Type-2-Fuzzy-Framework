#!/usr/bin/env python
from __future__ import annotations
import argparse,pickle,json
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import GroupShuffleSplit
from mchf.training import train_feature_model


def load(p):
    z=np.load(p,allow_pickle=True); return {k:z[k] for k in z.files}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--features',required=True); p.add_argument('--output-dir',default='checkpoints/final'); p.add_argument('--dataset',choices=['DDSM','INbreast','pooled'],default='pooled'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--epochs',type=int,default=50); p.add_argument('--seed',type=int,default=12345); a=p.parse_args(); d=load(a.features)
    if a.dataset!='pooled':
        keep=np.where(d['dataset'].astype(str)==a.dataset)[0]; n=len(d['label']); d={k:(v[keep] if getattr(v,'shape',()) and len(v)==n else v) for k,v in d.items()}
    y=d['label'].astype(int); groups=d['patient_id'].astype(str); ss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=a.seed); tr,va=next(ss.split(np.zeros(len(y)),y,groups))
    # Use validation both for early stopping and bundle sanity predictions; the independent holdout remains external to this utility.
    r=train_feature_model(d,tr,va,va,a.device,a.seed,epochs=a.epochs)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':r['model'].state_dict(),'wave_dim':r['pca_dim'],'threshold':r['threshold'],'dataset':a.dataset},out/'feature_model.pt')
    with open(out/'preprocessing.pkl','wb') as f: pickle.dump({'wavelet_pca':r['wavelet_pca'],'fusion_scaler':r['fusion_scaler'],'rule_normalizer':r['rule_normalizer'],'art':r['art'],'art_scaler':r['art_scaler']},f)
    np.save(out/'train_indices.npy',tr); np.save(out/'validation_indices.npy',va)
    (out/'bundle_info.json').write_text(json.dumps({'dataset':a.dataset,'threshold':r['threshold'],'pca_dim':r['pca_dim'],'val_auc':r['val_auc']},indent=2))
    print('saved final-model bundle to',out)
if __name__=='__main__':main()
