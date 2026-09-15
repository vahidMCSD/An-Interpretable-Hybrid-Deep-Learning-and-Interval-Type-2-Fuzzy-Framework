#!/usr/bin/env python
from __future__ import annotations
import argparse, copy
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from mchf.data.preprocess import read_grayscale, preprocess_roi, PreprocessConfig
from mchf.data.augment import augment_roi
from mchf.models.classifiers import EfficientNetCBAMClassifier, ViTClassifier
from mchf.utils.seed import seed_everything

class DS(Dataset):
    def __init__(self,df,train=False): self.df=df.reset_index(drop=True); self.train=train; self.cfg=PreprocessConfig()
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; x=preprocess_roi(read_grayscale(r.image_path),self.cfg)
        if self.train: x=augment_roi(x)
        return torch.from_numpy(x[None]).float(),torch.tensor(float(r.label))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--branch',choices=['effnet','vit'],required=True); p.add_argument('--output',required=True)
    p.add_argument('--dataset',choices=['DDSM','INbreast','pooled'],default='pooled'); p.add_argument('--epochs',type=int,default=50); p.add_argument('--patience',type=int,default=5)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--seed',type=int,default=12345); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--no-pretrained',action='store_true')
    a=p.parse_args(); seed_everything(a.seed); df=pd.read_csv(a.manifest)
    if a.dataset!='pooled': df=df[df.dataset==a.dataset].copy()
    split=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=a.seed); tr,va=next(split.split(df,df.label,df.patient_id))
    trdf,vadf=df.iloc[tr],df.iloc[va]; dev=torch.device(a.device)
    model=(EfficientNetCBAMClassifier(pretrained=not a.no_pretrained) if a.branch=='effnet' else ViTClassifier(pretrained=not a.no_pretrained)).to(dev)
    pos=max(int((trdf.label==1).sum()),1); neg=max(int((trdf.label==0).sum()),1); lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=dev)); opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    best=None; best_auc=-1; wait=0
    for ep in range(a.epochs):
        model.train()
        for x,y in DataLoader(DS(trdf,True),batch_size=a.batch_size,shuffle=True):
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); loss=lossfn(model(x),y); loss.backward(); opt.step()
        model.eval(); ys=[]; ss=[]
        with torch.no_grad():
            for x,y in DataLoader(DS(vadf,False),batch_size=a.batch_size): ys.extend(y.numpy()); ss.extend(torch.sigmoid(model(x.to(dev))).cpu().numpy())
        auc=roc_auc_score(ys,ss) if len(set(ys))>1 else .5
        print(ep+1,auc)
        if auc>best_auc+1e-5: best_auc=auc; best=copy.deepcopy(model.state_dict()); wait=0
        else:
            wait+=1
            if wait>=a.patience: break
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':best,'branch':a.branch,'val_auc':best_auc,'args':vars(a)},a.output); print('saved',a.output)
if __name__=='__main__': main()
