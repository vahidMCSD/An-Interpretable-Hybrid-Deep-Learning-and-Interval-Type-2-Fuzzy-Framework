#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from mchf.features.wavelet import WaveletPCA
from mchf.fuzzy.type2 import RuleFeatureNormalizer, Type2Config
from mchf.hybrid import FeatureLevelHybrid
from mchf.art.fuzzy_art import FuzzyART, FuzzyARTConfig
from mchf.metrics import youden_threshold, classification_metrics, bootstrap_ci
from mchf.utils.seed import seed_everything


def load_npz(path):
    z=np.load(path, allow_pickle=True)
    return {k:z[k] for k in z.files}


def subset(data, idx):
    return {k:(v[idx] if getattr(v,'shape',()) and len(v)==len(data['label']) else v) for k,v in data.items()}


def rule_rows(wave_rule, keys):
    return [{str(k):float(v) for k,v in zip(keys,row)} for row in wave_rule]


def normalize_wave_rules(normalizer, wave_rule, keys, device):
    out={}
    for j,k in enumerate(keys):
        arr=torch.tensor(wave_rule[:,j], dtype=torch.float32, device=device)
        out[str(k)] = normalizer.transform_value(str(k), arr)
    return out


def tensors(X, device):
    return torch.tensor(X, dtype=torch.float32, device=device)


def batch_iter(n, batch_size, shuffle=True, seed=0):
    idx=np.arange(n)
    if shuffle:
        rng=np.random.default_rng(seed); rng.shuffle(idx)
    for s in range(0,n,batch_size): yield idx[s:s+batch_size]


def train_one_fold(data, tr_idx, va_idx, te_idx, args, fold):
    device=torch.device(args.device)
    y=data['label'].astype(int); keys=[str(x) for x in data['wave_rule_keys']]

    # Training-only wavelet PCA.
    wpca=WaveletPCA(variance=args.pca_variance)
    Wtr=wpca.fit_transform(data['wave_raw'][tr_idx]); Wva=wpca.transform(data['wave_raw'][va_idx]); Wte=wpca.transform(data['wave_raw'][te_idx])
    Etr,Eva,Ete=data['eff'][tr_idx],data['eff'][va_idx],data['eff'][te_idx]
    Vtr,Vva,Vte=data['vit'][tr_idx],data['vit'][va_idx],data['vit'][te_idx]

    # Manuscript: concatenate -> MinMaxScaler fitted on training only.
    scaler=MinMaxScaler().fit(np.concatenate([Wtr,Etr,Vtr],axis=1))
    Ftr=scaler.transform(np.concatenate([Wtr,Etr,Vtr],axis=1)).astype(np.float32)
    Fva=scaler.transform(np.concatenate([Wva,Eva,Vva],axis=1)).astype(np.float32)
    Fte=scaler.transform(np.concatenate([Wte,Ete,Vte],axis=1)).astype(np.float32)

    rnorm=RuleFeatureNormalizer().fit(rule_rows(data['wave_rule'][tr_idx],keys))
    model=FeatureLevelHybrid(wave_dim=Wtr.shape[1], hidden_dim=256, heads=4, dropout=args.dropout,
                             fuzzy_cfg=Type2Config(sigma_uncertainty_fraction=args.fuzzy_uncertainty)).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos=max(int((y[tr_idx]==1).sum()),1); neg=max(int((y[tr_idx]==0).sum()),1)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=device))

    tr_t = dict(w=tensors(Wtr,device),e=tensors(Etr,device),v=tensors(Vtr,device),f=tensors(Ftr,device),y=tensors(y[tr_idx],device),
                wr=normalize_wave_rules(rnorm,data['wave_rule'][tr_idx],keys,device))
    va_t = dict(w=tensors(Wva,device),e=tensors(Eva,device),v=tensors(Vva,device),f=tensors(Fva,device),y=tensors(y[va_idx],device),
                wr=normalize_wave_rules(rnorm,data['wave_rule'][va_idx],keys,device))
    te_t = dict(w=tensors(Wte,device),e=tensors(Ete,device),v=tensors(Vte,device),f=tensors(Fte,device),y=tensors(y[te_idx],device),
                wr=normalize_wave_rules(rnorm,data['wave_rule'][te_idx],keys,device))

    best_state=None; best_auc=-np.inf; wait=0
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for bi in batch_iter(len(tr_idx), args.batch_size, True, args.seed+fold*1000+epoch):
            opt.zero_grad()
            wr={k:v[bi] for k,v in tr_t['wr'].items()}
            out=model(tr_t['w'][bi],tr_t['e'][bi],tr_t['v'][bi],wr,fusion_scaled=tr_t['f'][bi])
            loss=loss_fn(out['logits'],tr_t['y'][bi])
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            ov=model(va_t['w'],va_t['e'],va_t['v'],va_t['wr'],fusion_scaled=va_t['f'])
            pv=ov['prob'].cpu().numpy()
        auc=roc_auc_score(y[va_idx],pv) if len(np.unique(y[va_idx]))>1 else 0.5
        if auc>best_auc+1e-5:
            best_auc=auc; wait=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=args.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        otr=model(tr_t['w'],tr_t['e'],tr_t['v'],tr_t['wr'],fusion_scaled=tr_t['f'])
        ova=model(va_t['w'],va_t['e'],va_t['v'],va_t['wr'],fusion_scaled=va_t['f'])
        ote=model(te_t['w'],te_t['e'],te_t['v'],te_t['wr'],fusion_scaled=te_t['f'])

    val_prob=ova['prob'].cpu().numpy(); test_prob=ote['prob'].cpu().numpy()
    threshold=youden_threshold(y[va_idx],val_prob)

    # ART input = [fused, fuzzy score], scaled on training only.
    art_scaler=MinMaxScaler().fit(np.concatenate([otr['fused'].cpu().numpy(),otr['fuzzy_score'].cpu().numpy()[:,None]],axis=1))
    A_tr=art_scaler.transform(np.concatenate([otr['fused'].cpu().numpy(),otr['fuzzy_score'].cpu().numpy()[:,None]],axis=1))
    A_te=art_scaler.transform(np.concatenate([ote['fused'].cpu().numpy(),ote['fuzzy_score'].cpu().numpy()[:,None]],axis=1))
    A_tr=np.clip(A_tr,0,1); A_te=np.clip(A_te,0,1)
    art=FuzzyART(FuzzyARTConfig(vigilance=args.rho,beta=args.beta,alpha=args.art_alpha)).fit(A_tr,y[tr_idx])
    art_pred=art.predict(A_te); art_score=art.decision_score(A_te)

    rows=[]
    fuzzy=ote['fuzzy_score'].cpu().numpy(); flo=ote['fuzzy_lower'].cpu().numpy(); fup=ote['fuzzy_upper'].cpu().numpy()
    for j,idx in enumerate(te_idx):
        rows.append({
            'index':int(idx),'fold':fold,'dataset':str(data['dataset'][idx]),'patient_id':str(data['patient_id'][idx]),
            'y_true':int(y[idx]),'score':float(test_prob[j]),'threshold':float(threshold),
            'fuzzy_score':float(fuzzy[j]),'fuzzy_lower':float(flo[j]),'fuzzy_upper':float(fup[j]),
            'art_score':float(art_score[j]),'art_pred':int(art_pred[j]),'pca_dim':int(Wtr.shape[1]),
            'n_art_categories':int(len(art.weights)),'val_auc':float(best_auc),
        })
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--features',required=True); p.add_argument('--output',required=True)
    p.add_argument('--dataset',choices=['DDSM','INbreast','pooled'],default='pooled')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',type=int,default=12345); p.add_argument('--outer-folds',type=int,default=5); p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--pca-variance',type=float,default=0.98); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--epochs',type=int,default=50); p.add_argument('--patience',type=int,default=5)
    p.add_argument('--dropout',type=float,default=0.1); p.add_argument('--fuzzy-uncertainty',type=float,default=0.20)
    p.add_argument('--rho',type=float,default=0.85); p.add_argument('--beta',type=float,default=0.5); p.add_argument('--art-alpha',type=float,default=1e-3)
    p.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args(); seed_everything(args.seed)
    data=load_npz(args.features)
    if args.dataset!='pooled':
        keep=np.where(data['dataset'].astype(str)==args.dataset)[0]
        # build compact subset while keeping metadata arrays
        data={k:(v[keep] if getattr(v,'shape',()) and len(v)==len(data['label']) else v) for k,v in data.items()}
    y=data['label'].astype(int); groups=data['patient_id'].astype(str)
    outer=StratifiedGroupKFold(n_splits=args.outer_folds,shuffle=True,random_state=args.seed)
    all_rows=[]
    for fold,(trainval_idx,test_idx) in enumerate(outer.split(np.zeros(len(y)),y,groups),1):
        # Inner split: the first inner fold is reserved as validation for early stopping/threshold.
        inner=StratifiedGroupKFold(n_splits=args.inner_folds,shuffle=True,random_state=args.seed+fold)
        rel_tr,rel_va=next(inner.split(np.zeros(len(trainval_idx)),y[trainval_idx],groups[trainval_idx]))
        tr_idx=trainval_idx[rel_tr]; va_idx=trainval_idx[rel_va]
        print(f'fold {fold}: train={len(tr_idx)} val={len(va_idx)} test={len(test_idx)}')
        all_rows.extend(train_one_fold(data,tr_idx,va_idx,test_idx,args,fold))
    pred=pd.DataFrame(all_rows).sort_values('index')
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); pred.to_csv(out,index=False)
    # OOF metrics use each fold's validation-derived threshold.
    yv=pred.y_true.to_numpy(); score=pred.score.to_numpy(); yhat=(score>=pred.threshold.to_numpy()).astype(int)
    # For display, compute threshold-independent AUC + direct threshold metrics.
    from sklearn.metrics import confusion_matrix, roc_auc_score
    tn,fp,fn,tp=confusion_matrix(yv,yhat,labels=[0,1]).ravel()
    summary={
        'dataset':args.dataset,'n':len(pred),'accuracy':(tp+tn)/len(pred),'sensitivity':tp/max(tp+fn,1),'specificity':tn/max(tn+fp,1),
        'precision':tp/max(tp+fp,1),'f1':2*tp/max(2*tp+fp+fn,1),'auc':roc_auc_score(yv,score),
        'auc_ci':bootstrap_ci(yv,score,n_boot=args.bootstrap,seed=args.seed),
    }
    print(json.dumps(summary,indent=2))
    with open(out.with_suffix('.summary.json'),'w') as f: json.dump(summary,f,indent=2)

if __name__=='__main__': main()
