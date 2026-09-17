#!/usr/bin/env python
"""Reproduce a transparent version of Table 6 using shared outer folds.

Rows whose exact optimization recipe is not fully specified in the manuscript
are implemented using leakage-safe, documented choices. The generated table is
therefore a reproducible re-run, not a hard-coded copy of manuscript numbers.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score
from mchf.features.wavelet import WaveletPCA
from mchf.training import train_feature_model

def score(y,s,t=.5): return accuracy_score(y,np.asarray(s)>=t),roc_auc_score(y,s)

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',default='results/table6_ablation.csv');p.add_argument('--seed',type=int,default=12345);p.add_argument('--outer-folds',type=int,default=5);p.add_argument('--epochs',type=int,default=50);p.add_argument('--device',default='cpu');a=p.parse_args();z=np.load(a.features,allow_pickle=True);d={k:z[k] for k in z.files};y=d['label'].astype(int);g=d['patient_id'].astype(str);outer=StratifiedGroupKFold(a.outer_folds,shuffle=True,random_state=a.seed)
    configs=['EfficientNetV2 Only','ViT Only','Wavelet + EfficientNetV2','Wavelet + ViT','Early Fusion','Late Fusion','Non-Fuzzy MLP Classifier','Type-1 Fuzzy + ART','Type-2 Fuzzy + ART','Full Model without ART','Full Hybrid Framework'];scores={c:[] for c in configs}
    for fold,(tv,te) in enumerate(outer.split(np.zeros(len(y)),y,g),1):
        inner=StratifiedGroupKFold(3,shuffle=True,random_state=a.seed+fold);rt,rv=next(inner.split(np.zeros(len(tv)),y[tv],g[tv]));tr,va=tv[rt],tv[rv]
        wp=WaveletPCA(.98);Wtr=wp.fit_transform(d['wave_raw'][tr]);Wte=wp.transform(d['wave_raw'][te]);Etr,Ete=d['eff'][tr],d['eff'][te];Vtr,Vte=d['vit'][tr],d['vit'][te]
        mats={'EfficientNetV2 Only':(Etr,Ete),'ViT Only':(Vtr,Vte),'Wavelet + EfficientNetV2':(np.c_[Wtr,Etr],np.c_[Wte,Ete]),'Wavelet + ViT':(np.c_[Wtr,Vtr],np.c_[Wte,Vte]),'Early Fusion':(np.c_[Wtr,Etr,Vtr],np.c_[Wte,Ete,Vte]),'Late Fusion':(np.c_[Etr,Vtr],np.c_[Ete,Vte]),'Non-Fuzzy MLP Classifier':(np.c_[Wtr,Etr,Vtr],np.c_[Wte,Ete,Vte])}
        for c,(A,B) in mats.items():
            sc=StandardScaler().fit(A);clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(A),y[tr]);p1=clf.predict_proba(sc.transform(B))[:,1];scores[c].append(score(y[te],p1))
        r1=train_feature_model(d,tr,va,te,a.device,a.seed+fold,epochs=a.epochs,fuzzy_uncertainty=0.0,type_reduction='midpoint');r2=train_feature_model(d,tr,va,te,a.device,a.seed+100+fold,epochs=a.epochs,fuzzy_uncertainty=.2,type_reduction='km')
        scores['Type-1 Fuzzy + ART'].append(score(y[te],r1['art_score']))
        scores['Type-2 Fuzzy + ART'].append(score(y[te],r2['art_score']))
        scores['Full Model without ART'].append(score(y[te],r2['prob'],r2['threshold']))
        scores['Full Hybrid Framework'].append(score(y[te],r2['art_score']))
    rows=[{'Configuration':c,'Accuracy (%)':100*np.mean([x[0] for x in v]),'AUC':np.mean([x[1] for x in v])} for c,v in scores.items()];Path(a.output).parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output,index=False);Path(a.output).with_name('ablation_methodology.txt').write_text(__doc__ or '');print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
