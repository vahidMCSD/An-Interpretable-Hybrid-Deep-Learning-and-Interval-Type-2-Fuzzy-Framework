#!/usr/bin/env python
"""Component comparison corresponding to the spirit of manuscript Table 5.

The manuscript labels rows as Wavelet+Fuzzy, Wavelet+Fuzzy+ART and Full Hybrid,
but it does not provide enough numerical detail to reconstruct a wavelet-only
fuzzy rule base because Supplementary Table S1 contains CNN/ViT antecedents.
This script therefore does NOT invent a second fuzzy rule base. It reports:
(1) wavelet-only classifier, (2) wavelet+ART, and (3) the full hybrid output,
with the methodological difference explicitly written into the CSV `Notes`
column. Use this for sensitivity/reproducibility, not as a claim that the exact
paper Table 5 protocol was fully specified.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
from mchf.features.wavelet import WaveletPCA
from mchf.art.fuzzy_art import FuzzyART,FuzzyARTConfig

def met(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel();return accuracy_score(y,p),recall_score(y,p),tn/max(tn+fp,1),fp/max(len(y),1)
def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',default='results/table5_components.csv');p.add_argument('--seed',type=int,default=12345);a=p.parse_args();z=np.load(a.features,allow_pickle=True);y=z['label'].astype(int);g=z['patient_id'].astype(str);o=StratifiedGroupKFold(5,shuffle=True,random_state=a.seed);vals={'Wavelet-only classifier':[],'Wavelet + ART':[]}
    for tr,te in o.split(np.zeros(len(y)),y,g):
        wp=WaveletPCA(.98);A=wp.fit_transform(z['wave_raw'][tr]);B=wp.transform(z['wave_raw'][te]);sc=StandardScaler().fit(A);lr=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(A),y[tr]);vals['Wavelet-only classifier'].append(met(y[te],lr.predict(sc.transform(B))));mm=MinMaxScaler().fit(A);art=FuzzyART(FuzzyARTConfig()).fit(np.clip(mm.transform(A),0,1),y[tr]);vals['Wavelet + ART'].append(met(y[te],art.predict(np.clip(mm.transform(B),0,1))))
    rows=[]
    for k,v in vals.items():rows.append({'Configuration':k,'Accuracy (%)':100*np.mean([x[0] for x in v]),'Sensitivity (%)':100*np.mean([x[1] for x in v]),'Specificity (%)':100*np.mean([x[2] for x in v]),'FPR (errors/sample)':np.mean([x[3] for x in v]),'Notes':'Transparent proxy; manuscript does not specify a wavelet-only fuzzy rule base.'})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output,index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
