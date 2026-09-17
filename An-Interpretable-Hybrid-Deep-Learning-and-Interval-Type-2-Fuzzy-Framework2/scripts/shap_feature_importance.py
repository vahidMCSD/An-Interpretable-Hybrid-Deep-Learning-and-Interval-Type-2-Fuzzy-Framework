#!/usr/bin/env python
"""SHAP analysis of the trained feature-level hybrid model.

The explainer operates on the original per-ROI feature vector
[36 wavelet raw | 128 EfficientNet | 256 ViT | 6 wavelet-rule statistics].
It then invokes the exact saved preprocessing objects before model prediction.
"""
from __future__ import annotations
import argparse,pickle
from pathlib import Path
import numpy as np,torch
from mchf.hybrid import FeatureLevelHybrid

def main():
    p=argparse.ArgumentParser(); p.add_argument('--features',required=True); p.add_argument('--bundle',required=True); p.add_argument('--output',default='results/figures/figure7_shap.png'); p.add_argument('--background',type=int,default=50); p.add_argument('--samples',type=int,default=100); p.add_argument('--seed',type=int,default=12345); a=p.parse_args()
    try: import shap
    except Exception as e: raise SystemExit('Install shap>=0.45') from e
    import matplotlib.pyplot as plt
    z=np.load(a.features,allow_pickle=True); bundle=Path(a.bundle); ck=torch.load(bundle/'feature_model.pt',map_location='cpu'); prep=pickle.load(open(bundle/'preprocessing.pkl','rb')); keys=[str(x) for x in z['wave_rule_keys']]
    m=FeatureLevelHybrid(wave_dim=int(ck['wave_dim']));m.load_state_dict(ck['state_dict']);m.eval()
    X=np.c_[z['wave_raw'],z['eff'],z['vit'],z['wave_rule']].astype('float32'); rng=np.random.default_rng(a.seed); bg=X[rng.choice(len(X),min(a.background,len(X)),replace=False)]; sm=X[rng.choice(len(X),min(a.samples,len(X)),replace=False)]
    def pred(A):
        A=np.asarray(A,'float32'); wr=A[:,:36]; eff=A[:,36:164]; vit=A[:,164:420]; rvals=A[:,420:426]; wave=prep['wavelet_pca'].transform(wr); fusion=prep['fusion_scaler'].transform(np.c_[wave,eff,vit]).astype('float32'); vals={}
        for j,k in enumerate(keys): vals[k]=prep['rule_normalizer'].transform_value(k,torch.tensor(rvals[:,j],dtype=torch.float32))
        with torch.no_grad(): return m(torch.tensor(wave),torch.tensor(eff),torch.tensor(vit),vals,fusion_scaled=torch.tensor(fusion))['prob'].numpy()
    explainer=shap.Explainer(pred,bg,algorithm='permutation'); sv=explainer(sm,max_evals=max(2*X.shape[1]+1,900)); names=[f'wavelet_raw_{i+1}' for i in range(36)]+[f'EffNet_{i+1}' for i in range(128)]+[f'ViT_{i+1}' for i in range(256)]+keys
    shap.summary_plot(sv.values,sm,feature_names=names,show=False,max_display=20);Path(a.output).parent.mkdir(parents=True,exist_ok=True);plt.tight_layout();plt.savefig(a.output,dpi=300,bbox_inches='tight');plt.close();print('saved',a.output)
if __name__=='__main__':main()
