#!/usr/bin/env python
from __future__ import annotations
import argparse
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='results/synthetic_features.npz'); p.add_argument('--n',type=int,default=240); p.add_argument('--seed',type=int,default=7)
    a=p.parse_args(); rng=np.random.default_rng(a.seed); n=a.n
    dataset=np.where(np.arange(n)%2==0,'DDSM','INbreast')
    patient_id=np.asarray([f'{dataset[i]}_P{i//2:04d}' for i in range(n)])
    latent=rng.normal(size=n); label=(latent+rng.normal(scale=.8,size=n)>0).astype(int)
    wave=rng.normal(size=(n,36)).astype('float32'); wave[:,0]+=2*label; wave[:,1]+=label
    eff=rng.normal(size=(n,128)).astype('float32'); eff[:,:8]+=label[:,None]*.7
    vit=rng.normal(size=(n,256)).astype('float32'); vit[:,:8]+=label[:,None]*.7
    wr=np.column_stack([
      np.exp(latent)+1, np.abs(latent)+.5, np.abs(latent)*2+.2, 1/(1+np.abs(latent)), latent,
      latent**2-1,
    ]).astype('float32')
    keys=np.asarray(['wavelet_energy','wavelet_entropy','wavelet_contrast','wavelet_homogeneity','wavelet_skewness','wavelet_kurtosis'])
    from pathlib import Path
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,wave_raw=wave,wave_rule=wr,eff=eff,vit=vit,label=label,patient_id=patient_id,dataset=dataset,image_path=np.asarray(['synthetic']*n),wave_rule_keys=keys)
    print('saved',a.output)
if __name__=='__main__': main()
