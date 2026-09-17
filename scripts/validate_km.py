#!/usr/bin/env python
from __future__ import annotations
import argparse,json
import numpy as np,torch
from mchf.fuzzy.type2 import IntervalType2Fuzzy,Type2Config

def main():
    p=argparse.ArgumentParser(); p.add_argument('--samples',type=int,default=1000); p.add_argument('--seed',type=int,default=12345); p.add_argument('--output',default='results/km_validation.json'); a=p.parse_args(); rng=np.random.default_rng(a.seed)
    # Random normalized linguistic variables, compare exact KM vs midpoint approximation.
    vals={k:torch.tensor(rng.random(a.samples),dtype=torch.float32) for k in ['wavelet_energy','wavelet_entropy','wavelet_contrast','wavelet_homogeneity','wavelet_kurtosis','wavelet_skewness','eff_texture','eff_density','vit_irregular','vit_diffuse','vit_clustered','vit_linear','vit_spotty','vit_complex','vit_scattered']}
    exact=IntervalType2Fuzzy(Type2Config(type_reduction='km')); approx=IntervalType2Fuzzy(Type2Config(type_reduction='midpoint'))
    with torch.no_grad(): e=exact(vals)[0].numpy(); m=approx(vals)[0].numpy()
    res={'n':a.samples,'mean_absolute_difference':float(np.mean(np.abs(e-m))),'max_absolute_difference':float(np.max(np.abs(e-m)))}
    from pathlib import Path; Path(a.output).parent.mkdir(parents=True,exist_ok=True); open(a.output,'w').write(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__':main()
