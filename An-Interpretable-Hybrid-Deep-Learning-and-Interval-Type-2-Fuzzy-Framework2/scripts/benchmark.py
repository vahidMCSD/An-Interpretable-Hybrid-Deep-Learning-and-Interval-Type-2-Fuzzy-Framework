#!/usr/bin/env python
from __future__ import annotations
import argparse,time,json
from pathlib import Path
import numpy as np,torch
from mchf.data.preprocess import PreprocessConfig,read_grayscale,preprocess_roi
from mchf.features.wavelet import wavelet_raw_features
from mchf.models.effnet_cbam import EfficientNetV2CBAM
from mchf.models.vit import ViTBranch
from mchf.fuzzy.type2 import IntervalType2Fuzzy,Type2Config
from mchf.art.fuzzy_art import FuzzyART,FuzzyARTConfig

def sync(dev):
    if str(dev).startswith('cuda'): torch.cuda.synchronize()
def timed(fn,dev,n=50,warmup=10):
    for _ in range(warmup): fn(); sync(dev)
    t=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); sync(dev); t.append((time.perf_counter()-s)*1000)
    return float(np.mean(t)),float(np.std(t))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--output',default='results/table11_benchmark.json'); p.add_argument('--iterations',type=int,default=50); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--no-pretrained',action='store_true'); a=p.parse_args(); dev=torch.device(a.device); cfg=PreprocessConfig(); raw=read_grayscale(a.image); roi=preprocess_roi(raw,cfg); x=torch.from_numpy(roi[None,None]).float().to(dev)
    eff=EfficientNetV2CBAM(pretrained=not a.no_pretrained).to(dev).eval(); vit=ViTBranch(pretrained=not a.no_pretrained).to(dev).eval()
    res={};res['preprocessing_ms']=timed(lambda:preprocess_roi(raw,cfg),dev,a.iterations)[0];res['wavelet_ms']=timed(lambda:wavelet_raw_features(roi),dev,a.iterations)[0]
    with torch.no_grad(): res['efficientnet_cbam_ms']=timed(lambda:eff(x),dev,a.iterations)[0];res['vit_ms']=timed(lambda:vit(x),dev,a.iterations)[0]
    # Fuzzy/ART timing on representative normalized inputs.
    vals={k:torch.full((1,),.5,device=dev) for k in ['wavelet_energy','wavelet_entropy','wavelet_contrast','wavelet_homogeneity','wavelet_kurtosis','wavelet_skewness','eff_texture','eff_density','vit_irregular','vit_diffuse','vit_clustered','vit_linear','vit_spotty','vit_complex','vit_scattered']}; fuzzy=IntervalType2Fuzzy(Type2Config(type_reduction='km')).to(dev)
    res['type2_fuzzy_ms']=timed(lambda:fuzzy(vals),dev,a.iterations)[0]; X=np.random.default_rng(0).random((20,257)); y=np.array([0,1]*10); art=FuzzyART(FuzzyARTConfig()).fit(X,y); q=X[0:1]; res['art_ms']=timed(lambda:art.predict(q),'cpu',a.iterations)[0];res['total_component_sum_ms']=sum(v for k,v in res.items() if k.endswith('_ms') and k!='total_component_sum_ms');res['roi_per_second_from_sum']=1000/res['total_component_sum_ms']
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);open(a.output,'w').write(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
