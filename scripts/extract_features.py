#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from mchf.data.preprocess import PreprocessConfig, read_grayscale, preprocess_roi
from mchf.features.wavelet import wavelet_raw_features
from mchf.models.effnet_cbam import EfficientNetV2CBAM
from mchf.models.vit import ViTBranch
from mchf.models.classifiers import EfficientNetCBAMClassifier,ViTClassifier
from mchf.utils.seed import seed_everything

WAVE_RULE_KEYS=['wavelet_energy','wavelet_entropy','wavelet_contrast','wavelet_homogeneity','wavelet_skewness','wavelet_kurtosis']

def load_encoder(branch,ckpt,pretrained,device):
    if branch=='effnet':
        if ckpt:
            m=EfficientNetCBAMClassifier(pretrained=False); obj=torch.load(ckpt,map_location=device); m.load_state_dict(obj['state_dict']); return m.encoder.to(device).eval()
        return EfficientNetV2CBAM(pretrained=pretrained,freeze_backbone=True).to(device).eval()
    if ckpt:
        m=ViTClassifier(pretrained=False); obj=torch.load(ckpt,map_location=device); m.load_state_dict(obj['state_dict']); return m.encoder.to(device).eval()
    return ViTBranch(pretrained=pretrained,freeze_backbone=True).to(device).eval()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--no-pretrained',action='store_true'); p.add_argument('--eff-checkpoint'); p.add_argument('--vit-checkpoint'); p.add_argument('--seed',type=int,default=12345)
    a=p.parse_args(); seed_everything(a.seed); df=pd.read_csv(a.manifest); req={'image_path','label','patient_id','dataset'}
    if req-set(df.columns): raise ValueError(f'Manifest missing {sorted(req-set(df.columns))}')
    dev=torch.device(a.device); eff=load_encoder('effnet',a.eff_checkpoint,not a.no_pretrained,dev); vit=load_encoder('vit',a.vit_checkpoint,not a.no_pretrained,dev); cfg=PreprocessConfig()
    wr_all=[]; rules=[]; images=[]
    for _,r in tqdm(df.iterrows(),total=len(df),desc='preprocess/wavelet'):
        img=preprocess_roi(read_grayscale(r.image_path),cfg); wr,st=wavelet_raw_features(img); wr_all.append(wr); rules.append([st[k] for k in WAVE_RULE_KEYS]); images.append(torch.from_numpy(img[None]).float())
    E=[];V=[]
    with torch.no_grad():
        for s in tqdm(range(0,len(images),a.batch_size),desc='deep features'):
            x=torch.stack(images[s:s+a.batch_size]).to(dev); E.append(eff(x).cpu().numpy()); V.append(vit(x).cpu().numpy())
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,wave_raw=np.asarray(wr_all,np.float32),wave_rule=np.asarray(rules,np.float32),eff=np.concatenate(E).astype(np.float32),vit=np.concatenate(V).astype(np.float32),label=df.label.to_numpy(np.int64),patient_id=df.patient_id.astype(str).to_numpy(),dataset=df.dataset.astype(str).to_numpy(),image_path=df.image_path.astype(str).to_numpy(),wave_rule_keys=np.asarray(WAVE_RULE_KEYS))
    print('saved',out,'with',len(df),'samples')
if __name__=='__main__': main()
