#!/usr/bin/env python
"""Create ROI images from a canonical annotation CSV.

This is dataset-format agnostic because the manuscript does not specify a raw
DDSM/INbreast parser. Convert original annotations to the canonical columns:
image_path,dataset,patient_id,label,annotation_type,x1,y1,x2,y2,cx,cy
where annotation_type is bbox or point. Patient IDs must be correct before ROI
creation to preserve the leakage-prevention protocol.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from mchf.data.preprocess import read_grayscale, apply_clahe_and_smoothing, center_crop_bbox, resize_roi, PreprocessConfig

def main():
    p=argparse.ArgumentParser(); p.add_argument('--annotations',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--manifest',required=True)
    a=p.parse_args(); df=pd.read_csv(a.annotations); out=Path(a.output_dir); cfg=PreprocessConfig(); rows=[]
    req={'image_path','dataset','patient_id','label','annotation_type'}
    if req-set(df.columns): raise ValueError(f'missing {sorted(req-set(df.columns))}')
    for i,r in df.iterrows():
        img=apply_clahe_and_smoothing(read_grayscale(r.image_path),cfg)
        if r.annotation_type=='bbox': bbox=(int(r.x1),int(r.y1),int(r.x2),int(r.y2))
        elif r.annotation_type=='point':
            half=cfg.roi_size//2; bbox=(int(r.cx-half),int(r.cy-half),int(r.cx+half),int(r.cy+half))
        else: raise ValueError(f'unknown annotation_type {r.annotation_type}')
        roi=resize_roi(center_crop_bbox(img,bbox,cfg.roi_size),cfg.roi_size)
        cls='malignant' if int(r.label)==1 else 'benign'; d=out/str(r.dataset)/cls; d.mkdir(parents=True,exist_ok=True)
        path=d/f'{r.patient_id}__roi{i:06d}.png'; Image.fromarray((roi*255).astype('uint8')).save(path)
        rows.append({'image_path':str(path.resolve()),'label':int(r.label),'patient_id':str(r.patient_id),'dataset':str(r.dataset),'roi_id':f'roi{i:06d}'})
    pd.DataFrame(rows).to_csv(a.manifest,index=False); print('saved',len(rows),'ROIs and',a.manifest)
if __name__=='__main__': main()
