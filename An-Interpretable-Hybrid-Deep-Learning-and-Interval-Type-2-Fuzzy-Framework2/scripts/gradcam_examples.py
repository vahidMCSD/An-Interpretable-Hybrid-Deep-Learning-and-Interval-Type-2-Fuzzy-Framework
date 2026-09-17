#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,torch
from PIL import Image
from mchf.models.classifiers import EfficientNetCBAMClassifier
from mchf.data.preprocess import read_grayscale,preprocess_roi,PreprocessConfig

def last_conv(model):
    layer=None
    for m in model.modules():
        if isinstance(m,torch.nn.Conv2d): layer=m
    if layer is None: raise RuntimeError('No Conv2d layer found')
    return layer

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except Exception:
        # older grad-cam versions do not ship BinaryClassifierOutputTarget
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        BinaryClassifierOutputTarget=None
    m=EfficientNetCBAMClassifier(pretrained=False);obj=torch.load(a.checkpoint,map_location='cpu');m.load_state_dict(obj['state_dict']);m.eval();img=preprocess_roi(read_grayscale(a.image),PreprocessConfig());x=torch.from_numpy(img[None,None]).float();target=last_conv(m.encoder.backbone)
    cam=GradCAMPlusPlus(model=m,target_layers=[target]);targets=None if BinaryClassifierOutputTarget is None else [BinaryClassifierOutputTarget(1)];g=cam(input_tensor=x,targets=targets)[0];rgb=np.repeat(img[...,None],3,axis=-1);vis=show_cam_on_image(rgb.astype(np.float32),g,use_rgb=True);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Image.fromarray(vis).save(a.output);print('saved',a.output)
if __name__=='__main__':main()
