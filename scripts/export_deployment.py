#!/usr/bin/env python
"""Export deep branches to ONNX and optionally build TensorRT engines with trtexec."""
from __future__ import annotations
import argparse,subprocess,shutil
from pathlib import Path
import torch
from mchf.models.classifiers import EfficientNetCBAMClassifier,ViTClassifier

def export(branch,checkpoint,out):
    m=EfficientNetCBAMClassifier(pretrained=False) if branch=='effnet' else ViTClassifier(pretrained=False);obj=torch.load(checkpoint,map_location='cpu');m.load_state_dict(obj['state_dict']);m.eval();dummy=torch.randn(1,1,224,224);torch.onnx.export(m,dummy,out,input_names=['image'],output_names=['logit'],dynamic_axes={'image':{0:'batch'},'logit':{0:'batch'}},opset_version=17)
def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=['effnet','vit'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--onnx',required=True);p.add_argument('--engine');a=p.parse_args();Path(a.onnx).parent.mkdir(parents=True,exist_ok=True);export(a.branch,a.checkpoint,a.onnx);print('saved',a.onnx)
    if a.engine:
        exe=shutil.which('trtexec')
        if not exe: raise SystemExit('trtexec not found; ONNX export succeeded, TensorRT engine not built')
        subprocess.check_call([exe,f'--onnx={a.onnx}',f'--saveEngine={a.engine}','--fp16']);print('saved',a.engine)
if __name__=='__main__':main()
