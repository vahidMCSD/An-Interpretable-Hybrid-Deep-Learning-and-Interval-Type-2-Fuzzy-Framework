#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd, numpy as np
from mchf.plots import plot_dataset_roc,plot_confusion,plot_reliability

def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',required=True); p.add_argument('--output-dir',required=True)
    a=p.parse_args(); d=pd.read_csv(a.predictions); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    plot_dataset_roc(d,out/'figure3_roc.png')
    pred=(d.score.to_numpy()>=d.threshold.to_numpy()).astype(int) if 'threshold' in d else (d.score.to_numpy()>=.5).astype(int)
    plot_confusion(d.y_true,pred,out/'figure4_confusion_matrix.png'); plot_reliability(d.y_true,d.score,out/'reliability_diagram.png')
    print('saved figures to',out)
if __name__=='__main__': main()
