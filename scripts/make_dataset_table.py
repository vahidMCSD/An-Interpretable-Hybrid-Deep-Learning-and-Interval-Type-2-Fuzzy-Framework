#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from mchf.metrics import dataset_specific_table

def main():
    p=argparse.ArgumentParser(); p.add_argument('predictions',nargs='+'); p.add_argument('--output',required=True); p.add_argument('--bootstrap',type=int,default=1000); p.add_argument('--seed',type=int,default=12345)
    a=p.parse_args(); df=pd.concat([pd.read_csv(x) for x in a.predictions],ignore_index=True)
    # If separate DDSM and INbreast OOF files are supplied, pooled is their concat.
    tab=dataset_specific_table(df,threshold_col='threshold' if 'threshold' in df.columns else None,n_boot=a.bootstrap,seed=a.seed)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); tab.to_csv(a.output,index=False); print(tab.to_string(index=False))
if __name__=='__main__': main()
