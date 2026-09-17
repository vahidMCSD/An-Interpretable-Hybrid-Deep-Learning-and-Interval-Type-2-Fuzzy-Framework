#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output',default='results/supplementary_S4.csv');a=p.parse_args();d=pd.read_csv(a.predictions);rows=[]
    for f,g in d.groupby('fold'):
        cats=int(g.n_art_categories.iloc[0]) if 'n_art_categories' in g else np.nan
        acc=float(np.mean(g.art_pred==g.y_true)) if {'art_pred','y_true'}<=set(g.columns) else np.nan
        rows.append({'Outer Fold':int(f),'Dataset used (train+val)':'Pooled (after holdout removal)','Categories formed (count)':cats,'Majority-vote mapping accuracy':acc})
    pd.DataFrame(rows).to_csv(a.output,index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
