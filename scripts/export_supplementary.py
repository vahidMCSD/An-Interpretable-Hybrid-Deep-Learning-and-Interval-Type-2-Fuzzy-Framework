#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import argparse,pandas as pd
from mchf.fuzzy.rules import RULES

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--output-dir',default='results/supplementary');a=p.parse_args();o=Path(a.output_dir);o.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in RULES: rows.append({'Rule ID':r.rule_id,'Antecedent (IF…)':' AND '.join(f'{x.feature} = {x.term}' for x in r.antecedents),'Consequent (THEN…)':r.consequent})
    pd.DataFrame(rows).to_csv(o/'Table_S1_rules.csv',index=False)
    if a.manifest:
        d=pd.read_csv(a.manifest);s=d.groupby('dataset').agg(Included_ROIs=('label','size'),Malignant_ROIs=('label','sum')).reset_index();s['Benign_ROIs']=s.Included_ROIs-s.Malignant_ROIs;s.to_csv(o/'Table_S2_counts_from_manifest.csv',index=False)
    print('saved supplementary exports to',o)
if __name__=='__main__':main()
