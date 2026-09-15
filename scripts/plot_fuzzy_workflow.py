#!/usr/bin/env python
from pathlib import Path
import argparse,matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/figures/figure2_fuzzy_workflow.png');a=p.parse_args();fig,ax=plt.subplots(figsize=(10,3));ax.axis('off');labels=['Normalized branch features','12-rule interval firing','Aggregate lower/upper firing','Karnik–Mendel type reduction','Malignancy score s ∈ [0,1]'];xs=[.08,.28,.50,.72,.92]
    for x,l in zip(xs,labels):ax.text(x,.5,l,ha='center',va='center',bbox=dict(boxstyle='round',fc='white'),transform=ax.transAxes)
    for x1,x2 in zip(xs[:-1],xs[1:]):ax.annotate('',xy=(x2-.08,.5),xytext=(x1+.08,.5),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    fig.tight_layout();Path(a.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=300,bbox_inches='tight');plt.close(fig);print('saved',a.output)
if __name__=='__main__':main()
