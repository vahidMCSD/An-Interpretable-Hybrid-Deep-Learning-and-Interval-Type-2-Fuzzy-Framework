#!/usr/bin/env python
"""Build a manifest from already extracted ROI files.

Directory convention:
  root/DDSM/benign/<patient>__<roi>.png
  root/DDSM/malignant/<patient>__<roi>.png
  root/INbreast/benign/<patient>__<roi>.png
  root/INbreast/malignant/<patient>__<roi>.png

For raw DDSM/INbreast annotations, convert to ROIs first using your authorized
copies of the datasets. The paper's leakage-control requirement is patient-level,
so patient_id must be correct before any CV split.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    args=p.parse_args(); root=Path(args.root)
    rows=[]
    for dataset in ("DDSM","INbreast"):
        for cls,label in (("benign",0),("malignant",1)):
            d=root/dataset/cls
            if not d.exists(): continue
            for path in sorted(d.rglob("*")):
                if path.suffix.lower() not in {".png",".jpg",".jpeg",".tif",".tiff",".dcm"}: continue
                patient=path.stem.split("__")[0]
                rows.append({"image_path":str(path.resolve()),"label":label,"patient_id":patient,"dataset":dataset})
    df=pd.DataFrame(rows)
    if df.empty: raise SystemExit("No ROI files found")
    df.to_csv(args.output,index=False)
    print(df.groupby(["dataset","label"]).size())
    print("saved",args.output)
if __name__ == "__main__": main()
