#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mchf.data.preprocess import PreprocessConfig, read_grayscale, preprocess_roi
from mchf.features.wavelet import wavelet_raw_features
from mchf.models.effnet_cbam import EfficientNetV2CBAM
from mchf.models.vit import ViTBranch
from mchf.utils.seed import seed_everything

WAVE_RULE_KEYS = [
    "wavelet_energy", "wavelet_entropy", "wavelet_contrast",
    "wavelet_homogeneity", "wavelet_skewness", "wavelet_kurtosis",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    seed_everything(args.seed)

    df = pd.read_csv(args.manifest)
    required = {"image_path", "label", "patient_id", "dataset"}
    if missing := (required - set(df.columns)):
        raise ValueError(f"Manifest missing {sorted(missing)}")

    device = torch.device(args.device)
    eff = EfficientNetV2CBAM(pretrained=not args.no_pretrained, freeze_backbone=True).to(device).eval()
    vit = ViTBranch(pretrained=not args.no_pretrained, freeze_backbone=True).to(device).eval()
    cfg = PreprocessConfig()

    wave_raw_all, wave_rules_all, eff_all, vit_all = [], [], [], []
    # Wavelet + image preprocessing is deterministic and done sample-wise.
    images = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="preprocess/wavelet"):
        img = preprocess_roi(read_grayscale(row.image_path), cfg)
        wr, stats = wavelet_raw_features(img)
        wave_raw_all.append(wr)
        wave_rules_all.append([stats[k] for k in WAVE_RULE_KEYS])
        images.append(torch.from_numpy(img[None]).float())

    with torch.no_grad():
        for start in tqdm(range(0, len(images), args.batch_size), desc="deep features"):
            x = torch.stack(images[start:start+args.batch_size]).to(device)
            eff_all.append(eff(x).cpu().numpy())
            vit_all.append(vit(x).cpu().numpy())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        wave_raw=np.asarray(wave_raw_all, dtype=np.float32),
        wave_rule=np.asarray(wave_rules_all, dtype=np.float32),
        eff=np.concatenate(eff_all).astype(np.float32),
        vit=np.concatenate(vit_all).astype(np.float32),
        label=df.label.to_numpy(dtype=np.int64),
        patient_id=df.patient_id.astype(str).to_numpy(),
        dataset=df.dataset.astype(str).to_numpy(),
        image_path=df.image_path.astype(str).to_numpy(),
        wave_rule_keys=np.asarray(WAVE_RULE_KEYS),
    )
    print(f"saved {out} with {len(df)} samples")

if __name__ == "__main__":
    main()
