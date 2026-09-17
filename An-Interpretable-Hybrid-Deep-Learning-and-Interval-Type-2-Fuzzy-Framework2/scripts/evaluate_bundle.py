#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mchf.hybrid import FeatureLevelHybrid
from mchf.metrics import classification_metrics, bootstrap_ci


def load_features(path: str):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def select_indices(data, manifest=None, split_column='split', split_value=None, indices=None):
    if indices:
        idx = np.load(indices).astype(int)
        return idx

    if manifest and split_value is not None:
        m = pd.read_csv(manifest)
        if split_column not in m.columns:
            raise ValueError(f"Manifest does not contain split column {split_column!r}")
        if 'image_path' not in m.columns:
            raise ValueError("Manifest must contain image_path for split-based selection")
        wanted = set(
            m.loc[m[split_column].astype(str).str.lower() == split_value.lower(), 'image_path']
             .astype(str)
             .tolist()
        )
        if not wanted:
            raise ValueError(f"No manifest rows found for {split_column}={split_value!r}")
        feature_paths = np.asarray(data['image_path']).astype(str)
        idx = np.where(np.isin(feature_paths, list(wanted)))[0]
        if len(idx) == 0:
            raise ValueError("No selected manifest image_path values matched the feature file")
        return idx

    return np.arange(len(data['label']), dtype=int)


def normalize_rule_values(rule_normalizer, rule_array, keys, device):
    out = {}
    for j, key in enumerate(keys):
        t = torch.tensor(rule_array[:, j], dtype=torch.float32, device=device)
        out[key] = rule_normalizer.transform_value(key, t)
    return out


def main():
    p = argparse.ArgumentParser(description='Evaluate a saved final-model bundle without retraining.')
    p.add_argument('--features', required=True)
    p.add_argument('--bundle', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest')
    p.add_argument('--split-column', default='split')
    p.add_argument('--split-value')
    p.add_argument('--indices', help='Optional .npy file containing exact feature indices to evaluate')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=12345)
    a = p.parse_args()

    data = load_features(a.features)
    idx = select_indices(data, a.manifest, a.split_column, a.split_value, a.indices)

    bundle = Path(a.bundle)
    ck = torch.load(bundle / 'feature_model.pt', map_location=a.device)
    with open(bundle / 'preprocessing.pkl', 'rb') as f:
        prep = pickle.load(f)

    device = torch.device(a.device)
    model = FeatureLevelHybrid(wave_dim=int(ck['wave_dim']))
    model.load_state_dict(ck['state_dict'])
    model.to(device).eval()

    keys = [str(x) for x in data['wave_rule_keys']]
    wave = prep['wavelet_pca'].transform(data['wave_raw'][idx])
    eff = data['eff'][idx].astype('float32')
    vit = data['vit'][idx].astype('float32')
    fusion = prep['fusion_scaler'].transform(np.c_[wave, eff, vit]).astype('float32')
    rule_values = normalize_rule_values(prep['rule_normalizer'], data['wave_rule'][idx], keys, device)

    with torch.no_grad():
        out = model(
            torch.tensor(wave, dtype=torch.float32, device=device),
            torch.tensor(eff, dtype=torch.float32, device=device),
            torch.tensor(vit, dtype=torch.float32, device=device),
            rule_values,
            fusion_scaled=torch.tensor(fusion, dtype=torch.float32, device=device),
        )

    score = out['prob'].detach().cpu().numpy()
    fuzzy_score = out['fuzzy_score'].detach().cpu().numpy()
    fused = out['fused'].detach().cpu().numpy()
    y = data['label'][idx].astype(int)
    threshold = float(ck.get('threshold', 0.47))

    art_x = prep['art_scaler'].transform(np.c_[fused, fuzzy_score])
    art_x = np.clip(art_x, 0, 1)
    art_pred = prep['art'].predict(art_x)
    art_score = prep['art'].decision_score(art_x)

    metrics = classification_metrics(y, score, threshold)
    metrics['auc_ci'] = bootstrap_ci(y, score, n_boot=a.bootstrap, seed=a.seed)
    metrics['art_accuracy'] = float(np.mean(art_pred == y))
    metrics['bundle_dataset'] = str(ck.get('dataset', 'unknown'))

    pred = pd.DataFrame({
        'index': idx,
        'image_path': np.asarray(data['image_path'])[idx].astype(str),
        'dataset': np.asarray(data['dataset'])[idx].astype(str),
        'patient_id': np.asarray(data['patient_id'])[idx].astype(str),
        'y_true': y,
        'score': score,
        'threshold': threshold,
        'fuzzy_score': fuzzy_score,
        'art_score': art_score,
        'art_pred': art_pred,
    })

    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output, index=False)
    output.with_suffix('.summary.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
