import argparse, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from tqdm import tqdm

from config import Config
from data import ROIDataset, preprocess_roi
from wavelet import extract_wavelet_features
from model import HybridDeepModel
from fuzzy import IntervalType2FuzzyHead
from art import FuzzyART

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def collect_wavelet_raw(csv_path, split):
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split]
    out = []
    for p in tqdm(df["path"], desc=f"wavelet:{split}"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = preprocess_roi(img)
        out.append(extract_wavelet_features(img))
    return np.stack(out)

def fit_pca(train_raw, variance=0.95, target_dim=20):
    # Fit by variance threshold, as stated in the manuscript.
    pca = PCA(n_components=variance, whiten=False, svd_solver="full")
    z = pca.fit_transform(train_raw)

    # Paper reports a 20-D final wavelet representation.
    # If the exact data yields another dimension, truncate/pad only for
    # architecture compatibility and report it clearly.
    return pca

def pca20(pca, x):
    z = pca.transform(x)
    if z.shape[1] >= 20:
        return z[:, :20].astype(np.float32)
    pad = np.zeros((z.shape[0], 20-z.shape[1]), np.float32)
    return np.concatenate([z.astype(np.float32), pad], axis=1)

class FullNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep = HybridDeepModel()
        self.fuzzy = IntervalType2FuzzyHead(256)
        self.classifier = nn.Linear(257, 1)

    def forward(self, image, wave20):
        fused, _, _ = self.deep(image, wave20)
        fuzzy_score = self.fuzzy(fused).unsqueeze(1)
        logits = self.classifier(torch.cat([fused, fuzzy_score], dim=1)).squeeze(1)
        return logits, fused, fuzzy_score.squeeze(1)

def run_epoch(model, loader, wave20, opt, device, train=True):
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    scores, ys, fused_all, fuzzy_all = [], [], [], []
    offset = 0

    for img, y in loader:
        n = len(y)
        w = torch.from_numpy(wave20[offset:offset+n]).to(device)
        offset += n

        img, y = img.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits, fused, fuzzy = model(img, w)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

        losses.append(loss.item())
        scores.extend(torch.sigmoid(logits).detach().cpu().numpy())
        ys.extend(y.detach().cpu().numpy())
        fused_all.append(fused.detach().cpu().numpy())
        fuzzy_all.extend(fuzzy.detach().cpu().numpy())

    return (
        float(np.mean(losses)),
        np.asarray(ys),
        np.asarray(scores),
        np.concatenate(fused_all),
        np.asarray(fuzzy_all),
    )

def metrics(y, score, threshold=0.47):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    spec = tn / max(1, tn+fp)
    return {
        "accuracy": accuracy_score(y,pred),
        "sensitivity": recall_score(y,pred, zero_division=0),
        "specificity": spec,
        "precision": precision_score(y,pred, zero_division=0),
        "f1": f1_score(y,pred, zero_division=0),
        "auc": roc_auc_score(y,score),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV: path,label,patient_id,split")
    ap.add_argument("--out", default="runs/paper_model")
    args = ap.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Wavelet + PCA: fit ONLY on training data
    tr_raw = collect_wavelet_raw(args.csv, "train")
    va_raw = collect_wavelet_raw(args.csv, "val")
    te_raw = collect_wavelet_raw(args.csv, "test")

    pca = fit_pca(tr_raw, cfg.pca_variance)
    tr_w = pca20(pca, tr_raw)
    va_w = pca20(pca, va_raw)
    te_w = pca20(pca, te_raw)

    train_ds = ROIDataset(args.csv, "train")
    val_ds   = ROIDataset(args.csv, "val")
    test_ds  = ROIDataset(args.csv, "test")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)

    model = FullNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_auc = -1
    bad = 0
    for epoch in range(1, cfg.epochs+1):
        tr = run_epoch(model, train_loader, tr_w, opt, device, train=True)
        va = run_epoch(model, val_loader, va_w, opt, device, train=False)
        va_auc = roc_auc_score(va[1], va[2])

        print(f"epoch={epoch:02d} train_loss={tr[0]:.4f} val_auc={va_auc:.4f}")
        if va_auc > best_auc:
            best_auc = va_auc
            bad = 0
            torch.save(model.state_dict(), out/"best.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(torch.load(out/"best.pt", map_location=device))
    te = run_epoch(model, test_loader, te_w, opt, device, train=False)
    print(metrics(te[1], te[2], cfg.final_threshold))

    # Optional ART stage on [fused_embedding, fuzzy_score]
    # Normalize to [0,1] for fuzzy ART.
    tr = run_epoch(model, train_loader, tr_w, opt, device, train=False)
    scaler = MinMaxScaler()
    art_train_x = scaler.fit_transform(np.c_[tr[3], tr[4]])
    art_test_x = scaler.transform(np.c_[te[3], te[4]])

    art = FuzzyART(rho=cfg.art_rho, beta=cfg.art_beta)
    art.fit(art_train_x, tr[1].astype(int))
    art_pred = art.predict(art_test_x)
    print("ART accuracy:", accuracy_score(te[1], art_pred))

if __name__ == "__main__":
    main()
