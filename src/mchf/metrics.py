from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, brier_score_loss, roc_curve,
)


def youden_threshold(y_true, score) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, score)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    t = float(thresholds[idx])
    if not np.isfinite(t):
        t = 0.5
    return t


def classification_metrics(y_true, score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    auc = roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "accuracy": accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": auc,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "brier": brier_score_loss(y_true, np.clip(score, 0, 1)),
        "threshold": threshold,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n": int(len(y_true)), "prevalence": float(np.mean(y_true)),
    }


def bootstrap_ci(y_true, score, metric="auc", n_boot=1000, confidence=0.95, seed=12345):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, sc = y_true[idx], score[idx]
        if metric == "auc":
            if len(np.unique(yt)) < 2:
                continue
            vals.append(roc_auc_score(yt, sc))
        else:
            raise ValueError(metric)
    if not vals:
        return float("nan"), float("nan")
    alpha = (1 - confidence) / 2
    return float(np.quantile(vals, alpha)), float(np.quantile(vals, 1 - alpha))


def dataset_specific_table(predictions: pd.DataFrame, threshold_col: str | None = None, threshold: float | None = None, n_boot=1000, seed=12345):
    """Build DDSM, INbreast, and pooled table from out-of-fold predictions.

    predictions columns: dataset, y_true, score. If threshold_col is supplied,
    each row can carry the validation-derived fold threshold; otherwise a single
    supplied threshold is used. For publication-grade CV reporting, prefer fold-
    specific thresholds fixed from each validation split.
    """
    required = {"dataset", "y_true", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    rows = []
    groups = [(d, g) for d, g in predictions.groupby("dataset")]
    groups.append(("DDSM + INbreast pooled", predictions))
    for name, g in groups:
        if threshold_col and threshold_col in g.columns:
            pred = (g.score.to_numpy() >= g[threshold_col].to_numpy()).astype(int)
            # metrics helper expects a single threshold, so compute with an equivalent
            # direct path for threshold-dependent metrics while retaining AUC on scores.
            y = g.y_true.to_numpy(dtype=int); s = g.score.to_numpy(dtype=float)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
            m = {
                "accuracy": (tp+tn)/len(y), "sensitivity": tp/max(tp+fn,1),
                "specificity": tn/max(tn+fp,1), "precision": tp/max(tp+fp,1),
                "f1": 2*tp/max(2*tp+fp+fn,1), "auc": (roc_auc_score(y,s) if len(np.unique(y))==2 else np.nan),
            }
        else:
            t = 0.5 if threshold is None else threshold
            m = classification_metrics(g.y_true, g.score, t)
        lo, hi = bootstrap_ci(g.y_true, g.score, "auc", n_boot=n_boot, seed=seed)
        rows.append({
            "Dataset": name,
            "Accuracy (%)": 100*m["accuracy"],
            "Sensitivity (%)": 100*m["sensitivity"],
            "Specificity (%)": 100*m["specificity"],
            "Precision (%)": 100*m["precision"],
            "F1-score (%)": 100*m["f1"],
            "AUC": m["auc"], "AUC CI low": lo, "AUC CI high": hi,
            "AUC (95% CI)": f"{m['auc']:.3f} ({lo:.3f}–{hi:.3f})",
        })
    return pd.DataFrame(rows)
