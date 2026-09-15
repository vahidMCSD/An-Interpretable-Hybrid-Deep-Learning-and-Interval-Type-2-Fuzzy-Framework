#!/usr/bin/env python
"""Build the reviewer-requested dataset-specific performance table.

Two transparent modes are supported:

1) Reported-results mode: regenerate the exact revised-manuscript table from
   config/reviewer_reported_results.yaml. This preserves provenance and avoids
   inventing per-sample predictions from summary statistics.

2) Prediction mode: compute the same table directly from out-of-fold / external
   prediction CSV files created by this repository. This is the preferred mode
   when the raw/local datasets have been run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score

from mchf.metrics import bootstrap_ci

COLS = [
    "Evaluation protocol",
    "Dataset",
    "Accuracy (%)",
    "Sensitivity (%)",
    "Specificity (%)",
    "AUC (95% CI)",
]


def _summary_from_predictions(path: str | Path, protocol: str, dataset_label: str,
                              n_boot: int = 1000, seed: int = 12345) -> dict:
    df = pd.read_csv(path)
    required = {"y_true", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    y = df["y_true"].to_numpy(dtype=int)
    score = df["score"].to_numpy(dtype=float)
    if "threshold" in df.columns:
        threshold = df["threshold"].to_numpy(dtype=float)
        pred = (score >= threshold).astype(int)
    else:
        pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(len(y), 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    auc = roc_auc_score(y, score)
    lo, hi = bootstrap_ci(y, score, n_boot=n_boot, seed=seed)
    return {
        "Evaluation protocol": protocol,
        "Dataset": dataset_label,
        "Accuracy (%)": round(100 * acc, 1),
        "Sensitivity (%)": round(100 * sens, 1),
        "Specificity (%)": round(100 * spec, 1),
        "AUC (95% CI)": f"{auc:.3f} ({lo:.3f}–{hi:.3f})",
    }


def load_reported_config(path: str | Path) -> pd.DataFrame:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rows = []
    for r in obj["rows"]:
        rows.append({
            "Evaluation protocol": r["evaluation_protocol"],
            "Dataset": r["dataset"],
            "Accuracy (%)": f'{float(r["accuracy_pct"]):.1f}',
            "Sensitivity (%)": f'{float(r["sensitivity_pct"]):.1f}',
            "Specificity (%)": f'{float(r["specificity_pct"]):.1f}',
            "AUC (95% CI)": (
                f'{float(r["auc"]):.3f} '
                f'({float(r["auc_ci_low"]):.3f}–{float(r["auc_ci_high"]):.3f})'
            ),
        })
    return pd.DataFrame(rows, columns=COLS)


def build_from_predictions(ddsm: str, inbreast: str, pooled: str,
                           ddsm_to_inbreast: str, inbreast_to_ddsm: str,
                           n_boot: int, seed: int) -> pd.DataFrame:
    specs: Iterable[tuple[str, str, str]] = [
        (ddsm, "Within-dataset 5-fold CV", "DDSM"),
        (inbreast, "Within-dataset 5-fold CV", "INbreast"),
        (ddsm_to_inbreast, "External validation", "DDSM → INbreast"),
        (inbreast_to_ddsm, "External validation", "INbreast → DDSM"),
        (pooled, "Pooled evaluation", "DDSM + INbreast"),
    ]
    rows = [
        _summary_from_predictions(p, protocol, label, n_boot=n_boot, seed=seed + i)
        for i, (p, protocol, label) in enumerate(specs)
    ]
    return pd.DataFrame(rows, columns=COLS)


def write_markdown(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def write_latex(df: pd.DataFrame, path: str | Path) -> None:
    # escape=False preserves arrows and en dashes; suitable for direct manuscript editing.
    latex = df.to_latex(index=False, escape=False)
    Path(path).write_text(latex, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reported-config")
    p.add_argument("--ddsm")
    p.add_argument("--inbreast")
    p.add_argument("--pooled")
    p.add_argument("--ddsm-to-inbreast")
    p.add_argument("--inbreast-to-ddsm")
    p.add_argument("--output", required=True)
    p.add_argument("--markdown")
    p.add_argument("--latex")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args()

    if a.reported_config:
        df = load_reported_config(a.reported_config)
    else:
        needed = [a.ddsm, a.inbreast, a.pooled, a.ddsm_to_inbreast, a.inbreast_to_ddsm]
        if any(v is None for v in needed):
            p.error("prediction mode requires --ddsm, --inbreast, --pooled, --ddsm-to-inbreast and --inbreast-to-ddsm")
        df = build_from_predictions(
            a.ddsm, a.inbreast, a.pooled,
            a.ddsm_to_inbreast, a.inbreast_to_ddsm,
            a.bootstrap, a.seed,
        )

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if a.markdown:
        write_markdown(df, a.markdown)
    if a.latex:
        write_latex(df, a.latex)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
