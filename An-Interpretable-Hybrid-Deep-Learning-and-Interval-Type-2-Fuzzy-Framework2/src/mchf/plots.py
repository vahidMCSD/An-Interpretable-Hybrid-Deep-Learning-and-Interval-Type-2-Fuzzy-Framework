from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve


def plot_dataset_roc(predictions, out_path, ci=None):
    fig,ax=plt.subplots(figsize=(7,6))
    for name,g in list(predictions.groupby('dataset')) + [('Pooled',predictions)]:
        y=g.y_true.to_numpy(); s=g.score.to_numpy(); fpr,tpr,_=roc_curve(y,s); auc=roc_auc_score(y,s)
        ax.plot(fpr,tpr,label=f'{name} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'--',linewidth=1); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right'); fig.tight_layout(); Path(out_path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out_path,dpi=300); plt.close(fig)


def plot_confusion(y_true, y_pred, out_path, labels=('Benign','Malignant')):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1]); fig,ax=plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,cmap='Blues',colorbar=False)
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out_path,dpi=300); plt.close(fig)


def plot_reliability(y_true, score, out_path, n_bins=10):
    prob_true,prob_pred=calibration_curve(y_true,score,n_bins=n_bins,strategy='quantile')
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot(prob_pred,prob_true,marker='o'); ax.plot([0,1],[0,1],'--')
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed malignant fraction'); fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out_path,dpi=300); plt.close(fig)
