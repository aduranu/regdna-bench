"""pure scoring metrics shared across tasks (no model, no torch)."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def correlation_metrics(predicted, measured):
    # pearson + spearman between a predicted score and a continuous target.
    p = np.asarray(predicted, dtype=np.float64).ravel()
    m = np.asarray(measured, dtype=np.float64).ravel()

    return {
        "pearson": float(pearsonr(p, m)[0]),
        "spearman": float(spearmanr(p, m)[0]),
    }


def classification_metrics(scores, labels):
    # auroc/auprc for significant-vs-background variants; labels are 0/1, scores
    # are the (unsigned) effect score.
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)

    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }
