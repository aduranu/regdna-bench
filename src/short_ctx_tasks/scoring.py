"""shared scoring helpers for the zero-shot embedding variant task.

a generative DNA model is scored here by its last-layer embeddings: each
sequence is mean-pooled to one vector, and ref/alt allele vectors are compared.
only the minimal BenchModel interface is used (forward + the embedding hook), so
any wrapper exposing .last_layer_embedding can be evaluated.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

# pooling and metrics run in fp32 even if the model samples in fp16, for stability
METRIC_DTYPE = torch.float32


def extract_embeddings(model, sequences, batch_size=256):
    # mean-pool last-layer embeddings over length. the model must have its
    # extraction hook installed and expose .last_layer_embedding as (B, L, hidden).
    model.add_last_layer_embedding_extraction()

    pooled = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]

        model.forward(batch)
        embedding = model.last_layer_embedding

        pooled.append(embedding.mean(dim=1).to(METRIC_DTYPE).cpu())

    return torch.cat(pooled, dim=0)


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
    # are the (unsigned) cosine distance.
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)

    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }


def extract_allele_score_difference(model, windows, offsets, ref_tokens, alt_tokens,
                                    batch_size=256):
    # likelihood-based zero-shot score for a generative DNA LM that exposes
    # predict_logits. for D3 (uniform SEDD) predict_logits returns the log
    # concrete score (B,L,V): score[i,y] ~ log( p_t(x with i->y) / p_t(x) ), the
    # log-ratio of full-sequence marginals. feeding the reference window, the alt
    # readout score[off,alt] is directly log( p_t(x_alt)/p_t(x_ref) ) = the allele
    # log-likelihood difference; the ref slot is zeroed by the model, so taking
    # score[off,alt]-score[off,ref] is exact and robust (no per-base denominator).
    offsets = np.asarray(offsets, dtype=np.int64)
    ref_tokens = np.asarray(ref_tokens, dtype=np.int64)
    alt_tokens = np.asarray(alt_tokens, dtype=np.int64)

    diffs = []
    for start in range(0, len(windows), batch_size):
        end = start + batch_size

        score = model.predict_logits(windows[start:end]).to(METRIC_DTYPE).cpu()

        rows = torch.arange(score.shape[0])
        at_pos = score[rows, torch.as_tensor(offsets[start:end])]

        alt = at_pos[rows, torch.as_tensor(alt_tokens[start:end])]
        ref = at_pos[rows, torch.as_tensor(ref_tokens[start:end])]

        diffs.append((alt - ref).to(torch.float64).numpy())

    return np.concatenate(diffs)
