"""model-facing extraction for the VEP tasks.

these are the only helpers that touch a model. pooling and the score readout run
in fp32 even if the model samples in lower precision, for stability.
"""

from __future__ import annotations

import numpy as np
import torch

METRIC_DTYPE = torch.float32


def extract_embeddings(model, windows, batch_size=256):
    # mean-pool last-layer embeddings over length. model.embed returns (B,L,H).
    pooled = []
    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size]

        embedding = model.embed(batch)

        pooled.append(embedding.mean(dim=1).to(METRIC_DTYPE).cpu())

    return torch.cat(pooled, dim=0)


def extract_allele_score_difference(model, windows, offsets, ref_base, alt_base,
                                    batch_size=256):
    # likelihood-based zero-shot score for a model exposing predict_logits. for a
    # uniform-SEDD model predict_logits returns the log concrete score (B,L,V):
    # score[i,y] ~ log( p_t(x with i->y) / p_t(x) ), the log-ratio of full-sequence
    # marginals. feeding the reference window, score[off,alt] is directly
    # log( p_t(x_alt)/p_t(x_ref) ); the ref slot is zeroed by the model, so
    # score[off,alt]-score[off,ref] is exact (no per-base denominator). CANON base
    # ids are mapped to logit columns by the model's vocab_index.
    offsets = np.asarray(offsets, dtype=np.int64)
    ref_cols = np.asarray([model.vocab_index(int(b)) for b in ref_base], dtype=np.int64)
    alt_cols = np.asarray([model.vocab_index(int(b)) for b in alt_base], dtype=np.int64)

    diffs = []
    for start in range(0, len(windows), batch_size):
        end = start + batch_size

        score = model.predict_logits(windows[start:end]).to(METRIC_DTYPE).cpu()

        rows = torch.arange(score.shape[0])
        at_pos = score[rows, torch.as_tensor(offsets[start:end])]

        alt = at_pos[rows, torch.as_tensor(alt_cols[start:end])]
        ref = at_pos[rows, torch.as_tensor(ref_cols[start:end])]

        diffs.append((alt - ref).to(torch.float64).numpy())

    return np.concatenate(diffs)
