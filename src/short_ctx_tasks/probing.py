"""probing task: train a lightweight probe on frozen last-layer embeddings.

measures how linearly decodable a benchmark model's representations are for a
downstream activity target (e.g. lentiMPRA expression). model-agnostic: it only
uses the BenchModel embedding-extraction hook, so any wrapper implementing
add_last_layer_embedding_extraction + exposing .last_layer_embedding can be
probed (see examples/d3/wrapper.py).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# probe inputs/targets are fit in fp32; embeddings are pooled in fp32 too
EMBED_DTYPE = torch.float32


def extract_embeddings(model, sequences, batch_size=256):
    # run the model in batches and collect mean-pooled last-layer embeddings.
    # the model must have its extraction hook installed and expose
    # .last_layer_embedding as (B, L, hidden).
    model.add_last_layer_embedding_extraction()

    pooled = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]

        model.forward(batch)
        embedding = model.last_layer_embedding

        pooled.append(embedding.mean(dim=1).to(EMBED_DTYPE).cpu())

    return torch.cat(pooled, dim=0)


def fit_linear_probe(x_train, y_train, x_test, alpha=1e-3):
    # standardize then ridge-regress; deliberately simple so the score reflects
    # the representation rather than probe capacity.
    scaler = StandardScaler().fit(x_train)

    probe = Ridge(alpha=alpha).fit(scaler.transform(x_train), y_train)

    return probe.predict(scaler.transform(x_test))


def run_probing(model, train_sequences, train_labels, test_sequences, test_labels, alpha=1e-3):
    x_train = extract_embeddings(model, train_sequences).numpy()
    x_test = extract_embeddings(model, test_sequences).numpy()

    y_pred = fit_linear_probe(x_train, np.asarray(train_labels), x_test, alpha=alpha)

    y_true = np.asarray(test_labels)
    pearson = float(pearsonr(y_pred.ravel(), y_true.ravel())[0])
    spearman = float(spearmanr(y_pred.ravel(), y_true.ravel())[0])

    print(f"probing complete: pearson={pearson:.4f} spearman={spearman:.4f}")

    return {"pearson": pearson, "spearman": spearman}
