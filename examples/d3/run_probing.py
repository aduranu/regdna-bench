"""minimal runner: probe D3 last-layer embeddings on a lentiMPRA-style h5.

provide a trained D3 checkpoint; this loads the onehot/y splits, collects
last-layer representations and fits a probe (ridge on mean-pooled embeddings, or
cnn on the full per-position embeddings) reporting pearson/spearman. config and
probing noise level are taken from the checkpoint.

usage:
    PYTHONPATH=src python examples/d3/run_probing.py \
        --checkpoint path/to/d3.ckpt --data path/to/lenti_MPRA_HepG2_data.h5 \
        --probe cnn
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np
import torch

# src/ holds regdna_bench + short_ctx_tasks; this dir holds the d3 wrapper.
# done before the project imports since the repo ships no installable package.
_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from wrapper import D3Model  # noqa: E402
from short_ctx_tasks.probing import run_cnn_probing, run_probing  # noqa: E402


def _load_split(h5_path, x_key, y_key):
    with h5py.File(h5_path, "r") as handle:
        x = torch.as_tensor(np.asarray(handle[x_key]), dtype=torch.float32)
        y = np.asarray(handle[y_key])

    return x, y


def main():
    parser = argparse.ArgumentParser(description="probe D3 last-layer embeddings")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="h5 with onehot_/y_ splits")
    parser.add_argument("--config", default=None, help="optional; read from checkpoint if omitted")
    parser.add_argument("--device", default="cuda")
    # ridge = mean-pool + RidgeCV; cnn = full per-position embeddings + RepCNN
    parser.add_argument("--probe", default="ridge", choices=["ridge", "cnn"])
    # token_offset maps acgt onehot onto the model vocab (0 for k562, 1 for zoonomia)
    parser.add_argument("--token-offset", type=int, default=0)
    args = parser.parse_args()

    x_train, y_train = _load_split(args.data, "onehot_train", "y_train")
    x_test, y_test = _load_split(args.data, "onehot_test", "y_test")

    model = D3Model.from_pretrained(
        args.checkpoint, config=args.config, device=args.device,
        token_offset=args.token_offset,
    )

    if args.probe == "cnn":
        # cnn head needs a validation split for early stopping
        x_valid, y_valid = _load_split(args.data, "onehot_valid", "y_valid")
        run_cnn_probing(model, x_train, y_train, x_valid, y_valid, x_test, y_test, device=args.device)
    else:
        # alpha is tuned inside run_probing via RidgeCV, so the runner stays config-free
        run_probing(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
