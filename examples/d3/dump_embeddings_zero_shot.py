"""dump per-variant embeddings + scores for the zero-shot embedding VEP task.

assembles via CaqtlVariantDataset, then saves the raw ref/alt mean-pooled
embeddings, cosine distances, labels, betas and CRE classes to an npz so the
low-dim and classification plots can be made off-GPU. same computation that
produces the auroc/auprc/spearman numbers, just persisting the intermediates the
scoring summary strips.

requires `pip install -e .`. wrapper.py and config.py are siblings.

usage:
    python examples/d3/dump_embeddings_zero_shot.py \
        --variants data/caqtl/Afr.CaQTLS.tsv \
        --out results/embedding_zero_shot/embeddings.npz
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from config import DEFAULT_BED, DEFAULT_CKPT, DEFAULT_H5, DEFAULT_ZARR
from wrapper import D3Model

from regdna_bench.data.caqtl import CaqtlVariantDataset
from regdna_bench.tasks._scoring import extract_embeddings


def main():
    parser = argparse.ArgumentParser(description="dump embeddings for embedding zero-shot VEP")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--window-mode", choices=["element", "variant"], default="element")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--ccre-bed", default=DEFAULT_BED)
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--chroms", default="chr22,chrX")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/embedding_zero_shot/embeddings.npz")
    args = parser.parse_args()

    dataset = CaqtlVariantDataset(
        variants=args.variants, ccre_bed=args.ccre_bed,
        zarr_path=args.zarr, h5_path=args.h5,
        chroms=args.chroms.split(","), window_mode=args.window_mode,
    )
    data = dataset.load()

    model = D3Model.from_pretrained(args.checkpoint, device=args.device)

    ref_emb = extract_embeddings(model, data.ref_ids).numpy()
    alt_emb = extract_embeddings(model, data.alt_ids).numpy()

    cos = torch.nn.functional.cosine_similarity(
        torch.from_numpy(ref_emb), torch.from_numpy(alt_emb), dim=1,
    )
    distance = (1.0 - cos).numpy()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        ref_emb=ref_emb.astype(np.float32),
        alt_emb=alt_emb.astype(np.float32),
        cosine_distance=distance.astype(np.float32),
        labels=data.labels.astype(np.int64),
        betas=data.effect_sizes.astype(np.float64),
        cre_classes=data.groups.astype(str),
    )

    print(f"wrote {out_path}: {ref_emb.shape[0]} variants, embedding dim {ref_emb.shape[1]}, "
          f"backend {dataset.genome_backend}, drops {dict(dataset.drops)}")


if __name__ == "__main__":
    main()
