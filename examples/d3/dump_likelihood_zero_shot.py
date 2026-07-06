"""dump per-variant likelihood scores for the zero-shot likelihood VEP task.

assembles via CaqtlVariantDataset, then saves the raw signed allele log-likelihood
difference (llr), its magnitude, labels, betas, CRE classes and window offsets to
an npz so the diagnostic plots can be made off-GPU. same computation behind the
auroc/auprc/spearman summary, just persisting the per-variant scores that the
runner strips.

requires `pip install -e .`. wrapper.py and config.py are siblings.

usage:
    python examples/d3/dump_likelihood_zero_shot.py \
        --variants data/caqtl/Afr.CaQTLS.tsv --window-mode variant \
        --out results/likelihood_zero_shot_variant_centered/scores.npz
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from config import DEFAULT_BED, DEFAULT_CKPT, DEFAULT_H5, DEFAULT_ZARR
from wrapper import D3Model

from regdna_bench.data.caqtl import CaqtlVariantDataset
from regdna_bench.tasks._scoring import extract_allele_score_difference


def main():
    parser = argparse.ArgumentParser(description="dump per-variant likelihood scores")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--window-mode", choices=["element", "variant"], default="element")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--ccre-bed", default=DEFAULT_BED)
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--chroms", default="chr22,chrX")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = CaqtlVariantDataset(
        variants=args.variants, ccre_bed=args.ccre_bed,
        zarr_path=args.zarr, h5_path=args.h5,
        chroms=args.chroms.split(","), window_mode=args.window_mode,
    )
    data = dataset.load()

    model = D3Model.from_pretrained(args.checkpoint, device=args.device)

    llr = extract_allele_score_difference(
        model, data.ref_ids, data.offsets, data.ref_base, data.alt_base,
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        llr=llr.astype(np.float64),
        abs_score=np.abs(llr).astype(np.float64),
        labels=data.labels.astype(np.int64),
        betas=data.effect_sizes.astype(np.float64),
        cre_classes=data.groups.astype(str),
        offsets=data.offsets.astype(np.int64),
        window_mode=np.array(args.window_mode),
    )

    print(f"wrote {out_path}: {llr.shape[0]} variants, window_mode {args.window_mode}, "
          f"backend {dataset.genome_backend}, drops {dict(dataset.drops)}")


if __name__ == "__main__":
    main()
