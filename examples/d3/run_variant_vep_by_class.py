"""runner: zero-shot variant-effect prediction per CRE class, for D3.

assembles 350bp ref/alt windows for African caQTL variants and scores them with
one of the two zero-shot methods, reporting auroc/auprc (significant vs
background) overall and per cCRE class:

- embedding: cosine distance between mean-pooled ref/alt embeddings.
- likelihood: allele log-likelihood difference from the model's per-position
  scores (D3 uniform-SEDD log concrete score).

usage:
    PYTHONPATH=src python examples/d3/run_variant_vep_by_class.py \
        --variants /path/to/Afr.CaQTLS.tsv --method likelihood --window-mode variant
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# src/ holds regdna_bench + short_ctx_tasks; this dir holds the d3 wrapper and the
# shared variant_data module. done before project imports since the repo ships no
# installable package.
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from variant_data import assemble, load_ccre_intervals, norm_chrom, open_genome  # noqa: E402
from wrapper import D3Model  # noqa: E402
from short_ctx_tasks.variant_effect import (  # noqa: E402
    run_variant_zero_shot_by_class,
    run_variant_zero_shot_likelihood_by_class,
)

# zoonomia training defaults; the model, genome store and cCREs are all hg38.
DEFAULT_CKPT = "/grid/koo/home/shared/d3/trained_weights/zoonomia/best/model-epoch=207-val_loss=395.0214.ckpt"
DEFAULT_BED = "/grid/koo/home/shared/d3/data/zoonomia/GRCh38-cCREs.bed"
DEFAULT_ZARR = "~/scratch/d3-dna/zoonomia/zoonomia_v3.zarr"
DEFAULT_H5 = "/grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5"

# afr caQTL tsv columns (hg38)
COL_CHROM = "chr_hg38"
COL_POS = "pos_hg38"
COL_A1 = "allele1"
COL_A2 = "allele2"
COL_LABEL = "label"
COL_BETA = "beta"


def summarize(results):
    # strip the raw per-variant score arrays so the summary is json-dumpable.
    out = {}
    for key, entry in results.items():
        if "skipped" in entry:
            out[key] = {"n": entry["n"], "skipped": entry["skipped"]}
        else:
            out[key] = {"n": entry["n"], "metrics": entry.get("metrics", {})}

    return out


def print_summary(results, drops):
    summary = summarize(results)

    print("drop reasons:", dict(drops))
    for key in ["overall"] + sorted(k for k in summary if k != "overall"):
        entry = summary[key]
        if "skipped" in entry:
            print(f"  {key}: n={entry['n']} skipped ({entry['skipped']})")
            continue

        clf = entry["metrics"].get("classification", {})
        auroc = clf.get("auroc")
        auprc = clf.get("auprc")
        line = f"  {key}: n={entry['n']}"
        if auroc is not None:
            line += f" auroc={auroc:.4f} auprc={auprc:.4f}"
        print(line)


def run_method(method, model, data, min_per_class):
    # dispatch to the chosen zero-shot method; both report identical metric keys.
    if method == "embedding":
        return run_variant_zero_shot_by_class(
            model, data["ref_ids"], data["alt_ids"], data["cre_classes"],
            effect_sizes=data["betas"], labels=data["labels"],
            min_per_class=min_per_class,
        )

    return run_variant_zero_shot_likelihood_by_class(
        model, data["ref_ids"], data["offsets"], data["ref_tokens"],
        data["alt_tokens"], data["cre_classes"],
        effect_sizes=data["betas"], labels=data["labels"],
        min_per_class=min_per_class,
    )


def main():
    parser = argparse.ArgumentParser(description="zero-shot variant-effect prediction per CRE class for D3")
    parser.add_argument("--variants", required=True, help="caQTL tsv (hg38)")
    parser.add_argument("--method", choices=["embedding", "likelihood"], default="embedding")
    parser.add_argument("--window-mode", choices=["element", "variant"], default="element",
                        help="350bp window anchored on the cCRE midpoint (element) or the variant")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--ccre-bed", default=DEFAULT_BED)
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--chroms", default="chr22,chrX", help="held-out test chroms")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--out", default=None, help="optional json summary path")
    args = parser.parse_args()

    chroms = [norm_chrom(c) for c in args.chroms.split(",")]

    df = pd.read_csv(args.variants, sep="\t")
    # zero-shot eval set: variants used in the QTL test and located in peaks.
    if "IsUsed" in df.columns:
        df = df[df["IsUsed"].astype(bool)]
    if "in_peaks" in df.columns:
        df = df[df["in_peaks"].astype(bool)]

    df = df[df[COL_CHROM].map(norm_chrom).isin(chroms)]

    variants = [
        (norm_chrom(r[COL_CHROM]), int(r[COL_POS]) - 1, str(r[COL_A1]).upper(),
         str(r[COL_A2]).upper(), r[COL_LABEL], r[COL_BETA])
        for _, r in df.iterrows()
    ]

    intervals = load_ccre_intervals(args.ccre_bed, chroms)
    mode, handle, chrom_lengths = open_genome(args.zarr, args.h5, chroms)

    data, drops = assemble(variants, intervals, handle, chrom_lengths,
                           window_mode=args.window_mode)

    model = D3Model.from_pretrained(args.checkpoint, device=args.device)

    results = run_method(args.method, model, data, args.min_per_class)

    print_summary(results, drops)

    if args.out is not None:
        with open(args.out, "w") as handle_out:
            json.dump({"method": args.method, "window_mode": args.window_mode,
                       "genome_backend": mode, "drops": dict(drops),
                       "results": summarize(results)},
                      handle_out, indent=2)


if __name__ == "__main__":
    main()
