"""per-variant position table for the accessibility-overlay plots.

reproduces the cCRE containment + element-anchoring from the caqtl dataset, but
without the genome/model: for each eval variant it records the host cCRE midpoint
(the 350bp window center the model scores on), the signed offset of the variant
from that center, its CRE class and significance label. used to overlay variant
positions on the DNase accessibility meta-profile per CRE class.

requires `pip install -e .`. config.py is a sibling.

usage:
    python examples/d3/build_variant_position_table.py \
        --variants data/caqtl/Afr.CaQTLS.tsv \
        --out results/embedding_zero_shot/variant_positions.csv
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from config import COL_BETA, COL_CHROM, COL_LABEL, COL_POS, DEFAULT_BED

from regdna_bench.data.caqtl import HALF, SEQ_LEN
from regdna_bench.data.ccre import find_containing_ccre, load_ccre_intervals, norm_chrom


def main():
    parser = argparse.ArgumentParser(description="build per-variant position table")
    parser.add_argument("--variants", default="data/caqtl/Afr.CaQTLS.tsv")
    parser.add_argument("--ccre-bed", default=DEFAULT_BED)
    parser.add_argument("--chroms", default="chr22,chrX")
    parser.add_argument("--out", default="results/embedding_zero_shot/variant_positions.csv")
    args = parser.parse_args()

    chroms = [norm_chrom(c) for c in args.chroms.split(",")]

    df = pd.read_csv(args.variants, sep="\t")
    if "IsUsed" in df.columns:
        df = df[df["IsUsed"].astype(bool)]
    if "in_peaks" in df.columns:
        df = df[df["in_peaks"].astype(bool)]

    df = df[df[COL_CHROM].map(norm_chrom).isin(chroms)]

    intervals = load_ccre_intervals(args.ccre_bed, chroms)

    rows = []
    for _, r in df.iterrows():
        chrom = norm_chrom(r[COL_CHROM])
        pos0 = int(r[COL_POS]) - 1

        hit = find_containing_ccre(intervals, chrom, pos0)
        if hit is None:
            continue

        start, end, cls = hit
        mid = (start + end) // 2
        lo = mid - HALF
        off = pos0 - lo

        # keep only variants inside the central 350bp window, matching scoring
        if not (0 <= off < SEQ_LEN):
            continue

        # signed offset from the window center (model anchor), in bp
        rows.append({
            "chrom": chrom, "pos0": pos0, "ccre_start": start, "ccre_end": end,
            "ccre_mid": mid, "offset_from_center": pos0 - mid, "cre_class": cls,
            "label": int(bool(r[COL_LABEL])), "beta": float(r[COL_BETA]),
        })

    out = pd.DataFrame(rows)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"wrote {out_path}: {len(out)} variants")
    print("\nper-class counts (significant / total):")
    summary = out.groupby("cre_class")["label"].agg(["sum", "count"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
