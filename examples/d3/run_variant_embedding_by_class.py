"""runner: zero-shot embedding variant-effect prediction per CRE class, for D3.

assembles 350bp element-anchored ref/alt windows for African caQTL variants and
scores them with the embedding cosine-distance task, reporting auroc/auprc
(significant vs background) overall and per cCRE class.

windows are read straight from the Zoonomia uint8 token store (not a FASTA), so
sequences are native 5-token ids {N:0,A:1,C:2,G:3,T:4} and match exactly how D3
was trained: window = cCRE midpoint +/-175, N-padded at chrom edges, human
species_index 0, restricted to the held-out test chromosomes.

usage:
    PYTHONPATH=src python examples/d3/run_variant_embedding_by_class.py \
        --variants /path/to/Afr.CaQTLS.tsv
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import pathlib
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

# src/ holds regdna_bench + short_ctx_tasks; this dir holds the d3 wrapper.
# done before the project imports since the repo ships no installable package.
_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from wrapper import D3Model  # noqa: E402
from short_ctx_tasks.variant_effect import run_variant_zero_shot_by_class  # noqa: E402

# zoonomia training defaults; the model, genome store and cCREs are all hg38.
DEFAULT_CKPT = "/grid/koo/home/shared/d3/trained_weights/zoonomia/best/model-epoch=207-val_loss=395.0214.ckpt"
DEFAULT_BED = "/grid/koo/home/shared/d3/data/zoonomia/GRCh38-cCREs.bed"
DEFAULT_ZARR = "~/scratch/d3-dna/zoonomia/zoonomia_v3.zarr"
DEFAULT_H5 = "/grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5"

# native zoonomia tokenization (N=0); A,C,G,T below. windows read from the store
# are already these ids, so we only need the map to substitute the alt allele.
_BASE_TO_TOK = {"A": 1, "C": 2, "G": 3, "T": 4}
SEQ_LEN = 350
HALF = SEQ_LEN // 2
SPECIES_INDEX = 0

# afr caQTL tsv columns (hg38)
COL_CHROM = "chr_hg38"
COL_POS = "pos_hg38"
COL_A1 = "allele1"
COL_A2 = "allele2"
COL_LABEL = "label"
COL_BETA = "beta"


def _norm_chrom(chrom):
    # bed + store use "chr22"/"chrX"; normalize the tsv to match.
    chrom = str(chrom)

    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def _open_genome(zarr_path, h5_path, chroms):
    # zarr is ~50-100x faster than the h5 for many small single-species windows
    # (chunks (1,8192) vs (241,1000)); prefer it, fall back to h5. selection and
    # the [chrom]["seq"] layout mirror ZoonomiaDataset in d3-dna examples/zoonomia.
    expanded = os.path.expanduser(str(zarr_path)) if zarr_path else None

    if expanded and pathlib.Path(expanded).exists():
        import zarr

        handle = zarr.open_group(store=zarr.storage.LocalStore(expanded), mode="r")
        mode = "zarr"
    else:
        import h5py

        handle = h5py.File(h5_path, "r")
        mode = "h5"

    lengths = {c: int(handle[c]["seq"].shape[1]) for c in chroms}

    return mode, handle, lengths


def _read_window(handle, chrom, lo, hi, chrom_len):
    # single-species window with N-padding (token 0) past chromosome edges.
    lo_clip = max(lo, 0)
    hi_clip = min(hi, chrom_len)

    core = np.asarray(handle[chrom]["seq"][SPECIES_INDEX, lo_clip:hi_clip], dtype=np.int64)

    if lo < 0 or hi > chrom_len:
        pieces = []
        if lo < 0:
            pieces.append(np.zeros(-lo, dtype=np.int64))
        pieces.append(core)
        if hi > chrom_len:
            pieces.append(np.zeros(hi - chrom_len, dtype=np.int64))
        core = np.concatenate(pieces)

    return core


def _load_ccre_intervals(bed_path, chroms):
    # per-chrom (sorted starts, ends, classes) for variant->cCRE containment.
    starts = {c: [] for c in chroms}
    ends = {c: [] for c in chroms}
    classes = {c: [] for c in chroms}

    with open(bed_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue

            row = line.rstrip("\n").split("\t")
            chrom = row[0]
            if chrom not in starts:
                continue

            starts[chrom].append(int(row[1]))
            ends[chrom].append(int(row[2]))
            classes[chrom].append(row[5])

    # bed is coordinate-sorted already, but sort defensively so bisect is valid
    intervals = {}
    for c in chroms:
        order = np.argsort(starts[c], kind="stable")
        intervals[c] = (
            np.asarray(starts[c])[order],
            np.asarray(ends[c])[order],
            np.asarray(classes[c])[order],
        )

    return intervals


def _find_containing_ccre(intervals, chrom, pos0):
    # cCREs are non-overlapping; the candidate is the rightmost start <= pos0.
    # check it and its left neighbor to be safe against adjacency.
    if chrom not in intervals:
        return None

    starts, ends, classes = intervals[chrom]
    idx = bisect.bisect_right(starts, pos0) - 1

    for cand in (idx, idx - 1):
        if 0 <= cand < len(starts) and starts[cand] <= pos0 < ends[cand]:
            return int(starts[cand]), int(ends[cand]), str(classes[cand])

    return None


def _assemble(variants, intervals, handle, chrom_lengths):
    # turn caQTL rows into element-anchored ref/alt token windows + labels.
    ref_list, alt_list, cls_list, lab_list, beta_list = [], [], [], [], []
    drops = Counter()

    for chrom, pos, a1, a2, label, beta in variants:
        if len(a1) != 1 or len(a2) != 1 or a1 not in _BASE_TO_TOK or a2 not in _BASE_TO_TOK:
            drops["not_snp"] += 1
            continue

        hit = _find_containing_ccre(intervals, chrom, pos)
        if hit is None:
            drops["no_ccre"] += 1
            continue

        start, end, cls = hit
        mid = (start + end) // 2
        lo = mid - HALF
        off = pos - lo

        # variant inside the cCRE but outside its central 350bp window. a real
        # consequence of element-anchoring rather than variant-centering.
        if not (0 <= off < SEQ_LEN):
            drops["out_of_window"] += 1
            continue

        window = _read_window(handle, chrom, lo, lo + SEQ_LEN, chrom_lengths[chrom])
        genome_tok = int(window[off])

        # the genome holds one of the two alleles; the other is the substitution.
        # cosine distance is symmetric, so ref/alt orientation does not matter.
        if genome_tok == _BASE_TO_TOK[a1]:
            other = _BASE_TO_TOK[a2]
        elif genome_tok == _BASE_TO_TOK[a2]:
            other = _BASE_TO_TOK[a1]
        else:
            drops["ref_mismatch"] += 1
            continue

        alt = window.copy()
        alt[off] = other

        ref_list.append(window)
        alt_list.append(alt)
        cls_list.append(cls)
        lab_list.append(int(label))
        beta_list.append(float(beta))

    if not ref_list:
        raise ValueError("no variants survived assembly; check chroms/columns/genome build")

    arrays = (
        np.stack(ref_list).astype(np.int64),
        np.stack(alt_list).astype(np.int64),
        np.asarray(cls_list),
        np.asarray(lab_list, dtype=np.int64),
        np.asarray(beta_list, dtype=np.float64),
    )

    return arrays, drops


def _serializable(results):
    # strip the raw cosine_distance arrays so the summary is json-dumpable.
    out = {}
    for key, entry in results.items():
        if "skipped" in entry:
            out[key] = {"n": entry["n"], "skipped": entry["skipped"]}
        else:
            out[key] = {"n": entry["n"], "metrics": entry.get("metrics", {})}

    return out


def _print_summary(results, drops):
    summary = _serializable(results)

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


def main():
    parser = argparse.ArgumentParser(description="zero-shot embedding VEP per CRE class for D3")
    parser.add_argument("--variants", required=True, help="caQTL tsv (hg38)")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--ccre-bed", default=DEFAULT_BED)
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--chroms", default="chr22,chrX", help="held-out test chroms")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--out", default=None, help="optional json summary path")
    args = parser.parse_args()

    chroms = [_norm_chrom(c) for c in args.chroms.split(",")]

    df = pd.read_csv(args.variants, sep="\t")
    # zero-shot eval set: variants used in the QTL test and located in peaks.
    if "IsUsed" in df.columns:
        df = df[df["IsUsed"].astype(bool)]
    if "in_peaks" in df.columns:
        df = df[df["in_peaks"].astype(bool)]

    df = df[df[COL_CHROM].map(_norm_chrom).isin(chroms)]

    variants = [
        (_norm_chrom(r[COL_CHROM]), int(r[COL_POS]) - 1, str(r[COL_A1]).upper(),
         str(r[COL_A2]).upper(), r[COL_LABEL], r[COL_BETA])
        for _, r in df.iterrows()
    ]

    intervals = _load_ccre_intervals(args.ccre_bed, chroms)
    mode, handle, chrom_lengths = _open_genome(args.zarr, args.h5, chroms)

    (ref_ids, alt_ids, cre_classes, labels, betas), drops = _assemble(
        variants, intervals, handle, chrom_lengths,
    )

    model = D3Model.from_pretrained(args.checkpoint, device=args.device)

    results = run_variant_zero_shot_by_class(
        model, ref_ids, alt_ids, cre_classes,
        effect_sizes=betas, labels=labels, min_per_class=args.min_per_class,
    )

    _print_summary(results, drops)

    if args.out is not None:
        with open(args.out, "w") as handle_out:
            json.dump({"genome_backend": mode, "drops": dict(drops),
                       "results": _serializable(results)}, handle_out, indent=2)


if __name__ == "__main__":
    main()
