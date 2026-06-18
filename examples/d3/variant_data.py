"""shared data assembly for the zero-shot variant-effect runners.

reads element-anchored 350bp ref/alt token windows for caQTL variants straight
from the Zoonomia uint8 token store (not a FASTA), so sequences are native
5-token ids {N:0,A:1,C:2,G:3,T:4} and match how D3 was trained: window = cCRE
midpoint +/-175, N-padded at chrom edges, human species_index 0.

both the embedding and likelihood runners share this; the embedding method uses
the ref/alt windows, the likelihood method additionally uses the per-variant
offset + ref/alt token ids.
"""

from __future__ import annotations

import bisect
import os
import pathlib
from collections import Counter

import numpy as np

# native zoonomia tokenization (N=0); windows read from the store are already
# these ids, so we only need the map to substitute the alt allele.
_BASE_TO_TOK = {"A": 1, "C": 2, "G": 3, "T": 4}
SEQ_LEN = 350
HALF = SEQ_LEN // 2
SPECIES_INDEX = 0


def norm_chrom(chrom):
    # bed + store use "chr22"/"chrX"; normalize the tsv to match.
    chrom = str(chrom)

    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def open_genome(zarr_path, h5_path, chroms):
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


def load_ccre_intervals(bed_path, chroms):
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


def assemble(variants, intervals, handle, chrom_lengths, window_mode="element"):
    # turn caQTL rows into 350bp ref/alt token windows + labels. also returns the
    # per-variant offset and ref/alt token ids, which the likelihood method needs
    # (the embedding method only uses ref/alt windows). the containing cCRE is
    # always looked up: it assigns the per-class label and keeps the variant set
    # identical across window modes. window_mode controls where the window sits:
    #   "element" -> cCRE midpoint +/-175 (D3-native; variant off-center)
    #   "variant" -> variant +/-175 (variant always at index 175). flanks pull in
    #                real genomic sequence beyond the cCRE; _read_window N-pads only
    #                where the window runs off a chromosome edge.
    if window_mode not in ("element", "variant"):
        raise ValueError(f"unknown window_mode {window_mode!r}")

    ref_list, alt_list, cls_list, lab_list, beta_list = [], [], [], [], []
    off_list, reftok_list, alttok_list = [], [], []
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

        if window_mode == "variant":
            lo = pos - HALF
            off = HALF
        else:
            mid = (start + end) // 2
            lo = mid - HALF
            off = pos - lo

            # variant inside the cCRE but outside its central 350bp window: a real
            # consequence of element-anchoring (cannot happen when variant-centered).
            if not (0 <= off < SEQ_LEN):
                drops["out_of_window"] += 1
                continue

        window = _read_window(handle, chrom, lo, lo + SEQ_LEN, chrom_lengths[chrom])
        genome_tok = int(window[off])

        # the genome holds one of the two alleles (the ref); the other is the alt
        # substitution. cosine distance is symmetric, so this orientation only
        # matters for the (signed) likelihood readout.
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
        off_list.append(off)
        reftok_list.append(genome_tok)
        alttok_list.append(other)

    if not ref_list:
        raise ValueError("no variants survived assembly; check chroms/columns/genome build")

    arrays = {
        "ref_ids": np.stack(ref_list).astype(np.int64),
        "alt_ids": np.stack(alt_list).astype(np.int64),
        "cre_classes": np.asarray(cls_list),
        "labels": np.asarray(lab_list, dtype=np.int64),
        "betas": np.asarray(beta_list, dtype=np.float64),
        "offsets": np.asarray(off_list, dtype=np.int64),
        "ref_tokens": np.asarray(reftok_list, dtype=np.int64),
        "alt_tokens": np.asarray(alttok_list, dtype=np.int64),
    }

    return arrays, drops
