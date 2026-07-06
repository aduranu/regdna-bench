"""cCRE interval lookup for assigning variants to a host regulatory element.

used by windowed variant datasets to find the candidate cis-regulatory element
containing a variant (for per-class stratification and element-anchored windows).
"""

from __future__ import annotations

import bisect

import numpy as np


def norm_chrom(chrom):
    # bed + store use "chr22"/"chrX"; normalize tsv chrom values to match.
    chrom = str(chrom)

    return chrom if chrom.startswith("chr") else f"chr{chrom}"


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


def find_containing_ccre(intervals, chrom, pos0):
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
