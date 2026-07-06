"""genome token store access for windowed dataset assembly.

reads single-species DNA windows from the Zoonomia uint8 token store. windows
come out as native CANON ids {N:0,A:1,C:2,G:3,T:4} with no decode step, so the
read is a zero-copy passthrough into the rest of the pipeline.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np

# human is the first species row in the zoonomia store layout.
SPECIES_INDEX = 0


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


def read_window(handle, chrom, lo, hi, chrom_len):
    # single-species window; N (token 0) only where it runs past a chromosome edge.
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
