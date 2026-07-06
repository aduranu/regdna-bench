"""caQTL variant-effect dataset (DART-Eval task 5 data).

reads 350bp ref/alt CANON windows for caQTL variants straight from the Zoonomia
token store, so sequences are native CANON ids and match how zoonomia models were
trained. windows are N-padded only at chromosome edges (human, species_index 0).
two anchorings via window_mode: cCRE-midpoint ("element") or variant-centered
("variant").

the dataset is model-agnostic: it emits CANON windows + the CANON base id at the
variant offset; each model maps CANON to its own vocab.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from regdna_bench.data.base import Dataset, VariantSet
from regdna_bench.data.ccre import find_containing_ccre, load_ccre_intervals, norm_chrom
from regdna_bench.data.genome_store import open_genome, read_window

# CANON base ids (N=0); windows read from the store are already these ids, so we
# only need the map to substitute the alt allele.
BASE_TO_TOK = {"A": 1, "C": 2, "G": 3, "T": 4}
SEQ_LEN = 350
HALF = SEQ_LEN // 2

# default afr caQTL tsv columns (hg38)
COLUMNS = {
    "chrom": "chr_hg38",
    "pos": "pos_hg38",
    "a1": "allele1",
    "a2": "allele2",
    "label": "label",
    "beta": "beta",
}


def assemble(variants, intervals, handle, chrom_lengths, window_mode="element"):
    # turn caQTL rows into 350bp ref/alt CANON windows + labels, plus the
    # per-variant offset and ref/alt base ids (needed by the likelihood method).
    # the containing cCRE is always looked up: it assigns the per-class label and
    # keeps the variant set identical across window modes. window_mode controls
    # where the window sits:
    #   "element" -> cCRE midpoint +/-175 (native; variant off-center)
    #   "variant" -> variant +/-175 (variant at index 175); flanks pull in real
    #                genomic sequence beyond the cCRE, N only at chromosome edges.
    if window_mode not in ("element", "variant"):
        raise ValueError(f"unknown window_mode {window_mode!r}")

    ref_list, alt_list, cls_list, lab_list, beta_list = [], [], [], [], []
    off_list, reftok_list, alttok_list = [], [], []
    drops = Counter()

    for chrom, pos, a1, a2, label, beta in variants:
        if len(a1) != 1 or len(a2) != 1 or a1 not in BASE_TO_TOK or a2 not in BASE_TO_TOK:
            drops["not_snp"] += 1
            continue

        hit = find_containing_ccre(intervals, chrom, pos)
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

        window = read_window(handle, chrom, lo, lo + SEQ_LEN, chrom_lengths[chrom])
        genome_tok = int(window[off])

        # the genome holds one of the two alleles (the ref); the other is the alt
        # substitution. orientation only matters for the signed likelihood readout.
        if genome_tok == BASE_TO_TOK[a1]:
            other = BASE_TO_TOK[a2]
        elif genome_tok == BASE_TO_TOK[a2]:
            other = BASE_TO_TOK[a1]
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

    variant_set = VariantSet(
        ref_ids=np.stack(ref_list).astype(np.int64),
        alt_ids=np.stack(alt_list).astype(np.int64),
        offsets=np.asarray(off_list, dtype=np.int64),
        ref_base=np.asarray(reftok_list, dtype=np.int64),
        alt_base=np.asarray(alttok_list, dtype=np.int64),
        groups=np.asarray(cls_list),
        labels=np.asarray(lab_list, dtype=np.int64),
        effect_sizes=np.asarray(beta_list, dtype=np.float64),
    )

    return variant_set, drops


class CaqtlVariantDataset(Dataset):
    """assembles African caQTL variants into ref/alt CANON windows.

    drops (per-reason counts) and genome_backend ("zarr"/"h5") are recorded on
    the instance after load() for the runner's summary.
    """

    def __init__(self, variants, ccre_bed, zarr_path=None, h5_path=None,
                 chroms=("chr22", "chrX"), window_mode="element", columns=None):
        self.variants = variants
        self.ccre_bed = ccre_bed
        self.zarr_path = zarr_path
        self.h5_path = h5_path
        self.chroms = [norm_chrom(c) for c in chroms]
        self.window_mode = window_mode
        self.columns = dict(COLUMNS, **(columns or {}))

        self.drops = None
        self.genome_backend = None

    @property
    def name(self):
        return "caqtl"

    def _read_variants(self):
        import pandas as pd

        col = self.columns
        df = pd.read_csv(self.variants, sep="\t")

        # zero-shot eval set: variants used in the QTL test and located in peaks.
        if "IsUsed" in df.columns:
            df = df[df["IsUsed"].astype(bool)]
        if "in_peaks" in df.columns:
            df = df[df["in_peaks"].astype(bool)]

        df = df[df[col["chrom"]].map(norm_chrom).isin(self.chroms)]

        # tsv positions are 1-based; the store is 0-based.
        return [
            (norm_chrom(r[col["chrom"]]), int(r[col["pos"]]) - 1,
             str(r[col["a1"]]).upper(), str(r[col["a2"]]).upper(),
             r[col["label"]], r[col["beta"]])
            for _, r in df.iterrows()
        ]

    def load(self):
        variants = self._read_variants()

        intervals = load_ccre_intervals(self.ccre_bed, self.chroms)
        mode, handle, chrom_lengths = open_genome(self.zarr_path, self.h5_path, self.chroms)

        variant_set, drops = assemble(variants, intervals, handle, chrom_lengths,
                                      window_mode=self.window_mode)

        self.drops = drops
        self.genome_backend = mode

        return variant_set
