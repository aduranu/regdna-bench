"""example-specific defaults for the D3 / zoonomia caQTL runs.

paths and tsv column names live here (not in the benchmark core) so the core CLI
stays dataset-agnostic. consumed by run.py and the dump_/build_ scripts.
"""

# zoonomia training defaults; model, genome store and cCREs are all hg38.
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
