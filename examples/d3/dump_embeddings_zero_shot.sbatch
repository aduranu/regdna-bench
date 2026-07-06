#!/bin/bash
#SBATCH --job-name=d3_embed_dump
#SBATCH --output=sbatch_out/d3_embed_dump_%j.out
#SBATCH --error=sbatch_out/d3_embed_dump_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq

# dumps the ref/alt embeddings + cosine distances for the embedding zero-shot
# VEP task so the low-dim and classification plots can be built off-GPU.

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate d3-dna

cd "$SLURM_SUBMIT_DIR"

REPO="$SLURM_SUBMIT_DIR"

python examples/d3/dump_embeddings_zero_shot.py \
    --variants "$REPO/data/caqtl/Afr.CaQTLS.tsv" \
    --out "$REPO/results/embedding_zero_shot/embeddings.npz"

echo "done"
