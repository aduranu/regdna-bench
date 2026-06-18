#!/bin/bash
#SBATCH --job-name=d3_embed_vep
#SBATCH --output=sbatch_out/d3_embed_vep_%j.out
#SBATCH --error=sbatch_out/d3_embed_vep_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq

# zero-shot embedding variant-effect task (DART-Eval task 5) on the African
# caQTL set, scored with the zoonomia D3 checkpoint. h100 requested because the
# d3 forward runs FlashAttention in bf16 (needs Ampere+; v100 has no bf16).

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate d3-dna

cd "$SLURM_SUBMIT_DIR"

REPO="$SLURM_SUBMIT_DIR"
OUT_JSON="$REPO/results/embedding_zero_shot/results_embedding_by_class.json"
mkdir -p "$(dirname "$OUT_JSON")"

PYTHONPATH="$REPO/src" python examples/d3/run_variant_vep_by_class.py \
    --method embedding \
    --variants "$REPO/data/caqtl/Afr.CaQTLS.tsv" \
    --out "$OUT_JSON"

echo "wrote $OUT_JSON"
