#!/bin/bash
#SBATCH --job-name=d3_lik_vep_vc
#SBATCH --output=sbatch_out/d3_lik_vep_vc_%j.out
#SBATCH --error=sbatch_out/d3_lik_vep_vc_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq

# likelihood-based zero-shot variant-effect task (DART-Eval task 5), VARIANT-
# CENTERED windows: the 350bp window is centered on the variant (index 175) and
# flanks pull in real genomic sequence beyond the cCRE; N only at chromosome
# edges. h100 because the d3 forward runs FlashAttention in bf16.

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate d3-dna

cd "$SLURM_SUBMIT_DIR"

REPO="$SLURM_SUBMIT_DIR"
OUT_JSON="$REPO/results/likelihood_zero_shot_variant_centered/results_likelihood_by_class.json"
mkdir -p "$(dirname "$OUT_JSON")"

PYTHONPATH="$REPO/src" python examples/d3/run_variant_vep_by_class.py \
    --method likelihood \
    --window-mode variant \
    --variants "$REPO/data/caqtl/Afr.CaQTLS.tsv" \
    --out "$OUT_JSON"

echo "wrote $OUT_JSON"
