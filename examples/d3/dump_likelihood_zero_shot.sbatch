#!/bin/bash
#SBATCH --job-name=d3_lik_dump
#SBATCH --output=sbatch_out/d3_lik_dump_%j.out
#SBATCH --error=sbatch_out/d3_lik_dump_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq

# re-runs both likelihood VEP configs (element + variant-centered), regenerating
# the json summaries, and dumps the raw per-variant llr scores to npz so the
# diagnostic plots (roc/pr curves, score separation, effect-size scatter) can be
# made off-GPU. h100 because the d3 forward runs FlashAttention in bf16.

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate d3-dna

cd "$SLURM_SUBMIT_DIR"

REPO="$SLURM_SUBMIT_DIR"
TSV="$REPO/data/caqtl/Afr.CaQTLS.tsv"

run_one () {
    local mode="$1" dir="$2"
    mkdir -p "$REPO/results/$dir"

    python examples/d3/run.py \
        --method likelihood --window-mode "$mode" \
        --variants "$TSV" \
        --out "$REPO/results/$dir/results_likelihood_by_class.json"

    python examples/d3/dump_likelihood_zero_shot.py \
        --window-mode "$mode" \
        --variants "$TSV" \
        --out "$REPO/results/$dir/scores.npz"
}

run_one element likelihood_zero_shot
run_one variant likelihood_zero_shot_variant_centered

echo "done: re-ran likelihood experiments + dumped scores"
