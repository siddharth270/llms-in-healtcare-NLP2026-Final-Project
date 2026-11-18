#!/bin/bash
#SBATCH --job-name=enc
#SBATCH --partition=courses-gpu    # gpu has A100s but was fully allocated on Nov 18
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# logs/ must exist before sbatch, or Slurm cannot open the output file and the job dies silently
#
#   sbatch scripts/run_gpu.sh encoder.train --target hard --epochs 1
#   sbatch scripts/run_gpu.sh encoder.predict checkpoints/encoder/full_hard_s1

cd /scratch/$USER/llms-in-healtcare-NLP2026-Final-Project
source .venv/bin/activate
set -euo pipefail                  # after activate, which touches unset variables
export HF_HOME=/scratch/$USER/hf_cache
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m "$@"
