#!/bin/bash
# Quantify NFE=3 only (K=10 and K=20) for brain-CT iMF v2.
# Run inside the docker container. Reads pred_images_nfe3/, writes a results xlsx + prints a summary.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"                     # .../IMF_denoising
export PYTHONPATH="$(dirname "$REPO_ROOT"):$PYTHONPATH"
cd "$REPO_ROOT"

python Thinslice_experiments/eval_imf_v2.py \
    --epoch 200 \
    --num_steps 3 \
    --k_list 10 20 \
    --out /host/d/research/projects/denoising/results/imf_v2_brainCT_eval_nfe3.xlsx
