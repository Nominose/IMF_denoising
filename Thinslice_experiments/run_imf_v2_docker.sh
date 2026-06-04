#!/bin/bash
# Run iMF v2 (v-head) brain-CT inference + evaluation for NFE=3 and NFE=5.
# Intended to be run INSIDE your docker container (where torch sees the GPU and
# D:\research is mounted at /host/d).
#
#   bash Thinslice_experiments/run_imf_v2_docker.sh
#
# Edit EPOCH / K / NFES below if needed.
set -e

EPOCH=200
K=20                 # number of stochastic samples to average
NFES=(3 5)           # NFE values to run

# --- locate repo root from this script, put its PARENT on PYTHONPATH so `import IMF_denoising` works ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"                       # .../IMF_denoising
export PYTHONPATH="$(dirname "$REPO_ROOT"):$PYTHONPATH"    # dir that contains IMF_denoising
cd "$REPO_ROOT"
echo "repo: $REPO_ROOT"
echo "PYTHONPATH head: ${PYTHONPATH%%:*}"

# --- GPU sanity (inference on CPU would be far too slow) ---
python -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| cuda', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU'); import sys; sys.exit(0 if ok else 1)" || {
  echo '!! No CUDA GPU visible to torch inside this container — fix the container/torch before running.'; exit 1; }

# --- inference: for each NFE, generate K samples (pred) then cumulative-average (avg) ---
for NFE in "${NFES[@]}"; do
  echo "================ NFE=$NFE : pred (K=$K) ================"
  python Thinslice_experiments/predict_2D_imf_v2.py --epoch "$EPOCH" --mode pred --iteration_num "$K" --num_steps "$NFE"
  echo "================ NFE=$NFE : avg ================"
  python Thinslice_experiments/predict_2D_imf_v2.py --epoch "$EPOCH" --mode avg  --num_steps "$NFE"
done

# --- evaluation: MAE / SSIM / LPIPS on [0,100] HU, across K, for both NFE ---
echo "================ EVAL ================"
python Thinslice_experiments/eval_imf_v2.py --epoch "$EPOCH" --num_steps "${NFES[@]}" --k_list 10 20

echo "DONE."
