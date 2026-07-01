#!/usr/bin/env bash
# run_gan_nfe_sweep.sh — iMF+GAN brain-CT inference sweep over multiple NFEs.
#
# For each NFE it runs predict_2D_imf_v2.py twice:
#   pred : generate K stochastic samples per case (default K=20)
#   avg  : write the K=10 and K=20 averages (--cleanup drops the per-sample volumes to save disk)
# Full test set (batch 5, interval 2 -> 16 cases). No fp16, to match the baseline NFE sweep.
#
# `pred` mode is RESUMABLE: it skips samples whose pred_img.nii.gz already exists, so an
# interrupted NFE continues where it left off on re-run.
#
# Default NFE list = 5 10 20 30 50 (the tail of the brain-CT NFE table; nfe 2 and 3 already done).
# Run INSIDE the docker env (D: mounted at /host/d, repo reachable) on a GPU.
#
# Usage:
#   bash gan/run_gan_nfe_sweep.sh                 # sweep 5 10 20 30 50
#   bash gan/run_gan_nfe_sweep.sh 10 20           # only these NFEs
#   TRIAL=<name> EPOCH=<E> ITER=<K> bash gan/run_gan_nfe_sweep.sh 50
#
# After each NFE, get the metrics (brain window [0,100] HU, K=10/20, mean+-std) with:
#   python gan/eval_gan_nfe.py --trial "$TRIAL" --epoch "$EPOCH" --nfe <N>
#
# WARNING: high NFEs are slow. Rough full-set (16 cases) times, extrapolated from nfe3 (~3h):
#   nfe5 ~5h | nfe10 ~9h | nfe20 ~18h | nfe30 ~27h | nfe50 ~44h.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"                              # .../IMF_denoising
PRED="$REPO/Thinslice_experiments/predict_2D_imf_v2.py"

TRIAL="${TRIAL:-imf_gan_unsupervised_gaussian_brainCT}"
EPOCH="${EPOCH:-28}"
ITER="${ITER:-20}"
NFES="${*:-5 10 20 30 50}"

echo "iMF+GAN NFE sweep | trial=$TRIAL epoch=$EPOCH K=$ITER | NFEs=[$NFES]"
for NFE in $NFES; do
  echo "================ NFE=$NFE : pred ================"
  python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode pred --num_steps "$NFE" --iteration_num "$ITER" || { echo "pred NFE=$NFE FAILED"; exit 1; }
  echo "================ NFE=$NFE : avg  ================"
  python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode avg  --num_steps "$NFE" --cleanup       || { echo "avg  NFE=$NFE FAILED"; exit 1; }
  echo "================ NFE=$NFE done. metrics: python gan/eval_gan_nfe.py --trial $TRIAL --epoch $EPOCH --nfe $NFE"
done
echo "sweep complete: [$NFES]"
