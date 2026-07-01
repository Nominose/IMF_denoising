#!/bin/bash
#SBATCH --job-name=ts_gan_nfe_sweep
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_gan_nfe_sweep_%j.txt

# run_gan_nfe_sweep.sh (HPC/SLURM) — iMF+GAN brain-CT inference sweep over multiple NFEs.
#
# For each NFE it runs Thinslice_experiments/predict_2D_imf_v2.py twice:
#   pred : generate K stochastic samples per case (default K=20)
#   avg  : write the K=10 and K=20 averages (--cleanup drops the per-sample volumes to save disk)
# Full test set (batch 5, interval 2 -> 16 cases). No fp16, to match the baseline NFE sweep.
# Target checkpoint: trial imf_gan_unsupervised_gaussian_brainCT, model-28.pt (the original GAN
# fine-tune). xlsx / bins / study dir auto-resolve to /gpfs/work/aac/xingyiyao23 via
# predict_2D_imf_v2.py's _detect_base()/_he_bins().
#
# `pred` mode is RESUMABLE: it skips samples whose pred_img.nii.gz already exists, so a job that
# hits the 72h wall continues where it left off on re-submit.
#
# NFE list = positional args, else default "5 10 20 30 50". Override trial/epoch/K via env:
#   sbatch gan/run_gan_nfe_sweep.sh                    # sweep 5 10 20 30 50
#   sbatch gan/run_gan_nfe_sweep.sh 5 10 20 30         # chunk A (fits one 72h job)
#   sbatch gan/run_gan_nfe_sweep.sh 50                 # chunk B
#   TRIAL=<name> EPOCH=<E> ITER=<K> sbatch gan/run_gan_nfe_sweep.sh 50
#
# WALL-TIME WARNING: high NFEs are slow. Rough full-set (16 cases) times, extrapolated from nfe3
# (~3h on the docker RTX; the A800 may be faster): nfe5 ~5h | nfe10 ~9h | nfe20 ~18h | nfe30 ~27h |
# nfe50 ~44h. The full "5 10 20 30 50" sweep (~103h) EXCEEDS the 72h wall -> submit in chunks
# (e.g. "5 10 20 30" ~59h, then "50" ~44h), or rely on the resume-on-re-submit behaviour.
#
# After each NFE, get the metrics (brain window [0,100] HU, K=10/20, mean+-std) with:
#   python gan/eval_gan_nfe.py --trial <TRIAL> --epoch <EPOCH> --nfe <N>
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
REPO=/gpfs/work/aac/xingyiyao23/Code/IMF_denoising      # sbatch spools the script -> hardcode, don't use BASH_SOURCE
cd "$REPO"
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
