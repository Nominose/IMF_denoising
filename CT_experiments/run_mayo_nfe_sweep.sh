#!/bin/bash
#SBATCH --job-name=mayo_nfe_sweep
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_mayo_nfe_sweep_%j.txt

# run_mayo_nfe_sweep.sh (HPC/SLURM) — Mayo low-dose CT NFE sweep for the NO-GAN iMF model
# (trial imf_v2_unsupervised_gaussian_mayo, model-200.pt). This is the baseline the GAN runs get
# compared against.
#
# For each NFE it runs CT_experiments/predict_2D_imf.py twice:
#   pred : generate K stochastic samples per case (default K=20). With --input both (the default)
#          each sample = 2 generations (odd- and even-conditioned) averaged together.
#   avg  : write ONLY the K=10 and K=20 averages, then --cleanup drops the per-sample volumes.
# End state per case+NFE: epoch200avg/{pred_img_scans10,pred_img_scans20}.nii.gz + sample_std.npy.
#
# GROUND TRUTH IS NOT NEEDED: the Mayo gt (nii_imgs/<PID>/img.nii.gz) is not on the cluster, so the
# predict skips gt and no metrics are computed here. Download the epoch200avg/ folders and score
# them locally against your own gt. (Upload nii_imgs/ later and the script writes gt_img.nii.gz too.)
#
# Test set = 3 cases (L310, L192, L291) from the xlsx 'test' batch -> the whole sweep fits one job:
# per-sample cost is roughly 6s + 1.6s*NFE (measured on the A800 with adaptive batching), and
# 3 cases * 20 samples * 2 conditions = 120 generations per NFE:
#   nfe5 ~0.5h | nfe10 ~0.7h | nfe20 ~1.2h | nfe30 ~1.8h | nfe50 ~2.8h  -> ~7h total (pad to ~12h).
#
# `pred` is RESUMABLE: it skips samples whose pred_img.nii.gz already exists, so a killed/requeued
# job continues where it left off. NOTE: once `avg --cleanup` has run for an NFE it DELETES those
# per-sample files, so re-running `pred` for that NFE would regenerate all K samples from scratch.
#
# NFE list = positional args, else default "5 10 20 30 50" (ascending = cheapest first, so nfe5
# results land in ~30min instead of after the 50s). Override trial/epoch/K via env:
#   sbatch CT_experiments/run_mayo_nfe_sweep.sh                 # sweep 5 10 20 30 50
#   sbatch CT_experiments/run_mayo_nfe_sweep.sh 5 10            # just these
#   TRIAL=<name> EPOCH=<E> ITER=<K> sbatch CT_experiments/run_mayo_nfe_sweep.sh 50
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}   # ${..:-} so `set -u` tolerates an unset PYTHONPATH
REPO=/gpfs/work/aac/xingyiyao23/Code/IMF_denoising      # sbatch spools the script -> hardcode, don't use BASH_SOURCE
cd "$REPO"
PRED="$REPO/CT_experiments/predict_2D_imf.py"

TRIAL="${TRIAL:-imf_v2_unsupervised_gaussian_mayo}"
EPOCH="${EPOCH:-200}"
ITER="${ITER:-20}"
NFES="${*:-5 10 20 30 50}"

echo "Mayo no-GAN NFE sweep | trial=$TRIAL epoch=$EPOCH K=$ITER | NFEs=[$NFES]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for NFE in $NFES; do
  echo "================ NFE=$NFE : pred ================"
  python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode pred --num_steps "$NFE" --iteration_num "$ITER" \
    || { echo "pred NFE=$NFE FAILED"; exit 1; }
  echo "================ NFE=$NFE : avg (keep K=10,20 + cleanup) ================"
  python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode avg  --num_steps "$NFE" --k_save 10 20 --cleanup \
    || { echo "avg NFE=$NFE FAILED"; exit 1; }
  echo "================ NFE=$NFE done ================"
done
echo "sweep complete: [$NFES]"
echo "results: /gpfs/work/aac/xingyiyao23/projects/denoising/models/$TRIAL/pred_images_input_both_nfe*/<PID>/random_0/epoch${EPOCH}avg/"
