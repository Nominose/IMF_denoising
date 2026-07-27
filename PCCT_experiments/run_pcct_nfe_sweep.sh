#!/bin/bash
#SBATCH --job-name=pcct_nfe_sweep
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_pcct_nfe_sweep_%j.txt

# PCCT inference sweep + CNR, for either the no-GAN or the GAN model.
#
# Per NFE: pred (K=20 stochastic samples) -> avg (keep ONLY K=10,20, --cleanup drops the rest) ->
# CNR. End state per case+NFE: epoch<E>avg/{pred_img_scans10,pred_img_scans20,condition_img}.nii.gz
# plus a PCCT_CNR_epoch<E>_nfe<N>.xlsx per NFE.
#
# CNR (no-reference, Chen's formulation) is the metric because real PCCT has NO ground truth --
# MAE/SSIM/LPIPS cannot be computed at all here. The noisy input is scored too, as the baseline the
# denoised result must beat.
#
# 8 test cases (29-36) x 20 samples x 50 slices. With adaptive batching the whole volume goes in one
# forward, so this is fast: roughly 0.3-1.5h per NFE, the full 5-NFE sweep well inside one job.
#
#   sbatch PCCT_experiments/run_pcct_nfe_sweep.sh                       # no-GAN, NFE 1 2 3 5 10
#   sbatch PCCT_experiments/run_pcct_nfe_sweep.sh 3 5                   # just these NFEs
#   TRIAL=imf_gan_unsupervised_PCCT EPOCH=28 sbatch PCCT_experiments/run_pcct_nfe_sweep.sh 2 3
#
# pred is RESUMABLE (samples with a pred_img.nii.gz are skipped), but note that once `avg --cleanup`
# has run for an NFE those per-sample files are gone, so re-running pred for it regenerates all K.
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}
REPO=/gpfs/work/aac/xingyiyao23/Code/IMF_denoising
cd "$REPO"
PRED="$REPO/PCCT_experiments/predict_2D_imf_pcct.py"
EVAL="$REPO/PCCT_experiments/eval_pcct_cnr.py"

TRIAL="${TRIAL:-imf_v2_unsupervised_PCCT}"
EPOCH="${EPOCH:-200}"
ITER="${ITER:-20}"
NFES="${*:-1 2 3 5 10}"

echo "PCCT NFE sweep | trial=$TRIAL epoch=$EPOCH K=$ITER | NFEs=[$NFES]"
mmlsquota --block-size auto 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for NFE in $NFES; do
  echo "================ NFE=$NFE : pred ================"
  python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode pred --num_steps "$NFE" --iteration_num "$ITER" \
    || { echo "pred NFE=$NFE FAILED"; exit 1; }
  echo "================ NFE=$NFE : avg (keep K=10,20 + cleanup) ================"
  python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode avg --num_steps "$NFE" --k_save 10 20 --cleanup \
    || { echo "avg NFE=$NFE FAILED"; exit 1; }
  echo "================ NFE=$NFE : CNR ================"
  python "$EVAL" --trial "$TRIAL" --epoch "$EPOCH" --nfe "$NFE" --k 10 20 \
    || { echo "CNR NFE=$NFE FAILED"; exit 1; }
done
echo "sweep complete: [$NFES]"
echo "results: /gpfs/work/aac/xingyiyao23/projects/denoising/models/$TRIAL/pred_images_nfe*/PCCT_CNR_*.xlsx"
