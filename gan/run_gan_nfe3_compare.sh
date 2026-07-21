#!/bin/bash
#SBATCH --job-name=gan_nfe3_cmp
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_gan_nfe3_compare_%j.txt

# NFE=3 three-way A/B for brain CT: fills in the two arms the EMA-init GAN result needs to be
# interpretable, then prints all three RESULT blocks together.
#
#   no-GAN baseline   imf_v2_unsupervised_gaussian_brainCT      @ epoch 200
#   online-init GAN   imf_gan_unsupervised_gaussian_brainCT     @ epoch 28
#   EMA-init GAN      imf_gan_ema_unsupervised_gaussian_brainCT @ epoch 28   (already generated)
#
# The existing GAN sweep is NFE 5/10/20/30/50, so neither the online GAN nor the baseline has an
# NFE=3 result yet -- this generates them (pred K=20 -> avg keep K=10,20 + cleanup), then re-evals
# all three (the EMA arm's scans10/20 already exist, so its eval is free). Same predict + eval
# scripts, brain window [0,100] HU. Brain GT is written by the predict, so no upload needed.
#
# Comparison logic (what each contrast isolates):
#   online GAN  vs baseline  -> the GAN as it was actually run (adv loss + extra epochs + worse init)
#   EMA   GAN   vs baseline  -> the GAN starting from the SAME weights the baseline is scored on
#   EMA   GAN   vs online GAN-> the effect of the init change alone (both @ epoch 28, only init differs)
#
# RESUMABLE: pred skips samples whose pred_img.nii.gz exists. Peak disk ~3G per arm (cleaned after).
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}
REPO=/gpfs/work/aac/xingyiyao23/Code/IMF_denoising
cd "$REPO"
PRED="$REPO/Thinslice_experiments/predict_2D_imf_v2.py"
NFE="${NFE:-3}"
ITER="${ITER:-20}"

gen () {  # $1 = trial, $2 = epoch : generate NFE preds if not already there
  echo "======== generate: $1 @ epoch $2, NFE=$NFE ========"
  python "$PRED" --trial_name "$1" --epoch "$2" --mode pred --num_steps "$NFE" --iteration_num "$ITER" \
    || { echo "$1 pred FAILED"; exit 1; }
  python "$PRED" --trial_name "$1" --epoch "$2" --mode avg  --num_steps "$NFE" --cleanup \
    || { echo "$1 avg FAILED"; exit 1; }
}

mmlsquota --block-size auto 2>/dev/null || true

# EMA arm is already done (run_gan_ema_brainct_nfe3.sh); generate the two missing arms.
gen imf_gan_unsupervised_gaussian_brainCT 28    # online-init GAN
gen imf_v2_unsupervised_gaussian_brainCT  200   # no-GAN baseline

echo
echo "############### NFE=$NFE  three-way comparison (brain [0,100] HU, K=10/20) ###############"
echo "=== no-GAN baseline  (imf_v2 @200) ==="
python gan/eval_gan_nfe.py --trial imf_v2_unsupervised_gaussian_brainCT      --epoch 200 --nfe "$NFE"
echo "=== online-init GAN  (imf_gan @28) ==="
python gan/eval_gan_nfe.py --trial imf_gan_unsupervised_gaussian_brainCT     --epoch 28  --nfe "$NFE"
echo "=== EMA-init GAN     (imf_gan_ema @28) ==="
python gan/eval_gan_nfe.py --trial imf_gan_ema_unsupervised_gaussian_brainCT --epoch 28  --nfe "$NFE"
echo "############### done ###############"
