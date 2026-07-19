#!/bin/bash
#SBATCH --job-name=gan_ema_nfe3
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_gan_ema_brainct_nfe3_%j.txt

# Brain-CT iMF+GAN fine-tune from the flow checkpoint's EMA weights, then NFE=3 inference on the
# SAME epoch (28) + metrics -- all in one resubmit-safe job.
#
# WHY EMA init: the no-GAN baseline is scored on the flow model's EMA weights, but the GAN used to
# start from its ONLINE weights (the worse model at deployment K: brain 214841 K=10 MAE 2.156 online
# vs 1.997 EMA). Starting the GAN from EMA puts it on the same footing as the baseline, so a
# GAN-vs-baseline gap is not just an init handicap. Uses --pretrained_weights ema (gan/imf_gan.py
# ::load_generator filters the ema_model.* prefix). Separate trial dir -> the existing online-init
# GAN results are untouched and stay reproducible.
#
# WHY epoch 28: that is the epoch the original online GAN sweep evaluated (run_gan_nfe_sweep.sh
# EPOCH=28), so EMA@28 vs online@28 is a controlled A/B -- only the init differs. The GAN LR is a
# plain constant Adam (no schedule tied to train_num_steps), so model-28.pt is identical whether we
# train 28 or 50 epochs -> we train exactly 28 and stop (saves ~44% GPU). save_every=7 -> keeps
# epochs 7/14/21/28 (~2.3GB) as a fallback in case 28 is not the best (GAN quality is non-monotone);
# the per-epoch fv_evolution/ dumps let you eyeball all 28 regardless.
#
# RESUBMIT-SAFE: if model-28.pt already exists the training step is skipped, and `pred` skips samples
# whose pred_img.nii.gz exists -> a job killed mid-inference resumes without retraining.
#
# Pretrained flow model + xlsx/bins auto-resolve to /gpfs/work/aac/xingyiyao23. Metrics are printed
# by eval_gan_nfe.py (brain window [0,100] HU, K=10/20) -- brain GT is written by the predict, so no
# extra upload is needed (unlike the Mayo sweep).
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}
REPO=/gpfs/work/aac/xingyiyao23/Code/IMF_denoising
cd "$REPO"

TRIAL="${TRIAL:-imf_gan_ema_unsupervised_gaussian_brainCT}"
EPOCH="${EPOCH:-28}"
NFE="${NFE:-3}"
ITER="${ITER:-20}"
PRETRAINED="${PRETRAINED:-/gpfs/work/aac/xingyiyao23/projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt}"
CKPT="/gpfs/work/aac/xingyiyao23/projects/denoising/models/$TRIAL/models/model-$EPOCH.pt"
PRED="$REPO/Thinslice_experiments/predict_2D_imf_v2.py"

echo "=== EMA-init brain GAN -> NFE=$NFE @ epoch $EPOCH | trial=$TRIAL ==="
[ -f "$PRETRAINED" ] || { echo "pretrained flow model not found: $PRETRAINED"; exit 1; }
mmlsquota --block-size auto 2>/dev/null || true

# ---- 1) train (EMA init), 28 epochs. Skip if the target checkpoint already exists. ----
if [ -f "$CKPT" ]; then
  echo "[train] $CKPT exists -> skip training"
else
  python gan/train_2D_imf_gan.py \
    --trial_name "$TRIAL" \
    --pretrained "$PRETRAINED" \
    --pretrained_weights ema \
    --train_num_steps "$EPOCH" \
    --save_every 7 \
    || { echo "train FAILED"; exit 1; }
fi
[ -f "$CKPT" ] || { echo "no $CKPT after training -- aborting"; exit 1; }

# ---- 2) inference at NFE=3 on epoch 28: pred (K=20) -> avg (keep K=10,20) + cleanup ----
echo "================ NFE=$NFE : pred ================"
python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode pred --num_steps "$NFE" --iteration_num "$ITER" \
  || { echo "pred FAILED"; exit 1; }
echo "================ NFE=$NFE : avg (keep K=10,20 + cleanup) ================"
python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode avg --num_steps "$NFE" --cleanup \
  || { echo "avg FAILED"; exit 1; }

# ---- 3) metrics (brain window [0,100] HU, K=10/20, mean+-std over the test set) ----
echo "================ metrics ================"
python gan/eval_gan_nfe.py --trial "$TRIAL" --epoch "$EPOCH" --nfe "$NFE"
echo "done: $TRIAL epoch $EPOCH nfe $NFE"
echo "compare vs online GAN + no-GAN baseline at the SAME NFE with:"
echo "  python gan/eval_gan_nfe.py --trial imf_gan_unsupervised_gaussian_brainCT --epoch 28 --nfe $NFE   # (needs its nfe$NFE preds)"
