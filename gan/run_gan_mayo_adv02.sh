#!/usr/bin/env bash
# run_gan_mayo_adv02.sh — Mayo iMF+GAN fine-tune at adv_weight=0.2 (the default trainer uses 0.5).
#
# Thin wrapper over gan/train_2D_imf_gan_mayo.py: SAME recipe, but a lighter adversarial weight
# and a SEPARATE trial dir so the 0.2 checkpoints never collide with the 0.5 run's
# (imf_gan_unsupervised_gaussian_mayo). No code is duplicated — the trainer already exposes
# --adv_weight; this just fixes it to 0.2 for a reproducible run.
#
# Why: on Mayo the adv=0.5 GAN adds COARSE texture that slightly HURTS vs the no-GAN flow at every
# K — it converges toward but never beats noGAN (LPIPS/MAE/SSIM). adv_weight scales the AMOUNT of
# that texture (L_G = L_flow + adv_weight * L_adv), so a lighter 0.2 adds less of it and is
# EXPECTED to sit closer to (break even with) noGAN, not surpass it — the texture *direction* is
# unchanged. This run is to confirm that empirically before deciding the GAN's place in the paper.
#
# Run inside docker (tmux), on the GPU that trained the Mayo flow model:
#   bash gan/run_gan_mayo_adv02.sh
#   bash gan/run_gan_mayo_adv02.sh --train_num_steps 30 --batch_size 2   # extra args pass through
#   TRIAL=my_name bash gan/run_gan_mayo_adv02.sh                          # override the trial dir
#
# Then pick the best epoch from fv_evolution/ + a short predict/eval sweep, exactly like the 0.5 run
# (fv HF-match is only a proxy — confirm the pick with an actual K-sweep LPIPS vs the no-GAN flow).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"                              # .../IMF_denoising
TRIAL="${TRIAL:-imf_gan_adv02_unsupervised_gaussian_mayo}"

echo "Mayo iMF+GAN fine-tune | adv_weight=0.2 | trial=$TRIAL"
python "$REPO/gan/train_2D_imf_gan_mayo.py" \
    --trial_name "$TRIAL" \
    --adv_weight 0.2 \
    "$@"
