#!/bin/bash
#SBATCH --job-name=ts_gan_mayo_aw02
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_gan_mayo_adv02_%j.txt

# Mayo iMF+GAN fine-tune at adv_weight=0.2 (the trainer default is 0.5) -> a SEPARATE trial dir
# imf_gan_adv02_unsupervised_gaussian_mayo, so the 0.2 checkpoints never collide with the 0.5 run
# (imf_gan_unsupervised_gaussian_mayo, launched by run_train_imf_gan_mayo.sh). SLURM counterpart of
# the docker wrapper gan/run_gan_mayo_adv02.sh: SAME recipe, only --adv_weight + --trial_name change.
# No code is duplicated — train_2D_imf_gan_mayo.py already exposes --adv_weight.
#
# Why 0.2: at adv=0.5 the Mayo GAN adds COARSE texture that slightly HURTS vs the no-GAN flow at
# every K (converges toward but never beats noGAN on LPIPS/MAE/SSIM). adv_weight scales the AMOUNT
# of that texture (L_G = L_flow + adv_weight*L_adv), so a lighter 0.2 adds less of it and is EXPECTED
# to sit closer to / break even with noGAN, not surpass it (the texture *direction* is unchanged).
# This run confirms that empirically before deciding the GAN's place in the paper.
#
# Data + pretrained flow model (imf_v2_unsupervised_gaussian_mayo/model-200.pt, confirmed present)
# auto-resolve to /gpfs/work/aac/xingyiyao23 via the trainer's _detect_base()/_remap().
# Defaults otherwise unchanged: 50 epochs, batch 2, adv_nfe 1, save every epoch (GAN quality is
# non-monotone -> pick the best epoch from fv_evolution/ + a K-sweep LPIPS vs no-GAN, then prune).

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

python gan/train_2D_imf_gan_mayo.py \
  --trial_name imf_gan_adv02_unsupervised_gaussian_mayo \
  --adv_weight 0.2 \
  --pretrained /gpfs/work/aac/xingyiyao23/projects/denoising/models/imf_v2_unsupervised_gaussian_mayo/models/model-200.pt
