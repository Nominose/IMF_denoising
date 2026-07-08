#!/bin/bash
#SBATCH --job-name=ts_gan_mayo_aw08
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_gan_mayo_adv08_%j.txt

# Mayo iMF+GAN fine-tune at adv_weight=0.8 (the trainer default is 0.5) -> a SEPARATE trial dir
# imf_gan_adv08_unsupervised_gaussian_mayo, so the 0.8 checkpoints never collide with the 0.5 / 0.2
# runs. Sibling of run_train_imf_gan_mayo.sh (0.5) and run_train_imf_gan_mayo_adv02.sh (0.2): SAME
# recipe, only --adv_weight + --trial_name change. No code duplicated — the trainer exposes --adv_weight.
#
# Why 0.8: this is the HEAVY end of the adv_weight bracket (0.2 / 0.5 / 0.8). adv_weight scales the
# AMOUNT of the (coarse) GAN texture in L_G = L_flow + adv_weight*L_adv. On Mayo that texture already
# slightly HURTS at 0.5 vs the no-GAN flow, so 0.8 is EXPECTED to move FURTHER from noGAN (more
# texture), not closer — run it to map the upper end of the trade-off and check whether heavier adv
# ever wins at the very lowest NFE (sharper texture) despite worse MAE/SSIM. Bracketing the axis
# (0.2 vs 0.5 vs 0.8) is what tells the paper where, if anywhere, the GAN helps.
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
  --trial_name imf_gan_adv08_unsupervised_gaussian_mayo \
  --adv_weight 0.8 \
  --pretrained /gpfs/work/aac/xingyiyao23/projects/denoising/models/imf_v2_unsupervised_gaussian_mayo/models/model-200.pt
