#!/bin/bash
#SBATCH --job-name=ts_gan_mayo_aw00
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_gan_mayo_adv0_%j.txt

# Mayo iMF+GAN adv_weight=0 -- the CONTROL run. Separate trial dir imf_gan_adv0_unsupervised_gaussian_mayo.
# Sibling of run_train_imf_gan_mayo_adv0{2,8}.sh and run_train_imf_gan_mayo.sh (0.5): SAME recipe,
# only --adv_weight + --trial_name change.
#
# WHY THIS RUN EXISTS (it is not a sweep point -- it is the control the comparison needs):
# the GAN fine-tune gives the generator 50 extra epochs that the no-GAN baseline never had. So a
# GAN-vs-no-GAN difference has TWO possible causes -- the adversarial loss, or just the extra
# training -- and you cannot separate them without this run. With it:
#     adv0  vs no-GAN baseline  -> isolates the effect of the EXTRA EPOCHS alone
#     adv{0.2,0.5,0.8} vs adv0  -> isolates the ADVERSARIAL LOSS itself
# Without adv0 the whole sweep can only say "GAN differs from baseline", not "the adversarial loss
# did it" -- which is the claim the paper actually needs.
#
# It is an EXACT control, not an approximation: imf_gan.py computes
#     g_loss = loss_forward + self.adv_weight * g_adv
# so at adv_weight=0 the adversarial term contributes exactly zero gradient to G -- G trains on the
# pure flow loss. D still trains and g_adv is still computed, so wall-clock and data order match the
# other runs bit-for-bit; only the adversarial contribution is off. That is what makes it comparable.
#
# KEEP --pretrained_weights THE SAME ACROSS ALL FOUR RUNS (0 / 0.2 / 0.5 / 0.8) or the sweep is not
# internally comparable. Default 'model' (= online weights) reproduces the existing tables; 'ema'
# starts from the better model that the no-GAN baseline is actually scored on and removes the init
# mismatch, but then the baseline side must be re-scored too. See gan/imf_gan.py::load_generator.

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

python gan/train_2D_imf_gan_mayo.py \
  --trial_name imf_gan_adv0_unsupervised_gaussian_mayo \
  --adv_weight 0 \
  --pretrained /gpfs/work/aac/xingyiyao23/projects/denoising/models/imf_v2_unsupervised_gaussian_mayo/models/model-200.pt
