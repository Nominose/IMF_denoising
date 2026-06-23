#!/bin/bash
#SBATCH --job-name=ts_imf_gan
#SBATCH --partition=aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_gan_%j.txt

# EXPERIMENTAL: adversarial fine-tuning of the flow-pretrained iMF v2 generator (model-200).
# L_flow + beta*L_adv; the discriminator pushes the one-step output toward the noisy x2 distribution.
# Inference is unchanged afterward (discriminator discarded). New trial dir, touches nothing existing.
# xlsx / bins / save dir auto-resolve to /gpfs/work/aac/xingyiyao23 via _detect_base()/_he_bins().

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

python gan/train_2D_imf_gan.py \
  --pretrained /gpfs/work/aac/xingyiyao23/projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt
