#!/bin/bash
#SBATCH --job-name=ts_imf_gan_aw08
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_gan_aw08_%j.txt

# NFE=1 GAN fine-tune of the flow-pretrained iMF v2 generator (model-200), adv_weight=0.8.
# Identical to run_train_imf_gan.sh EXCEPT a STRONGER adversarial pull: beta 0.5 -> 0.8 (flow still
# present but the D pushes the single-step F(v) harder toward the real noisy-x2 texture). adv_nfe=1
# = the cheapest NFE=1 inference object. Separate --trial_name so it never clobbers the 0.5 run.
# Inference is unchanged afterward (discriminator discarded). xlsx / bins / save dir auto-resolve
# to /gpfs/work/aac/xingyiyao23 via _detect_base()/_he_bins().

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

python gan/train_2D_imf_gan.py \
  --pretrained /gpfs/work/aac/xingyiyao23/projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt \
  --trial_name imf_gan_aw0.8_nfe1_brainCT \
  --adv_weight 0.8 \
  --adv_nfe 1
