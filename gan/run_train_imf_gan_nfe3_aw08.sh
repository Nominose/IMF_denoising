#!/bin/bash
#SBATCH --job-name=ts_imf_gan_nfe3_aw08
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_gan_nfe3_aw08_%j.txt

# NFE=3-DIRECT GAN fine-tune of the flow-pretrained iMF v2 generator (model-200), adv_weight=0.8.
# The adversarial fake is the model's TRUE 3-step generation (adv_nfe=3) -> optimises the NFE=3
# LPIPS DIRECTLY, with a STRONG adversarial pull (beta 0.8). The 3-step rollout backprops through
# 3 chained forwards (memory-heavy): batch=24 (~40GB on the 80GB A800) is the safe high-throughput
# point; 32 (~53GB+frag) sits at the ceiling -> drop to 16 if it OOMs. Separate --trial_name so
# it never clobbers the NFE=1 runs or the default 0.5 run. Inference unchanged (D discarded; sample
# at NFE=3). xlsx / bins / save dir auto-resolve to /gpfs/work/aac/xingyiyao23 via _detect_base()/_he_bins().

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

python gan/train_2D_imf_gan_nfe3.py \
  --pretrained /gpfs/work/aac/xingyiyao23/projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt \
  --trial_name imf_gan_nfe3_aw0.8_brainCT \
  --adv_weight 0.8 \
  --adv_nfe 3 \
  --batch_size 24
