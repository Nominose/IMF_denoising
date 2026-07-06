#!/bin/bash
#SBATCH --job-name=ts_imf_gan_mayo
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_gan_mayo_%j.txt

# GAN fine-tuning of the Mayo flow-pretrained iMF v2 generator (imf_v2_unsupervised_gaussian_mayo/
# model-200.pt). Two-stage: L_flow + beta*L_adv; an UNCONDITIONAL high-pass PatchGAN D pushes the
# few-step generation toward the real noisy even-recon texture (better low-NFE LPIPS). Inference is
# UNCHANGED afterward (D discarded; sample with CT_experiments/predict_2D_imf.py). Writes a NEW trial
# dir imf_gan_unsupervised_gaussian_mayo/, touches nothing existing. Data (xlsx + recon volumes) and
# the pretrained model auto-resolve to /gpfs/work/aac/xingyiyao23 via _detect_base()/_remap().
#
# PREREQ: the Mayo flow run finished and left
#   .../projects/denoising/models/imf_v2_unsupervised_gaussian_mayo/models/model-200.pt
# (verify with `ls`; if the best flow epoch differs, edit --pretrained below to model-<E>.pt).
#
# KNOBS (defaults are the proven brain recipe): --batch_size 2 (raise for throughput/VRAM, but it
# changes GAN dynamics), --adv_weight 0.5, --adv_nfe 1 (single-step F(v); 3 = heavier NFE=3-direct),
# --train_num_steps 50 (epochs), --save_every 1 (GAN quality is non-monotone -> checkpoint every
# epoch, prune bad ones later). Pick the best epoch from fv_evolution/ + a short predict/eval sweep.

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

python gan/train_2D_imf_gan_mayo.py \
  --pretrained /gpfs/work/aac/xingyiyao23/projects/denoising/models/imf_v2_unsupervised_gaussian_mayo/models/model-200.pt
