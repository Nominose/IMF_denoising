#!/bin/bash
#SBATCH --job-name=ts_imf_mayo
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_mayo_%j.txt

# Mayo low-dose CT iMF (NO GAN) flow training. Self-supervised Noise2Noise: predict recon_even from
# recon_odd (two independent noisy halves of the SAME slice) -- no clean image used. Abdomen HU
# window [-200, 250], no histogram-eq, patch 256x256, 1-channel condition. Backbone = the brain-CT
# v2 U-Net (+aux v-head) + ImprovedMeanFlow. Data must be uploaded to
# /gpfs/work/aac/xingyiyao23/新mayo_data/ (xlsx + simulation_highnoise_v2/); _detect_base()/_remap()
# resolve the xlsx-stored E:-drive paths (/host/e/D/Data/low_dose_CT/...) onto gpfs. Models ->
# projects/denoising/models/<trial_name>/models/.

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

# batch=1 is the script default (a single-step flow forward is light -> the 80GB A800 is way
# under-used). Raise --train_batch_size for throughput; from-scratch flow over 200 epochs means a
# bigger batch = fewer updates/epoch, so bump --train_num_steps too if you grow the batch a lot.
python CT_experiments/train_2D_imf_mayo.py \
  --trial_name imf_v2_unsupervised_gaussian_mayo \
  --train_num_steps 200
