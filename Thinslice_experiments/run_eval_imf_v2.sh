#!/bin/bash
#SBATCH --job-name=ts_imf_v2_eval
#SBATCH --partition=aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=log_eval_imf_v2_%j.txt

# Evaluate the iMF v2 predictions written by run_predict_imf_v2.sh.
# Run AFTER predict (both `pred` and `avg`) has finished for NFE=3 and NFE=5.

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

MODELS=/gpfs/work/aac/xingyiyao23/projects/denoising/models
RESULTS=/gpfs/work/aac/xingyiyao23/projects/denoising/results
mkdir -p $RESULTS

# (1) GT-free x2-reference (N,K) selection check. Dependency-light (numpy/nibabel/pandas), MSE only.
#     Checks argmin_{N,K} R_obs == argmin R_true and that C_noise = R_obs - R_true is (N,K)-constant.
python Thinslice_experiments/eval_x2ref.py --epoch 200 --num_steps 3 5 --k_list 1 10 20 \
  --study_folder $MODELS --out $RESULTS/imf_v2_x2ref.xlsx

# (2) Full metric eval (MAE/SSIM/LPIPS on [0,100] HU). Needs lpips + skimage in the env;
#     if lpips is missing this step errors out but (1) above has already been written.
python Thinslice_experiments/eval_imf_v2.py --epoch 200 --num_steps 3 5 --k_list 1 5 10 20 \
  --study_folder $MODELS --out $RESULTS/imf_v2_brainCT_eval.xlsx
