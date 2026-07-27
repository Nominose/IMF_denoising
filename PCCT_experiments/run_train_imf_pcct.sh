#!/bin/bash
#SBATCH --job-name=imf_pcct
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_pcct_%j.txt

# Stage 1 of the PCCT pipeline: iMF flow training (NO GAN) on real-world PCCT.
# Self-supervised adjacent-slice Noise2Noise; no ground truth is used or needed.
#
# PREREQUISITES (run once, on the login node):
#   bash   PCCT_experiments/unpack_pcct_hpc.sh              # unzip + verify 33 volumes / 8 ROI pairs
#   python PCCT_experiments/fix_pcct_xlsx.py                # preview the rewritten patient list
#   python PCCT_experiments/fix_pcct_xlsx.py --write        # write PCCT_split_hpc.xlsx
#
# 22 train cases x 50 slices x 2 patches = 2200 patches/epoch. At batch 32 that is ~69 updates per
# epoch, ~14k over 200 epochs. Checkpoints every 10 epochs (~571MB each -> ~11GB total): watch the
# quota, and prune down to the epoch you deploy once training is done.
#
# Stage 2 (optional adversarial fine-tune) is gan/run_train_imf_gan_pcct.sh, which needs model-200.pt
# from THIS run. Inference + CNR is PCCT_experiments/run_pcct_nfe_sweep.sh.
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

XLSX=/gpfs/work/aac/xingyiyao23/Data/PCCT/Patient_lists/PCCT_split_hpc.xlsx
[ -f "$XLSX" ] || { echo "patient list missing: $XLSX"; echo "run unpack_pcct_hpc.sh + fix_pcct_xlsx.py --write first"; exit 1; }

mmlsquota --block-size auto 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python PCCT_experiments/train_2D_imf_pcct.py \
  --trial_name imf_v2_unsupervised_PCCT \
  --train_num_steps 200 \
  --train_batch_size 32 \
  --save_every 10
