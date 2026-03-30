#!/bin/bash
# ============================================================
# SLURM Job Template — XJTLU HPC (SIP)
# User: xingyiyao23
# ============================================================

# ---------- SLURM Configuration ----------
#SBATCH --job-name=JOB_NAME          # Job name (shown in squeue), keep short
#SBATCH --partition=aiaca800          # Partition: aiaca800 or gpua800 (A800 GPU)
#SBATCH --qos=1a800                   # QoS: 1a800 (max 1 running job, 4 in queue)
#SBATCH --gpus=1                      # Number of GPUs (max 1 for 1a800 QoS)
#SBATCH --time=72:00:00               # Max wall time (max 7 days for 1a800)
#SBATCH --mem=64G                     # Memory (A800 node max ~500G, 64G usually enough)
#SBATCH --output=log_%x_%j.txt        # Log file: %x=job name, %j=job ID

# ---------- Environment Setup ----------
# NOTE: `module load anaconda3 && conda activate` does NOT work in sbatch.
#       Must source conda.sh first to initialize conda in non-interactive shell.
source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm                  # Python 3.10, PyTorch 2.5.1+cu121

# Add parent Code directory to PYTHONPATH so imports like
# `import Diffusion_denoising_thin_slice.*` and `import IMF_denoising.*` work
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH

# cd to project root
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

# ---------- Run ----------
# Replace with your actual command, examples:
#
# [Training - Thinslice]
# python Thinslice_experiments/train_2D_imf.py
#
# [Training - CT]
# python CT_experiments/train_2D_imf.py
#
# [Predict - Thinslice, NFE=2, 20 iterations then average]
# python Thinslice_experiments/predict_2D_imf.py --epoch 30 --mode pred --iteration_num 20 --num_steps 2
# python Thinslice_experiments/predict_2D_imf.py --epoch 30 --mode avg --num_steps 2
#
# [Predict - CT]
# python CT_experiments/predict_2D_imf.py --epoch 30 --mode pred --iteration_num 20 --num_steps 2
# python CT_experiments/predict_2D_imf.py --epoch 30 --mode avg --num_steps 2

python YOUR_SCRIPT.py --YOUR_ARGS

# ============================================================
# Quick Reference:
#   Submit:    sbatch this_script.sh
#   Monitor:   squeue -u xingyiyao23
#   Cancel:    scancel <JOBID>
#   Cancel all: scancel -u xingyiyao23
#   View log:  cat log_JOB_NAME_<JOBID>.txt
#   Interactive debug:
#     srun --partition=aiaca800 --qos=1a800 --gpus=1 --time=01:00:00 --pty bash
# ============================================================
