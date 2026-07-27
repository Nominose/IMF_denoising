#!/bin/bash
#SBATCH --job-name=gan_pcct
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_train_imf_gan_pcct_%j.txt

# Stage 2 of the PCCT pipeline: adversarial fine-tune of the flow model from stage 1.
# REQUIRES imf_v2_unsupervised_PCCT/models/model-200.pt (PCCT_experiments/run_train_imf_pcct.sh) --
# GAN-from-scratch is unstable, so the script hard-fails rather than silently training from noise.
#
# Init = the checkpoint's ONLINE weights (--pretrained_weights default 'model'), matching how the
# brain and Mayo GAN numbers were produced. On brain, EMA init was worth only ~0.5% MAE / ~3% LPIPS,
# which does not justify making PCCT incomparable with the other two datasets.
#
# adv_weight 0.5 (flow loss stays dominant), adv_nfe 1 (single-step F(v) adversary), 50 epochs.
# save_every 10 DIVIDES 50 -> checkpoints at 10/20/30/40/50, i.e. the final epoch is actually saved
# (with save_every 7 the last one would be 49, and model-50.pt would never exist). ~2.9GB total.
# GAN quality is NON-MONOTONE, so pick the deployed epoch by CNR across these checkpoints
# (run_pcct_all.sh does this automatically) rather than assuming the last is best.
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

PRETRAINED="${PRETRAINED:-/gpfs/work/aac/xingyiyao23/projects/denoising/models/imf_v2_unsupervised_PCCT/models/model-200.pt}"
[ -f "$PRETRAINED" ] || { echo "flow model missing: $PRETRAINED"; echo "run PCCT_experiments/run_train_imf_pcct.sh first"; exit 1; }

mmlsquota --block-size auto 2>/dev/null || true

python gan/train_2D_imf_gan_pcct.py \
  --trial_name imf_gan_unsupervised_PCCT \
  --pretrained "$PRETRAINED" \
  --pretrained_weights model \
  --adv_weight 0.5 \
  --train_num_steps 50 \
  --batch_size 16 \
  --save_every 10
