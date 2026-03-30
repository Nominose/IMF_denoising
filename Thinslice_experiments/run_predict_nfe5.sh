#!/bin/bash
#SBATCH --job-name=ts_nfe5
#SBATCH --partition=aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_pred_nfe5_%j.txt

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

python Thinslice_experiments/predict_2D_imf.py --epoch 30 --mode pred --iteration_num 20 --num_steps 5
python Thinslice_experiments/predict_2D_imf.py --epoch 30 --mode avg --num_steps 5
