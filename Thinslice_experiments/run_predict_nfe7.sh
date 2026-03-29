#!/bin/bash
#SBATCH --job-name=ts_nfe7
#SBATCH --partition=aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_pred_nfe7_%j.txt

module load anaconda3
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

python Thinslice_experiments/predict_2D_imf.py --epoch 30 --mode pred --iteration_num 20 --num_steps 7
python Thinslice_experiments/predict_2D_imf.py --epoch 30 --mode avg --num_steps 7
