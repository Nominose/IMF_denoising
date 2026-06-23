#!/bin/bash
#SBATCH --job-name=ts_imf_v2_pred
#SBATCH --partition=aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_pred_imf_v2_%j.txt

# Inference for the iMF v2 (aux v-head) checkpoint model-200.
# Builds the U-Net WITH auxiliary_v_head=True so model-200 strict-loads.
# Runs `pred` (generate K stochastic samples) then `avg` (cumulative average) for NFE=3 and NFE=5.

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

DATA=/gpfs/work/aac/xingyiyao23/Data
MODELS=/gpfs/work/aac/xingyiyao23/projects/denoising/models
XLSX=$DATA/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx
BINS=$DATA/histogram_equalization/bins.npy
BINSM=$DATA/histogram_equalization/bins_mapped.npy

# HPC paths passed explicitly so the script (whose defaults point at the docker /host/d mount)
# resolves to gpfs without editing the .py.
COMMON="--study_folder $MODELS --patient_list_file $XLSX --bins $BINS --bins_mapped $BINSM"

# K=20 stochastic samples per case; k_save writes the cumulative averages eval needs (k=1,5,10,20).
for NFE in 3 5; do
  python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode pred --iteration_num 20 --num_steps $NFE $COMMON
  python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode avg  --num_steps $NFE --k_save 1 5 10 20 $COMMON
done
