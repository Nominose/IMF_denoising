#!/bin/bash
#SBATCH --job-name=ts_imf_v2_sweep
#SBATCH --partition=aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_imf_v2_sweep_%j.txt

# iMF v2 (v-head) NFE sweep on brain CT — HPC version of the docker loop.
# For each NFE: generate K=20 stochastic samples (pred), cumulative-average to k=10/20 (avg),
# then eval MAE/SSIM/LPIPS vs GT. gpfs paths are passed explicitly because the script defaults
# point at the docker /host/d mount.

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

DATA=/gpfs/work/aac/xingyiyao23/Data
MODELS=/gpfs/work/aac/xingyiyao23/projects/denoising/models
BASE=$MODELS/imf_v2_unsupervised_gaussian_brainCT
XLSX=$DATA/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx
BINS=$DATA/histogram_equalization/bins.npy
BINSM=$DATA/histogram_equalization/bins_mapped.npy

# slice_batch 16 = batch slices per GPU forward (A800 has the headroom); auto-halves on CUDA OOM.
# Result is numerically identical to slice_batch=1 (each slice keeps its own init noise).
PRED_COMMON="--study_folder $MODELS --patient_list_file $XLSX --bins $BINS --bins_mapped $BINSM --slice_batch 16"

# ---- guards: fail clearly instead of 5x FileNotFoundError / silent missing eval ----
CKPT=$BASE/models/model-200.pt
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT — has the 200-epoch training finished?"; exit 1
fi
if python -c "import lpips, skimage" 2>/dev/null; then
  echo "[precheck] lpips + skimage OK"
else
  echo "[precheck] WARNING: lpips/skimage missing -> eval steps will fail."
  echo "           predictions + scans10/20 are still written, so you can re-run eval later"
  echo "           after: pip install lpips scikit-image"
fi

# NOTE: --cleanup deletes the per-sample volumes after averaging (keeps only scans10/20 + gt + std).
#       This run is self-consistent (k_save 10 20 == eval k_list 10 20), but you will NOT be able to
#       compute other K (e.g. k=1 single-sample) later without re-running pred. Drop --cleanup to keep them.
for NFE in 2 10 20 30 50; do
  echo "===========  NFE=$NFE  $(date)  ==========="
  python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode pred --iteration_num 20 --num_steps $NFE $PRED_COMMON
  python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode avg  --num_steps $NFE --k_save 10 20 --cleanup $PRED_COMMON
  python Thinslice_experiments/eval_imf_v2.py --epoch 200 --num_steps $NFE --k_list 10 20 \
         --study_folder $MODELS --out "$BASE/pred_images_nfe$NFE/eval_nfe$NFE.xlsx" \
    2>&1 | tee "$BASE/pred_images_nfe$NFE/eval_nfe$NFE.log"
  echo "----  NFE=$NFE done  $(date)  ----"
done

# combined table across all NFE (reads the kept scans10/20; cheap, gives one side-by-side SUMMARY)
echo "===========  combined eval  $(date)  ==========="
python Thinslice_experiments/eval_imf_v2.py --epoch 200 --num_steps 2 10 20 30 50 --k_list 10 20 \
       --study_folder $MODELS --out "$BASE/eval_all_nfe.xlsx" \
  2>&1 | tee "$BASE/eval_all_nfe.log"
echo "ALL DONE $(date)"
