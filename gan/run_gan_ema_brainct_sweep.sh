#!/bin/bash
#SBATCH --job-name=gan_ema_sweep
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_gan_ema_brainct_sweep_%j.txt

# Full NFE sweep for the EMA-init brain-CT GAN, then flatten the results into the shareable tree.
#
# The trial imf_gan_ema_unsupervised_gaussian_brainCT was fine-tuned from the flow checkpoint's EMA
# weights instead of its online ones, but only NFE=3 was ever scored (MAE 2.598 / SSIM 0.763 /
# LPIPS 0.0485 at K=10), and that prediction folder was since reclaimed. This regenerates every NFE
# in the reported table so the EMA-init row can sit beside the online-init one.
#
# Per NFE: pred (K=20 stochastic samples) -> avg (writes K=1,10,20) -> eval. K=1 is the
# single-sample row of the tables and needs the --k flag added to eval_gan_nfe.py; --cleanup then
# drops the per-sample volumes so disk stays flat at roughly one NFE's worth (~10GB).
#
# Finally tmp/collect_results.py flattens
#     pred_images_nfe<N>/<patient>/<subid>/random_<r>/epoch28avg/pred_img_scans<K>.nii.gz
# into the layout the result folders use:
#     <OUT>/pred_images_NFE<N>/<patient>/epoch28avg/pred_img_scans<K>.nii.gz
# Hardlinked, so the flattened tree costs no extra disk (pass COPY=1 for real copies).
#
# RESUMABLE: pred skips samples that already exist, and an NFE whose eval already ran is skipped
# entirely, so a job killed by the wall clock resumes with `sbatch` alone.
#
# Cost: 16 cases x 20 samples per NFE, roughly 6s + 1.56s*NFE per sample -> about 20h for the full
# 2 3 5 10 20 30 50 list; the high NFEs dominate. Trim with NFES if only part of the table is wanted.
#
#   sbatch gan/run_gan_ema_brainct_sweep.sh
#   NFES="2 3 5" sbatch gan/run_gan_ema_brainct_sweep.sh
#   TRIAL=imf_gan_unsupervised_gaussian_brainCT EPOCH=28 sbatch gan/run_gan_ema_brainct_sweep.sh
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}
REPO=/gpfs/work/aac/xingyiyao23/Code/IMF_denoising
cd "$REPO"

MODELS=/gpfs/work/aac/xingyiyao23/projects/denoising/models
PRED="$REPO/Thinslice_experiments/predict_2D_imf_v2.py"
EVAL="$REPO/gan/eval_gan_nfe.py"
COLLECT="$REPO/tmp/collect_results.py"

TRIAL="${TRIAL:-imf_gan_ema_unsupervised_gaussian_brainCT}"
EPOCH="${EPOCH:-28}"
NFES="${NFES:-2 3 5 10 20 30 50}"
ITER="${ITER:-20}"
KS="${KS:-1 10 20}"
OUT="${OUT:-/gpfs/work/aac/xingyiyao23/collected/${TRIAL}_epoch${EPOCH}}"
COPY="${COPY:-0}"

CKPT="$MODELS/$TRIAL/models/model-$EPOCH.pt"
echo "================ EMA-init brain GAN sweep ================"
echo "trial : $TRIAL"
echo "epoch : $EPOCH   NFEs: [$NFES]   K: [$KS]   samples/case: $ITER"
echo "out   : $OUT"
date '+%F %T'
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mmlsquota --block-size auto 2>/dev/null | tail -1

for NFE in $NFES; do
  echo; echo "######## NFE=$NFE ########"; date '+%F %T'
  AVG_ONE=$(ls -d "$MODELS/$TRIAL/pred_images_nfe$NFE"/*/*/random_*/epoch${EPOCH}avg 2>/dev/null | head -1)
  if [ -n "$AVG_ONE" ] && [ -f "$AVG_ONE/pred_img_scans20.nii.gz" ]; then
    echo "[skip] NFE=$NFE already averaged"
  else
    python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode pred \
      --num_steps "$NFE" --iteration_num "$ITER" || { echo "pred NFE=$NFE FAILED"; exit 1; }
    python "$PRED" --trial_name "$TRIAL" --epoch "$EPOCH" --mode avg \
      --num_steps "$NFE" --k_save $KS --cleanup || { echo "avg NFE=$NFE FAILED"; exit 1; }
  fi
  echo "---- metrics NFE=$NFE (brain window [0,100] HU) ----"
  python "$EVAL" --trial "$TRIAL" --epoch "$EPOCH" --nfe "$NFE" --k $KS \
    || { echo "eval NFE=$NFE FAILED"; exit 1; }
  mmlsquota --block-size auto 2>/dev/null | tail -1
done

echo; echo "######## collect into the shareable tree ########"
COPYFLAG=""; [ "$COPY" = "1" ] && COPYFLAG="--copy"
python "$COLLECT" --trial "$TRIAL" --epoch "$EPOCH" --nfe $NFES --k $KS \
  --models_root "$MODELS" --out "$OUT" $COPYFLAG || { echo "collect FAILED"; exit 1; }

echo; echo "######## summary ########"
grep -E "^RESULT|^nfe=" "$SLURM_SUBMIT_DIR/log_gan_ema_brainct_sweep_${SLURM_JOB_ID}.txt" 2>/dev/null \
  | sed 's/^/  /' || true
echo
echo "result tree : $OUT"
echo "layout      : pred_images_NFE<N>/<patient>/epoch${EPOCH}avg/pred_img_scans<K>.nii.gz"
echo "download    : rsync -avP xingyiyao23@xpszlogin2.xjtlu.edu.cn:$OUT ~/Downloads/"
date '+%F %T'
