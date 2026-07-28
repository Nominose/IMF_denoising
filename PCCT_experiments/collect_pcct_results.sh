#!/bin/bash
# collect_pcct_results.sh — gather one PCCT run's results into a single self-contained folder,
# ready to download in one go.
#
# The results normally sit four levels deep and interleaved with every other epoch and NFE
# (pred_images_nfe<N>/<case>/random_<r>/epoch<E>avg/...), which is awkward to fetch selectively.
# This flattens ONE (trial, epoch, NFE) into <out>/<case>/, and copies in the things needed to make
# sense of it without going back to the cluster: the noisy input (the CNR baseline), the GM/WM ROIs
# the CNR was measured over, and the CNR spreadsheet itself.
#
# Copies, never moves -- the originals stay where the pipeline expects them.
#
#   bash PCCT_experiments/collect_pcct_results.sh
#   TRIAL=imf_gan_adv02_PCCT EPOCH=30 NFE=3 bash PCCT_experiments/collect_pcct_results.sh
#   INCLUDE_STD=1 bash PCCT_experiments/collect_pcct_results.sh    # also the per-pixel std maps
#
# Then, from your own machine:
#   rsync -avP xingyiyao23@xpszlogin2.xjtlu.edu.cn:<the path printed at the end>/ ~/Downloads/
set -u

BASE=/gpfs/work/aac/xingyiyao23
M="$BASE/projects/denoising/models"
TRIAL="${TRIAL:-imf_gan_unsupervised_PCCT}"
EPOCH="${EPOCH:-48}"
NFE="${NFE:-3}"
ROI_DIR="${ROI_DIR:-$BASE/Data/PCCT/ROI}"
OUT="${OUT:-$BASE/collected/${TRIAL}_epoch${EPOCH}_nfe${NFE}}"
INCLUDE_STD="${INCLUDE_STD:-0}"        # sample_std.npy is ~52MB/case and rarely needed for viewing

SRC="$M/$TRIAL/pred_images_nfe$NFE"
[ -d "$SRC" ] || { echo "not found: $SRC"; echo "(check TRIAL / NFE)"; exit 1; }

echo "=== collect PCCT results ==="
echo "trial : $TRIAL"
echo "epoch : $EPOCH   NFE: $NFE"
echo "from  : $SRC"
echo "to    : $OUT"
echo

rm -rf "$OUT"; mkdir -p "$OUT"
n_case=0; n_missing=0; missing_list=""

for cdir in "$SRC"/*/; do
  case_id=$(basename "$cdir")
  [ -d "$cdir" ] || continue
  # the avg folder lives under random_<r>/; there is normally exactly one
  avg=$(ls -d "$cdir"random_*/epoch${EPOCH}avg 2>/dev/null | head -1)
  [ -n "$avg" ] || continue

  dst="$OUT/$case_id"; mkdir -p "$dst"
  got=""
  for f in pred_img_scans10.nii.gz pred_img_scans20.nii.gz condition_img.nii.gz; do
    if [ -f "$avg/$f" ]; then
      cp "$avg/$f" "$dst/" && got="$got $f"
    else
      n_missing=$((n_missing + 1)); missing_list="$missing_list $case_id/$f"
    fi
  done
  [ "$INCLUDE_STD" = "1" ] && [ -f "$avg/sample_std.npy" ] && cp "$avg/sample_std.npy" "$dst/"

  # the ROIs the CNR was measured over -- without them the numbers cannot be reproduced
  for r in GM_ROI WM_ROI; do
    [ -f "$ROI_DIR/$case_id/$r.nii.gz" ] && cp "$ROI_DIR/$case_id/$r.nii.gz" "$dst/"
  done

  n_case=$((n_case + 1))
  printf "  %-6s ->%s\n" "$case_id" "$got"
done

# the CNR numbers for this exact run
for x in "$SRC/PCCT_CNR_epoch${EPOCH}_nfe${NFE}.xlsx" "$SRC/PCCT_CNR_sel_epoch${EPOCH}_nfe${NFE}.xlsx"; do
  [ -f "$x" ] && cp "$x" "$OUT/" && echo "  xlsx  -> $(basename "$x")"
done

cat > "$OUT/README.txt" <<EOF
PCCT denoising results
======================
trial      : $TRIAL
epoch      : $EPOCH
NFE        : $NFE
collected  : $(date '+%Y-%m-%d %H:%M')
source     : $SRC

Per case folder:
  pred_img_scans10.nii.gz  denoised, average of K=10 stochastic samples
  pred_img_scans20.nii.gz  denoised, average of K=20 samples (absent if that K was not written)
  condition_img.nii.gz     the NOISY input -- this is the CNR baseline to compare against
  GM_ROI.nii.gz            grey-matter ROI used for CNR
  WM_ROI.nii.gz            white-matter ROI used for CNR

There is no ground truth: this is real clinical PCCT, which is why quality is measured by CNR
over the ROIs rather than by MAE/SSIM/LPIPS.

CNR (Chen's formulation, as used in PCCT_experiments/eval_pcct_cnr.py):
    clip the image to [0, 100] HU, round the ROI masks, then
    CNR = (GM_mean - WM_mean) / sqrt(GM_std^2 + WM_std^2)
pooled over all ROI voxels of the volume.

All volumes are 512 x 512 x 50 and share the same slice indexing, so the ROIs apply directly
to every image in the folder.
EOF

echo
echo "cases collected : $n_case"
if [ "$n_missing" -gt 0 ]; then
  echo "MISSING files   : $n_missing ->$missing_list"
  echo "  (pred_img_scans20 is absent when that NFE was only ever evaluated at K=10)"
fi
echo "total size      : $(du -sh "$OUT" 2>/dev/null | cut -f1)"
echo
echo "folder ready:"
echo "  $OUT"
echo
echo "download it with (run on YOUR machine):"
echo "  rsync -avP xingyiyao23@xpszlogin2.xjtlu.edu.cn:$OUT ~/Downloads/"
