#!/bin/bash
# unpack_pcct_hpc.sh — unpack + VERIFY the PCCT dataset on the HPC cluster.
#
# Input  (already uploaded): /gpfs/work/aac/xingyiyao23/Data/{soft_thins_xy.zip, ROI.zip}
# Output (this script):      /gpfs/work/aac/xingyiyao23/Data/PCCT/
#                              soft_thins_xy/<case>/soft_thins_0_noblank_sliced.nii.gz   (33 cases)
#                              ROI/<case>/{GM_ROI,WM_ROI}.nii.gz                         (8 cases, 29-36)
#
# The zips may contain the case folders either at the top level (soft_thins_xy/29/...) or nested one
# level deeper (soft_thins_xy/soft_thins_xy/29/...), depending on how they were created -- this
# script auto-detects which and flattens to a single known layout, so the xlsx paths are predictable.
#
# It VERIFIES rather than assuming: 33 image cases, 8 ROI cases, and that every ROI case has both
# GM and WM. Anything off is printed loudly, because a silently-missing case would only surface much
# later as a wrong CNR (ROI) or a skipped patient (image).
#
# Run on the login node (small, no GPU needed):   bash PCCT_experiments/unpack_pcct_hpc.sh
set -u

BASE="${BASE:-/gpfs/work/aac/xingyiyao23}"
SRC="$BASE/Data"
DST="$BASE/Data/PCCT"

echo "=== PCCT unpack ==="
echo "source: $SRC   ->   dest: $DST"
df_quota () { mmlsquota --block-size auto 2>/dev/null | tail -2 || true; }
df_quota

for z in soft_thins_xy.zip ROI.zip; do
  [ -f "$SRC/$z" ] || { echo "MISSING: $SRC/$z"; exit 1; }
done

mkdir -p "$DST"
TMP="$DST/.unzip_tmp"
rm -rf "$TMP"; mkdir -p "$TMP"

# ---- unzip both into a scratch dir, then normalise the layout ----
for z in soft_thins_xy ROI; do
  echo; echo "--- unzip $z.zip ---"
  rm -rf "${TMP:?}/$z"; mkdir -p "$TMP/$z"
  # -x __MACOSX/*: zips made on macOS carry a parallel __MACOSX/ tree of AppleDouble stubs
  # (._GM_ROI.nii.gz, a few hundred bytes each). It mirrors the real folder names, so without this
  # the layout probe below can lock onto __MACOSX/ROI/29 and "find" cases that hold no real data.
  unzip -q -o "$SRC/$z.zip" -x '__MACOSX/*' -d "$TMP/$z" || { echo "unzip $z FAILED"; exit 1; }
  find "$TMP/$z" -name '._*' -delete 2>/dev/null || true

  # Find the directory that actually CONTAINS the numeric case folders (handles both the flat and
  # the nested layout). Pick the candidate holding the most real .nii.gz, so any decoy loses.
  root=$(find "$TMP/$z" -type d -regex '.*/[0-9]+$' -not -path '*__MACOSX*' -printf '%h\n' 2>/dev/null \
         | sort -u \
         | while read -r cand; do
             echo "$(find "$cand" -name '*.nii.gz' -not -name '._*' | wc -l) $cand"
           done | sort -rn | head -1 | cut -d' ' -f2-)
  [ -n "$root" ] || { echo "no numeric case folders found inside $z.zip"; exit 1; }
  echo "case folders found under: $root  ($(find "$root" -name '*.nii.gz' -not -name '._*' | wc -l) volumes)"

  rm -rf "${DST:?}/$z"
  mv "$root" "$DST/$z" || { echo "move FAILED"; exit 1; }
done
rm -rf "$TMP"

# ---- verify ----
echo; echo "=== verify ==="
IMG=$(find "$DST/soft_thins_xy" -name 'soft_thins_0_noblank_sliced.nii.gz' | wc -l)
GM=$(find "$DST/ROI" -name 'GM_ROI.nii.gz' | wc -l)
WM=$(find "$DST/ROI" -name 'WM_ROI.nii.gz' | wc -l)
echo "image volumes : $IMG   (expected 33: cases 1-36 minus 8,10,13)"
echo "GM ROI        : $GM    (expected 8: cases 29-36)"
echo "WM ROI        : $WM    (expected 8)"

echo; echo "image cases : $(ls "$DST/soft_thins_xy" | sort -n | tr '\n' ' ')"
echo "ROI cases   : $(ls "$DST/ROI" | sort -n | tr '\n' ' ')"

# every ROI case must have BOTH masks, and a matching image
bad=0
for c in $(ls "$DST/ROI"); do
  [ -f "$DST/ROI/$c/GM_ROI.nii.gz" ] || { echo "  [BAD] case $c missing GM_ROI"; bad=1; }
  [ -f "$DST/ROI/$c/WM_ROI.nii.gz" ] || { echo "  [BAD] case $c missing WM_ROI"; bad=1; }
  [ -f "$DST/soft_thins_xy/$c/soft_thins_0_noblank_sliced.nii.gz" ] || { echo "  [BAD] case $c has ROI but no image"; bad=1; }
done

echo
if [ "$IMG" -eq 33 ] && [ "$GM" -eq 8 ] && [ "$WM" -eq 8 ] && [ "$bad" -eq 0 ]; then
  echo "OK — layout verified."
  echo "next (fix_pcct_xlsx.py needs pandas+nibabel, which live in the n2ndm env, NOT in base):"
  echo "  conda activate n2ndm"
  echo "  python PCCT_experiments/fix_pcct_xlsx.py            # preview the rewritten patient list"
  echo "  python PCCT_experiments/fix_pcct_xlsx.py --write    # write it"
else
  echo "*** VERIFICATION FAILED — do not train until this is resolved ***"
  exit 1
fi
du -sh "$DST"
df_quota
