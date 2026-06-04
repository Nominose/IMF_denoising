#!/bin/bash
# Free disk space by deleting NFE=3 *per-sample* prediction volumes and the redundant
# cumulative averages. The averages you actually use are already computed, so this is safe.
#   KEEPS : pred_img_scans10.nii.gz, pred_img_scans20.nii.gz, sample_std.npy, gt_img, condition_img
#   DELETES: every epoch200_*/pred_img.nii.gz  +  pred_img_scans{1-9,11-19}.nii.gz
# Pure deletion (no writes), so it works even at 0 bytes free.
D=/host/d/research/projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/pred_images_nfe3

echo "BEFORE:"; df -h /host/d | tail -1
find "$D" -name 'pred_img.nii.gz' -delete
find "$D" -name 'pred_img_scans*.nii.gz' ! -name 'pred_img_scans10.nii.gz' ! -name 'pred_img_scans20.nii.gz' -delete
echo "AFTER:";  df -h /host/d | tail -1
echo "done."
