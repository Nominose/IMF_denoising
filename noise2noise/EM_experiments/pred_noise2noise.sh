#!/bin/bash
# EPOCH=405
# MODE="avg"   # or: avg
# RANGE="100-200"
# echo "running input both ..."
# python3 predict_2D.py --trial_name unsupervised_gaussian --epoch $EPOCH --mode $MODE --input both --slice_range $RANGE
# epoch "running input all ..."
# python3 predict_2D.py --trial_name unsupervised_gaussian --epoch $EPOCH --mode $MODE --input all --slice_range $RANGE
# echo "Finished all jobs"

# # ============ USER SETTINGS ============
TRIAL="noise2noise_EM_range-1+1"
RANGE="10-15"
NOISETYPE="gaussian"  # or: gaussian
# =======================================

# list of epochs you want to run
EPOCH_LIST=(50)

# loop through epochs
for EPOCH in "${EPOCH_LIST[@]}"; do
    echo "Running epoch $EPOCH ..."
    
    python3 pred_noise2noise.py \
        --trial_name $TRIAL \
        --epoch $EPOCH \
        --slice_range $RANGE \
        --noise_type $NOISETYPE
done

echo "Finished all jobs."
