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
TRIAL="unsupervised_gaussian_EM_range01"
MODE="avg"   # or: avg
RANGE="10-15"
FINALMAX_FINALMIN="1_0"  # final max and min values for clipping, such as 1_-1
# =======================================

# list of epochs you want to run
EPOCH_LIST=(125 105 75 180)

# loop through epochs
for EPOCH in "${EPOCH_LIST[@]}"; do
    echo "Running epoch $EPOCH ..."
    
    python3 predict_2D.py \
        --trial_name $TRIAL \
        --epoch $EPOCH \
        --mode $MODE \
        --slice_range $RANGE \
        --finalmax_finalmin $FINALMAX_FINALMIN
done

echo "Finished all jobs."
