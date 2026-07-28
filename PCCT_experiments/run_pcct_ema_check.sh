#!/bin/bash
#SBATCH --job-name=pcct_ema_chk
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --output=log_pcct_ema_check_%j.txt

# Does the flow trainer's stale EMA explain the PCCT "GAN" gains? ~1.5h, and it decides whether
# any of the existing numbers need redoing -- so run it before any further training.
#
# THE PROBLEM. improved_mean_flow.py calls ema.update() once per EPOCH (the loop was converted from
# lucidrains' step-based trainer without moving the call inside the batch loop). With
# update_every=10 and ema_pytorch's update_after_step=100, a 200-epoch run does 9 real averaging
# steps at beta=0.995 -> its "EMA" is ~96% the epoch-100 weights. gan/imf_gan.py updates per
# optimizer step, so a GAN trial's EMA is a genuine average. Every no-GAN-vs-GAN comparison so far
# has therefore been roughly epoch 100 versus epoch 200-250 -- a 150-epoch gap, not the 50 epochs of
# fine-tuning, and worst exactly at NFE=1 where the solution is least converged (no-GAN 0.28 vs GAN
# 0.99, the most extreme number in the table).
#
# THE TEST. Re-run the no-GAN model with --weights raw (the epoch-200 online weights) and compare:
#     no-GAN(ema, ~epoch100)   vs   no-GAN(raw, epoch200)   vs   GAN(ema)
#   * raw jumps most of the way to GAN  -> the gain was largely this artefact; baselines need
#                                          re-scoring on raw before any GAN claim is made.
#   * raw barely moves                  -> the confound is empirically negligible; ignore it and
#                                          move on to the adv_weight=0 control.
#
# K=10 only and a subset of NFEs: this is an A/B on the weights, not a results sweep, so it does not
# need the full K=20 averaging. Raw runs write to pred_images_nfe<N>_raw/ and cannot overwrite
# anything existing.
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}
REPO=/gpfs/work/aac/xingyiyao23/Code/IMF_denoising
cd "$REPO"

MODELS=/gpfs/work/aac/xingyiyao23/projects/denoising/models
PRED="$REPO/PCCT_experiments/predict_2D_imf_pcct.py"
EVAL="$REPO/PCCT_experiments/eval_pcct_cnr.py"
FLOW_TRIAL=imf_v2_unsupervised_PCCT
FLOW_EPOCH=200
GAN_TRIAL=imf_gan_unsupervised_PCCT
GAN_EPOCH=48
NFES="${NFES:-1 3 10}"
K="${K:-10}"

echo "=== EMA staleness check | no-GAN raw vs ema, NFEs=[$NFES], K=$K ==="
date '+%F %T'
mmlsquota --block-size auto 2>/dev/null | tail -1

for NFE in $NFES; do
  RAWDIR="$MODELS/$FLOW_TRIAL/pred_images_nfe${NFE}_raw"
  echo; echo "---- no-GAN NFE=$NFE with ONLINE weights ----"
  python "$PRED" --trial_name "$FLOW_TRIAL" --epoch "$FLOW_EPOCH" --mode pred \
    --num_steps "$NFE" --iteration_num "$K" --weights raw || { echo "pred NFE=$NFE FAILED"; exit 1; }
  python "$PRED" --trial_name "$FLOW_TRIAL" --epoch "$FLOW_EPOCH" --mode avg \
    --num_steps "$NFE" --k_save "$K" --cleanup --weights raw || { echo "avg NFE=$NFE FAILED"; exit 1; }
  python "$EVAL" --trial "$FLOW_TRIAL" --epoch "$FLOW_EPOCH" --nfe "$NFE" --k "$K" \
    --folder "$RAWDIR" || { echo "CNR NFE=$NFE FAILED"; exit 1; }
done

echo; echo "################ verdict ################"
python - "$MODELS" "$FLOW_TRIAL" "$FLOW_EPOCH" "$GAN_TRIAL" "$GAN_EPOCH" "$NFES" "$K" <<'PY'
import sys, os
import numpy as np, pandas as pd

models, ft, fe, gt, ge, nfes, k = sys.argv[1:8]
nfes = [int(x) for x in nfes.split()]; k = int(k)

def mean_cnr(folder, epoch, nfe, col):
    f = os.path.join(folder, f'PCCT_CNR_epoch{epoch}_nfe{nfe}.xlsx')
    if not os.path.isfile(f):
        return None
    d = pd.read_excel(f)
    if col not in d.columns:
        return None
    v = d[col].dropna().values
    return float(np.mean(v)) if len(v) else None

print(f'\nPCCT CNR, K={k} (higher is better). noisy input = 0.385\n')
hdr = f'{"NFE":>4} | {"no-GAN (ema~ep100)":>19} {"no-GAN (raw ep200)":>19} | {"GAN (ema)":>10} | {"raw closes":>11}'
print(hdr); print('-' * len(hdr))
closed = []
for nfe in nfes:
    e = mean_cnr(os.path.join(models, ft, f'pred_images_nfe{nfe}'), fe, nfe, f'CNR_K{k}')
    r = mean_cnr(os.path.join(models, ft, f'pred_images_nfe{nfe}_raw'), fe, nfe, f'CNR_K{k}')
    g = mean_cnr(os.path.join(models, gt, f'pred_images_nfe{nfe}'), ge, nfe, f'CNR_K{k}')
    if g is None:
        g = mean_cnr(os.path.join(models, gt, f'pred_images_nfe{nfe}'), ge, nfe, f'CNR_K{k}')
    pct = ''
    if None not in (e, r, g) and abs(g - e) > 1e-9:
        frac = (r - e) / (g - e) * 100
        closed.append(frac); pct = f'{frac:6.1f}%'
    f = lambda v: f'{v:.4f}' if v is not None else '   -  '
    print(f'{nfe:>4} | {f(e):>19} {f(r):>19} | {f(g):>10} | {pct:>11}')

print('\n"raw closes" = how much of the no-GAN -> GAN gap is recovered just by dropping the stale EMA.')
if closed:
    m = float(np.mean(closed))
    print(f'\nmean across NFEs: {m:.1f}%')
    if m > 60:
        print('>> Most of the "GAN gain" is the EMA artefact. Re-score the baselines on raw weights')
        print('   before making any GAN claim, and fix ema.update() placement for future training.')
    elif m > 25:
        print('>> A substantial part is the EMA artefact. Report baselines on raw weights; the')
        print('   remainder still needs the adv_weight=0 control to attribute.')
    else:
        print('>> The EMA confound is small in practice. Leave it, and go to the adv_weight=0')
        print('   control to separate the adversarial loss from the extra fine-tuning epochs.')
else:
    print('(not enough data to compare -- check that the GAN xlsx for these NFEs exist)')
PY

echo; echo "done"; date '+%F %T'
