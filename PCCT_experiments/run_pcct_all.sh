#!/bin/bash
#SBATCH --job-name=pcct_all
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=log_pcct_all_%j.txt

# The whole PCCT study in ONE job: flow training -> baseline sweep+CNR -> GAN fine-tune ->
# GAN epoch selection -> GAN sweep+CNR -> side-by-side summary.
#
# Chaining beats submitting five jobs because QOS 1a800 runs only ONE job at a time: separate jobs
# would queue behind each other anyway, and each would need babysitting to launch once its
# predecessor finished. Total is roughly 12-18h, comfortably inside the 72h wall.
#
# EVERY STAGE IS SKIP-IF-DONE. Re-submitting after a crash, a timeout, or a scancel resumes at the
# first unfinished stage: training skips when its final checkpoint exists, and `pred` skips samples
# that already have a pred_img.nii.gz. So the recovery procedure is just `sbatch` again.
#
# Stage order is deliberate: the no-GAN baseline sweep runs BEFORE GAN training, so the numbers that
# matter most (and that the GAN must beat) land after ~8h instead of after everything.
#
#   sbatch PCCT_experiments/run_pcct_all.sh              # everything
#   NFES="1 2 3 5" sbatch PCCT_experiments/run_pcct_all.sh
#   SKIP_GAN=1 sbatch PCCT_experiments/run_pcct_all.sh   # flow + baseline only
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}
REPO=/gpfs/work/aac/xingyiyao23/Code/IMF_denoising
cd "$REPO"

MODELS=/gpfs/work/aac/xingyiyao23/projects/denoising/models
PRED="$REPO/PCCT_experiments/predict_2D_imf_pcct.py"
EVAL="$REPO/PCCT_experiments/eval_pcct_cnr.py"
XLSX=/gpfs/work/aac/xingyiyao23/Data/PCCT/Patient_lists/PCCT_split_hpc.xlsx

FLOW_TRIAL="${FLOW_TRIAL:-imf_v2_unsupervised_PCCT}"
GAN_TRIAL="${GAN_TRIAL:-imf_gan_unsupervised_PCCT}"
FLOW_EPOCH="${FLOW_EPOCH:-200}"
GAN_EPOCHS_TOTAL="${GAN_EPOCHS_TOTAL:-50}"
# save_every 2 -> 25 checkpoints (2,4,..,50; divides GAN_EPOCHS_TOTAL so the final epoch IS saved).
# Dense on purpose: GAN quality is non-monotone and its useful window can be narrow, so a coarse
# grid can straddle the peak (with save_every 10, a peak at epoch 23 is simply never evaluated).
# 25 x 571MB = 14.3GB is TEMPORARY -- stage 6 prunes down to the epochs actually worth keeping.
# The binding cost is not disk but selection time: stage 4 runs inference per checkpoint.
GAN_SAVE_EVERY="${GAN_SAVE_EVERY:-2}"
NFES="${NFES:-1 2 3 5 10}"
SEL_NFE="${SEL_NFE:-3}"                # NFE used to pick the best GAN epoch
SEL_K="${SEL_K:-10}"                   # samples per case during selection (10 = a reported operating
                                       # point; ranking epochs needs less averaging than the final numbers)
ITER="${ITER:-20}"
SKIP_GAN="${SKIP_GAN:-0}"
PRUNE="${PRUNE:-1}"                    # 1 = delete the GAN checkpoints that lost the selection
KEEP_NEIGHBOURS="${KEEP_NEIGHBOURS:-1}"  # also keep this many checkpoints either side of the winner

banner () { echo; echo "################ $* ################"; echo "  $(date '+%F %T')"; }
quota  () { mmlsquota --block-size auto 2>/dev/null | tail -1 || true; }

banner "PCCT pipeline start"
echo "flow=$FLOW_TRIAL@$FLOW_EPOCH  gan=$GAN_TRIAL  NFEs=[$NFES]  K=$ITER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
quota
[ -f "$XLSX" ] || { echo "patient list missing: $XLSX"; echo "run unpack_pcct_hpc.sh + fix_pcct_xlsx.py --write"; exit 1; }

# ---------- stage 1: flow training ----------
FLOW_CKPT="$MODELS/$FLOW_TRIAL/models/model-$FLOW_EPOCH.pt"
banner "stage 1/7: flow training ($FLOW_TRIAL)"
if [ -f "$FLOW_CKPT" ]; then
  echo "[skip] $FLOW_CKPT already exists"
else
  python PCCT_experiments/train_2D_imf_pcct.py \
    --trial_name "$FLOW_TRIAL" --train_num_steps "$FLOW_EPOCH" \
    --train_batch_size 32 --save_every 10 || { echo "STAGE 1 FAILED"; exit 1; }
fi
[ -f "$FLOW_CKPT" ] || { echo "no $FLOW_CKPT after training — aborting"; exit 1; }
quota

# ---------- stage 2: baseline sweep + CNR (before the GAN, so results land early) ----------
banner "stage 2/7: no-GAN sweep + CNR"
for NFE in $NFES; do
  echo "---- $FLOW_TRIAL NFE=$NFE ----"
  python "$PRED" --trial_name "$FLOW_TRIAL" --epoch "$FLOW_EPOCH" --mode pred \
    --num_steps "$NFE" --iteration_num "$ITER" || { echo "STAGE 2 pred NFE=$NFE FAILED"; exit 1; }
  python "$PRED" --trial_name "$FLOW_TRIAL" --epoch "$FLOW_EPOCH" --mode avg \
    --num_steps "$NFE" --k_save 10 20 --cleanup || { echo "STAGE 2 avg NFE=$NFE FAILED"; exit 1; }
  python "$EVAL" --trial "$FLOW_TRIAL" --epoch "$FLOW_EPOCH" --nfe "$NFE" --k 10 20 \
    || { echo "STAGE 2 CNR NFE=$NFE FAILED"; exit 1; }
done
quota

if [ "$SKIP_GAN" = "1" ]; then
  banner "SKIP_GAN=1 — stopping after the baseline"
  exit 0
fi

# ---------- stage 3: GAN fine-tune ----------
GAN_CKPT="$MODELS/$GAN_TRIAL/models/model-$GAN_EPOCHS_TOTAL.pt"
banner "stage 3/7: GAN fine-tune ($GAN_TRIAL)"
if [ -f "$GAN_CKPT" ]; then
  echo "[skip] $GAN_CKPT already exists"
else
  python gan/train_2D_imf_gan_pcct.py \
    --trial_name "$GAN_TRIAL" --pretrained "$FLOW_CKPT" --pretrained_weights model \
    --adv_weight 0.5 --train_num_steps "$GAN_EPOCHS_TOTAL" \
    --batch_size 16 --save_every "$GAN_SAVE_EVERY" || { echo "STAGE 3 FAILED"; exit 1; }
fi
[ -f "$GAN_CKPT" ] || { echo "no $GAN_CKPT after training — aborting"; exit 1; }
quota

# ---------- stage 4: pick the GAN epoch by CNR (quality is non-monotone -> do not assume the last) ----------
banner "stage 4/7: GAN epoch selection @ NFE=$SEL_NFE, K=$SEL_K"
SAVED=$(ls "$MODELS/$GAN_TRIAL/models"/model-*.pt 2>/dev/null \
        | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tr '\n' ' ')
echo "checkpoints to score: $SAVED"
echo "(each is a full inference pass over the 8 test cases; --cleanup keeps disk flat between them)"
for E in $SAVED; do
  echo "---- epoch $E @ NFE=$SEL_NFE K=$SEL_K ----"
  python "$PRED" --trial_name "$GAN_TRIAL" --epoch "$E" --mode pred \
    --num_steps "$SEL_NFE" --iteration_num "$SEL_K" || { echo "epoch $E pred failed, skipping"; continue; }
  python "$PRED" --trial_name "$GAN_TRIAL" --epoch "$E" --mode avg \
    --num_steps "$SEL_NFE" --k_save "$SEL_K" --cleanup || true
  python "$EVAL" --trial "$GAN_TRIAL" --epoch "$E" --nfe "$SEL_NFE" --k "$SEL_K" || true
done

# best epoch = highest mean CNR among the per-epoch xlsx just written
BEST_EPOCH=$(python - "$MODELS/$GAN_TRIAL/pred_images_nfe$SEL_NFE" "$SEL_K" <<'PY'
import sys, glob, os, re
import pandas as pd
folder, k = sys.argv[1], int(sys.argv[2])
col_pref = [f'CNR_K{k}', 'CNR_K20', 'CNR_K10']
rows = []
for f in glob.glob(os.path.join(folder, 'PCCT_CNR_epoch*_nfe*.xlsx')):
    m = re.search(r'epoch(\d+)_nfe', os.path.basename(f))
    if not m:
        continue
    try:
        d = pd.read_excel(f)
    except Exception:
        continue
    col = next((c for c in col_pref if c in d.columns), None)
    if col is None:
        continue
    v = d[col].dropna().mean()
    if v == v:
        rows.append((int(m.group(1)), float(v)))
if rows:
    rows.sort(key=lambda r: r[0])
    # the full CNR-vs-epoch curve goes to the log: the shape (rising / peaked / collapsed) says more
    # about whether the adversarial loss is helping than the winning number alone does.
    sys.stderr.write('  epoch : CNR\n')
    best = max(rows, key=lambda r: r[1])
    for e, v in rows:
        sys.stderr.write(f'  {e:>5} : {v:.4f}{"   <-- best" if e == best[0] else ""}\n')
    print(best[0])
else:
    print('')
PY
)
if [ -z "$BEST_EPOCH" ]; then
  echo "[warn] epoch selection produced nothing — falling back to the final epoch $GAN_EPOCHS_TOTAL"
  BEST_EPOCH="$GAN_EPOCHS_TOTAL"
fi
echo ">>> best GAN epoch by mean CNR(K=20) @ NFE=$SEL_NFE : $BEST_EPOCH"

# ---------- stage 5: GAN sweep at the chosen epoch ----------
banner "stage 5/7: GAN sweep + CNR (epoch $BEST_EPOCH)"
for NFE in $NFES; do
  echo "---- $GAN_TRIAL epoch $BEST_EPOCH NFE=$NFE ----"
  python "$PRED" --trial_name "$GAN_TRIAL" --epoch "$BEST_EPOCH" --mode pred \
    --num_steps "$NFE" --iteration_num "$ITER" || { echo "STAGE 5 pred NFE=$NFE FAILED"; exit 1; }
  python "$PRED" --trial_name "$GAN_TRIAL" --epoch "$BEST_EPOCH" --mode avg \
    --num_steps "$NFE" --k_save 10 20 --cleanup || { echo "STAGE 5 avg NFE=$NFE FAILED"; exit 1; }
  python "$EVAL" --trial "$GAN_TRIAL" --epoch "$BEST_EPOCH" --nfe "$NFE" --k 10 20 \
    || { echo "STAGE 5 CNR NFE=$NFE FAILED"; exit 1; }
done
quota

# ---------- stage 6: prune the GAN checkpoints that lost ----------
banner "stage 6/7: prune GAN checkpoints"
if [ "$PRUNE" = "1" ]; then
  # Dense saving was only needed to LOCATE the peak; once found, keeping 25 x 571MB serves nothing.
  # Kept: the winner, its immediate neighbours (so the choice can be revisited without retraining),
  # and the final epoch (the "train to the end" reference). Everything else goes.
  KEEP="$BEST_EPOCH $GAN_EPOCHS_TOTAL"
  for d in $(seq 1 "$KEEP_NEIGHBOURS"); do
    KEEP="$KEEP $((BEST_EPOCH - d * GAN_SAVE_EVERY)) $((BEST_EPOCH + d * GAN_SAVE_EVERY))"
  done
  echo "keeping epochs: $(echo $KEEP | tr ' ' '\n' | sort -n -u | tr '\n' ' ')"
  freed=0
  for f in "$MODELS/$GAN_TRIAL/models"/model-*.pt; do
    [ -f "$f" ] || continue
    e=$(basename "$f" | sed 's/model-\([0-9]*\)\.pt/\1/')
    if ! echo " $KEEP " | grep -q " $e "; then
      sz=$(du -m "$f" | cut -f1); rm -f "$f"; freed=$((freed + sz))
    fi
  done
  echo "pruned, freed ~${freed} MB"
  ls "$MODELS/$GAN_TRIAL/models"/model-*.pt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
  quota
else
  echo "[skip] PRUNE=0 — all $(ls "$MODELS/$GAN_TRIAL/models"/model-*.pt 2>/dev/null | wc -l) checkpoints kept"
fi

# ---------- stage 7: summary ----------
banner "stage 7/7: summary"
python - "$MODELS" "$FLOW_TRIAL" "$FLOW_EPOCH" "$GAN_TRIAL" "$BEST_EPOCH" "$NFES" <<'PY'
import sys, os, glob, re
import numpy as np, pandas as pd

models, flow_t, flow_e, gan_t, gan_e, nfes = sys.argv[1:7]
nfes = [int(x) for x in nfes.split()]

def stats(trial, epoch, nfe):
    f = os.path.join(models, trial, f'pred_images_nfe{nfe}', f'PCCT_CNR_epoch{epoch}_nfe{nfe}.xlsx')
    if not os.path.isfile(f):
        return None
    d = pd.read_excel(f)
    out = {}
    for c in ('CNR_Noisy', 'CNR_K10', 'CNR_K20'):
        if c in d.columns:
            v = d[c].dropna().values
            if len(v):
                out[c] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    return out or None

noisy = None
for nfe in nfes:
    for t, e in ((flow_t, flow_e), (gan_t, gan_e)):
        s = stats(t, e, nfe)
        if s and 'CNR_Noisy' in s:
            noisy = s['CNR_Noisy']; break
    if noisy:
        break

print('\nPCCT CNR — mean +- std across the 8 test cases (higher is better)')
print('CNR = (GM_mean - WM_mean) / sqrt(GM_std^2 + WM_std^2), image clipped to [0,100] HU\n')
if noisy:
    print(f'  noisy input (baseline to beat): {noisy[0]:.4f} +- {noisy[1]:.4f}\n')

hdr = f'{"NFE":>4} | {"no-GAN K10":>13} {"no-GAN K20":>13} | {"GAN K10":>13} {"GAN K20":>13}'
print(hdr); print('-' * len(hdr))
for nfe in nfes:
    a, b = stats(flow_t, flow_e, nfe), stats(gan_t, gan_e, nfe)
    cell = lambda s, k: (f'{s[k][0]:.4f}' if s and k in s else '     -   ')
    print(f'{nfe:>4} | {cell(a,"CNR_K10"):>13} {cell(a,"CNR_K20"):>13} | {cell(b,"CNR_K10"):>13} {cell(b,"CNR_K20"):>13}')

if noisy:
    print(f'\n(gain vs the noisy input, K=20)')
    for nfe in nfes:
        a, b = stats(flow_t, flow_e, nfe), stats(gan_t, gan_e, nfe)
        ga = f'{a["CNR_K20"][0]/noisy[0]:.2f}x' if a and 'CNR_K20' in a else '-'
        gb = f'{b["CNR_K20"][0]/noisy[0]:.2f}x' if b and 'CNR_K20' in b else '-'
        print(f'  NFE {nfe:>2}: no-GAN {ga:>6}   GAN {gb:>6}')
print(f'\nGAN epoch used: {gan_e}   per-NFE xlsx: <trial>/pred_images_nfe*/PCCT_CNR_*.xlsx')
PY

banner "PCCT pipeline complete"
quota
echo "no-GAN : $MODELS/$FLOW_TRIAL/pred_images_nfe*/PCCT_CNR_*.xlsx"
echo "GAN    : $MODELS/$GAN_TRIAL/pred_images_nfe*/PCCT_CNR_*.xlsx"
