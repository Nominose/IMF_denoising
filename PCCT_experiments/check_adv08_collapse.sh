#!/bin/bash
#SBATCH --job-name=adv08_chk
#SBATCH --partition=gpua800,aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --output=log_adv08_collapse_%j.txt

# Did adv_weight=0.8 actually break the one-step model, or did the epoch picker just choose badly?
#
# adv0.8 scored 0.1520 at NFE=1 (K=10) against adv0.5's 0.9925 -- far below even the noisy input
# (0.385) -- while matching or beating adv0.5 at every NFE >= 2. Two explanations fit that:
#   (a) COLLAPSE. adv_nfe=1 means the adversarial loss trains the ONE-STEP generation, so an
#       over-weighted adversary damages exactly NFE=1 and leaves multi-step sampling alone.
#   (b) SELECTION ARTEFACT. The epoch was chosen by CNR at NFE=3; that epoch may simply be poor at
#       NFE=1, with other epochs fine.
#
# Two independent probes, because the checkpoints needed for the obvious test were pruned:
#
#   PROBE 1 (no GPU, all 50 epochs): fv_evolution/fv_epoch<N>.npy is the one-step output for a FIXED
#     slice under FIXED noise, dumped every epoch and untouched by checkpoint pruning. It is exactly
#     the quantity the adversarial loss optimises, so a genuine collapse must be visible here across
#     the whole run -- independent of which epoch was selected.
#
#   PROBE 2 (GPU): score whatever adv0.8 checkpoints survived at NFE=1. Only 45 and 50 remain, so
#     this alone is thin -- it corroborates probe 1 rather than standing on its own.
#
# Agreement between the two settles it.
set -u

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:${PYTHONPATH:-}
REPO=/gpfs/work/aac/xingyiyao23/Code/IMF_denoising
cd "$REPO"
M=/gpfs/work/aac/xingyiyao23/projects/denoising/models
T=imf_gan_adv08_PCCT

echo "############ PROBE 1: one-step output across every epoch (fv_evolution) ############"
python - "$M" <<'PY'
import sys, os, glob, re
import numpy as np
M = sys.argv[1]
TRIALS = [('adv0', 'imf_gan_adv0_PCCT'), ('adv0.2', 'imf_gan_adv02_PCCT'),
          ('adv0.5', 'imf_gan_unsupervised_PCCT'), ('adv0.8', 'imf_gan_adv08_PCCT')]
hf = lambda a: float((a[1:, :] - a[:-1, :]).std())     # high-freq proxy = what D keys on
summary = {}
for name, t in TRIALS:
    F = os.path.join(M, t, 'models', 'fv_evolution')
    tgt = os.path.join(F, 'real_x2.npy')
    if not os.path.isfile(tgt):
        print(f'\n=== {name} ===  (no fv_evolution)'); continue
    r = np.load(tgt)
    print(f'\n=== {name} ===  target real_x2: mean {r.mean():+.4f}  std {r.std():.4f}  hf {hf(r):.4f}')
    print(f'{"ep":>4}{"mean":>10}{"std":>9}{"hf":>9}{"MSE vs target":>15}')
    files = sorted(glob.glob(os.path.join(F, 'fv_epoch*.npy')),
                   key=lambda p: int(re.search(r'epoch(\d+)', p).group(1)))
    last = None
    for p in files:
        e = int(re.search(r'epoch(\d+)', p).group(1))
        a = np.load(p)
        last = (a.std(), hf(a), float(((a - r) ** 2).mean()))
        if e % 5 and e not in (1, 2, 3):
            continue
        print(f'{e:>4}{a.mean():+10.4f}{a.std():9.4f}{hf(a):9.4f}{((a - r) ** 2).mean():15.4f}')
    if last:
        summary[name] = last
if len(summary) >= 2:
    print('\n--- final-epoch one-step output, relative to the target ---')
    print(f'{"cfg":>8}{"std":>9}{"hf":>9}{"MSE":>10}')
    for k, v in summary.items():
        print(f'{k:>8}{v[0]:9.4f}{v[1]:9.4f}{v[2]:10.4f}')
    if 'adv0.8' in summary and 'adv0.5' in summary:
        a8, a5 = summary['adv0.8'], summary['adv0.5']
        ratio = a8[2] / a5[2] if a5[2] > 0 else float('inf')
        print(f'\nadv0.8 MSE / adv0.5 MSE = {ratio:.2f}x')
        if ratio > 3 or a8[0] > 3 * a5[0] or a8[0] < a5[0] / 3:
            print('>> the one-step output is DEGENERATE at adv0.8 -> genuine collapse (a)')
        else:
            print('>> the one-step output looks comparable to adv0.5 -> favours a selection artefact (b)')
PY

echo
echo "############ PROBE 2: surviving adv0.8 checkpoints at NFE=1 ############"
EPOCHS=$(ls "$M/$T/models"/model-*.pt 2>/dev/null | sed 's/.*model-\([0-9]*\)\.pt/\1/' | sort -n | tr '\n' ' ')
echo "checkpoints that survived pruning: ${EPOCHS:-none}"
for E in $EPOCHS; do
  echo "---- epoch $E @ NFE=1 ----"
  python PCCT_experiments/predict_2D_imf_pcct.py --trial_name "$T" --epoch "$E" \
    --mode pred --num_steps 1 --iteration_num 10 >/dev/null 2>&1
  python PCCT_experiments/predict_2D_imf_pcct.py --trial_name "$T" --epoch "$E" \
    --mode avg --num_steps 1 --k_save 10 --cleanup >/dev/null 2>&1
  python PCCT_experiments/eval_pcct_cnr.py --trial "$T" --epoch "$E" --nfe 1 --k 10 2>/dev/null \
    | grep -E "denoised K=10|noisy input"
done

echo
echo "############ how to read this ############"
cat <<'EOF'
reference at NFE=1, K=10:   noisy 0.3850 | adv0 0.3498 | adv0.2 0.6977 | adv0.5 0.9925 | adv0.8 0.1520

  every surviving epoch also ~0.15, AND probe 1 shows a degenerate one-step output
      -> COLLAPSE is real. adv0.8 over-drives the adversary and destroys exactly the
         operating point it trains (NFE=1), while NFE>=2 is untouched. adv_weight=0.5 is
         the optimum, now bracketed on both sides rather than asserted.

  some epoch scores ~0.8-1.0 at NFE=1, and probe 1 looks normal
      -> SELECTION ARTEFACT. The best epoch depends on the deployment NFE, so the epoch
         must be chosen per NFE (or at NFE=1) rather than once at NFE=3. Worth a sentence
         in the paper either way, since it also explains why adv0.8 looked fine at NFE>=2.
EOF
