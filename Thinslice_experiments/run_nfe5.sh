#!/bin/bash
# NFE=5 full pass: inference (K=20) -> average (K=10,20) -> quantification.
# Run inside the docker container (ideally under tmux: the pred step is the long part).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"                     # .../IMF_denoising
export PYTHONPATH="$(dirname "$REPO_ROOT"):$PYTHONPATH"
cd "$REPO_ROOT"

python -c "import torch,sys; ok=torch.cuda.is_available(); print('cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { echo '!! No GPU visible to torch in this container.'; exit 1; }

echo "===== NFE=5 : pred (K=20) ====="
python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode pred --iteration_num 20 --num_steps 5 --slice_batch 8
echo "===== NFE=5 : avg (K=10,20) ====="
python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode avg --num_steps 5
echo "===== NFE=5 : eval ====="
python Thinslice_experiments/eval_imf_v2.py --epoch 200 --num_steps 5 --k_list 10 20 \
    --out /host/d/research/projects/denoising/results/imf_v2_brainCT_eval_nfe5.xlsx
echo "DONE (NFE=5)."
