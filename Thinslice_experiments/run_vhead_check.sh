#!/bin/bash
#SBATCH --job-name=vhead_chk
#SBATCH --partition=aiaca800
#SBATCH --qos=1a800
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=log_vhead_check_%j.txt

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate n2ndm
export PYTHONPATH=/gpfs/work/aac/xingyiyao23/Code:$PYTHONPATH
cd /gpfs/work/aac/xingyiyao23/Code/IMF_denoising

echo "=== Hostname / Date ==="
hostname
date
echo ""
echo "=== Python / Torch / GPU ==="
which python
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

python Thinslice_experiments/vhead_check.py
RC=$?

echo ""
echo "=== End of sbatch (rc=$RC) ==="
date
exit $RC
