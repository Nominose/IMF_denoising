# N2NDM + iMF: Self-supervised CT Denoising with Improved Mean Flow

This repository implements **Noise2Noise Diffusion Model (N2NDM)** combined with **improved Mean Flow (iMF)** for self-supervised CT image denoising. The method eliminates the need for clean training data and achieves comparable denoising quality to the original N2NDM while being **333x faster** at inference.

## Method Overview

### N2NDM (Base Framework)
N2NDM trains a diffusion model to sample from `p(x2|x1)`, where `{x1, x2}` is a Noise2Noise pair sharing the same clean target but with independent noise. By sampling K times and averaging, the noise cancels out while structures are preserved.

### iMF (Our Acceleration)
We replace the cDDPM backbone in N2NDM with improved Mean Flow (iMF), a flow matching model that supports few-step sampling. Key advantages:

- **17x speedup**: K=20 denoising requires only 60 NFE (vs 1,000 for N2NDM with DDIM 50-step sampling)
- **No distillation needed**: K is a flexible inference-time hyperparameter
- **NFE analysis**: We prove NFE=1 collapses to posterior mean (equivalent to N2N regression). NFE=1 to NFE=2 is a **qualitative leap** (from zero to non-zero diversity), while NFE=2 to NFE=3 provides further quantitative improvement

## Results on Simulated Thin-slice Brain CT

| Method | MAE (↓) | SSIM (↑) | LPIPS (↓) | Total NFE (K=20) |
|--------|---------|----------|-----------|-------------------|
| FBP (noisy) | 6.28±0.61 | 0.412±0.055 | 0.154±0.025 | - |
| N2NDM (distilled cDDPM) | 2.98±0.32 | 0.763±0.028 | 0.047±0.009 | 1 (but K fixed) |
| Ours (iMF, K=1) | 4.08±0.43 | 0.582±0.027 | 0.085±0.016 | 3 |
| Ours (iMF, K=10) | 3.09±0.42 | 0.747±0.033 | 0.061±0.010 | 30 |
| **Ours (iMF, K=20)** | **3.02±0.43** | **0.761±0.034** | **0.064±0.011** | **60** |

Metrics computed on brain tissue window [0, 100] HU, 16 test cases.

## Results on AAPM Low-dose Abdominal CT

| Method | MAE | SSIM | LPIPS | Total NFE (K=20) |
|--------|------|------|-------|-------------------|
| N2NDM (distilled cDDPM) | 11.4±1.1 | 0.765±0.014 | 0.045±0.009 | 1 (but K fixed) |
| **Ours (iMF, K=20)** | **12.22±0.98** | **0.747±0.017** | **0.053±0.011** | **60** |

Metrics computed on abdominal window [-160, 240] HU.

## Repository Structure

```
IMF_denoising/
|
|-- improved_mean_flow.py          # iMF model, trainer, and sampler
|-- conditional_flow_matching.py   # Standard flow matching baseline
|
|-- Generator.py                   # Dataset class for Mayo low-dose CT
|-- Generator_thinslice.py         # Dataset class for brain CT (adjacent slice N2N)
|-- Generator_EM.py                # Dataset class for electron microscopy
|-- Generator_MR.py                # Dataset class for MR data
|-- Data_processing.py             # Normalization, histogram equalization, augmentation
|
|-- Build_lists/
|   |-- Build_list.py              # Patient list builders for all datasets
|   |-- Build_train_test_file_spreadsheet_brainCT.ipynb
|
|-- denoising_diffusion_pytorch/   # U-Net architecture and cDDPM implementation
|-- functions_collection/          # Utility functions
|-- help_data/                     # Histogram equalization bins
|
|-- CT_experiments/                # Mayo low-dose CT experiments
|   |-- train_2D.py               # cDDPM training
|   |-- train_2D_imf.py           # iMF training
|   |-- predict_2D_imf.py         # iMF inference
|
|-- Thinslice_experiments/         # Brain CT thin-slice experiments
|   |-- train_2D.py               # cDDPM training
|   |-- train_2D_imf.py           # iMF training
|   |-- predict_2D_imf.py         # iMF inference (pred + avg modes)
|   |-- main_quantitative_imf.ipynb  # Evaluation metrics
|   |-- main_quantitative_new.ipynb  # Evaluation (all methods comparison)
|
|-- EM_experiments/                # Electron microscopy experiments
|-- MR_experiments/                # MR denoising experiments
|-- PCCT_experiments/              # Photon-counting CT experiments
|-- noise2noise/                   # Deterministic N2N baselines
|-- Manuscript/                    # Paper figures and analysis
|-- simulation/                    # CT noise simulation code
```

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 1.13+
- CUDA-compatible GPU (12GB+ VRAM recommended)
- nibabel, numpy, pandas, openpyxl, scikit-image, lpips, ema-pytorch, accelerate

### Data Preparation

1. Prepare NIfTI (.nii.gz) files organized as:
```
Data/
  fixedCT/{Patient_ID}/{Sub_ID}/img_thinslice_partial.nii.gz   # Ground truth
  simulation/{Patient_ID}/{Sub_ID}/gaussian_random_0/recon.nii.gz  # Noisy simulation
```

2. Create a patient list Excel file with columns: `batch, Patient_ID, Patient_subID, random_num, noise_file, ground_truth_file`

3. Pre-compute histogram equalization bins (for brain CT):
```
help_data/histogram_equalization/bins.npy
help_data/histogram_equalization/bins_mapped.npy
```

### Training (Brain CT)

```bash
python Thinslice_experiments/train_2D_imf.py
```

Key parameters in the script:
- `supervision = 'unsupervised'` — N2N mode with adjacent slices
- `condition_channel = 2` — adjacent slices s-1, s+1 as 2-channel condition
- `ratio_r_neq_t = 0.50` — 50% MeanFlow + 50% flow matching training
- `train_batch_size = 4` — adjust based on GPU memory
- `patch_size = [128, 128]` — training patch size

### Inference

**Step 1: Generate K predictions**
```bash
python Thinslice_experiments/predict_2D_imf.py \
  --epoch 30 \
  --mode pred \
  --iteration_num 20 \
  --num_steps 3
```

**Step 2: Average predictions**
```bash
python Thinslice_experiments/predict_2D_imf.py \
  --epoch 30 \
  --mode avg
```

### Evaluation

Run `Thinslice_experiments/main_quantitative_imf.ipynb` to compute MAE, SSIM, and LPIPS against ground truth.

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--epoch` | Model checkpoint epoch | required |
| `--mode` | `pred` (generate samples) or `avg` (average samples) | required |
| `--iteration_num` | Number of K samples to generate | 20 |
| `--num_steps` | NFE per sample (1=one-step, 3=recommended) | 3 |
| `--slice_range` | Slice range, e.g. `30-80` or `all` | `30-80` |

## Datasets

| Dataset | Modality | N2N Pair Construction | Condition Channel |
|---------|----------|----------------------|-------------------|
| AAPM Mayo | Low-dose abdominal CT | Sinogram odd/even split | 1 |
| Brain CT | Thin-slice brain CT | Adjacent slices (s-1, s+1) | 2 |
| PCCT | Photon-counting brain CT | Adjacent slices | 2 |
| EM | Electron microscopy | Independent simulations | 1 |

## Citation

If you use this code, please cite:

```
@article{n2ndm2025,
  title={Noise2Noise Diffusion Model for CT Denoising without Clean Training Data},
  author={Anonymized},
  year={2025}
}
```

## License

This project is for research purposes only.