# N2NDM + iMF: Self-supervised CT Denoising with Improved Mean Flow

This repository implements **Noise2Noise Diffusion Model (N2NDM)** accelerated with an
**improved Mean Flow (iMF)** backbone for **self-supervised** CT image denoising — no clean
training data is ever used. iMF turns the many-step conditional-diffusion sampler of N2NDM into a
**few-step (NFE 1–5)** flow-matching sampler, and an optional **self-supervised adversarial (GAN)
fine-tuning** stage sharpens the one-step reconstruction. Experiments cover thin-slice **brain CT**,
**Mayo low-dose abdominal CT**, and **photon-counting CT (PCCT)**.

## Method Overview

**N2NDM (base).** Trains a conditional generative model to sample from `p(x2|x1)`, where `{x1, x2}`
is a Noise2Noise pair — two independent noisy realizations of the *same* clean slice. Drawing `K`
samples and averaging cancels the noise while preserving structure.

**iMF (acceleration).** The cDDPM backbone is replaced by improved Mean Flow (`ImprovedMeanFlow`),
a flow-matching model supporting few-step Euler/midpoint/Heun sampling:
- `K` (number of N2N samples averaged) is a **flexible inference-time** knob — no distillation, no
  fixed-`K` second stage.
- Mixed training objective: `ratio_r_neq_t=0.50` blends MeanFlow (r≠t) with plain flow matching,
  plus an auxiliary velocity head (`auxiliary_v_head=True`, the "v2" model).
- NFE analysis: NFE=1 collapses to the posterior mean (≈ N2N regression); NFE=1→2 is the
  qualitative jump to non-zero sample diversity; NFE=2→3 refines quantitatively.

**Self-supervised GAN fine-tuning (`gan/`, experimental).** Loads a flow-pretrained checkpoint and
adds a small adversarial term, `L_total = L_flow + beta * L_adv`, pushing the model's one-step
output `x0_gen = z - u(z, r=0, t=1, c)` toward the **noisy `x2`** distribution (the discriminator's
"real" is the noisy N2N target — never clean data). A high-pass PatchGAN discriminator (`img -
blur(img)`, hinge loss + lazy R1) targets the high-frequency noise texture. Inference is unchanged:
the discriminator is discarded and the same few-step, K-flexible sampler is used.

## Repository Structure

```
IMF_denoising/
|-- improved_mean_flow.py            # iMF model, Trainer, and Sampler (few-step ODE solvers)
|-- conditional_flow_matching.py     # standard flow-matching baseline
|-- Generator.py                     # Mayo low-dose CT dataset (odd/even recon N2N pairs)
|-- Generator_thinslice.py           # brain CT dataset (adjacent-slice N2N pairs)
|-- Generator_EM.py / Generator_MR.py# EM / MR dataset variants
|-- Data_processing.py               # normalization, histogram equalization, augmentation
|-- optimal_schedule.py              # non-uniform time-step schedule for sampling
|
|-- Build_lists/Build_list.py        # patient-list builders (Build, Build_thinsliceCT, ...)
|-- denoising_diffusion_pytorch/     # conditional U-Net backbone (+ cDDPM)
|-- functions_collection/            # utilities (make_folder, preload_data, metrics helpers)
|-- help_data/                       # histogram-equalization bins
|
|-- Thinslice_experiments/           # brain CT thin-slice experiments
|   |-- train_2D_imf.py              # iMF training (v2, aux v-head, hist-eq)
|   |-- predict_2D_imf.py            # iMF inference (pred / avg; euler|midpoint|heun)
|   |-- predict_2D_imf_v2.py         # inference for the v2 checkpoint; per-NFE output folders
|   |-- train_2D.py / predict_2D*.py # cDDPM / distill / flow-matching / EDM baselines
|   |-- eval_imf_v2.py               # MAE/SSIM/LPIPS on brain window
|   |-- eval_x2ref.py                # GT-free (N,K) selection via the x2-reference identity
|   |-- compare_nfe.py               # compare metrics across NFEs
|   |-- fuse_imf_v2.py               # fuse per-K averages
|   |-- verify_ambient_oracle.py     # ambient/oracle sanity check
|   |-- run_imf_v2_docker.sh / run_eval_nfe3.sh / run_nfe5.sh / free_space_nfe3.sh
|   |-- main_quantitative_imf.ipynb / main_quantitative_new.ipynb  # eval notebooks
|
|-- CT_experiments/                  # Mayo low-dose abdominal CT experiments
|   |-- train_2D_imf_mayo.py         # Mayo iMF training (no GAN), odd/even N2N, argparse paths
|   |-- train_2D_imf.py              # older (stale-path) Mayo iMF training
|   |-- predict_2D_imf.py            # Mayo iMF inference
|   |-- train_2D*.py / predict_2D*.py# cDDPM / distill / flow-matching baselines
|   |-- pred_2D_copy.sh              # inference shell wrapper
|
|-- gan/                             # self-supervised adversarial fine-tuning (training-only)
|   |-- imf_gan.py                   # high-pass PatchGAN discriminator + GANTrainer
|   |-- train_2D_imf_gan.py          # GAN fine-tuning (one-step adversarial, adv_nfe=1)
|   |-- train_2D_imf_gan_nfe3.py     # GAN fine-tuning (NFE=3-direct adversarial)
|   |-- gen_baseline_fv.py           # one-off baseline (no-GAN) F(v) dump -> epoch 0
|   |-- run_gan_nfe_sweep.sh         # sweep inference over NFEs (5 10 20 30 50) via predict_2D_imf_v2
|   |-- eval_gan_nfe.py              # full-set eval at one NFE (K=10/20; brain window [0,100] HU)
|   |-- view_fv_evolution.py / view_probe.py / view_nfe1_fullslice.py  # visual diagnostics
|
|-- PCCT_experiments/                # photon-counting CT experiments
|-- EM_experiments/ / MR_experiments/# electron-microscopy / MR denoising experiments
|-- natural_image_experiments/       # natural-image variant (train/eval/generator)
|-- noise2noise/                     # deterministic N2N baselines
|-- simulation/                      # CT noise simulation code
|-- Manuscript/                      # paper figures and analysis
```

## Requirements

- Python 3.10+, PyTorch 1.13+, CUDA GPU (12 GB+ recommended)
- `nibabel`, `numpy`, `pandas`, `openpyxl`, `scikit-image`, `lpips`, `ema-pytorch`, `accelerate`

Scripts are written for a Docker environment that mounts `D:\research` at `/host/d` and the repo
under `/host/c/Users/ROG/Documents/GitHub`. Newer scripts auto-detect the data root (`/host/d/research`
then `/host/d`) and remap stale paths, so absolute paths degrade gracefully.

## How to Run

### Brain CT (thin-slice, adjacent-slice N2N)

Train (edit params at the top of the script — `supervision='unsupervised'`, `condition_channel=2`,
`ratio_r_neq_t=0.50`, `patch_size=[128,128]`):
```bash
python Thinslice_experiments/train_2D_imf.py
```

Inference — run twice per NFE (generate K samples, then average). Use `predict_2D_imf_v2.py` for the
aux-v-head "v2" checkpoint (writes to a per-NFE folder so NFE runs don't collide):
```bash
python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode pred --iteration_num 20 --num_steps 3
python Thinslice_experiments/predict_2D_imf_v2.py --epoch 200 --mode avg  --num_steps 3
```
`predict_2D_imf.py` also supports `--solver {euler,midpoint,heun}` and `--schedule {uniform,optimal}`.

Evaluate:
```bash
python Thinslice_experiments/eval_imf_v2.py   --epoch 200 --num_steps 3      # MAE/SSIM/LPIPS
python Thinslice_experiments/eval_x2ref.py    --epoch 200 --num_steps 3 5 --k_list 1 10 20  # GT-free (N,K)
python Thinslice_experiments/compare_nfe.py                                  # compare across NFEs
```

### Mayo low-dose abdominal CT (odd/even recon N2N, no GAN)

```bash
python CT_experiments/train_2D_imf_mayo.py
python CT_experiments/train_2D_imf_mayo.py --train_num_steps 200 --train_batch_size 1
python CT_experiments/train_2D_imf_mayo.py --patient_list_file <xlsx> --trial_name <name>
```
Abdomen window HU `[-200, 250]`, no histogram equalization, `condition_channel=1`, patch `256x256`;
evaluation region = slices `[150, 200]`. `switch_odd_and_even_frequency=0.5` randomizes which half
is condition vs target.

### GAN fine-tuning (self-supervised, brain CT)

```bash
python gan/train_2D_imf_gan.py \
  --pretrained /host/d/research/projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt
python gan/train_2D_imf_gan_nfe3.py          # NFE=3-direct adversarial variant

# inference sweep over NFEs + per-NFE metrics
bash gan/run_gan_nfe_sweep.sh                 # NFEs 5 10 20 30 50 (override: pass a list, or TRIAL/EPOCH/ITER env)
python gan/eval_gan_nfe.py --trial imf_gan_unsupervised_gaussian_brainCT --epoch 28 --nfe 5
```

## Data Layout & Key Specifics

- **Self-supervision:** Noise2Noise — brain/PCCT use adjacent slices (s−1, s+1) as the 2-channel
  condition; Mayo uses odd/even half-projection reconstructions (1-channel). No clean data in training.
- **HU windows / metrics:** brain tissue `[0, 100]` HU; abdomen `[-200, 250]` HU (eval) / `[-160, 240]`
  in the paper tables. Metrics: MAE, SSIM, LPIPS (AlexNet), reported mean ± std across test cases.
- **Brain CT:** histogram equalization on (bins under `help_data/histogram_equalization/`),
  `background_cutoff=-1000`, `maximum_cutoff=2000`; default inference slice range `30-80`
  (`--slice_range all` to cover the whole volume).
- **Checkpoints:** the v2 model requires `auxiliary_v_head=True`; load it with `predict_2D_imf_v2.py`
  (the old `predict_2D_imf.py` builds the U-Net without the v-head and will fail to load v2).
- **Patient lists:** Excel sheets with per-batch splits (e.g. brain batches 0–3 train / 4 val / 5 test;
  Mayo batches `train`/`val`/`test`), read by `Build_lists/Build_list.py`.

## Notes

- The validated contributions are the **NFE=1 posterior-mean collapse analysis** and the **GT-free
  (N,K) selection** via the x2-reference identity (`eval_x2ref.py`). The `gan/` adversarial path is
  **experimental and training-only**; whether it improves small-K LPIPS is still under investigation.

## License

Research use only.
