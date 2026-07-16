# iMF + self-supervised GAN — what to run, and what bites

Adversarial fine-tune of a flow-pretrained iMF generator. `L_G = L_flow + adv_weight · L_adv`;
the discriminator is an unconditional high-pass PatchGAN pushing the few-step generation toward
the **noisy** N2N target `x2`, so the whole thing stays self-supervised (no clean data) and the D
is discarded at inference.

## Which weights the fine-tune starts from — `--pretrained_weights`

A flow checkpoint carries **two** different models and they are not equally good.

The flow trainer's legacy EMA path (still the default, see `improved_mean_flow.py`) calls
`ema.update()` once per **epoch**. With `ema_pytorch`'s `update_after_step=100` / `update_every=10`
defaults that means a 200-epoch run gets ~11 pure copies (last at epoch ~101) and only ~9 real
averaging events — epochs 192-200 never enter it. So `checkpoint['ema']` is essentially a
**snapshot of epoch ~101**, not an average, and `‖raw − ema‖/‖raw‖` measures 29% (brain) / 33% (Mayo).

That snapshot is the better model, and the gap survives the K we deploy at:

| | | MAE ↓ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| brain 214841, nfe3, K=10 | `ema` | **1.997** | **0.8338** | **0.0583** |
| | `model` (raw) | 2.156 | 0.8001 | 0.0639 |
| Mayo L310, nfe3, K=10 | `ema` | **13.682** | **0.6733** | **0.0517** |
| | `model` (raw) | 14.038 | 0.6620 | 0.0547 |

The flow model degrades late; the buggy EMA accidentally freezes it near a better epoch. (Measure
this at K=10, never K=1 — at K=1 the same gaps are 3-4× larger and a GAN checkpoint's own raw-vs-EMA
gap looks decisive at K=1 yet collapses to nothing by K=10, because that one is only sample texture,
which averaging removes.)

`load_generator` historically read `data['model']` only, so **every GAN result so far started from the
worse of the two**, while the no-GAN baseline it is compared against is scored on the better one.
`--pretrained_weights ema` starts from those same weights and removes that mismatch.

- Default stays `model` — it reproduces every reported GAN number.
- `ema` results are a different régime: **rerun the baseline side before putting them in one table.**
- It does **not** make the comparison clean by itself: the GAN still gets epochs the baseline never
  had. To attribute anything to the adversarial loss you still need an `adv_weight=0` control —
  `--adv_start_step 999999999` gives one free, since `use_adv` gates both the G-side loss and the
  entire D update, leaving a pure low-LR flow fine-tune with the same init, epochs and EMA.

Implementation note: `data['ema']` is not a plain state_dict. It holds `initted`, `step`, **and both**
`ema_model.*` and `online_model.*`. Only the first is averaged — `online_model.*` has identical key
names and measures **0.00%** away from `data['model']`, so a sloppy prefix filter silently loads the
raw weights back and quietly wastes the run. The loader matches the prefix explicitly and asserts no
generator param is missing.

## Verified configs

Each trial's `ema.step ÷ epoch` pins down the batch size it actually ran with (the trainer updates the
EMA once per optimizer step), and all three CLI defaults check out. Sample count is
`pairs × num_slices_per_image × num_patches_per_slice`.

| trial | samples/epoch | steps/epoch | batch | script | default |
|---|---|---|---|---|---|
| `imf_gan_unsupervised_gaussian_brainCT` | 68×50×2 = 6800 | 1700 | 4 | `train_2D_imf_gan.py` | ✅ 4 |
| `imf_gan_nfe3_unsupervised_gaussian_brainCT` | 6800 | 3400 | 2 | `train_2D_imf_gan_nfe3.py` | ✅ 2 |
| `imf_gan_unsupervised_gaussian_mayo` | 6×50×2 = 600 | 300 | 2 | `train_2D_imf_gan_mayo.py` | ✅ 2 |
| `imf_gan_adv02_unsupervised_gaussian_mayo` | 600 | 300 | 2 | `run_gan_mayo_adv02.sh` | ✅ 2 |

Reproduce the deployed brain GAN (adv 0.5, ep28) — defaults are already correct:

```bash
python gan/train_2D_imf_gan.py \
  --pretrained .../imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt \
  --adv_weight 0.5 --adv_nfe 1 --train_num_steps 28
```

Same run, but starting from the better weights (new trial name — see below):

```bash
python gan/train_2D_imf_gan.py \
  --trial_name imf_gan_emainit_unsupervised_gaussian_brainCT \
  --pretrained .../imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt \
  --pretrained_weights ema \
  --adv_weight 0.5 --adv_nfe 1 --train_num_steps 28
```

## Landmines

**`--trial_name` defaults to the existing trial, and `--save_every` defaults to 1.** Re-running
`train_2D_imf_gan.py` with defaults writes `model-1 … model-N` straight over
`imf_gan_unsupervised_gaussian_brainCT/models/`, i.e. over `model-28.pt` — the deployed brain GAN
every reported number came from. Always pass a fresh `--trial_name` for a new configuration.

**`gan_train.log` in the repo root is not the adv-0.5 run.** Its tqdm shows 3400 steps/epoch, which is
the nfe3 trial's batch-2 config; the adv-0.5 trial ran 1700. Don't read timings or settings for one
trial out of the other's log — check `ema.step ÷ epoch` instead.

**`ema.step` is the régime marker.** GAN trials update the EMA per optimizer step, so their step counts
land in the thousands; flow trials used the legacy per-epoch path and land at ~200 (= the epoch count).
A GAN checkpoint whose step is far *below* its neighbours was written by a restarted run and is not
"N epochs of the same trajectory" — `imf_gan_unsupervised_gaussian_brainCT/model-50.pt` (step 10650,
against model-44's 74800) and `imf_gan_nfe3_.../model-21.pt` (5964, against model-2's 6800) are both
like this. Treat their epoch numbers as unreliable.

**`Sampler.load_model` (`improved_mean_flow.py`) is what reads these checkpoints, not
`Trainer.load_model`.** The Sampler needs only `model`/`step`/`ema`, which is why
`predict_2D_imf_v2.py --trial_name imf_gan_...` works even though a GAN checkpoint has no `opt` or
`decay_steps`.

## Evaluating a GAN trial

Inference is the ordinary predict pointed at the GAN trial; `--weights {ema,raw}` picks which slot to
sample from and is encoded into the output directory so the two never collide:

```bash
python Thinslice_experiments/predict_2D_imf_v2.py --mode pred --trial_name imf_gan_... --epoch 28 \
  --num_steps 3 --iteration_num 10 --patient 214841 [--weights raw]
python Thinslice_experiments/predict_2D_imf_v2.py --mode avg  --trial_name imf_gan_... --epoch 28 \
  --num_steps 3 --iteration_num 10 --k_save 10 [--weights raw]
python gan/eval_gan_nfe.py --trial imf_gan_... --epoch 28 --nfe 3
```

Judge at the deployment K (10/20), never K=1 — both the `fv_evolution` high-frequency proxy and K=1
metrics have each picked the wrong epoch/method in this project before.
