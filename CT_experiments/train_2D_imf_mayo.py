"""
train_2D_imf_mayo.py — Mayo Low-Dose CT denoising with improved MeanFlow (iMF). **NO GAN.**

Self-supervised, Noise2Noise style: the network predicts one half-projection reconstruction
(`recon_even`) from the other (`recon_odd`) — two independent noisy realizations of the SAME
slice. No clean image is used during training (`ground_truth_file` is for evaluation only).
`switch_odd_and_even_frequency=0.5` randomises which half is condition vs target each draw.

This is the Mayo counterpart of `Thinslice_experiments/train_2D_imf.py` (brain CT). Differences:
  * pairing      : odd/even recon (N2N) instead of adjacent slices  -> uses `Build_list.Build`
                   and `IMF_denoising.Generator` (NOT Generator_thinslice).
  * window       : abdomen HU [-200, 250], NO histogram equalization.
  * condition    : 1 channel (a single recon), patch 256x256.
Backbone is identical to the brain-CT v2 model: U-Net (+ auxiliary v-head) + ImprovedMeanFlow.

Cleaned/parameterised replacement for the old `CT_experiments/train_2D_imf.py`, whose hard-coded
paths (`/host/d/file/新建文件夹/mayo/...`) are stale. Paths are now auto-detected + argparse-able.

Data layout
-----------
  patient list : <base>/新mayo_data/mayo_low_dose_CT_gaussian_simulation_highnoise_v2.xlsx
                 (columns: batch, Patient_ID, simulation_file_odd/even/all, ground_truth_file, ...)
  recon files  : stored in the xlsx as `/host/e/D/Data/low_dose_CT/...` (the **E:** drive).
                 -> the training environment MUST mount E: at /host/e. The default
                    pytorch_container only mounts C:/D:, so remount E: before training.
  models out   : <base>/projects/denoising/models/<trial_name>/models/

Usage (inside the docker env, with E: mounted at /host/e and a GPU)
    python CT_experiments/train_2D_imf_mayo.py
    python CT_experiments/train_2D_imf_mayo.py --train_num_steps 200 --train_batch_size 1
    python CT_experiments/train_2D_imf_mayo.py --patient_list_file <xlsx> --trial_name <name>
"""
import os
import sys
import argparse

# --- make `import IMF_denoising...` work regardless of where the repo is mounted ---
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../IMF_denoising/CT_experiments
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))        # dir that CONTAINS the IMF_denoising folder
for _p in (_REPO_PARENT, '/host/c/Users/ROG/Documents/GitHub'):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import torch  # noqa: F401  (imported so a missing-torch env fails early, before data work)

import IMF_denoising.improved_mean_flow as imf
import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator as Generator          # odd/even Mayo generator (NOT Generator_thinslice)
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet


def _detect_base():
    """Where the Mayo xlsx + model outputs live. Docker mounts D: at /host/d (data under
    D:\\research). Falls back across the usual mount points."""
    for b in ('/host/d/research', '/host/d', 'D:/research', '/d/research'):
        if os.path.isdir(os.path.join(b, '新mayo_data')) or os.path.isdir(os.path.join(b, 'projects/denoising')):
            return b
    return '/host/d/research'


_BASE = _detect_base()


def _remap(p):
    """Resolve a stored recon path across mounts. The xlsx stores `/host/e/...` (E: drive); on a
    Windows host that is `E:/` (Git-Bash `/e/`), in docker it is `/host/e` (needs E: mounted)."""
    if p is None:
        return p
    p = str(p)
    if os.path.exists(p):
        return p
    for src, dsts in (('/host/e', ('/host/e', 'E:', '/e')),
                      ('/host/d', ('/host/d/research', '/host/d', 'D:', '/d'))):
        if p.startswith(src + '/'):
            tail = p[len(src):]
            for d in dsts:
                cand = d + tail
                if os.path.exists(cand):
                    return cand
    return p


def get_args():
    ap = argparse.ArgumentParser('Mayo low-dose CT iMF (no GAN) training')
    ap.add_argument('--trial_name', default='imf_v2_unsupervised_gaussian_mayo')
    ap.add_argument('--patient_list_file',
                    default=os.path.join(_BASE, '新mayo_data/mayo_low_dose_CT_gaussian_simulation_highnoise_v2.xlsx'))
    ap.add_argument('--study_folder', default=os.path.join(_BASE, 'projects/denoising/models'))
    ap.add_argument('--batch_train', nargs='+', default=['train'], help="xlsx 'batch' value(s) for training")
    ap.add_argument('--batch_val',   nargs='+', default=['val'],   help="xlsx 'batch' value(s) for validation")
    ap.add_argument('--train_num_steps', type=int, default=200, help='number of epochs')
    ap.add_argument('--train_batch_size', type=int, default=1)
    ap.add_argument('--train_lr', type=float, default=1e-4)
    ap.add_argument('--patch_size', type=int, nargs=2, default=[256, 256])
    ap.add_argument('--num_patches_per_slice', type=int, default=2)
    ap.add_argument('--background_cutoff', type=float, default=-200.0, help='abdomen window low (HU)')
    ap.add_argument('--maximum_cutoff', type=float, default=250.0, help='abdomen window high (HU)')
    ap.add_argument('--no_preload', action='store_true', help='disable in-RAM preload of recon volumes')
    ap.add_argument('--save_every', type=int, default=5)
    ap.add_argument('--validation_every', type=int, default=5)
    ap.add_argument('--pre_trained_model', default=None)
    ap.add_argument('--start_step', type=int, default=0)
    return ap.parse_args()


def main():
    args = get_args()
    supervision = 'unsupervised'
    condition_channel = 1            # Mayo: a single recon as condition (odd OR even)
    image_size = [512, 512]
    histogram_equalization = False   # abdomen: plain HU window, no hist-eq
    normalize_factor = 'equation'
    preload = not args.no_preload

    print('data base   :', _BASE)
    print('patient list:', args.patient_list_file)
    if not os.path.isfile(args.patient_list_file):
        raise FileNotFoundError(args.patient_list_file)

    # ---- patient list (Mayo Build: odd/even recon columns) ----
    build_sheet = Build_list.Build(args.patient_list_file)
    _, _, _, _, odd_tr, even_tr, _, _ = build_sheet.__build__(batch_list=args.batch_train)
    _, _, _, _, odd_va, even_va, _, _ = build_sheet.__build__(batch_list=args.batch_val)

    # N2N pairing: target = EVEN recon, condition = ODD recon (switch_odd_and_even randomises it)
    x0_tr,  cond_tr = np.array([_remap(p) for p in even_tr]), np.array([_remap(p) for p in odd_tr])
    x0_va,  cond_va = np.array([_remap(p) for p in even_va]), np.array([_remap(p) for p in odd_va])
    print('train cases:', x0_tr.shape[0], '| val cases:', x0_va.shape[0])
    print('x0[0]  :', x0_tr[0])
    print('cond[0]:', cond_tr[0])

    # ---- backbone: same U-Net (+aux v-head) as the brain-CT v2 model ----
    base_model = Unet(
        problem_dimension='2D', init_dim=64, out_dim=1, channels=1,
        conditional_diffusion=True, condition_channels=condition_channel,
        downsample_list=(True, True, True, False), upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True), auxiliary_v_head=True,
    )
    diffusion_model = imf.ImprovedMeanFlow(
        base_model, image_size=args.patch_size, ratio_r_neq_t=0.50,
        clip_or_not=False, auto_normalize=False, adaptive_weight_power=1.0, v_loss_weight=0.5,
    )

    # ---- optional in-RAM preload of the recon volumes (faster, needs the data mounted) ----
    pre_tr = (ff.preload_data(x0_tr), ff.preload_data(cond_tr)) if preload else None
    pre_va = (ff.preload_data(x0_va), ff.preload_data(cond_va)) if preload else None

    G = Generator.Dataset_2D
    gen_tr = G(
        supervision=supervision, preload=preload, preload_data=pre_tr,
        img_list=x0_tr, condition_list=cond_tr, image_size=image_size,
        num_slices_per_image=50, random_pick_slice=True, slice_range=None,
        num_patches_per_slice=args.num_patches_per_slice, patch_size=args.patch_size,
        histogram_equalization=histogram_equalization, bins=None, bins_mapped=None,
        background_cutoff=args.background_cutoff, maximum_cutoff=args.maximum_cutoff,
        normalize_factor=normalize_factor, shuffle=True, augment=True, augment_frequency=0.5,
        switch_odd_and_even_frequency=0.5,
    )
    gen_va = G(
        supervision=supervision, preload=preload, preload_data=pre_va,
        img_list=x0_va, condition_list=cond_va, image_size=image_size,
        num_slices_per_image=90, random_pick_slice=False, slice_range=[100, 190],
        num_patches_per_slice=1, patch_size=[512, 512],
        histogram_equalization=histogram_equalization, bins=None, bins_mapped=None,
        background_cutoff=args.background_cutoff, maximum_cutoff=args.maximum_cutoff,
        normalize_factor=normalize_factor,
    )

    # ---- train ----
    save_folder = os.path.join(args.study_folder, args.trial_name, 'models')
    ff.make_folder([os.path.dirname(save_folder), save_folder])
    print('save models ->', save_folder)

    trainer = imf.Trainer(
        diffusion_model=diffusion_model, generator_train=gen_tr, generator_val=gen_va,
        train_batch_size=args.train_batch_size, accum_iter=1,
        train_num_steps=args.train_num_steps, results_folder=save_folder,
        train_lr=args.train_lr, train_lr_decay_every=200,
        save_models_every=args.save_every, validation_every=args.validation_every,
    )
    trainer.train(pre_trained_model=args.pre_trained_model, start_step=args.start_step)


if __name__ == '__main__':
    main()
