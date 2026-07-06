"""
train_2D_imf_gan_mayo.py — GAN fine-tuning of the iMF generator on Mayo low-dose CT (N2N, gaussian sim).

Mayo counterpart of gan/train_2D_imf_gan.py (brain CT). Two-stage recipe: load the FLOW-pretrained
Mayo model (imf_v2_unsupervised_gaussian_mayo/model-200.pt), then fine-tune it with a small
adversarial loss (L_flow + beta * L_adv). The UNCONDITIONAL high-pass PatchGAN D pushes the model's
few-step generation toward the real NOISY even-recon distribution -> more faithful few-step texture
(better LPIPS at low NFE). Inference is UNCHANGED (D discarded; sample with CT_experiments/predict_2D_imf.py).
Writes to a NEW trial dir; touches nothing existing.

Differences vs the brain GAN trainer (this mirrors CT_experiments/train_2D_imf_mayo.py's data setup):
  * pairing   : odd/even half-projection recon (Noise2Noise) via Build_list.Build + IMF_denoising.Generator
                (NOT adjacent slices / Generator_thinslice). target = EVEN recon, condition = ODD recon;
                switch_odd_and_even_frequency=0.5 randomises which half is condition vs target each draw.
  * window    : abdomen HU [-200, 250], NO histogram equalization (brain used hist-eq [-1000, 2000]).
  * condition : 1 channel (a single recon), patch 256x256 (matches the Mayo flow pretrain).
The GAN machinery (PatchDiscriminator + GANTrainer: hinge + lazy-R1, adv_nfe-step differentiable
rollout, EMA, per-epoch F(v) dump) is identical to the brain run and lives in gan/imf_gan.py.

Run inside docker (tmux), on the GPU that trained the Mayo flow model:
    python gan/train_2D_imf_gan_mayo.py
    python gan/train_2D_imf_gan_mayo.py --adv_weight 0.5 --batch_size 2
    python gan/train_2D_imf_gan_mayo.py --adv_nfe 3            # NFE=3-direct variant (heavier: 3x rollout backprop)
Then pick the best epoch from fv_evolution/ + a short predict/eval sweep, exactly like the brain run.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))              # dir that CONTAINS the IMF_denoising folder
for _p in (_PARENT, '/host/c/Users/ROG/Documents/GitHub'):
    if _p not in sys.path:
        sys.path.append(_p)

import argparse
import numpy as np

import IMF_denoising.improved_mean_flow as imf
import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator as Generator          # Mayo odd/even generator (NOT Generator_thinslice)
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet
from IMF_denoising.gan.imf_gan import PatchDiscriminator, GANTrainer


def _detect_base():
    """Data root holding Data/新mayo_data + projects/denoising. Docker mounts D: at /host/d
    (data under D:\\research); falls back across the usual mount points / host drive letters."""
    for b in ('/host/d/research', '/host/d', 'D:/research', '/d/research'):
        if os.path.isdir(os.path.join(b, 'Data', '新mayo_data')):
            return b
    return '/host/d/research'


_BASE = _detect_base()
_MAYO_DATA = os.path.join(_BASE, 'Data', '新mayo_data')


def _remap(p):
    """Resolve a stored recon path. The xlsx records `/host/e/D/Data/low_dose_CT/<rest>` (old E:
    layout); the data now also lives on D: at `<base>/Data/新mayo_data/<rest>`. Try as-is first
    (works if E: is mounted), then remap onto the D: copy so NO E: mount is needed."""
    if p is None:
        return p
    p = str(p)
    if os.path.exists(p):
        return p
    old = '/host/e/D/Data/low_dose_CT'
    if p.startswith(old + '/'):
        cand = _MAYO_DATA + p[len(old):]
        if os.path.exists(cand):
            return cand
    return p


COND_CH = 1                          # Mayo: a single recon (odd OR even) as condition
IMG = [512, 512]
HE, BG, MX, NF = False, -200.0, 250.0, 'equation'   # abdomen window, NO histogram equalization


def get_args():
    p = argparse.ArgumentParser('iMF GAN fine-tuning (Mayo low-dose CT)')
    p.add_argument('--trial_name', type=str, default='imf_gan_unsupervised_gaussian_mayo')
    p.add_argument('--pretrained', type=str,
                   default=os.path.join(_BASE, 'projects/denoising/models/imf_v2_unsupervised_gaussian_mayo/models/model-200.pt'),
                   help='flow-pretrained Mayo generator to fine-tune (loads its "model" weights)')
    p.add_argument('--patient_list_file',
                   default=os.path.join(_MAYO_DATA, 'mayo_low_dose_CT_gaussian_simulation_highnoise_v2.xlsx'))
    p.add_argument('--batch_train', nargs='+', default=['train'], help="xlsx 'batch' value(s) for training")
    p.add_argument('--train_num_steps', type=int, default=50, help='epochs of GAN fine-tuning')
    p.add_argument('--batch_size', type=int, default=2,
                   help="2 @ patch 256 (~2x the brain run's activation memory). Raise if VRAM allows; "
                        "lower it (or drop --patch_size to 128 128) if you OOM.")
    p.add_argument('--patch_size', type=int, nargs=2, default=[256, 256],
                   help='GAN patch size (matches the Mayo flow pretrain). 128 128 is lighter.')
    p.add_argument('--num_patches_per_slice', type=int, default=2)
    p.add_argument('--lr_g', type=float, default=1e-4, help='<= lr_d so G responds to adv; watch flow loss stays bounded')
    p.add_argument('--lr_d', type=float, default=4e-4, help='faster D so it learns the subtle noise-texture signal')
    p.add_argument('--adv_weight', type=float, default=0.5,
                   help='beta in L_flow + beta*L_adv; 0.5 keeps flow dominant (~0.7) while adv has real influence')
    p.add_argument('--r1_gamma', type=float, default=0.01, help='R1 strength (the ONLY D regulariser); keep > 0')
    p.add_argument('--adv_nfe', type=int, default=1,
                   help='adversarial fake = differentiable adv_nfe-step generation. 1 = single-step F(v) '
                        '(proven brain recipe); 3 = NFE=3-direct (heavier: backprops through 3 forwards).')
    p.add_argument('--adv_start_step', type=int, default=0, help='warmup: flow-only steps before turning GAN on')
    p.add_argument('--save_every', type=int, default=1,
                   help='1 = every epoch. GAN quality is non-monotone (peaks then can degrade), so '
                        'checkpoint finely to catch the best epoch (prune bad ones after).')
    p.add_argument('--fs_slice', type=int, default=100, help='slice index for the full-slice F(v) monitor probe')
    p.add_argument('--no_preload', action='store_true', help='disable in-RAM preload of the recon volumes')
    return p.parse_args()


def main():
    args = get_args()
    supervision = 'unsupervised'
    PATCH = list(args.patch_size)
    preload = not args.no_preload
    print('data base:', _BASE, '| trial:', args.trial_name, '| adv_nfe:', args.adv_nfe, '| patch:', PATCH)
    print('patient list:', args.patient_list_file)
    if not os.path.isfile(args.patient_list_file):
        raise FileNotFoundError(args.patient_list_file)

    # ---- patient list (Mayo Build: odd/even recon columns) ----
    build_sheet = Build_list.Build(args.patient_list_file)
    _, _, _, _, odd_tr, even_tr, _, _ = build_sheet.__build__(batch_list=args.batch_train)
    # N2N pairing: target = EVEN recon, condition = ODD recon (switch_odd_and_even randomises it)
    x0_tr = np.array([_remap(p) for p in even_tr])
    cond_tr = np.array([_remap(p) for p in odd_tr])
    print('train pairs:', x0_tr.shape[0])
    print('x0[0]  :', x0_tr[0])
    print('cond[0]:', cond_tr[0])

    # optional in-RAM preload of the recon volumes (faster; needs the data reachable)
    pre_tr = (ff.preload_data(x0_tr), ff.preload_data(cond_tr)) if preload else None

    gen_tr = Generator.Dataset_2D(
        supervision=supervision, preload=preload, preload_data=pre_tr,
        img_list=x0_tr, condition_list=cond_tr, image_size=IMG,
        num_slices_per_image=50, random_pick_slice=True, slice_range=None,   # full volume: 150-200 impossible for 128-slice cases & the generator doesn't clamp; train range doesn't affect the 150-200 eval
        num_patches_per_slice=args.num_patches_per_slice, patch_size=PATCH,
        histogram_equalization=HE, bins=None, bins_mapped=None,
        background_cutoff=BG, maximum_cutoff=MX, normalize_factor=NF,
        shuffle=True, augment=True, augment_frequency=0.5,
        switch_odd_and_even_frequency=0.5)

    # generator: same arch as the Mayo flow model (aux v-head, 1-channel condition)
    base = Unet(problem_dimension='2D', init_dim=64, out_dim=1, channels=1,
                conditional_diffusion=True, condition_channels=COND_CH,
                downsample_list=(True, True, True, False), upsample_list=(True, True, True, False),
                full_attn=(None, None, False, True), auxiliary_v_head=True)
    G = imf.ImprovedMeanFlow(base, image_size=PATCH, ratio_r_neq_t=0.5, clip_or_not=False, auto_normalize=False)

    # UNCONDITIONAL high-pass PatchGAN D (cond_channels=0): the real-vs-fake signal here is "noisy
    # recon vs smooth few-step generation", which lives in the image alone -> dropping the condition
    # removes its dilution of that subtle noise-texture signal. Set cond_channels=COND_CH to go back.
    D = PatchDiscriminator(img_channels=1, cond_channels=0, base=64, n_layers=3)

    save_models = os.path.join(_BASE, 'projects/denoising/models', args.trial_name, 'models')
    ff.make_folder([os.path.dirname(os.path.dirname(save_models)), os.path.dirname(save_models), save_models])

    # full-slice F(v) probe: one fixed slice of case 0 -> the trainer dumps a COMPLETE 512x512 slice
    # each epoch (fv_fullslice_epoch*.npy in fv_evolution/), not just the training patch. Best-effort:
    # disabled gracefully if that slice index is out of range for case 0.
    fs_probe = None
    try:
        gen_fs = Generator.Dataset_2D(
            supervision=supervision, preload=False, preload_data=None,
            img_list=x0_tr[:1], condition_list=cond_tr[:1], image_size=IMG,
            num_slices_per_image=1, random_pick_slice=False, slice_range=[args.fs_slice, args.fs_slice + 1],
            num_patches_per_slice=1, patch_size=[512, 512],
            histogram_equalization=HE, bins=None, bins_mapped=None,
            background_cutoff=BG, maximum_cutoff=MX, normalize_factor=NF,
            shuffle=False, augment=False)
        fs_real, fs_cond = gen_fs[0]
        fs_probe = (fs_real.unsqueeze(0), fs_cond.unsqueeze(0))   # (1,1,512,512), (1,1,512,512)
        print(f'[fs] full-slice probe = slice {args.fs_slice}')
    except Exception as e:
        print(f'[fs] full-slice probe disabled ({str(e)[:90]})')

    trainer = GANTrainer(
        diffusion_model=G, discriminator=D, generator_train=gen_tr,
        train_batch_size=args.batch_size, train_num_steps=args.train_num_steps,
        results_folder=save_models, lr_g=args.lr_g, lr_d=args.lr_d,
        adv_weight=args.adv_weight, r1_gamma=args.r1_gamma, adv_start_step=args.adv_start_step,
        adv_nfe=args.adv_nfe, save_every=args.save_every, fs_probe=fs_probe)

    if args.pretrained and os.path.isfile(args.pretrained):
        trainer.load_generator(args.pretrained, key='model')
    else:
        print(f'[GAN] WARNING: pretrained not found ({args.pretrained}) — GAN-from-scratch is unstable.', flush=True)

    trainer.train()


if __name__ == '__main__':
    main()
