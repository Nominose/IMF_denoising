"""
train_2D_imf_gan_pcct.py — GAN fine-tuning of the iMF generator on real-world PCCT.

PCCT counterpart of gan/train_2D_imf_gan.py (brain CT). Same two-stage recipe: load the
FLOW-pretrained checkpoint from PCCT_experiments/train_2D_imf_pcct.py, then fine-tune it with a
small adversarial loss (L_flow + adv_weight * L_adv). The discriminator pushes the one-step
prediction toward the real (noisy) target distribution, which recovers texture the few-step flow
tends to over-smooth. Inference is unchanged afterwards (the discriminator is discarded).

Differences from the brain-CT GAN, both inherited from the PCCT data itself:
  * NO histogram equalisation, HU window [-1000, 1000]   (brain: hist-eq on, [-1000, 2000])
  * all 22 training cases are used (the brain script subsampled every 2nd case)
Pairing is identical -- adjacent-slice N2N via Generator_thinslice, condition_channels=2.

Init: --pretrained_weights defaults to 'model' (the checkpoint's ONLINE weights), matching how the
brain and Mayo GAN results in the paper tables were produced. 'ema' is available and is the fairer
init when the no-GAN baseline is scored on EMA weights, but on brain it was worth only ~0.5% MAE /
~3% LPIPS -- not enough to justify making PCCT incomparable with the other two datasets.

    python gan/train_2D_imf_gan_pcct.py \
        --pretrained <base>/projects/denoising/models/imf_v2_unsupervised_PCCT/models/model-200.pt
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_PARENT, '/gpfs/work/aac/xingyiyao23/Code', '/host/c/Users/ROG/Documents/GitHub'):
    if _p not in sys.path:
        sys.path.append(_p)

import argparse
import numpy as np

import IMF_denoising.improved_mean_flow as imf
import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator_thinslice as Generator
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet
from IMF_denoising.gan.imf_gan import PatchDiscriminator, GANTrainer


def _detect_base():
    for b in ('/gpfs/work/aac/xingyiyao23', '/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'Data')):
            return b
    return '/gpfs/work/aac/xingyiyao23'


_BASE = _detect_base()
_PCCT = os.path.join(_BASE, 'Data', 'PCCT')
_SLICED = 'soft_thins_0_noblank_sliced.nii.gz'

COND_CH = 2
IMG = [512, 512]
PATCH = [128, 128]
HE, BG, MX, NF = False, -1000, 1000, 'equation'    # PCCT: no hist-eq, [-1000,1000] HU


def _remap(p, case=None):
    """Resolve a patient-list path onto the uploaded PCCT data (directory AND filename may be stale;
    see PCCT_experiments/fix_pcct_xlsx.py)."""
    if p is None:
        return p
    p = str(p)
    if os.path.exists(p):
        return p
    if case is not None:
        for name in (_SLICED, 'soft_thins_0_noblank.nii.gz'):
            cand = os.path.join(_PCCT, 'soft_thins_xy', str(case), name)
            if os.path.isfile(cand):
                return cand
    for marker in ('/soft_thins/', '/soft_thins_xy/'):
        if marker in p:
            cand = os.path.join(_PCCT, 'soft_thins_xy', p.split(marker, 1)[1])
            if os.path.isfile(cand):
                return cand
            cand = os.path.join(os.path.dirname(cand), _SLICED)
            if os.path.isfile(cand):
                return cand
    return p


def get_args():
    p = argparse.ArgumentParser('iMF GAN fine-tuning (real-world PCCT)')
    p.add_argument('--trial_name', type=str, default='imf_gan_unsupervised_PCCT')
    p.add_argument('--pretrained', type=str,
                   default=os.path.join(_BASE, 'projects/denoising/models/imf_v2_unsupervised_PCCT/models/model-200.pt'),
                   help='flow-pretrained PCCT generator to fine-tune')
    p.add_argument('--pretrained_weights', type=str, default='model', choices=['model', 'ema'],
                   help="Which weights inside --pretrained to start from. 'model' (default) matches "
                        "how the brain/Mayo GAN tables were produced. 'ema' removes the init mismatch "
                        "vs an EMA-scored no-GAN baseline but makes the numbers incomparable to them.")
    p.add_argument('--patient_list_file', default=os.path.join(_PCCT, 'Patient_lists', 'PCCT_split_hpc.xlsx'))
    p.add_argument('--batch_train', nargs='+', type=int, default=[0])
    p.add_argument('--train_num_steps', type=int, default=50, help='epochs of GAN fine-tuning')
    p.add_argument('--batch_size', type=int, default=16,
                   help='16: bigger batch -> less noisy D gradient, and keeps the A800 busy')
    p.add_argument('--lr_g', type=float, default=1e-4)
    p.add_argument('--lr_d', type=float, default=2e-4)
    p.add_argument('--adv_weight', type=float, default=0.5,
                   help='beta in L_flow + beta*L_adv; 0.5 keeps the flow loss dominant')
    p.add_argument('--r1_gamma', type=float, default=1.0)
    p.add_argument('--adv_start_step', type=int, default=0)
    p.add_argument('--adv_nfe', type=int, default=1,
                   help='adversarial fake = a differentiable adv_nfe-step generation. 1 = single-step F(v)')
    p.add_argument('--save_every', type=int, default=7,
                   help='7 -> keeps epochs 7/14/.. (~571MB each). GAN quality is non-monotone, so keep '
                        'a few; fv_evolution/ dumps every epoch regardless.')
    return p.parse_args()


def main():
    args = get_args()
    print('data base:', _BASE, '| trial:', args.trial_name)
    if not os.path.isfile(args.patient_list_file):
        raise FileNotFoundError(
            f'{args.patient_list_file}\nRun PCCT_experiments/unpack_pcct_hpc.sh then '
            'python PCCT_experiments/fix_pcct_xlsx.py --write')

    bs = Build_list.Build_thinsliceCT(args.patient_list_file)
    _, pid_tr, _, _, cond_tr, _ = bs.__build__(batch_list=args.batch_train)
    cond_tr = np.array([_remap(p, c) for p, c in zip(cond_tr, pid_tr)])
    x0_tr = cond_tr                       # N2N: target and condition are the SAME volume
    miss = [p for p in cond_tr if not os.path.isfile(p)]
    if miss:
        raise FileNotFoundError(f'{len(miss)} training volume(s) missing, e.g. {miss[:3]}')
    print('train cases:', x0_tr.shape[0])

    gen_tr = Generator.Dataset_2D(
        supervision='unsupervised', img_list=x0_tr, condition_list=cond_tr, image_size=IMG,
        # 48, not 50 -- adjacent-slice conditioning drops the first and last slice; see
        # PCCT_experiments/train_2D_imf_pcct.py for why asking for 50 raises IndexError.
        num_slices_per_image=48, random_pick_slice=True, slice_range=None,
        num_patches_per_slice=2, patch_size=PATCH,
        histogram_equalization=HE, bins=None, bins_mapped=None,
        background_cutoff=BG, maximum_cutoff=MX, normalize_factor=NF,
        shuffle=True, augment=True, augment_frequency=0.5)

    # generator: same arch as the flow checkpoint (v-head, 2-channel condition)
    base = Unet(problem_dimension='2D', init_dim=64, out_dim=1, channels=1,
                conditional_diffusion=True, condition_channels=COND_CH,
                downsample_list=(True, True, True, False), upsample_list=(True, True, True, False),
                full_attn=(None, None, False, True), auxiliary_v_head=True)
    G = imf.ImprovedMeanFlow(base, image_size=PATCH, ratio_r_neq_t=0.5, clip_or_not=False, auto_normalize=False)

    # UNCONDITIONAL D (cond_channels=0): the real-vs-fake signal is "noisy target vs smooth
    # generation", which lives in the image alone -> dropping the condition avoids diluting it.
    D = PatchDiscriminator(img_channels=1, cond_channels=0, base=64, n_layers=3)

    save_models = os.path.join(_BASE, 'projects/denoising/models', args.trial_name, 'models')
    ff.make_folder([os.path.dirname(os.path.dirname(save_models)), os.path.dirname(save_models), save_models])

    # full-slice F(v) probe: a fixed slice of case 0, dumped complete (512x512) every epoch so the
    # one-step output can be watched evolving (fv_evolution/fv_fullslice_epoch*.npy).
    FS_SLICE = 25                          # mid-volume; every PCCT volume is 50 slices
    fs_probe = None
    try:
        gen_fs = Generator.Dataset_2D(
            supervision='unsupervised', img_list=x0_tr[:1], condition_list=cond_tr[:1], image_size=IMG,
            num_slices_per_image=1, random_pick_slice=False, slice_range=[FS_SLICE, FS_SLICE + 1],
            num_patches_per_slice=None, patch_size=None,
            histogram_equalization=HE, bins=None, bins_mapped=None,
            background_cutoff=BG, maximum_cutoff=MX, normalize_factor=NF,
            shuffle=False, augment=False)
        fs_real, fs_cond = gen_fs[0]
        fs_probe = (fs_real.unsqueeze(0), fs_cond.unsqueeze(0))
        print(f'[fs] full-slice probe = slice {FS_SLICE}')
    except Exception as e:
        print(f'[fs] full-slice probe disabled ({str(e)[:90]})')

    trainer = GANTrainer(
        diffusion_model=G, discriminator=D, generator_train=gen_tr,
        train_batch_size=args.batch_size, train_num_steps=args.train_num_steps,
        results_folder=save_models, lr_g=args.lr_g, lr_d=args.lr_d,
        adv_weight=args.adv_weight, r1_gamma=args.r1_gamma, adv_start_step=args.adv_start_step,
        adv_nfe=args.adv_nfe, save_every=args.save_every, fs_probe=fs_probe)

    if args.pretrained and os.path.isfile(args.pretrained):
        trainer.load_generator(args.pretrained, key=args.pretrained_weights)
    else:
        raise FileNotFoundError(
            f'pretrained flow model not found: {args.pretrained}\n'
            'GAN fine-tuning REQUIRES it (from-scratch GAN is unstable) -- run '
            'PCCT_experiments/run_train_imf_pcct.sh first.')

    trainer.train()


if __name__ == '__main__':
    main()
