"""
train_2D_imf_gan_nfe3.py — NFE=3-DIRECT adversarial fine-tuning of the iMF generator (brain CT).

Same recipe as train_2D_imf_gan.py, but the adversarial fake is the model's TRUE 3-step generation
from pure noise (adv_nfe=3) instead of the single-step NFE=1 output. Use this when you have committed
to NFE=3 as the deploy point and want to optimise the NFE=3 LPIPS DIRECTLY (no reliance on the
NFE=1 -> NFE>1 propagation).

  * fake = differentiable 3-step Euler rollout from z ~ N(0,I)  (the actual NFE=3 inference object)
  * D pushes that NFE=3 output toward the real noisy x2 distribution; flow loss keeps G a denoiser
  * COST: the adversarial path backprops through 3 chained forwards -> ~3x the generator compute and
    more memory than NFE=1, and the longer gradient is somewhat less stable. batch is smaller here.

Writes to a SEPARATE trial folder (imf_gan_nfe3_...) so it never collides with the NFE=1 GAN run.
Inference is unchanged (discriminator discarded; sample with predict_2D_imf_v2.py at NFE=3).

Run inside docker (tmux):
    python gan/train_2D_imf_gan_nfe3.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_PARENT, '/host/c/Users/ROG/Documents/GitHub'):
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
    for b in ('/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'Data')):
            return b
    return '/host/d/research'


_BASE = _detect_base()


def _remap(p):
    """Remap stale '/host/d/...' patient-list paths onto the real data root (/host/d/research/...)."""
    if p is None:
        return p
    p = str(p)
    if os.path.exists(p):
        return p
    if p.startswith('/host/d/') and not p.startswith(_BASE + '/'):
        c = _BASE + p[len('/host/d'):]
        if os.path.exists(c):
            return c
    return p


COND_CH = 2
IMG = [512, 512]
PATCH = [128, 128]
HE, BG, MX, NF = True, -1000, 2000, 'equation'


def get_args():
    p = argparse.ArgumentParser('iMF GAN fine-tuning, NFE=3-direct (brain CT)')
    p.add_argument('--trial_name', type=str, default='imf_gan_nfe3_unsupervised_gaussian_brainCT',
                   help='SEPARATE folder from the NFE=1 run so they never collide')
    p.add_argument('--pretrained', type=str,
                   default=os.path.join(_BASE, 'projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt'),
                   help='flow-pretrained generator checkpoint to fine-tune (loads its "model" weights)')
    p.add_argument('--train_num_steps', type=int, default=50, help='epochs of GAN fine-tuning')
    p.add_argument('--batch_size', type=int, default=2,
                   help='2 (smaller than the NFE=1 run): the 3-step rollout backprops through 3 forwards -> ~3x activation memory. Raise if VRAM allows.')
    p.add_argument('--lr_g', type=float, default=1e-4, help='<= lr_d so G responds to adv; watch flow loss does not blow up')
    p.add_argument('--lr_d', type=float, default=4e-4, help='faster D so it learns the (now NFE=3) real-vs-fake signal')
    p.add_argument('--adv_weight', type=float, default=0.5,
                   help='beta in L_flow + beta*L_adv. You have MAE/SSIM margin vs DDIM@30 -> can raise (0.7-1.0) to push LPIPS harder.')
    p.add_argument('--r1_gamma', type=float, default=0.01, help='R1 strength (the only D regulariser)')
    p.add_argument('--adv_nfe', type=int, default=3, help='3 = adversarial on the TRUE 3-step generation (this script). The NFE=3 inference object.')
    p.add_argument('--adv_start_step', type=int, default=0, help='warmup: flow-only steps before turning GAN on')
    p.add_argument('--save_every', type=int, default=1,
                   help='1 = every epoch (GAN quality is non-monotone; checkpoint finely to catch the best epoch).')
    return p.parse_args()


def main():
    args = get_args()
    print('data base:', _BASE, '| trial:', args.trial_name, '| adv_nfe:', args.adv_nfe)

    bins = np.load(os.path.join(_BASE, 'file/histogram_equalization/bins.npy')) if HE else None
    binsm = np.load(os.path.join(_BASE, 'file/histogram_equalization/bins_mapped.npy')) if HE else None

    xlsx = os.path.join(_BASE, 'Data/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx')
    bs = Build_list.Build_thinsliceCT(xlsx)
    _, _, _, _, cond_tr, x0_tr = bs.__build__(batch_list=[0, 1, 2, 3])
    n = ff.get_X_numbers_in_interval(total_number=x0_tr.shape[0], start_number=0, end_number=1, interval=2)
    x0_tr, cond_tr = x0_tr[n], cond_tr[n]
    x0_tr = np.array([_remap(p) for p in x0_tr])
    cond_tr = np.array([_remap(p) for p in cond_tr])
    print('train pairs:', x0_tr.shape[0])

    gen_tr = Generator.Dataset_2D(
        supervision='unsupervised', img_list=x0_tr, condition_list=cond_tr, image_size=IMG,
        num_slices_per_image=50, random_pick_slice=True, slice_range=None,
        num_patches_per_slice=2, patch_size=PATCH,
        histogram_equalization=HE, bins=bins, bins_mapped=binsm,
        background_cutoff=BG, maximum_cutoff=MX, normalize_factor=NF,
        shuffle=True, augment=True, augment_frequency=0.5)

    # generator: same arch as model-200 (v-head, 2-channel condition)
    base = Unet(problem_dimension='2D', init_dim=64, out_dim=1, channels=1,
                conditional_diffusion=True, condition_channels=COND_CH,
                downsample_list=(True, True, True, False), upsample_list=(True, True, True, False),
                full_attn=(None, None, False, True), auxiliary_v_head=True)
    G = imf.ImprovedMeanFlow(base, image_size=PATCH, ratio_r_neq_t=0.5, clip_or_not=False, auto_normalize=False)

    # UNCONDITIONAL high-pass PatchGAN D (same as the NFE=1 run)
    D = PatchDiscriminator(img_channels=1, cond_channels=0, base=64, n_layers=3)

    save_models = os.path.join(_BASE, 'projects/denoising/models', args.trial_name, 'models')
    ff.make_folder([os.path.dirname(os.path.dirname(save_models)), os.path.dirname(save_models), save_models])

    trainer = GANTrainer(
        diffusion_model=G, discriminator=D, generator_train=gen_tr,
        train_batch_size=args.batch_size, train_num_steps=args.train_num_steps,
        results_folder=save_models, lr_g=args.lr_g, lr_d=args.lr_d,
        adv_weight=args.adv_weight, r1_gamma=args.r1_gamma, adv_start_step=args.adv_start_step,
        adv_nfe=args.adv_nfe, save_every=args.save_every)

    if args.pretrained and os.path.isfile(args.pretrained):
        trainer.load_generator(args.pretrained, key='model')
    else:
        print('[GAN] WARNING: no pretrained generator loaded — training GAN from scratch is unstable.', flush=True)

    trainer.train()


if __name__ == '__main__':
    main()
