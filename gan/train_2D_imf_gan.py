"""
train_2D_imf_gan.py — GAN fine-tuning of the iMF generator on brain CT (N2N, gaussian sim).

Two-stage recipe: load your FLOW-pretrained model-200, then fine-tune it with a small
adversarial loss (L_flow + beta * L_adv). The discriminator pushes the model's one-step
prediction to match the real (noisy) target distribution -> more faithful few-step samples.
Inference is unchanged (discriminator discarded). New trial dir; touches nothing existing.

Run inside docker (tmux):
    python gan/train_2D_imf_gan.py --pretrained /host/d/research/projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt
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
    p = argparse.ArgumentParser('iMF GAN fine-tuning (brain CT)')
    p.add_argument('--trial_name', type=str, default='imf_gan_unsupervised_gaussian_brainCT')
    p.add_argument('--pretrained', type=str,
                   default=os.path.join(_BASE, 'projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt'),
                   help='flow-pretrained generator checkpoint to fine-tune (loads its "model" weights)')
    p.add_argument('--train_num_steps', type=int, default=50, help='epochs of GAN fine-tuning')
    p.add_argument('--batch_size', type=int, default=4, help='4 (was 2): bigger batch -> less noisy D gradient; raise if VRAM allows (3-step rollout backprop is memory-heavy)')
    p.add_argument('--lr_g', type=float, default=1e-4, help='normal GAN-finetune lr (<= lr_d) so G actually responds to adv; watch flow loss does not blow up')
    p.add_argument('--lr_d', type=float, default=4e-4, help='4e-4 (was 2e-4): faster D so it actually learns the subtle signal')
    p.add_argument('--adv_weight', type=float, default=0.5, help='beta in L_flow + beta*L_adv; 0.5 keeps flow dominant (~0.7) while adv has real influence')
    p.add_argument('--r1_gamma', type=float, default=0.01, help='0.01 (was 0.1): R1 was over-regularizing a struggling D; with the 0.5 fix this is ~20x lighter than the original')
    p.add_argument('--adv_nfe', type=int, default=1, help='adversarial fake = a differentiable adv_nfe-step generation. 1 = single-step F(v) (this NFE=1 experiment). For NFE=3-direct, use train_2D_imf_gan_nfe3.py.')
    p.add_argument('--adv_start_step', type=int, default=0, help='warmup: flow-only steps before turning GAN on')
    p.add_argument('--save_every', type=int, default=1,
                   help='1 = every epoch. GAN quality is non-monotone (can peak then degrade), so '
                        'checkpoint finely to catch the best epoch (each ckpt ~0.58 GB; prune bad ones).')
    return p.parse_args()


def main():
    args = get_args()
    print('data base:', _BASE, '| trial:', args.trial_name)

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

    # UNCONDITIONAL D (cond_channels=0): the real-vs-fake signal here is "noisy x2 vs smooth
    # generation", which lives in the image alone -> dropping the condition removes its dilution
    # of that subtle (~8% amplitude) noise-texture signal. Set cond_channels=COND_CH to go back.
    D = PatchDiscriminator(img_channels=1, cond_channels=0, base=64, n_layers=3)

    save_models = os.path.join(_BASE, 'projects/denoising/models', args.trial_name, 'models')
    ff.make_folder([os.path.dirname(os.path.dirname(save_models)), os.path.dirname(save_models), save_models])

    # full-slice F(v) probe: one fixed slice (FS_SLICE) of case 0 -> the trainer dumps a COMPLETE
    # 512x512 slice each epoch (fv_fullslice_epoch*.npy in fv_evolution/), not just the 128 patch.
    FS_SLICE = 25
    fs_probe = None
    try:
        gen_fs = Generator.Dataset_2D(
            supervision='unsupervised', img_list=x0_tr[:1], condition_list=cond_tr[:1], image_size=IMG,
            num_slices_per_image=1, random_pick_slice=False, slice_range=[FS_SLICE, FS_SLICE + 1],
            num_patches_per_slice=None, patch_size=None,
            histogram_equalization=HE, bins=bins, bins_mapped=binsm,
            background_cutoff=BG, maximum_cutoff=MX, normalize_factor=NF,
            shuffle=False, augment=False)
        fs_real, fs_cond = gen_fs[0]
        fs_probe = (fs_real.unsqueeze(0), fs_cond.unsqueeze(0))   # (1,1,512,512), (1,2,512,512)
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
        trainer.load_generator(args.pretrained, key='model')
    else:
        print('[GAN] WARNING: no pretrained generator loaded — training GAN from scratch is unstable.', flush=True)

    trainer.train()


if __name__ == '__main__':
    main()
