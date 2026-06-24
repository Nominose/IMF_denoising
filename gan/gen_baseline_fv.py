"""
gen_baseline_fv.py — generate the BASELINE (model-200, no GAN) full-slice F(v) and drop it into a
run's fv_evolution/ as `fv_fullslice_epoch0.npy`, WITHOUT restarting training.

Use when a GAN run was started before the epoch-0 baseline dump existed, but you want the
"original NFE" reference frame for the evolution montage. After running this, the montage shows
baseline (no GAN) -> ep1 -> ep2 -> ..., so the divergence from frame 0 = what the GAN did.

NOTE: the noise seed here differs from the run's own fixed probe noise (that wasn't saved), so this
baseline is a TEXTURE/QUALITY-level reference for the same slice, not pixel-aligned to ep1+. For a
pixel-aligned baseline, restart the run instead (the trainer now dumps epoch 0 automatically).

    python gan/gen_baseline_fv.py --trial_name imf_gan_nfe3_unsupervised_gaussian_brainCT --adv_nfe 3
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
import torch

import IMF_denoising.improved_mean_flow as imf
import IMF_denoising.functions_collection as ff
import IMF_denoising.Build_lists.Build_list as Build_list
import IMF_denoising.Generator_thinslice as Generator
from IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion import Unet


def _detect_base():
    for b in ('/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'Data')):
            return b
    return '/host/d/research'


_BASE = _detect_base()


def _remap(p):
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
    p = argparse.ArgumentParser('generate baseline (no-GAN) full-slice F(v) as epoch 0')
    p.add_argument('--trial_name', default='imf_gan_nfe3_unsupervised_gaussian_brainCT',
                   help='the GAN run folder to drop fv_fullslice_epoch0.npy into')
    p.add_argument('--pretrained', default=os.path.join(_BASE, 'projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/models/model-200.pt'),
                   help='the baseline checkpoint (no GAN)')
    p.add_argument('--adv_nfe', type=int, default=3, help='NFE of the generation (match the run: 3)')
    p.add_argument('--fs_slice', type=int, default=25, help='which slice (match the run; default 25)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    bins = np.load(os.path.join(_BASE, 'file/histogram_equalization/bins.npy'))
    binsm = np.load(os.path.join(_BASE, 'file/histogram_equalization/bins_mapped.npy'))

    xlsx = os.path.join(_BASE, 'Data/brain_CT/Patient_lists/fixedCT_static_simulation_train_test_gaussian_xjtlu.xlsx')
    bs = Build_list.Build_thinsliceCT(xlsx)
    _, _, _, _, cond_tr, x0_tr = bs.__build__(batch_list=[0, 1, 2, 3])
    n = ff.get_X_numbers_in_interval(total_number=x0_tr.shape[0], start_number=0, end_number=1, interval=2)
    x0_tr, cond_tr = x0_tr[n], cond_tr[n]
    x0_tr = np.array([_remap(p) for p in x0_tr])
    cond_tr = np.array([_remap(p) for p in cond_tr])

    # SAME full-slice probe as the trainer: slice fs_slice of case 0, normalized
    gen_fs = Generator.Dataset_2D(
        supervision='unsupervised', img_list=x0_tr[:1], condition_list=cond_tr[:1], image_size=IMG,
        num_slices_per_image=1, random_pick_slice=False, slice_range=[args.fs_slice, args.fs_slice + 1],
        num_patches_per_slice=None, patch_size=None,
        histogram_equalization=HE, bins=bins, bins_mapped=binsm,
        background_cutoff=BG, maximum_cutoff=MX, normalize_factor=NF,
        shuffle=False, augment=False)
    fs_real, fs_cond = gen_fs[0]
    fs_real = fs_real.unsqueeze(0).to(dev)
    fs_cond = fs_cond.unsqueeze(0).to(dev)

    # baseline generator (same arch as the GAN's G), load the 'model' weights (matches trainer dump)
    base = Unet(problem_dimension='2D', init_dim=64, out_dim=1, channels=1,
                conditional_diffusion=True, condition_channels=COND_CH,
                downsample_list=(True, True, True, False), upsample_list=(True, True, True, False),
                full_attn=(None, None, False, True), auxiliary_v_head=True)
    G = imf.ImprovedMeanFlow(base, image_size=PATCH, ratio_r_neq_t=0.5, clip_or_not=False, auto_normalize=False).to(dev)
    G.load_state_dict(torch.load(args.pretrained, map_location=dev)['model'])
    G.eval()
    print('loaded baseline:', args.pretrained)

    @torch.no_grad()
    def rollout(z, cond, nfe):
        b = z.shape[0]
        ts = torch.linspace(1.0, 0.0, nfe + 1, device=dev)
        for i in range(nfe):
            dt = (ts[i] - ts[i + 1]).item()
            t_b = torch.full((b,), ts[i].item(), device=dev)
            r_b = torch.full((b,), ts[i + 1].item(), device=dev)
            z = z - dt * G._fn_u(z, r_b, t_b, cond)
        return z

    x0 = rollout(torch.randn_like(fs_real), fs_cond, args.adv_nfe)
    outdir = os.path.join(_BASE, 'projects/denoising/models', args.trial_name, 'models', 'fv_evolution')
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'fv_fullslice_epoch0.npy'), x0[0, 0].cpu().numpy())
    real_p = os.path.join(outdir, 'real_x2_fullslice.npy')
    if not os.path.isfile(real_p):
        np.save(real_p, fs_real[0, 0].cpu().numpy())
    hf = lambda t: float((t[1:, :] - t[:-1, :]).std())
    print(f'baseline NFE={args.adv_nfe} hf {hf(x0[0,0].cpu().numpy()):.4f}  vs real {hf(fs_real[0,0].cpu().numpy()):.4f}')
    print('saved baseline epoch-0 ->', os.path.join(outdir, 'fv_fullslice_epoch0.npy'))


if __name__ == '__main__':
    main()
