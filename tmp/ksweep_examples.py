r"""Example images for the K ablation (layout of Fig. 3B of the Noise2Noise-diffusion paper): the
same slice as noisy input (FBP) and as the average of K = 1, 2, 4, 8, 20 generated outputs, each
with a magnified region (yellow box on the panel, inset in the upper-right corner).

Display follows tmp/image_showcase.py: crop around the centre, orient with np.flip(img.T, 0), brain
window WL 40 / WW 80 through functions_collection.set_window.

    python tmp/ksweep_examples.py --patient 00214841 --slice 25 --zoom 150,230,36
    python tmp/ksweep_examples.py --patient 00214841 --slice 25 --grid      # coordinate grid to pick the ROI
"""
import argparse
import glob
import os
import sys

import numpy as np
import nibabel as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from IMF_denoising.functions_collection.functions import set_window

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'D:\research\projects\denoising\models\ksweep'
GTC = r'D:\research\projects\denoising\results\brianct\gt_and_conditions'
TRIALS = {'gan': ('imf_gan_unsupervised_gaussian_brainCT', 28), 'flow_only': ('imf_v2_unsupervised_gaussian_brainCT', 200)}


def panel(vol, sl, center, half):
    a = vol[center - half:center + half, center - half:center + half, sl]
    return set_window(np.flip(a.T, 0), 40, 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient', default='00214841')
    ap.add_argument('--slice', type=int, default=25)
    ap.add_argument('--model', choices=list(TRIALS), default='gan')
    ap.add_argument('--nfe', type=int, default=3)
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 4, 8, 20])
    ap.add_argument('--zoom', default=None, help='x,y,half of the magnified region in panel pixels')
    ap.add_argument('--center', type=int, default=256); ap.add_argument('--half', type=int, default=200)
    ap.add_argument('--grid', action='store_true', help='overlay a coordinate grid instead of the inset')
    ap.add_argument('--with_gt', action='store_true', help='append the clean reference as a last panel')
    ap.add_argument('--out', default=r'D:\research\projects\denoising\results\paper_revision')
    a = ap.parse_args()
    trial, ep = TRIALS[a.model]
    avg = glob.glob(os.path.join(ROOT, trial, f'pred_images_nfe{a.nfe}', a.patient, '*', 'random_*', f'epoch{ep}avg'))[0]
    ld = lambda p: np.asarray(nb.load(p).dataobj, np.float32)
    cond = ld(glob.glob(os.path.join(GTC, a.patient, '**', 'condition_img.nii.gz'), recursive=True)[0])
    items = [('FBP', cond)] + [(f'K={k}', ld(os.path.join(avg, f'pred_img_scans{k}.nii.gz'))) for k in a.ks]
    if a.with_gt:
        items.append(('Clean reference', ld(glob.glob(os.path.join(GTC, a.patient, '**', 'gt_img.nii.gz'), recursive=True)[0])))
    imgs = [(t, panel(v, a.slice, a.center, a.half)) for t, v in items]
    n = len(imgs); cols = 3 if n <= 6 else 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.25), facecolor='black')
    zoom = tuple(int(v) for v in a.zoom.split(',')) if a.zoom else None
    for ax in np.atleast_1d(axes).flat:
        ax.axis('off')
    for ax, (t, im) in zip(np.atleast_1d(axes).flat, imgs):
        ax.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(t, color='white', fontsize=13, fontweight='bold', pad=4)
        if a.grid:
            for g in range(0, im.shape[0], 50):
                ax.axhline(g, color='cyan', lw=0.4, alpha=0.7); ax.axvline(g, color='cyan', lw=0.4, alpha=0.7)
                ax.text(g + 2, 12, str(g), color='cyan', fontsize=6); ax.text(2, g + 12, str(g), color='cyan', fontsize=6)
        elif zoom:
            zx, zy, zh = zoom
            ax.add_patch(Rectangle((zx - zh, zy - zh), 2 * zh, 2 * zh, fill=False, ec='yellow', lw=1.2))
            ins = ax.inset_axes([0.60, 0.60, 0.40, 0.40])
            ins.imshow(im[zy - zh:zy + zh, zx - zh:zx + zh], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ins.set_xticks([]); ins.set_yticks([])
            for sp in ins.spines.values():
                sp.set_edgecolor('yellow'); sp.set_linewidth(1.2)
    plt.subplots_adjust(0.005, 0.005, 0.995, 0.955, 0.02, 0.08)
    base = os.path.join(a.out, f'ksweep_examples_{a.model}_{a.patient}_s{a.slice}' + ('_grid' if a.grid else ''))
    for ext in (('png',) if a.grid else ('png', 'pdf')):
        fig.savefig(f'{base}.{ext}', dpi=200 if a.grid else 300, facecolor='black')
    print('wrote', base)


if __name__ == '__main__':
    main()
