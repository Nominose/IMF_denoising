"""
view_fv_evolution.py — montage of the per-epoch one-step F(v) dumps written by imf_gan.py.

Run after (or during) GAN training to SEE whether F(v) drifts from the smooth posterior-mean
toward the noisy x2 as the adversarial loss kicks in:

    python gan/view_fv_evolution.py                      # auto-detects the fv_evolution folder
    python gan/view_fv_evolution.py --folder <path> --cols 6

Writes <folder>/evolution.png : first panel = real x2 (target), then F(v) at each epoch, all on a
common grayscale window so the change is comparable. (.npy isn't viewable in ITK-SNAP; this is the
easy way to eyeball the drift. Needs matplotlib — present in the docker env.)
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _digits(name):
    d = ''.join(ch for ch in os.path.basename(name) if ch.isdigit())
    return int(d) if d else 0


def _detect_default():
    trial = 'imf_gan_unsupervised_gaussian_brainCT'
    for b in ('/host/d/research', '/d/research', '/host/d'):
        p = os.path.join(b, 'projects/denoising/models', trial, 'models', 'fv_evolution')
        if os.path.isdir(p):
            return p
    return 'fv_evolution'


def main():
    ap = argparse.ArgumentParser('montage of per-epoch F(v) dumps')
    ap.add_argument('--folder', default=_detect_default(), help='the fv_evolution folder')
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--vmin', type=float, default=None, help='grayscale low (default: 1st pct of real)')
    ap.add_argument('--vmax', type=float, default=None, help='grayscale high (default: 99th pct of real)')
    ap.add_argument('--fullslice', action='store_true',
                    help='montage the FULL-SLICE dumps (fv_fullslice_epoch*.npy) instead of the 128 patch ones')
    args = ap.parse_args()

    stub = 'fv_fullslice_epoch' if args.fullslice else 'fv_epoch'
    realname = 'real_x2_fullslice.npy' if args.fullslice else 'real_x2.npy'
    fvs = sorted(glob.glob(os.path.join(args.folder, stub + '*.npy')), key=_digits)
    if not fvs:
        raise SystemExit(f'no {stub}*.npy under {args.folder}')

    panels = []
    real_p = os.path.join(args.folder, realname)
    if os.path.isfile(real_p):
        panels.append(('real x2 (target)', np.load(real_p)))
    for p in fvs:
        panels.append((f'epoch {_digits(p)}', np.load(p)))

    ref = panels[0][1]
    vmin = args.vmin if args.vmin is not None else float(np.percentile(ref, 1))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(ref, 99))

    n = len(panels)
    cols = max(1, args.cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)

    out = os.path.join(args.folder, 'evolution_fullslice.png' if args.fullslice else 'evolution.png')
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'saved {out}  ({n} panels, window [{vmin:.3f}, {vmax:.3f}])')


if __name__ == '__main__':
    main()
