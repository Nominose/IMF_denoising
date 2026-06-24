"""
view_probe.py — compare probe_real.npy (real x2) vs probe_fake.npy (the NFE=1 one-step output)
written by imf_gan.py's sanity probe. Saves probe_compare.png:

    row 1:  real x2  |  fake NFE=1  |  (fake - real)
    row 2:  high-pass(real)  |  high-pass(fake)        <- the NOISE TEXTURE of each, directly

The high-pass row is the point: it strips the anatomy and shows ONLY the high-frequency content,
so you can judge whether the fake's roughness is x2-LIKE noise (similar fine grain -> good, the
adversarial target makes sense) or model-error JUNK (blotchy / structured-different / much stronger
-> the NFE=1 roughness is reconstruction error, not real noise).

    python gan/view_probe.py        # auto-detects the models folder; writes probe_compare.png

(.npy can't be opened in ITK-SNAP; this is the easy way. Needs matplotlib; scipy optional.)
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _detect():
    trial = 'imf_gan_unsupervised_gaussian_brainCT'
    for b in ('/host/d/research', '/d/research', '/host/d'):
        p = os.path.join(b, 'projects/denoising/models', trial, 'models')
        if os.path.isfile(os.path.join(p, 'probe_real.npy')):
            return p
    return '.'


def highpass(img, k=7):
    """img - local mean (matches the discriminator's avg_pool front-end). scipy if present."""
    try:
        from scipy.ndimage import uniform_filter
        return img - uniform_filter(img, size=k, mode='reflect')
    except Exception:
        return None


def hf_std(x):
    return float(np.std(x[1:, :] - x[:-1, :]))   # vertical-diff std = the probe's hf proxy


def main():
    ap = argparse.ArgumentParser('compare probe_real vs probe_fake')
    ap.add_argument('--folder', default=_detect())
    ap.add_argument('--k', type=int, default=7, help='high-pass window (match D hp_kernel)')
    args = ap.parse_args()

    real = np.load(os.path.join(args.folder, 'probe_real.npy'))
    fake = np.load(os.path.join(args.folder, 'probe_fake.npy'))
    sr, sf = hf_std(real), hf_std(fake)

    vmin, vmax = np.percentile(real, 1), np.percentile(real, 99)
    hr, hk = highpass(real, args.k), highpass(fake, args.k)
    have_hp = hr is not None

    nrow = 2 if have_hp else 1
    fig, axes = plt.subplots(nrow, 3, figsize=(9, 3 * nrow))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(real, cmap='gray', vmin=vmin, vmax=vmax); axes[0, 0].set_title(f'real x2   (hf {sr:.3f})')
    axes[0, 1].imshow(fake, cmap='gray', vmin=vmin, vmax=vmax); axes[0, 1].set_title(f'fake NFE=1   (hf {sf:.3f})')
    d = fake - real
    dl = max(np.percentile(np.abs(d), 99), 1e-6)
    axes[0, 2].imshow(d, cmap='RdBu_r', vmin=-dl, vmax=dl); axes[0, 2].set_title('fake - real')

    if have_hp:
        hl = max(np.percentile(np.abs(hr), 99), np.percentile(np.abs(hk), 99), 1e-6)
        axes[1, 0].imshow(hr, cmap='gray', vmin=-hl, vmax=hl); axes[1, 0].set_title('high-pass(real) = x2 noise')
        axes[1, 1].imshow(hk, cmap='gray', vmin=-hl, vmax=hl); axes[1, 1].set_title('high-pass(fake) = NFE=1 detail')
        axes[1, 2].axis('off')

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    out = os.path.join(args.folder, 'probe_compare.png')
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'saved {out}')
    print(f'hf std:  real={sr:.4f}  fake={sf:.4f}  (fake/real = {sf / max(sr, 1e-9):.2f}x)')
    print('Look at the high-pass row: fake grain ~ real grain -> x2-like noise (good);')
    print('fake blotchy / different / much stronger -> NFE=1 roughness is model error, not x2 noise.')


if __name__ == '__main__':
    main()
