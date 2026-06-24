"""
view_nfe1_fullslice.py — FULL-slice (not patch) version of the probe comparison, read from the
existing pred_images_nfe1 folder (model-200, NFE=1 one-step inference). Shows, on a real 512x512
brain slice:

    row 1:  noisy x2          |  NFE=1 one-step output  |  clean GT
    row 2:  high-pass(x2)     |  high-pass(NFE=1)        |  NFE=1 - x2

The high-pass row strips anatomy and shows ONLY the high-frequency content, so you can judge
whether the NFE=1 roughness is x2-like noise or model-error junk (e.g. grain in the air/background,
where x2 has none, is junk). This is the model-200 starting point (before any GAN fine-tuning).

    python gan/view_nfe1_fullslice.py                 # middle slice, brain window [0,100] HU
    python gan/view_nfe1_fullslice.py --slice 12 --vmin 0 --vmax 80

You can ALSO just open condition_img.nii.gz / pred_img.nii.gz / gt_img.nii.gz from that folder
directly in ITK-SNAP -- this script only adds the high-pass (noise-texture) view on top.
"""
import os
import glob
import argparse
import numpy as np
import nibabel as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _detect():
    trial = 'imf_v2_unsupervised_gaussian_brainCT'
    for b in ('/host/d/research', '/d/research', '/host/d'):
        p = os.path.join(b, 'projects/denoising/models', trial, 'pred_images_nfe1')
        if os.path.isdir(p):
            return p
    return '.'


def load(p):
    return np.asarray(nb.load(p).get_fdata(), dtype=np.float32) if os.path.isfile(p) else None


def highpass(img, k=7):
    try:
        from scipy.ndimage import uniform_filter
        return img - uniform_filter(img, size=k, mode='reflect')
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser('full-slice NFE=1 vs x2 comparison')
    ap.add_argument('--folder', default=_detect(), help='pred_images_nfe1 folder')
    ap.add_argument('--slice', type=int, default=None, help='slice index (default: middle)')
    ap.add_argument('--vmin', type=float, default=0.0)
    ap.add_argument('--vmax', type=float, default=100.0)
    ap.add_argument('--k', type=int, default=7, help='high-pass window (match D hp_kernel)')
    ap.add_argument('--rot', type=int, default=0, help='rot90 k times for display orientation')
    args = ap.parse_args()

    e1 = glob.glob(os.path.join(args.folder, '*', '*', 'random_*', 'epoch*_1'))
    if not e1:
        raise SystemExit(f'no epoch*_1 case under {args.folder}')
    d = sorted(e1)[0]
    x2 = load(os.path.join(d, 'condition_img.nii.gz'))
    fake = load(os.path.join(d, 'pred_img.nii.gz'))
    gt = load(os.path.join(d, 'gt_img.nii.gz'))
    if x2 is None or fake is None:
        raise SystemExit('missing condition_img / pred_img in ' + d)
    if x2.ndim == 4:
        x2 = x2[..., 0]

    S = fake.shape[-1]
    s = S // 2 if args.slice is None else max(0, min(S - 1, args.slice))
    X, Fk = x2[:, :, s], fake[:, :, s]
    G = gt[:, :, s] if gt is not None else None

    def orient(im):
        return np.rot90(im, args.rot) if args.rot else im

    hX, hF = highpass(X, args.k), highpass(Fk, args.k)
    sx = float(np.std(X[1:, :] - X[:-1, :]))
    sf = float(np.std(Fk[1:, :] - Fk[:-1, :]))

    panels = [(f'noisy x2  (hf {sx:.3f})', X, 'gray', args.vmin, args.vmax),
              (f'NFE=1 output  (hf {sf:.3f})', Fk, 'gray', args.vmin, args.vmax)]
    if G is not None:
        panels.append(('clean GT', G, 'gray', args.vmin, args.vmax))
    if hX is not None:
        hl = max(np.percentile(np.abs(hX), 99), np.percentile(np.abs(hF), 99), 1e-6)
        panels.append(('high-pass(x2) = noise', hX, 'gray', -hl, hl))
        panels.append(('high-pass(NFE=1)', hF, 'gray', -hl, hl))
        dd = Fk - X
        dl = max(np.percentile(np.abs(dd), 99), 1e-6)
        panels.append(('NFE=1 - x2', dd, 'RdBu_r', -dl, dl))

    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, (t, im, cm, lo, hi) in zip(axes, panels):
        ax.imshow(orient(im), cmap=cm, vmin=lo, vmax=hi)
        ax.set_title(t, fontsize=10)

    out = os.path.join(args.folder, f'nfe1_fullslice_s{s}.png')
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'saved {out}  | slice {s} of {S} | hf: x2={sx:.4f} NFE1={sf:.4f} ({sf / max(sx,1e-9):.2f}x)')


if __name__ == '__main__':
    main()
