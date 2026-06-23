"""
fuse_imf_v2.py — Variance-gated detail fusion (training-free post-processing) for iMF v2.

Idea (the "averaging method" discussed early on): plain K-averaging removes noise but over-smooths,
hurting LPIPS. Instead:
    structure  <- the K-average  (clean, low distortion)
    detail     <- high-freq of a single sample, injected back
    gate       <- exp(-std^2 / 2 tau^2), so detail is added only where the K samples AGREE
                  (low std = real consistent structure) and suppressed where they disagree (noise).
    fused = base + gamma * w * (sample - lowpass(sample))

Compares: avg10, avg20, and fused(base=avg20) for several gamma. Metrics on [vmin,vmax] HU vs GT.
Pure numpy/scipy + lpips; runs on CPU. Reads the files written by predict_2D_imf_v2.py.

Example (host):  python fuse_imf_v2.py --root D:/research/projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/pred_images_nfe3
Example (docker): python Thinslice_experiments/fuse_imf_v2.py --root /host/d/research/projects/denoising/models/imf_v2_unsupervised_gaussian_brainCT/pred_images_nfe3
"""
import os
import glob
import argparse
import numpy as np
import nibabel as nb
import torch
import lpips
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity


def load(p):
    return np.asarray(nb.load(p).get_fdata(), dtype=np.float32) if os.path.isfile(p) else None


def calc_mae(img, ref, lo, hi):
    out = []
    for s in range(img.shape[-1]):
        m = ((ref[:, :, s] >= lo) & (ref[:, :, s] <= hi)).astype(np.float32)
        if m.sum() > 0:
            out.append(float((np.abs(img[:, :, s] - ref[:, :, s]) * m).sum() / m.sum()))
    return float(np.mean(out)) if out else np.nan


def calc_ssim(img, ref, lo, hi):
    out = []
    for s in range(img.shape[-1]):
        m = ((ref[:, :, s] >= lo) & (ref[:, :, s] <= hi)).astype(np.float32)
        if m.sum() == 0:
            continue
        _, smap = structural_similarity(img[:, :, s], ref[:, :, s], data_range=hi - lo, full=True)
        out.append(float((smap * m).sum() / m.sum()))
    return float(np.mean(out)) if out else np.nan


def calc_lpips(img, ref, lo, hi, fn, dev):
    if fn is None:
        return np.nan
    out = []
    for s in range(img.shape[-1]):
        a = np.clip(img[:, :, s], lo, hi); b = np.clip(ref[:, :, s], lo, hi)
        a = (a - lo) / (hi - lo) * 2 - 1; b = (b - lo) / (hi - lo) * 2 - 1
        ta = torch.from_numpy(np.stack([a, a, a])[None].astype(np.float32)).to(dev)
        tb = torch.from_numpy(np.stack([b, b, b])[None].astype(np.float32)).to(dev)
        with torch.no_grad():
            out.append(float(fn(ta, tb).item()))
    return float(np.mean(out)) if out else np.nan


def fuse(base, sample, std, gamma, tau, lp_sigma, gated=True):
    low = ndi.gaussian_filter(sample, sigma=(lp_sigma, lp_sigma, 0.0))
    detail = sample - low
    w = np.exp(-(std ** 2) / (2.0 * tau ** 2)) if gated else 1.0
    return base + gamma * w * detail


def main():
    ap = argparse.ArgumentParser('variance-gated detail fusion + eval')
    ap.add_argument('--root', required=True, help='pred_images_nfeN directory')
    ap.add_argument('--gamma', type=float, nargs='+', default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument('--tau_pct', type=float, default=50.0, help='percentile of in-tissue std used as gate scale tau')
    ap.add_argument('--lp_sigma', type=float, default=1.5, help='gaussian sigma (px) for hi/lo split')
    ap.add_argument('--no_gate', action='store_true', help='disable variance gate (global gamma blend)')
    ap.add_argument('--no_lpips', action='store_true', help='skip LPIPS (e.g. if alexnet weights cannot be downloaded)')
    ap.add_argument('--vmin', type=float, default=0.0)
    ap.add_argument('--vmax', type=float, default=100.0)
    ap.add_argument('--cases', type=int, default=0, help='limit number of cases (0=all)')
    ap.add_argument('--slice_step', type=int, default=1, help='subsample slices for a quick run')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn = None
    if not args.no_lpips:
        try:
            fn = lpips.LPIPS(net='alex').to(dev)
        except Exception as e:
            print(f'[warn] LPIPS unavailable ({str(e)[:60]}) -> reporting MAE/SSIM only', flush=True)
            fn = None
    avg_dirs = sorted(glob.glob(os.path.join(args.root, '*', '*', 'random_*', 'epoch*avg')))
    if args.cases:
        avg_dirs = avg_dirs[:args.cases]
    print(f'gate={"OFF" if args.no_gate else "ON"} lp_sigma={args.lp_sigma} tau_pct={args.tau_pct} | {len(avg_dirs)} cases', flush=True)

    rows = {}  # method -> list of (mae,ssim,lpips)
    def add(method, t):
        rows.setdefault(method, []).append(t)

    for d in avg_dirs:
        case_root = os.path.dirname(d)
        gt = load(os.path.join(d, 'gt_img.nii.gz'))
        if gt is None:
            gt = load(os.path.join(case_root, 'epoch200_1', 'gt_img.nii.gz'))
        a20 = load(os.path.join(d, 'pred_img_scans20.nii.gz'))
        a10 = load(os.path.join(d, 'pred_img_scans10.nii.gz'))
        std = None
        sp = os.path.join(d, 'sample_std.npy')
        if os.path.isfile(sp):
            std = np.load(sp).astype(np.float32)
        samp = load(os.path.join(case_root, 'epoch200_1', 'pred_img.nii.gz'))
        if gt is None or a20 is None or std is None or samp is None:
            print('  [skip] missing inputs in', d); continue

        if args.slice_step > 1:
            sl = slice(None, None, args.slice_step)
            gt, a20, std, samp = gt[:, :, sl], a20[:, :, sl], std[:, :, sl], samp[:, :, sl]
            a10 = a10[:, :, sl] if a10 is not None else None

        mask = (gt >= args.vmin) & (gt <= args.vmax)
        tau = float(np.percentile(std[mask], args.tau_pct)) if mask.sum() else float(np.percentile(std, args.tau_pct))
        tau = max(tau, 1e-3)

        m = lambda x: (calc_mae(x, gt, args.vmin, args.vmax), calc_ssim(x, gt, args.vmin, args.vmax), calc_lpips(x, gt, args.vmin, args.vmax, fn, dev))
        if a10 is not None:
            add('avg10', m(a10))
        add('avg20', m(a20))
        for g in args.gamma:
            fused = fuse(a20, samp, std, g, tau, args.lp_sigma, gated=not args.no_gate)
            add(f'fuse20_g{g}', m(fused))
        print('  done', '/'.join(os.path.normpath(case_root).split(os.sep)[-3:]), f'(tau={tau:.2f})', flush=True)

    print('\n================ FUSION SUMMARY (window [%g,%g] HU, mean over cases) ================' % (args.vmin, args.vmax))
    print(f"{'method':>14} | {'MAE(down)':>10} | {'SSIM(up)':>10} | {'LPIPS(down)':>11}")
    order = ['avg10', 'avg20'] + [f'fuse20_g{g}' for g in args.gamma]
    for meth in order:
        if meth not in rows:
            continue
        arr = np.array(rows[meth])
        mae, ssim, lp = np.nanmean(arr, axis=0)
        print(f"{meth:>14} | {mae:>10.3f} | {ssim:>10.3f} | {lp:>11.4f}")


if __name__ == '__main__':
    main()
