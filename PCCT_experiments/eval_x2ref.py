r"""Ground-truth-free check of the PCCT results with the x2-reference identity.

PCCT has no clean reference, so the tables report only a CNR from hand-drawn ROIs. For any method
whose output for slice j is computed WITHOUT looking at slice j, the noisy slice itself is an
independent observation of the same anatomy, x_j = z_j + n_j, and

    R_obs = E|| pred_j - x_j ||^2  =  E|| pred_j - z_j ||^2  +  E|| n_j ||^2          (*)
          =        true MSE         +  a constant that does not depend on the method

so DIFFERENCES of R_obs between methods equal differences of the true (unobservable) MSE.

Where (*) holds and where it does not
  * Conditional diffusion (DDIM), iMF and iMF+GAN generate slice j from slices j-1 and j+1 only:
    valid.
  * The Noise2Noise baseline and DDM2 take slice j itself as input, so their output carries n_j and
    the cross term does not vanish; R_obs would flatter them. They are listed but marked invalid.
  * The two outer slices of the 50-slice extracts are skipped: the local NFE=5 run had to replicate
    the missing outer neighbour, i.e. it did see the slice it was predicting there.
  * (*) needs the noise of adjacent slices to be independent. That is also the premise of training
    on adjacent-slice pairs; --noise_corr measures it in homogeneous ROIs (PCCT: +0.03 at lag 1,
    against -0.01 +- 0.03 from the same estimator on simulated data with independent noise).
  * On simulated brain CT, where a clean reference exists, the identity was checked directly
    (tmp/x2ref_brain_check.py): R_obs - true MSE is one constant, 63.66-63.73 HU^2, over DDIM, flow
    matching, iMF and iMF+GAN at seven NFE, and R_obs orders all 42 method pairs correctly. EDM comes
    out at 41-48: its output is correlated with the target slice's own noise, i.e. that model does
    see the slice, and R_obs must not be used for it.

The mask is GT-free as well: brain tissue ([0,100] HU) on the Gaussian-smoothed mean of the two
NEIGHBOURING noisy slices, which is independent of n_j, so conditioning on it does not bias (*).
No clipping is applied (clipping is non-linear and would break the identity).

    python PCCT_experiments/eval_x2ref.py
    python PCCT_experiments/eval_x2ref.py --noise_corr --xlsx results/pcct/PCCT_x2ref.xlsx
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import nibabel as nb
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding='utf-8')
R = os.environ.get('RESULTS_ROOT', r'D:\research\projects\denoising\results')
P = os.path.join(R, 'pcct')
CASES = list(range(29, 37))
ld = lambda p: np.asarray(nb.load(p).dataobj, np.float32)


def first(pattern):
    g = sorted(glob.glob(pattern))
    return g[0] if g else None


def methods():
    """-> list of (label, nfe, k, valid, path_of_case)"""
    m = [('DDIM', 100, 8, True, lambda c: first(os.path.join(P, 'DDIM_results', str(c), 'epoch*avg', 'pred_img_scans8.nii.gz')))]
    for kind, label in (('nogan', 'iMF'), ('gan', 'iMF + GAN')):
        dirs = [d for d in glob.glob(os.path.join(P, 'imf', kind, '*')) if os.path.isdir(d) and '_raw' not in d]
        for d in sorted(dirs, key=lambda d: int(re.search(r'nfe(\d+)', d).group(1))):
            nfe = int(re.search(r'nfe(\d+)', d).group(1))
            for k in (10, 20):
                m.append((label, nfe, k, True, (lambda d, k: lambda c: os.path.join(d, str(c), f'pred_img_scans{k}.nii.gz'))(d, k)))
    m.append(('Noise2Noise', 1, '-', False, lambda c: first(os.path.join(P, 'noise2noise_results', str(c), 'epoch*', 'pred_img.nii.gz'))))
    m.append(('DDM2 (first step)', '-', '-', False, lambda c: os.path.join(P, 'DDM2_results', str(c), 'ddm2_firststep_image.nii.gz')))
    return m


def noisy_and_mask(case):
    x = ld(os.path.join(P, 'imf', 'gan', 'imf_gan_unsupervised_PCCT_epoch48_nfe3', str(case), 'condition_img.nii.gz'))
    nbr = 0.5 * (x[:, :, :-2] + x[:, :, 2:])                       # neighbours of slices 1..n-2
    sm = np.stack([gaussian_filter(nbr[:, :, s], 2.0) for s in range(nbr.shape[2])], axis=2)
    return x, ((sm >= 0) & (sm <= 100))                             # mask aligned with slices 1..n-2


def r_obs(pred, x, mask):
    d = (pred[:, :, 1:-1] - x[:, :, 1:-1]) ** 2
    per = [(d[:, :, s][mask[:, :, s]]).mean() for s in range(mask.shape[2]) if mask[:, :, s].sum() > 2000]
    return float(np.mean(per))


def _detrend(v, m):
    yy, xx = np.nonzero(m)
    A = np.c_[xx, yy, np.ones_like(xx)].astype(np.float64); b = v[m].astype(np.float64)
    return b - A @ np.linalg.lstsq(A, b, rcond=None)[0]


def _roi_rho(x, anat, m, s, lag):
    """Correlation of slices s and s+lag over the ROI pixels after removing a plane per slice.
    -> (raw, anatomy-corrected, anatomy share of the variance, noise sd). The correction subtracts the
    covariance and variance of `anat` (a denoised or clean volume) measured the same way."""
    a, b = _detrend(x[:, :, s], m), _detrend(x[:, :, s + lag], m)
    za, zb = _detrend(anat[:, :, s], m), _detrend(anat[:, :, s + lag], m)
    cx, cz = np.mean(a * b), np.mean(za * zb)
    va, vb, vza, vzb = a.var(), b.var(), za.var(), zb.var()
    return (cx / np.sqrt(va * vb), (cx - cz) / np.sqrt(max(va - vza, 1e-9) * max(vb - vzb, 1e-9)),
            (vza + vzb) / (va + vb), np.sqrt(0.5 * (va + vb)))


def noise_corr():
    """Is the noise of adjacent slices independent? Measured inside homogeneous regions, because
    anywhere else the anatomy shared by neighbouring slices swamps the estimate (over all brain tissue
    the raw correlation of high-pass residuals is 0.6-0.8 even where the noise is independent).
    PCCT uses the hand-drawn WM/GM ROIs; the same estimator is run on the simulated brain data, whose
    noise is independent by construction, to show what it returns when the answer is zero."""
    from scipy.ndimage import binary_erosion, label
    print()
    print('adjacent-slice noise correlation inside homogeneous ROIs (plane removed, anatomy-corrected)')
    r1, r2 = [], []
    for c in CASES:
        d = os.path.join(P, 'imf', 'gan', 'imf_gan_unsupervised_PCCT_epoch48_nfe3', str(c))
        x = ld(os.path.join(d, 'condition_img.nii.gz'))
        den = ld(os.path.join(P, 'imf', 'gan', 'imf_gan_unsupervised_PCCT_epoch48_nfe10', str(c), 'pred_img_scans20.nii.gz'))
        for name in ('WM', 'GM'):
            roi = np.round(ld(os.path.join(d, f'{name}_ROI.nii.gz'))).astype(bool)
            a1, a2 = [], []
            for s in sorted(set(np.where(roi)[2].tolist())):
                m = roi[:, :, s]
                if m.sum() < 80 or s < 2 or s > x.shape[2] - 3:
                    continue
                a1 += [_roi_rho(x, den, m, s, 1), _roi_rho(x, den, m, s - 1, 1)]
                a2 += [_roi_rho(x, den, m, s - 1, 2)]            # the two inputs of slice s
            if a1:
                r1.append(np.mean(a1, 0)); r2.append(np.mean(a2, 0))
    r1, r2 = np.array(r1), np.array(r2)
    print(f'  PCCT, {len(r1)} ROIs (8 cases x WM, GM): lag 1  {r1[:, 1].mean():+.3f} +- {r1[:, 1].std(ddof=1) / np.sqrt(len(r1)):.3f} (s.e.)   '
          f'lag 2  {r2[:, 1].mean():+.3f}   | noise sd {r1[:, 3].mean():.1f} HU, anatomy {100 * r1[:, 2].mean():.0f}% of the ROI variance')
    B = os.path.join(R, 'brianct', 'gt_and_conditions')
    o1, o2 = [], []
    for gdir in sorted(glob.glob(os.path.join(B, '*', '*', 'random_0')))[:10]:
        g, x = ld(os.path.join(gdir, 'gt_img.nii.gz')), ld(os.path.join(gdir, 'condition_img.nii.gz'))
        for s in (10, 18, 25, 32, 40):
            gs = gaussian_filter(g[:, :, s], 2.0); gy, gx = np.gradient(gs)
            lab, n = label(binary_erosion((gs > 18) & (gs < 45) & (np.hypot(gx, gy) < 0.6), iterations=2))
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())[1:]; k = int(np.argmax(sizes)) + 1
            yy, xx = np.nonzero(lab == k); cy, cx = int(yy.mean()), int(xx.mean())
            m = (lab == k) & (np.hypot(*np.ogrid[-cy:g.shape[0] - cy, -cx:g.shape[1] - cx]) < 14)
            if m.sum() < 250:
                continue
            o1.append(_roi_rho(x, g, m, s, 1)); o2.append(_roi_rho(x, g, m, s - 1, 2))
    o1, o2 = np.array(o1), np.array(o2)
    print(f'  simulated brain, {len(o1)} flat ROIs (independent by construction): lag 1  {o1[:, 1].mean():+.3f} +- {o1[:, 1].std(ddof=1) / np.sqrt(len(o1)):.3f} (s.e.)   '
          f'lag 2  {o2[:, 1].mean():+.3f}   | noise sd {o1[:, 3].mean():.1f} HU')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise_corr', action='store_true')
    ap.add_argument('--xlsx', default=None)
    a = ap.parse_args()
    ms = methods()
    table = {i: [] for i in range(len(ms))}
    for c in CASES:
        x, mask = noisy_and_mask(c)
        for i, (_, _, _, _, path_of) in enumerate(ms):
            p = path_of(c)
            table[i].append(r_obs(ld(p), x, mask) if p and os.path.isfile(p) else np.nan)
    ref = np.array(table[0])                                          # DDIM
    print('R_obs = masked MSE against the held-out noisy slice (HU^2), 8 PCCT test cases, slices 1-48')
    print('differences between VALID rows equal differences of the true MSE\n')
    print(f"{'method':<19} {'NFE':>4} {'K':>3} | {'R_obs mean':>10} {'std':>7} | {'vs DDIM':>9} {'(%)':>7} | better than DDIM | valid")
    rows = []
    for i, (label, nfe, k, valid, _) in enumerate(ms):
        v = np.array(table[i]); d = v - ref
        wins = int((d < 0).sum())
        rows.append((label, nfe, k, valid, v, d, wins))
        print(f"{label:<19} {str(nfe):>4} {str(k):>3} | {np.nanmean(v):>10.2f} {np.nanstd(v, ddof=1):>7.2f} | {np.nanmean(d):>+9.2f} {100 * np.nanmean(d) / np.nanmean(ref):>+6.1f}% | "
              + (f"{wins}/8" if i else ' - ').center(16) + ' | ' + ('yes' if valid else 'NO (saw the slice)'))
    if a.noise_corr:
        noise_corr()
    if a.xlsx:
        import openpyxl
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = 'x2-reference'
        ws.append(['Method', 'NFE', 'K'] + [f'case{c}' for c in CASES] + ['mean', 'std', 'mean - DDIM', 'cases better than DDIM', 'valid'])
        for label, nfe, k, valid, v, d, wins in rows:
            ws.append([label, nfe, k] + [round(float(t), 2) for t in v] + [round(float(np.nanmean(v)), 2), round(float(np.nanstd(v, ddof=1)), 2),
                      round(float(np.nanmean(d)), 2), f'{wins}/8', 'yes' if valid else 'no: the method sees the slice it predicts'])
        ws.append([]); ws.append(['R_obs = MSE between the output and the held-out noisy slice (HU^2), brain-tissue mask from the neighbouring slices, slices 1-48.'])
        ws.append(['For valid methods R_obs = true MSE + noise power of the reference, so differences between rows are differences of the true MSE.'])
        wb.save(a.xlsx); print('\nwrote', a.xlsx)


if __name__ == '__main__':
    main()
