"""Chen's CNR on the PCCT test cases for every method in the result tree.

CNR = (mean_GM - mean_WM) / sqrt(var_GM + var_WM) on HU clipped to [0, 100], pooled over ALL ROI
voxels of the volume (the GM and WM ROIs sit on different slices, so a per-slice CNR is undefined).
Baselines are the fixed folders below; every imf/gan/* and imf/nogan/* folder is picked up
automatically, so a new NFE run appears as soon as its folder exists.

    python PCCT_experiments/eval_cnr.py                 # table to stdout
    python PCCT_experiments/eval_cnr.py --xlsx out.xlsx
Runs on the host (no GPU); RESULTS_ROOT overrides the result tree location.
"""
import argparse
import glob
import os
import sys

import numpy as np
import nibabel as nb

sys.stdout.reconfigure(encoding='utf-8')
R = os.environ.get('RESULTS_ROOT', r'D:\research\projects\denoising\results')
P = os.path.join(R, 'pcct')
CASES = list(range(29, 37))


def first(pattern):
    g = sorted(glob.glob(pattern))
    return g[0] if g else None


def methods():
    m = {
        'Noisy input': lambda c: os.path.join(P, 'imf', 'gan', 'imf_gan_unsupervised_PCCT_epoch48_nfe3', str(c), 'condition_img.nii.gz'),
        'Noise2Noise': lambda c: first(os.path.join(P, 'noise2noise_results', str(c), 'epoch*', 'pred_img.nii.gz')),
        'DDM2 first':  lambda c: os.path.join(P, 'DDM2_results', str(c), 'ddm2_firststep_image.nii.gz'),
        'DDM2 final':  lambda c: os.path.join(P, 'DDM2_results', str(c), 'ddm2_finalstep_image.nii.gz'),
        'DDIM K8':     lambda c: first(os.path.join(P, 'DDIM_results', str(c), 'epoch*avg', 'pred_img_scans8.nii.gz')),
    }
    for kind in ('nogan', 'gan'):
        for d in sorted(glob.glob(os.path.join(P, 'imf', kind, '*'))):
            if not os.path.isdir(d):
                continue
            tag = os.path.basename(d).replace('imf_gan_unsupervised_PCCT_epoch48_', 'iMF+GAN ').replace(
                'imf_v2_unsupervised_PCCT_epoch200_', 'iMF ')
            for k in (10, 20):
                m[f'{tag} K{k}'] = (lambda d, k: lambda c: os.path.join(d, str(c), f'pred_img_scans{k}.nii.gz'))(d, k)
    return m


def cnr(vol, gm, wm):
    x = np.clip(vol, 0, 100)
    g, w = x[gm], x[wm]
    return float((g.mean() - w.mean()) / np.sqrt(g.std() ** 2 + w.std() ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', default=None)
    ap.add_argument('--only', nargs='*', default=None, help='substring filters on method names')
    a = ap.parse_args()
    rois = {}
    for c in CASES:
        d = os.path.join(P, 'imf', 'gan', 'imf_gan_unsupervised_PCCT_epoch48_nfe3', str(c))
        rois[c] = tuple(np.round(np.asarray(nb.load(os.path.join(d, f'{t}_ROI.nii.gz')).dataobj)).astype(bool) for t in ('GM', 'WM'))
    table = {}
    for name, path_of in methods().items():
        if a.only and not any(s.lower() in name.lower() for s in a.only):
            continue
        row = []
        for c in CASES:
            p = path_of(c)
            row.append(cnr(np.asarray(nb.load(p).dataobj, np.float32), *rois[c]) if p and os.path.isfile(p) else np.nan)
        if not all(np.isnan(row)):
            table[name] = row
    w = max(len(n) for n in table)
    print(f"{'method':<{w}} " + ' '.join(f'{c:>6}' for c in CASES) + '    mean   std')
    for name, row in table.items():
        r = np.array(row, dtype=float)
        print(f'{name:<{w}} ' + ' '.join('     -' if np.isnan(v) else f'{v:6.2f}' for v in r)
              + f'  {np.nanmean(r):6.3f} {np.nanstd(r, ddof=1):6.3f}')
    if a.xlsx:
        import openpyxl
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = 'CNR'
        ws.append(['method'] + [f'case{c}' for c in CASES] + ['mean', 'std'])
        for name, row in table.items():
            r = np.array(row, dtype=float)
            ws.append([name] + [None if np.isnan(v) else round(float(v), 3) for v in r]
                      + [round(float(np.nanmean(r)), 3), round(float(np.nanstd(r, ddof=1)), 3)])
        wb.save(a.xlsx); print('wrote', a.xlsx)


if __name__ == '__main__':
    main()
