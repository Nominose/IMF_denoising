"""Chen's CNR on the PCCT test cases for every method in the result tree.

CNR = (mean_GM - mean_WM) / sqrt(var_GM + var_WM) on HU clipped to [0, 100], pooled over ALL ROI
voxels of the volume (the GM and WM ROIs sit on different slices, so a per-slice CNR is undefined).
Baselines are the fixed folders below; every imf/gan/* and imf/nogan/* folder is picked up
automatically, so a new NFE run appears as soon as its folder exists.

    python PCCT_experiments/eval_cnr.py                                   # table to stdout
    python PCCT_experiments/eval_cnr.py --xlsx results/pcct/PCCT_CNR_comparison.xlsx
The workbook has the layout of the original comparison file: Summary (one row per method/NFE/K,
sorted by mean CNR), Per-case, and GAN ablation (iMF vs iMF+GAN at the same NFE and K).
Runs on the host (no GPU); RESULTS_ROOT overrides the result tree location.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import nibabel as nb

sys.stdout.reconfigure(encoding='utf-8')
R = os.environ.get('RESULTS_ROOT', r'D:\research\projects\denoising\results')
P = os.path.join(R, 'pcct')
CASES = list(range(29, 37))
ROI_SRC = os.path.join(P, 'imf', 'gan', 'imf_gan_unsupervised_PCCT_epoch48_nfe3')


def first(pattern):
    g = sorted(glob.glob(pattern))
    return g[0] if g else None


def methods():
    """-> list of (method, nfe, k, path_of_case). nfe/k are ints or '-'."""
    m = [
        ('Noisy input', '-', '-', lambda c: os.path.join(ROI_SRC, str(c), 'condition_img.nii.gz')),
        ('Noise2Noise', 1, '-', lambda c: first(os.path.join(P, 'noise2noise_results', str(c), 'epoch*', 'pred_img.nii.gz'))),
        ('DDM2 (first step)', '-', '-', lambda c: os.path.join(P, 'DDM2_results', str(c), 'ddm2_firststep_image.nii.gz')),
        ('DDM2 (final step)', '-', '-', lambda c: os.path.join(P, 'DDM2_results', str(c), 'ddm2_finalstep_image.nii.gz')),
        ('DDIM', 100, 8, lambda c: first(os.path.join(P, 'DDIM_results', str(c), 'epoch*avg', 'pred_img_scans8.nii.gz'))),
    ]
    for kind, label in (('nogan', 'iMF'), ('gan', 'iMF + GAN')):
        dirs = [d for d in glob.glob(os.path.join(P, 'imf', kind, '*')) if os.path.isdir(d)]
        for d in sorted(dirs, key=lambda d: sort_key(_nfe_of(d))):       # numeric NFE order, not alphabetical
            nfe = _nfe_of(d)
            raw = '_raw' in os.path.basename(d)
            for k in (10, 20):
                m.append((label + (' (raw weights)' if raw else ''), nfe, k,
                          (lambda d, k: lambda c: os.path.join(d, str(c), f'pred_img_scans{k}.nii.gz'))(d, k)))
    return m


def cnr(vol, gm, wm):
    x = np.clip(vol, 0, 100)
    g, w = x[gm], x[wm]
    return float((g.mean() - w.mean()) / np.sqrt(g.std() ** 2 + w.std() ** 2))


def sort_key(nfe):
    return nfe if isinstance(nfe, int) else -1


def _nfe_of(folder):
    m = re.search(r'nfe(\d+)', os.path.basename(folder))
    return int(m.group(1)) if m else '-'


def write_xlsx(rows, path):
    """rows: list of dicts method/nfe/k/per (list of 8)/mean/std/n, in the original workbook's layout."""
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, Side
    hdr = Font(name='宋体', size=11, bold=True); body = Font(name='宋体', size=11)
    bottom = Border(bottom=Side(style='thin')); center = Alignment(horizontal='center')

    def sheet(ws, header, data, widths):
        ws.append(header)
        for c in ws[1]:
            c.font, c.alignment, c.border = hdr, center, bottom
        for r in data:
            ws.append(r)
        for r in ws.iter_rows(min_row=2):
            for c in r:
                c.font = body
        for col, w in widths.items():
            ws.column_dimensions[col].width = w

    wb = openpyxl.Workbook()
    r3 = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)
    # Summary: ascending mean CNR, like the original
    ws = wb.active; ws.title = 'Summary'
    sheet(ws, ['Method', 'NFE', 'K', 'CNR', 'CNR_mean', 'CNR_std', 'n'],
          [[r['method'], r['nfe'], r['k'], f"{r['mean']:.3f} ± {r['std']:.3f}", r3(r['mean']), r3(r['std']), r['n']]
           for r in sorted(rows, key=lambda r: r['mean'])],
          {'A': 20, 'B': 5, 'C': 4, 'D': 15, 'E': 10, 'F': 9, 'G': 3})
    # Per-case: method order as computed (baselines, then iMF by NFE, then iMF+GAN by NFE)
    ws = wb.create_sheet('Per-case')
    sheet(ws, ['Method', 'NFE', 'K'] + [f'case{c}' for c in CASES] + ['mean', 'std'],
          [[r['method'], r['nfe'], r['k']] + [r3(v) for v in r['per']] + [r3(r['mean']), r3(r['std'])] for r in rows],
          {'A': 20, 'B': 5, 'C': 4, 'L': 7})
    # GAN ablation: same NFE and K, iMF vs iMF + GAN; rows where one side is missing are kept blank
    ws = wb.create_sheet('GAN ablation')
    by = {(r['method'], r['nfe'], r['k']): r for r in rows}
    keys = sorted({(r['nfe'], r['k']) for r in rows if r['method'] in ('iMF', 'iMF + GAN') and isinstance(r['nfe'], int)})
    data = []
    for nfe, k in keys:
        a, b = by.get(('iMF', nfe, k)), by.get(('iMF + GAN', nfe, k))
        if a and b:
            wins = sum(1 for x, y in zip(a['per'], b['per']) if y > x)
            data.append([nfe, k, r3(a['mean']), r3(b['mean']), r3(b['mean'] - a['mean']),
                         round(100 * (b['mean'] - a['mean']) / a['mean'], 1), f'{wins}/{len(a["per"])}'])
        else:
            data.append([nfe, k, r3(a['mean']) if a else None, r3(b['mean']) if b else None, None, None,
                         'no-GAN iMF not run at this NFE' if b and not a else 'iMF+GAN not run at this NFE'])
    sheet(ws, ['NFE', 'K', 'iMF', 'iMF+GAN', 'Delta', 'Delta_%', 'GAN wins'], data,
          {'A': 5, 'B': 4, 'C': 7, 'D': 9, 'E': 7, 'F': 9, 'G': 30})
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', default=None)
    ap.add_argument('--only', nargs='*', default=None, help='substring filters on method names')
    a = ap.parse_args()
    rois = {}
    for c in CASES:
        d = os.path.join(ROI_SRC, str(c))
        rois[c] = tuple(np.round(np.asarray(nb.load(os.path.join(d, f'{t}_ROI.nii.gz')).dataobj)).astype(bool) for t in ('GM', 'WM'))
    rows = []
    for method, nfe, k, path_of in methods():
        if a.only and not any(s.lower() in method.lower() for s in a.only):
            continue
        per = []
        for c in CASES:
            p = path_of(c)
            per.append(cnr(np.asarray(nb.load(p).dataobj, np.float32), *rois[c]) if p and os.path.isfile(p) else np.nan)
        if all(np.isnan(per)):
            continue
        v = np.array(per, dtype=float)
        rows.append(dict(method=method, nfe=nfe, k=k, per=per, mean=float(np.nanmean(v)),
                         std=float(np.nanstd(v, ddof=1)), n=int(np.sum(~np.isnan(v)))))
    w = max(len(r['method']) for r in rows)
    print(f"{'method':<{w}} {'NFE':>4} {'K':>3} " + ' '.join(f'{c:>6}' for c in CASES) + '    mean   std')
    for r in rows:
        print(f"{r['method']:<{w}} {str(r['nfe']):>4} {str(r['k']):>3} " + ' '.join('     -' if np.isnan(v) else f'{v:6.2f}' for v in r['per'])
              + f"  {r['mean']:6.3f} {r['std']:6.3f}")
    if a.xlsx:
        write_xlsx(rows, a.xlsx); print('wrote', a.xlsx)


if __name__ == '__main__':
    main()
