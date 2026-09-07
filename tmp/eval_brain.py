"""Per-case MAE / SSIM / LPIPS for every brain-CT method in the result tree, all-clipped convention.

Brain window [0, 100] HU. Mask = voxels whose ORIGINAL ground truth lies in the window; both volumes
are clipped to the window before MAE and SSIM (data_range 100) and before LPIPS (mapped to [-1, 1],
AlexNet). Per-slice values are averaged over the 50 slices, then mean +- std (ddof=1) over cases.
Same convention as tmp/eval_mayo.py, so the numbers are directly comparable with the tables.

Reads whichever of these exist under results/brianct (patient IDs are matched as integers because
EDM / flow_matching drop the leading zeros):
    DDIM_results/pred_images_NFE{n}/<pid>/<sub>/random_0/epoch*avg/pred_img_scans{k}.nii.gz
    edm/pred_images_steps{n}_k20*/<pid>/...              flow_matching/pred_images_NFE{n}/<pid>/...
    imf_v2_unsupervised_gaussian_brainCT/pred_images_nfe{n}/<pid>/...   imf_gan_.../pred_images_nfe{n}/...
Writes one CSV row per (method, nfe, k, patient); aggregate with tmp/fill_brain_xlsx.py.

    python tmp/eval_brain.py --out D:/research/projects/denoising/results/brianct/brainCT_metrics_per_case.csv
    python tmp/eval_brain.py --methods EDM FM --procs 12
"""
import argparse
import csv
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import nibabel as nb

R = os.environ.get('RESULTS_ROOT', r'D:\research\projects\denoising\results')
B = os.path.join(R, 'brianct')
VMIN, VMAX = 0.0, 100.0
NFES = [2, 3, 5, 10, 20, 30, 50]
METHODS = {
    'DDIM': lambda n: os.path.join(B, 'DDIM_results', f'pred_images_NFE{n}'),
    'EDM':  lambda n: (glob.glob(os.path.join(B, 'edm', f'pred_images_steps{n}_k20*')) or [None])[0],
    'FM':   lambda n: os.path.join(B, 'flow_matching', f'pred_images_NFE{n}'),
    'iMF':  lambda n: os.path.join(B, 'imf_v2_unsupervised_gaussian_brainCT', f'pred_images_nfe{n}'),
    'GAN':  lambda n: os.path.join(B, 'imf_gan_unsupervised_gaussian_brainCT', f'pred_images_nfe{n}'),
}

_fn = None


def _init():
    global _fn
    import torch, lpips
    torch.set_num_threads(2)
    _fn = lpips.LPIPS(net='alex', verbose=False)


def epoch_of(p):
    m = re.search(r'epoch(\d+)avg', p.replace('\\', '/'))
    return int(m.group(1)) if m else -1


def gt_volumes():
    out = {}
    for p in glob.glob(os.path.join(B, 'gt_and_conditions', '*', '**', 'gt_img.nii.gz'), recursive=True):
        pid = int(os.path.relpath(p, os.path.join(B, 'gt_and_conditions')).replace('\\', '/').split('/')[0])
        out.setdefault(pid, p)
    return out


def pred_volume(folder, pid_int, k):
    """highest-epoch K-average of this patient, or None"""
    best = None
    for pdir in glob.glob(os.path.join(folder, '*')):
        try:
            if int(os.path.basename(pdir)) != pid_int:
                continue
        except ValueError:
            continue
        for p in glob.glob(os.path.join(pdir, '**', 'epoch*avg', f'pred_img_scans{k}.nii.gz'), recursive=True):
            if best is None or epoch_of(p) > epoch_of(best):
                best = p
    return best


def metrics(pred_path, gt_path):
    import torch
    from skimage.metrics import structural_similarity
    a = np.asarray(nb.load(pred_path).dataobj, np.float32)
    g = np.asarray(nb.load(gt_path).dataobj, np.float32)
    d = min(a.shape[-1], g.shape[-1]); a, g = a[..., :d], g[..., :d]
    mae, ssim, lp = [], [], []
    for s in range(d):
        gs = g[:, :, s]; m = ((gs >= VMIN) & (gs <= VMAX)).astype(np.float32); den = m.sum()
        if den == 0:
            continue
        x, y = np.clip(a[:, :, s], VMIN, VMAX), np.clip(gs, VMIN, VMAX)
        mae.append(float((np.abs(x - y) * m).sum() / den))
        _, smap = structural_similarity(x, y, data_range=VMAX - VMIN, full=True)
        ssim.append(float((smap * m).sum() / den))
        xn = (x - VMIN) / (VMAX - VMIN) * 2 - 1; yn = (y - VMIN) / (VMAX - VMIN) * 2 - 1
        tx = torch.from_numpy(np.stack([xn] * 3)[None]); ty = torch.from_numpy(np.stack([yn] * 3)[None])
        with torch.no_grad():
            lp.append(float(_fn(tx, ty).item()))
    # Never let an empty result through as NaN: under host memory pressure a pool of 12 workers once
    # returned 790 rows of silent NaNs (no exception, no error text) that looked like data.
    if len(mae) < 10 or not all(np.isfinite(v) for v in (np.mean(mae), np.mean(ssim), np.mean(lp))):
        raise RuntimeError(f'only {len(mae)} usable slices / non-finite metric for {os.path.basename(pred_path)}')
    return float(np.mean(mae)), float(np.mean(ssim)), float(np.mean(lp))


def work(task):
    method, nfe, k, pid, pred_path, gt_path = task
    try:
        m, s, l = metrics(pred_path, gt_path)
        return (method, nfe, k, pid, m, s, l, '')
    except Exception as e:
        return (method, nfe, k, pid, np.nan, np.nan, np.nan, str(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(B, 'brainCT_metrics_per_case.csv'))
    ap.add_argument('--methods', nargs='+', default=list(METHODS))
    ap.add_argument('--nfe', type=int, nargs='+', default=NFES)
    ap.add_argument('--k', type=int, nargs='+', default=[10, 20])
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    gts = gt_volumes()
    tasks, missing = [], []
    for method in a.methods:
        for n in a.nfe:
            folder = METHODS[method](n)
            if not folder or not os.path.isdir(folder):
                continue
            for pid, gt in sorted(gts.items()):
                for k in a.k:
                    p = pred_volume(folder, pid, k)
                    (tasks if p else missing).append((method, n, k, pid, p, gt))
    print(f'{len(tasks)} volumes to score, {len(missing)} (method,nfe,k,patient) combos without a prediction', flush=True)
    done = {}
    if os.path.isfile(a.out):                       # resume: keep rows already scored
        with open(a.out, newline='') as f:
            for row in csv.DictReader(f):
                done[(row['method'], int(row['nfe']), int(row['k']), int(row['patient']))] = row
    todo = [t for t in tasks if (t[0], t[1], t[2], t[3]) not in done]
    print(f'{len(done)} already in {a.out}, {len(todo)} to compute', flush=True)
    rows = list(done.values())
    with Pool(a.procs, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(work, todo), 1):
            rows.append(dict(method=r[0], nfe=r[1], k=r[2], patient=r[3], mae=r[4], ssim=r[5], lpips=r[6], error=r[7]))
            if i % 20 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)}', flush=True)
                _write(a.out, rows)
    _write(a.out, rows)
    print('wrote', a.out)


def _write(path, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'nfe', 'k', 'patient', 'mae', 'ssim', 'lpips', 'error'])
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r['method'], int(r['nfe']), int(r['k']), int(r['patient']))))
    os.replace(tmp, path)


if __name__ == '__main__':
    main()
