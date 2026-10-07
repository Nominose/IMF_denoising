r"""Does the x2-reference identity hold on brain CT, where a clean reference exists to check it?

For a method that generates slice j from its neighbours only, the noisy slice x_j = z_j + n_j is an
independent observation of the same anatomy and
    R_obs = MSE(pred_j, x_j) = MSE(pred_j, z_j) + E|n_j|^2 = true MSE + a method-independent constant.
This script measures R_obs and the true MSE for every method x NFE at K=20 (16 test patients) and
for the K-sweep, on a brain-tissue mask that does NOT use the clean image (it is taken from the
smoothed mean of the two neighbouring noisy slices, hence independent of n_j), and reports
    C_hat = R_obs - true MSE     one constant if the identity holds
    the rank agreement between R_obs and the true MSE, and every method pair it would order wrongly.
It is the evidence behind PCCT_experiments/eval_x2ref.py, which applies R_obs where no clean
reference exists. A method whose C_hat departs from the common constant is one whose output is
correlated with n_j, i.e. one that sees the slice it predicts; R_obs is not valid for it.

    python tmp/x2ref_brain_check.py           # numpy only, a few minutes with 4 processes
"""
import csv
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import nibabel as nb
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_brain as E

sys.stdout.reconfigure(encoding='utf-8')
B = E.B
KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
KSWEEP = r'D:\research\projects\denoising\models\ksweep\imf_gan_unsupervised_gaussian_brainCT\pred_images_nfe3'
ld = lambda p: np.asarray(nb.load(p).dataobj, np.float32)


def one(args):
    pid, items = args
    lp = f'{pid:08d}'
    g = ld(glob.glob(os.path.join(B, 'gt_and_conditions', lp, '**', 'gt_img.nii.gz'), recursive=True)[0])
    x = ld(glob.glob(os.path.join(B, 'gt_and_conditions', lp, '**', 'condition_img.nii.gz'), recursive=True)[0])
    n = g.shape[2]
    nbr = np.empty_like(x)                                   # mean of the neighbouring NOISY slices
    nbr[:, :, 1:-1] = 0.5 * (x[:, :, :-2] + x[:, :, 2:]); nbr[:, :, 0] = x[:, :, 1]; nbr[:, :, -1] = x[:, :, -2]
    sm = np.stack([gaussian_filter(nbr[:, :, s], 2.0) for s in range(n)], axis=2)
    mask = ((sm >= 0) & (sm <= 100)).astype(np.float32)      # no clean image involved

    def mse(a, b):
        d = mask.sum((0, 1)); v = d > 0
        return float(((((a - b) ** 2) * mask).sum((0, 1))[v] / d[v]).mean())
    return [(key, pid, mse(ld(p)[:, :, :n], x), mse(ld(p)[:, :, :n], g)) for key, p in items]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); n = len(a)
    return 1 - 6 * np.sum((ra - rb) ** 2) / (n * (n * n - 1))


def main():
    gts = E.gt_volumes(); per = {pid: [] for pid in gts}
    for m in ('DDIM', 'EDM', 'FM', 'iMF', 'GAN'):
        for nfe in E.NFES:
            f = E.METHODS[m](nfe)
            if not f or not os.path.isdir(f):
                continue
            for pid in gts:
                p = E.pred_volume(f, pid, 20, E.DEPLOYED.get(m))
                if p:
                    per[pid].append(((m, nfe, 20), p))
    for avg in glob.glob(os.path.join(KSWEEP, '*', '*', 'random_*', 'epoch28avg')):
        pid = int(os.path.relpath(avg, KSWEEP).split(os.sep)[0])
        for k in KS:
            per[pid].append((('Ksweep', 3, k), os.path.join(avg, f'pred_img_scans{k}.nii.gz')))
    print('volumes:', sum(len(v) for v in per.values()), flush=True)
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(one, list(per.items())):
            rows += r
    with open(os.path.join(B, 'brainCT_x2ref_check.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'nfe', 'k', 'patient', 'R_obs', 'true_mse'])
        for (m, nfe, k), pid, a, b in rows:
            w.writerow([m, nfe, k, pid, a, b])

    agg = {}
    for key, _, a, b in rows:
        agg.setdefault(key, []).append((a, b))
    agg = {k: np.mean(v, 0) for k, v in agg.items()}
    print()
    print('C_hat = R_obs - true MSE (HU^2), K=20, 16 patients')
    print('        ' + '  '.join(f'NFE{n:<3}' for n in E.NFES))
    for m in ('DDIM', 'FM', 'iMF', 'GAN', 'EDM'):
        print(f'  {m:<5} ' + '  '.join(f'{agg[(m, n, 20)][0] - agg[(m, n, 20)][1]:6.2f}' if (m, n, 20) in agg else '   -  ' for n in E.NFES))
    keys = [k for k in sorted(agg) if k[0] not in ('EDM', 'Ksweep')]
    o = np.array([agg[k][0] for k in keys]); t = np.array([agg[k][1] for k in keys]); ch = o - t
    bad = tot = 0; worst = 0.0
    for n in E.NFES:
        ks = [k for k in keys if k[1] == n]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                do, dt = agg[ks[i]][0] - agg[ks[j]][0], agg[ks[i]][1] - agg[ks[j]][1]
                tot += 1; bad += int(do * dt < 0); worst = max(worst, abs(do - dt))
    print()
    print(f'without EDM: {len(keys)} settings, Spearman(R_obs, true MSE) = {spearman(o, t):.4f}, C_hat spread {ch.max() - ch.min():.3f} HU^2')
    print(f'             method pairs at equal NFE ordered wrongly: {bad} of {tot}; largest error of a difference {worst:.3f} HU^2')
    ks = sorted((k for k in agg if k[0] == 'Ksweep'), key=lambda k: k[2])
    if ks:
        c = np.array([agg[k][0] - agg[k][1] for k in ks])
        print(f'K-sweep (iMF+GAN, NFE=3, K=1..20): C_hat {c.min():.3f}-{c.max():.3f} HU^2')


if __name__ == '__main__':
    main()
