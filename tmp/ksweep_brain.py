r"""MAE / SSIM / LPIPS as a function of the number of averaged outputs K, brain CT.

Reads the K-sweep produced by Thinslice_experiments/predict_2D_imf_v2.py run with
`--k_save 1 2 4 6 8 10 12 14 16 18 20` under a dedicated study folder:

    <root>/<trial>/pred_images_nfe{N}/<patient>/<sub>/random_0/epoch{E}avg/pred_img_scans{K}.nii.gz

Scores every K with the table's convention (tmp/eval_brain.py: window [0,100] HU, mask from the
original GT, clipped MAE / SSIM / LPIPS), then writes
    ksweep_brain_per_case.csv      one row per (model, patient, K)
    ksweep_brain.xlsx              mean +- std over patients per K, one sheet per model
    ksweep_brain_{gan,flow_only}.png/.pdf   per model: MAE, SSIM, LPIPS vs K with exponential,
                                            logarithmic and power-law fits (layout of Fig. 3A of
                                            the Noise2Noise-diffusion paper)
    ksweep_brain_compare.png/.pdf           both models on the same axes, mean +- 1 std

    python tmp/ksweep_brain.py --out D:/research/projects/denoising/results/paper_revision
"""
import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_brain as E

sys.stdout.reconfigure(encoding='utf-8')
KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
MODELS = [('N2N-iMF (adversarial fine-tuning)', 'imf_gan_unsupervised_gaussian_brainCT', 28),
          ('N2N-iMF (flow loss only)', 'imf_v2_unsupervised_gaussian_brainCT', 200)]


def tasks(root, nfe):
    gts = E.gt_volumes()
    out = []
    for label, trial, ep in MODELS:
        for avg in sorted(glob.glob(os.path.join(root, trial, f'pred_images_nfe{nfe}', '*', '*', 'random_*', f'epoch{ep}avg'))):
            pid = int(os.path.relpath(avg, os.path.join(root, trial, f'pred_images_nfe{nfe}')).split(os.sep)[0])
            if not all(os.path.isfile(os.path.join(avg, f'pred_img_scans{k}.nii.gz')) for k in KS):
                continue                                   # patient not finished yet
            for k in KS:
                out.append((label, nfe, k, pid, os.path.join(avg, f'pred_img_scans{k}.nii.gz'), gts[pid]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'D:\research\projects\denoising\models\ksweep')
    ap.add_argument('--out', default=r'D:\research\projects\denoising\results\paper_revision')
    ap.add_argument('--nfe', type=int, default=3)
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, 'ksweep_brain_per_case.csv')
    done = {}
    if os.path.isfile(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                done[(r['model'], int(r['k']), int(r['patient']))] = r
    todo = [t for t in tasks(a.root, a.nfe) if (t[0], t[2], t[3]) not in done]
    print(f'{len(done)} scored, {len(todo)} to score', flush=True)
    rows = list(done.values())
    if todo:
        with Pool(a.procs, initializer=E._init) as pool:
            for i, r in enumerate(pool.imap_unordered(E.work, todo), 1):
                assert not r[7], f'scoring failed: {r}'
                rows.append(dict(model=r[0], nfe=r[1], k=r[2], patient=r[3], mae=r[4], ssim=r[5], lpips=r[6]))
                if i % 20 == 0 or i == len(todo):
                    print(f'  {i}/{len(todo)}', flush=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['model', 'nfe', 'k', 'patient', 'mae', 'ssim', 'lpips']); w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r['model'], int(r['patient']), int(r['k']))))

    # ---- aggregate -----------------------------------------------------------------------------
    stat = {}
    for label, _, _ in MODELS:
        pats = sorted({int(r['patient']) for r in rows if r['model'] == label})
        if not pats:
            continue
        stat[label] = dict(patients=pats)
        for m in ('mae', 'ssim', 'lpips'):
            arr = np.array([[float(next(r[m] for r in rows if r['model'] == label and int(r['patient']) == p and int(r['k']) == k))
                             for k in KS] for p in pats])                      # (patients, K)
            stat[label][m] = arr
    for label, s in stat.items():
        print(f'\n{label}   (n = {len(s["patients"])} patients: {s["patients"]})')
        print('   K      MAE              SSIM             LPIPS')
        for j, k in enumerate(KS):
            print(f'  {k:>2}   ' + '   '.join(f'{s[m][:, j].mean():.{d}f} ± {s[m][:, j].std(ddof=1):.{d}f}' for m, d in (('mae', 3), ('ssim', 3), ('lpips', 4))))

    # ---- workbook ------------------------------------------------------------------------------
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, Side
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    th = Side(style='thin'); bd = Border(left=th, right=th, top=th, bottom=th)
    for label, s in stat.items():
        ws = wb.create_sheet('GAN' if 'adversarial' in label else 'flow only')
        ws.append([f'{label}, NFE = {a.nfe}, n = {len(s["patients"])} patients']); ws.append(['K', 'MAE', 'SSIM', 'LPIPS'])
        for j, k in enumerate(KS):
            ws.append([k] + [f'{s[m][:, j].mean():.{d}f} ± {s[m][:, j].std(ddof=1):.{d}f}' for m, d in (('mae', 3), ('ssim', 3), ('lpips', 4))])
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.font = Font(name='等线', size=12); c.border = bd; c.alignment = Alignment(horizontal='center')
        ws['A1'].font = Font(name='等线', size=12)
        for col, wd in zip('ABCD', (6, 18, 18, 20)):
            ws.column_dimensions[col].width = wd
    wb.save(os.path.join(a.out, 'ksweep_brain.xlsx'))

    # ---- figures -------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label, st in stat.items():
        tag = 'gan' if 'adversarial' in label else 'flow_only'
        fig_fits(plt, st, os.path.join(a.out, f'ksweep_brain_{tag}'))
    fig_compare(plt, stat, os.path.join(a.out, 'ksweep_brain_compare'))
    print('\nwrote', os.path.join(a.out, 'ksweep_brain*.{csv,xlsx,png,pdf}'))


# The three curve families of the earlier Noise2Noise-diffusion study (its Fig. 3A), so the two
# figures can be read side by side.
FITS = [('Exponential', 'r', lambda x, a, b, c: a * np.exp(-b * x) + c),
        ('Logarithmic', 'g', lambda x, a, b: a * np.log(x) + b),
        ('Power-law', 'b', lambda x, a, b, c: a * np.power(x, -b) + c)]
PANELS = [('mae', 'MAE (HU)', 'upper right'), ('ssim', 'SSIM', 'lower right'), ('lpips', 'LPIPS', 'upper right')]


def fig_fits(plt, st, base):
    """One model, three stacked panels in the style of the reference figure: black markers = mean
    over patients at each K, dashed curves = the three fits with their R^2 in the legend."""
    from scipy.optimize import curve_fit
    x = np.array(KS, float); xf = np.linspace(1, 20, 200)
    fig, axes = plt.subplots(3, 1, figsize=(5, 10.2), dpi=200)
    for ax, (m, name, loc) in zip(axes, PANELS):
        y = st[m].mean(0)
        ax.plot(x, y, 'ko-', label=f'{name.split(" ")[0]} for each K', linewidth=1.5, markersize=6)
        # All three families are monotone in K. When the measured curve is not (LPIPS of the
        # adversarially fine-tuned model is lowest at K = 6 and rises afterwards), drawing them would
        # suggest a plateau the data do not have, so the panel shows the data and marks the optimum.
        best = int(np.argmax(y) if m == 'ssim' else np.argmin(y))
        if best != len(y) - 1:
            ax.plot(x[best], y[best], 'o', ms=13, mfc='none', mec='tab:red', mew=1.8, label=f'Best at K = {int(x[best])}')
            print(f'  {os.path.basename(base):<22} {m:<5} non-monotone: best at K = {int(x[best])} ({y[best]:.4f}), K = 20 gives {y[-1]:.4f}; fits omitted')
            fits = []
        else:
            fits = FITS
        for fname, col, fn in fits:
            try:
                p0 = (y[0] - y[-1], 0.5, y[-1]) if fname != 'Logarithmic' else None
                p, _ = curve_fit(fn, x, y, p0=p0, maxfev=20000)
                r2 = 1 - np.sum((y - fn(x, *p)) ** 2) / np.sum((y - y.mean()) ** 2)
                ax.plot(xf, fn(xf, *p), col + '--', label=f'{fname} (R²={r2:.2f})', linewidth=1.2)
                print(f'  fit {os.path.basename(base):<22} {m:<5} {fname:<11} R2 = {r2:.3f}   params {np.round(p, 4).tolist()}')
            except Exception as ex:
                print(f'  fit {m} {fname}: failed ({str(ex)[:60]})')
        ax.set_xlim(0, 20.5); ax.set_xticks(np.arange(2, 21, 2))
        ax.set_xlabel('Number of Sampling Inferences, K', fontsize=12, fontweight='bold')
        ax.set_ylabel(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-'); ax.legend(fontsize=9, frameon=True, loc=loc)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_compare(plt, stat, base):
    """Both models on the same axes, no fits. Means only: the spread between patients (std ~0.45 HU
    in MAE) is several times the difference between the models and would bury it under two
    overlapping bands, although the comparison is paired -- see the printed per-patient counts."""
    if len(stat) < 2:
        return
    x = np.array(KS, float)
    cols = ['#c0392b', '#2c6fbb']
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=200)
    for ax, (m, name, _) in zip(axes, PANELS):
        for (label, st), c in zip(stat.items(), cols):
            ax.plot(x, st[m].mean(0), 'o-', color=c, ms=5, lw=1.5, label=label)
        ax.set_xlim(0, 20.5); ax.set_xticks(np.arange(2, 21, 2))
        ax.set_xlabel('Number of Sampling Inferences, K', fontsize=11, fontweight='bold')
        ax.set_ylabel(name, fontsize=11, fontweight='bold'); ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=True, fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    (la, a), (lb, b) = list(stat.items())
    print()
    print(f'patients (of {len(a["patients"])}) where "{la}" is better than "{lb}", per K = {KS}:')
    for m in ('mae', 'ssim', 'lpips'):
        wins = [(a[m][:, j] > b[m][:, j]).sum() if m == 'ssim' else (a[m][:, j] < b[m][:, j]).sum() for j in range(len(KS))]
        print(f'   {m:<5} {[int(w) for w in wins]}')


if __name__ == '__main__':
    main()
