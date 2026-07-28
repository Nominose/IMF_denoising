"""
eval_pcct_cnr.py — CNR evaluation for real-world PCCT, using Chen's original formulation.

Real PCCT has no ground truth, so image quality is measured by the grey/white-matter
contrast-to-noise ratio over hand-drawn ROIs. The three lines that define the metric are copied
verbatim from PCCT_experiments/main_quantitative_new.ipynb (`calculate_CNR`) so these numbers stay
comparable with the previously reported ones:

    CNR = (GM_mean - WM_mean) / sqrt(GM_std^2 + WM_std^2)

and the three conventions that go with it, each of which changes the number if dropped:
    1. the image is CLIPPED to [0, 100] HU first  (brain-tissue window)
    2. the ROI masks are np.round()-ed, then compared == 1
    3. statistics are pooled over ALL ROI voxels of the volume (not averaged per slice)

Note the GM and WM ROIs sit on DIFFERENT slices (e.g. case 29: GM on z=30..44, WM on z=3..10), so
the prediction must cover the whole 50-slice volume -- which is why the predict script runs
--slice_range all. A pooled CNR is meaningful here because it contrasts two tissue populations,
not two regions of one slice.

    python PCCT_experiments/eval_pcct_cnr.py --trial imf_v2_unsupervised_PCCT --epoch 200 --nfe 5
    python PCCT_experiments/eval_pcct_cnr.py --trial imf_gan_unsupervised_PCCT --epoch 28 --nfe 5 --k 10 20
"""
import os
import glob
import argparse

import numpy as np
import nibabel as nb
import pandas as pd


def _detect_base():
    for b in ('/gpfs/work/aac/xingyiyao23', '/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'projects/denoising/models')):
            return b
    return '/gpfs/work/aac/xingyiyao23'


_BASE = _detect_base()
CLIP_LO, CLIP_HI = 0.0, 100.0          # Chen's brain-tissue window


def calculate_CNR(img, wm_roi, gm_roi):
    """Verbatim from main_quantitative_new.ipynb — do not "improve" this: the reported PCCT numbers
    were produced by exactly this arithmetic."""
    wm_mean = np.mean(img[wm_roi == 1])
    wm_std = np.std(img[wm_roi == 1])
    gm_mean = np.mean(img[gm_roi == 1])
    gm_std = np.std(img[gm_roi == 1])
    contrast = gm_mean - wm_mean
    contrast_to_noise_ratio = (gm_mean - wm_mean) / np.sqrt(gm_std ** 2 + wm_std ** 2)
    return wm_mean, wm_std, gm_mean, gm_std, contrast, contrast_to_noise_ratio


def load_clipped(path):
    """Load a volume and apply Chen's [0,100] clip (the notebook does this to every image before CNR)."""
    img = np.asarray(nb.load(path).get_fdata(), dtype=np.float64)
    img[img < CLIP_LO] = CLIP_LO
    img[img > CLIP_HI] = CLIP_HI
    return img


def load_roi(path):
    return np.round(np.asarray(nb.load(path).get_fdata(), dtype=np.float64))


def main():
    ap = argparse.ArgumentParser('PCCT CNR evaluation (Chen formulation)')
    ap.add_argument('--trial', required=True, help='e.g. imf_v2_unsupervised_PCCT or imf_gan_unsupervised_PCCT')
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--nfe', type=int, default=5, help="matches the predict's pred_images_nfe{N} folder")
    ap.add_argument('--k', type=int, nargs='+', default=[10, 20], help='which avg-of-K volumes to score')
    ap.add_argument('--study_folder', default=os.path.join(_BASE, 'projects/denoising/models'))
    ap.add_argument('--roi_dir', default=os.path.join(_BASE, 'Data/PCCT/ROI'))
    ap.add_argument('--out', default=None, help='xlsx to write (default: alongside the predictions)')
    ap.add_argument('--folder', default=None,
                    help='prediction folder to score, overriding the default pred_images_nfe<N>. '
                         'Needed for runs whose config is encoded in the folder name, e.g. the '
                         '--weights raw sweeps that land in pred_images_nfe<N>_raw.')
    args = ap.parse_args()

    folder = args.folder or os.path.join(args.study_folder, args.trial, f'pred_images_nfe{args.nfe}')
    if not os.path.isdir(folder):
        raise SystemExit(f'predictions not found: {folder}\n(run the predict script for this NFE first)')

    avg_dirs = sorted(glob.glob(os.path.join(folder, '*', 'random_*', f'epoch{args.epoch}avg')))
    print(f'trial={args.trial} epoch={args.epoch} nfe={args.nfe}')
    print(f'CNR = (GM_mean - WM_mean) / sqrt(GM_std^2 + WM_std^2), image clipped to [{CLIP_LO},{CLIP_HI}] HU')
    print(f'{len(avg_dirs)} case(s) under {folder}\n')

    rows, missing_roi = [], []
    for d in avg_dirs:
        case = os.path.basename(os.path.dirname(os.path.dirname(d)))
        gm_f = os.path.join(args.roi_dir, case, 'GM_ROI.nii.gz')
        wm_f = os.path.join(args.roi_dir, case, 'WM_ROI.nii.gz')
        if not (os.path.isfile(gm_f) and os.path.isfile(wm_f)):
            missing_roi.append(case)
            print(f'case {case}: NO ROI — skipped')
            continue
        gm_roi, wm_roi = load_roi(gm_f), load_roi(wm_f)

        row = {'Patient_ID': case}
        # baseline: the noisy input the denoiser has to beat
        cond_f = os.path.join(d, 'condition_img.nii.gz')
        if os.path.isfile(cond_f):
            cond = load_clipped(cond_f)
            if cond.shape != gm_roi.shape:
                print(f'case {case}: [warn] condition {cond.shape} vs ROI {gm_roi.shape} — skipped')
            else:
                wm_m, wm_s, gm_m, gm_s, c, cnr = calculate_CNR(cond, wm_roi, gm_roi)
                row.update({'WM_Mean_Noisy': wm_m, 'WM_Std_Noisy': wm_s, 'GM_Mean_Noisy': gm_m,
                            'GM_Std_Noisy': gm_s, 'Contrast_Noisy': c, 'CNR_Noisy': cnr})

        for k in args.k:
            pf = os.path.join(d, f'pred_img_scans{k}.nii.gz')
            if not os.path.isfile(pf):
                print(f'case {case}: no scans{k}')
                continue
            pred = load_clipped(pf)
            if pred.shape != gm_roi.shape:
                print(f'case {case} K{k}: [warn] pred {pred.shape} vs ROI {gm_roi.shape} — skipped')
                continue
            wm_m, wm_s, gm_m, gm_s, c, cnr = calculate_CNR(pred, wm_roi, gm_roi)
            row.update({f'WM_Mean_K{k}': wm_m, f'WM_Std_K{k}': wm_s, f'GM_Mean_K{k}': gm_m,
                        f'GM_Std_K{k}': gm_s, f'Contrast_K{k}': c, f'CNR_K{k}': cnr})

        base = row.get('CNR_Noisy', float('nan'))
        parts = [f'noisy {base:6.4f}'] if base == base else ['noisy   n/a']
        for k in args.k:
            v = row.get(f'CNR_K{k}')
            if v is not None:
                gain = f' ({v / base:.2f}x)' if base == base and base != 0 else ''
                parts.append(f'K{k} {v:6.4f}{gain}')
        print(f'case {case}: ' + '   '.join(parts))
        rows.append(row)

    if not rows:
        raise SystemExit('no cases scored — check the predictions and the ROI folder')

    df = pd.DataFrame(rows)
    out = args.out or os.path.join(folder, f'PCCT_CNR_epoch{args.epoch}_nfe{args.nfe}.xlsx')
    df.to_excel(out, index=False)

    print('\n=== summary (mean +- std across cases) ===')
    std = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0
    cols = [('noisy input', 'CNR_Noisy')] + [(f'denoised K={k}', f'CNR_K{k}') for k in args.k]
    base_mean = None
    for label, col in cols:
        if col not in df.columns:
            continue
        v = df[col].dropna().values
        if len(v) == 0:
            continue
        m = float(np.mean(v))
        if col == 'CNR_Noisy':
            base_mean = m
            print(f'{label:>15}: CNR {m:6.4f} +- {std(v):6.4f}   (n={len(v)})')
        else:
            g = f'   {m / base_mean:5.2f}x vs noisy' if base_mean else ''
            print(f'{label:>15}: CNR {m:6.4f} +- {std(v):6.4f}   (n={len(v)}){g}')
    if missing_roi:
        print(f'\n[warn] {len(missing_roi)} case(s) had no ROI and were skipped: {missing_roi}')
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()
