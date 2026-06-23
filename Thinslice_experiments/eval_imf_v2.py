"""
eval_imf_v2.py — Evaluate iMF v2 brain-CT predictions: MAE / SSIM / LPIPS vs GT.

Metrics are computed on the brain-tissue window [0, 100] HU (matches main_quantitative_imf.ipynb).
It reads the per-NFE folders written by predict_2D_imf_v2.py and reports, for each NFE and each
K (number of averaged samples), the mean +/- std over the test cases. This directly gives you the
NFE=3 vs NFE=5 comparison AND the perception-distortion curve (LPIPS vs K).

This script is dependency-light (numpy / nibabel / skimage / lpips / torch / pandas) and does NOT
import the heavy IMF_denoising data stack, so it runs even on a plain CPU box.

Usage:
    python Thinslice_experiments/eval_imf_v2.py --epoch 200 --num_steps 3 5 --k_list 1 5 10 20
"""
import os
import glob
import argparse
import numpy as np
import nibabel as nb
import pandas as pd
import torch
import lpips
from skimage.metrics import structural_similarity


def calc_mae(img, ref, vmin, vmax):
    maes = []
    for s in range(img.shape[-1]):
        a, b = img[:, :, s], ref[:, :, s]
        mask = ((b >= vmin) & (b <= vmax)).astype(np.float32)
        denom = mask.sum()
        if denom > 0:
            maes.append(float((np.abs(a - b) * mask).sum() / denom))
    return float(np.mean(maes)) if maes else np.nan


def calc_ssim(img, ref, vmin, vmax):
    ssims = []
    for s in range(img.shape[-1]):
        a, b = img[:, :, s], ref[:, :, s]
        mask = ((b >= vmin) & (b <= vmax)).astype(np.float32)
        denom = mask.sum()
        if denom == 0:
            continue
        _, smap = structural_similarity(a, b, data_range=vmax - vmin, full=True)
        ssims.append(float((smap * mask).sum() / denom))
    return float(np.mean(ssims)) if ssims else np.nan


def calc_lpips(img, ref, vmin, vmax, loss_fn, device):
    vals = []
    for s in range(img.shape[-1]):
        a = np.clip(img[:, :, s], vmin, vmax).astype(np.float32)
        b = np.clip(ref[:, :, s], vmin, vmax).astype(np.float32)
        a = (a - vmin) / (vmax - vmin) * 2 - 1
        b = (b - vmin) / (vmax - vmin) * 2 - 1
        ta = torch.from_numpy(np.stack([a, a, a])[None]).to(device)
        tb = torch.from_numpy(np.stack([b, b, b])[None]).to(device)
        with torch.no_grad():
            vals.append(float(loss_fn(ta, tb).item()))
    return float(np.mean(vals)) if vals else np.nan


def load(path):
    return nb.load(path).get_fdata() if os.path.isfile(path) else None


def _detect_base():
    # whole D: is mounted at /host/d; real data lives under /host/d/research
    for b in ('/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'projects/denoising/models')):
            return b
    return '/host/d/research'


_BASE = _detect_base()


def get_args():
    p = argparse.ArgumentParser('Evaluate iMF v2 brain-CT predictions')
    p.add_argument('--trial_name', type=str, default='imf_v2_unsupervised_gaussian_brainCT')
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--num_steps', type=int, nargs='+', default=[3, 5], help='NFE folders to evaluate')
    p.add_argument('--k_list', type=int, nargs='+', default=[10, 20], help='K values (averaged samples)')
    p.add_argument('--study_folder', type=str, default=os.path.join(_BASE, 'projects/denoising/models'))
    p.add_argument('--vmin', type=float, default=0.0)
    p.add_argument('--vmax', type=float, default=100.0)
    p.add_argument('--lpips_net', type=str, default='alex', choices=['alex', 'vgg'])
    p.add_argument('--out', type=str, default=os.path.join(_BASE, 'projects/denoising/results/imf_v2_brainCT_eval.xlsx'))
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, '| lpips net:', args.lpips_net)
    loss_fn = lpips.LPIPS(net=args.lpips_net).to(device)
    E = args.epoch

    rows = []
    for nfe in args.num_steps:
        save_folder = os.path.join(args.study_folder, args.trial_name, f'pred_images_nfe{nfe}')
        avg_dirs = sorted(glob.glob(os.path.join(save_folder, '*', '*', 'random_*', f'epoch{E}avg')))
        print(f'\n=== NFE={nfe} : found {len(avg_dirs)} cases under {save_folder} ===')
        if not avg_dirs:
            print('  (nothing found — did `--mode avg` finish for this NFE?)')
            continue

        for avg_dir in avg_dirs:
            case_root = os.path.dirname(avg_dir)
            e1_dir = os.path.join(case_root, f'epoch{E}_1')
            tag = '/'.join(case_root.split(os.sep)[-3:])

            gt = load(os.path.join(avg_dir, 'gt_img.nii.gz'))
            if gt is None:
                print('  [skip] no gt in', avg_dir); continue

            # method -> image
            methods = {}
            cond = load(os.path.join(e1_dir, 'condition_img.nii.gz'))
            if cond is not None:
                # condition is 2-channel (s-1, s+1) saved as the noisy reference; take first channel-equivalent
                methods['noisy'] = cond if cond.ndim == 3 else cond[..., 0]
            for k in args.k_list:
                if k == 1:
                    methods['k1'] = load(os.path.join(e1_dir, 'pred_img.nii.gz'))
                else:
                    methods[f'k{k}'] = load(os.path.join(avg_dir, f'pred_img_scans{k}.nii.gz'))

            for m, img in methods.items():
                if img is None:
                    continue
                if img.shape != gt.shape:
                    # e.g. condition stored with different slice count; align by min depth
                    d = min(img.shape[-1], gt.shape[-1])
                    img_e, gt_e = img[..., :d], gt[..., :d]
                else:
                    img_e, gt_e = img, gt
                rows.append({
                    'nfe': nfe, 'case': tag, 'method': m,
                    'mae': calc_mae(img_e, gt_e, args.vmin, args.vmax),
                    'ssim': calc_ssim(img_e, gt_e, args.vmin, args.vmax),
                    'lpips': calc_lpips(img_e, gt_e, args.vmin, args.vmax, loss_fn, device),
                })
            print('  done', tag)

    if not rows:
        print('\nNo results. Run the `pred` and `avg` steps first.'); return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_excel(args.out, index=False)

    # summary: mean +/- std across cases, per (nfe, method)
    g = df.groupby(['nfe', 'method'])[['mae', 'ssim', 'lpips']].agg(['mean', 'std'])
    method_order = ['noisy'] + [f'k{k}' for k in args.k_list]
    print('\n================ SUMMARY (window [%g,%g] HU, mean+/-std over cases) ================' % (args.vmin, args.vmax))
    for nfe in args.num_steps:
        if nfe not in df['nfe'].unique():
            continue
        print(f'\n--- NFE = {nfe} ---')
        print(f"{'method':>8} | {'MAE(down)':>16} | {'SSIM(up)':>16} | {'LPIPS(down)':>16}")
        for m in method_order:
            if (nfe, m) not in g.index:
                continue
            r = g.loc[(nfe, m)]
            print(f"{m:>8} | {r[('mae','mean')]:7.3f}+/-{r[('mae','std')]:<6.3f} | "
                  f"{r[('ssim','mean')]:7.3f}+/-{r[('ssim','std')]:<6.3f} | "
                  f"{r[('lpips','mean')]:7.4f}+/-{r[('lpips','std')]:<6.4f}")
    print('\nsaved per-case results to:', args.out)


if __name__ == '__main__':
    main()
