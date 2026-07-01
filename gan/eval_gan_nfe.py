"""
eval_gan_nfe.py — full-set eval of ONE checkpoint at ONE NFE, printing parseable RESULT lines.

Companion to run_gan_nfe_sweep.sh. Reads the averaged predictions written by predict_2D_imf_v2.py
(pred_images_nfe{N}/.../epoch{E}avg/pred_img_scans{10,20}.nii.gz + gt_img.nii.gz) and reports
MAE / SSIM / LPIPS on the brain window [0,100] HU (same convention as compare_nfe.py), as
mean+-std across cases, for K=10 and K=20.

Usage (inside the docker env, GPU optional — LPIPS runs on CPU if no CUDA):
    python gan/eval_gan_nfe.py --trial imf_gan_unsupervised_gaussian_brainCT --epoch 28 --nfe 5

Output (parseable):
    RESULT K10 MAE 2.657+-0.406 SSIM 0.773+-0.038 LPIPS 0.0682+-0.0151
    RESULT K20 MAE 2.608+-0.408 SSIM 0.782+-0.039 LPIPS 0.0683+-0.0159
"""
import os, glob, argparse
import numpy as np, nibabel as nb, torch, lpips
from skimage.metrics import structural_similarity


def _detect_base():
    for b in ('/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'projects/denoising/models')):
            return b
    return '/host/d/research'


STUDY = os.path.join(_detect_base(), 'projects/denoising/models')
VMIN, VMAX = 0.0, 100.0
load = lambda p: np.asarray(nb.load(p).get_fdata(), dtype=np.float32) if os.path.isfile(p) else None


def calc_mae(a, b):
    out = []
    for s in range(a.shape[-1]):
        m = ((b[:, :, s] >= VMIN) & (b[:, :, s] <= VMAX)).astype(np.float32); d = m.sum()
        if d > 0: out.append(float((np.abs(a[:, :, s] - b[:, :, s]) * m).sum() / d))
    return float(np.mean(out)) if out else np.nan


def calc_ssim(a, b):
    out = []
    for s in range(a.shape[-1]):
        m = ((b[:, :, s] >= VMIN) & (b[:, :, s] <= VMAX)).astype(np.float32); d = m.sum()
        if d == 0: continue
        _, smap = structural_similarity(a[:, :, s], b[:, :, s], data_range=VMAX - VMIN, full=True)
        out.append(float((smap * m).sum() / d))
    return float(np.mean(out)) if out else np.nan


def calc_lpips(a, b, fn, dev):
    out = []
    for s in range(a.shape[-1]):
        x = (np.clip(a[:, :, s], VMIN, VMAX) - VMIN) / (VMAX - VMIN) * 2 - 1
        y = (np.clip(b[:, :, s], VMIN, VMAX) - VMIN) / (VMAX - VMIN) * 2 - 1
        tx = torch.from_numpy(np.stack([x, x, x])[None].astype(np.float32)).to(dev)
        ty = torch.from_numpy(np.stack([y, y, y])[None].astype(np.float32)).to(dev)
        with torch.no_grad(): out.append(float(fn(tx, ty).item()))
    return float(np.mean(out)) if out else np.nan


def main():
    ap = argparse.ArgumentParser('full-set eval of one checkpoint at one NFE')
    ap.add_argument('--trial', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--nfe', type=int, required=True)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn = lpips.LPIPS(net='alex').to(dev)

    folder = os.path.join(STUDY, args.trial, f'pred_images_nfe{args.nfe}')
    dirs = sorted(glob.glob(os.path.join(folder, '*', '*', 'random_*', f'epoch{args.epoch}avg')))
    print(f'nfe={args.nfe} epoch={args.epoch}: {len(dirs)} cases under {folder}')
    res = {10: {'mae': [], 'ssim': [], 'lpips': []}, 20: {'mae': [], 'ssim': [], 'lpips': []}}
    for d in dirs:
        gt = load(os.path.join(d, 'gt_img.nii.gz'))
        if gt is None: continue
        for k in (10, 20):
            pr = load(os.path.join(d, f'pred_img_scans{k}.nii.gz'))
            if pr is None: continue
            dd = min(pr.shape[-1], gt.shape[-1]); p, g = pr[..., :dd], gt[..., :dd]
            res[k]['mae'].append(calc_mae(p, g)); res[k]['ssim'].append(calc_ssim(p, g)); res[k]['lpips'].append(calc_lpips(p, g, fn, dev))

    std = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0
    for k in (10, 20):
        m = res[k]
        if not m['mae']:
            print(f'RESULT K{k} NO_DATA'); continue
        print(f"RESULT K{k} MAE {np.mean(m['mae']):.3f}+-{std(m['mae']):.3f} "
              f"SSIM {np.mean(m['ssim']):.3f}+-{std(m['ssim']):.3f} "
              f"LPIPS {np.mean(m['lpips']):.4f}+-{std(m['lpips']):.4f}")


if __name__ == '__main__':
    main()
