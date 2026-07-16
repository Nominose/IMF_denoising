"""Full-set eval of a Mayo iMF checkpoint at one NFE. Prints parseable RESULT lines.
Reads pred_images_input_<input>/<pid>/random_*/epoch<E>avg/{gt_img, pred_img_scansK}.nii.gz.
Metrics on the abdomen HU window [-160,240] (same as DDM2 eval_mayo.py / the paper), mean+-std
across test patients, for K in --k. Volumes are already cropped to slices 150-200 by the predict.
Usage: python eval_mayo.py --trial imf_v2_unsupervised_gaussian_mayo --epoch 200 --input both
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
VMIN, VMAX = -160.0, 240.0
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--input', default='both')
    ap.add_argument('--k', type=int, nargs='+', default=[10, 20])
    ap.add_argument('--nfe', type=int, default=3, help="NFE — matches predict's config-encoded dir suffix _nfe{N}")
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn = lpips.LPIPS(net='alex').to(dev)

    folder = os.path.join(STUDY, args.trial, f'pred_images_input_{args.input}_nfe{args.nfe}')  # match predict's config-encoded dir
    dirs = sorted(glob.glob(os.path.join(folder, '*', 'random_*', f'epoch{args.epoch}avg')))
    print(f'trial={args.trial} epoch={args.epoch} input={args.input}: {len(dirs)} cases | window [{VMIN},{VMAX}] HU')
    res = {k: {'mae': [], 'ssim': [], 'lpips': []} for k in args.k}
    for d in dirs:
        gt = load(os.path.join(d, 'gt_img.nii.gz'))
        if gt is None:
            print('  no gt in', d); continue
        for k in args.k:
            pr = load(os.path.join(d, f'pred_img_scans{k}.nii.gz'))
            if pr is None:
                print(f'  no scans{k} in', d); continue
            dd = min(pr.shape[-1], gt.shape[-1]); p, g = pr[..., :dd], gt[..., :dd]
            res[k]['mae'].append(calc_mae(p, g)); res[k]['ssim'].append(calc_ssim(p, g)); res[k]['lpips'].append(calc_lpips(p, g, fn, dev))

    std = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0
    for k in args.k:
        m = res[k]
        if not m['mae']:
            print(f'RESULT K{k} NO_DATA'); continue
        print(f"RESULT K{k} MAE {np.mean(m['mae']):.3f}+-{std(m['mae']):.3f} "
              f"SSIM {np.mean(m['ssim']):.3f}+-{std(m['ssim']):.3f} "
              f"LPIPS {np.mean(m['lpips']):.4f}+-{std(m['lpips']):.4f}  (n={len(m['mae'])})")


if __name__ == '__main__':
    main()
