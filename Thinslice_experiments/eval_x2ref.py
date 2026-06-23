"""
eval_x2ref.py — Ground-truth-FREE (N,K) selection check via the x2-reference identity.

Theory (your x2_reference derivation):
    R_obs(N,K) = E|| y_hat(N,K) - x2 ||^2  =  R_true(N,K) + C_noise
where:
    R_true(N,K) = E|| y_hat(N,K) - z ||^2   (true clean MSE, needs GT)
    x2          = an INDEPENDENT noisy observation of the same clean slice z
    C_noise     = E||n2||^2  is (N,K)-INDEPENDENT  ==> ranking/argmin over (N,K) is preserved.

Here the valid x2 reference is the *held-out middle slice* of the noisy volume:
    pred[:,:,j] was generated from neighbour slices j-1, j+1 (condition),
    so condition_img[:,:,j] (the noisy slice j) is independent of the prediction -> valid x2.

What this script does, for each (NFE, K):
    R_obs  = MSE( pred ,  noisy condition_img )   <- NO ground truth used
    R_true = MSE( pred ,  clean gt_img )          <- uses GT, ONLY to validate the ranking
and then checks:
    (1) does argmin_{N,K} R_obs == argmin_{N,K} R_true ?   (can we pick (N,K) GT-free?)
    (2) is  C_hat = R_obs - R_true  roughly constant across (N,K) ?  (the identity holds?)

MSE is computed on the brain window [vmin,vmax] HU, with a FIXED mask defined by the CLEAN GT
(same pixels for R_obs and R_true, and same across all (N,K) -> C_noise stays constant).
NOTE: x2-reference is an L2 (MSE) tool only. It does NOT apply to LPIPS (LPIPS-vs-noisy is
meaningless) and is not rigorous for MAE. So this script reports MSE only.

Dependency-light: numpy / nibabel / pandas. No torch / lpips needed.

Usage:
    python Thinslice_experiments/eval_x2ref.py --epoch 200 --num_steps 3 5 --k_list 1 10 20
"""
import os
import glob
import time
import argparse
import numpy as np
import nibabel as nb
import pandas as pd


def load(path):
    # float32 (avoids nibabel's default float64 copy). The real cost here is gzip decompression
    # + reading off the slow docker/Windows mount, NOT this conversion.
    return np.asarray(nb.load(path).get_fdata(dtype=np.float32)) if os.path.isfile(path) else None


def calc_mse(img, ref, mask):
    """Vectorized per-slice MSE over masked pixels, averaged over slices. mask is (H,W,S) {0,1}.
    No Python per-slice loop -> this is milliseconds; if the script is slow it is file I/O."""
    se = ((img - ref) ** 2) * mask
    denom = mask.sum(axis=(0, 1))            # (S,)
    valid = denom > 0
    if not valid.any():
        return np.nan
    per_slice = se.sum(axis=(0, 1))[valid] / denom[valid]
    return float(per_slice.mean())


def _align(*arrs):
    """Align a set of (H,W,S) volumes to the common minimum depth (from the start), mirroring
    eval_imf_v2.py's shape handling."""
    d = min(a.shape[-1] for a in arrs)
    return [a[..., :d] for a in arrs]


def _detect_base():
    for b in ('/host/d/research', '/host/d'):
        if os.path.isdir(os.path.join(b, 'projects/denoising/models')):
            return b
    return '/host/d/research'


_BASE = _detect_base()


def get_args():
    p = argparse.ArgumentParser('x2-reference GT-free (N,K) selection check (MSE only)')
    p.add_argument('--trial_name', type=str, default='imf_v2_unsupervised_gaussian_brainCT')
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--num_steps', type=int, nargs='+', default=[3, 5], help='NFE folders to evaluate')
    p.add_argument('--k_list', type=int, nargs='+', default=[1, 10, 20], help='K values (averaged samples)')
    p.add_argument('--study_folder', type=str, default=os.path.join(_BASE, 'projects/denoising/models'))
    p.add_argument('--vmin', type=float, default=0.0)
    p.add_argument('--vmax', type=float, default=100.0)
    p.add_argument('--out', type=str, default=os.path.join(_BASE, 'projects/denoising/results/imf_v2_x2ref.xlsx'))
    return p.parse_args()


def main():
    args = get_args()
    E = args.epoch
    rows = []

    for nfe in args.num_steps:
        save_folder = os.path.join(args.study_folder, args.trial_name, f'pred_images_nfe{nfe}')
        avg_dirs = sorted(glob.glob(os.path.join(save_folder, '*', '*', 'random_*', f'epoch{E}avg')))
        print(f'\n=== NFE={nfe} : found {len(avg_dirs)} cases under {save_folder} ===')
        if not avg_dirs:
            print('  (nothing found - did `--mode avg` finish for this NFE?)')
            continue

        for avg_dir in avg_dirs:
            case_root = os.path.dirname(avg_dir)
            e1_dir = os.path.join(case_root, f'epoch{E}_1')
            tag = '/'.join(case_root.split(os.sep)[-3:])
            load_t, comp_t = 0.0, 0.0

            tL = time.time()
            gt = load(os.path.join(avg_dir, 'gt_img.nii.gz'))          # clean GT (z)
            noisy = load(os.path.join(e1_dir, 'condition_img.nii.gz'))  # noisy volume (x2 reference)
            load_t += time.time() - tL
            if gt is None or noisy is None:
                print('  [skip] missing gt/condition in', tag); continue
            if noisy.ndim != 3:  # safety: condition is the noisy volume (H,W,S)
                noisy = noisy[..., 0]

            for k in args.k_list:
                tL = time.time()
                if k == 1:
                    pred = load(os.path.join(e1_dir, 'pred_img.nii.gz'))
                else:
                    pred = load(os.path.join(avg_dir, f'pred_img_scans{k}.nii.gz'))
                load_t += time.time() - tL
                if pred is None:
                    continue
                tC = time.time()
                p_, g_, n_ = _align(pred, gt, noisy)
                mask = ((g_ >= args.vmin) & (g_ <= args.vmax)).astype(np.float32)  # FIXED region from clean GT
                rows.append({
                    'nfe': nfe, 'k': k, 'case': tag,
                    'R_obs': calc_mse(p_, n_, mask),    # MSE vs noisy x2  (GT-FREE)
                    'R_true': calc_mse(p_, g_, mask),   # MSE vs clean GT  (validation only)
                })
                comp_t += time.time() - tC
            print(f'  done {tag}  (load {load_t:.2f}s | mse {comp_t:.3f}s)')

    if not rows:
        print('\nNo results. Run the `pred` and `avg` steps first.'); return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_excel(args.out, index=False)

    # ---- aggregate per (nfe,k): mean over cases ----
    g = df.groupby(['nfe', 'k'])[['R_obs', 'R_true']].mean()
    g['C_hat'] = g['R_obs'] - g['R_true']          # estimate of C_noise; should be ~constant
    g = g.reset_index().sort_values(['nfe', 'k']).reset_index(drop=True)

    print('\n================ x2-reference check (MSE on [%g,%g] HU) ================' % (args.vmin, args.vmax))
    print(f"{'NFE':>4} {'K':>4} | {'R_obs (vs x2)':>14} | {'R_true (vs GT)':>14} | {'C_hat=Robs-Rtrue':>16}")
    for _, r in g.iterrows():
        print(f"{int(r['nfe']):>4} {int(r['k']):>4} | {r['R_obs']:>14.5f} | {r['R_true']:>14.5f} | {r['C_hat']:>16.5f}")

    # ---- the two things we want to demonstrate ----
    obs_best = g.loc[g['R_obs'].idxmin()]
    true_best = g.loc[g['R_true'].idxmin()]
    print('\n--- (1) GT-free selection ---')
    print(f"  argmin R_obs  (GT-free) : NFE={int(obs_best['nfe'])}, K={int(obs_best['k'])}")
    print(f"  argmin R_true (with GT) : NFE={int(true_best['nfe'])}, K={int(true_best['k'])}")
    print('  -> MATCH: picking (N,K) by the GT-free x2-reference selects the SAME setting as GT.'
          if (obs_best['nfe'], obs_best['k']) == (true_best['nfe'], true_best['k'])
          else '  -> MISMATCH: x2-ref argmin differs from GT argmin (inspect ranking below).')

    # rank agreement across all settings (manual Spearman, no scipy dependency)
    ro = g['R_obs'].rank().to_numpy()
    rt = g['R_true'].rank().to_numpy()
    n = len(ro)
    spearman = 1.0 - 6.0 * np.sum((ro - rt) ** 2) / (n * (n * n - 1)) if n > 1 else np.nan
    print(f'\n--- (2) ranking agreement over all {n} (N,K) settings ---')
    print(f"  Spearman rank corr (R_obs vs R_true) = {spearman:.4f}   (1.0 = identical ranking)")
    print(f"  C_hat across settings: mean={g['C_hat'].mean():.5f}, std={g['C_hat'].std():.5f}, "
          f"spread={g['C_hat'].max() - g['C_hat'].min():.5f}")
    print("  (small C_hat std/spread => C_noise is ~(N,K)-independent => the identity holds)")
    print('\nsaved per-case results to:', args.out)


if __name__ == '__main__':
    main()
