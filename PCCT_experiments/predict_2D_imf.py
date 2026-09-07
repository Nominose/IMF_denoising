"""iMF / iMF+GAN inference on the real photon-counting CT (PCCT) test set, any NFE, K samples averaged.

PCCT has no clean ground truth: the deliverable per case is the K-average volume plus the GM/WM ROI
masks that eval_cnr.py turns into Chen's CNR. Output layout matches the result folders the paper's
tables and figures are built from (tmp/image_showcase.py, PCCT_experiments/eval_cnr.py):

    <out>/<case>/{condition_img, GM_ROI, WM_ROI, pred_img_scans10, pred_img_scans20}.nii.gz

Per-sample volumes go to a work dir and are reused on re-run, so an interrupted run resumes.

Two things this script decides that the model checkpoint does not record:
  * normalisation: PCCT models use HU cut to [-1000, 1000] mapped linearly to [-1, 1], no histogram
    equalisation (the brain-CT convention is [-1000, 2000] + equalisation). --bg/--mx/--he override.
  * edge slices: the condition is the pair of neighbouring slices, and the local volumes are the
    50-slice extracts (30-80 of the original scan), so slices 0 and 49 have one neighbour missing.
    The missing one is replaced by the edge slice itself (--edge replicate); the original run had the
    real neighbours, so those two slices can differ slightly.

Usage (inside the docker env, GPU):
    python PCCT_experiments/predict_2D_imf.py --nfe 5                       # 8 test cases, K=20, EMA weights
    python PCCT_experiments/predict_2D_imf.py --nfe 3 --cases 31 --k 4 \
        --out /tmp/check --compare_to <delivered nfe3 folder>              # reproduce-check one case
"""
import argparse
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO_PARENT, '/host/c/Users/ROG/Documents/GitHub'):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import nibabel as nb
import torch

import IMF_denoising.improved_mean_flow as imf
import IMF_denoising.Generator_thinslice as Generator
from IMF_denoising.Thinslice_experiments.predict_2D_imf_v2 import build_model, _save_nifti, _detect_base

_BASE = _detect_base()
PCCT = os.path.join(_BASE, 'file/new/real_world_PCCT/soft_thins_xy')
RESULTS = os.path.join(_BASE, 'projects/denoising/results/pcct')


def get_args():
    p = argparse.ArgumentParser('PCCT iMF inference')
    p.add_argument('--nfe', type=int, required=True, help='Euler steps per sample')
    p.add_argument('--k', type=int, default=20, help='number of stochastic samples per case')
    p.add_argument('--k_save', type=int, nargs='+', default=[10, 20], help='which K-averages to write')
    p.add_argument('--cases', type=int, nargs='+', default=list(range(29, 37)), help='test cases (batch 2 = 29-36)')
    p.add_argument('--ckpt', default=os.path.join(RESULTS, 'model-48.pt'),
                   help='GANTrainer checkpoint (keys model/ema/D/opt_*); the iMF+GAN PCCT epoch-48 model')
    p.add_argument('--weights', choices=['ema', 'raw'], default='ema',
                   help="'ema' = the averaged weights (deployed); 'raw' = the online weights")
    p.add_argument('--out', default=None,
                   help='result folder; default results/pcct/imf/gan/imf_gan_unsupervised_PCCT_epoch48_nfe<N>[_raw]')
    p.add_argument('--work', default=None, help='per-sample volumes; default models/pcct_samples/<out name>')
    p.add_argument('--roi_from', default=os.path.join(RESULTS, 'imf/gan/imf_gan_unsupervised_PCCT_epoch48_nfe3'),
                   help='folder whose <case>/{GM,WM}_ROI.nii.gz are copied next to the predictions')
    p.add_argument('--data', default=PCCT, help='folder with <case>/soft_thins_0_noblank_sliced.nii.gz')
    p.add_argument('--bg', type=float, default=-1000.0, help='HU mapped to -1')
    p.add_argument('--mx', type=float, default=1000.0, help='HU mapped to +1')
    p.add_argument('--he', action='store_true', help='histogram equalisation (brain-CT convention; not PCCT)')
    p.add_argument('--edge', choices=['replicate'], default='replicate')
    p.add_argument('--slice_batch', type=int, default=2, help='slices per forward; 2 is fastest on a 12 GB card')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--compare_to', default=None,
                   help='another result folder: after averaging, print per-slice MAE and GM/WM means against '
                        'its <case>/pred_img_scans20.nii.gz (reproduce-check)')
    return p.parse_args()


def load_sampler(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model(condition_channel=2)
    sampler = imf.Sampler(model, generator=None, batch_size=1)
    sampler.background_cutoff = args.bg
    sampler.maximum_cutoff = args.mx
    sampler.normalize_factor = 'equation'
    sampler.histogram_equalization = args.he
    if args.he:
        hd = os.path.join(os.path.dirname(_HERE), 'help_data/histogram_equalization')
        sampler.bins = np.load(os.path.join(hd, 'bins.npy'))
        sampler.bins_mapped = np.load(os.path.join(hd, 'bins_mapped.npy'))
    else:
        sampler.bins = sampler.bins_mapped = None
    data = torch.load(args.ckpt, map_location=sampler.device)
    # ImprovedMeanFlow's own state_dict is already 'wrapped_model.base_unet.*', the same names the
    # GANTrainer saved, so both entries load strictly.
    sampler.model.load_state_dict(data['model'])
    sampler.ema.load_state_dict(data['ema'])
    if args.weights == 'ema':
        sampler.model = sampler.ema.ema_model
    sampler.model.eval()
    sampler.slice_batch = args.slice_batch
    print(f"checkpoint {args.ckpt} | step {data.get('step')} | weights={args.weights} | "
          f"norm [{args.bg:.0f},{args.mx:.0f}] he={args.he}", flush=True)
    return sampler


def cnr_stats(vol, gm, wm):
    x = np.clip(vol, 0, 100)
    g, w = x[gm], x[wm]
    return g.mean(), w.mean(), (g.mean() - w.mean()) / np.sqrt(g.std() ** 2 + w.std() ** 2)


def main():
    args = get_args()
    name = f'imf_gan_unsupervised_PCCT_epoch48_nfe{args.nfe}' + ('_raw' if args.weights == 'raw' else '')
    out = args.out or os.path.join(RESULTS, 'imf', 'gan', name)
    work = args.work or os.path.join(_BASE, 'projects/denoising/models/pcct_samples', os.path.basename(out.rstrip('/\\')))
    os.makedirs(out, exist_ok=True); os.makedirs(work, exist_ok=True)
    print('out :', out); print('work:', work, flush=True)
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed)

    sampler = load_sampler(args)

    for case in args.cases:
        t_case = time.time()
        src = os.path.join(args.data, str(case), 'soft_thins_0_noblank_sliced.nii.gz')
        if not os.path.isfile(src):
            print(f'[skip] case {case}: missing {src}', flush=True); continue
        img = nb.load(src)
        vol = np.asarray(img.get_fdata(), dtype=np.float32)          # (512, 512, 50) HU
        n = vol.shape[2]
        case_work = os.path.join(work, str(case)); os.makedirs(case_work, exist_ok=True)
        case_out = os.path.join(out, str(case)); os.makedirs(case_out, exist_ok=True)

        # padded copy so slices 0 and n-1 get a neighbour on both sides (replicated edge)
        padded_path = os.path.join(case_work, 'condition_padded.nii')
        if not os.path.isfile(padded_path):
            padded = np.concatenate([vol[:, :, :1], vol, vol[:, :, -1:]], axis=2)
            nb.save(nb.Nifti1Image(padded, img.affine), padded_path)
        generator = Generator.Dataset_2D(
            supervision='unsupervised',
            img_list=np.array([padded_path]), condition_list=np.array([padded_path]),
            image_size=[512, 512], num_slices_per_image=n, random_pick_slice=False,
            slice_range=[1, n + 1],                       # padded index p <-> original slice p-1
            histogram_equalization=args.he, bins=sampler.bins, bins_mapped=sampler.bins_mapped,
            background_cutoff=args.bg, maximum_cutoff=args.mx, normalize_factor='equation',
            shuffle=False, augment=False)
        sampler.generator = generator

        # K samples, resumable
        samples = []
        for i in range(1, args.k + 1):
            sp = os.path.join(case_work, f'sample_{i:02d}.nii.gz')
            if os.path.isfile(sp):
                try:
                    samples.append(np.asarray(nb.load(sp).dataobj, dtype=np.float32)); continue
                except Exception as e:                      # half-written file from an interrupted run
                    print(f'  [corrupt] {sp}: {str(e)[:60]} -> regenerating', flush=True); os.remove(sp)
            t0 = time.time()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                pred = sampler.sample_2D(args.ckpt, vol, direct_use_of_model=True,
                                         num_steps=args.nfe, solver='euler', schedule='uniform')
            pred = pred.astype(np.float32)
            _save_nifti(pred, img.affine, sp)
            samples.append(pred)
            print(f'  case {case} sample {i}/{args.k}  {time.time() - t0:.1f}s', flush=True)

        stack = np.stack(samples, axis=-1)
        for k in sorted(set(args.k_save)):
            if k > stack.shape[-1]:
                print(f'  [warn] K={k} requested but only {stack.shape[-1]} samples', flush=True); continue
            _save_nifti(stack[..., :k].mean(axis=-1), img.affine, os.path.join(case_out, f'pred_img_scans{k}.nii.gz'))
        _save_nifti(vol, img.affine, os.path.join(case_out, 'condition_img.nii.gz'))
        for roi in ('GM_ROI.nii.gz', 'WM_ROI.nii.gz'):
            s = os.path.join(args.roi_from, str(case), roi)
            if os.path.isfile(s):
                shutil.copyfile(s, os.path.join(case_out, roi))
        print(f'case {case} done: {stack.shape[-1]} samples, {(time.time() - t_case) / 60:.1f} min', flush=True)

        if args.compare_to:
            ref_p = os.path.join(args.compare_to, str(case), 'pred_img_scans20.nii.gz')
            if os.path.isfile(ref_p):
                ref = np.asarray(nb.load(ref_p).dataobj, dtype=np.float32)
                mine = stack.mean(axis=-1)
                body = (vol > -500)                                   # ignore air
                mae_slice = [np.abs(mine[:, :, s] - ref[:, :, s])[body[:, :, s]].mean() for s in range(n)]
                print(f'  vs {args.compare_to}: K={stack.shape[-1]} vs 20 | body MAE: interior {np.mean(mae_slice[1:-1]):.2f} HU, '
                      f'slice0 {mae_slice[0]:.2f}, slice{n - 1} {mae_slice[-1]:.2f} | bias {(mine - ref)[body].mean():+.2f}')
                gm_p, wm_p = (os.path.join(case_out, r) for r in ('GM_ROI.nii.gz', 'WM_ROI.nii.gz'))
                if os.path.isfile(gm_p) and os.path.isfile(wm_p):
                    gm = np.round(np.asarray(nb.load(gm_p).dataobj)).astype(bool)
                    wm = np.round(np.asarray(nb.load(wm_p).dataobj)).astype(bool)
                    a, b = cnr_stats(mine, gm, wm), cnr_stats(ref, gm, wm)
                    print(f'  GM/WM mean  mine {a[0]:.2f}/{a[1]:.2f} (CNR {a[2]:.2f} at K={stack.shape[-1]})   '
                          f'ref {b[0]:.2f}/{b[1]:.2f} (CNR {b[2]:.2f} at K=20)', flush=True)


if __name__ == '__main__':
    main()
