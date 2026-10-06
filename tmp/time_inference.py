"""Wall-clock inference time per generated output for the samplers compared in the paper.

One 512x512 slice, batch size 1, two condition channels (the thin-slice brain-CT setting), fp32, on
whatever GPU is visible. All three models use the same conditional U-Net backbone (35.7 M parameters;
iMF adds one 1x1 output head), so the cost of a generated output is essentially NFE x one network
evaluation; this script measures it end to end instead of assuming it:

    DDIM            conditional DDPM sampled with DDIM, NFE = 50 (the Noise2Noise diffusion setting)
    flow matching   Euler, NFE = N
    iMF             average-velocity updates, NFE = N

Weights are randomly initialised -- timing does not depend on them. Each configuration is warmed up,
then timed --reps times with torch.cuda.synchronize() on both sides; the median is reported.
Run inside the docker env:
    python tmp/time_inference.py --out /host/d/research/projects/denoising/results/running_time.csv
"""
import argparse
import csv
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO_PARENT, '/host/c/Users/ROG/Documents/GitHub'):
    if _p not in sys.path:
        sys.path.append(_p)

import torch

import IMF_denoising.improved_mean_flow as imf
import IMF_denoising.conditional_flow_matching as fm
import IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm

UNET = dict(problem_dimension='2D', init_dim=64, out_dim=1, channels=1, conditional_diffusion=True,
            condition_channels=2, downsample_list=(True, True, True, False),
            upsample_list=(True, True, True, False), full_attn=(None, None, False, True))


def timed(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--batch', type=int, default=1, help='slices per forward')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'needs a GPU'
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    gpu = torch.cuda.get_device_name(0)
    cond = torch.randn(a.batch, 2, 512, 512, device=dev)
    rows = []

    def add(method, nfe, fn, reps):
        med, lo, hi = timed(fn, reps)
        rows.append(dict(method=method, nfe=nfe, batch=a.batch, sec_per_output=med / a.batch, ms_per_nfe=1000 * med / a.batch / nfe,
                         min_s=lo / a.batch, max_s=hi / a.batch, gpu=gpu))
        print(f'{method:<14} NFE={nfe:<3} {med / a.batch:7.3f} s per output per slice   ({1000 * med / a.batch / nfe:6.1f} ms per network evaluation)', flush=True)

    # ---- iMF -------------------------------------------------------------------------------
    m = imf.ImprovedMeanFlow(ddpm.Unet(auxiliary_v_head=True, **UNET), image_size=[512, 512], ratio_r_neq_t=0.5,
                             clip_or_not=False, auto_normalize=False).to(dev).eval()
    n_imf = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.inference_mode():
        for nfe in (1, 2, 3, 5, 10):
            fn = (lambda: m.sample(condition=cond, batch_size=a.batch)) if nfe == 1 else \
                 (lambda nfe=nfe: m.sample_multistep(condition=cond, batch_size=a.batch, num_steps=nfe, solver='euler', schedule='uniform'))
            add('iMF', nfe, fn, a.reps)
    del m; torch.cuda.empty_cache()

    # ---- flow matching (Euler) ---------------------------------------------------------------
    with torch.inference_mode():
        for nfe in (3, 50):
            f = fm.ConditionalFlowMatching(ddpm.Unet(**UNET), image_size=[512, 512], sampling_timesteps=nfe).to(dev).eval()
            add('flow matching', nfe, lambda: f.sample(condition=cond, batch_size=a.batch), a.reps if nfe <= 5 else max(5, a.reps // 4))
            del f; torch.cuda.empty_cache()

    # ---- conditional diffusion with DDIM sampling -------------------------------------------
    for nfe in (50,):
        d = ddpm.GaussianDiffusion(ddpm.Unet(**UNET), image_size=[512, 512], timesteps=1000, sampling_timesteps=nfe,
                                   ddim_sampling_eta=1., force_ddim=False, auto_normalize=False, objective='pred_x0',
                                   clip_or_not=True, clip_range=[-1, 1]).to(dev).eval()
        n_ddim = sum(p.numel() for p in d.parameters()) / 1e6
        with torch.inference_mode():
            add('DDIM', nfe, lambda: d.sample(condition=cond, batch_size=a.batch), max(5, a.reps // 4))
        del d; torch.cuda.empty_cache()

    print(f'\nGPU: {gpu} | torch {torch.__version__} | fp32 | batch {a.batch} | parameters: iMF {n_imf:.2f} M, DDIM {n_ddim:.2f} M')
    ref = next(r for r in rows if r['method'] == 'DDIM')
    print('\nPer slice, K = 20 averaged outputs:')
    for r in rows:
        print(f"  {r['method']:<14} NFE={r['nfe']:<3} {20 * r['sec_per_output']:7.1f} s   ({ref['sec_per_output'] / r['sec_per_output']:5.1f}x faster than DDIM-50)")
    if a.out:
        with open(a.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
