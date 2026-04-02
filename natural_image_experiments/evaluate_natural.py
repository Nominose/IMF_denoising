"""
Evaluate cDDPM (various NFE) and N2N baseline on BSD68.

Usage:
    # cDDPM with different NFE
    python evaluate_natural.py --method cddpm --sigma 25 --epoch 200 --nfe 1
    python evaluate_natural.py --method cddpm --sigma 25 --epoch 200 --nfe 2
    python evaluate_natural.py --method cddpm --sigma 25 --epoch 200 --nfe 5
    python evaluate_natural.py --method cddpm --sigma 25 --epoch 200 --nfe 10
    python evaluate_natural.py --method cddpm --sigma 25 --epoch 200 --nfe 50

    # N2N baseline
    python evaluate_natural.py --method n2n --sigma 25 --epoch 200
"""
import sys
sys.path.append('/host/c/Users/ROG/Documents/GitHub')
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import pandas as pd
import glob

import IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm


def get_args():
    parser = argparse.ArgumentParser('Evaluate Natural Image Denoising')
    parser.add_argument('--method', type=str, required=True, choices=['cddpm', 'n2n'])
    parser.add_argument('--sigma', type=int, default=25)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--nfe', type=int, default=1, help='Number of DDIM steps (only for cddpm)')
    parser.add_argument('--test_dir', type=str,
                        default='/host/c/Users/ROG/Documents/GitHub/IMF_denoising/natural_image_experiments/data/denoising-datasets/BSD68/original')
    parser.add_argument('--save_dir', type=str,
                        default='/host/d/projects/denoising/models/natural_image')
    return parser.parse_args()


def load_test_images(test_dir):
    """Load BSD68 test images as grayscale numpy arrays in [0,1]."""
    paths = sorted(glob.glob(os.path.join(test_dir, '*.png')))
    images = []
    for p in paths:
        img = np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0
        images.append(img)
    print(f'Loaded {len(images)} test images')
    return images


def compute_metrics(pred, clean, data_range=1.0):
    """Compute PSNR and SSIM."""
    psnr = peak_signal_noise_ratio(clean, pred, data_range=data_range)
    ssim = structural_similarity(clean, pred, data_range=data_range)
    return psnr, ssim


def evaluate_cddpm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigma = args.sigma / 255.0

    trial_name = f'cddpm_n2n_sigma{args.sigma}'
    model_path = os.path.join(args.save_dir, trial_name, 'models', f'model-{args.epoch}.pt')

    # Build model
    model = ddpm.Unet(
        problem_dimension='2D',
        init_dim=64,
        out_dim=1,
        channels=1,
        conditional_diffusion=True,
        condition_channels=1,
        downsample_list=(True, True, True, False),
        upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True),
    )

    diffusion_model = ddpm.GaussianDiffusion(
        model,
        image_size=[256, 256],  # dummy, we use full images
        timesteps=1000,
        sampling_timesteps=args.nfe,
        objective='pred_x0',
        clip_or_not=False,
        auto_normalize=False,
    )

    # Load checkpoint
    data = torch.load(model_path, map_location=device)
    diffusion_model.load_state_dict(data['model'])
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()

    # Load test images
    test_images = load_test_images(args.test_dir)

    results = []
    noisy_results = []

    with torch.no_grad():
        for i, clean in enumerate(test_images):
            h, w = clean.shape

            # Generate noisy input
            np.random.seed(i)  # reproducible noise
            n1 = np.random.randn(h, w).astype(np.float32) * sigma
            noisy = clean + n1

            # Normalize to [-1, 1]
            noisy_norm = (noisy * 2.0 - 1.0).astype(np.float32)
            cond = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device)

            # DDIM sampling
            if args.nfe == 1:
                # For pred_x0, single step from T
                # Use the model's internal single-step prediction
                noise = torch.randn(1, 1, h, w, device=device)
                t = torch.full((1,), diffusion_model.num_timesteps - 1, device=device, dtype=torch.long)
                pred_norm = diffusion_model.model(
                    torch.cat([noise, cond], dim=1) if diffusion_model.model.conditional_diffusion
                    else noise, t, cond
                )
                # Actually, let's use the proper sampling interface
                pred_norm = diffusion_model.ddim_sample(cond=cond, shape=(1, 1, h, w))
            else:
                pred_norm = diffusion_model.ddim_sample(cond=cond, shape=(1, 1, h, w))

            # Denormalize back to [0,1]
            pred = (pred_norm[0, 0].cpu().numpy() + 1.0) / 2.0
            pred = np.clip(pred, 0, 1)

            # Metrics
            psnr, ssim = compute_metrics(pred, clean)
            psnr_noisy, ssim_noisy = compute_metrics(np.clip(noisy, 0, 1), clean)

            results.append({'image': i, 'psnr': psnr, 'ssim': ssim})
            noisy_results.append({'image': i, 'psnr': psnr_noisy, 'ssim': ssim_noisy})

            if i % 10 == 0:
                print(f'Image {i}: PSNR={psnr:.2f} (noisy: {psnr_noisy:.2f}), SSIM={ssim:.4f} (noisy: {ssim_noisy:.4f})')

    df = pd.DataFrame(results)
    print(f'\n=== cDDPM NFE={args.nfe} sigma={args.sigma} ===')
    print(f'PSNR: {df["psnr"].mean():.2f} +/- {df["psnr"].std():.2f}')
    print(f'SSIM: {df["ssim"].mean():.4f} +/- {df["ssim"].std():.4f}')

    df_noisy = pd.DataFrame(noisy_results)
    print(f'\nNoisy input:')
    print(f'PSNR: {df_noisy["psnr"].mean():.2f}, SSIM: {df_noisy["ssim"].mean():.4f}')

    # Save results
    result_dir = os.path.join(args.save_dir, trial_name, 'results')
    os.makedirs(result_dir, exist_ok=True)
    df.to_excel(os.path.join(result_dir, f'results_nfe{args.nfe}.xlsx'), index=False)


def evaluate_n2n(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigma = args.sigma / 255.0

    trial_name = f'n2n_baseline_sigma{args.sigma}'
    model_path = os.path.join(args.save_dir, trial_name, 'models', f'model-{args.epoch}.pt')

    # Build model (same architecture as cDDPM for fair comparison)
    model = ddpm.Unet(
        problem_dimension='2D',
        init_dim=64,
        out_dim=1,
        channels=1,
        conditional_diffusion=True,
        condition_channels=1,
        downsample_list=(True, True, True, False),
        upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True),
    )

    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data['model'])
    model = model.to(device)
    model.eval()

    test_images = load_test_images(args.test_dir)

    results = []

    with torch.no_grad():
        for i, clean in enumerate(test_images):
            h, w = clean.shape

            np.random.seed(i)
            n1 = np.random.randn(h, w).astype(np.float32) * sigma
            noisy = clean + n1

            noisy_norm = (noisy * 2.0 - 1.0).astype(np.float32)
            x = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device)

            dummy_time = torch.zeros(1, device=device)
            pred_norm = model(x, dummy_time, x)

            pred = (pred_norm[0, 0].cpu().numpy() + 1.0) / 2.0
            pred = np.clip(pred, 0, 1)

            psnr, ssim = compute_metrics(pred, clean)
            results.append({'image': i, 'psnr': psnr, 'ssim': ssim})

            if i % 10 == 0:
                print(f'Image {i}: PSNR={psnr:.2f}, SSIM={ssim:.4f}')

    df = pd.DataFrame(results)
    print(f'\n=== N2N Baseline sigma={args.sigma} ===')
    print(f'PSNR: {df["psnr"].mean():.2f} +/- {df["psnr"].std():.2f}')
    print(f'SSIM: {df["ssim"].mean():.4f} +/- {df["ssim"].std():.4f}')

    result_dir = os.path.join(args.save_dir, trial_name, 'results')
    os.makedirs(result_dir, exist_ok=True)
    df.to_excel(os.path.join(result_dir, f'results.xlsx'), index=False)


if __name__ == '__main__':
    args = get_args()
    if args.method == 'cddpm':
        evaluate_cddpm(args)
    elif args.method == 'n2n':
        evaluate_n2n(args)
