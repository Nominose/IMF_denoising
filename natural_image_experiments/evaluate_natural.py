"""
Evaluate cDDPM (various NFE) and N2N baseline on BSD68.

Usage:
    # cDDPM with different NFE (DDIM sampling)
    python evaluate_natural.py --method cddpm --sigma 50 --epoch 200 --nfe 1
    python evaluate_natural.py --method cddpm --sigma 50 --epoch 200 --nfe 2
    python evaluate_natural.py --method cddpm --sigma 50 --epoch 200 --nfe 5
    python evaluate_natural.py --method cddpm --sigma 50 --epoch 200 --nfe 10
    python evaluate_natural.py --method cddpm --sigma 50 --epoch 200 --nfe 50

    # N2N baseline
    python evaluate_natural.py --method n2n --sigma 50 --epoch 200
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
    parser.add_argument('--method', type=str, required=True, choices=['cddpm', 'n2n', 'supervised'])
    parser.add_argument('--sigma', type=int, default=50)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--nfe', type=int, default=1, help='Number of DDIM steps (only for cddpm)')
    parser.add_argument('--test_dir', type=str,
                        default='/host/c/Users/ROG/Documents/GitHub/IMF_denoising/natural_image_experiments/data/denoising-datasets/BSD68/original')
    parser.add_argument('--save_dir', type=str,
                        default='/host/d/projects/denoising/models/natural_image')
    parser.add_argument('--save_images', action='store_true', help='Save denoised images')
    return parser.parse_args()


def load_test_images(test_dir):
    paths = sorted(glob.glob(os.path.join(test_dir, '*.png')))
    images = []
    names = []
    for p in paths:
        img = np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0
        images.append(img)
        names.append(os.path.basename(p))
    print(f'Loaded {len(images)} test images')
    return images, names


def compute_metrics(pred, clean, data_range=1.0):
    psnr = peak_signal_noise_ratio(clean, pred, data_range=data_range)
    ssim = structural_similarity(clean, pred, data_range=data_range)
    return psnr, ssim


def evaluate_cddpm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigma = args.sigma / 255.0

    trial_name = f'cddpm_n2n_sigma{args.sigma}'
    model_path = os.path.join(args.save_dir, trial_name, 'models', f'model-{args.epoch}.pt')

    # Build model (same as training)
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
        image_size=[256, 256],  # dummy, actual size determined by input
        timesteps=1000,
        sampling_timesteps=args.nfe,
        objective='pred_x0',
        clip_or_not=False,
        auto_normalize=False,
        ddim_sampling_eta=0.0,  # deterministic DDIM
        force_ddim=True,
    )

    # Load checkpoint
    data = torch.load(model_path, map_location=device)
    diffusion_model.load_state_dict(data['model'])
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()

    test_images, names = load_test_images(args.test_dir)

    # Output dir
    result_dir = os.path.join(args.save_dir, trial_name, f'results_nfe{args.nfe}')
    os.makedirs(result_dir, exist_ok=True)

    results = []

    with torch.inference_mode():
        for i, clean in enumerate(test_images):
            h, w = clean.shape

            # Generate noisy input (fixed seed for reproducibility)
            np.random.seed(i + 1000)
            n1 = np.random.randn(h, w).astype(np.float32) * sigma
            noisy = clean + n1

            # Normalize to [-1, 1]
            noisy_norm = (noisy * 2.0 - 1.0).astype(np.float32)
            cond = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device)

            # DDIM sampling: pass condition, sample from noise
            shape = (1, 1, h, w)
            pred_norm = diffusion_model.ddim_sample(shape, condition=cond)

            # Denormalize back to [0,1]
            pred = (pred_norm[0, 0].cpu().numpy() + 1.0) / 2.0
            pred = np.clip(pred, 0, 1)

            # Metrics vs clean
            psnr_pred, ssim_pred = compute_metrics(pred, clean)
            psnr_noisy, ssim_noisy = compute_metrics(np.clip(noisy, 0, 1), clean)

            results.append({
                'image': names[i],
                'psnr_denoised': psnr_pred,
                'ssim_denoised': ssim_pred,
                'psnr_noisy': psnr_noisy,
                'ssim_noisy': ssim_noisy,
            })

            if i % 10 == 0:
                print(f'{names[i]}: PSNR={psnr_pred:.2f} (noisy: {psnr_noisy:.2f}), SSIM={ssim_pred:.4f}')

            # Save images
            if args.save_images:
                Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(result_dir, f'denoised_{names[i]}'))
                if i == 0:
                    Image.fromarray((np.clip(noisy, 0, 1) * 255).astype(np.uint8)).save(os.path.join(result_dir, f'noisy_{names[i]}'))
                    Image.fromarray((clean * 255).astype(np.uint8)).save(os.path.join(result_dir, f'clean_{names[i]}'))

    df = pd.DataFrame(results)
    print(f'\n=== cDDPM NFE={args.nfe} sigma={args.sigma} ===')
    print(f'PSNR: {df["psnr_denoised"].mean():.2f} +/- {df["psnr_denoised"].std():.2f}')
    print(f'SSIM: {df["ssim_denoised"].mean():.4f} +/- {df["ssim_denoised"].std():.4f}')
    print(f'Noisy: PSNR={df["psnr_noisy"].mean():.2f}, SSIM={df["ssim_noisy"].mean():.4f}')

    df.to_excel(os.path.join(result_dir, 'metrics.xlsx'), index=False)
    print(f'Results saved to {result_dir}')


def evaluate_n2n(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigma = args.sigma / 255.0

    trial_name = f'n2n_baseline_sigma{args.sigma}'
    model_path = os.path.join(args.save_dir, trial_name, 'models', f'model-{args.epoch}.pt')

    # Same architecture as cDDPM
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

    test_images, names = load_test_images(args.test_dir)

    result_dir = os.path.join(args.save_dir, trial_name, 'results')
    os.makedirs(result_dir, exist_ok=True)

    results = []

    with torch.inference_mode():
        for i, clean in enumerate(test_images):
            h, w = clean.shape

            np.random.seed(i + 1000)
            n1 = np.random.randn(h, w).astype(np.float32) * sigma
            noisy = clean + n1

            noisy_norm = (noisy * 2.0 - 1.0).astype(np.float32)
            x = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device)

            dummy_time = torch.zeros(1, device=device)
            pred_norm = model(x, dummy_time, x)  # x as both input and condition

            pred = (pred_norm[0, 0].cpu().numpy() + 1.0) / 2.0
            pred = np.clip(pred, 0, 1)

            # Debug: check value ranges
            if i == 0:
                print(f'DEBUG clean: min={clean.min():.4f} max={clean.max():.4f} shape={clean.shape}')
                print(f'DEBUG noisy: min={noisy.min():.4f} max={noisy.max():.4f}')
                print(f'DEBUG pred_norm: min={pred_norm[0,0].min().item():.4f} max={pred_norm[0,0].max().item():.4f}')
                print(f'DEBUG pred (after denorm+clip): min={pred.min():.4f} max={pred.max():.4f}')
                print(f'DEBUG MSE(pred, clean)={np.mean((pred - clean)**2):.6f}')
                print(f'DEBUG MSE(noisy_clip, clean)={np.mean((np.clip(noisy,0,1) - clean)**2):.6f}')

            psnr_pred, ssim_pred = compute_metrics(pred, clean)
            psnr_noisy, ssim_noisy = compute_metrics(np.clip(noisy, 0, 1), clean)

            results.append({
                'image': names[i],
                'psnr_denoised': psnr_pred,
                'ssim_denoised': ssim_pred,
                'psnr_noisy': psnr_noisy,
                'ssim_noisy': ssim_noisy,
            })

            if i % 10 == 0:
                print(f'{names[i]}: PSNR={psnr_pred:.2f} (noisy: {psnr_noisy:.2f}), SSIM={ssim_pred:.4f}')

            if args.save_images:
                Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(result_dir, f'denoised_{names[i]}'))

    df = pd.DataFrame(results)
    print(f'\n=== N2N Baseline sigma={args.sigma} ===')
    print(f'PSNR: {df["psnr_denoised"].mean():.2f} +/- {df["psnr_denoised"].std():.2f}')
    print(f'SSIM: {df["ssim_denoised"].mean():.4f} +/- {df["ssim_denoised"].std():.4f}')

    df.to_excel(os.path.join(result_dir, 'metrics.xlsx'), index=False)
    print(f'Results saved to {result_dir}')


def evaluate_supervised(args):
    """Evaluate supervised baseline — same code as N2N eval, different trial_name."""
    args_copy = argparse.Namespace(**vars(args))
    # Temporarily override to reuse n2n eval logic (same architecture, same inference)
    original_method = args.method
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sigma = args.sigma / 255.0

    trial_name = f'supervised_sigma{args.sigma}'
    model_path = os.path.join(args.save_dir, trial_name, 'models', f'model-{args.epoch}.pt')

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

    test_images, names = load_test_images(args.test_dir)
    result_dir = os.path.join(args.save_dir, trial_name, 'results')
    os.makedirs(result_dir, exist_ok=True)

    results = []

    with torch.inference_mode():
        for i, clean in enumerate(test_images):
            h, w = clean.shape
            np.random.seed(i + 1000)
            n1 = np.random.randn(h, w).astype(np.float32) * sigma
            noisy = clean + n1

            noisy_norm = (noisy * 2.0 - 1.0).astype(np.float32)
            x = torch.from_numpy(noisy_norm).unsqueeze(0).unsqueeze(0).to(device)

            dummy_time = torch.zeros(1, device=device)
            pred_norm = model(x, dummy_time, x)

            pred = (pred_norm[0, 0].cpu().numpy() + 1.0) / 2.0
            pred = np.clip(pred, 0, 1)

            psnr_pred, ssim_pred = compute_metrics(pred, clean)
            psnr_noisy, ssim_noisy = compute_metrics(np.clip(noisy, 0, 1), clean)

            results.append({
                'image': names[i],
                'psnr_denoised': psnr_pred,
                'ssim_denoised': ssim_pred,
                'psnr_noisy': psnr_noisy,
                'ssim_noisy': ssim_noisy,
            })

            if i % 10 == 0:
                print(f'{names[i]}: PSNR={psnr_pred:.2f} (noisy: {psnr_noisy:.2f}), SSIM={ssim_pred:.4f}')

    df = pd.DataFrame(results)
    print(f'\n=== Supervised sigma={args.sigma} ===')
    print(f'PSNR: {df["psnr_denoised"].mean():.2f} +/- {df["psnr_denoised"].std():.2f}')
    print(f'SSIM: {df["ssim_denoised"].mean():.4f} +/- {df["ssim_denoised"].std():.4f}')

    df.to_excel(os.path.join(result_dir, 'metrics.xlsx'), index=False)


if __name__ == '__main__':
    args = get_args()
    if args.method == 'cddpm':
        evaluate_cddpm(args)
    elif args.method == 'n2n':
        evaluate_n2n(args)
    elif args.method == 'supervised':
        evaluate_supervised(args)
