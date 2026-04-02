"""
Train cDDPM and N2N baseline on natural images (BSD400) for denoising.

Usage:
    # cDDPM training
    python train_natural.py --method cddpm --sigma 25

    # N2N baseline
    python train_natural.py --method n2n --sigma 25
"""
import sys
sys.path.append('/host/c/Users/ROG/Documents/GitHub')
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import IMF_denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion as ddpm
import IMF_denoising.functions_collection as ff
from IMF_denoising.natural_image_experiments.Generator_natural import NaturalImageDataset


def get_args():
    parser = argparse.ArgumentParser('Natural Image Denoising')
    parser.add_argument('--method', type=str, required=True, choices=['cddpm', 'n2n'])
    parser.add_argument('--sigma', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str,
                        default='/host/c/Users/ROG/Documents/GitHub/IMF_denoising/natural_image_experiments/data/denoising-datasets/BSD400')
    parser.add_argument('--save_dir', type=str,
                        default='/host/d/projects/denoising/models/natural_image')
    return parser.parse_args()


def train_cddpm(args):
    """Train conditional DDPM with N2N pairs."""
    trial_name = f'cddpm_n2n_sigma{args.sigma}'

    # Dataset
    dataset_train = NaturalImageDataset(
        image_dir=args.data_dir,
        noise_sigma=args.sigma,
        patch_size=args.patch_size,
        num_patches_per_image=8,
        augment=True,
        mode='train',
    )

    dataset_val = NaturalImageDataset(
        image_dir=args.data_dir.replace('BSD400', 'BSD68/original'),
        noise_sigma=args.sigma,
        patch_size=args.patch_size,
        num_patches_per_image=1,
        augment=False,
        mode='train',  # still use patches for val loss
    )

    # Model: same U-Net as CT experiments
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
        image_size=[args.patch_size, args.patch_size],
        timesteps=1000,
        sampling_timesteps=50,
        objective='pred_x0',
        clip_or_not=False,
        auto_normalize=False,
    )

    # Training
    save_dir = os.path.join(args.save_dir, trial_name, 'models')
    ff.make_folder([os.path.dirname(save_dir), save_dir])

    trainer = ddpm.Trainer(
        diffusion_model=diffusion_model,
        generator_train=dataset_train,
        generator_val=dataset_val,
        train_batch_size=args.batch_size,
        accum_iter=1,
        train_num_steps=args.epochs,
        results_folder=save_dir,
        train_lr=args.lr,
        train_lr_decay_every=args.epochs,
        save_models_every=10,
        validation_every=10,
    )

    trainer.train(pre_trained_model=None, start_step=0, beta=0, lpips_weight=0, edge_weight=0)
    print(f'cDDPM training complete. Models saved to {save_dir}')


def train_n2n(args):
    """Train direct N2N regression baseline (same U-Net, no diffusion)."""
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    from tqdm import tqdm

    trial_name = f'n2n_baseline_sigma{args.sigma}'

    dataset_train = NaturalImageDataset(
        image_dir=args.data_dir,
        noise_sigma=args.sigma,
        patch_size=args.patch_size,
        num_patches_per_image=8,
        augment=True,
        mode='train',
    )

    dataset_val = NaturalImageDataset(
        image_dir=args.data_dir.replace('BSD400', 'BSD68/original'),
        noise_sigma=args.sigma,
        patch_size=args.patch_size,
        num_patches_per_image=1,
        augment=False,
        mode='train',
    )

    # Same U-Net architecture as cDDPM (identical parameters for fair comparison)
    model = ddpm.Unet(
        problem_dimension='2D',
        init_dim=64,
        out_dim=1,
        channels=1,
        conditional_diffusion=True,  # Same architecture as cDDPM
        condition_channels=1,
        downsample_list=(True, True, True, False),
        upsample_list=(True, True, True, False),
        full_attn=(None, None, False, True),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.epochs, gamma=0.95)

    dl_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    save_dir = os.path.join(args.save_dir, trial_name, 'models')
    ff.make_folder([os.path.dirname(save_dir), save_dir])
    log_dir = os.path.join(args.save_dir, trial_name, 'log')
    ff.make_folder([log_dir])

    training_log = []
    best_val_loss = float('inf')

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        epoch_loss = []

        for batch in dl_train:
            x2, x1 = batch  # x2=target, x1=condition
            x1 = x1.to(device)
            x2 = x2.to(device)

            # N2N regression: predict x2 from x1
            # Use x1 as both input and condition (same architecture as cDDPM)
            dummy_time = torch.zeros(x1.shape[0], device=device)
            pred = model(x1, dummy_time, x1)

            loss = nn.functional.mse_loss(pred, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)

        # Validation
        val_loss = float('inf')
        if epoch % 10 == 0:
            model.eval()
            vl = []
            with torch.no_grad():
                for vbatch in dl_val:
                    vx2, vx1 = vbatch
                    vx1 = vx1.to(device)
                    vx2 = vx2.to(device)
                    dummy_time = torch.zeros(vx1.shape[0], device=device)
                    vpred = model(vx1, dummy_time, vx1)
                    vl.append(nn.functional.mse_loss(vpred, vx2).item())
            val_loss = np.mean(vl)
            print(f'Epoch {epoch} | train_loss={avg_loss:.6f} | val_loss={val_loss:.6f}')

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_dir, f'model-{epoch}.pt'))

        training_log.append([epoch, avg_loss, val_loss])
        scheduler.step()

    # Save final log
    import pandas as pd
    df = pd.DataFrame(training_log, columns=['epoch', 'train_loss', 'val_loss'])
    df.to_excel(os.path.join(log_dir, 'training_log.xlsx'), index=False)
    print(f'N2N training complete. Models saved to {save_dir}')


if __name__ == '__main__':
    args = get_args()
    if args.method == 'cddpm':
        train_cddpm(args)
    elif args.method == 'n2n':
        train_n2n(args)
