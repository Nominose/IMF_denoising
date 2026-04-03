"""
Dataset class for natural image denoising with synthetic Gaussian noise.
Constructs N2N pairs by adding independent noise to the same clean image.

Usage:
    dataset = NaturalImageDataset(
        image_dir='data/denoising-datasets/BSD400',
        noise_sigma=25,
        patch_size=128,
        num_patches_per_image=8,
    )
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob


class NaturalImageDataset(Dataset):
    def __init__(
        self,
        image_dir,
        noise_sigma=25,
        patch_size=128,
        num_patches_per_image=8,
        augment=True,
        mode='train',  # 'train' or 'test'
        split=None,  # None=all, 'train'=first 360, 'val'=last 40 (for BSD400)
        supervised=False,  # if True, target is clean (no noise on target)
    ):
        self.noise_sigma = noise_sigma / 255.0  # normalize to [0,1] range
        self.supervised = supervised
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.augment = augment and (mode == 'train')
        self.mode = mode

        # Load all images
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self.image_paths.sort()

        # Split if requested
        if split == 'val':
            self.image_paths = self.image_paths[-40:]
        elif split == 'train':
            self.image_paths = self.image_paths[:360]

        assert len(self.image_paths) > 0, f"No images found in {image_dir}"
        print(f"Loaded {len(self.image_paths)} images from {image_dir}, mode={mode}, split={split}")

        # Preload images
        self.images = []
        for p in self.image_paths:
            img = np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0  # grayscale, [0,1]
            self.images.append(img)

    def __len__(self):
        if self.mode == 'train':
            return len(self.images) * self.num_patches_per_image
        else:
            return len(self.images)

    def _random_crop(self, img, size):
        h, w = img.shape
        if h < size or w < size:
            # Pad if needed
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape
        top = np.random.randint(0, h - size + 1)
        left = np.random.randint(0, w - size + 1)
        return img[top:top+size, left:left+size]

    def _augment(self, img):
        # Random flip and rotation
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
        k = np.random.randint(0, 4)
        img = np.rot90(img, k).copy()
        return img

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_idx = idx // self.num_patches_per_image
            img = self.images[img_idx]

            # Random crop
            patch = self._random_crop(img, self.patch_size)

            # Augment
            if self.augment:
                patch = self._augment(patch)

            # Generate noisy pair
            n1 = np.random.randn(*patch.shape).astype(np.float32) * self.noise_sigma
            x1 = patch + n1  # condition (always noisy)

            if self.supervised:
                x2 = patch.copy()  # target is clean
            else:
                n2 = np.random.randn(*patch.shape).astype(np.float32) * self.noise_sigma
                x2 = patch + n2  # target is noisy (N2N)

            # Normalize to [-1, 1] for diffusion model
            x1 = (x1 * 2.0 - 1.0).astype(np.float32)
            x2 = (x2 * 2.0 - 1.0).astype(np.float32)

            # Add channel dimension [1, H, W]
            x2_tensor = torch.from_numpy(x2).unsqueeze(0)
            x1_tensor = torch.from_numpy(x1).unsqueeze(0)

            return x2_tensor, x1_tensor

        else:
            # Test mode: return full image with noise
            img = self.images[idx]

            n1 = np.random.randn(*img.shape).astype(np.float32) * self.noise_sigma
            n2 = np.random.randn(*img.shape).astype(np.float32) * self.noise_sigma

            x1 = img + n1
            x2 = img + n2

            x1 = (x1 * 2.0 - 1.0).astype(np.float32)
            x2 = (x2 * 2.0 - 1.0).astype(np.float32)
            clean = (img * 2.0 - 1.0).astype(np.float32)

            x2_tensor = torch.from_numpy(x2).unsqueeze(0)
            x1_tensor = torch.from_numpy(x1).unsqueeze(0)
            clean_tensor = torch.from_numpy(clean).unsqueeze(0)

            return x2_tensor, x1_tensor, clean_tensor

    def on_epoch_end(self):
        """Called at the end of each epoch (compatibility with existing Trainer)."""
        pass


class N2NRegressionDataset(Dataset):
    """Direct N2N regression dataset (baseline comparison).
    Same data as NaturalImageDataset but without diffusion timestep structure.
    """
    def __init__(
        self,
        image_dir,
        noise_sigma=25,
        patch_size=128,
        num_patches_per_image=8,
        augment=True,
        mode='train',
    ):
        self.inner = NaturalImageDataset(
            image_dir=image_dir,
            noise_sigma=noise_sigma,
            patch_size=patch_size,
            num_patches_per_image=num_patches_per_image,
            augment=augment,
            mode=mode,
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self.inner[idx]

    def on_epoch_end(self):
        pass
