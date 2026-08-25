"""
data_loader.py  —  Dataset classes and data utilities for Cycle-CNN SR training.

Supports:
  - UnpairedSRDataset: loads HR + LR images from disk (any image dataset)
  - SyntheticSRDataset: generates smooth random patches for quick demo/testing
  - get_dataloaders(): convenience factory

Paper: Wang et al., IGARSS 2019 — Unsupervised Remote Sensing Image SR Using Cycle CNN
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# ── Colour-space helpers ───────────────────────────────────────────────────────

def rgb_to_y_channel(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to the Y channel of YCbCr (float32, 0–255)."""
    if image.mode == 'L':
        return np.array(image, dtype=np.float32)
    ycbcr = image.convert('YCbCr')
    y, _, _ = ycbcr.split()
    return np.array(y, dtype=np.float32)


def load_y_channel(path: str) -> np.ndarray:
    """Load an image file and return its Y channel as float32 numpy array."""
    img = Image.open(path).convert('RGB')
    return rgb_to_y_channel(img)


# ── Patch-cropping helpers ─────────────────────────────────────────────────────

def _ensure_min_size(img: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    h, w = img.shape
    if h < min_h or w < min_w:
        new_w = max(w, min_w)
        new_h = max(h, min_h)
        img = np.array(
            Image.fromarray(img.astype(np.uint8)).resize((new_w, new_h), Image.BICUBIC),
            dtype=np.float32,
        )
    return img


def _random_crop(img: np.ndarray, patch_size: int) -> torch.Tensor:
    img = _ensure_min_size(img, patch_size, patch_size)
    h, w = img.shape
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    patch = img[top : top + patch_size, left : left + patch_size]
    return torch.from_numpy(patch).unsqueeze(0)  # (1, P, P)


# ── Datasets ───────────────────────────────────────────────────────────────────

class UnpairedSRDataset(Dataset):
    """
    Unpaired super-resolution dataset for Cycle-CNN training.

    HR images are expected to be ~1024×1024 (PAN equivalent).
    LR images are expected to be ~256×256 (MS Y-channel equivalent).
    If lr_dir is None, LR patches are derived by 4× bicubic downsampling
    of a *different* HR image (unpaired by design).

    At each __getitem__ call:
      - HR patch: random 384×384 crop from an HR image
      - LR patch: random  96×96 crop from a different LR image (unpaired)
    """

    _EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str | None = None,
        hr_patch_size: int = 384,
        lr_patch_size: int = 96,
        mode: str = 'train',
        test_size: int = 8,
    ):
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = lr_patch_size
        self.mode = mode

        all_hr = sorted(
            str(p) for p in Path(hr_dir).rglob('*') if p.suffix.lower() in self._EXTS
        )
        if not all_hr:
            raise FileNotFoundError(f'No images found in HR directory: {hr_dir}')

        if mode == 'train':
            self.hr_paths = all_hr[:-test_size] if len(all_hr) > test_size else all_hr
        else:
            self.hr_paths = all_hr[-test_size:] if len(all_hr) >= test_size else all_hr

        if lr_dir is not None:
            all_lr = sorted(
                str(p) for p in Path(lr_dir).rglob('*') if p.suffix.lower() in self._EXTS
            )
            if mode == 'train':
                self.lr_paths: list[str] | None = (
                    all_lr[:-test_size] if len(all_lr) > test_size else all_lr
                )
            else:
                self.lr_paths = all_lr[-test_size:] if len(all_lr) >= test_size else all_lr
        else:
            self.lr_paths = None  # derive LR from HR via bicubic downsample

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _get_lr_patch(self, current_hr_idx: int) -> torch.Tensor:
        """Return a random unpaired LR patch."""
        if self.lr_paths is not None:
            lr_idx = random.randint(0, len(self.lr_paths) - 1)
            lr_img = load_y_channel(self.lr_paths[lr_idx])
        else:
            # Pick a *different* HR image and bicubic-downsample it (unpaired)
            lr_idx = random.choice(
                [i for i in range(len(self.hr_paths)) if i != current_hr_idx]
                or [current_hr_idx]
            )
            hr_img = load_y_channel(self.hr_paths[lr_idx])
            h, w = hr_img.shape
            lr_img = np.array(
                Image.fromarray(hr_img.astype(np.uint8)).resize(
                    (w // 4, h // 4), Image.BICUBIC
                ),
                dtype=np.float32,
            )
        return _random_crop(lr_img, self.lr_patch_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_patch = _random_crop(load_y_channel(self.hr_paths[idx]), self.hr_patch_size)
        lr_patch = self._get_lr_patch(idx)
        return lr_patch, hr_patch  # (1,96,96), (1,384,384)


class SyntheticSRDataset(Dataset):
    """
    Synthetic dataset for quick demo and unit testing.

    Generates smooth random-noise images using Gaussian blur so the patches
    look like diffuse textures rather than pure white noise.
    LR and HR images come from *separate* random bases to simulate unpaired data.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        hr_patch_size: int = 384,
        lr_patch_size: int = 96,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = lr_patch_size

        rng = np.random.RandomState(seed)
        n_base = 50
        # Create smooth base images by blurring random noise
        from scipy.ndimage import gaussian_filter as gf

        self._hr_bases = [
            gf(rng.uniform(0, 255, (hr_patch_size * 2, hr_patch_size * 2)).astype(np.float32), sigma=8)
            for _ in range(n_base)
        ]
        self._lr_bases = [
            gf(rng.uniform(0, 255, (lr_patch_size * 2, lr_patch_size * 2)).astype(np.float32), sigma=3)
            for _ in range(n_base)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr_base = self._hr_bases[idx % len(self._hr_bases)]
        lr_base = self._lr_bases[(idx + 7) % len(self._lr_bases)]  # offset → unpaired

        hr_patch = _random_crop(hr_base, self.hr_patch_size)
        lr_patch = _random_crop(lr_base, self.lr_patch_size)
        return lr_patch, hr_patch


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders(
    hr_dir: str | None = None,
    lr_dir: str | None = None,
    batch_size: int = 8,
    num_workers: int = 2,
    synthetic: bool = False,
    num_synthetic_train: int = 5600,
    num_synthetic_test: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """
    Factory function to build train and test DataLoaders.

    Args:
        hr_dir:               Path to the HR image directory.
        lr_dir:               Path to the LR image directory (optional).
        batch_size:           Minibatch size (paper: 8).
        num_workers:          DataLoader worker processes.
        synthetic:            Use SyntheticSRDataset (demo/testing).
        num_synthetic_train:  Training samples for synthetic mode.
        num_synthetic_test:   Test samples for synthetic mode.

    Returns:
        (train_loader, test_loader)
    """
    if synthetic or hr_dir is None:
        train_ds: Dataset = SyntheticSRDataset(num_samples=num_synthetic_train)
        test_ds: Dataset = SyntheticSRDataset(num_samples=num_synthetic_test, seed=99)
    else:
        train_ds = UnpairedSRDataset(hr_dir, lr_dir, mode='train')
        test_ds = UnpairedSRDataset(hr_dir, lr_dir, mode='test')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
