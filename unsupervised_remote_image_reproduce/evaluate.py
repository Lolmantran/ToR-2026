"""
evaluate.py  —  Evaluation script for Cycle-CNN SR.

Replicates Table 1 from Wang et al., IGARSS 2019:
  Methods    : Bi-cubic  |  SRCNN  |  Cycle-CNN (ours)
  Conditions : None  |  Blur  |  Noise (σ=10)  |  Blur+Noise

Usage (CLI)
-----------
  python evaluate.py \\
      --checkpoint checkpoints/ckpt_iter_0180000.pth \\
      --test_hr_dir /data/GaoFen2/test/PAN \\
      --test_lr_dir /data/GaoFen2/test/MS_Y
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from data_loader import load_y_channel
from models import G1, SRCNN
from utils import load_checkpoint, tensor_to_numpy, visualize_patches


# ── Degradation helpers ────────────────────────────────────────────────────────

def add_gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian blur (σ=2 by default) to a float32 image."""
    return gaussian_filter(img.astype(np.float32), sigma=sigma)


def add_gaussian_noise(
    img: np.ndarray, sigma: float = 10.0, seed: int = 0
) -> np.ndarray:
    """Add zero-mean Gaussian white noise (σ=10) to a float32 image."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0.0, 255.0)


def apply_degradation(lr_np: np.ndarray, condition: str) -> np.ndarray:
    """
    Apply the requested degradation to a float32 LR array.

    condition : 'none' | 'blur' | 'noise' | 'blur_noise'
    """
    if condition == 'none':
        return lr_np.astype(np.float32)
    if condition == 'blur':
        return add_gaussian_blur(lr_np)
    if condition == 'noise':
        return add_gaussian_noise(lr_np)
    if condition == 'blur_noise':
        return add_gaussian_noise(add_gaussian_blur(lr_np))
    raise ValueError(f"Unknown condition '{condition}'. "
                     "Choose from: none, blur, noise, blur_noise")


# ── Bicubic baseline ───────────────────────────────────────────────────────────

def bicubic_upsample(lr_np: np.ndarray, scale: int = 4) -> np.ndarray:
    """Bicubic upsampling via PIL (float32 in/out, 0–255)."""
    h, w = lr_np.shape
    img = Image.fromarray(lr_np.astype(np.uint8))
    img_hr = img.resize((w * scale, h * scale), Image.BICUBIC)
    return np.array(img_hr, dtype=np.float32)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    sr: np.ndarray, hr: np.ndarray
) -> tuple[float, float]:
    """Return (PSNR_dB, SSIM) between SR and HR images (uint8 range 0-255)."""
    sr_u8 = np.clip(sr, 0, 255).astype(np.uint8)
    hr_u8 = np.clip(hr, 0, 255).astype(np.uint8)
    psnr_val = peak_signal_noise_ratio(hr_u8, sr_u8, data_range=255)
    ssim_val = structural_similarity(hr_u8, sr_u8, data_range=255)
    return float(psnr_val), float(ssim_val)


# ── Main evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_all(
    test_images: list[tuple[np.ndarray, np.ndarray]],
    g1_model: Optional[nn.Module] = None,
    srcnn_model: Optional[nn.Module] = None,
    device: torch.device = torch.device('cpu'),
    conditions: tuple[str, ...] = ('none', 'blur', 'noise', 'blur_noise'),
    scale: int = 4,
) -> dict:
    """
    Evaluate all methods under all degradation conditions.

    Args:
        test_images:  List of (lr_array, hr_array) tuples (float32, 0–255).
                      lr should be at 1/scale resolution of hr.
        g1_model:     Trained G1 (Cycle-CNN).
        srcnn_model:  Trained SRCNN baseline.
        device:       Torch device.
        conditions:   Degradation conditions to evaluate.
        scale:        Upscale factor (4 per paper).

    Returns:
        Nested dict: results[condition][method]['psnr' | 'ssim'] = list[float]
    """
    results: dict = {}

    for cond in conditions:
        results[cond] = {
            'bicubic':   {'psnr': [], 'ssim': []},
            'srcnn':     {'psnr': [], 'ssim': []},
            'cycle_cnn': {'psnr': [], 'ssim': []},
        }

        for lr_np, hr_np in test_images:
            lr_deg = apply_degradation(lr_np, cond)

            # ── Bi-cubic ──────────────────────────────────────────────────────
            bic_sr = bicubic_upsample(lr_deg, scale=scale)
            p, s = compute_metrics(bic_sr, hr_np)
            results[cond]['bicubic']['psnr'].append(p)
            results[cond]['bicubic']['ssim'].append(s)

            # ── SRCNN ─────────────────────────────────────────────────────────
            if srcnn_model is not None:
                bic_for_srcnn = bicubic_upsample(lr_deg, scale=scale)
                inp = (
                    torch.from_numpy(bic_for_srcnn)
                    .unsqueeze(0).unsqueeze(0)
                    .to(device)
                )
                out = srcnn_model(inp).squeeze().cpu().numpy()
                # Crop to HR size in case of padding differences
                out = out[: hr_np.shape[0], : hr_np.shape[1]]
                p, s = compute_metrics(out, hr_np)
                results[cond]['srcnn']['psnr'].append(p)
                results[cond]['srcnn']['ssim'].append(s)

            # ── Cycle-CNN ─────────────────────────────────────────────────────
            if g1_model is not None:
                inp = (
                    torch.from_numpy(lr_deg.copy())
                    .unsqueeze(0).unsqueeze(0)
                    .to(device)
                )
                # Pad to minimum 96×96 if patch is smaller
                _, _, h, w = inp.shape
                pad_h = max(0, 96 - h)
                pad_w = max(0, 96 - w)
                if pad_h or pad_w:
                    inp = F.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')

                out = g1_model(inp).squeeze().cpu().numpy()
                # Crop to expected HR size
                out = out[: hr_np.shape[0], : hr_np.shape[1]]
                p, s = compute_metrics(out, hr_np)
                results[cond]['cycle_cnn']['psnr'].append(p)
                results[cond]['cycle_cnn']['ssim'].append(s)

    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_results_table(results: dict) -> None:
    """
    Print an evaluation summary table in the style of Table 1 from the paper.
    """
    conditions = list(results.keys())
    method_order  = ['bicubic', 'srcnn', 'cycle_cnn']
    method_labels = {
        'bicubic':   'Bi-cubic',
        'srcnn':     'SRCNN',
        'cycle_cnn': 'Cycle-CNN (ours)',
    }
    cond_labels = {
        'none':       'None',
        'blur':       'Blur',
        'noise':      'Noise',
        'blur_noise': 'Blur+Noise',
    }

    col_w = 22
    header = (
        f"{'Degradation':<14}"
        f"{'Method':<{col_w}}"
        f"{'PSNR (dB)':>12}"
        f"{'SSIM':>10}"
    )
    sep = '─' * len(header)
    print(sep)
    print(header)
    print(sep)

    for cond in conditions:
        first = True
        for method in method_order:
            data = results[cond].get(method, {})
            psnr_list = data.get('psnr', [])
            ssim_list = data.get('ssim', [])
            if not psnr_list:
                continue
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            cond_str = cond_labels.get(cond, cond) if first else ''
            first = False
            print(
                f'{cond_str:<14}'
                f'{method_labels[method]:<{col_w}}'
                f'{avg_psnr:>12.2f}'
                f'{avg_ssim:>10.4f}'
            )
        print(sep)


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Evaluate Cycle-CNN SR — reproduces Table 1 metrics.'
    )
    p.add_argument('--checkpoint',   type=str, required=True,
                   help='Path to a .pth checkpoint file (contains G1 weights).')
    p.add_argument('--test_hr_dir',  type=str, required=True,
                   help='Directory of HR test images.')
    p.add_argument('--test_lr_dir',  type=str, default=None,
                   help='Directory of LR test images. If omitted, LR is '
                        'derived from HR by 4× bicubic downsampling.')
    p.add_argument('--scale',        type=int, default=4)
    p.add_argument('--visualize',    action='store_true', default=False,
                   help='Show visual comparison for the first test image.')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load G1 ───────────────────────────────────────────────────────────────
    g1 = G1(num_resblocks=16).to(device)
    load_checkpoint(g1, None, None, args.checkpoint, device)
    g1.eval()

    # ── Load test images ──────────────────────────────────────────────────────
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    hr_paths = sorted(
        str(p) for p in Path(args.test_hr_dir).rglob('*') if p.suffix.lower() in exts
    )[:8]

    if args.test_lr_dir:
        lr_paths = sorted(
            str(p) for p in Path(args.test_lr_dir).rglob('*') if p.suffix.lower() in exts
        )[:8]
    else:
        lr_paths = None

    test_images: list[tuple[np.ndarray, np.ndarray]] = []
    for i, hp in enumerate(hr_paths):
        hr_np = load_y_channel(hp)
        if lr_paths and i < len(lr_paths):
            lr_np = load_y_channel(lr_paths[i])
        else:
            h, w = hr_np.shape
            lr_np = np.array(
                Image.fromarray(hr_np.astype(np.uint8)).resize(
                    (w // args.scale, h // args.scale), Image.BICUBIC
                ),
                dtype=np.float32,
            )
        test_images.append((lr_np, hr_np))

    # ── SRCNN baseline ────────────────────────────────────────────────────────
    srcnn = SRCNN().to(device)
    srcnn.eval()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate_all(
        test_images,
        g1_model=g1,
        srcnn_model=srcnn,
        device=device,
        scale=args.scale,
    )
    print_results_table(results)

    # ── Optional visual comparison ────────────────────────────────────────────
    if args.visualize and test_images:
        import torch as _t
        lr_np, hr_np = test_images[0]
        lr_t  = _t.from_numpy(lr_np).unsqueeze(0).unsqueeze(0)
        hr_t  = _t.from_numpy(hr_np).unsqueeze(0).unsqueeze(0)
        inp = lr_t.to(device)
        _, _, h, w = inp.shape
        if h < 96 or w < 96:
            inp = F.pad(inp, (0, max(0, 96 - w), 0, max(0, 96 - h)), mode='reflect')
        with _t.no_grad():
            sr_t = g1(inp).cpu()
        visualize_patches(lr_t, sr_t, hr_t)
