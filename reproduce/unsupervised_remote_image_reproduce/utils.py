"""
utils.py  —  Checkpoint I/O, metrics, and visualisation helpers for Cycle-CNN.
"""

import math
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ── Tensor / array helpers ─────────────────────────────────────────────────────

def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Convert a (1,H,W) or (1,1,H,W) float tensor  →  (H,W) uint8 numpy array.
    Values are clipped to [0, 255] before casting.
    """
    arr = t.detach().cpu().float().squeeze().numpy()
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


# ── Image quality metrics ─────────────────────────────────────────────────────

def psnr_metric(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 255.0,
) -> float:
    """Peak signal-to-noise ratio between two uint8/float arrays."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────

def save_checkpoint(
    g1: nn.Module,
    g2: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    loss: float,
    checkpoint_dir: str = 'checkpoints',
) -> None:
    """Persist model weights and optimiser state to disk."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'ckpt_iter_{iteration:07d}.pth')
    torch.save(
        {
            'iteration': iteration,
            'g1_state_dict': g1.state_dict(),
            'g2_state_dict': g2.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        path,
    )
    print(f'[Checkpoint] Saved → {path}')


def load_checkpoint(
    g1: nn.Module,
    g2: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device,
) -> int:
    """
    Restore model (and optionally optimiser) state from a checkpoint file.

    Returns:
        The iteration number stored in the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    g1.load_state_dict(ckpt['g1_state_dict'])
    if g2 is not None and 'g2_state_dict' in ckpt:
        g2.load_state_dict(ckpt['g2_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    iteration = ckpt.get('iteration', 0)
    print(f'[Checkpoint] Loaded iter {iteration} ← {checkpoint_path}')
    return iteration


# ── Logging ────────────────────────────────────────────────────────────────────

def log_metrics(
    iteration: int,
    loss: float,
    l_cyc: float,
    l_idt: float,
    psnr_val: Optional[float] = None,
) -> None:
    """Print a one-line training status message."""
    msg = (
        f'[Iter {iteration:7d}]  '
        f'Loss={loss:9.4f}  '
        f'Cyc={l_cyc:9.4f}  '
        f'Idt={l_idt:9.4f}'
    )
    if psnr_val is not None:
        msg += f'  PSNR={psnr_val:.2f} dB'
    print(msg)


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_patches(
    lr_patch: torch.Tensor,
    sr_patch: torch.Tensor,
    hr_patch: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Display LR input, SR output, and (optionally) HR ground truth side by side.

    Args:
        lr_patch:  (1,H,W) or (1,1,H,W) tensor — low-resolution input.
        sr_patch:  (1,H,W) or (1,1,H,W) tensor — super-resolved output.
        hr_patch:  (1,H,W) or (1,1,H,W) tensor — HR reference (optional).
        save_path: If given, the figure is saved to this path.
    """
    panels = [('LR Input', tensor_to_numpy(lr_patch)),
              ('SR Output (G1)', tensor_to_numpy(sr_patch))]
    if hr_patch is not None:
        panels.append(('HR Ground Truth', tensor_to_numpy(hr_patch)))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_loss_curve(
    loss_history: list[float],
    log_interval: int = 1,
    title: str = 'Training Loss',
    save_path: Optional[str] = None,
) -> None:
    """Plot the training loss curve."""
    iters = [i * log_interval for i in range(1, len(loss_history) + 1)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iters, loss_history, linewidth=1.2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
