"""
train.py  —  Training loop for Cycle-CNN super-resolution.

Paper: Wang et al., IGARSS 2019
  • Adam  β1=0.9, β2=0.999, ε=1e-8,  lr=1e-4
  • LR decay ×0.1 at iteration 90 000 (halfway of 180 000 total)
  • Batch size 8,  G1 and G2 updated simultaneously
  • Unpaired mode by default (identity loss)
  • Checkpoint saved every 10 000 iterations
  • Validation PSNR logged every 5 000 iterations

Usage (CLI)
-----------
  # Real data (HR directory, optionally separate LR directory)
  python train.py --hr_dir /data/GaoFen2/PAN --lr_dir /data/GaoFen2/MS_Y

  # Synthetic data (for quick demo / smoke-test)
  python train.py --synthetic --total_iters 500 --log_interval 50
"""

import argparse
import time
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import get_dataloaders
from losses import cycle_loss, identity_loss, total_loss
from models import G1, G2
from utils import log_metrics, psnr_metric, save_checkpoint, tensor_to_numpy


# ── Helpers ────────────────────────────────────────────────────────────────────

def bicubic_downsample(y_t: torch.Tensor, scale: float = 0.25) -> torch.Tensor:
    """Bicubic 4× downsampling used to compute the identity loss."""
    return F.interpolate(
        y_t,
        scale_factor=scale,
        mode='bicubic',
        align_corners=False,
        antialias=True,
    )


@torch.no_grad()
def quick_val_psnr(
    g1: G1,
    test_loader: DataLoader,
    device: torch.device,
    max_batches: int = 4,
) -> float:
    """Compute average PSNR of G1 outputs vs. HR patches on a few test batches."""
    g1.eval()
    total, count = 0.0, 0
    for lr, hr in test_loader:
        sr = g1(lr.to(device))
        for b in range(sr.size(0)):
            total += psnr_metric(tensor_to_numpy(sr[b]), tensor_to_numpy(hr[b]))
            count += 1
        if count >= max_batches:
            break
    g1.train()
    return total / max(count, 1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> tuple[G1, G2, list[float]]:
    """
    Run the full Cycle-CNN training loop.

    Returns:
        (g1, g2, loss_history)   — trained models and per-log-interval loss values.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ── Models ────────────────────────────────────────────────────────────────
    g1 = G1(num_resblocks=16).to(device)
    g2 = G2(num_resblocks=5).to(device)

    # ── Optimizer (single Adam for both generators, updated simultaneously) ──
    params = list(g1.parameters()) + list(g2.parameters())
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        synthetic=args.synthetic,
    )
    train_iter = iter(train_loader)
    print(
        f'Train samples : {len(train_loader.dataset)}'
        f'  |  batch size : {args.batch_size}'
        f'  |  total iters : {args.total_iters}'
        f'  |  LR decay @ iter {args.lr_decay_iter}'
    )

    # ── Loop ─────────────────────────────────────────────────────────────────
    g1.train()
    g2.train()
    loss_history: list[float] = []
    t0 = time.time()

    for iteration in range(1, args.total_iters + 1):

        # Learning-rate step decay at the halfway point
        if iteration == args.lr_decay_iter:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1
            print(
                f'[Iter {iteration}] LR decayed → {optimizer.param_groups[0]["lr"]:.2e}'
            )

        # ── Fetch next batch (cycle through the loader) ───────────────────────
        try:
            x_t, y_t = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_t, y_t = next(train_iter)

        x_t = x_t.to(device)   # (B,1, 96, 96) LR patches
        y_t = y_t.to(device)   # (B,1,384,384) HR patches

        # ── Forward cycle:  x_t → G1 → y_fake → G2 → x_rec ─────────────────
        y_fake       = g1(x_t)
        x_rec        = g2(y_fake)

        # ── Backward cycle: y_t → G2 → x_fake → G1 → y_rec ─────────────────
        x_fake       = g2(y_t)
        y_rec        = g1(x_fake)

        # ── Losses ───────────────────────────────────────────────────────────
        l_cyc = cycle_loss(x_t, x_rec, y_t, y_rec)

        # Identity loss: G1 applied to 4× bicubic-downsampled HR should ≈ HR
        y_t_down = bicubic_downsample(y_t)      # (B,1,384,384) → (B,1,96,96)
        y_t_sr   = g1(y_t_down)                 # (B,1,384,384)
        l_idt    = identity_loss(y_t, y_t_sr)

        l_tot = total_loss(l_cyc, l_idt)

        # ── Backward pass ────────────────────────────────────────────────────
        optimizer.zero_grad()
        l_tot.backward()
        optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────────
        if iteration % args.log_interval == 0:
            psnr_val: Optional[float] = None
            if iteration % args.val_interval == 0:
                psnr_val = quick_val_psnr(g1, test_loader, device)
            log_metrics(iteration, l_tot.item(), l_cyc.item(), l_idt.item(), psnr_val)
            loss_history.append(l_tot.item())

        # ── Checkpoint ───────────────────────────────────────────────────────
        if iteration % args.ckpt_interval == 0:
            save_checkpoint(g1, g2, optimizer, iteration, l_tot.item(), args.ckpt_dir)

    elapsed = time.time() - t0
    print(f'\nTraining complete in {elapsed/60:.1f} min.')

    # Final checkpoint
    save_checkpoint(
        g1, g2, optimizer, args.total_iters,
        loss_history[-1] if loss_history else float('nan'),
        args.ckpt_dir,
    )
    return g1, g2, loss_history


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Train Cycle-CNN for unsupervised image super-resolution.'
    )
    p.add_argument('--hr_dir',       type=str,   default=None,
                   help='Path to HR image directory (1024×1024).')
    p.add_argument('--lr_dir',       type=str,   default=None,
                   help='Path to LR image directory (256×256). If omitted, '
                        'LR patches are derived from HR via 4× bicubic downsample.')
    p.add_argument('--synthetic',    action='store_true', default=False,
                   help='Use synthetic random data (for demo/smoke-test).')
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--total_iters',  type=int,   default=180_000)
    p.add_argument('--lr_decay_iter',type=int,   default=90_000,
                   help='Iteration at which LR is multiplied by 0.1.')
    p.add_argument('--log_interval', type=int,   default=100,
                   help='Print training stats every N iterations.')
    p.add_argument('--val_interval', type=int,   default=5_000,
                   help='Compute validation PSNR every N iterations.')
    p.add_argument('--ckpt_interval',type=int,   default=10_000,
                   help='Save checkpoint every N iterations.')
    p.add_argument('--ckpt_dir',     type=str,   default='checkpoints')
    p.add_argument('--num_workers',  type=int,   default=2)
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    train(args)
