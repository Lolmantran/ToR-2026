"""
losses.py  —  Loss functions for Cycle-CNN.

Paper equations (Wang et al., IGARSS 2019)
------------------------------------------
L_cyc   = (1/N) Σ [ ||G2(G1(x_t)) - x_t||₂  +  ||G1(G2(y_t)) - y_t||₂ ]
L_idt   = (1/N) Σ  ||G1(y_t ↓s) - y_t||₂          (unpaired mode)
L_idt^p = (1/N) Σ  ||G1(x_t) - y_t||₂              (paired mode)
L_total =  ω₁·L_cyc  +  ω₂·L_idt      (ω₁=2, ω₂=1)

Implementation note
-------------------
The paper's ||·||₂ notation is interpreted as mean-squared error (MSE),
which is the standard choice in SR literature and gives stable gradients.
"""

import torch
import torch.nn.functional as F


def cycle_loss(
    x_t: torch.Tensor,
    x_reconstructed: torch.Tensor,
    y_t: torch.Tensor,
    y_reconstructed: torch.Tensor,
) -> torch.Tensor:
    """
    Cycle consistency loss.

    Forward  cycle: x_t → G1 → y_fake → G2 → x_reconstructed
    Backward cycle: y_t → G2 → x_fake → G1 → y_reconstructed

    L_cyc = MSE(x_reconstructed, x_t) + MSE(y_reconstructed, y_t)
    """
    return F.mse_loss(x_reconstructed, x_t) + F.mse_loss(y_reconstructed, y_t)


def identity_loss(
    y_t: torch.Tensor,
    y_t_sr: torch.Tensor,
) -> torch.Tensor:
    """
    Identity loss for unpaired training.

    y_t_sr = G1(bicubic_downsample(y_t))   should reconstruct  y_t.

    L_idt = MSE(G1(y_t ↓4), y_t)
    """
    return F.mse_loss(y_t_sr, y_t)


def paired_identity_loss(
    y_t: torch.Tensor,
    g1_xt: torch.Tensor,
) -> torch.Tensor:
    """
    Identity loss for paired training (replaces identity_loss when paired data
    is available).

    g1_xt = G1(x_t)   should reconstruct the paired HR image y_t.

    L_idt^p = MSE(G1(x_t), y_t)
    """
    return F.mse_loss(g1_xt, y_t)


def total_loss(
    l_cyc: torch.Tensor,
    l_idt: torch.Tensor,
    w1: float = 2.0,
    w2: float = 1.0,
) -> torch.Tensor:
    """
    Weighted combination of cycle and identity losses.

    L_total = w1 · L_cyc  +  w2 · L_idt    (default: w1=2, w2=1 per paper)
    """
    return w1 * l_cyc + w2 * l_idt
