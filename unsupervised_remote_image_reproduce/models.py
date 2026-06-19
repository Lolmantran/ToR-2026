"""
models.py  —  Cycle-CNN architecture: G1 (SR) and G2 (down-sampling).

Paper: Wang et al., IGARSS 2019 — Unsupervised Remote Sensing Image SR Using Cycle CNN

Architecture summary
--------------------
ResBlock : Conv(3×3,64) → BN → ReLU → Conv(3×3,64) → BN  (+skip)

G1 (LR → HR, ×4 upscale)
  Input  : (B, 1,  96,  96)
  ↓ Conv(3×3,64)+ReLU → Conv(3×3,64)+ReLU
  ↓ 16 × ResBlock(64)
  ↓ Conv(3×3,64)
  ↓ SubpixelConv [Conv(3×3,256) → PixelShuffle(2) → ReLU]  96→192
  ↓ SubpixelConv [Conv(3×3,256) → PixelShuffle(2) → ReLU] 192→384
  ↓ Conv(3×3,1)  →  Clip(0,255)
  Output : (B, 1, 384, 384)

G2 (HR → LR, ×4 downscale)
  Input  : (B, 1, 384, 384)
  ↓ AvgPool(2×2) → AvgPool(2×2)   384→96
  ↓ Conv(3×3,64)+ReLU → Conv(3×3,64)+ReLU
  ↓ 5 × ResBlock(64)
  ↓ Conv(3×3,64)+ReLU → Conv(3×3,1)  →  Clip(0,255)
  Output : (B, 1,  96,  96)
"""

import torch
import torch.nn as nn


# ── Building blocks ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual block used in both G1 and G2.
    Structure: Conv→BN→ReLU→Conv→BN with an identity skip connection.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SubpixelConv(nn.Module):
    """
    Sub-pixel convolution block for ×2 spatial upsampling.

    Conv(3×3, 256 filters) → PixelShuffle(2) → ReLU
    (B, in_ch, H, W)  →  (B, 64, 2H, 2W)

    PixelShuffle(2) rearranges 256 = 64 × 2² channels into 64 channels
    at double spatial resolution.
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)   # 256 → 64 ch, H,W → 2H,2W
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.shuffle(self.conv(x)))


# ── Generators ────────────────────────────────────────────────────────────────

class G1(nn.Module):
    """
    Super-resolution generator  G1 : LR → HR  (×4 upscale).

    Input  shape : (B, 1,  96,  96)
    Output shape : (B, 1, 384, 384)

    Uses 16 residual blocks (paper Table/Fig 1) and two sub-pixel
    convolution layers for ×2+×2 = ×4 upsampling.
    """

    def __init__(self, num_resblocks: int = 16):
        super().__init__()

        # Entry convolutions
        self.head = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual body
        self.resblocks = nn.Sequential(*[ResBlock(64) for _ in range(num_resblocks)])

        # Post-residual conv (before upsampling)
        self.post_res = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # ×2 upsampling stages
        self.up1 = SubpixelConv(64)   # 96  → 192
        self.up2 = SubpixelConv(64)   # 192 → 384

        # Output projection to 1 channel
        self.tail = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)                   # (B,64, H, W)
        feat = self.post_res(self.resblocks(feat)) + feat  # global skip
        feat = self.up1(feat)                 # (B,64,2H,2W)
        feat = self.up2(feat)                 # (B,64,4H,4W)
        out  = self.tail(feat)                # (B, 1,4H,4W)
        return torch.clamp(out, 0.0, 255.0)


class G2(nn.Module):
    """
    Down-sampling generator  G2 : HR → LR  (×4 downscale).

    Input  shape : (B, 1, 384, 384)
    Output shape : (B, 1,  96,  96)

    Two 2×2 average-pooling layers first reduce spatial size ×4,
    then 5 residual blocks refine the result.
    """

    def __init__(self, num_resblocks: int = 5):
        super().__init__()

        # Spatial downscaling: 384 → 192 → 96
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Entry convolutions (after pooling)
        self.head = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual body
        self.resblocks = nn.Sequential(*[ResBlock(64) for _ in range(num_resblocks)])

        # Output projection
        self.tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = self.pool(x)           # (B, 1, H/4, W/4)
        feat = self.head(x)           # (B,64, H/4, W/4)
        feat = self.resblocks(feat)   # (B,64, H/4, W/4)
        out  = self.tail(feat)        # (B, 1, H/4, W/4)
        return torch.clamp(out, 0.0, 255.0)


# ── SRCNN baseline ─────────────────────────────────────────────────────────────

class SRCNN(nn.Module):
    """
    SRCNN baseline (Dong et al., 2016) with 9-1-5 kernel sizes.
    Operates on a bicubic-upsampled LR image.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(x), 0.0, 255.0)


# ── Parameter count helper ────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
