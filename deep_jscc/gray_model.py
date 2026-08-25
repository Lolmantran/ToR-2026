"""Grayscale Deep-JSCC variant.

Two changes requested by supervisor:
  1. A learnable "middle CNN" adapter that maps a 1-channel grayscale image to the
     3-channel input the encoder expects (replaces fixed R=G=B replication).
  2. A reduced bottleneck: the encoder's inner channel `c` is set smaller than the
     colour model's c=8, so conv5 emits fewer channel-symbol planes -> fewer
     transmitted symbols k = c*24*24 -> less data for grayscale.

The encoder/decoder themselves are reused unchanged from jscc_model.py; only the
input front-end (adapter) and the value of `c` change.
"""
import os, sys
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from jscc_model import DeepJSCC


class ChannelAdapter(nn.Module):
    """Small trainable CNN: 1 channel -> 3 channels (the middle CNN Phi).

    Uses 1x1 convs so the mapping is pixel-wise (no spatial mixing needed --
    the encoder's own conv1, 5x5, handles spatial context afterwards).

    Initialised to reproduce EXACT replication (R=G=B=gray) at step 0:
    conv1 copies the input into every hidden channel, PReLU is the identity
    for x >= 0 (true here since input is normalised to [0,1]), and conv2
    averages the hidden channels back out to 3 identical copies. So training
    starts at -- and can only improve on -- the old fixed-replication
    baseline, instead of a random init that has to first "find" a reasonable
    mapping from scratch.
    """
    def __init__(self, hidden=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=1)
        self.act   = nn.PReLU(hidden)
        self.conv2 = nn.Conv2d(hidden, 3, kernel_size=1)
        self._init_as_replication()

    def _init_as_replication(self):
        with torch.no_grad():
            self.conv1.weight.fill_(1.0)
            self.conv1.bias.zero_()
            self.act.weight.fill_(0.25)                 # unused for x>=0, but a sane default
            self.conv2.weight.fill_(1.0 / self.conv1.out_channels)
            self.conv2.bias.zero_()

    def forward(self, x):          # x: (B, 1, H, W), values in [0, 1]
        h = self.act(self.conv1(x))
        return self.conv2(h)       # -> (B, 3, H, W)


class DeepJSCC_Gray(nn.Module):
    """Adapter (1->3) + standard Deep-JSCC encoder/channel/decoder with reduced c."""
    def __init__(self, c=6, channel_type='AWGN', snr=None, adapter_hidden=8):
        super().__init__()
        self.adapter = ChannelAdapter(adapter_hidden)
        self.jscc = DeepJSCC(c=c, channel_type=channel_type, snr=snr)
        self.c = c

    def forward(self, x):          # x: (B, 1, H, W) grayscale in [0,1]
        x3 = self.adapter(x)       # (B, 3, H, W)
        return self.jscc(x3)       # (B, 3, H, W) reconstruction

    def change_channel(self, channel_type='AWGN', snr=None):
        self.jscc.change_channel(channel_type, snr)


if __name__ == '__main__':
    for c in (8, 6, 5, 4):
        m = DeepJSCC_Gray(c=c, channel_type='AWGN', snr=19.0)
        x = torch.rand(2, 1, 96, 96)
        y = m(x)
        k = c * 24 * 24
        print(f'c={c}: in {tuple(x.shape)} -> out {tuple(y.shape)} | '
              f'k={k:,} complex symbols | params={sum(p.numel() for p in m.parameters()):,}')
