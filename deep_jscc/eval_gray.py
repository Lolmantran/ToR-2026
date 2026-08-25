"""Evaluate the trained grayscale Deep-JSCC (c=6, adapter) checkpoint:
PSNR vs SNR, and the actual bandwidth saved by the smaller bottleneck."""
import os, sys, math, io, json
import numpy as np
import torch, torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gray_model import DeepJSCC_Gray

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--c', type=int, default=6)
_a = _ap.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = os.path.join(os.path.dirname(HERE), 'data')
CKPT = os.path.join(HERE, 'checkpoints', f'stl10_GRAY_c{_a.c}_snr19_AWGN_best.pth')

ck = torch.load(CKPT, map_location=DEVICE)
c = ck['c']
model = DeepJSCC_Gray(c=c, channel_type='AWGN', snr=19.0).to(DEVICE)
model.load_state_dict(ck['state_dict'])
model.eval()
print(f"Loaded checkpoint: c={c}  best_psnr(train-time val)={ck['best_psnr']:.2f} dB  epoch={ck['epoch']}")

test_full = datasets.STL10(DATA_ROOT, split='test', download=True, transform=transforms.ToTensor())
N = 200
rgb_pool = test_full.data[:N]  # (N,3,96,96) uint8

def to_gray(chw_uint8):
    r, g, b = chw_uint8[0].astype(np.float32), chw_uint8[1].astype(np.float32), chw_uint8[2].astype(np.float32)
    y = 0.299*r + 0.587*g + 0.114*b
    return np.clip(np.round(y), 0, 255).astype(np.uint8)

gray_pool = np.stack([to_gray(img) for img in rgb_pool])  # (N,96,96) uint8
gray_tensor = torch.from_numpy(gray_pool).unsqueeze(1).float().to(DEVICE) / 255.0  # (N,1,96,96)

def psnr_from_mse(mse):
    return 100.0 if mse < 1e-10 else 10*math.log10(255.0**2/mse)

SNR_RANGE = [0, 4, 7, 10, 13, 16, 19, 22, 25]
results = {}
for snr in SNR_RANGE:
    model.change_channel('AWGN', snr)
    with torch.no_grad():
        recon = model(gray_tensor)
        mse = F.mse_loss(recon*255.0, gray_tensor.repeat(1,3,1,1)*255.0).item()
    results[snr] = psnr_from_mse(mse)
    print(f"  SNR={snr:2d} dB  PSNR={results[snr]:5.2f} dB")

# JPEG-grayscale baseline sizes (same pool)
def jpeg_size(gray_hw, q=73):
    buf = io.BytesIO()
    Image.fromarray(gray_hw, 'L').save(buf, format='JPEG', quality=q)
    return len(buf.getvalue())

jpeg_sizes = np.array([jpeg_size(g) for g in gray_pool])

k_new = c * 24 * 24
k_old = 8 * 24 * 24  # colour/old-grayscale baseline bottleneck
print(f"\nBottleneck: c={c} -> k={k_new:,} complex symbols  "
      f"({k_new*2:,} real channel uses)")
print(f"vs old c=8 baseline: k={k_old:,} complex symbols  ({k_old*2:,} real channel uses)")
print(f"Symbol reduction: {(1 - k_new/k_old)*100:.1f}% fewer channel symbols")
print(f"\nJPEG-grayscale (Q=73) on same pool: mean={jpeg_sizes.mean():.0f} B  std={jpeg_sizes.std():.0f} B")

# equivalent-byte comparison at a few SNRs
print(f"\n{'SNR':>5}  {'PSNR':>6}  {'JSCC-new bytes':>15}  {'JSCC-old(c=8) bytes':>20}  {'vs JPEG-gray':>13}")
summary_rows = []
for snr in SNR_RANGE:
    cap = math.log2(1 + 10**(snr/10))
    new_bytes = k_new * cap / 8
    old_bytes = k_old * cap / 8
    save_vs_jpeg = (jpeg_sizes.mean() - new_bytes) / jpeg_sizes.mean() * 100
    print(f"{snr:>5}  {results[snr]:>6.2f}  {new_bytes:>13.0f} B  {old_bytes:>18.0f} B  {save_vs_jpeg:>+11.1f}%")
    summary_rows.append({'snr': snr, 'psnr': results[snr], 'jscc_new_bytes': new_bytes,
                          'jscc_old_bytes': old_bytes, 'save_vs_jpeg_gray_pct': save_vs_jpeg})

out = {
    'c': c, 'k_complex_new': k_new, 'k_complex_old': k_old,
    'symbol_reduction_pct': (1 - k_new/k_old)*100,
    'jpeg_gray_mean_bytes': float(jpeg_sizes.mean()),
    'train_time_best_psnr': ck['best_psnr'],
    'sweep': summary_rows,
}
out_path = os.path.join(HERE, 'results', f'gray_c{c}_eval.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {out_path}")
