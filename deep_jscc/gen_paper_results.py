"""Generate the grayscale result figures + numbers for the paper:
  results/gray_vs_jpeg.png       (JSCC c=8..4 vs JPEG+BPSK+AWGN, grayscale)
  results/gray_training_curves.png
  results/gray_results.json      (all numbers used in the paper)
"""
import os, io, json, math
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from gray_model import DeepJSCC_Gray

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
DATA = os.path.join(os.path.dirname(HERE), 'data')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CS = [8, 7, 6, 5, 4]; SNRS = [0, 2, 4, 7, 10, 13, 16, 19, 22, 25]
COL = {8: '#2980B9', 7: '#E67E22', 6: '#16A085', 5: '#8E44AD', 4: '#2C3E50'}
MRK = {8: 'o', 7: 's', 6: '^', 5: 'D', 4: 'v'}

def rgb2gray(u8):
    a = u8.astype(np.float32); y = 0.299*a[..., 0]+0.587*a[..., 1]+0.114*a[..., 2]
    return np.clip(np.round(y), 0, 255).astype(np.uint8)

torch.manual_seed(42); np.random.seed(42)
test = datasets.STL10(DATA, split='test', download=False)
idx = np.random.choice(len(test), 8000, replace=False)
gray_val = torch.from_numpy(rgb2gray(test.data[idx].transpose(0, 2, 3, 1))[:, None])  # (N,1,96,96)
loader = DataLoader(TensorDataset(gray_val), batch_size=128, shuffle=False)

# ── JSCC grayscale sweep (5 models) ────────────────────────────────────────────
def eval_jscc(model, snr):
    model.change_channel('AWGN', snr); tot, n = 0.0, 0
    with torch.no_grad():
        for (b,) in loader:
            x = b.float().to(DEVICE)/255.0; tgt = x.repeat(1, 3, 1, 1)
            r = model(x); tot += torch.mean((r-tgt)**2).item()*x.size(0); n += x.size(0)
    return 10*math.log10(1.0/(tot/n))

sweep, k = {}, {}
for c in CS:
    ck = torch.load(f'checkpoints/stl10_v2gray_c{c}_demo_snr19_AWGN_best.pth',
                    map_location=DEVICE, weights_only=False)
    m = DeepJSCC_Gray(c=c, channel_type='AWGN', snr=19.0).to(DEVICE)
    m.load_state_dict(ck['state_dict']); m.eval()
    sweep[c] = [eval_jscc(m, s) for s in SNRS]; k[c] = c*576
    print(f'JSCC c={c}: PSNR@19={sweep[c][SNRS.index(19)]:.2f}')

# ── JPEG + BPSK + AWGN grayscale baseline (uncoded, same method as notebook) ────
JPEG_Q, POOL = 73, 200
pool = [rgb2gray(test.data[idx[i]].transpose(1, 2, 0)) for i in range(POOL)]

def jpeg_bpsk(img, snr):
    buf = io.BytesIO(); Image.fromarray(img, 'L').save(buf, 'JPEG', quality=JPEG_Q)
    bits = np.unpackbits(np.frombuffer(buf.getvalue(), dtype=np.uint8))
    sym = 2*bits.astype(np.float32)-1.0
    sigma = math.sqrt(1.0/(10**(snr/10.0)))
    rec_bits = (sym + np.random.randn(*sym.shape)*sigma > 0).astype(np.uint8)
    try:
        rec = np.array(Image.open(io.BytesIO(np.packbits(rec_bits).tobytes())).convert('L'))
        if rec.shape != img.shape: return None
        return rec
    except Exception:
        return None

def psnr8(a, b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 100.0 if mse < 1e-10 else 10*math.log10(255.0**2/mse)

def jpeg_nbytes(img):
    b = io.BytesIO(); Image.fromarray(img, 'L').save(b, 'JPEG', quality=JPEG_Q)
    return len(b.getvalue())

jpeg_curve = []
jpeg_bytes = float(np.mean([jpeg_nbytes(p) for p in pool]))
for s in SNRS:
    vals = []
    for p in pool:
        r = jpeg_bpsk(p, s)
        vals.append(psnr8(p, r) if r is not None else 0.0)  # failed decode -> 0 dB
    jpeg_curve.append(float(np.mean(vals)))
    print(f'JPEG+BPSK @ {s}: {jpeg_curve[-1]:.2f}')

# ── Figure D: JSCC vs JPEG (grayscale) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 4.6))
for c in CS:
    ax.plot(SNRS, sweep[c], color=COL[c], marker=MRK[c], ms=5, lw=1.8,
            label=f'Deep JSCC c={c} (k={k[c]:,})')
ax.plot(SNRS, jpeg_curve, color='#B03A2E', ls='--', marker='x', ms=6, lw=1.8,
        label=f'JPEG Q73 + BPSK + AWGN')
ax.axvline(19, color='#999', ls=':', lw=1)
ax.set_xlabel('Channel SNR (dB)'); ax.set_ylabel('PSNR (dB)')
ax.set_title('Grayscale STL-10: Deep JSCC vs digital JPEG+BPSK baseline')
ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, loc='lower right')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'gray_vs_jpeg.png'), dpi=150); plt.close(fig)

# ── Figure: training curves ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 3.4))
for c in CS:
    h = json.load(open(f'train_v2gray_c{c}_demo_history.json'))
    ax.plot(h['epoch'], h['val_psnr'], color=COL[c], lw=1.7, label=f'c={c}')
ax.set_xlabel('epoch'); ax.set_ylabel('validation PSNR (dB)')
ax.set_title('Grayscale training (identical recipe, all five widths)')
ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, loc='lower right')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(RES, 'gray_training_curves.png'), dpi=150); plt.close(fig)

# ── consolidated numbers ───────────────────────────────────────────────────────
out = {'snrs': SNRS, 'jscc_sweep': {str(c): sweep[c] for c in CS},
       'k': {str(c): k[c] for c in CS}, 'jpeg_bpsk_curve': jpeg_curve,
       'jpeg_mean_bytes': float(jpeg_bytes), 'jpeg_quality': JPEG_Q}
json.dump(out, open(os.path.join(RES, 'gray_results.json'), 'w'), indent=2)
print('JPEG mean bytes:', round(jpeg_bytes, 1))
print('wrote gray_vs_jpeg.png, gray_training_curves.png, gray_results.json')
