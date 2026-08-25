"""Evaluate a colour Deep-JSCC checkpoint: PSNR vs test-SNR sweep.

Usage:
  python eval_c.py --ckpt checkpoints/stl10_v2_c8_base_snr19_AWGN_best.pth
Writes results/eval_<ckpt-stem>.json
"""
import os, sys, math, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jscc_model import DeepJSCC

p = argparse.ArgumentParser()
p.add_argument('--ckpt', type=str, required=True)
p.add_argument('--n-val', type=int, default=8_000)
p.add_argument('--batch', type=int, default=128)
p.add_argument('--snrs', type=float, nargs='+',
               default=[0, 2, 4, 7, 10, 13, 16, 19, 22, 25])
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()

DATA_ROOT = os.path.join(os.path.dirname(HERE), 'data')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
c = ck['c']
model = DeepJSCC(c=c, channel_type='AWGN', snr=ck.get('snr_train', 19.0)).to(DEVICE)
model.load_state_dict(ck['state_dict'])
model.eval()
print(f"Loaded {args.ckpt}: c={c}, best_psnr={ck.get('best_psnr'):.2f} dB "
      f"(epoch {ck.get('epoch')})")

torch.manual_seed(args.seed); np.random.seed(args.seed)
test_full = datasets.STL10(DATA_ROOT, split='test', download=False)
val_idx = np.random.choice(len(test_full), args.n_val, replace=False)
val_tensor = torch.from_numpy(test_full.data[val_idx].copy())
loader = DataLoader(TensorDataset(val_tensor), batch_size=args.batch, shuffle=False)

results = {'ckpt': os.path.basename(args.ckpt), 'c': c,
           'k_complex': c * 24 * 24, 'snr_train': ck.get('snr_train', 19.0),
           'best_psnr_train_time': ck.get('best_psnr'), 'sweep': []}

for snr in args.snrs:
    model.change_channel('AWGN', snr)
    tot, n = 0.0, 0
    with torch.no_grad():
        for (batch,) in loader:
            imgs = batch.float().to(DEVICE) / 255.0
            recon = model(imgs)
            tot += torch.mean((recon - imgs) ** 2).item() * imgs.size(0)
            n += imgs.size(0)
    mse = tot / n
    psnr = 10 * math.log10(1.0 / mse)
    results['sweep'].append({'snr': snr, 'psnr': psnr})
    print(f'  SNR {snr:5.1f} dB -> PSNR {psnr:.2f} dB')

stem = os.path.splitext(os.path.basename(args.ckpt))[0]
out = os.path.join(HERE, 'results', f'eval_{stem}.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print('Wrote', out)
