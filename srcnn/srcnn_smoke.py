"""SRCNN smoke test: patch training + residual learning. ~5-10 min on CPU."""
import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import TensorDataset, DataLoader

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

class SRCNN(nn.Module):
    def __init__(self, n1=64, n2=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n1, 9, padding=4)
        self.conv2 = nn.Conv2d(n1, n2, 1, padding=0)
        self.conv3 = nn.Conv2d(n2, 1,  5, padding=2)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        r = F.relu(self.conv1(x), inplace=True)
        r = F.relu(self.conv2(r), inplace=True)
        return self.conv3(r)

model = SRCNN().to(DEVICE)
print(f'SRCNN params: {sum(p.numel() for p in model.parameters()):,}')

# ── Data ───────────────────────────────────────────────────────────────────────
print('Loading STL-10 (train, 100 images)...')
ds = torchvision.datasets.STL10(root=DATA_ROOT, split='train', download=True)

def rgb_to_y(pil_img):
    a = np.array(pil_img, dtype=np.float32)
    return 16 + 65.481*a[:,:,0]/255 + 128.553*a[:,:,1]/255 + 24.966*a[:,:,2]/255

def psnr(ref, rec):
    mse = np.mean((ref.astype(np.float64) - rec.astype(np.float64))**2)
    return 100.0 if mse < 1e-10 else 10*math.log10(255.0**2 / mse)

def degrade_full(y, device=DEVICE):
    t = torch.from_numpy(y/255.0).unsqueeze(0).unsqueeze(0).to(device)
    lr = F.interpolate(t, scale_factor=0.25, mode='bicubic', align_corners=False, antialias=True)
    return F.interpolate(lr, size=(96,96), mode='bicubic', align_corners=False, antialias=True).clamp(0,1)

N       = 100
PATCH   = 48
STRIDE  = 12

ys = np.stack([rgb_to_y(ds[i][0]) for i in range(N)])  # (100,96,96)

# Extract patches
bic_patches, hr_patches = [], []
for y in ys:
    for top in range(0, y.shape[0] - PATCH + 1, STRIDE):
        for left in range(0, y.shape[1] - PATCH + 1, STRIDE):
            hr   = y[top:top+PATCH, left:left+PATCH]
            hr_t = torch.from_numpy(hr/255.0).unsqueeze(0).unsqueeze(0)
            lr_t = F.interpolate(hr_t, scale_factor=0.25, mode='bicubic',
                                 align_corners=False, antialias=True)
            bic_t = F.interpolate(lr_t, size=(PATCH, PATCH), mode='bicubic',
                                  align_corners=False, antialias=True).clamp(0,1)
            bic_patches.append(bic_t.squeeze(0))
            hr_patches.append(hr_t.clamp(0,1).squeeze(0))

bic_all = torch.stack(bic_patches)
hr_all  = torch.stack(hr_patches)
loader  = DataLoader(TensorDataset(bic_all, hr_all), batch_size=64, shuffle=True)
print(f'Patches: {len(bic_patches):,}   Batches/epoch: {len(loader)}')

# ── Eval function (full 96x96 images) ─────────────────────────────────────────
def evaluate(m):
    m.eval()
    bics, cnns = [], []
    with torch.no_grad():
        for y in ys:
            bic_t = degrade_full(y)
            sr_t  = (bic_t + m(bic_t)).clamp(0,1)
            bics.append(psnr(y, bic_t.squeeze().cpu().numpy()*255))
            cnns.append(psnr(y, sr_t.squeeze().cpu().numpy()*255))
    m.train()
    return bics, cnns

bics0, cnns0 = evaluate(model)
print(f'\nBicubic baseline : {np.mean(bics0):.2f} dB')
print(f'SRCNN at init    : {np.mean(cnns0):.2f} dB  (should equal bicubic due to zero init)')

# ── Training ───────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-4},
    {'params': model.conv2.parameters(), 'lr': 1e-4},
    {'params': model.conv3.parameters(), 'lr': 1e-5},
])
EPOCHS = 300
print(f'\nTraining {EPOCHS} epochs x {len(loader)} batches ...')
t0 = time.time()

for epoch in range(1, EPOCHS+1):
    for bic, hr in loader:
        bic, hr = bic.to(DEVICE), hr.to(DEVICE)
        sr   = (bic + model(bic)).clamp(0,1)
        loss = criterion = nn.MSELoss()(sr, hr)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        bics, cnns = evaluate(model)
        bm, cm = np.mean(bics), np.mean(cnns)
        n30 = int(np.sum(np.array(cnns) > 30))
        print(f'  Epoch {epoch:3d}  SRCNN={cm:.2f} dB  bicubic={bm:.2f} dB  '
              f'gain={cm-bm:+.2f} dB  above30dB={n30}/{N}  '
              f'({(time.time()-t0)/60:.1f} min)')

# ── Final ──────────────────────────────────────────────────────────────────────
bics, cnns = evaluate(model)
bm, cm = np.mean(bics), np.mean(cnns)
ca = np.array(cnns)
print(f'\n=== FINAL ({N} images, {EPOCHS} epochs) ===')
print(f'  Bicubic : {bm:.2f} dB   above30dB={int(np.sum(np.array(bics)>30))}/{N}')
print(f'  SRCNN   : {cm:.2f} dB   above30dB={int(np.sum(ca>30))}/{N}  gain={cm-bm:+.2f} dB')
print(f'  Per-image SRCNN:  min={ca.min():.1f}  median={np.median(ca):.1f}  max={ca.max():.1f}')
