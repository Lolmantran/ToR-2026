"""Train the grayscale Deep-JSCC (adapter 1->3 + reduced bottleneck c) FROM SCRATCH.

Supervisor instructions applied:
  * reduced output layer (c below the colour model's 8) -> less data for grayscale
  * learnable middle CNN adapter 1xHxW -> 3xHxW
  * trained from scratch (NO warm-start from the colour checkpoint)

The script still saves its OWN best/last checkpoints so an interrupted run can
resume itself -- that is not the forbidden warm-start from the old colour model.
"""
import os, sys, math, time, json, argparse
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gray_model import DeepJSCC_Gray

_ap = argparse.ArgumentParser()
_ap.add_argument('--c', type=int, default=6, help='encoder bottleneck (channel-symbol planes)')
_args = _ap.parse_args()

DATA_ROOT = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR   = os.path.join(HERE, 'checkpoints')
LOG_PATH  = os.path.join(HERE, f'train_gray_c{_args.c}.log')
HIST_PATH = os.path.join(HERE, f'train_gray_c{_args.c}_history.json')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
C_BOTTLENECK = _args.c    # 8 = adapter-only (isolate adapter effect); 6 = + reduced bottleneck
SNR_TRAIN    = 19.0
CHANNEL_TYPE = 'AWGN'
N_TRAIN      = 30_000
N_VAL        = 8_000
BATCH_SIZE   = 128
INIT_LR      = 2e-4
WEIGHT_DECAY = 5e-4
MAX_EPOCHS   = 120
GRAD_CLIP    = 0.5
PLATEAU_PAT  = 8
PLATEAU_FAC  = 0.5
MIN_LR       = 1e-7
EARLY_STOP   = 20
MIN_DELTA    = 0.02
SEED         = 42

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def psnr_from_mse(mse):
    return 100.0 if mse < 1e-10 else 10 * math.log10(255.0**2 / mse)

def rgb_to_gray_uint8(chw_uint8):
    """(N,3,H,W) uint8 -> (N,1,H,W) uint8 luminance (BT.601)."""
    a = chw_uint8.astype(np.float32)
    y = 0.299*a[:,0] + 0.587*a[:,1] + 0.114*a[:,2]
    return np.clip(np.round(y), 0, 255).astype(np.uint8)[:, None]

if not os.path.exists(LOG_PATH):
    open(LOG_PATH, 'w').close()
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
log('=' * 60)
log(f'Device: {DEVICE}   c={C_BOTTLENECK}  (k={C_BOTTLENECK*24*24:,} symbols)')
log(f'FROM SCRATCH | SNR_train={SNR_TRAIN} channel={CHANNEL_TYPE} lr={INIT_LR}')

log('Loading STL-10 ...')
train_full = datasets.STL10(DATA_ROOT, split='train+unlabeled', download=True,
                            transform=transforms.ToTensor())
test_full  = datasets.STL10(DATA_ROOT, split='test', download=True,
                            transform=transforms.ToTensor())

idx = np.random.choice(len(train_full), N_TRAIN, replace=False)
log('Building grayscale tensors ...')
gray_train = torch.from_numpy(rgb_to_gray_uint8(train_full.data[idx]))         # (N,1,96,96)
gray_val   = torch.from_numpy(rgb_to_gray_uint8(test_full.data[:N_VAL]))
log(f'Train {gray_train.shape}  Val {gray_val.shape}')

train_loader = DataLoader(TensorDataset(gray_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(gray_val),   batch_size=BATCH_SIZE, shuffle=False)

model = DeepJSCC_Gray(c=C_BOTTLENECK, channel_type=CHANNEL_TYPE, snr=SNR_TRAIN).to(DEVICE)
log(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
opt   = torch.optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=PLATEAU_FAC,
                                                    patience=PLATEAU_PAT, min_lr=MIN_LR)

ckpt_best = os.path.join(OUT_DIR, f'stl10_GRAY_c{C_BOTTLENECK}_snr{int(SNR_TRAIN)}_{CHANNEL_TYPE}_best.pth')
ckpt_last = os.path.join(OUT_DIR, f'stl10_GRAY_c{C_BOTTLENECK}_snr{int(SNR_TRAIN)}_{CHANNEL_TYPE}_last.pth')

best_psnr, start_epoch, since = 0.0, 1, 0
history = {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_psnr': [], 'lr': []}
if os.path.exists(ckpt_last):  # resume own run only (not the colour checkpoint)
    ck = torch.load(ckpt_last, map_location=DEVICE)
    if ck.get('c') == C_BOTTLENECK:
        model.load_state_dict(ck['state_dict']); opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        best_psnr = ck['best_psnr']; start_epoch = ck['epoch']+1
        history = ck['history']; since = ck['since']
        log(f'Resumed own run: best={best_psnr:.2f} dB epoch={start_epoch-1}')

def run_epoch(loader, train):
    model.train(train)
    tot, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (batch,) in loader:
            gray = batch.float().to(DEVICE) / 255.0            # (B,1,H,W)
            target = gray.repeat(1, 3, 1, 1)                    # (B,3,H,W) replicated
            recon = model(gray)                                # (B,3,H,W)
            loss = F.mse_loss(recon * 255.0, target * 255.0)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            tot += loss.item(); n += 1
    return tot / max(n, 1)

def save(path, epoch):
    torch.save({'state_dict': model.state_dict(), 'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(), 'c': C_BOTTLENECK, 'snr_train': SNR_TRAIN,
                'k_complex': C_BOTTLENECK*24*24, 'epoch': epoch, 'best_psnr': best_psnr,
                'since': since, 'history': history}, path)

t0 = time.time(); stop = 'max_epochs'
for epoch in range(start_epoch, MAX_EPOCHS + 1):
    tr = run_epoch(train_loader, True)
    vl = run_epoch(val_loader, False)
    vp = psnr_from_mse(vl)
    sched.step(vp); lr = opt.param_groups[0]['lr']
    history['epoch'].append(epoch); history['train_mse'].append(tr)
    history['val_mse'].append(vl); history['val_psnr'].append(vp); history['lr'].append(lr)
    improved = vp > best_psnr + MIN_DELTA
    is_best = vp > best_psnr
    since = 0 if improved else since + 1
    if is_best:
        best_psnr = vp; save(ckpt_best, epoch)
    save(ckpt_last, epoch)
    with open(HIST_PATH, 'w') as f: json.dump(history, f, indent=2)
    log(f'Ep {epoch:3d} {"*" if is_best else " "} tr={tr:7.2f} val={vl:7.2f} '
        f'psnr={vp:5.2f} dB best={best_psnr:5.2f} no_imp={since:2d} lr={lr:.1e} '
        f'({(time.time()-t0)/60:5.1f} min)')
    if since >= EARLY_STOP:
        stop = f'early_stop (no >{MIN_DELTA} dB gain in {EARLY_STOP} epochs)'; break
    if lr <= MIN_LR*1.01:
        stop = 'min_lr'; break

log(f'STOP: {stop}  best={best_psnr:.2f} dB  epochs={epoch}  ({(time.time()-t0)/60:.1f} min)')
