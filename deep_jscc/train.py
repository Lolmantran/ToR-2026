"""Train Deep-JSCC on STL-10 until val PSNR > 30 dB, with early stopping.

Changes vs previous version
---------------------------
* **Preloads the dataset into memory as uint8** — eliminates per-batch PIL
  decode, ~4x faster epochs on CPU.
* **ReduceLROnPlateau** instead of StepLR — automatically drops LR when val
  PSNR stalls.
* **Early stopping** — terminates if val PSNR hasn't improved by `MIN_DELTA`
  for `PATIENCE` consecutive epochs.
* **Auto-stops on target** — exits cleanly when best val PSNR > 30 dB.
* **Resume-safe** — every epoch overwrites the best checkpoint; restart picks
  up where it left off.
"""
import os, sys, math, time, json
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jscc_model import DeepJSCC, ratio2filtersize

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR    = os.path.join(HERE, 'checkpoints')
LOG_PATH   = os.path.join(HERE, 'train_to_30db.log')
HIST_PATH  = os.path.join(HERE, 'train_history.json')

RATIO        = 1/6
SNR_TRAIN    = 19.0
CHANNEL_TYPE = 'AWGN'
N_TRAIN      = 30_000
N_VAL        = 8_000      # full STL-10 test set — stable val measurement
BATCH_SIZE   = 64         # restored: the original hyperparameters that reached
                          # 28.99 dB, valid again now that jscc_model.py /
                          # jscc_channel.py are reverted to the reference repo's
                          # SNR convention (see those files' docstrings)
INIT_LR      = 1e-3       # restored: paper's LR, known-good under the reverted
                           # (reference-repo-matching) SNR convention
WEIGHT_DECAY = 5e-4
MAX_EPOCHS   = 300
TARGET_PSNR  = 30.0
HFLIP_AUG    = False       # disabled: destabilised training in an earlier experiment
                           # (29 -> 22 dB collapse), never re-enable without re-verifying

# ── Stability / schedule ────────────────────────────────────────────────────────
# The reference repo (chunbaobao/Deep-JSCC-PyTorch) trains CIFAR-10 for up to
# 1000 epochs with StepLR(step_size=640, gamma=0.1) -- LR stays CONSTANT at
# 1e-3 for 640 epochs before its one and only decay. Our previous attempts
# used ReduceLROnPlateau(patience=10), which decayed LR far too early relative
# to the actual recipe -- likely the real cause of the repeated ~22-23 dB
# plateau. Fixed: LR now follows the same long-constant-then-decay shape.
# Early stopping is kept (unlike the reference repo) as a genuine safety net,
# but decoupled from the LR schedule and much more patient than before.
GRAD_CLIP      = 0.5
STEP_SIZE      = 100      # epochs of constant lr=1e-3 before the one decay
GAMMA          = 0.1
EARLY_STOP_PAT = 50       # stop if no >MIN_DELTA improvement in this many epochs
MIN_DELTA      = 0.02     # dB
SEED           = 42

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def psnr_from_mse(mse):
    return 100.0 if mse < 1e-10 else 10 * math.log10(255.0 ** 2 / mse)

# ── Setup ──────────────────────────────────────────────────────────────────────
open(LOG_PATH, 'w').close()
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f'Device: {DEVICE}')
log(f'Config: ratio={RATIO:.3f} SNR_train={SNR_TRAIN} channel={CHANNEL_TYPE}')
log(f'        N_train={N_TRAIN} batch={BATCH_SIZE} init_lr={INIT_LR}')
log(f'        target>{TARGET_PSNR} dB step_size={STEP_SIZE} gamma={GAMMA} '
    f'max_epochs={MAX_EPOCHS}')

log('Loading STL-10 (train+unlabeled) ...')
train_full = datasets.STL10(DATA_ROOT, split='train+unlabeled',
                            download=True, transform=transforms.ToTensor())
test_full  = datasets.STL10(DATA_ROOT, split='test',
                            download=True, transform=transforms.ToTensor())

train_idx = np.random.choice(len(train_full), N_TRAIN, replace=False)
val_idx   = np.random.choice(len(test_full),  N_VAL,   replace=False)

# Preload as uint8 (3.3GB float32 -> 830MB uint8 for 30k images)
log('Preloading train tensor (uint8) ...')
t_pre = time.time()
train_tensor = torch.empty(N_TRAIN, 3, 96, 96, dtype=torch.uint8)
for i, idx in enumerate(train_idx):
    train_tensor[i] = torch.from_numpy(train_full.data[idx].copy())  # already CHW
log(f'Train preload done ({(time.time()-t_pre):.1f} s, '
    f'{train_tensor.element_size()*train_tensor.numel()/1e6:.1f} MB)')

log('Preloading val tensor (uint8) ...')
val_tensor = torch.empty(N_VAL, 3, 96, 96, dtype=torch.uint8)
for i, idx in enumerate(val_idx):
    val_tensor[i] = torch.from_numpy(test_full.data[idx].copy())

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader   = DataLoader(TensorDataset(val_tensor),   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
log(f'Train batches/ep: {len(train_loader)}   Val batches: {len(val_loader)}')

# ── Model ──────────────────────────────────────────────────────────────────────
c = ratio2filtersize(train_full[0][0], RATIO)
log(f'Inner channel c = {c}   k = {c*24*24:,} complex symbols/image')

model = DeepJSCC(c=c, channel_type=CHANNEL_TYPE, snr=SNR_TRAIN).to(DEVICE)
log(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

opt   = torch.optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

ckpt_path = os.path.join(OUT_DIR, f'stl10_c{c}_snr{int(SNR_TRAIN)}_{CHANNEL_TYPE}_best.pth')

best_psnr, start_epoch = 0.0, 1
since_improve = 0
history = {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_psnr': [], 'lr': []}

if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck['state_dict'])
    opt.load_state_dict(ck['optimizer'])
    sched.load_state_dict(ck['scheduler'])
    best_psnr     = ck.get('best_psnr', 0.0)
    start_epoch   = ck.get('epoch', 0) + 1
    history       = ck.get('history', history)
    since_improve = ck.get('since_improve', 0)
    log(f'Resumed from {ckpt_path}: best={best_psnr:.2f} dB epoch={start_epoch-1} '
        f'lr={opt.param_groups[0]["lr"]:.1e}')

# ── Train loop ────────────────────────────────────────────────────────────────
def run_epoch(loader, train_mode):
    model.train(train_mode)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for (batch,) in loader:
            imgs = batch.float().to(DEVICE) / 255.0
            if train_mode and HFLIP_AUG:
                # random horizontal flip per sample
                flip_mask = torch.rand(imgs.size(0)) < 0.5
                if flip_mask.any():
                    imgs[flip_mask] = torch.flip(imgs[flip_mask], dims=[-1])
            recon = model(imgs)
            loss = F.mse_loss(recon * 255.0, imgs * 255.0)
            if train_mode:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            total += loss.item(); n += 1
    return total / max(n, 1)

t0 = time.time()
stop_reason = 'max_epochs'
for epoch in range(start_epoch, MAX_EPOCHS + 1):
    tr_mse = run_epoch(train_loader, train_mode=True)
    val_mse  = run_epoch(val_loader, train_mode=False)
    val_psnr = psnr_from_mse(val_mse)

    cur_lr = opt.param_groups[0]['lr']

    history['epoch'].append(epoch)
    history['train_mse'].append(tr_mse)
    history['val_mse'].append(val_mse)
    history['val_psnr'].append(val_psnr)
    history['lr'].append(cur_lr)

    improved = val_psnr > best_psnr + MIN_DELTA
    is_best  = val_psnr > best_psnr
    since_improve = 0 if improved else since_improve + 1
    if is_best:
        best_psnr = val_psnr
        torch.save({'state_dict': model.state_dict(),
                    'optimizer':  opt.state_dict(),
                    'scheduler':  sched.state_dict(),
                    'c':          c,
                    'snr_train':  SNR_TRAIN,
                    'epoch':      epoch,
                    'best_psnr':  best_psnr,
                    'since_improve': since_improve,
                    'history':    history},
                   ckpt_path)

    with open(HIST_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    mark = '*' if is_best else ' '
    log(f'Ep {epoch:3d} {mark} tr_mse={tr_mse:7.2f} val_mse={val_mse:7.2f} '
        f'val_psnr={val_psnr:5.2f} dB best={best_psnr:5.2f} dB '
        f'no_imp={since_improve:2d} lr={cur_lr:.1e} ({(time.time()-t0)/60:5.1f} min)')

    sched.step()  # StepLR: unconditional, independent of the early-stop check below

    # ── stopping conditions ──
    if best_psnr >= TARGET_PSNR:
        stop_reason = f'target_reached ({best_psnr:.2f} dB >= {TARGET_PSNR})'
        break
    if since_improve >= EARLY_STOP_PAT:
        stop_reason = f'early_stop (no >{MIN_DELTA} dB gain in {EARLY_STOP_PAT} epochs)'
        break

log(f'STOP: {stop_reason}')
log(f'Final: best val PSNR = {best_psnr:.2f} dB after {epoch} epochs '
    f'({(time.time()-t0)/60:.1f} min). Checkpoint: {ckpt_path}')
