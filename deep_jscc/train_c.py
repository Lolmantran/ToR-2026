"""Parameterized Deep-JSCC trainer for STL-10 colour models.

Purpose: recover the original ~29 dB recipe and train reduced-bottleneck
variants (c < 8) with the SAME recipe so they are directly comparable.

Key differences vs train.py (the script whose 6 retries plateaued at ~23 dB):
* NO gradient clipping by default. The reference repo
  (chunbaobao/Deep-JSCC-PyTorch) does not clip; train.py's GRAD_CLIP=0.5 on a
  255^2-scaled loss forced every minibatch gradient to a constant tiny norm.
* Loss is computed on the [0,1] scale (like the reference repo's model.loss),
  not multiplied by 255. PSNR reported identically (max_val=255 equivalent).
* --c is a CLI flag so c=8 baseline and c=7/c=6 variants share one script.
* Checkpoints are written to checkpoints/stl10_v2_c{c}_... so the old
  checkpoints are never overwritten.

Usage:
  python train_c.py --c 8 --tag base
  python train_c.py --c 7 --tag reduced
"""
import os, sys, math, time, json, argparse
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jscc_model import DeepJSCC
from gray_model import DeepJSCC_Gray

p = argparse.ArgumentParser()
p.add_argument('--c', type=int, required=True)
p.add_argument('--tag', type=str, default='')
p.add_argument('--snr', type=float, default=19.0)
p.add_argument('--channel', type=str, default='AWGN')
p.add_argument('--n-train', type=int, default=30_000)
p.add_argument('--n-val', type=int, default=8_000)
p.add_argument('--batch', type=int, default=64)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--wd', type=float, default=5e-4)
p.add_argument('--clip', type=float, default=0.0, help='0 = no clipping')
p.add_argument('--loss-scale', type=float, default=1.0,
               help='1.0 = MSE on [0,1] (reference repo); 255.0 = train.py style')
p.add_argument('--epochs', type=int, default=300)
p.add_argument('--sched', type=str, default='step', choices=['step', 'plateau'],
               help='plateau = ReduceLROnPlateau(factor .5, patience 15, min_lr 1e-5), '
                    'the reference repo non-CIFAR recipe')
p.add_argument('--step-size', type=int, default=100, help='StepLR step')
p.add_argument('--gamma', type=float, default=0.1)
p.add_argument('--target', type=float, default=30.0)
p.add_argument('--patience', type=int, default=50)
p.add_argument('--min-delta', type=float, default=0.02)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--no-tf32', action='store_true',
               help='force full FP32 conv/matmul (TF32 off)')
p.add_argument('--warmup', type=int, default=0,
               help='linear LR warmup epochs (from lr/10 up to lr)')
p.add_argument('--gray', action='store_true',
               help='grayscale mode: BT.601 luminance input, DeepJSCC_Gray '
                    '(1->3 adapter + JSCC core), loss vs replicated 3-ch target')
args = p.parse_args()

if args.no_tf32:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

DATA_ROOT = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR   = os.path.join(HERE, 'checkpoints')
suffix    = ('v2gray' if args.gray else 'v2') + f'_c{args.c}' + (f'_{args.tag}' if args.tag else '')
LOG_PATH  = os.path.join(HERE, f'train_{suffix}.log')
HIST_PATH = os.path.join(HERE, f'train_{suffix}_history.json')
CKPT_PATH = os.path.join(OUT_DIR, f'stl10_{suffix}_snr{int(args.snr)}_{args.channel}_best.pth')
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def psnr01(mse01):
    return 100.0 if mse01 < 1e-12 else 10 * math.log10(1.0 / mse01)

open(LOG_PATH, 'w').close()
torch.manual_seed(args.seed); np.random.seed(args.seed)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f'Device: {DEVICE}')
log(f'Args: {vars(args)}')
log(f'Checkpoint: {CKPT_PATH}')

log('Loading STL-10 ...')
train_full = datasets.STL10(DATA_ROOT, split='train+unlabeled', download=False)
test_full  = datasets.STL10(DATA_ROOT, split='test', download=False)

train_idx = np.random.choice(len(train_full), args.n_train, replace=False)
val_idx   = np.random.choice(len(test_full),  args.n_val,   replace=False)

def rgb_to_gray_uint8(chw_uint8):
    """(N,3,H,W) uint8 -> (N,1,H,W) uint8 luminance (BT.601), chunked to
    avoid materializing the whole set as float32 (~12 GB for 105k images)."""
    n = len(chw_uint8)
    out = np.empty((n, 1) + chw_uint8.shape[2:], dtype=np.uint8)
    for i in range(0, n, 10_000):
        a = chw_uint8[i:i+10_000].astype(np.float32)
        y = 0.299 * a[:, 0] + 0.587 * a[:, 1] + 0.114 * a[:, 2]
        out[i:i+10_000, 0] = np.clip(np.round(y), 0, 255).astype(np.uint8)
    return out

if args.gray:
    train_tensor = torch.from_numpy(rgb_to_gray_uint8(train_full.data[train_idx]))
    val_tensor   = torch.from_numpy(rgb_to_gray_uint8(test_full.data[val_idx]))
else:
    train_tensor = torch.from_numpy(train_full.data[train_idx].copy())  # uint8 CHW
    val_tensor   = torch.from_numpy(test_full.data[val_idx].copy())
log(f'Train {tuple(train_tensor.shape)}  Val {tuple(val_tensor.shape)}')

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=args.batch,
                          shuffle=True, num_workers=0)
val_loader   = DataLoader(TensorDataset(val_tensor),   batch_size=args.batch,
                          shuffle=False, num_workers=0)

if args.gray:
    model = DeepJSCC_Gray(c=args.c, channel_type=args.channel, snr=args.snr).to(DEVICE)
else:
    model = DeepJSCC(c=args.c, channel_type=args.channel, snr=args.snr).to(DEVICE)
k_complex = args.c * 24 * 24
log(f'c={args.c}  k={k_complex:,} complex symbols/image  '
    f'params={sum(p.numel() for p in model.parameters()):,}')

opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
if args.sched == 'plateau':
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=15, min_lr=1e-5)
else:
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

best_psnr, since_improve = 0.0, 0
history = {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_psnr': [], 'lr': []}

def run_epoch(loader, train_mode):
    model.train(train_mode)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for (batch,) in loader:
            imgs = batch.float().to(DEVICE) / 255.0
            recon = model(imgs)
            target = imgs.repeat(1, 3, 1, 1) if args.gray else imgs
            loss = F.mse_loss(recon * args.loss_scale, target * args.loss_scale)
            if train_mode:
                opt.zero_grad(); loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
            total += loss.item() / (args.loss_scale ** 2); n += 1  # store on [0,1] scale
    return total / max(n, 1)

t0 = time.time()
stop_reason = 'max_epochs'
for epoch in range(1, args.epochs + 1):
    if args.warmup and epoch <= args.warmup:
        wlr = args.lr * (0.1 + 0.9 * epoch / args.warmup)
        for g in opt.param_groups:
            g['lr'] = wlr
    tr_mse = run_epoch(train_loader, True)
    val_mse = run_epoch(val_loader, False)
    val_psnr = psnr01(val_mse)
    cur_lr = opt.param_groups[0]['lr']

    history['epoch'].append(epoch)
    history['train_mse'].append(tr_mse * 255.0**2)   # keep 0-255^2 units for plots
    history['val_mse'].append(val_mse * 255.0**2)
    history['val_psnr'].append(val_psnr)
    history['lr'].append(cur_lr)

    improved = val_psnr > best_psnr + args.min_delta
    is_best  = val_psnr > best_psnr
    since_improve = 0 if improved else since_improve + 1
    if is_best:
        best_psnr = val_psnr
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': sched.state_dict(),
                    'c': args.c, 'snr_train': args.snr, 'k_complex': k_complex,
                    'epoch': epoch, 'best_psnr': best_psnr,
                    'args': vars(args), 'history': history}, CKPT_PATH)

    with open(HIST_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    mark = '*' if is_best else ' '
    log(f'Ep {epoch:3d} {mark} tr_mse={tr_mse*255**2:7.2f} val_mse={val_mse*255**2:7.2f} '
        f'val_psnr={val_psnr:5.2f} dB best={best_psnr:5.2f} dB '
        f'no_imp={since_improve:2d} lr={cur_lr:.1e} ({(time.time()-t0)/60:5.1f} min)')

    if args.warmup and epoch <= args.warmup:
        pass  # LR controlled by warmup ramp; scheduler starts after
    elif args.sched == 'plateau':
        sched.step(val_mse)
    else:
        sched.step()

    if best_psnr >= args.target:
        stop_reason = f'target_reached ({best_psnr:.2f} dB >= {args.target})'
        break
    if since_improve >= args.patience:
        stop_reason = f'early_stop (no >{args.min_delta} dB gain in {args.patience} epochs)'
        break

log(f'STOP: {stop_reason}')
log(f'Final: best val PSNR = {best_psnr:.2f} dB after {epoch} epochs '
    f'({(time.time()-t0)/60:.1f} min). Checkpoint: {CKPT_PATH}')
