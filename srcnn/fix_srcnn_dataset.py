import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'srcnn_super_resolution.ipynb')

nb = json.load(open(NB_PATH, encoding='utf-8'))

# ── Cell 5: replace SRCNNDataset with preloaded version (no per-sample PIL/disk I/O)
new_cell5 = """\
HR_PATCH      = 64
LR_PATCH      = HR_PATCH // 4   # 16
MAX_TRAIN_IMG = 5_000           # cap unlabeled; 5k x4 patches = 20k samples/epoch

class SRCNNDataset(Dataset):
    \"\"\"Pre-loads all Y channels into RAM so __getitem__ is pure in-memory.\"\"\"
    def __init__(self, stl_ds, patches_per_image=4, max_images=MAX_TRAIN_IMG):
        n    = min(len(stl_ds), max_images)
        data = stl_ds.data[:n]                       # (N,3,96,96) uint8 numpy
        rgb  = data.transpose(0, 2, 3, 1).astype('float32')  # (N,96,96,3)
        # BT.601 Y channel, scaled to [0,255]
        self.ys  = (16 + 65.481*rgb[:,:,:,0]/255
                       + 128.553*rgb[:,:,:,1]/255
                       + 24.966*rgb[:,:,:,2]/255)    # (N,96,96) float32
        self.ppi = patches_per_image
        print(f'Pre-loaded {n} images into RAM  '
              f'({n * patches_per_image:,} samples/epoch, '
              f'{n * patches_per_image // 64} batches/epoch)')

    def __len__(self):
        return len(self.ys) * self.ppi

    def __getitem__(self, idx):
        y    = self.ys[idx // self.ppi]              # (96,96) float32
        h, w = y.shape
        top  = int(np.random.randint(0, h - HR_PATCH + 1))
        left = int(np.random.randint(0, w - HR_PATCH + 1))
        hr   = y[top:top+HR_PATCH, left:left+HR_PATCH]   # (64,64)

        hr_t  = torch.from_numpy(hr / 255.0).unsqueeze(0).unsqueeze(0)
        lr_t  = F.interpolate(hr_t, size=(LR_PATCH, LR_PATCH),
                              mode='bicubic', align_corners=False, antialias=True)
        bic_t = F.interpolate(lr_t, size=(HR_PATCH, HR_PATCH),
                              mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        return bic_t.squeeze(0), hr_t.clamp(0, 1).squeeze(0)   # (1,64,64) each

_raw_unlabeled = torchvision.datasets.STL10(root=DATA_ROOT, split='unlabeled', download=True)
_raw_test      = torchvision.datasets.STL10(root=DATA_ROOT, split='train',     download=True)

train_ds     = SRCNNDataset(_raw_unlabeled, patches_per_image=4)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)"""

# ── Cell 6: training config (keep validate but reference _raw_test PIL images)
new_cell6 = """\
TOTAL_EPOCHS = 200
LOG_EVERY    = 10
VAL_EVERY    = 20
CKPT_EVERY   = 20

optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-4},
    {'params': model.conv2.parameters(), 'lr': 1e-4},
    {'params': model.conv3.parameters(), 'lr': 1e-5},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
criterion = nn.MSELoss()

def validate(m, raw_ds, n=64):
    \"\"\"Evaluate on full 96x96 images using PIL-based loading (eval only, infrequent).\"\"\"
    m.eval()
    bics, cnns = [], []
    with torch.no_grad():
        for i in range(min(n, len(raw_ds))):
            pil = raw_ds[i][0]
            y   = rgb_to_y(pil)                          # (96,96) float32
            hr_t  = torch.from_numpy(y/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
            lr_t  = F.interpolate(hr_t, scale_factor=0.25, mode='bicubic',
                                  align_corners=False, antialias=True)
            bic_t = F.interpolate(lr_t, size=(96, 96), mode='bicubic',
                                  align_corners=False, antialias=True).clamp(0, 1)
            sr_t  = m(bic_t).clamp(0, 1)
            bics.append(psnr_np(y, bic_t.squeeze().cpu().numpy()*255))
            cnns.append(psnr_np(y, sr_t.squeeze().cpu().numpy()*255))
    m.train()
    return float(np.mean(bics)), float(np.mean(cnns))

print(f'Training: {TOTAL_EPOCHS} epochs, {len(train_loader)} batches/epoch')
print(f'Expected time: ~{TOTAL_EPOCHS * len(train_loader) * 0.01 / 60:.0f}-{TOTAL_EPOCHS * len(train_loader) * 0.05 / 60:.0f} min (GPU/CPU)')"""

# find code cells and replace cells 5 (index 5) and 6 (index 6)
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print('Code cells:', [(i, ''.join(c['source'])[:50]) for i, c in code_cells])

def set_source(cell, src):
    lines = src.split('\n')
    cell['source'] = [l + '\n' for l in lines]
    cell['source'][-1] = cell['source'][-1].rstrip('\n')
    cell['outputs'] = []
    cell['execution_count'] = None

# Cell indices in nb['cells']: 0=md, 1=code(imports), 2=md, 3=code(model),
# 4=md, 5=code(dataset), 6=code(trainconfig), 7=code(trainloop), ...
set_source(nb['cells'][5], new_cell5)
set_source(nb['cells'][6], new_cell6)

json.dump(nb, open(NB_PATH, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('Notebook updated.')
