"""Transfer-initialise the grayscale c=6 (adapter) model from the fixed-SNR
colour c=8 checkpoint. Encoder conv1-4 and decoder tconv2-5 are c-independent
and copy directly. conv5 / tconv1 depend on c (2*c channels at the bottleneck),
so we take the first 2*c_new of the colour model's 2*c_old channels. The
adapter (1->3 ch) has no colour-model equivalent; it keeps its own safe
replication-preserving init (see gray_model.ChannelAdapter).
"""
import os, sys, math, torch
import torch.nn.functional as F
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jscc_model import DeepJSCC
from gray_model import DeepJSCC_Gray

COLOUR_CKPT = os.path.join(HERE, 'checkpoints', 'stl10_c8_snr19_AWGN_best.pth')
C_NEW = 6
OUT_PATH = os.path.join(HERE, 'checkpoints', f'stl10_GRAY_c{C_NEW}_TRANSFER_INIT.pth')

ck = torch.load(COLOUR_CKPT, map_location='cpu')
c_old = ck['c']
print(f'Loaded colour checkpoint: c={c_old}  best_psnr={ck["best_psnr"]:.2f} dB  epoch={ck["epoch"]}')

colour = DeepJSCC(c=c_old, channel_type='AWGN', snr=19.0)
colour.load_state_dict(ck['state_dict'])

gray = DeepJSCC_Gray(c=C_NEW, channel_type='AWGN', snr=19.0)
print(f'Colour params: {sum(p.numel() for p in colour.parameters()):,}')
print(f'Gray   params: {sum(p.numel() for p in gray.parameters()):,}')


def copy_direct(src, dst):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)


def copy_sliced(src, dst, is_transpose):
    """Copy the first N channels along the dimension that depends on c."""
    with torch.no_grad():
        if is_transpose:
            n = dst.weight.shape[0]           # ConvTranspose2d: (in, out, k, k)
            dst.weight.copy_(src.weight[:n])
            dst.bias.copy_(src.bias)          # bias dim = out_channels, c-independent here
        else:
            n = dst.weight.shape[0]           # Conv2d: (out, in, k, k)
            dst.weight.copy_(src.weight[:n])
            dst.bias.copy_(src.bias[:n])


def copy_prelu(src, dst):
    with torch.no_grad():
        dst.weight.copy_(src.weight)


# ── Encoder: conv1-4 direct, conv5 sliced ──────────────────────────────────────
copy_direct(colour.encoder.conv1.conv, gray.jscc.encoder.conv1.conv)
copy_direct(colour.encoder.conv2.conv, gray.jscc.encoder.conv2.conv)
copy_direct(colour.encoder.conv3.conv, gray.jscc.encoder.conv3.conv)
copy_direct(colour.encoder.conv4.conv, gray.jscc.encoder.conv4.conv)
copy_sliced(colour.encoder.conv5.conv, gray.jscc.encoder.conv5.conv, is_transpose=False)

copy_prelu(colour.encoder.conv1.prelu, gray.jscc.encoder.conv1.prelu)
copy_prelu(colour.encoder.conv2.prelu, gray.jscc.encoder.conv2.prelu)
copy_prelu(colour.encoder.conv3.prelu, gray.jscc.encoder.conv3.prelu)
copy_prelu(colour.encoder.conv4.prelu, gray.jscc.encoder.conv4.prelu)
copy_prelu(colour.encoder.conv5.prelu, gray.jscc.encoder.conv5.prelu)

# ── Decoder: tconv1 sliced, tconv2-5 direct ───────────────────────────────────
copy_sliced(colour.decoder.tconv1.transconv, gray.jscc.decoder.tconv1.transconv, is_transpose=True)
copy_direct(colour.decoder.tconv2.transconv, gray.jscc.decoder.tconv2.transconv)
copy_direct(colour.decoder.tconv3.transconv, gray.jscc.decoder.tconv3.transconv)
copy_direct(colour.decoder.tconv4.transconv, gray.jscc.decoder.tconv4.transconv)
copy_direct(colour.decoder.tconv5.transconv, gray.jscc.decoder.tconv5.transconv)

copy_prelu(colour.decoder.tconv1.activate, gray.jscc.decoder.tconv1.activate)
copy_prelu(colour.decoder.tconv2.activate, gray.jscc.decoder.tconv2.activate)
copy_prelu(colour.decoder.tconv3.activate, gray.jscc.decoder.tconv3.activate)
copy_prelu(colour.decoder.tconv4.activate, gray.jscc.decoder.tconv4.activate)
# tconv5.activate is Sigmoid (no params) -- nothing to copy.

# adapter (1->3 channels) keeps its own safe replication-preserving init, untouched.

# ── Quick sanity eval ──────────────────────────────────────────────────────────
print('\nQuick eval (200 STL-10 test images, grayscale-fed through gray model):')
ds = datasets.STL10(os.path.join(os.path.dirname(HERE), 'data'), split='test',
                    download=False, transform=transforms.ToTensor())
rgb = torch.stack([ds[i][0] for i in range(200)])
gray_input = (0.299*rgb[:,0:1] + 0.587*rgb[:,1:2] + 0.114*rgb[:,2:3])

def psnr_255(mse_255):
    return 100.0 if mse_255 < 1e-10 else 10*math.log10(255.0**2/mse_255)

gray.eval()
with torch.no_grad():
    recon = gray(gray_input)
    target = gray_input.repeat(1,3,1,1)
    mse_255 = F.mse_loss(recon*255, target*255).item()
print(f'  Transfer-init PSNR = {psnr_255(mse_255):.2f} dB')

torch.save({'state_dict': gray.state_dict(), 'c': C_NEW, 'snr_train': 19.0,
            'transfer_from': COLOUR_CKPT, 'transfer_from_epoch': ck['epoch'],
            'epoch': 0, 'best_psnr': 0.0,
            'history': {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_psnr': [], 'lr': []}},
           OUT_PATH)
print(f'\nSaved transfer-init checkpoint -> {OUT_PATH}')
