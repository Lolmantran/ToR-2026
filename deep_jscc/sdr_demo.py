"""SDR-style demo for Deep-JSCC: image <-> bitstream.

Two functions, as requested:

    bits  = encode_to_bits(image)      # TRANSMITTER: image  -> bitstream
    image = decode_from_bits(bits)     # RECEIVER:    bitstream -> image

The bitstream is the thing that would travel "over the air" in a real
Software-Defined-Radio link. Between the two functions a normal SDR would
modulate the bits (QPSK/QAM), send them, and demodulate back to bits; here we
assume that digital link is reliable and simulate the *wireless channel* the
network was trained on (AWGN at SNR 19 dB) on the recovered symbols before the
neural decoder. This is "option (a)": the analog channel acts on the JSCC
symbols, which is exactly how the model was trained.

Pipeline
--------
    TX:  image -> [neural encoder] -> continuous symbols z (k complex = 2k real)
                -> [quantize: clip +/-4, 7 bits] -> pack -> bits
    RX:  bits -> unpack -> [dequantize] -> z_hat
              -> [AWGN channel, SNR 19] -> [neural decoder] -> image

Default model: grayscale c = 4 (checkpoints/stl10_v2gray_c4_demo...). Pass a
different loaded model to use colour or another bottleneck width.
"""
import os
import numpy as np
import torch

from jscc_channel import Channel
from gray_model import DeepJSCC_Gray
from jscc_model import DeepJSCC

HERE   = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Quantization settings (must match the report / measurements) ───────────────
CLIP  = 4.0   # symbols are power-normalized: mean~0, std~1, |value|<4 covers 99.9%
NBITS = 7     # bits per real value; 7 is near-lossless (measured -0.03 dB at c=4)


# ── Model loading (cached) ─────────────────────────────────────────────────────
_MODELS = {}

def get_model(mode='gray', c=4, snr_train=19.0):
    """Load (and cache) a trained Deep-JSCC model.

    mode='gray' -> DeepJSCC_Gray (1-channel input), mode='colour' -> DeepJSCC.
    """
    key = (mode, c)
    if key in _MODELS:
        return _MODELS[key]
    if mode == 'gray':
        ckpt = os.path.join(HERE, 'checkpoints',
                            f'stl10_v2gray_c{c}_demo_snr19_AWGN_best.pth')
        model = DeepJSCC_Gray(c=c, channel_type='AWGN', snr=snr_train)
    else:
        ckpt = os.path.join(HERE, 'checkpoints',
                            f'stl10_v2_c{c}_final_snr19_AWGN_best.pth')
        model = DeepJSCC(c=c, channel_type='AWGN', snr=snr_train)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state['state_dict'])
    model.to(DEVICE).eval()
    model._mode, model._c = mode, c
    _MODELS[key] = model
    return model


# ── Small helpers ──────────────────────────────────────────────────────────────
def _rgb_to_gray(u8):
    """(H,W,3) uint8 -> (H,W) uint8 luminance (BT.601)."""
    a = u8.astype(np.float32)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def _prep_image(image, model):
    """Accept a file path or a numpy array; return a (1,C,96,96) tensor in [0,1]."""
    if isinstance(image, str):
        from PIL import Image
        image = np.array(Image.open(image))
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if model._mode == 'gray':
        if image.ndim == 3:
            image = _rgb_to_gray(image)
        if image.shape != (96, 96):
            from PIL import Image
            image = np.array(Image.fromarray(image).resize((96, 96)))
        x = torch.from_numpy(image).float().div(255.0)[None, None]   # (1,1,96,96)
    else:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[:2] != (96, 96):
            from PIL import Image
            image = np.array(Image.fromarray(image).resize((96, 96)))
        x = torch.from_numpy(image).float().div(255.0).permute(2, 0, 1)[None]  # (1,3,96,96)
    return x.to(DEVICE)


def _quantize(z, nbits=NBITS, clip=CLIP):
    """Real array -> integer level indices in [0, 2^nbits - 1]."""
    levels = 2 ** nbits
    step = 2 * clip / (levels - 1)
    zc = np.clip(z, -clip, clip)
    return np.round((zc + clip) / step).astype(np.int64)


def _dequantize(idx, nbits=NBITS, clip=CLIP):
    """Level indices -> reconstructed real values."""
    levels = 2 ** nbits
    step = 2 * clip / (levels - 1)
    return idx.astype(np.float32) * step - clip


def _ints_to_bits(idx, nbits=NBITS):
    """Integer levels -> flat uint8 bit array (MSB first), length len(idx)*nbits."""
    shifts = np.arange(nbits - 1, -1, -1)
    return ((idx[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _bits_to_ints(bits, nbits=NBITS):
    """Flat bit array -> integer levels (inverse of _ints_to_bits)."""
    b = np.asarray(bits, dtype=np.int64).reshape(-1, nbits)
    weights = (1 << np.arange(nbits - 1, -1, -1))
    return (b * weights).sum(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
#  THE TWO FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def encode_to_bits(image, model=None, nbits=NBITS, clip=CLIP):
    """TRANSMITTER.  image (path or numpy array) -> bitstream (numpy uint8 of 0/1).

    Steps: neural encoder -> continuous symbols -> quantize -> pack into bits.
    No channel noise here; the symbols are the clean transmitted signal.
    """
    model = model or get_model()
    x = _prep_image(image, model)
    with torch.no_grad():
        if model._mode == 'gray':
            z = model.jscc.encoder(model.adapter(x))
        else:
            z = model.encoder(x)
    z = z.squeeze(0).cpu().numpy().astype(np.float32)   # (2c, 24, 24)
    idx = _quantize(z, nbits, clip)
    return _ints_to_bits(idx.reshape(-1), nbits)


def decode_from_bits(bits, snr_db=19.0, model=None, nbits=NBITS, clip=CLIP):
    """RECEIVER.  bitstream -> reconstructed image (numpy uint8).

    Steps: unpack bits -> dequantize -> simulate the AWGN wireless channel
    (SNR = snr_db, the link the network was trained for) -> neural decoder.
    Returns an (H,W) grayscale image, or (H,W,3) colour, depending on the model.
    Use snr_db='clean' to skip the channel (perfect link).
    """
    model = model or get_model()
    c = model._c
    n_real = 2 * c * 24 * 24
    idx = _bits_to_ints(bits, nbits)[:n_real]
    z = _dequantize(idx, nbits, clip).reshape(2 * c, 24, 24)
    z = torch.from_numpy(z).float().unsqueeze(0).to(DEVICE)   # (1, 2c, 24, 24)

    if snr_db != 'clean':
        z = Channel('AWGN', float(snr_db)).to(DEVICE)(z)      # simulate the channel

    with torch.no_grad():
        x_hat = (model.jscc.decoder(z) if model._mode == 'gray'
                 else model.decoder(z))
    img = x_hat.squeeze(0).clamp(0, 1).cpu().numpy()
    if model._mode == 'gray':
        img = img.mean(axis=0)                                # 3 replicated ch -> 1
    else:
        img = img.transpose(1, 2, 0)                          # CHW -> HWC
    return (img * 255).round().astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  Round-trip demo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import math
    from torchvision import datasets

    model = get_model('gray', c=4)
    c = model._c
    k = c * 24 * 24

    # grab a few STL-10 test images
    test = datasets.STL10(os.path.join(os.path.dirname(HERE), 'data'),
                          split='test', download=False)
    imgs = [test.data[i].transpose(1, 2, 0) for i in range(6)]   # HWC RGB

    # --- single-image round trip (for the picture) ---
    img0 = imgs[0]
    bits = encode_to_bits(img0)
    recon = decode_from_bits(bits, snr_db=19.0)

    n_bits = len(bits)
    print('=== SDR round-trip (grayscale c=4) ===')
    print(f'image in            : 96x96 grayscale')
    print(f'symbols (k)         : {k:,} complex  ({2*k:,} real I/Q values)')
    print(f'bitstream length    : {n_bits:,} bits = {n_bits//8:,} bytes '
          f'(actual, {NBITS} bits per real value)')
    print(f'data-transfer metric: {k*NBITS/8:,.0f} bytes (k x {NBITS} / 8, per-symbol)')
    print(f'first 32 bits       : {"".join(map(str, bits[:32].tolist()))} ...')

    # --- average PSNR over several images (stable number) ---
    def psnr(a, b):
        mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
        return 100.0 if mse < 1e-9 else 10 * math.log10(255.0 ** 2 / mse)

    ps = []
    for im in imgs:
        gin = _rgb_to_gray(im)
        out = decode_from_bits(encode_to_bits(im), snr_db=19.0)
        ps.append(psnr(gin, out))
    print(f'\
round-trip PSNR (6 imgs, SNR 19): {np.mean(ps):.2f} dB  (per-image spread)')
    for snr in ('clean', 19.0, 10.0, 4.0):
        out = decode_from_bits(encode_to_bits(img0), snr_db=snr)
        print(f'  img0 @ SNR {str(snr):>5}: PSNR = {psnr(_rgb_to_gray(img0), out):.2f} dB')

    # --- save a before/after picture ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
        ax[0].imshow(_rgb_to_gray(img0), cmap='gray', vmin=0, vmax=255)
        ax[0].set_title('input (grayscale)')
        ax[1].imshow(recon, cmap='gray', vmin=0, vmax=255)
        ax[1].set_title('decoded @ SNR 19\n{:,} bytes sent'.format(n_bits // 8))
        ax[2].imshow(decode_from_bits(bits, snr_db='clean'), cmap='gray', vmin=0, vmax=255)
        ax[2].set_title('decoded (clean link)')
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle('Deep-JSCC SDR round trip: image -> bits -> image', y=1.02)
        fig.tight_layout()
        out_png = os.path.join(HERE, 'results', 'sdr_roundtrip.png')
        fig.savefig(out_png, dpi=140, bbox_inches='tight')
        print(f'\
saved {out_png}')
    except Exception as e:
        print('plot skipped:', e)
