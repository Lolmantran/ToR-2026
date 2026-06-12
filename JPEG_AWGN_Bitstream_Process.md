# JPEG Bitstream Transmission Over AWGN Channel

## Overview

This document explains the full process of transmitting a JPEG-compressed image over an
Additive White Gaussian Noise (AWGN) channel at the **bit level**, measuring when the image
becomes unrecoverable (corrupted) as channel noise increases.

---

## Concrete Walkthrough — One Image End-to-End

The numbers below come from a real STL-10 image (class: *airplane*, 96×96 RGB) processed
at Q = 73 and two SNR levels: **14 dB** (clean regime) and **10 dB** (transition regime).

---

### A. Load image from STL-10

```
dataset = STL10(root='./data', split='test', download=True, transform=ToTensor())
img_tensor, label = dataset[42]               # class = 'airplane'
img_uint8 = (img_tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
# shape: (96, 96, 3)   dtype: uint8   pixel range: [0, 255]
```

STL-10 stores images as 96×96 RGB (3 channels × 96 × 96 = 27 648 raw bytes per image).
Each pixel has three 8-bit values (R, G, B).

---

### B. JPEG Compress at Q = 73

```python
buf = io.BytesIO()
Image.fromarray(img_uint8).save(buf, format='JPEG', quality=73)
jpeg_bytes = buf.getvalue()   # raw JPEG bitstream as a Python bytes object
```

| Quantity | Value |
|---|---|
| Raw image size | 27 648 bytes |
| JPEG Q=73 size | **1 613 bytes** |
| Compression ratio | **17.1×** |
| Total bits to transmit | **12 904 bits** |

The first two bytes of every JPEG file are always `FF D8` (the Start-Of-Image marker).
In binary:

```
Byte 0 = 0xFF = 1111 1111
Byte 1 = 0xD8 = 1101 1000

First 16 bits of the bitstream:
Index:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Bit:     1  1  1  1  1  1  1  1  1  1  0  1  1  0  0  0
```

The full JPEG file has the internal structure:

| Marker | Role | Sensitivity |
|---|---|---|
| `FF D8` — SOI | Start-of-image | **1 bit flip → file rejected** |
| `FF E0` — APP0 | JFIF header / metadata | High |
| `FF DB` — DQT | Quantisation tables (one per channel) | **Critical — used for every 8×8 block** |
| `FF C0` — SOF | Frame header (width, height, components) | Critical |
| `FF C4` — DHT | Huffman code tables | **Critical — used to decode all scan data** |
| `FF DA` — SOS | Compressed scan data (actual pixel information) | Moderate |
| `FF D9` — EOI | End-of-image | Low |

---

### C. Convert Bytes → Bits

```python
bits = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8)).astype(np.float32)
# bits.shape = (12904,)   values in {0.0, 1.0}
```

```
First 16 bits: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0]
                ← byte 0xFF ──────────────── →← byte 0xD8 ───── →
```

---

### D. BPSK Modulation — Bits → Symbols

Each bit is mapped to a symbol on the real line:

```
bit 0  →  symbol −1   (negative electrical signal)
bit 1  →  symbol +1   (positive electrical signal)

symbols = 2 × bits − 1
```

```
First 16 symbols: [+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,−1,+1,+1,−1,−1,−1]
```

Signal power is exactly 1 (since all symbols are ±1), which simplifies the SNR formula.

---

### E. AWGN Channel — Add Noise

Independent Gaussian noise is added to every symbol:

```
σ = sqrt( 1 / 10^(SNR_dB / 10) )     [signal power = 1 for BPSK]

received[i] = symbol[i] + noise[i]    noise[i] ~ N(0, σ²)
```

| SNR | σ (noise std) | Received values (first 16 symbols) |
|---|---|---|
| **14 dB** | 0.1995 | `+1.34, +0.91, +1.01, +1.08, +0.84, +1.00, +1.00, +0.65, +1.20, +1.12, −1.12, +0.97, +1.10, −1.05, −1.05, −1.29` |
| **10 dB** | 0.3162 | `+0.73, +1.08, +0.52, +1.15, +0.59, +0.89, +0.72, +0.67, +1.37, +1.02, −0.60, +0.95, +1.11, −0.56, −1.06, −1.15` |

Note that all received values still have the correct sign here (no bit error in the first 16
symbols). Errors occur later in the bitstream where noise occasionally flips the sign.

---

### F. Hard-Decision Decoding

A single threshold at **0** recovers the bit:

```
decoded_bit = 1  if received > 0
decoded_bit = 0  if received ≤ 0
```

This is the simplest possible receiver — no soft information, no error correction.

```
Received (10 dB):   +0.73 +1.08 +0.52 +1.15 ... −0.60 ...
Decoded bits:          1     1     1     1   ...    0   ...   ← matches original ✓

But somewhere deeper in the 12 904-bit stream:
  symbol sent  = +1   received = −0.15   decoded = 0   ← BIT ERROR ✗
```

| SNR | Total bit errors | BER |
|---|---|---|
| **14 dB** | **0** out of 12 904 | 0.000000 |
| **10 dB** | **5** out of 12 904 | 0.000387 |

---

### G. Reconstruct Bytes → JPEG Decode Attempt

```python
# pack bits back into bytes (same length as original JPEG)
rec_bytes = np.packbits((decoded_bits > 0.5).astype(np.uint8)).tobytes()[:n_bytes]

# attempt JPEG decode
try:
    img_recovered = np.array(Image.open(io.BytesIO(rec_bytes)).convert('RGB'))
    # SUCCESS — measure PSNR against original
except Exception:
    img_recovered = None   # CORRUPTED — JPEG decoder threw an exception
```

| SNR | Outcome | PSNR vs original |
|---|---|---|
| **14 dB** | **OK** — 0 bit errors, file structure intact | **39.4 dB** |
| **10 dB** | **OK** — 5 bit errors, all landed in scan data | **6.2 dB** (heavy artefacts) |

> **Why does 0 errors give 39.4 dB instead of ∞?**
> JPEG compression itself is lossy. The 39.4 dB is the JPEG compression distortion at Q=73,
> not channel noise — the channel added nothing extra at 14 dB.

> **Why does 5 bit errors give only 6.2 dB?**
> In JPEG, errors in the compressed scan data (SOS) corrupt the Huffman-coded symbol stream.
> One flip misaligns the decoder's parsing position, causing a cascade of wrong coefficient
> values in every 8×8 DCT block decoded after the error position.

---

### H. How Corruption is Detected

**Corruption = JPEG decoder raises an exception.**

The JPEG decoder (`PIL.Image.open().load()`) performs internal consistency checks:

1. **Marker sequence validation** — checks that `FF D8` is present, markers are in a valid order
2. **Huffman table integrity** — verifies code lengths sum correctly
3. **Quantisation table range** — all 64 entries must be in [1, 255]
4. **Image dimensions** — width/height from SOF must be positive and consistent
5. **End-of-image marker** — `FF D9` must be present at the correct position

Any inconsistency raises an exception. The image is then classified as **corrupted** and
counts toward the corruption rate:

```
Corruption Rate @ SNR = N_corrupted / N_total

Example results (N=5000, Q=73):
  SNR= 0 dB  →  5000/5000 (100.0%) corrupted
  SNR=10 dB  →  2991/5000  (59.8%) corrupted
  SNR=11 dB  →  1009/5000  (20.2%) corrupted
  SNR=12 dB  →   198/5000   (4.0%) corrupted
  SNR=13 dB  →    31/5000   (0.6%) corrupted
  SNR=14 dB  →     2/5000   (0.0%) corrupted
  SNR≥16 dB  →     0/5000   (0.0%) corrupted
```

---

### I. Summary: Full Pipeline in One Table

| Stage | Input | Output | Key formula |
|---|---|---|---|
| **Load** | STL-10 index | RGB uint8 (96×96×3) | — |
| **JPEG encode** | RGB uint8 | bytes (≈1–4 KB) | PIL save Q=73 |
| **Unpack bits** | bytes | {0,1} array, length = 8×bytes | `np.unpackbits` |
| **BPSK map** | {0,1} | {−1,+1} | `s = 2b − 1` |
| **AWGN** | {−1,+1} | floats | `r = s + N(0, σ²)`, `σ=1/√10^(SNR/10)` |
| **Hard decide** | floats | {0,1} | `b̂ = (r > 0)` |
| **Pack bytes** | {0,1} | bytes | `np.packbits` |
| **JPEG decode** | bytes | RGB uint8 **or** exception | PIL open/load |
| **PSNR** | original vs recovered | dB | `10 log₁₀(255²/MSE)` |

---

## Step-by-Step Process

### Step 1 — JPEG Compression (Source Coding)

A raw RGB image is compressed by the JPEG encoder at quality factor **Q = 73**.
The encoder outputs a **binary bitstream** — a sequence of bytes representing:

| JPEG Structure | Role |
|---|---|
| SOI marker (`FF D8`) | Start-of-image — must be intact |
| APP0 / EXIF headers | Metadata |
| DQT — Quantization Tables | Defines how DCT coefficients are quantised |
| DHT — Huffman Tables | Defines variable-length entropy codes |
| SOS — Scan Data | Actual compressed pixel information |

The bitstream looks like:

```
... 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | ...
            (byte FF = SOI high byte)           (byte D8 = SOI low byte)
```

Any bit error in the SOI marker, Huffman tables, or quantisation tables will make the
**entire image undecodable**.

---

### Step 2 — BPSK Modulation (Bit-to-Symbol Mapping)

Each bit is mapped to a physical electrical signal using
**BPSK (Binary Phase Shift Keying)**:

```
Bit 0  →  Symbol -1  (negative electric signal)
Bit 1  →  Symbol +1  (positive electric signal)
```

Example:

```
JPEG bits:    0  | 1  | 0  | 0  | 1  | 1  | 0  | 1
BPSK symbols: -1 | +1 | -1 | -1 | +1 | +1 | -1 | +1
```

---

### Step 3 — AWGN Channel (Noise Addition)

The symbols are transmitted over a noisy channel. Gaussian noise `n ~ N(0, σ²)` is
added to every symbol independently:

```
received = symbol + noise
```

Noise standard deviation from SNR:

```
σ = sqrt( signal_power / 10^(SNR_dB / 10) )
  = sqrt( 1 / 10^(SNR_dB / 10) )        [for BPSK, signal power = 1]
```

After the channel, the received values are **continuous** (no longer discrete ±1):

```
Sent:     -1   | +1   | -1   | -1   | +1   | +1
Received: -2.3 | +3.1 | +0.2 | -0.8 | +1.5 | +0.9   <- noisy float values
                         ^
                         This +0.2 was sent as -1 !  -> BIT ERROR
```

---

### Step 4 — Hard Decision Decoding (Recovery)

A threshold at `0` is applied to each received value to recover the bit:

```
if received > 0  →  decoded bit = 1   (bipolar: +1)
if received <= 0 →  decoded bit = 0   (bipolar: -1)
```

Example:

```
Received:       -2.3 | +3.1 | +0.2 | -0.8 | +1.5 | +0.9
Decoded bits:    0   |  1   |  1   |  0   |  1   |  1
                              ^
                         ERROR: was 0, decoded as 1
Decoded bipolar: -1  |  +1  |  +1  |  -1  |  +1  |  +1
```

The decoded stream `-1 | +1 | +1 | -1 | +1 | +1 | ...` is converted back to bytes
and fed into the JPEG decoder.

---

### Step 5 — JPEG Reconstruction (Outcome)

Two outcomes are possible:

#### Outcome A — Successful Recovery (Non-Corrupted)
- No critical bits were flipped (or errors landed in non-critical regions)
- The JPEG decoder parses the file structure successfully
- An image is produced — quality depends on how many scan-data bits were flipped
- **PSNR** is measured against the original image

#### Outcome B — Corruption
- One or more bit errors hit the SOI, APP headers, Huffman tables, or quantisation tables
- The JPEG decoder throws an exception — the file structure is broken
- **The image cannot be recovered**

```
Corruption Rate = N_corrupted / N_total
```

---

## Why JPEG is Extremely Sensitive to Bit Errors

Unlike pixel-domain noise (where each pixel is independently degraded), **one bit error
in the JPEG header destroys the entire image**:

- **SOI / APP markers**: defines the file as a valid JPEG. 1-bit error → file rejected.
- **DHT (Huffman table)**: defines the variable-length code tree used to decode ALL scan data.
  Any corruption here makes the entire image undecodable.
- **DQT (Quantisation table)**: used to dequantise every 8×8 DCT block.
- **SOS header**: specifies which components / scans to decode.

Only errors landing in the compressed scan data (SOS payload) may produce a corrupted-but-decodable
image with visible block artefacts.

---

## Corruption Rate vs SNR — Expected Behaviour

| SNR (dB) | Approx. BER | Expected Behaviour |
|---|---|---|
| 0 dB | ~15.9 % | Almost every image corrupted |
| 4 dB | ~6.3 % | Nearly all corrupted |
| 8 dB | ~0.5 % | Most corrupted (100s of errors per file) |
| 10 dB | ~0.08 % | High corruption |
| 12 dB | ~0.003 % | Transitioning — some survive |
| 14 dB | ~0.0001 % | Most survive |
| 16–20 dB | ~0 % | Near-zero corruption |

The **critical transition SNR** (where corruption drops below 50%) is typically
**10–14 dB** for JPEG files of a few KB.

The corruption curve is a **steep sigmoid** that drops from 1.0 to 0.0 as SNR
increases from ~8 dB to ~16 dB.

---

## What is SNR?

**Signal-to-Noise Ratio (SNR)** measures how much stronger the intended signal is
compared to the background noise, expressed in decibels (dB):

```
SNR_dB = 10 × log₁₀( signal power / noise power )
```

A higher SNR means a cleaner channel. For BPSK with unit signal power (symbols = ±1),
the noise standard deviation is:

```
σ = 1 / √10^(SNR/10)

SNR =  0 dB  →  σ = 1.000  (noise as strong as the signal — many flips)
SNR = 10 dB  →  σ = 0.316  (noise about 1/3 of signal strength)
SNR = 14 dB  →  σ = 0.200  (noise about 1/5 of signal strength)
SNR = 20 dB  →  σ = 0.100  (noise about 1/10 of signal strength)
```

In practice, SNR encodes the physical quality of the wireless or wired channel — a weak
Wi-Fi signal, a satellite link, or interference from other devices all lower the SNR.

---

## What is PSNR?

**Peak Signal-to-Noise Ratio (PSNR)** measures image reconstruction quality — how
similar a received image is to the original:

```
PSNR = 10 × log₁₀( 255² / MSE )

where  MSE = mean squared error per pixel, averaged over all pixels and channels.
```

The value 255 is the peak possible pixel value for an 8-bit image. PSNR is in dB —
higher means better. Rough perceptual guidelines:

| PSNR | Perceptual quality |
|---|---|
| > 40 dB | Essentially identical — differences invisible |
| 30–40 dB | Good quality — minor compression artefacts at most |
| 20–30 dB | Noticeable degradation — visible noise or block artefacts |
| < 20 dB | Severely corrupted — content barely recognisable |
| ∞ | Perfect reconstruction (MSE = 0) |

PSNR is used here instead of a subjective rating because it is deterministic,
fast to compute, and widely used as a baseline metric in image transmission research.
Its main limitation is that it does not always match human perception (two images can
have the same PSNR but look very different), which is why later work often adds SSIM
or a learned perceptual metric.

---

## Why 14 dB and 10 dB Were Chosen as Reference Points

The two SNR values appear repeatedly throughout the walkthrough and plots because they
bracket the **critical transition zone** of the corruption curve:

**14 dB — the "clean" reference point**

At SNR = 14 dB the experimental corruption rate drops to ≈ 0% (only 2 out of 5 000
images failed). This is the point where the BPSK channel becomes effectively error-free
for a ~13 000-bit JPEG file: the expected number of bit errors per image is

```
E[errors] = BER × L = 0.004% × 12 904 ≈ 0.5 bits per image on average
```

so most images have zero errors, and the few that have one error usually survive
because the error lands in the scan data rather than a critical header.
At 14 dB, PSNR equals the JPEG-only compression PSNR (~30–39 dB depending on image
content) — the channel contributes no extra distortion.

**10 dB — the "transitional" reference point**

At SNR = 10 dB the corruption rate is ≈ 60%. This is the steepest part of the sigmoid
curve — small changes in SNR cause large changes in reliability. The expected number
of bit errors per image is

```
E[errors] = 0.08% × 12 904 ≈ 10 bits per image
```

Images that survive (≈ 40%) did so because their errors all landed in the SOS scan
payload. Their PSNR is low (≈ 10 dB) because a single mis-parsed Huffman codeword
cascades into hundreds of wrong DCT coefficients.

Together, these two points define the **boundary of the transition region**:

```
SNR < 10 dB  →  essentially all images destroyed (classical JPEG fails completely)
10–13 dB     →  unreliable — some survive, some do not (transition zone)
SNR ≥ 14 dB  →  essentially all images survive (JPEG succeeds)
```

This is why the ToR 2026 project targets SNRs of **0–14 dB** as the operating range
for the deep JSCC system: these are exactly the conditions where JPEG transmission is
unreliable or impossible.

---

## PSNR vs SNR (Non-Corrupted Images)

For images that are successfully decoded:

- At **high SNR** (≥ 14 dB): zero or near-zero bit errors — PSNR equals the JPEG-only
  compression quality (~30–39 dB at Q=73); the channel adds no extra distortion.
- At **medium SNR** (10–13 dB): a few errors survive into the scan data, misaligning
  Huffman decoding and corrupting entire DCT blocks — PSNR can drop to 6–20 dB even
  for images that technically decoded without an exception.
- At **low SNR** (< 10 dB): almost no images survive; the rare ones that do had 0
  errors purely by chance, so their PSNR equals the clean JPEG PSNR, but the sample
  size is too small to be statistically meaningful.

---
