# JPEG Bitstream Transmission Over AWGN Channel

## Overview

This document explains the full process of transmitting a JPEG-compressed image over an
Additive White Gaussian Noise (AWGN) channel at the **bit level**, measuring when the image
becomes unrecoverable (corrupted) as channel noise increases.

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

## Corruption Rate vs SNR Curve

```
Corruption
Rate (R)
  1.0 |**
      |  **
  0.8 |    *
      |     *
  0.6 |      *
      |       **
  0.4 |         *
      |          **
  0.2 |            ***
      |               ****
  0.0 +----+----+----+----+----+----
      0    4    8   12   16   20
                  SNR (dB)
```

---

## PSNR vs SNR (Non-Corrupted Images)

For images that are successfully decoded:

- At **high SNR** (> 14 dB): almost no bit errors — PSNR approaches the JPEG-only
  compression PSNR (~30–35 dB at Q=73)
- At **medium SNR** (10–14 dB): some scan-data errors produce visible artefacts —
  PSNR drops below JPEG-only quality
- At **low SNR** (< 10 dB): very few images survive; those that do had 0 errors by
  chance — PSNR equals clean JPEG PSNR but sample size is very small

---

## Connection to the ToR 2026 Project

This experiment models the classical **separate source-channel coding** pipeline:

```
[Image]
  -> JPEG Encoder  (source coding, Q=73)
  -> BPSK Modulator
  -> AWGN Channel
  -> BPSK Hard Decoder
  -> JPEG Decoder
  -> [Recovered Image or CORRUPTED]
```

**Key limitation**: JPEG was designed for compression, not channel robustness.
It has zero error protection — a single bit flip in the wrong place destroys the image.

The goal of this project is to design a **Joint Source-Channel Coding (JSCC)** deep
learning system that encodes images directly into channel-robust representations,
outperforming this separate pipeline, especially at low SNR (0–10 dB) where JPEG
transmission fails almost completely.

---

*Dataset: STL-10 (96×96 RGB) | JPEG Quality: Q=73 | Channel: BPSK over AWGN | SNR range: 0–20 dB*
