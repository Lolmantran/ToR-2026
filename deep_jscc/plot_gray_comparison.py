"""Regenerate results/gray_compare_psnr.png and gray_compare_bytes.png from
gray_c8_eval.json / gray_c6_eval.json (produced by eval_gray.py --c 8 / --c 6).

This script was previously only run as an inline one-off command and never
saved -- reconstructed here so it's rerunnable without digging through chat
history. Verify the input JSONs are fresh (same SNR convention as the current
jscc_model.py/jscc_channel.py -- see HANDOFF.md section 2) before trusting
the output.
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

with open(os.path.join(RES, 'gray_c8_eval.json')) as f:
    c8 = json.load(f)
with open(os.path.join(RES, 'gray_c6_eval.json')) as f:
    c6 = json.load(f)

s8 = [r['snr'] for r in c8['sweep']]; p8 = [r['psnr'] for r in c8['sweep']]
b8 = [r['jscc_new_bytes'] for r in c8['sweep']]
s6 = [r['snr'] for r in c6['sweep']]; p6 = [r['psnr'] for r in c6['sweep']]
b6 = [r['jscc_new_bytes'] for r in c6['sweep']]
jpeg_gray_mean = c8['jpeg_gray_mean_bytes']

# NOTE: this "original" reference curve (colour-pretrained, zero-shot grayscale)
# is hardcoded from an earlier notebook run -- if the colour baseline has been
# retrained since, re-pull these numbers from the notebook's Section 11 output
# instead of trusting this hardcoded list.
orig_snr = [0, 2, 4, 7, 10, 13, 16, 19, 22, 25]
orig_psnr = [21.76, 23.46, 25.01, 27.00, 28.51, 29.52, 30.13, 30.47, 30.65, 30.75]

# ── Chart 1: PSNR vs SNR ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(orig_snr, orig_psnr, 'o-', color='#16A085', lw=2.5,
        label='Original: c=8, replication (colour-pretrained, zero-shot)')
ax.plot(s8, p8, 's--', color='#2980B9', lw=2,
        label='New: c=8, middle-CNN adapter (from scratch)')
ax.plot(s6, p6, '^--', color='#8E44AD', lw=2,
        label='New: c=6, middle-CNN adapter (from scratch, -25% symbols)')
ax.axvline(19, color='grey', ls=':', lw=1, label='SNR_train = 19 dB')
ax.set_xlabel('Channel SNR (dB)'); ax.set_ylabel('PSNR (dB)')
ax.set_title('Grayscale Deep-JSCC: PSNR vs SNR across the three configurations')
ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RES, 'gray_compare_psnr.png'), dpi=120)
print('saved gray_compare_psnr.png')

# ── Chart 2: equivalent bytes vs SNR (c=8 vs c=6, fair comparison) ──────────
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(s8, b8, 's-', color='#2980B9', lw=2, label='c=8 + adapter equivalent size')
ax.plot(s6, b6, '^-', color='#8E44AD', lw=2, label='c=6 + adapter equivalent size (-25% symbols)')
ax.axhline(jpeg_gray_mean, color='#C0392B', ls='--', lw=1.5,
           label=f'JPEG-grayscale average = {jpeg_gray_mean:.0f} B')
ax.axvline(19, color='grey', ls=':', lw=1, label='SNR_train = 19 dB')
ax.set_xlabel('Channel SNR (dB)'); ax.set_ylabel('Equivalent compressed size (bytes)')
ax.set_title('Bandwidth impact of the reduced bottleneck (c=8 vs c=6, both with adapter)')
ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RES, 'gray_compare_bytes.png'), dpi=120)
print('saved gray_compare_bytes.png')
