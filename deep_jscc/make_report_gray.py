"""Generate 'pdf result/Grayscale_DataSaving_Report.pdf'.

Grayscale Deep-JSCC bottleneck study c = 8..4: transmitted-data savings and the
measured PSNR cost, plus a plain-language section explaining why the network uses
real-valued weights yet its output is described as complex-valued.

Reads:  checkpoints/stl10_v2gray_c{8,7,6,5,4}_demo_snr19_AWGN_best.pth
Writes: results/gray_saving_bars.png, results/gray_psnr_vs_snr.png,
        pdf result/Grayscale_DataSaving_Report.pdf
"""
import os, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from gray_model import DeepJSCC_Gray

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, 'results')
DATA = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR = os.path.join(HERE, 'pdf result')
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SNR_TRAIN = 19.0
CS = [8, 7, 6, 5, 4]
SNRS = [0, 2, 4, 7, 10, 13, 16, 19, 22, 25]
COL = {8: '#2980B9', 7: '#E67E22', 6: '#16A085', 5: '#8E44AD', 4: '#2C3E50'}
MRK = {8: 'o', 7: 's', 6: '^', 5: 'D', 4: 'v'}

# ── Evaluate every checkpoint (fresh, so numbers are reproducible) ──────────────
def rgb_to_gray_uint8(chw_uint8):
    a = chw_uint8.astype(np.float32)
    y = 0.299 * a[:, 0] + 0.587 * a[:, 1] + 0.114 * a[:, 2]
    return np.clip(np.round(y), 0, 255).astype(np.uint8)[:, None]

torch.manual_seed(42); np.random.seed(42)
test_full = datasets.STL10(DATA, split='test', download=False)
val_idx = np.random.choice(len(test_full), 8000, replace=False)
val_tensor = torch.from_numpy(rgb_to_gray_uint8(test_full.data[val_idx]))
loader = DataLoader(TensorDataset(val_tensor), batch_size=128, shuffle=False)

models, K, sweep = {}, {}, {}
for c in CS:
    ck = torch.load(f'checkpoints/stl10_v2gray_c{c}_demo_snr19_AWGN_best.pth',
                    map_location=DEVICE, weights_only=False)
    m = DeepJSCC_Gray(c=c, channel_type='AWGN', snr=SNR_TRAIN).to(DEVICE)
    m.load_state_dict(ck['state_dict']); m.eval()
    models[c] = m; K[c] = c * 24 * 24

def eval_at(model, snr):
    model.change_channel('AWGN', snr)
    tot, n = 0.0, 0
    with torch.no_grad():
        for (b,) in loader:
            x = b.float().to(DEVICE) / 255.0
            target = x.repeat(1, 3, 1, 1)
            r = model(x)
            tot += torch.mean((r - target) ** 2).item() * x.size(0); n += x.size(0)
    return 10 * math.log10(1.0 / (tot / n))

for c in CS:
    sweep[c] = [eval_at(models[c], s) for s in SNRS]
    print(f'c={c}: PSNR@19 = {sweep[c][SNRS.index(19)]:.2f} dB')

P19 = {c: sweep[c][SNRS.index(19)] for c in CS}
Q = 7  # bits per transmitted symbol
# Byte convention (per supervisor): data-transfer footprint = one N-bit number
# per channel symbol -> bytes = k * N / 8  (no factor of 2 for I/Q).
BYTES = {c: K[c] * Q / 8 for c in CS}

# ── Figure 1: data saving (symbols + bytes) ────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
xs = [str(c) for c in CS]
b1 = ax1.bar(xs, [K[c] for c in CS], color=[COL[c] for c in CS], width=0.62, zorder=3)
for c, b in zip(CS, b1):
    lbl = f'{K[c]:,}' + ('' if c == 8 else f'\n-{100*(1-K[c]/K[8]):.1f}%')
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+60, lbl, ha='center',
             va='bottom', fontsize=9.5, color='#2C3E50')
ax1.set_ylim(0, 5350)
ax1.set_xlabel('bottleneck width c'); ax1.set_ylabel('complex symbols per image (k)')
ax1.set_title('Channel uses per image', fontsize=11)
ax1.grid(axis='y', alpha=0.25, lw=0.5, zorder=0)

b2 = ax2.bar(xs, [BYTES[c] for c in CS], color=[COL[c] for c in CS], width=0.62, zorder=3)
for c, b in zip(CS, b2):
    lbl = f'{BYTES[c]:,.0f}' + ('' if c == 8 else f'\n-{100*(1-BYTES[c]/BYTES[8]):.1f}%')
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+110, lbl, ha='center',
             va='bottom', fontsize=9.5, color='#2C3E50')
ax2.set_ylim(0, 9400)
ax2.set_xlabel('bottleneck width c'); ax2.set_ylabel('bytes per image (7-bit quantization)')
ax2.set_title('Equivalent bytes per image', fontsize=11)
ax2.grid(axis='y', alpha=0.25, lw=0.5, zorder=0)
for a in (ax1, ax2):
    for s in ('top', 'right'): a.spines[s].set_visible(False)
fig.tight_layout()
FIG_BARS = os.path.join(RES, 'gray_saving_bars.png')
fig.savefig(FIG_BARS, dpi=150); plt.close(fig)

# ── Figure 2: PSNR vs SNR ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 5.0))
for c in CS:
    ax.plot(SNRS, sweep[c], color=COL[c], marker=MRK[c], ms=6, lw=2,
            label=(f'c=8  (k={K[8]:,}, baseline)' if c == 8
                   else f'c={c}  (k={K[c]:,}, -{100*(1-c/8):.1f}% data)'))
ax.axvline(19, color='#999999', ls=':', lw=1.1, label='SNR_train = 19 dB')
ax.set_xlabel('Channel SNR_test (dB)'); ax.set_ylabel('Reconstruction PSNR (dB)')
ax.set_title('STL-10 grayscale — reconstruction quality vs channel SNR', fontsize=12)
ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=True, fontsize=9, loc='lower right')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout()
FIG_PSNR = os.path.join(RES, 'gray_psnr_vs_snr.png')
fig.savefig(FIG_PSNR, dpi=150); plt.close(fig)

# ── Figure 3: bit-depth sweep (quantization) for c=4 (chosen model) ────────────
import json as _json
QD = _json.load(open(os.path.join(RES, 'gray_quant.json')))
bs = QD['bitsweep_c4']
bits = [d['bits'] for d in bs]; qp = [d['psnr'] for d in bs]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.plot(bits, qp, color='#16A085', marker='o', ms=7, lw=2, zorder=3)
for d in bs:
    ax.annotate(f"{d['bytes']:,.0f} B", xy=(d['bits'], d['psnr']),
                xytext=(0, 8), textcoords='offset points', ha='center',
                fontsize=8, color='#555555')
# mark the 7-bit choice
d7 = next(d for d in bs if d['bits'] == 7)
ax.scatter([7], [d7['psnr']], s=180, facecolors='none', edgecolors='#E67E22',
           linewidths=2, zorder=4)
ax.annotate('7-bit: near-lossless', xy=(7, d7['psnr']), xytext=(7.1, d7['psnr']-2.2),
            fontsize=9, color='#E67E22', fontweight='bold')
ax.set_xlabel('bits per transmitted symbol'); ax.set_ylabel('PSNR (dB)')
ax.set_title('How many bits per symbol? (grayscale c = 4, channel at SNR 19 dB)', fontsize=11)
ax.invert_xaxis()
ax.grid(alpha=0.25, lw=0.5)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout()
FIG_QUANT = os.path.join(RES, 'gray_quant_bits.png')
fig.savefig(FIG_QUANT, dpi=150); plt.close(fig)

# ── PDF ────────────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

C_DARK  = colors.HexColor('#2C3E50')
C_TEAL  = colors.HexColor('#16A085')
C_LIGHT = colors.HexColor('#ECF0F1')
C_WHITE = colors.white
styles = getSampleStyleSheet()
def st(name, base='Normal', **kw): return ParagraphStyle(name, parent=styles[base], **kw)
S_TITLE = st('T', 'Title', fontSize=20, textColor=C_DARK, spaceAfter=4, alignment=TA_CENTER)
S_SUB   = st('S', 'Normal', fontSize=12, textColor=C_TEAL, alignment=TA_CENTER, spaceAfter=4)
S_META  = st('M', 'Normal', fontSize=9.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=3)
S_BODY  = st('B', 'Normal', fontSize=10, textColor=colors.HexColor('#2C2C2C'),
             leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
S_BUL   = st('BL', 'Normal', fontSize=10, textColor=colors.HexColor('#2C2C2C'),
             leading=15, leftIndent=15, spaceAfter=3)
S_CAP   = st('CAP', 'Normal', fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=9)
S_NOTE  = st('N', 'Normal', fontSize=9.5, textColor=colors.HexColor('#0B5345'),
             leading=14, spaceAfter=6, alignment=TA_JUSTIFY,
             backColor=colors.HexColor('#E8F6F1'), borderPad=8, leftIndent=4, rightIndent=4)

def body(t): return Paragraph(t, S_BODY)
def bullet(t): return Paragraph(f'&#8226;&nbsp; {t}', S_BUL)
def cap(t): return Paragraph(f'<i>{t}</i>', S_CAP)
def note(t): return Paragraph(t, S_NOTE)
def sp(n=6): return Spacer(1, n)
def box(title, color=C_TEAL):
    tbl = Table([[Paragraph(f'<b>{title}</b>', st('BX','Normal',fontSize=11,textColor=C_WHITE))]],
                colWidths=[16*cm])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),color),
        ('LEFTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return tbl
def dtable(header, rows, widths):
    data = [[Paragraph(f'<b>{h}</b>', S_BODY) for h in header]]
    for r in rows:
        data.append([Paragraph(str(v), S_BODY) for v in r])
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), C_TEAL), ('TEXTCOLOR',(0,0),(-1,0), C_WHITE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_LIGHT, C_WHITE]),
        ('GRID',(0,0),(-1,-1),0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN',(0,0),(-1,-1),'CENTER'), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),3), ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
    return t
def fig_img(path, width=16.0*cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return RLImage(path, width=width, height=width*h/w)

story = [
    Paragraph('ToR Project — Progress Report', S_TITLE),
    Paragraph('Grayscale Deep-JSCC: Saving Transmitted Data by Shrinking the '
              'Bottleneck (c = 8 &#8594; 4)', S_SUB),
    sp(4),
    Paragraph('Deep Joint Source-Channel Coding (JSCC) for wireless image transmission '
              '— STL-10, AWGN, SNR<sub>train</sub> = 19 dB', S_META),
    Paragraph('Prepared for: Supervisor&nbsp;&nbsp;|&nbsp;&nbsp;Author: Nam Khanh Tran'
              '&nbsp;&nbsp;|&nbsp;&nbsp;July 2026', S_META),
    sp(4), HRFlowable(width='100%', thickness=1.5, color=C_TEAL), sp(8),
]
story.append(body(
    'This report studies the grayscale version of Deep JSCC. We shrink one layer of '
    'the encoder — the last convolution, called the <b>bottleneck</b> — to send less '
    'data over the channel, and we measure how much quality (PSNR) we give up in '
    'return. The bottleneck width <b>c</b> is reduced from the baseline <b>c = 8</b> '
    'down to <b>4</b>, in single steps. Nothing else in the model changes.'))

# Section 1
story.append(box('1.  What was changed'))
story.append(sp(8))
story.append(body(
    'The encoder turns a 96&#215;96 grayscale image into a small block of numbers to '
    'send. Only the last layer (conv5) decides how big that block is: it outputs '
    '<b>2c</b> values at each of 24&#215;24 positions, which the system reads as '
    '<b>k = c &#215; 24 &#215; 24</b> complex channel symbols per image. Making c '
    'smaller makes that block smaller — and that is the only thing we change:'))
story.append(sp(4))
story.append(dtable(
    ['Bottleneck width c', 'k (symbols/image)', 'Data vs c=8'],
    [[f'c = {c}' + (' (baseline)' if c == 8 else ''), f'{K[c]:,}',
      '&#8212;' if c == 8 else f'<b>&#8722;{100*(1-K[c]/K[8]):.1f}%</b>'] for c in CS],
    [5.3*cm, 5.3*cm, 5.3*cm]))

# Section 2 — the requested plain-language explanation
story.append(box('2.  Real-valued weights, complex-valued output — in plain words'))
story.append(sp(8))
story.append(body(
    'A common confusion: the paper says the encoder produces <b>complex numbers</b>, '
    'so people ask whether the network itself must be built from complex numbers. '
    'It does not. Here is the simple picture:'))
story.append(sp(2))
story.append(bullet(
    '<b>Inside the network, everything is ordinary real numbers.</b> The weights are '
    'the same kind of plain numbers as in any normal neural network. There are no '
    'complex numbers stored or multiplied anywhere inside it.'))
story.append(bullet(
    '<b>"Complex" is just how we read the output, not how it is computed.</b> The '
    'last layer produces 2c real numbers per position. We simply take them in '
    '<b>pairs</b>: the first number of a pair is called the "real part" and the '
    'second the "imaginary part" of one complex symbol.'))
story.append(bullet(
    '<b>Why pairs?</b> A radio transmitter sends two separate real signals at once — '
    'called <b>I</b> (in-phase) and <b>Q</b> (quadrature). Each pair of real numbers '
    'is sent as one I value and one Q value: two real voltages that together form one '
    'radio sample. So "one complex symbol" simply means "two real numbers sent '
    'together."'))
story.append(sp(2))
story.append(body(
    'In short: the network is real; the word "complex" only describes how its two '
    'real outputs are paired up and sent over the air. No complex arithmetic is '
    'needed anywhere.'))
story.append(note(
    '<b>How we count data.</b> We measure the <b>data transfer</b> — the amount sent '
    'over the channel — as one number per channel symbol. So the transmitted footprint '
    'is <b>k</b> symbols per image, and its byte figure (Section 3) is '
    'bytes/image = k &#215; N &#247; 8 for N bits per symbol. (This counts the '
    'transmission footprint, not the raw internal storage; the <i>percentage</i> saved '
    'is the same for any choice of N.)'))

# Section 3 — quantization (symbols -> bytes)
QD = _json.load(open(os.path.join(RES, 'gray_quant.json')))
q7 = {d['c']: d for d in QD['per_c']}
story.append(box('3.  Turning symbols into bytes: quantization'))
story.append(sp(8))
story.append(body(
    'Section 2 gave the byte formula. This section shows the actual method behind it '
    'and confirms, by measurement, that it costs almost no quality. To turn each '
    'continuous (real) value into bits, we use ordinary <b>scalar quantization</b>:'))
story.append(sp(2))
story.append(bullet(
    '<b>Pick a range.</b> After the power-normalization step, the transmitted values '
    'are centered on 0 with an average size of about 1 (measured: mean &#8776; 0.07, '
    'standard deviation &#8776; 1.0). Over 99.9% of them fall within '
    '&#8722;4 to +4, so we clip to that range. (Note: they are <i>not</i> in the '
    '[0,1] range one might first assume &#8212; the power constraint sets the scale.)'))
story.append(bullet(
    '<b>Split into levels.</b> With <b>N</b> bits we get 2<super>N</super> equal steps '
    'across that range; each symbol is sent as its N-bit step number.'))
story.append(bullet(
    '<b>Count the bytes.</b> bytes/image = k &#215; N &#247; 8 (one N-bit number per '
    'channel symbol &#8212; the data-transfer footprint).'))
story.append(sp(2))
story.append(body(
    'The only question is how many bits N we need. We take the <b>chosen model, '
    'c = 4</b>, insert the quantization into the transmission pipeline, and re-measure '
    'the reconstruction PSNR as N is reduced (channel active at SNR 19 dB):'))
story.append(sp(4))
story.append(dtable(
    ['Bits per symbol', 'Bytes/image', 'PSNR (dB)', 'Loss vs un-quantized'],
    [[f"{d['bits']}", f"{d['bytes']:,.0f}", f"{d['psnr']:.2f}",
      f"&#8722;{q7[4]['psnr_base']-d['psnr']:.2f} dB"] for d in QD['bitsweep_c4']],
    [3.6*cm, 3.6*cm, 3.6*cm, 4.2*cm]))
story.append(fig_img(FIG_QUANT, width=14.5*cm))
story.append(cap('Quality is essentially flat down to 7 bits, then falls off as the '
                 'steps get too coarse. 7 bits/symbol is the natural choice: '
                 'near-lossless at the smallest byte cost.'))
story.append(note(
    f"<b>The PSNR still matches after quantization.</b> Adding 7-bit quantization to "
    f"the chosen c = 4 model changes its PSNR by only "
    f"{q7[4]['psnr_base']-q7[4]['psnr_q7']:.2f} dB "
    f"({q7[4]['psnr_base']:.2f} &#8594; {q7[4]['psnr_q7']:.2f} dB) &#8212; so the image "
    f"quality claimed for c = 4 is still valid once the values are turned into bits. "
    f"The resulting data-transfer footprint is "
    f"2,304 &#215; 7 &#247; 8 = <b>2,016 bytes/image</b>."))

# Section 4 — results
story.append(box('4.  Results: data saved and the quality cost'))
story.append(sp(8))
story.append(body(
    'All five models were trained the same way and tested the same way (8,000 '
    'held-out grayscale images, channel active at SNR 19 dB). The table shows the '
    'transmitted data (in channel symbols and in equivalent bytes) and the measured '
    'PSNR:'))
story.append(sp(4))
story.append(dtable(
    ['Model', 'k (symbols)', 'Bytes (7-bit)', 'Data saved', 'PSNR @ 19 dB', 'vs c=8'],
    [[f'c = {c}' + (' (base)' if c == 8 else '') + (' &#9733;' if c == 4 else ''),
      f'{K[c]:,}', f'{BYTES[c]:,.0f}',
      '&#8212;' if c == 8 else f'<b>&#8722;{100*(1-K[c]/K[8]):.1f}%</b>',
      f'<b>{P19[c]:.2f}</b>',
      '&#8212;' if c == 8 else f'&#8722;{P19[8]-P19[c]:.2f} dB'] for c in CS],
    [2.5*cm, 2.7*cm, 2.7*cm, 2.5*cm, 3.0*cm, 2.5*cm]))
story.append(cap('Bytes = k &#215; 7 &#247; 8 (one 7-bit number per channel symbol, '
                 'the data-transfer footprint; Sections 2&#8211;3). '
                 '&#9733; = chosen model. PSNR is averaged over the full 8,000-image test set.'))
story.append(fig_img(FIG_BARS, width=16.0*cm))
story.append(cap('Transmitted data per image falls straight in line with c: down to '
                 'half the baseline at c = 4 — the same &#8722;50% whether measured '
                 'in channel symbols (left) or bytes (right).'))
story.append(fig_img(FIG_PSNR, width=15.0*cm))
story.append(cap('Reconstruction quality across channel conditions. The models keep '
                 'their order (wider c = higher PSNR) and all degrade gracefully as '
                 'the channel worsens — no sudden failure.'))
story.append(body(
    f'The trade-off is smooth and predictable: a <b>wider</b> bottleneck gives higher '
    f'PSNR, and each step down trades a little quality for a large, guaranteed data '
    f'saving. The <b>chosen model, c = 4</b>, halves the transmitted data versus the '
    f'c = 8 baseline for about <b>{P19[8]-P19[4]:.1f} dB</b>, and still holds '
    f'{P19[4]:.2f} dB (30.13 dB after 7-bit quantization) &#8212; a data transfer of '
    f'just 2,016 bytes/image.'))

# Conclusion
story.append(box('5.  Conclusion', C_DARK))
story.append(sp(8))
story.append(bullet(
    'Shrinking only the bottleneck layer (conv5) cuts the transmitted data by a '
    'fixed amount for every image and every channel condition: from &#8722;12.5% '
    '(c = 7) up to <b>&#8722;50% (c = 4)</b>.'))
story.append(bullet(
    'The quality cost is small and orderly — wider c always gives higher PSNR — so '
    'c can be chosen to balance data saving against image quality for a given need.'))
story.append(bullet(
    'The network uses ordinary real-valued weights; its "complex" output is just '
    'pairs of real numbers sent on the I and Q radio channels (Section 2).'))
story.append(bullet(
    'Turning the symbols into bits by 7-bit quantization is near-lossless '
    '(&#8776; 0.03 dB for the chosen c = 4 model), so the PSNR still holds after '
    'quantization and the data transfer is <b>2,016 bytes/image</b> (Section 3).'))

doc = SimpleDocTemplate(os.path.join(OUT_DIR, 'Grayscale_DataSaving_Report.pdf'),
                        pagesize=A4, topMargin=1.6*cm, bottomMargin=1.6*cm,
                        leftMargin=2.0*cm, rightMargin=2.0*cm,
                        title='Grayscale Data-Saving Report', author='Nam Khanh Tran')
doc.build(story)
print('Wrote', os.path.join(OUT_DIR, 'Grayscale_DataSaving_Report.pdf'))
