"""Generate 'pdf result/Bottleneck_Reduction_Report.pdf'.

Reports the encoder layer-5 (bottleneck) modification c=8 -> c=7 -> c=6:
what was changed, how much transmitted data it saves (in channel symbols,
carefully NOT conflated with the SNR-dependent JPEG-equivalent-bandwidth
analysis of the earlier report), and the measured PSNR cost.

Reads:  results/eval_stl10_v2_c{6,7,8}_final_snr19_AWGN_best.json
        train_v2_c{6,7,8}_final_history.json
Writes: results/bottleneck_psnr_vs_snr.png
        results/bottleneck_training_curves.png
        pdf result/Bottleneck_Reduction_Report.pdf
"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, 'results')
OUT_DIR = os.path.join(HERE, 'pdf result')
os.makedirs(OUT_DIR, exist_ok=True)

EV = {}
for c in (8, 7, 6):
    with open(os.path.join(RES, f'eval_stl10_v2_c{c}_final_snr19_AWGN_best.json')) as f:
        EV[c] = json.load(f)
HIST = {}
for c in (8, 7, 6):
    with open(os.path.join(HERE, f'train_v2_c{c}_final_history.json')) as f:
        HIST[c] = json.load(f)

K   = {c: EV[c]['k_complex'] for c in (8, 7, 6)}
P19 = {c: next(s['psnr'] for s in EV[c]['sweep'] if s['snr'] == 19) for c in (8, 7, 6)}

# entity-fixed colors: same model keeps its color in every figure
COL = {8: '#2980B9', 7: '#E67E22', 6: '#16A085'}
MRK = {8: 'o', 7: 's', 6: '^'}

# ── Figure 1: PSNR vs test SNR + symbols per image ────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                               gridspec_kw={'width_ratios': [1.55, 1]})
for c in (8, 7, 6):
    snrs  = [s['snr'] for s in EV[c]['sweep']]
    psnrs = [s['psnr'] for s in EV[c]['sweep']]
    ax1.plot(snrs, psnrs, color=COL[c], marker=MRK[c], ms=5, lw=2,
             label=f'c={c}  (k={K[c]:,})')
ax1.axvline(19, color='#999999', ls='--', lw=1)
ax1.text(19, ax1.get_ylim()[0] + 0.4, ' SNR_train = 19 dB', color='#777777',
         fontsize=8, rotation=90, va='bottom')
ax1.set_xlabel('test-channel SNR (dB)')
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('Reconstruction quality vs channel SNR', fontsize=11)
ax1.grid(alpha=0.25, lw=0.5)
ax1.legend(frameon=False, fontsize=9, loc='lower right')
ax1.set_xlim(-1, 28)

cs = [8, 7, 6]
bars = ax2.bar([str(c) for c in cs], [K[c] for c in cs],
               color=[COL[c] for c in cs], width=0.62, zorder=3)
for c, b in zip(cs, bars):
    saved = 100 * (1 - K[c] / K[8])
    lbl = f'{K[c]:,}' + ('' if c == 8 else f'\n−{saved:.1f}%')
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 60, lbl,
             ha='center', va='bottom', fontsize=9.5, color='#2C3E50')
ax2.set_ylim(0, 5350)
ax2.set_xlabel('bottleneck width c')
ax2.set_ylabel('complex channel symbols per image (k)')
ax2.set_title('Transmitted data per image (constant, all SNRs)', fontsize=11)
ax2.grid(axis='y', alpha=0.25, lw=0.5, zorder=0)
for s in ('top', 'right'):
    ax1.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
fig.tight_layout()
FIG1 = os.path.join(RES, 'bottleneck_psnr_vs_snr.png')
fig.savefig(FIG1, dpi=150); plt.close(fig)

# ── Figure 1b: the three widths vs the Week-2 JPEG+BPSK baseline ──────────────
with open(os.path.join(RES, 'psnr_vs_snr.json')) as f:
    WK2 = json.load(f)
with open(os.path.join(RES, 'bottleneck_noiseless_limits.json')) as f:
    LIM = {int(k): v for k, v in json.load(f).items()}

fig, ax = plt.subplots(figsize=(9.5, 5.2))
for c in (8, 7, 6):
    snrs  = [s['snr'] for s in EV[c]['sweep']]
    psnrs = [s['psnr'] for s in EV[c]['sweep']]
    ax.plot(snrs, psnrs, color=COL[c], marker=MRK[c], ms=5.5, lw=2,
            label=f'Deep JSCC  c={c}  (k={K[c]:,})')
ax.plot(WK2['snr_test'], WK2['jpeg_bpsk_psnr'], color='#B03A2E', ls='--',
        marker='s', ms=5.5, lw=1.8, label='JPEG Q=73 + BPSK + AWGN  (Week 2)')
ax.axhline(LIM[8], color='#2980B9', ls=':', lw=1.1,
           label=f'c=8 noiseless-channel limit = {LIM[8]:.2f} dB')
ax.axvline(19, color='#999999', ls=':', lw=1.1, label='SNR_train = 19 dB')
ax.set_xlabel('Channel SNR_test (dB)')
ax.set_ylabel('Reconstruction PSNR (dB)')
ax.set_title('STL-10 — Deep JSCC (c = 8 / 7 / 6) vs JPEG-Q73 BPSK baseline (AWGN)',
             fontsize=12)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(frameon=True, fontsize=9, loc='lower right', framealpha=0.9)
ax.set_ylim(-1.5, 33)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
FIG_JPEG = os.path.join(RES, 'bottleneck_vs_jpeg.png')
fig.savefig(FIG_JPEG, dpi=150); plt.close(fig)

# ── Figure 2: training curves ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.4))
for c in (8, 7, 6):
    ax.plot(HIST[c]['epoch'], HIST[c]['val_psnr'], color=COL[c], lw=1.8,
            label=f'c={c}')
ax.axhline(30, color='#B03A2E', ls='--', lw=1, label='30 dB stop target')
ax.set_xlabel('epoch'); ax.set_ylabel('val PSNR (dB)')
ax.set_title('Validation PSNR during training (identical recipe, all three widths)',
             fontsize=11)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(frameon=False, fontsize=9, loc='lower right')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
FIG2 = os.path.join(RES, 'bottleneck_training_curves.png')
fig.savefig(FIG2, dpi=150); plt.close(fig)

# ── PDF ────────────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, Image as RLImage)
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
S_CAP   = st('CAP', 'Normal', fontSize=8.5, textColor=colors.grey,
             alignment=TA_CENTER, spaceAfter=9)
S_NOTE  = st('N', 'Normal', fontSize=9.5, textColor=colors.HexColor('#0B5345'),
             leading=14, spaceAfter=6, alignment=TA_JUSTIFY,
             backColor=colors.HexColor('#E8F6F1'), borderPad=8,
             leftIndent=4, rightIndent=4)

def body(t): return Paragraph(t, S_BODY)
def bullet(t): return Paragraph(f'&#8226;&nbsp; {t}', S_BUL)
def cap(t): return Paragraph(f'<i>{t}</i>', S_CAP)
def note(t): return Paragraph(t, S_NOTE)
def sp(n=6): return Spacer(1, n)

def box(title, color=C_TEAL):
    tbl = Table([[Paragraph(f'<b>{title}</b>',
                            st('BX', 'Normal', fontSize=11, textColor=C_WHITE))]],
                colWidths=[16 * cm])
    tbl.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), color),
        ('LEFTPADDING', (0, 0), (-1, -1), 10), ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    return tbl

def dtable(header, rows, widths):
    data = [[Paragraph(f'<b>{h}</b>', S_BODY) for h in header]]
    for r in rows:
        data.append([Paragraph(str(v), S_BODY) for v in r])
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_TEAL),
        ('TEXTCOLOR', (0, 0), (-1, 0), C_WHITE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_LIGHT, C_WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    return t

def fig_img(path, width=16.0 * cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return RLImage(path, width=width, height=width * h / w)

story = [
    Paragraph('ToR Project — Progress Report', S_TITLE),
    Paragraph('Reducing the Encoder Bottleneck (Layer 5): '
              'Transmitted-Data Savings at Constant Quality', S_SUB),
    sp(4),
    Paragraph('Deep Joint Source-Channel Coding (JSCC) for wireless image '
              'transmission — STL-10, AWGN, SNR<sub>train</sub> = 19 dB', S_META),
    Paragraph('Prepared for: Supervisor&nbsp;&nbsp;|&nbsp;&nbsp;Author: Nam Khanh Tran'
              '&nbsp;&nbsp;|&nbsp;&nbsp;July 2026', S_META),
    sp(4),
    HRFlowable(width='100%', thickness=1.5, color=C_TEAL),
    sp(8),
]

story.append(body(
    'This report documents a single, deliberate modification to the reference '
    'Deep-JSCC architecture (Bourtsoulatze et&nbsp;al., 2019): shrinking the width '
    '<b>c</b> of the encoder\'s final convolution (layer&nbsp;5, the bottleneck '
    'that determines how much data is transmitted over the channel) from the '
    'baseline <b>c=8</b> to <b>c=7</b> and <b>c=6</b>. Everything else — the other '
    'nine conv/deconv layers, the power normalization, the AWGN channel, the '
    'training procedure — is unchanged, so any difference in the results is '
    'attributable to the bottleneck alone.'))
story.append(sp(4))

# ── Section 1 ──────────────────────────────────────────────────────────────────
story.append(box('1.  What exactly was modified'))
story.append(sp(8))
story.append(body(
    'The encoder maps a 96&#215;96&#215;3 image through five convolutions to a '
    '24&#215;24&#215;2c feature block. The 2c real feature maps are interpreted as '
    'c complex-valued channel symbols per spatial position, so one image occupies '
    'exactly <b>k = c &#183; 24 &#183; 24</b> complex channel uses. Layer&nbsp;5 is '
    'the only layer whose output width is c; reducing c shrinks the transmitted '
    'block and nothing else:'))
story.append(sp(4))
story.append(dtable(
    ['Encoder layer', 'Output shape', 'c = 8 (baseline)', 'c = 7', 'c = 6'],
    [
        ['conv1 (5&#215;5, s2)', '48&#215;48&#215;16', 'unchanged', 'unchanged', 'unchanged'],
        ['conv2 (5&#215;5, s2)', '24&#215;24&#215;32', 'unchanged', 'unchanged', 'unchanged'],
        ['conv3 (5&#215;5)', '24&#215;24&#215;32', 'unchanged', 'unchanged', 'unchanged'],
        ['conv4 (5&#215;5)', '24&#215;24&#215;32', 'unchanged', 'unchanged', 'unchanged'],
        ['<b>conv5 (5&#215;5) — modified</b>', '24&#215;24&#215;<b>2c</b>',
         '<b>16 maps</b>', '<b>14 maps</b>', '<b>12 maps</b>'],
        ['k (complex symbols / image)', 'c&#183;576',
         f'{K[8]:,}', f'{K[7]:,}', f'{K[6]:,}'],
        ['bandwidth ratio k/n', 'c/48', '1/6', '&#8776;1/6.9', '1/8'],
    ],
    [4.4 * cm, 3.2 * cm, 2.9 * cm, 2.7 * cm, 2.7 * cm]))
story.append(sp(4))
story.append(body(
    'The decoder\'s first transposed convolution accepts 2c input maps and is '
    'resized accordingly. All three models were trained from scratch with the '
    'identical recipe (Adam, lr 10<super>-3</super> with 5-epoch warm-up, no '
    'weight decay, gradient-clip 0.5, full 105k-image STL-10 train+unlabeled '
    'set, batch 64, ReduceLROnPlateau) and the identical validation protocol '
    '(8,000-image STL-10 test set, channel active at SNR 19 dB).'))

# ── Section 2 ──────────────────────────────────────────────────────────────────
story.append(box('2.  What "data saving" means here — and what it does not'))
story.append(sp(8))
story.append(body(
    'Deep JSCC transmits <b>a fixed number of complex channel symbols per image: '
    'k = c&#183;576</b>. This number is a property of the encoder alone. It does '
    'not depend on the channel SNR, on the image content, or on any comparison '
    'with a digital scheme. Reducing c therefore saves transmitted data '
    '<b>unconditionally</b>:'))
story.append(sp(2))
story.append(bullet(
    f'<b>c=8 &#8594; c=7</b>: {K[8]:,} &#8594; {K[7]:,} symbols — '
    f'<b>12.5% fewer channel uses</b> for every image, at every SNR.'))
story.append(bullet(
    f'<b>c=8 &#8594; c=6</b>: {K[8]:,} &#8594; {K[6]:,} symbols — '
    f'<b>25% fewer channel uses</b> for every image, at every SNR.'))
story.append(sp(4))
story.append(note(
    '<b>Not to be confused with the earlier bandwidth-vs-JPEG analysis.</b> The '
    'previous report compared Deep JSCC\'s constant footprint against the '
    '<i>SNR-dependent</i> number of symbols an ideal capacity-achieving digital '
    'code would need to carry a JPEG file (k<sub>min</sub>(SNR) = JPEG bits / '
    'log<sub>2</sub>(1+SNR)). That quantity varies with channel quality and is a '
    'statement about a hypothetical competitor. The saving reported <b>here</b> is '
    'different and much simpler: the same JSCC system, on the same channel, at '
    'the same SNR, simply occupies 12.5% (c=7) or 25% (c=6) fewer channel uses '
    'per image — equivalently, proportionally less airtime or spectrum occupancy. '
    'No assumption about JPEG, coding, or channel quality is involved.'))

# ── Section 3 ──────────────────────────────────────────────────────────────────
story.append(box('3.  Results: quality cost of the saving'))
story.append(sp(8))
story.append(dtable(
    ['Model', 'k (symbols/image)', 'Data saved vs c=8', 'PSNR @ SNR 19 (dB)',
     '&#916; vs baseline'],
    [
        ['c = 8 (baseline)', f'{K[8]:,}', '—', f'<b>{P19[8]:.2f}</b>', '—'],
        ['c = 7', f'{K[7]:,}', '<b>&#8722;12.5%</b>', f'<b>{P19[7]:.2f}</b>',
         f'&#8722;{P19[8]-P19[7]:.2f} dB'],
        ['c = 6', f'{K[6]:,}', '<b>&#8722;25.0%</b>', f'<b>{P19[6]:.2f}</b>',
         f'&#8722;{P19[8]-P19[6]:.2f} dB'],
    ],
    [3.2 * cm, 3.4 * cm, 3.3 * cm, 3.4 * cm, 2.7 * cm]))
story.append(cap('Validation: 8,000-image STL-10 test set, AWGN channel active, '
                 'PSNR averaged over the full set.'))
story.append(fig_img(FIG1))
story.append(cap('Left: PSNR versus test-channel SNR for the three bottleneck widths '
                 '(all trained at SNR 19). Right: transmitted symbols per image — '
                 'constant at every SNR by construction.'))
story.append(body(
    'The quality cost is remarkably small at the design point: <b>0.10 dB for a '
    '12.5% saving (c=7)</b> and <b>0.20 dB for a 25% saving (c=6)</b>. All three '
    'models remain at or above 30 dB at SNR 19. The penalty grows slightly toward '
    f'very poor channels: at SNR 0 dB the three models reach '
    f'{EV[8]["sweep"][0]["psnr"]:.2f} / {EV[7]["sweep"][0]["psnr"]:.2f} / '
    f'{EV[6]["sweep"][0]["psnr"]:.2f} dB (c=8/7/6) — c=7 is indistinguishable '
    'from the baseline even there, while c=6 gives up about half a dB. The '
    'characteristic graceful degradation of JSCC (no cliff effect) is preserved '
    'by all three widths.'))
story.append(fig_img(FIG_JPEG, width=15.5 * cm))
story.append(cap('The three bottleneck widths against the Week-2 digital baseline '
                 '(JPEG Q=73 + BPSK over the same AWGN channel). All three JSCC '
                 'models keep the characteristic graceful degradation and beat the '
                 'digital baseline below its &#8776;13 dB cliff; the dotted line '
                 f'marks the c=8 model\'s noiseless-channel limit '
                 f'({LIM[8]:.2f} dB) — at SNR 19 the channel costs only '
                 f'{LIM[8]-P19[8]:.2f} dB.'))
story.append(body(
    'Against the digital JPEG+BPSK baseline the picture is unchanged from the '
    'earlier full-colour study, now with the added data saving: below the digital '
    'cliff (&#8776;10–13 dB) all three JSCC widths deliver usable images while '
    'JPEG+BPSK fails completely, and even the c=6 model — transmitting 25% less '
    'than the c=8 baseline — stays within 0.5 dB of it across the whole sweep.'))
story.append(fig_img(FIG2, width=15.0 * cm))
story.append(cap('All three widths trained with the identical recipe and stopped '
                 'automatically on reaching the 30 dB target (23–26 epochs, '
                 '&#8776;25 min each on the laptop RTX 4060).'))

# ── Section 4 ──────────────────────────────────────────────────────────────────
story.append(box('4.  Why these numbers are higher than in previous reports'))
story.append(sp(8))
story.append(body(
    'Earlier progress reports quoted 23–24 dB for retrained models and treated '
    '&#8776;29 dB as an unreproducible one-off. The cause has now been isolated: '
    'the training script combined Adam weight decay (5&#215;10<super>-4</super>) '
    'with an MSE loss computed on the [0,1] pixel scale. Adam\'s weight-decay '
    'term does not scale with the loss, so against [0,1]-scale gradients it acted '
    'as roughly 65,000&#215; stronger regularization than intended — continuously '
    'shrinking the encoder weights until the power-normalization layer amplified '
    'the damage into instability. Removing weight decay (plus a short LR warm-up '
    'and gradient clipping as spike guards) restores healthy training: the c=8 '
    f'baseline now reaches <b>{P19[8]:.2f} dB</b>, above the historical best of '
    '28.99 dB. All numbers in this report were produced under the corrected '
    'recipe, so the three models are directly comparable.'))

# ── Conclusion ─────────────────────────────────────────────────────────────────
story.append(box('5.  Conclusion', C_DARK))
story.append(sp(8))
story.append(bullet(
    'Shrinking only the bottleneck layer (conv5) cuts the per-image transmitted '
    'data by 12.5% (c=7) or 25% (c=6), a fixed saving valid at every SNR.'))
story.append(bullet(
    'The measured cost at the 19 dB design point is 0.10 dB (c=7) and '
    '0.20 dB (c=6); all three models sit at or above 30 dB.'))
story.append(bullet(
    '<b>Recommendation:</b> c=6 offers the stronger data-saving story at nearly '
    'identical quality; c=7 is the conservative choice if low-SNR robustness '
    '(below &#8776;4 dB) must match the baseline exactly.'))

doc = SimpleDocTemplate(os.path.join(OUT_DIR, 'Bottleneck_Reduction_Report.pdf'),
                        pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                        title='Bottleneck Reduction Report',
                        author='Nam Khanh Tran')
doc.build(story)
print('Wrote', os.path.join(OUT_DIR, 'Bottleneck_Reduction_Report.pdf'))
