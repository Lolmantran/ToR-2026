"""Generate Deep_JSCC_Pipeline.pdf — full process explanation with charts,
focused on the grayscale transmission results."""
import os, json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak, Image as RLImage)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, 'results')

with open(os.path.join(RES, 'data_size_summary.json')) as _f:
    DSZ = json.load(_f)

C_DARK   = colors.HexColor('#2C3E50')
C_BLUE   = colors.HexColor('#2980B9')
C_GREEN  = colors.HexColor('#27AE60')
C_ORANGE = colors.HexColor('#E67E22')
C_RED    = colors.HexColor('#C0392B')
C_LIGHT  = colors.HexColor('#ECF0F1')
C_TEAL   = colors.HexColor('#16A085')
C_PURPLE = colors.HexColor('#8E44AD')
C_WHITE  = colors.white

styles = getSampleStyleSheet()
def st(name, base='Normal', **kw): return ParagraphStyle(name, parent=styles[base], **kw)

S_TITLE = st('T', 'Title', fontSize=25, textColor=C_DARK, spaceAfter=6, alignment=TA_CENTER)
S_SUB   = st('S', 'Normal', fontSize=13, textColor=C_TEAL, alignment=TA_CENTER, spaceAfter=18)
S_H1    = st('H1', 'Heading1', fontSize=15, textColor=C_DARK, spaceBefore=14, spaceAfter=6)
S_H2    = st('H2', 'Heading2', fontSize=12, textColor=C_TEAL, spaceBefore=10, spaceAfter=4)
S_BODY  = st('B', 'Normal', fontSize=10, textColor=colors.HexColor('#333333'),
             leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
S_BULLET= st('BL', 'Normal', fontSize=10, textColor=colors.HexColor('#333333'),
             leading=15, leftIndent=16, spaceAfter=3)
S_CAP   = st('CAP', 'Normal', fontSize=9, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10)
S_MATH  = st('M', 'Normal', fontSize=10, textColor=C_DARK, fontName='Courier',
             backColor=C_LIGHT, leftIndent=16, spaceAfter=6)

def h1(t): return Paragraph(t, S_H1)
def h2(t): return Paragraph(t, S_H2)
def body(t): return Paragraph(t, S_BODY)
def bullet(t): return Paragraph(f'&#8226; {t}', S_BULLET)
def cap(t): return Paragraph(f'<i>{t}</i>', S_CAP)
def math(t): return Paragraph(t, S_MATH)
def sp(n=6): return Spacer(1, n)
def hr(): return HRFlowable(width='100%', thickness=1, color=C_LIGHT, spaceAfter=8, spaceBefore=4)

def section_box(title, color=C_TEAL):
    tbl = Table([[Paragraph(f'<b>{title}</b>', st('SB','Normal',fontSize=11,textColor=C_WHITE))]],
                colWidths=[16*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1), color),
        ('LEFTPADDING',(0,0),(-1,-1),10), ('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    return tbl

def info_table(rows, widths=None):
    w = widths or [4.5*cm, 11*cm]
    data = [[Paragraph(f'<b>{k}</b>', S_BODY), Paragraph(v, S_BODY)] for k,v in rows]
    tbl = Table(data, colWidths=w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1), C_LIGHT),
        ('GRID',(0,0),(-1,-1),0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    return tbl

def data_table(header, rows, widths):
    data = [[Paragraph(f'<b>{h}</b>', S_BODY) for h in header]]
    for r in rows:
        data.append([Paragraph(str(v), S_BODY) for v in r])
    tbl = Table(data, colWidths=widths)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), C_TEAL),
        ('TEXTCOLOR',(0,0),(-1,0), C_WHITE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_LIGHT, C_WHITE]),
        ('GRID',(0,0),(-1,-1),0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    return tbl

def fig(path, width=15.5*cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    ratio = h / w
    return RLImage(path, width=width, height=width*ratio)

# ═══════════════════════════════════════════════════════════════════════════════
story = []

# ── Cover ────────────────────────────────────────────────────────────────────
story += [sp(30),
          Paragraph('Deep JSCC', S_TITLE),
          Paragraph('Joint Source-Channel Coding for Wireless Image Transmission', S_SUB),
          Paragraph('Full Pipeline, Results &amp; Grayscale Transmission', S_SUB),
          sp(6),
          HRFlowable(width='80%', thickness=2, color=C_TEAL, hAlign='CENTER'),
          sp(12),
          body('Reference: Bourtsoulatze, E., Kurka, D. B., &amp; Gündüz, D. (2019). '
               '<i>Deep Joint Source-Channel Coding for Wireless Image Transmission.</i> '
               'IEEE Trans. on Cognitive Communications and Networking.'),
          sp(4),
          body('Implementation: PyTorch, trained on STL-10 (96×96 RGB) with an '
               'NVIDIA RTX 4060 GPU · Bandwidth ratio 1/6 · SNR<sub>train</sub>=19 dB'),
          PageBreak()]

# ── 1. Overview ──────────────────────────────────────────────────────────────
story.append(section_box('1.  What is Deep JSCC?'))
story.append(sp(8))
story.append(body(
    'Deep JSCC replaces the conventional two-step "compress, then protect" pipeline '
    '(source coding + channel coding) with a <b>single trained neural network</b> that '
    'maps image pixels directly to complex-valued channel symbols.  The encoder and '
    'decoder are trained jointly, end-to-end, together with a model of the wireless '
    'channel sitting between them — hence the name <i>joint</i> source-channel coding.'))
story.append(sp(6))
story.append(h2('Why not just use JPEG + error-correcting codes?'))
story += [
    bullet('Digital pipelines have a <b>"cliff effect"</b>: below the SNR the channel '
           'code was designed for, the bitstream fails outright and the image is lost.'),
    bullet('Deep JSCC has no hard bit boundary to corrupt — quality degrades '
           '<b>gracefully</b> as the channel worsens.'),
    bullet('The whole pipeline is trained to directly minimise reconstruction error '
           '(MSE), rather than optimising compression and error-protection separately.'),
    sp(8)]
story.append(h2('End-to-end pipeline'))
story.append(math('image  &#8594;  Encoder f&#952;  &#8594;  z &#8712; C^k  &#8594;  AWGN channel  &#8594;  z_hat  &#8594;  Decoder g&#966;  &#8594;  reconstruction'))
story.append(PageBreak())

# ── 2. Architecture ──────────────────────────────────────────────────────────
story.append(section_box('2.  Model Architecture'))
story.append(sp(8))
story.append(h2('Encoder (5 conv layers, PReLU activations)'))
story.append(data_table(
    ['Layer', 'Config', 'Role'],
    [['Conv1', '5×5, stride 2, 3&#8594;16ch', 'Downsample 96&#8594;48, extract low-level features'],
     ['Conv2', '5×5, stride 2, 16&#8594;32ch', 'Downsample 48&#8594;24, mid-level features'],
     ['Conv3', '5×5, stride 1, 32&#8594;32ch', 'Feature refinement'],
     ['Conv4', '5×5, stride 1, 32&#8594;32ch', 'Feature refinement'],
     ['Conv5', '5×5, stride 1, 32&#8594;2c ch', 'Project to 2c channel-symbol planes'],
     ['Power norm', '&#8730;(kP) · z_tilde / &#8730;(z_tilde*z_tilde)', 'Enforce average transmit power constraint']],
    [2.6*cm, 5.2*cm, 7.7*cm]))
story.append(sp(8))
story.append(h2('Decoder (5 transposed-conv layers, mirrors the encoder)'))
story.append(body(
    'The decoder inverts the encoder: 3 stride-1 transposed convolutions refine the '
    'noisy received symbols, then 2 stride-2 transposed convolutions upsample back to '
    '96×96.  The final layer uses a Sigmoid activation so pixel outputs land in [0, 1].'))
story.append(sp(8))
story.append(h2('AWGN channel (non-trainable layer)'))
story.append(math('z_hat = z + n,   n ~ CN(0, &#963;&#178; I_k)'))
story.append(body(
    'The channel is inserted as an ordinary (non-trainable) layer inside the network, '
    'so gradients flow through it during training — the encoder/decoder learn '
    'representations that are inherently robust to the specific noise level '
    '(SNR<sub>train</sub>) they were trained at.'))
story.append(sp(8))
story.append(h2('Bandwidth ratio'))
story.append(math('&#961; = k/n = c &#183; H_enc &#183; W_enc / (C &#183; H &#183; W)'))
story.append(body(
    'For STL-10 (3×96×96) at ratio 1/6: <b>c = 8</b>, giving <b>k = 4,608</b> complex '
    'channel symbols per image (9,216 real values transmitted as I/Q pairs).'))
story.append(PageBreak())

# ── 3. Training ──────────────────────────────────────────────────────────────
story.append(section_box('3.  Training Pipeline'))
story.append(sp(8))
story.append(body(
    'The encoder and decoder are trained jointly on paired data — there is no separate '
    'compression step to supervise, only the final reconstruction.'))
story.append(sp(6))
story.append(info_table([
    ('Dataset',      'STL-10 (105,000 train+unlabeled images, 8,000 test images), 3×96×96 RGB'),
    ('Loss',         'Mean Squared Error between input and reconstruction'),
    ('Optimiser',    'Adam, lr starting at 1e-4, ReduceLROnPlateau schedule'),
    ('Channel',      'AWGN, trained at a fixed SNR<sub>train</sub> = 19 dB'),
    ('Bandwidth ratio', '1/6  (c = 8, k = 4,608 complex symbols/image)'),
    ('Stability',    'Gradient clipping (max-norm 0.5) to prevent training divergence'),
    ('Early stopping', 'Stop when validation PSNR has not improved for a patience window'),
    ('Hardware',     'NVIDIA RTX 4060 laptop GPU (8 GB) via CUDA-enabled PyTorch'),
    ('Result',       '<b>28.99 dB</b> best validation PSNR (full-colour test set)'),
]))
story.append(sp(8))
story.append(h2('Training curve'))
story.append(fig(os.path.join(RES, 'training_curve.png')))
story.append(cap('Figure 1: training/validation MSE and validation PSNR across training epochs.'))
story.append(PageBreak())

# ── 4. Colour transmission results ────────────────────────────────────────────
story.append(section_box('4.  Full-Colour Transmission Results'))
story.append(sp(8))
story.append(body(
    'The trained model is compared against the Week-2 baseline: JPEG (quality 73) '
    'compression followed by BPSK modulation and transmission over the same AWGN '
    'channel.  Both schemes are evaluated across a sweep of test-time channel SNR '
    'values, 0–25 dB.'))
story.append(sp(6))
story.append(fig(os.path.join(RES, 'psnr_vs_snr_comparison.png')))
story.append(cap('Figure 2: PSNR vs channel SNR — Deep JSCC vs JPEG-Q73 + BPSK + AWGN, full-colour STL-10.'))
story.append(sp(6))
story.append(data_table(
    ['SNR (dB)', 'Deep JSCC', 'JPEG+BPSK', 'Gap'],
    [['0',  '21.5', '0.0 (fails)',  '+21.5'],
     ['7',  '26.3', '0.0 (fails)',  '+26.3'],
     ['10', '27.5', '4.3',          '+23.2'],
     ['13', '28.3', '30.2',         '-1.9'],
     ['19', '28.9', '30.8',         '-1.8'],
     ['25', '29.2', '30.8',         '-1.6']],
    [3*cm, 4.2*cm, 4.2*cm, 3*cm]))
story.append(sp(6))
story.append(body(
    '<b>Deep JSCC wins decisively below SNR ≈ 12 dB</b> — JPEG\'s bitstream cannot '
    'survive a noisy channel and the reconstruction collapses to 0 dB, while JSCC '
    'degrades gracefully.  <b>JPEG wins by ~1.5–2 dB once the channel is clean</b> '
    'enough (SNR ≥ 13 dB) for its bitstream to decode intact, since it then reaches '
    'its quality ceiling above the JSCC autoencoder\'s capacity limit.  Averaged '
    'across the full 0–25 dB range, <b>Deep JSCC outperforms JPEG+BPSK by roughly '
    '+10.9 dB</b>.'))
story.append(PageBreak())

# ── 5. Grayscale — headline section ───────────────────────────────────────────
story.append(section_box('5.  Grayscale Image Transmission', C_PURPLE))
story.append(sp(8))
story.append(body(
    'The trained model always expects a 3-channel (RGB) input, since its first '
    'convolution layer is fixed at <b>in_channels=3</b>.  To transmit a grayscale '
    'image, the single luminance channel is <b>replicated into all three input '
    'channels</b> (<i>R = G = B = gray</i>) before it is passed through the encoder — '
    'the model itself is unchanged.'))
story.append(sp(6))
story.append(h2('5.1  Shape adaptation: 1×H×W &#8594; 3×H×W'))
story.append(math('grayscale (H, W)  &#8594;  replicate  &#8594;  (H, W, 3)  &#8594;  batch  &#8594;  (B, 3, H, W)'))
story.append(body(
    'Concretely: <font face="Courier">gray_to_rgb3(gray) = np.stack([gray, gray, gray], '
    'axis=-1)</font>.  A true single-channel tensor is rejected by the first '
    'convolution layer (its weight tensor is shaped [16, 3, 5, 5] — it structurally '
    'requires 3 input channels); duplicating the luminance plane sidesteps that '
    'restriction cleanly, since every input channel still carries real (if repeated) '
    'signal rather than blank zeros.'))
story.append(sp(8))
story.append(fig(os.path.join(RES, 'grayscale_samples.png'), width=13.5*cm))
story.append(cap('Figure 3: sample STL-10 test images — original colour (top row) vs the '
                 'grayscale-replicated input actually fed to the encoder (bottom row).'))
story.append(sp(6))
story.append(h2('5.2  Headline result: grayscale reconstructs BETTER than colour'))
story.append(body(
    'Feeding grayscale-replicated images through the model produces <b>consistently '
    'higher PSNR than full-colour input</b>, at every tested channel SNR:'))
story.append(sp(4))
story.append(data_table(
    ['SNR (dB)', 'Colour PSNR', 'Grayscale PSNR', 'Gap'],
    [['0',  '21.50', '21.76', '+0.26'],
     ['4',  '24.49', '25.00', '+0.51'],
     ['10', '27.45', '28.50', '+1.05'],
     ['16', '28.69', '30.13', '+1.44'],
     ['19', '28.94', '30.47', '+1.53'],
     ['25', '29.14', '30.75', '+1.61']],
    [3*cm, 4.2*cm, 4.2*cm, 3*cm]))
story.append(sp(6))
story.append(body(
    'The gap grows from +0.26 dB at SNR=0 dB to <b>+1.61 dB at SNR=25 dB</b>.  The '
    'likely explanation: replicating R=G=B removes all colour (chroma) variation from '
    'the input, so the fixed channel capacity (k=4,608 complex symbols) no longer '
    'needs to spend any of its limited budget encoding colour information — it can '
    'dedicate the entire budget to luminance and texture detail, which is exactly what '
    'PSNR (computed per-pixel across all 3 identical channels) rewards most.'))
story.append(PageBreak())

# ── 5.3 Grayscale vs JPEG-grayscale ──────────────────────────────────────────
story.append(h2('5.3  Grayscale transmission vs JPEG-grayscale + BPSK + AWGN'))
story.append(body(
    'For a fair apples-to-apples grayscale comparison, the JPEG baseline is also run '
    'in true single-channel mode (no chroma planes at all).'))
story.append(sp(6))
story.append(fig(os.path.join(RES, 'grayscale_psnr_vs_snr.png')))
story.append(cap('Figure 4: Deep JSCC (grayscale-as-RGB) vs JPEG-grayscale Q=73 + BPSK + AWGN, '
                 'with the full-colour Deep JSCC curve shown for reference.'))
story.append(sp(6))
story.append(data_table(
    ['SNR range', 'Winner', 'Why'],
    [['0–10 dB',  'Deep JSCC  (+14 to +28 dB)',
      'JPEG-grayscale bitstream fails below SNR&#8776;10 dB'],
     ['13–25 dB', 'JPEG-grayscale  (+3 dB)',
      'Bitstream survives intact; JPEG ceiling (33.6 dB) exceeds JSCC ceiling (~30.8 dB)'],
     ['Average, 0–25 dB', 'Deep JSCC,  +10.2 dB',
      'Same graceful-degradation pattern as the full-colour comparison']],
    [3.2*cm, 4.8*cm, 7.4*cm]))
story.append(sp(6))
story.append(body(
    'Interestingly, <b>both schemes score higher on grayscale than on colour</b> at '
    'high SNR: JPEG-grayscale\'s ceiling (33.6 dB) exceeds JPEG-colour\'s (30.8 dB) '
    'because there are no chroma planes left to introduce reconstruction error, and '
    'Deep-JSCC grayscale (30.8 dB) similarly exceeds Deep-JSCC colour (29.3 dB) for '
    'the same reason described in 5.2.  The ranking between the two schemes is '
    'unchanged either way — Deep JSCC dominates the low/mid-SNR range where any '
    'realistic wireless channel spends most of its time.'))
story.append(PageBreak())

# ── 5.4 Data size ──────────────────────────────────────────────────────────
story.append(h2('5.4  Compressed data size — the bandwidth asymmetry'))
story.append(body(
    'JPEG produces a literal, countable byte size.  Deep JSCC has no discrete '
    'bitstream at all — it transmits a <b>fixed number of continuous-valued channel '
    'symbols</b> regardless of image content.  This creates a notable asymmetry worth '
    'flagging directly:'))
story.append(sp(6))
story.append(data_table(
    ['Scheme', 'Size / image', 'Notes'],
    [['Raw RGB (uncompressed)', '27,648 bytes', '96&#215;96&#215;3'],
     ['Raw grayscale (uncompressed)', '9,216 bytes', '96&#215;96&#215;1'],
     ['JPEG Q=73, colour', f'{DSZ["jpeg_color"]["mean"]:,.0f} bytes (avg)', '9.9&#215; compression vs raw'],
     ['JPEG Q=73, grayscale', f'{DSZ["jpeg_gray"]["mean"]:,.0f} bytes (avg)', '4.0&#215; compression; <b>18% smaller than colour JPEG</b>'],
     ['Deep JSCC (colour or grayscale)', f'{DSZ["k_complex"]:,} complex symbols<br/>({DSZ["k_real"]:,} real channel uses)',
      '<b>Identical for colour and grayscale input</b> — fixed by architecture']],
    [4.3*cm, 5.2*cm, 5.9*cm]))
story.append(sp(8))
story.append(body(
    '<b>Switching JPEG to grayscale saves bandwidth automatically</b> — the encoder '
    'simply omits the two chroma planes, shrinking the file by 18%.  <b>Deep JSCC gets '
    'no such saving.</b>  Its channel bandwidth k = c·24·24 = 4,608 complex symbols is '
    'fixed by the encoder architecture, not by the input content — feeding a '
    'grayscale-replicated image through the (always 3-channel) encoder still spends '
    'the same k channel symbols.  Two of the three input channels carry duplicate '
    'information that the network happens to make good use of (Section 5.2), but the '
    'channel-use budget itself is never reclaimed.  A pipeline that actually gains '
    'bandwidth from grayscale content would need a purpose-built single-channel model '
    '(in_channels=1 on the first encoder layer, out_channels=1 on the last decoder '
    'layer) — a natural next step beyond this notebook.'))
story.append(PageBreak())

# ── 5.5 Per-image JPEG size variability ─────────────────────────────────────
story.append(h2('5.5  JPEG size varies with image content — Deep JSCC does not'))
story.append(body(
    'The averages above hide substantial per-image spread.  Measured over the same '
    '200-image test pool used throughout this notebook:'))
story.append(sp(4))
story.append(data_table(
    ['Scheme', 'Mean', 'Std dev', 'Min', 'Max'],
    [['JPEG Q=73, colour',
      f'{DSZ["jpeg_color"]["mean"]:,.0f} B', f'{DSZ["jpeg_color"]["std"]:,.0f} B',
      f'{DSZ["jpeg_color"]["min"]:,} B', f'{DSZ["jpeg_color"]["max"]:,} B'],
     ['JPEG Q=73, grayscale',
      f'{DSZ["jpeg_gray"]["mean"]:,.0f} B', f'{DSZ["jpeg_gray"]["std"]:,.0f} B',
      f'{DSZ["jpeg_gray"]["min"]:,} B', f'{DSZ["jpeg_gray"]["max"]:,} B'],
     ['Deep JSCC (any content)',
      f'{DSZ["k_real"]:,} real values', '0 (constant)', 'same', 'same']],
    [4.6*cm, 2.9*cm, 2.9*cm, 2.5*cm, 2.5*cm]))
story.append(sp(8))
story.append(fig(os.path.join(RES, 'jpeg_size_distribution.png'), width=14.5*cm))
story.append(cap('Figure 6: distribution of per-image JPEG compressed size across the 200-image '
                 'pool — colour vs grayscale. Deep JSCC has no equivalent spread: every image '
                 'costs exactly the same 9,216 real channel uses.'))
story.append(PageBreak())

# ── 5.6 Equivalent size vs SNR + crossover ──────────────────────────────────
story.append(h2('5.6  Deep JSCC\'s equivalent size vs channel SNR — the full picture'))
story.append(body(
    'Since Deep JSCC has no literal bitstream, an SNR-dependent equivalent byte size '
    'can be computed the same way the original paper bounds its digital baseline '
    '(capacity of a complex AWGN channel, C = log<sub>2</sub>(1+SNR) bits per complex '
    'symbol, converted to bytes). Unlike JPEG\'s fixed file size, this equivalent size '
    '<b>grows with SNR</b> — more information can be carried per channel use as the '
    'channel gets cleaner, even though Deep JSCC does not exploit that extra headroom '
    '(its k=4,608 complex symbols never change):'))
story.append(sp(6))

_rows = []
for r in DSZ['savings_table']:
    _rows.append([
        f'{r["snr"]:.0f} dB',
        f'{r["jscc_bytes"]:,.0f} B',
        f'{r["jpeg_color_bytes"]:,.0f} B',
        f'{r["save_color_pct"]:+.1f}%',
        f'{r["jpeg_gray_bytes"]:,.0f} B',
        f'{r["save_gray_pct"]:+.1f}%',
    ])
story.append(data_table(
    ['SNR', 'JSCC bytes', 'JPEG-colour', 'vs colour', 'JPEG-gray', 'vs gray'],
    _rows,
    [2.0*cm, 2.7*cm, 2.7*cm, 2.4*cm, 2.7*cm, 2.4*cm]))
story.append(sp(4))
story.append(cap('Positive % = Deep JSCC smaller than JPEG (saves bandwidth); '
                 'negative % = Deep JSCC larger (costs more).'))
story.append(sp(6))
story.append(fig(os.path.join(RES, 'jscc_equivalent_size_vs_snr.png'), width=14.5*cm))
story.append(cap(f'Figure 7: Deep JSCC\'s equivalent size crosses the JPEG-colour average around '
                 f'SNR &#8776; {DSZ["crossover_color_snr"]} dB, and the JPEG-grayscale average '
                 f'around SNR &#8776; {DSZ["crossover_gray_snr"]} dB.'))
story.append(PageBreak())

# ── 5.7 Per-image savings distribution + operating-point caveat ────────────
story.append(h2('5.7  Per-image savings distribution, and the model\'s actual operating point'))
_ref = next(r for r in DSZ['savings_table'] if abs(r['snr'] - DSZ['snr_train']) < 0.5)
story.append(body(
    f'Because JPEG\'s size varies per image (5.5) while Deep JSCC\'s footprint is '
    f'constant, the <i>percentage saved</i> also varies per image, purely due to '
    f'JPEG\'s side of the comparison.  At the model\'s own training SNR '
    f'({DSZ["snr_train"]:.0f} dB), Deep JSCC\'s equivalent size is '
    f'<b>{_ref["jscc_bytes"]:,.0f} bytes/image</b> — averaging '
    f'<b>{_ref["save_color_pct"]:+.1f}%</b> vs JPEG-colour and '
    f'<b>{_ref["save_gray_pct"]:+.1f}%</b> vs JPEG-grayscale:'))
story.append(sp(6))
story.append(fig(os.path.join(RES, 'jscc_savings_distribution.png'), width=14.5*cm))
story.append(cap(f'Figure 8: per-image bandwidth-saving distribution at SNR_train = '
                 f'{DSZ["snr_train"]:.0f} dB. Both distributions sit almost entirely on the '
                 f'negative side — at this SNR, Deep JSCC costs more bandwidth than JPEG for '
                 f'nearly every image in the pool.'))
story.append(sp(8))
story.append(body(
    f'<b>Important framing for a paper built on this result:</b> the model\'s own '
    f'SNR_train = {DSZ["snr_train"]:.0f} dB sits <i>past both crossover points</i> '
    f'(colour &#8776; {DSZ["crossover_color_snr"]} dB, grayscale &#8776; '
    f'{DSZ["crossover_gray_snr"]} dB).  At its actual trained operating point, Deep '
    f'JSCC is <b>not</b> the bandwidth-cheaper option.  The genuine '
    f'bandwidth-saving regime is the <b>low-SNR range</b> (below roughly 12–15 dB) — '
    f'which is also exactly where Deep JSCC\'s reconstruction-quality advantage over '
    f'JPEG+BPSK is largest (Section 4).  A paper claiming "Deep JSCC saves bandwidth '
    f'<i>and</i> reconstructs better" should present this as a <b>low-SNR result</b>, '
    f'not a universal one — training a dedicated low-SNR model (e.g. SNR_train &#8776; '
    f'7–10 dB) would make this the honest, defensible headline claim.'))
story.append(PageBreak())

# ── 6. Qualitative ────────────────────────────────────────────────────────────
story.append(section_box('6.  Qualitative Reconstructions', C_ORANGE))
story.append(sp(8))
story.append(fig(os.path.join(RES, 'qualitative_examples.png')))
story.append(cap('Figure 9: original vs Deep-JSCC vs JPEG+BPSK reconstructions at SNR = 4, 10, 19 dB.'))
story.append(PageBreak())

# ── 7. Summary ─────────────────────────────────────────────────────────────
story.append(section_box('7.  Summary', C_DARK))
story.append(sp(8))
story.append(h2('Key findings'))
story += [
    bullet('Deep JSCC degrades <b>gracefully</b> with channel SNR — no cliff effect, '
           'unlike JPEG+BPSK which fails outright below SNR&#8776;12 dB.'),
    bullet('Averaged across 0–25 dB, Deep JSCC beats JPEG+BPSK by roughly '
           '<b>+10.9 dB (colour)</b> and <b>+10.2 dB (grayscale)</b>.'),
    bullet('Grayscale input (fed as a replicated 3-channel image) reconstructs '
           '<b>better than colour input</b> at every SNR — up to +1.6 dB higher — '
           'because the fixed channel budget no longer needs to encode chroma.'),
    bullet('JPEG automatically shrinks by 18% when switched to grayscale mode; '
           '<b>Deep JSCC\'s channel bandwidth is fixed by the encoder architecture</b> '
           'and does not shrink for grayscale content.'),
    bullet('A dedicated single-channel Deep JSCC model would be needed to actually '
           'reclaim that bandwidth for grayscale-only applications — a natural next step.'),
]
story.append(sp(10))
story.append(h2('Deep JSCC vs JPEG+BPSK — at a glance'))
story.append(data_table(
    ['Property', 'Deep JSCC', 'JPEG + BPSK'],
    [['Cliff effect at low SNR', 'No', 'Yes'],
     ['Optimised end-to-end', 'Yes (MSE loss)', 'No (separate compression/coding)'],
     ['Bandwidth adapts to grayscale', 'No (fixed by architecture)', 'Yes (18% smaller)'],
     ['Best colour PSNR (high SNR)', '29.3 dB', '30.8 dB'],
     ['Best grayscale PSNR (high SNR)', '30.8 dB', '33.6 dB'],
     ['Average PSNR, 0-25 dB (colour)', '26.7 dB', '15.9 dB'],
     ['Average PSNR, 0-25 dB (grayscale)', '27.7 dB', '17.5 dB']],
    [5.5*cm, 4.8*cm, 5.1*cm]))
story.append(sp(10))
story.append(hr())
story.append(body('<i>Generated for ToR 2026 project. Deep JSCC reference: Bourtsoulatze, '
                  'Kurka &amp; Gündüz, IEEE TCCN 2019. Trained on STL-10 with an '
                  'RTX 4060 GPU.</i>'))

# ═══════════════════════════════════════════════════════════════════════════════
doc = SimpleDocTemplate(os.path.join(HERE, 'Deep_JSCC_Pipeline.pdf'), pagesize=A4,
                        leftMargin=2.2*cm, rightMargin=2.2*cm,
                        topMargin=2.2*cm, bottomMargin=2.2*cm)
doc.build(story)
print('Saved Deep_JSCC_Pipeline.pdf')
