"""Generate two pipeline PDFs: SRCNN and CycleCNN."""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle
from reportlab.graphics import renderPDF
from reportlab.platypus.flowables import Flowable

W, H = A4
C_DARK   = colors.HexColor('#2C3E50')
C_BLUE   = colors.HexColor('#2980B9')
C_GREEN  = colors.HexColor('#27AE60')
C_ORANGE = colors.HexColor('#E67E22')
C_RED    = colors.HexColor('#C0392B')
C_LIGHT  = colors.HexColor('#ECF0F1')
C_PURPLE = colors.HexColor('#8E44AD')
C_TEAL   = colors.HexColor('#16A085')
C_WHITE  = colors.white

# ── Styles ─────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def make_style(name, base='Normal', **kw):
    return ParagraphStyle(name, parent=styles[base], **kw)

S_TITLE   = make_style('MyTitle',  'Title',   fontSize=26, textColor=C_DARK, spaceAfter=6, alignment=TA_CENTER)
S_SUBTITLE= make_style('MySub',    'Normal',  fontSize=13, textColor=C_BLUE, alignment=TA_CENTER, spaceAfter=20)
S_H1      = make_style('MyH1',     'Heading1',fontSize=15, textColor=C_DARK, spaceBefore=14, spaceAfter=6)
S_H2      = make_style('MyH2',     'Heading2',fontSize=12, textColor=C_BLUE, spaceBefore=10, spaceAfter=4)
S_BODY    = make_style('MyBody',   'Normal',  fontSize=10, textColor=colors.HexColor('#333333'),
                       leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
S_BULLET  = make_style('MyBullet', 'Normal',  fontSize=10, textColor=colors.HexColor('#333333'),
                       leading=15, leftIndent=16, spaceAfter=3)
S_MATH    = make_style('MyMath',   'Normal',  fontSize=10, textColor=C_DARK,
                       fontName='Courier', backColor=C_LIGHT, leftIndent=20, spaceAfter=6,
                       borderPad=4)
S_CAP     = make_style('MyCap',    'Normal',  fontSize=9, textColor=colors.grey,
                       alignment=TA_CENTER, spaceAfter=8)
S_BOX     = make_style('MyBox',    'Normal',  fontSize=9, textColor=C_WHITE,
                       alignment=TA_CENTER, leading=13)

def h1(text): return Paragraph(text, S_H1)
def h2(text): return Paragraph(text, S_H2)
def body(text): return Paragraph(text, S_BODY)
def bullet(text): return Paragraph(f'&#8226; {text}', S_BULLET)
def math(text): return Paragraph(text, S_MATH)
def cap(text): return Paragraph(f'<i>{text}</i>', S_CAP)
def sp(n=6): return Spacer(1, n)
def hr(): return HRFlowable(width='100%', thickness=1, color=C_LIGHT, spaceAfter=8, spaceBefore=4)

# ── Coloured box table helper ──────────────────────────────────────────────────
def colbox(items_colors, col_widths=None):
    """Single-row coloured boxes as a table."""
    row = [Paragraph(f'<b>{t}</b>', S_BOX) for t, _ in items_colors]
    fills = [c for _, c in items_colors]
    n = len(row)
    w = col_widths or [14*cm/n]*n
    tbl = Table([row], colWidths=w, rowHeights=1.1*cm)
    ts = [('ALIGN',(0,0),(-1,-1),'CENTER'),
          ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
          ('GRID',(0,0),(-1,-1),0.5,C_WHITE),
          ('ROWBACKGROUNDS',(0,0),(-1,-1),[None])]
    for i,c in enumerate(fills):
        ts.append(('BACKGROUND',(i,0),(i,0),c))
    tbl.setStyle(TableStyle(ts))
    return tbl

def arrow_row(items, bg_colors, arrow='  →  '):
    """Items separated by arrows in one paragraph."""
    parts = []
    for i,(item,bg) in enumerate(zip(items,bg_colors)):
        parts.append(f'<b>{item}</b>')
        if i < len(items)-1: parts.append(arrow)
    return Paragraph('  '.join(parts), make_style('Arr','Normal',fontSize=10,
                     alignment=TA_CENTER, leading=16, spaceAfter=4))

def info_table(rows, col_widths=None):
    """Two-column label/value table."""
    w = col_widths or [4.5*cm, 10*cm]
    data = [[Paragraph(f'<b>{k}</b>', S_BODY), Paragraph(v, S_BODY)] for k,v in rows]
    tbl = Table(data, colWidths=w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1), C_LIGHT),
        ('GRID',(0,0),(-1,-1),0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),6),
        ('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    return tbl

def section_box(title, color=C_BLUE):
    data = [[Paragraph(f'<b>{title}</b>',
             make_style('SB','Normal',fontSize=11,textColor=C_WHITE,leading=14))]]
    tbl = Table(data, colWidths=[15*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1), color),
        ('LEFTPADDING',(0,0),(-1,-1),10),
        ('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('ROUNDEDCORNERS',[3,3,3,3]),
    ]))
    return tbl

# ══════════════════════════════════════════════════════════════════════════════
#  SRCNN PDF
# ══════════════════════════════════════════════════════════════════════════════
def build_srcnn():
    doc = SimpleDocTemplate('SRCNN_Pipeline.pdf', pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story += [sp(40),
              Paragraph('SRCNN', S_TITLE),
              Paragraph('Super-Resolution Convolutional Neural Network', S_SUBTITLE),
              Paragraph('Full Pipeline — Architecture, Training &amp; Transmission', S_SUBTITLE),
              sp(8),
              HRFlowable(width='80%', thickness=2, color=C_BLUE, hAlign='CENTER'),
              sp(12),
              body('Reference: Dong, C., Loy, C. C., He, K., &amp; Tang, X. (2014). '
                   '<i>Image Super-Resolution Using Deep Convolutional Networks.</i> '
                   'IEEE TPAMI.'),
              sp(4),
              body('Implementation: PyTorch · Dataset: STL-10 (96×96) · Scale factor: 4×'),
              PageBreak()]

    # ── Section 1: Overview ──────────────────────────────────────────────────
    story.append(section_box('1.  What is SRCNN?', C_BLUE))
    story.append(sp(8))
    story.append(body(
        'SRCNN is a <b>supervised</b> convolutional neural network for single-image '
        'super-resolution (SR). It learns a direct end-to-end mapping from a '
        'low-resolution (LR) input to a high-resolution (HR) output using paired '
        'training data. Because the loss function is pixel-wise MSE, training directly '
        'maximises PSNR — unlike unsupervised methods that use indirect proxy losses.'))
    story.append(sp(6))
    story.append(h2('Key properties'))
    story += [bullet('<b>Supervised</b> — requires paired (LR, HR) training images.'),
              bullet('<b>Lightweight</b> — only 3 convolutional layers, ~8 000 parameters.'),
              bullet('<b>Input</b> — bicubic-upsampled LR image (already at HR resolution).'),
              bullet('<b>Output</b> — refined HR image with sharper edges and details.'),
              bullet('<b>Loss</b> — Mean Squared Error (MSE) between predicted and true HR.'),
              sp(10)]

    story.append(h2('Comparison with CycleCNN'))
    story.append(info_table([
        ('Training',    'Supervised — needs paired (LR, HR) images'),
        ('Loss',        'MSE → directly maximises PSNR'),
        ('Architecture','3 plain conv layers (~8 K params)'),
        ('Convergence', 'Fast — loss = what you measure'),
        ('Limitation',  'Requires paired data; limited capacity for very hard content'),
    ]))
    story.append(PageBreak())

    # ── Section 2: Architecture ──────────────────────────────────────────────
    story.append(section_box('2.  Model Architecture', C_BLUE))
    story.append(sp(8))
    story.append(body(
        'SRCNN consists of exactly three convolutional layers, each playing a '
        'distinct role in the super-resolution process:'))
    story.append(sp(6))

    arch_data = [
        [Paragraph('<b>Layer</b>', S_BODY),
         Paragraph('<b>Config</b>', S_BODY),
         Paragraph('<b>Role</b>', S_BODY),
         Paragraph('<b>Output size</b>', S_BODY)],
        [Paragraph('Conv 1<br/>(Patch Extraction)', S_BODY),
         Paragraph('9×9 kernel<br/>64 filters, ReLU', S_BODY),
         Paragraph('Extracts overlapping patches; detects low-level features '
                   '(edges, gradients)', S_BODY),
         Paragraph('(H, W, 64)', S_BODY)],
        [Paragraph('Conv 2<br/>(Non-linear Mapping)', S_BODY),
         Paragraph('1×1 kernel<br/>32 filters, ReLU', S_BODY),
         Paragraph('Maps patch features non-linearly to HR feature space', S_BODY),
         Paragraph('(H, W, 32)', S_BODY)],
        [Paragraph('Conv 3<br/>(Reconstruction)', S_BODY),
         Paragraph('5×5 kernel<br/>1 filter, linear', S_BODY),
         Paragraph('Aggregates HR features into the final SR image', S_BODY),
         Paragraph('(H, W, 1)', S_BODY)],
    ]
    arch_tbl = Table(arch_data, colWidths=[3.5*cm, 3*cm, 5.5*cm, 2.5*cm])
    arch_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), C_BLUE),
        ('TEXTCOLOR',(0,0),(-1,0), C_WHITE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_LIGHT, C_WHITE]),
        ('GRID',(0,0),(-1,-1),0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(arch_tbl)
    story.append(sp(12))

    story.append(h2('Architecture Diagram'))
    story.append(colbox([
        ('Bicubic-up LR\n(1, H, W)', C_TEAL),
        ('Conv1\n9×9, 64ch\nReLU', C_BLUE),
        ('Conv2\n1×1, 32ch\nReLU', C_BLUE),
        ('Conv3\n5×5, 1ch\nlinear', C_BLUE),
        ('SR Output\n(1, H, W)', C_GREEN),
    ]))
    story.append(cap('Figure 1: SRCNN forward pass. Input is a bicubic-upsampled LR image.'))
    story.append(sp(10))

    story.append(h2('Residual Learning Variant'))
    story.append(body(
        'In this implementation, SRCNN uses <b>residual learning</b>: the network '
        'predicts the <i>difference</i> (residual) between the SR output and the '
        'bicubic input, rather than the full SR image. The final output is:'))
    story.append(math('SR = bicubic_input  +  SRCNN(bicubic_input)'))
    story.append(body(
        'This means at initialisation (near-zero weights), the output equals the '
        'bicubic image, so PSNR starts at the bicubic baseline and only improves '
        'during training — much faster convergence.'))
    story.append(PageBreak())

    # ── Section 3: Training Pipeline ────────────────────────────────────────
    story.append(section_box('3.  Training Pipeline', C_GREEN))
    story.append(sp(8))
    story.append(body(
        'SRCNN is trained in a fully <b>supervised</b> manner. Paired (LR, HR) '
        'training images are generated synthetically by downsampling and then '
        'upsampling the original HR image.'))
    story.append(sp(8))

    story.append(h2('Step 1 — Data Preparation'))
    story.append(colbox([
        ('HR image\n(96×96)', C_DARK),
        ('Bicubic\ndownsample ÷4', C_ORANGE),
        ('LR image\n(24×24)', C_ORANGE),
        ('Bicubic\nupsample ×4', C_ORANGE),
        ('Bicubic-HR\n(96×96)', C_TEAL),
    ]))
    story.append(cap('Figure 2: Degradation pipeline for creating training pairs.'))
    story.append(sp(6))
    story.append(body(
        'The pair <b>(Bicubic-HR, true HR)</b> forms one training sample. '
        'The model learns to map the blurry bicubic-HR to the sharp true HR.'))
    story.append(sp(8))

    story.append(h2('Step 2 — Loss Function'))
    story.append(math('L = (1/N) * SUM[ || SRCNN(bic_i) - HR_i ||^2 ]'))
    story.append(body(
        'Mean Squared Error (MSE) loss. Minimising MSE directly maximises PSNR, '
        'since PSNR = 10 * log10(255^2 / MSE).'))
    story.append(sp(8))

    story.append(h2('Step 3 — Optimisation'))
    story.append(info_table([
        ('Optimiser',   'Adam'),
        ('LR (Conv1,2)','1 × 10<super>-4</super>'),
        ('LR (Conv3)',  '1 × 10<super>-5</super>  (10× smaller, as in original paper)'),
        ('Scheduler',   'Step decay ×0.1 at epoch 100'),
        ('Batch size',  '64 patches of 48×48'),
        ('Dataset',     'STL-10 unlabeled split (5 000 images used), 4 patches/image'),
    ]))
    story.append(sp(8))

    story.append(h2('Full Training Loop'))
    train_steps = [
        ['1', 'Sample random 48×48 crop from HR image'],
        ['2', 'Downsample ÷4 (bicubic) → LR patch (12×12)'],
        ['3', 'Upsample ×4 (bicubic) → Bicubic-HR patch (48×48)  — SRCNN input'],
        ['4', 'Forward pass: residual = SRCNN(bicubic_HR)'],
        ['5', 'SR = bicubic_HR + residual  (clamped to [0, 1])'],
        ['6', 'Loss = MSE(SR, true_HR)'],
        ['7', 'Backpropagate → update Conv1, Conv2, Conv3 weights'],
        ['8', 'Validate every 20 epochs on full 96×96 test images'],
    ]
    t = Table([[Paragraph(f'<b>{n}</b>', S_BODY), Paragraph(s, S_BODY)]
               for n,s in train_steps], colWidths=[1*cm, 13.5*cm])
    t.setStyle(TableStyle([
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[C_LIGHT, C_WHITE]),
        ('GRID',(0,0),(-1,-1),0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN',(0,0),(0,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ── Section 4: Inference + Transmission ─────────────────────────────────
    story.append(section_box('4.  Transmission Pipeline (BPSK + AWGN)', C_ORANGE))
    story.append(sp(8))
    story.append(body(
        'This section describes the end-to-end pipeline used in the project, '
        'connecting SRCNN to the Week 2 JPEG/BPSK/AWGN transmission scheme. '
        'Three schemes are compared at various SNR levels (0–20 dB).'))
    story.append(sp(10))

    story.append(h2('Scheme 1 — HR-Direct (Week 2 Baseline)'))
    story.append(colbox([
        ('HR image\n96×96', C_DARK),
        ('JPEG encode\nQ=73', C_ORANGE),
        ('BPSK mod\n+AWGN', C_RED),
        ('Hard decode\n>0 → 1', C_RED),
        ('JPEG decode', C_ORANGE),
        ('Received HR', C_TEAL),
    ]))
    story.append(cap('Transmit the full 96×96 image directly. High quality at high SNR.'))
    story.append(sp(12))

    story.append(h2('Scheme 2 — LR + Bicubic Upsample'))
    story.append(colbox([
        ('HR → LR\n24×24', C_DARK),
        ('JPEG encode\nQ=73', C_ORANGE),
        ('BPSK+AWGN', C_RED),
        ('JPEG decode\nLR 24×24', C_ORANGE),
        ('Bicubic\nupsample ×4', C_TEAL),
        ('SR 96×96', C_TEAL),
    ]))
    story.append(cap('Transmit 24×24 LR; reconstruct with bicubic upsampling only.'))
    story.append(sp(12))

    story.append(h2('Scheme 3 — LR + SRCNN (Proposed)'))
    story.append(colbox([
        ('HR → LR\n24×24', C_DARK),
        ('JPEG encode\nQ=73', C_ORANGE),
        ('BPSK+AWGN', C_RED),
        ('JPEG decode\nLR 24×24', C_ORANGE),
        ('Bicubic ×4\n→ SRCNN', C_BLUE),
        ('SR 96×96', C_GREEN),
    ]))
    story.append(cap('Transmit 24×24 LR; reconstruct with bicubic then SRCNN refinement.'))
    story.append(sp(10))

    story.append(h2('AWGN Channel Model'))
    story.append(math('symbols = 2 * bits - 1         (BPSK: 0→-1, 1→+1)'))
    story.append(math('sigma = sqrt( 1 / 10^(SNR_dB / 10) )'))
    story.append(math('received = symbols + Normal(0, sigma)'))
    story.append(math('decoded_bits = (received > 0)   (hard decision)'))
    story.append(sp(10))

    story.append(h2('Bandwidth'))
    story.append(info_table([
        ('HR (96×96) JPEG', '~18 162 bits per image  (Q=73)'),
        ('LR (24×24) JPEG', '~3 969 bits per image  (Q=73)'),
        ('Reduction',       '~4.6× fewer bits for LR transmission'),
    ]))
    story.append(PageBreak())

    # ── Section 5: Results ───────────────────────────────────────────────────
    story.append(section_box('5.  Results & Key Findings', C_PURPLE))
    story.append(sp(8))

    story.append(h2('PSNR vs SNR Summary (N = 200 STL-10 images)'))
    res_data = [
        [Paragraph(f'<b>{h}</b>', S_BODY) for h in
         ['SNR (dB)', 'HR-direct', 'LR + Bicubic', 'LR + SRCNN']],
        *[[Paragraph(str(v), S_BODY) for v in row] for row in [
            ['< 8',  '~0 dB (corrupt)', '~0 dB (corrupt)', '~0 dB (corrupt)'],
            ['8',    '9 dB',  '10 dB', '10 dB'],
            ['10',   '13 dB', '18 dB', '18 dB'],
            ['11',   '23 dB', '21 dB', '21 dB'],
            ['12',   '31 dB', '22 dB', '22 dB'],
            ['14+',  '33 dB', '22 dB', '23 dB'],
        ]],
    ]
    rt = Table(res_data, colWidths=[3*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    rt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), C_PURPLE),
        ('TEXTCOLOR',(0,0),(-1,0), C_WHITE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_LIGHT, C_WHITE]),
        ('GRID',(0,0),(-1,-1),0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(rt)
    story.append(sp(10))

    story.append(h2('Three-Regime Finding'))
    story += [
        bullet('<b>SNR &lt; 8 dB</b> — Complete failure for all schemes. Channel too '
               'noisy for any JPEG bitstream to survive.'),
        bullet('<b>SNR 8–11 dB (marginal zone)</b> — LR transmission wins (+5 dB). '
               'Shorter LR bitstream accumulates fewer errors per surviving packet.'),
        bullet('<b>SNR &ge; 12 dB (reliable zone)</b> — HR-direct wins decisively '
               '(33 dB vs 22–23 dB plateau). Once the channel is clean, transmitting '
               'full resolution beats reconstructing from a quarter-size image.'),
        sp(8),
        bullet('<b>SRCNN vs Bicubic</b> — SRCNN adds ~0.5 dB gain within the LR '
               'transmission scheme at high SNR, matching clean-data SR performance.'),
        sp(8),
        body('<b>Information-theoretic ceiling:</b> the ~22–23 dB plateau of LR+CNN '
             'schemes at high SNR is not a model failure. It reflects the irreversible '
             'information loss from 4× downsampling before transmission. No '
             'reconstruction algorithm can exceed this ceiling.'),
    ]
    story.append(sp(10))
    story.append(hr())
    story.append(body('<i>Generated for ToR 2026 project. SRCNN reference: Dong et al., '
                      'IGARSS 2019 / IEEE TPAMI 2016.</i>'))

    doc.build(story)
    print('Saved: SRCNN_Pipeline.pdf')

# ══════════════════════════════════════════════════════════════════════════════
#  CycleCNN PDF
# ══════════════════════════════════════════════════════════════════════════════
def build_cyclecnn():
    doc = SimpleDocTemplate('CycleCNN_Pipeline.pdf', pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story += [sp(40),
              Paragraph('Cycle-CNN', S_TITLE),
              Paragraph('Unsupervised Remote Sensing Image Super-Resolution', S_SUBTITLE),
              Paragraph('Full Pipeline — Architecture, Training &amp; Transmission', S_SUBTITLE),
              sp(8),
              HRFlowable(width='80%', thickness=2, color=C_ORANGE, hAlign='CENTER'),
              sp(12),
              body('Reference: Wang, Y. et al. (2019). <i>Unsupervised Remote Sensing '
                   'Image Super-Resolution Using Cycle CNN.</i> IGARSS 2019.'),
              sp(4),
              body('Implementation: PyTorch · Dataset: STL-10 (96×96) · Scale factor: 4×'),
              PageBreak()]

    # ── Section 1: Overview ──────────────────────────────────────────────────
    story.append(section_box('1.  What is Cycle-CNN?', C_ORANGE))
    story.append(sp(8))
    story.append(body(
        'Cycle-CNN is an <b>unsupervised</b> super-resolution framework inspired by '
        'CycleGAN. Unlike SRCNN, it does <i>not</i> require paired (LR, HR) training '
        'images. Instead, it learns from two <b>unpaired</b> pools of images — one HR '
        'pool and one LR pool — using cycle-consistency to ensure the two generators '
        'are inverses of each other.'))
    story.append(sp(6))
    story.append(h2('Key properties'))
    story += [bullet('<b>Unsupervised</b> — no paired (LR, HR) data needed.'),
              bullet('<b>Two generators</b> — G1 (LR→HR) and G2 (HR→LR).'),
              bullet('<b>Cycle consistency</b> — G2(G1(LR)) ≈ LR  and  G1(G2(HR)) ≈ HR.'),
              bullet('<b>Identity loss</b> — stabilises training (dominant weight).'),
              bullet('<b>Limitation</b> — indirect loss proxy; PSNR ceiling ~22–24 dB.'),
              sp(10)]

    story.append(h2('Comparison with SRCNN'))
    story.append(info_table([
        ('Training',    'Unsupervised — only needs unpaired HR pool + LR pool'),
        ('Loss',        'Cycle-consistency + identity (indirect proxy for PSNR)'),
        ('Architecture','G1: 16 ResBlocks + SubpixelConv (~large); G2: 5 ResBlocks'),
        ('Convergence', 'Slow — loss does not directly optimise PSNR'),
        ('Advantage',   'Works when ground-truth HR data is unavailable'),
    ]))
    story.append(PageBreak())

    # ── Section 2: Architecture ──────────────────────────────────────────────
    story.append(section_box('2.  Model Architecture', C_ORANGE))
    story.append(sp(8))
    story.append(body(
        'Cycle-CNN contains two generator networks that form an inverse pair:'))
    story.append(sp(6))

    story.append(h2('Generator G1 — SR Network (LR → HR)'))
    story.append(colbox([
        ('LR Input\n(1, 24, 24)', C_DARK),
        ('Conv\n3×3, 24ch', C_ORANGE),
        ('16 × ResBlock\n(BN + ReLU)', C_ORANGE),
        ('SubPixelConv\n×2  (24→48)', C_BLUE),
        ('SubPixelConv\n×2  (48→96)', C_BLUE),
        ('HR Output\n(1, 96, 96)', C_GREEN),
    ]))
    story.append(cap('Figure 3: G1 upsamples 24×24 → 96×96 via two sub-pixel conv layers.'))
    story.append(sp(6))
    story.append(info_table([
        ('Input size',   '24×24 (raw LR image, not bicubic-upsampled)'),
        ('Output size',  '96×96 (4× upsampling)'),
        ('ResBlocks',    '16 blocks, each: Conv(3×3) → BN → ReLU → Conv(3×3) → BN + skip'),
        ('Upsampling',   '2 × SubpixelConv (pixel-shuffle): 24→48→96 channels rearranged'),
        ('Channels',     '24 feature channels throughout'),
        ('Parameters',   '~large (dominates total model size)'),
    ]))
    story.append(sp(10))

    story.append(h2('Generator G2 — Downsampling Network (HR → LR)'))
    story.append(colbox([
        ('HR Input\n(1, 96, 96)', C_DARK),
        ('Conv\n3×3, 96ch', C_TEAL),
        ('5 × ResBlock\n(BN + ReLU)', C_TEAL),
        ('AvgPool\n÷2', C_RED),
        ('AvgPool\n÷2', C_RED),
        ('LR Output\n(1, 24, 24)', C_ORANGE),
    ]))
    story.append(cap('Figure 4: G2 downsamples 96×96 → 24×24 via two average-pool layers.'))
    story.append(sp(6))
    story.append(info_table([
        ('Input size',  '96×96 (HR image)'),
        ('Output size', '24×24 (LR image)'),
        ('ResBlocks',   '5 blocks with BatchNorm'),
        ('Downsampling','2 × AvgPool(2×2): 96→48→24'),
    ]))
    story.append(PageBreak())

    # ── Section 3: Training ──────────────────────────────────────────────────
    story.append(section_box('3.  Unsupervised Training Pipeline', C_ORANGE))
    story.append(sp(8))
    story.append(body(
        'Training uses two unpaired image pools drawn from the STL-10 dataset. '
        'No image in the HR pool is paired with any image in the LR pool.'))
    story.append(sp(8))

    story.append(h2('Cycle-Consistency Loss'))
    story.append(body(
        'The core training signal: applying both generators in sequence should '
        'recover the original image.'))
    story.append(colbox([
        ('LR pool\nimage', C_ORANGE),
        ('G1\nLR→HR', C_BLUE),
        ('Fake HR', C_GREEN),
        ('G2\nHR→LR', C_TEAL),
        ('Rec LR', C_ORANGE),
        ('≈ LR?\nL_cyc', C_RED),
    ]))
    story.append(cap('Forward cycle: LR → G1 → fake-HR → G2 → rec-LR  (should ≈ LR)'))
    story.append(sp(6))
    story.append(colbox([
        ('HR pool\nimage', C_GREEN),
        ('G2\nHR→LR', C_TEAL),
        ('Fake LR', C_ORANGE),
        ('G1\nLR→HR', C_BLUE),
        ('Rec HR', C_GREEN),
        ('≈ HR?\nL_cyc', C_RED),
    ]))
    story.append(cap('Backward cycle: HR → G2 → fake-LR → G1 → rec-HR  (should ≈ HR)'))
    story.append(sp(10))

    story.append(h2('Identity Loss'))
    story.append(body(
        'Prevents generators from hallucinating content when the input is already '
        'in the target domain:'))
    story.append(math('L_idt = ||G2(LR) - LR|| + ||G1(HR) - HR||'))
    story.append(sp(10))

    story.append(h2('Total Loss'))
    story.append(math('L_total = w1 * L_cycle  +  w2 * L_identity'))
    story.append(math('       w1 = 0.5  (cycle)     w2 = 10.0  (identity)'))
    story.append(body(
        'Identity loss dominates (20× larger weight). This keeps the generators '
        'close to identity mappings, which stabilises training but also limits '
        'how aggressively the SR generator can sharpen images.'))
    story.append(sp(10))

    story.append(h2('Training Configuration'))
    story.append(info_table([
        ('Optimiser',       'Adam (G1 and G2 separately)'),
        ('Learning rate',   '2 × 10<super>-4</super>, halved at iteration 125 000'),
        ('Total iterations','250 000'),
        ('Batch size',      '8 images'),
        ('Patch size',      '64×64 HR  (with random crop + dihedral augmentation)'),
        ('LR patch',        '16×16 (bicubic ÷4 from 64×64 HR crop)'),
        ('Dataset',         'STL-10 unlabeled (up to 40 000 images)'),
        ('Checkpoints',     'Every 25 000 iterations → checkpoints_fixed/'),
    ]))
    story.append(PageBreak())

    # ── Section 4: Transmission ──────────────────────────────────────────────
    story.append(section_box('4.  Transmission Pipeline (BPSK + AWGN)', C_ORANGE))
    story.append(sp(8))
    story.append(body(
        'Same transmission scheme as SRCNN and Week 2 — three schemes compared '
        'across SNR = 0–20 dB. The key difference from SRCNN: G1 takes raw '
        '24×24 LR as input (no bicubic pre-upsampling needed).'))
    story.append(sp(10))

    story.append(h2('Scheme 1 — HR-Direct (Week 2 Baseline)'))
    story.append(colbox([
        ('HR image\n96×96', C_DARK),
        ('JPEG encode\nQ=73', C_ORANGE),
        ('BPSK mod\n+AWGN', C_RED),
        ('Hard decode', C_RED),
        ('JPEG decode', C_ORANGE),
        ('Received HR', C_TEAL),
    ]))
    story.append(sp(10))

    story.append(h2('Scheme 2 — LR + Bicubic'))
    story.append(colbox([
        ('HR→LR\n24×24', C_DARK),
        ('JPEG encode\nQ=73', C_ORANGE),
        ('BPSK+AWGN', C_RED),
        ('JPEG decode\n24×24', C_ORANGE),
        ('Bicubic\n×4', C_TEAL),
        ('SR 96×96', C_TEAL),
    ]))
    story.append(sp(10))

    story.append(h2('Scheme 3 — LR + Cycle-CNN G1 (Proposed)'))
    story.append(colbox([
        ('HR→LR\n24×24', C_DARK),
        ('JPEG encode\nQ=73', C_ORANGE),
        ('BPSK+AWGN', C_RED),
        ('JPEG decode\n24×24', C_ORANGE),
        ('G1\n(CycleCNN)', C_BLUE),
        ('SR 96×96', C_GREEN),
    ]))
    story.append(cap('G1 takes the raw 24×24 decoded LR image directly — no pre-upsampling.'))
    story.append(sp(10))

    story.append(h2('AWGN Channel Model'))
    story.append(math('symbols = 2 * bits - 1         (BPSK: 0→-1, 1→+1)'))
    story.append(math('sigma = sqrt( 1 / 10^(SNR_dB / 10) )'))
    story.append(math('received = symbols + Normal(0, sigma)'))
    story.append(math('decoded_bits = (received > 0)   (hard decision)'))
    story.append(PageBreak())

    # ── Section 5: Results ───────────────────────────────────────────────────
    story.append(section_box('5.  Results & Key Findings', C_PURPLE))
    story.append(sp(8))

    story.append(h2('Clean SR Evaluation (No Transmission Noise)'))
    story.append(info_table([
        ('Test set',       '64 STL-10 train images (full random set)'),
        ('Bicubic x4',     '~22–25 dB average  (range 17–32 dB depending on content)'),
        ('Cycle-CNN (G1)', '~24–25 dB average  (+1–2 dB over bicubic)'),
        ('Curated easy-8', 'Cycle-CNN: 31.1–31.2 dB vs bicubic 28.7 dB (+2.4 dB)'),
        ('Full random set', 'Average ~22 dB — limited by hard/textured content in STL-10'),
    ]))
    story.append(sp(8))

    story.append(h2('PSNR vs SNR (N = 200 images, Transmission Experiment)'))
    res_data2 = [
        [Paragraph(f'<b>{h}</b>', S_BODY) for h in
         ['SNR (dB)', 'HR-direct', 'LR + Bicubic', 'LR + Cycle-CNN']],
        *[[Paragraph(str(v), S_BODY) for v in row] for row in [
            ['< 8',  '~0 dB (corrupt)', '~0 dB (corrupt)', '~0 dB (corrupt)'],
            ['8',    '5.5 dB',  '9.2 dB', '8.5 dB'],
            ['10',   '12.7 dB', '18.3 dB', '17.9 dB'],
            ['11',   '23.0 dB', '20.6 dB', '20.9 dB'],
            ['12',   '31.2 dB', '21.6 dB', '22.0 dB'],
            ['14+',  '33.5 dB', '21.9 dB', '22.3 dB'],
        ]],
    ]
    rt2 = Table(res_data2, colWidths=[3*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    rt2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), C_ORANGE),
        ('TEXTCOLOR',(0,0),(-1,0), C_WHITE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_LIGHT, C_WHITE]),
        ('GRID',(0,0),(-1,-1),0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(rt2)
    story.append(sp(10))

    story.append(h2('Three-Regime Finding'))
    story += [
        bullet('<b>SNR &lt; 8 dB</b> — All schemes fail completely. '
               'JPEG headers are corrupted regardless of bitstream length.'),
        bullet('<b>SNR 8–11 dB</b> — LR transmission wins (+5 dB at SNR=10 dB). '
               'Shorter LR bitstream has fewer total bit errors per surviving image.'),
        bullet('<b>SNR &ge; 12 dB</b> — HR-direct wins decisively (31–34 dB vs ~22 dB). '
               'The 4× downsampling discards information that no CNN can recover.'),
        sp(8),
        bullet('<b>Why Cycle-CNN plateaus at ~22 dB:</b> the unsupervised cycle-consistency '
               'loss is only an indirect proxy for PSNR. The identity-dominant weighting '
               '(w2=10 >> w1=0.5) further limits aggressive SR learning. SRCNN\'s '
               'supervised MSE loss directly optimises PSNR, yielding better quality '
               'when sufficient clean training data is available.'),
    ]
    story.append(sp(10))
    story.append(hr())
    story.append(body('<i>Generated for ToR 2026 project. Reference: Wang et al., '
                      'IGARSS 2019. Trained for 250 000 iterations on STL-10.</i>'))

    doc.build(story)
    print('Saved: CycleCNN_Pipeline.pdf')

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build_srcnn()
    build_cyclecnn()
    print('Both PDFs created.')
