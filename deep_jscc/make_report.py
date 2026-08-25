"""Generate ToR_DataSize_Grayscale_Report.pdf — a supervisor progress report on
(1) the data-size / bandwidth comparison and (2) the proposed 1xHxW -> 3xHxW
channel-expansion CNN for grayscale input. Numbers are pulled live from
results/data_size_summary.json."""
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
with open(os.path.join(RES, 'data_size_summary.json')) as f:
    D = json.load(f)

C_DARK  = colors.HexColor('#2C3E50')
C_BLUE  = colors.HexColor('#2980B9')
C_TEAL  = colors.HexColor('#16A085')
C_PURPLE= colors.HexColor('#8E44AD')
C_LIGHT = colors.HexColor('#ECF0F1')
C_WHITE = colors.white

styles = getSampleStyleSheet()
def st(name, base='Normal', **kw): return ParagraphStyle(name, parent=styles[base], **kw)

S_TITLE = st('T','Title', fontSize=22, textColor=C_DARK, spaceAfter=4, alignment=TA_CENTER)
S_SUB   = st('S','Normal', fontSize=12, textColor=C_TEAL, alignment=TA_CENTER, spaceAfter=4)
S_META  = st('M','Normal', fontSize=9.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=3)
S_H1    = st('H1','Heading1', fontSize=14, textColor=C_DARK, spaceBefore=13, spaceAfter=6)
S_H2    = st('H2','Heading2', fontSize=11.5, textColor=C_TEAL, spaceBefore=9, spaceAfter=4)
S_BODY  = st('B','Normal', fontSize=10, textColor=colors.HexColor('#2C2C2C'),
             leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
S_BUL   = st('BL','Normal', fontSize=10, textColor=colors.HexColor('#2C2C2C'),
             leading=15, leftIndent=15, spaceAfter=3)
S_CAP   = st('CAP','Normal', fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=9)
S_MATH  = st('MA','Normal', fontSize=9.5, textColor=C_DARK, fontName='Courier',
             backColor=C_LIGHT, leftIndent=14, spaceAfter=6)

def h1(t): return Paragraph(t, S_H1)
def h2(t): return Paragraph(t, S_H2)
def body(t): return Paragraph(t, S_BODY)
def bullet(t): return Paragraph(f'&#8226;&nbsp; {t}', S_BUL)
def cap(t): return Paragraph(f'<i>{t}</i>', S_CAP)
def math(t): return Paragraph(t, S_MATH)
def sp(n=6): return Spacer(1, n)
def hr(): return HRFlowable(width='100%', thickness=1, color=C_LIGHT, spaceBefore=4, spaceAfter=8)

def box(title, color=C_TEAL):
    tbl = Table([[Paragraph(f'<b>{title}</b>', st('BX','Normal',fontSize=11,textColor=C_WHITE))]],
                colWidths=[16*cm])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),color),
        ('LEFTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return tbl

def dtable(header, rows, widths, head_color=C_TEAL):
    data = [[Paragraph(f'<b>{h}</b>', S_BODY) for h in header]]
    for r in rows:
        data.append([Paragraph(str(v), S_BODY) for v in r])
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), head_color),
        ('TEXTCOLOR',(0,0),(-1,0), C_WHITE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_LIGHT, C_WHITE]),
        ('GRID',(0,0),(-1,-1),0.5, colors.HexColor('#CCCCCC')),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
    return t

def fig(name, width=14.5*cm):
    from PIL import Image as PILImage
    path = os.path.join(RES, name)
    w, h = PILImage.open(path).size
    return RLImage(path, width=width, height=width*h/w)

story = []

# ── Header ─────────────────────────────────────────────────────────────────────
story += [
    Paragraph('ToR Project — Progress Report', S_TITLE),
    Paragraph('Data-Size Comparison &amp; Grayscale Channel-Expansion Proposal', S_SUB),
    sp(4),
    Paragraph('Deep Joint Source-Channel Coding (JSCC) for wireless image transmission — STL-10',
              S_META),
    Paragraph('Prepared for: Supervisor&nbsp;&nbsp;|&nbsp;&nbsp;Author: Nam Khanh Tran'
              '&nbsp;&nbsp;|&nbsp;&nbsp;July 2026', S_META),
    sp(4),
    HRFlowable(width='100%', thickness=1.5, color=C_TEAL),
    sp(8),
]

# ── Executive summary ──────────────────────────────────────────────────────────
story.append(body(
    'This report covers two items from the recent work on the Deep-JSCC model. '
    '<b>(1)</b> A quantitative comparison of the compressed data size of Deep JSCC '
    'against the JPEG baseline (both colour and grayscale), and where each scheme is '
    'the bandwidth-cheaper option. <b>(2)</b> A proposal to replace the current '
    'grayscale input handling (channel replication) with a small learnable CNN that '
    'maps a 1-channel image to the 3-channel input the encoder expects.'))
story.append(sp(4))

# ── Section 1 ──────────────────────────────────────────────────────────────────
story.append(box('1.  Compressed Data-Size Comparison'))
story.append(sp(8))
story.append(h2('1.1  Motivation and method'))
story.append(body(
    'JPEG produces a literal, countable file size that <b>varies with image content</b>. '
    'Deep JSCC has no discrete bitstream — it transmits a <b>fixed number of continuous-'
    'valued channel symbols</b>, the same count for every image. To compare the two on a '
    'common axis, the JSCC channel usage is converted to an equivalent byte size using '
    'the AWGN channel-capacity formula from the reference paper (Bourtsoulatze et al., '
    'Section IV-A, Eq. 6-7):'))
story.append(math('equivalent_bytes(SNR) = k &#215; log<sub>2</sub>(1 + SNR) / 8'))
story.append(body(
    f'where <b>k = {D["k_complex"]:,}</b> complex channel symbols/image '
    f'(= {D["k_real"]:,} real channel uses), fixed by the encoder architecture at the '
    f'1/6 bandwidth ratio. All measurements below are over a fixed 200-image STL-10 '
    f'test pool.'))
story.append(sp(4))

story.append(h2('1.2  JPEG size varies with content; Deep JSCC does not'))
story.append(dtable(
    ['Scheme', 'Mean', 'Std dev', 'Min', 'Max'],
    [['JPEG Q=73, colour',
      f'{D["jpeg_color"]["mean"]:,.0f} B', f'{D["jpeg_color"]["std"]:,.0f} B',
      f'{D["jpeg_color"]["min"]:,} B', f'{D["jpeg_color"]["max"]:,} B'],
     ['JPEG Q=73, grayscale',
      f'{D["jpeg_gray"]["mean"]:,.0f} B', f'{D["jpeg_gray"]["std"]:,.0f} B',
      f'{D["jpeg_gray"]["min"]:,} B', f'{D["jpeg_gray"]["max"]:,} B'],
     ['Deep JSCC (any image)',
      f'{D["k_real"]:,} values', '0 (constant)', 'same', 'same']],
    [4.4*cm, 2.9*cm, 2.9*cm, 2.5*cm, 2.5*cm]))
story.append(sp(6))
story.append(fig('jpeg_size_distribution.png', width=13.5*cm))
story.append(cap('Figure 1: per-image JPEG compressed size spreads widely with content '
                 '(colour std ' + f'{D["jpeg_color"]["std"]:,.0f}' + ' B, grayscale std '
                 + f'{D["jpeg_gray"]["std"]:,.0f}' + ' B); Deep JSCC has zero spread.'))
story.append(body(
    'Switching JPEG to grayscale mode alone shrinks the average file by ~18% '
    f'({D["jpeg_color"]["mean"]:,.0f} &#8594; {D["jpeg_gray"]["mean"]:,.0f} bytes) because '
    'the two chroma planes are simply not encoded. <b>Deep JSCC gets no such saving</b> — '
    'its channel-symbol count is architecture-fixed and identical for colour and '
    'grayscale-replicated input.'))
story.append(PageBreak())

story.append(h2('1.3  Equivalent size vs channel SNR, and the crossover'))
story.append(body(
    'Unlike JPEG&#8217;s fixed file, Deep JSCC&#8217;s equivalent size grows with SNR '
    '(more information can be carried per channel use as the channel improves). It '
    f'crosses the JPEG-colour average around <b>SNR &#8776; {D["crossover_color_snr"]} dB</b> '
    f'and the JPEG-grayscale average around <b>SNR &#8776; {D["crossover_gray_snr"]} dB</b>:'))
story.append(sp(4))

_rows = []
for r in D['savings_table']:
    _rows.append([
        f'{r["snr"]:.0f} dB',
        f'{r["jscc_bytes"]:,.0f} B',
        f'{r["jpeg_color_bytes"]:,.0f} B',
        f'{r["save_color_pct"]:+.1f}%',
        f'{r["jpeg_gray_bytes"]:,.0f} B',
        f'{r["save_gray_pct"]:+.1f}%',
    ])
story.append(dtable(
    ['SNR', 'JSCC', 'JPEG-clr', 'vs clr', 'JPEG-gry', 'vs gry'],
    _rows, [1.9*cm, 2.6*cm, 2.6*cm, 2.4*cm, 2.6*cm, 2.4*cm]))
story.append(cap('Positive % = Deep JSCC uses fewer bytes than JPEG (saves bandwidth); '
                 'negative % = uses more.'))
story.append(sp(4))
story.append(fig('jscc_equivalent_size_vs_snr.png', width=13.5*cm))
story.append(cap('Figure 2: Deep JSCC equivalent size vs SNR, with JPEG averages (dashed).'))
story.append(PageBreak())

story.append(h2('1.4  Key finding for the project direction'))
_ref = next(r for r in D['savings_table'] if abs(r['snr'] - D['snr_train']) < 0.5)
story.append(fig('jscc_savings_distribution.png', width=13.0*cm))
story.append(cap(f'Figure 3: per-image saving distribution at SNR_train = {D["snr_train"]:.0f} dB; '
                 'almost entirely negative (JSCC costs more bytes here).'))
story.append(body(
    f'<b>Deep JSCC is the bandwidth-cheaper scheme only in the low-SNR regime</b> '
    f'(below roughly {D["crossover_gray_snr"]}-{D["crossover_color_snr"]} dB). At low SNR '
    f'the savings are large — e.g. at 7 dB, JSCC uses ~46% fewer bytes than JPEG-colour '
    f'and ~35% fewer than JPEG-grayscale. However, the model&#8217;s own training point '
    f'(<b>SNR_train = {D["snr_train"]:.0f} dB</b>) sits <i>past</i> both crossovers, so at '
    f'its trained operating point Deep JSCC actually uses more bytes than JPEG '
    f'({_ref["save_color_pct"]:+.0f}% vs colour, {_ref["save_gray_pct"]:+.0f}% vs grayscale).'))
story.append(sp(2))
story.append(body(
    '<b>Recommendation.</b> The bandwidth-saving story is most defensible as a '
    '<b>low-SNR result</b>, which is also exactly where Deep JSCC&#8217;s reconstruction '
    'quality most strongly beats JPEG (JPEG&#8217;s bitstream fails to decode below '
    '~10 dB). Training a dedicated low-SNR model (e.g. SNR_train &#8776; 7-10 dB) would let '
    'us claim &#8220;better reconstruction AND lower bandwidth&#8221; at the same operating '
    'point. <i>(Crossover / low-SNR framing is our own analysis; the reference paper only '
    'states that Deep JSCC wins on PSNR at low SNR and low bandwidth, pp. 8 and 11.)</i>'))
story.append(PageBreak())

# ── Section 2 ──────────────────────────────────────────────────────────────────
story.append(box('2.  Grayscale Input: Learnable 1&#215;H&#215;W &#8594; 3&#215;H&#215;W Adapter',
                 C_PURPLE))
story.append(sp(8))
story.append(h2('2.1  Current handling and its limitation'))
story.append(body(
    'The encoder&#8217;s first layer is fixed at 3 input channels, so a grayscale image '
    '(1&#215;H&#215;W) cannot be fed in directly. The current approach <b>replicates</b> '
    'the single luminance channel into all three inputs (R = G = B = gray). This works — '
    'in fact it reconstructs with <i>higher</i> PSNR than colour input, because no channel '
    'budget is spent on chroma — but it is a fixed, hand-coded rule: two of the three '
    'input channels carry duplicated information.'))
story.append(sp(4))
story.append(fig('grayscale_samples.png', width=12.5*cm))
story.append(cap('Figure 4: original colour (top) vs the grayscale-replicated input currently '
                 'fed to the encoder (bottom).'))
story.append(sp(2))

story.append(h2('2.2  Proposal: a small trainable channel-expansion module'))
story.append(body(
    'Replace the fixed replication with a small learnable module &#934; that maps '
    '1 channel to 3, inserted immediately before the existing encoder:'))
story.append(math('grayscale (1, H, W)  &#8594;  &#934; (Conv2d 1&#8594;3)  &#8594;  '
                  '(3, H, W)  &#8594;  existing encoder'))
story.append(sp(2))
story.append(h2('2.3  Feasibility'))
story += [
    bullet('<b>Tiny.</b> A 1&#215;1 conv (1&#8594;3) adds only 6 parameters; a 3&#215;3 '
           'version adds 30 — negligible next to the ~150K-parameter encoder.'),
    bullet('<b>Safe initialisation.</b> &#934; can be initialised to reproduce exact '
           'replication (each output channel&#8217;s kernel = 1, bias = 0), so training '
           'starts at the current baseline and can only improve.'),
    bullet('<b>Two training options.</b> (a) freeze the existing encoder/decoder and train '
           'only &#934; on grayscale data — fast and cheap; or (b) fine-tune &#934; jointly '
           'with the encoder/decoder end-to-end for maximum flexibility.'),
    sp(6),
]
story.append(h2('2.4  What it does and does not change'))
story.append(dtable(
    ['Aspect', 'Effect of adding &#934;'],
    [['Reconstruction quality', 'May improve slightly; replication is already a strong baseline'],
     ['Paper narrative', 'Stronger — &#8220;learned projection&#8221; vs &#8220;copy the channel 3&#215;&#8221;'],
     ['Channel bandwidth (k)', 'Unchanged — &#934; only prepares the input; encoder still emits the same k symbols'],
     ['Training cost', 'Small extra run to fit &#934; (or fine-tune with encoder/decoder)']],
    [4.8*cm, 11.0*cm]))
story.append(sp(6))
story.append(body(
    '<b>Important:</b> &#934; does <u>not</u> reduce transmitted bandwidth — that would '
    'require a genuinely single-channel encoder (in_channels = 1 on the first layer, '
    'out_channels = 1 on the last decoder layer), a larger architectural change. &#934; is '
    'about a cleaner, more principled grayscale front-end, not bandwidth savings.'))
story.append(sp(6))
story.append(hr())
story.append(h2('Questions for discussion'))
story += [
    bullet('For the data-size result, do we retrain a dedicated low-SNR model (SNR_train '
           '&#8776; 7-10 dB) to align the bandwidth-saving and quality-win at one operating point?'),
    bullet('For grayscale: proceed with the learnable adapter &#934; (option a: frozen '
           'backbone), or go straight to a full single-channel encoder for a genuine '
           'bandwidth reduction?'),
]

doc = SimpleDocTemplate(os.path.join(HERE, 'ToR_DataSize_Grayscale_Report.pdf'), pagesize=A4,
                        leftMargin=2.2*cm, rightMargin=2.2*cm, topMargin=2.0*cm, bottomMargin=2.0*cm)
doc.build(story)
print('Saved ToR_DataSize_Grayscale_Report.pdf')
