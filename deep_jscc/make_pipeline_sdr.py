"""Draw the full Deep-JSCC-over-USRP demo pipeline (TX -> air -> RX) and put it
in a short PDF for the supervisor discussion.

Colour code:
  green  = already built (in sdr_demo.py: encoder, quantize/bits, dequantize, decoder)
  orange = to build next (the digital modem: modulation + pulse shaping + sync)
  blue   = real SDR hardware (USRP)
  grey   = the image at each end
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, 'results')
OUT_DIR = os.path.join(HERE, 'pdf result')
os.makedirs(OUT_DIR, exist_ok=True)

DONE = ('#16A085', '#D1F2EB')   # edge, fill
TODO = ('#E67E22', '#FDEBD0')
HW   = ('#2980B9', '#D6EAF8')
END  = ('#7F8C8D', '#EAECEE')

fig, ax = plt.subplots(figsize=(13.5, 6.6))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7); ax.axis('off')

BW, BH = 1.85, 1.05
xs = [1.25, 3.45, 5.65, 7.85, 10.05, 12.25]
TOP, BOT = 5.0, 1.2

def box(xc, yc, text, col):
    edge, fill = col
    ax.add_patch(FancyBboxPatch((xc-BW/2, yc-BH/2), BW, BH,
                 boxstyle='round,pad=0.02,rounding_size=0.12',
                 linewidth=1.8, edgecolor=edge, facecolor=fill))
    ax.text(xc, yc, text, ha='center', va='center', fontsize=9.5,
            color='#2C3E50', fontweight='bold')

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=16, linewidth=1.6, color='#566573'))

# ── TX row (left -> right) ─────────────────────────────────────────────────────
top = [
    ('Image\n(96x96 gray)', END),
    ('JSCC\nEncoder', DONE),
    ('Quantize\n-> bits', DONE),
    ('QPSK\nmodulate', TODO),
    ('RRC pulse\nshape (SPS=50)', TODO),
    ('USRP TX\n(RF out)', HW),
]
for xc, (t, c) in zip(xs, top):
    box(xc, TOP, t, c)
for i in range(len(xs)-1):
    arrow(xs[i]+BW/2, TOP, xs[i+1]-BW/2, TOP)

# ── over-the-air (down the right side) ─────────────────────────────────────────
arrow(xs[-1], TOP-BH/2, xs[-1], BOT+BH/2)
ax.text(xs[-1]+0.15, (TOP+BOT)/2, 'over the air\n(RF channel,\nreal noise)',
        ha='left', va='center', fontsize=8.5, color='#B03A2E', style='italic')

# ── RX row (right -> left) ─────────────────────────────────────────────────────
bot = [
    ('Image out\n(reconstructed)', END),   # x=1.25
    ('JSCC\nDecoder', DONE),                # 3.45
    ('Dequantize', DONE),                   # 5.65
    ('QPSK\ndemod -> bits', TODO),          # 7.85
    ('RRC matched\nfilter + sync', TODO),   # 10.05
    ('USRP RX\n(RF in)', HW),               # 12.25
]
for xc, (t, c) in zip(xs, bot):
    box(xc, BOT, t, c)
for i in range(len(xs)-1, 0, -1):
    arrow(xs[i]-BW/2, BOT, xs[i-1]+BW/2, BOT)

# ── row labels ─────────────────────────────────────────────────────────────────
ax.text(0.15, TOP+0.85, 'TRANSMITTER', fontsize=11, fontweight='bold', color='#2C3E50')
ax.text(0.15, BOT-0.85, 'RECEIVER', fontsize=11, fontweight='bold', color='#2C3E50')

# ── legend ─────────────────────────────────────────────────────────────────────
leg = [('Already built (sdr_demo.py)', DONE), ('To build next (digital modem)', TODO),
       ('SDR hardware (USRP)', HW), ('Image', END)]
leg_x = [0.4, 4.2, 7.9, 11.1]
for lx, (t, (edge, fill)) in zip(leg_x, leg):
    ax.add_patch(FancyBboxPatch((lx, 6.55), 0.32, 0.26,
                 boxstyle='round,pad=0.01,rounding_size=0.05',
                 linewidth=1.5, edgecolor=edge, facecolor=fill))
    ax.text(lx+0.42, 6.68, t, ha='left', va='center', fontsize=8.6, color='#2C3E50')

ax.set_title('Deep-JSCC over USRP — full transmit / receive pipeline',
             fontsize=13.5, fontweight='bold', color='#2C3E50', pad=14)
fig.tight_layout()
FIG = os.path.join(RES, 'sdr_pipeline.png')
fig.savefig(FIG, dpi=150, bbox_inches='tight'); plt.close(fig)
print('wrote', FIG)

# ── short PDF ──────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

C_DARK = colors.HexColor('#2C3E50'); C_TEAL = colors.HexColor('#16A085')
C_LIGHT = colors.HexColor('#ECF0F1'); C_WHITE = colors.white
styles = getSampleStyleSheet()
def stl(n, b='Normal', **k): return ParagraphStyle(n, parent=styles[b], **k)
S_TITLE = stl('T','Title', fontSize=19, textColor=C_DARK, alignment=TA_CENTER, spaceAfter=4)
S_SUB   = stl('S', fontSize=12, textColor=C_TEAL, alignment=TA_CENTER, spaceAfter=4)
S_META  = stl('M', fontSize=9.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=3)
S_BODY  = stl('B', fontSize=10.5, textColor=colors.HexColor('#2C2C2C'), leading=16,
              spaceAfter=6, alignment=TA_JUSTIFY)
S_CAP   = stl('CAP', fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=9)

def body(t): return Paragraph(t, S_BODY)
def cap(t): return Paragraph(f'<i>{t}</i>', S_CAP)
def sp(n=6): return Spacer(1, n)
def box_(title, color=C_TEAL):
    tb = Table([[Paragraph(f'<b>{title}</b>', stl('BX',fontSize=11,textColor=C_WHITE))]],
               colWidths=[16*cm])
    tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),color),
        ('LEFTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5)])); return tb
def dtable(header, rows, widths):
    data = [[Paragraph(f'<b>{h}</b>', S_BODY) for h in header]]
    for r in rows:
        data.append([Paragraph(str(v), S_BODY) for v in r])
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),C_TEAL),('TEXTCOLOR',(0,0),(-1,0),C_WHITE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_LIGHT,C_WHITE]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#CCCCCC')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)])); return t
def fig_img(path, width=17.0*cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return RLImage(path, width=width, height=width*h/w)

story = [
    Paragraph('ToR Project — SDR Demo Pipeline', S_TITLE),
    Paragraph('Transmitting an Image with Deep-JSCC over a USRP', S_SUB),
    sp(3),
    Paragraph('Prepared for: Supervisor&nbsp;&nbsp;|&nbsp;&nbsp;Author: Nam Khanh Tran'
              '&nbsp;&nbsp;|&nbsp;&nbsp;July 2026', S_META),
    sp(3), HRFlowable(width='100%', thickness=1.5, color=C_TEAL), sp(8),
]
story.append(body(
    'The full demo transmits a picture over a real radio (USRP). Bits cannot be put '
    'on the antenna directly &#8212; they must first be turned into radio symbols '
    '(modulation) and then into a smooth waveform (pulse shaping). The diagram shows '
    'the whole chain; green blocks are already built, orange blocks are the next step.'))
story.append(fig_img(FIG))
story.append(cap('Green = already built (sdr_demo.py). Orange = the digital modem to add '
                 'next. Blue = the USRP hardware.'))
story.append(box_('Each stage, in one line'))
story.append(sp(6))
story.append(dtable(
    ['Stage', 'What it does', 'Status'],
    [['JSCC Encoder', 'picture &#8594; compact complex symbols', 'done'],
     ['Quantize &#8594; bits', 'round each value to 7 bits', 'done'],
     ['QPSK modulate', 'map bit pairs to constellation points (00,01,10,11)', 'next'],
     ['RRC pulse shape', 'spread each symbol into a smooth waveform (no ISI)', 'next'],
     ['USRP TX / RX', 'send / receive the actual radio signal', 'hardware'],
     ['RRC matched filter + sync', 'clean up and time-align the received samples', 'next'],
     ['QPSK demod &#8594; bits', 'read constellation points back to bits', 'next'],
     ['Dequantize', 'bits &#8594; complex symbols', 'done'],
     ['JSCC Decoder', 'complex symbols &#8594; picture', 'done']],
    [4.6*cm, 8.8*cm, 2.4*cm]))

doc = SimpleDocTemplate(os.path.join(OUT_DIR, 'SDR_Pipeline.pdf'), pagesize=A4,
                        topMargin=1.5*cm, bottomMargin=1.5*cm, leftMargin=1.8*cm,
                        rightMargin=1.8*cm, title='SDR Demo Pipeline', author='Nam Khanh Tran')
doc.build(story)
print('Wrote', os.path.join(OUT_DIR, 'SDR_Pipeline.pdf'))
