"""Generate 'pdf result/SDR_Demo.pdf' — a short, simple explainer of the
image <-> bits SDR demo (sdr_demo.py), for the supervisor to review.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, 'results')
OUT_DIR = os.path.join(HERE, 'pdf result')
os.makedirs(OUT_DIR, exist_ok=True)

C_DARK  = colors.HexColor('#2C3E50')
C_TEAL  = colors.HexColor('#16A085')
C_LIGHT = colors.HexColor('#ECF0F1')
C_WHITE = colors.white
C_CODE  = colors.HexColor('#1B2631')

styles = getSampleStyleSheet()
def st(name, base='Normal', **kw): return ParagraphStyle(name, parent=styles[base], **kw)
S_TITLE = st('T', 'Title', fontSize=20, textColor=C_DARK, spaceAfter=4, alignment=TA_CENTER)
S_SUB   = st('S', 'Normal', fontSize=12, textColor=C_TEAL, alignment=TA_CENTER, spaceAfter=4)
S_META  = st('M', 'Normal', fontSize=9.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=3)
S_BODY  = st('B', 'Normal', fontSize=10.5, textColor=colors.HexColor('#2C2C2C'),
             leading=16, spaceAfter=6, alignment=TA_JUSTIFY)
S_BUL   = st('BL', 'Normal', fontSize=10.5, textColor=colors.HexColor('#2C2C2C'),
             leading=16, leftIndent=15, spaceAfter=4)
S_CAP   = st('CAP', 'Normal', fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=9)
S_CODE  = st('CO', 'Code', fontSize=9.5, textColor=C_WHITE, leading=14)
S_QN    = st('QN', 'Normal', fontSize=10, textColor=colors.HexColor('#7A4A00'),
             leading=15, leftIndent=15, spaceAfter=5)

def body(t): return Paragraph(t, S_BODY)
def bullet(t): return Paragraph(f'&#8226;&nbsp; {t}', S_BUL)
def cap(t): return Paragraph(f'<i>{t}</i>', S_CAP)
def sp(n=6): return Spacer(1, n)
def box(title, color=C_TEAL):
    tbl = Table([[Paragraph(f'<b>{title}</b>', st('BX','Normal',fontSize=11,textColor=C_WHITE))]],
                colWidths=[16*cm])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),color),
        ('LEFTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return tbl
def codebox(lines):
    p = [Paragraph(l, S_CODE) for l in lines]
    tbl = Table([[p]], colWidths=[16*cm])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),C_CODE),
        ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
        ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    return tbl
def fig_img(path, width=15.0*cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return RLImage(path, width=width, height=width*h/w)

story = [
    Paragraph('ToR Project — SDR Demo', S_TITLE),
    Paragraph('Sending an Image as Bits with Deep-JSCC', S_SUB),
    sp(4),
    Paragraph('Software-Defined Radio (SDR) round trip: image &#8594; bits &#8594; image',
              S_META),
    Paragraph('Prepared for: Supervisor&nbsp;&nbsp;|&nbsp;&nbsp;Author: Nam Khanh Tran'
              '&nbsp;&nbsp;|&nbsp;&nbsp;July 2026', S_META),
    sp(4), HRFlowable(width='100%', thickness=1.5, color=C_TEAL), sp(8),
]

story.append(body(
    'This is a small demo of how the Deep-JSCC model could send a picture over a '
    'radio. In a Software-Defined Radio (SDR), what travels over the air is '
    '<b>bits</b>. So the demo has two simple functions: one turns a picture into '
    'bits, and the other turns the bits back into a picture.'))

# 1. The two functions
story.append(box('1.  The two functions'))
story.append(sp(8))
story.append(codebox([
    'bits  = encode_to_bits(image)&nbsp;&nbsp;&nbsp;&nbsp;# transmitter: picture &#8594; bits',
    'image = decode_from_bits(bits)&nbsp;&nbsp;&nbsp;# receiver: bits &#8594; picture',
]))
story.append(sp(6))
story.append(bullet('<b>encode_to_bits</b> &#8212; the <b>transmitter</b>. It takes a '
                    'picture and gives back a string of 0s and 1s (the bits to send).'))
story.append(bullet('<b>decode_from_bits</b> &#8212; the <b>receiver</b>. It takes the '
                    'bits and rebuilds the picture.'))

# 2. How it works
story.append(box('2.  How it works (step by step)'))
story.append(sp(8))
story.append(body('<b>Transmitter</b> (inside encode_to_bits):'))
story.append(bullet('The neural <b>encoder</b> turns the picture into a small set of '
                    'numbers (the "symbols" to send).'))
story.append(bullet('Each number is <b>rounded to 7 bits</b> (quantized) so it can be '
                    'written as bits. These bits are the output.'))
story.append(sp(2))
story.append(body('<b>Over the air:</b> a real SDR sends the bits as a radio signal '
                  'and receives them on the other side. The radio channel is noisy, '
                  'set by the signal-to-noise ratio (SNR).'))
story.append(sp(2))
story.append(body('<b>Receiver</b> (inside decode_from_bits):'))
story.append(bullet('Read the bits back into numbers.'))
story.append(bullet('Add the <b>channel noise</b> (here SNR = 19 dB, the same setting '
                    'the model was trained with).'))
story.append(bullet('The neural <b>decoder</b> turns the numbers back into a picture.'))

# 3. Result
story.append(box('3.  Result'))
story.append(sp(8))
story.append(body('We ran one grayscale picture through the whole loop '
                  '(picture &#8594; bits &#8594; picture):'))
story.append(sp(2))
story.append(bullet('Picture size: <b>96 &#215; 96</b> grayscale'))
story.append(bullet('Bits sent: <b>32,256 bits</b> (about 4,032 bytes)'))
story.append(bullet('Quality after the trip: about <b>30 dB</b> at SNR 19 &#8212; the '
                    'decoded picture looks almost the same as the original.'))
story.append(sp(4))
sdr_png = os.path.join(RES, 'sdr_roundtrip.png')
if os.path.exists(sdr_png):
    story.append(fig_img(sdr_png, width=15.5*cm))
    story.append(cap('Left: the original picture. Middle: decoded after the noisy radio '
                     'channel (SNR 19). Right: decoded over a clean link. The picture '
                     'survives the trip.'))

doc = SimpleDocTemplate(os.path.join(OUT_DIR, 'SDR_Demo.pdf'),
                        pagesize=A4, topMargin=1.6*cm, bottomMargin=1.6*cm,
                        leftMargin=2.0*cm, rightMargin=2.0*cm,
                        title='SDR Demo', author='Nam Khanh Tran')
doc.build(story)
print('Wrote', os.path.join(OUT_DIR, 'SDR_Demo.pdf'))
