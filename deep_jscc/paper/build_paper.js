// Build a conference-format (IEEE-style, two-column, Arial) paper skeleton.
// No em-dashes anywhere. Each section briefly introduces its concept.
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, SectionType,
  ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, VerticalAlign,
} = require('C:/nvm4w/nodejs/node_modules/docx');
const RES = path.join(__dirname, '..', 'results');

const PAGE = {
  size: { width: 12240, height: 15840 },            // US Letter
  margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
};

// ── paragraph helpers ──────────────────────────────────────────────────────────
const title = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ text: t, bold: true, size: 40 })],
});
const authors = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: t, size: 22 })],
});
const affil = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: t, italics: true, size: 20 })],
});
const abstractPara = (t) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
  children: [
    new TextRun({ text: 'Abstract—', bold: true, italics: true, size: 18 }),
    new TextRun({ text: t, italics: true, size: 18 }),
  ],
});
const indexTerms = (t) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
  children: [
    new TextRun({ text: 'Index Terms—', bold: true, italics: true, size: 18 }),
    new TextRun({ text: t, italics: true, size: 18 }),
  ],
});
const heading = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text: t, bold: true, size: 20 })],
});
const subheading = (t) => new Paragraph({
  spacing: { before: 120, after: 60 },
  children: [new TextRun({ text: t, italics: true, size: 20 })],
});
const body = (t) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 },
  children: [new TextRun({ text: t, size: 20 })],
});
const note = (t) => new Paragraph({          // editing placeholder, grey italic
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 140 },
  children: [new TextRun({ text: t, italics: true, size: 18, color: '888888' })],
});
const refItem = (t) => new Paragraph({
  spacing: { after: 40 }, indent: { left: 200, hanging: 200 },
  children: [new TextRun({ text: t, size: 18 })],
});

// ── figure / caption / table helpers ────────────────────────────────────────────
const FIGW = 300;   // px, fits a two-column IEEE column
function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function figure(name) {
  const p = path.join(RES, name); const { w, h } = pngSize(p);
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 80, after: 20 },
    children: [new ImageRun({ type: 'png', data: fs.readFileSync(p),
      transformation: { width: FIGW, height: Math.round(FIGW * h / w) } })],
  });
}
const caption = (n, t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 140 },
  children: [
    new TextRun({ text: `Fig. ${n}. `, size: 16 }),
    new TextRun({ text: t, size: 16 }),
  ],
});
const tableCaption = (n, t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
  children: [
    new TextRun({ text: `TABLE ${n}. `, size: 16, bold: true }),
    new TextRun({ text: t, size: 16 }),
  ],
});
function cell(text, w, header) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: header ? { type: ShadingType.CLEAR, fill: 'DDDDDD', color: 'auto' } : undefined,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 20, after: 20 },
      children: [new TextRun({ text: String(text), bold: !!header, size: 16 })] })],
  });
}
function makeTable(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const head = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true)) });
  const body = rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false)) }));
  return new Table({ columnWidths: widths, width: { size: total, type: WidthType.DXA }, rows: [head, ...body] });
}

// ── front matter (single column, full width) ────────────────────────────────────
const front = [
  title('Grayscale Deep Joint Source-Channel Coding with '
      + 'Bottleneck-Adaptive Bandwidth Saving and a '
      + 'Software-Defined-Radio Proof of Concept'),
  authors('Nam Khanh Tran, Tom [Last Name], Zhitong Ni'),
  affil('[Department], [University], [City, Country]'),
  affil('Email: [nam.email], [tom.email], [zhitong.email]'),
];

// ── two-column body ─────────────────────────────────────────────────────────────
const cols = [];

cols.push(abstractPara(
  'This paper studies a grayscale variant of deep joint source-channel coding '
  + '(Deep JSCC) for wireless image transmission. Starting from the colour '
  + 'reference model, we adapt the system to grayscale images, reduce the amount '
  + 'of transmitted data by narrowing the encoder bottleneck, and demonstrate the '
  + 'end-to-end system as a software-defined-radio pipeline that transmits an image '
  + 'as a bitstream. [Draft abstract: fill in the final numbers for PSNR, data '
  + 'saving, and the demonstration result.]'));
cols.push(indexTerms(
  'Deep joint source-channel coding, grayscale image transmission, bandwidth '
  + 'reduction, quantization, software-defined radio, USRP, STL-10.'));

// I. Introduction
cols.push(heading('I. Introduction'));
cols.push(body(
  'Wireless image transmission is traditionally built from two separate stages: '
  + 'source coding (for example JPEG) that compresses the image into bits, and '
  + 'channel coding that protects those bits for transmission. This separation '
  + 'works well at the signal-to-noise ratio (SNR) it was designed for, but the '
  + 'quality collapses sharply once the channel becomes worse than expected, an '
  + 'effect known as the cliff effect. Deep JSCC replaces both stages with a single '
  + 'neural network that maps image pixels directly to channel symbols, and it '
  + 'degrades gracefully as the channel changes.'));
cols.push(body(
  'In this work we focus on a grayscale version of Deep JSCC and make three '
  + 'contributions: (i) a grayscale model built from the colour reference; (ii) a '
  + 'bandwidth-saving study that reduces the transmitted data by narrowing the '
  + 'encoder bottleneck; and (iii) a software-defined-radio (SDR) proof of concept '
  + 'that sends an image as a real bitstream.'));
cols.push(note(
  '[Expand: motivation for grayscale, why bandwidth matters, and an outline of the '
  + 'paper organization.]'));

// II. Background and Related Work
cols.push(heading('II. Background and Related Work'));
cols.push(body(
  'This section reviews the Deep JSCC model that our work builds on and places it '
  + 'next to classical digital transmission. We summarize the encoder-decoder '
  + 'architecture, the additive white Gaussian noise (AWGN) channel model, and the '
  + 'bandwidth ratio that measures how many channel uses are spent per source '
  + 'pixel.'));
cols.push(note(
  '[Expand: the reference Deep JSCC paper, other related JSCC works, and the JPEG '
  + 'plus digital modulation baseline used later for comparison.]'));

// III. System Model
cols.push(heading('III. System Model'));
cols.push(body(
  'We describe the end-to-end system: a convolutional encoder that maps an image to '
  + 'complex channel symbols under an average power constraint, an AWGN channel, and '
  + 'a convolutional decoder that reconstructs the image. An important clarification '
  + 'is that the network uses real-valued weights throughout; the complex channel '
  + 'symbols are formed by pairing the real outputs of the last layer into '
  + 'in-phase (I) and quadrature (Q) components.'));
cols.push(body(
  'For grayscale operation, a small learnable adapter maps the single luminance '
  + 'channel to the three-channel input that the encoder expects, so the rest of the '
  + 'network is reused without change.'));
cols.push(note(
  '[Expand: layer table, the power-normalization equation, and the exact symbol '
  + 'count k = 24 x 24 x c.]'));

// IV. Proposed Method
cols.push(heading('IV. Proposed Method'));
cols.push(body(
  'Our method keeps the reference architecture and changes only what is needed for '
  + 'grayscale operation and for controlling the amount of transmitted data.'));
cols.push(subheading('A. Grayscale Variant'));
cols.push(body(
  'Images are converted to luminance, and a learnable one-to-three-channel adapter '
  + 'is placed before the encoder. The rest of the model is unchanged, which lets us '
  + 'isolate the effect of moving from colour to grayscale.'));
cols.push(note('[Expand: adapter design and the reconstruction target.]'));
cols.push(subheading('B. Bottleneck-Adaptive Bandwidth Saving'));
cols.push(body(
  'The final encoder layer sets the number of transmitted symbols, k = 24 x 24 x c, '
  + 'where c is the bottleneck width. Reducing c lowers the transmitted data by a '
  + 'fixed fraction for every image and at every channel condition. We study c from '
  + '8 down to 4.'));
cols.push(note('[Expand: symbol counting and the data-size framing in channel uses.]'));
cols.push(subheading('C. Quantization to Bytes'));
cols.push(body(
  'To express the transmitted data in bytes and to prepare for a digital radio link, '
  + 'each symbol value is quantized to a small number of bits. We find that seven '
  + 'bits per value is near lossless.'));
cols.push(note('[Expand: scalar quantization, the clip range, and the byte metric.]'));

// V. Experimental Setup
cols.push(heading('V. Experimental Setup'));
cols.push(body(
  'All experiments use the STL-10 dataset at 96 by 96 resolution, an AWGN channel, '
  + 'and a training SNR of 19 dB. Reconstruction quality is measured by peak '
  + 'signal-to-noise ratio (PSNR) on a held-out test set of 8000 images.'));
cols.push(note(
  '[Expand: optimizer, learning rate and schedule, number of training images, and a '
  + 'generic statement of the compute used.]'));

// VI. Results
cols.push(heading('VI. Results'));
cols.push(body(
  'We first reproduce and strengthen the baseline, then present the grayscale '
  + 'bottleneck sweep, the quantization analysis, and a comparison against a '
  + 'classical JPEG plus modulation baseline.'));
cols.push(subheading('A. Baseline and Training'));
cols.push(body(
  'Under the corrected training recipe, the colour reference model reaches '
  + '30.25 dB validation PSNR at 19 dB SNR, above the 28.99 dB obtained with the '
  + 'original configuration. The grayscale models train stably to convergence with '
  + 'the same recipe. Fig. 1 shows the validation PSNR during training for the five '
  + 'grayscale bottleneck widths.'));
cols.push(figure('gray_training_curves.png'));
cols.push(caption(1, 'Validation PSNR during grayscale training, widths c = 8 to 4, '
  + 'identical recipe.'));

cols.push(subheading('B. Grayscale Bottleneck Sweep'));
cols.push(body(
  'Table I reports the transmitted data and reconstruction quality for bottleneck '
  + 'widths c from 8 down to 4. Reducing c lowers the number of channel uses '
  + 'k = 24 x 24 x c proportionally, from 4,608 symbols at c = 8 to 2,304 at c = 4, a '
  + '50 percent reduction. All five models stay at or above 30 dB at the 19 dB '
  + 'operating point, and quality decreases smoothly as c is reduced. Fig. 2 shows '
  + 'PSNR against channel SNR.'));
cols.push(tableCaption('I', 'Grayscale bottleneck sweep (PSNR at 19 dB SNR).'));
cols.push(makeTable(
  ['c', 'k (symbols)', 'Data saved', 'PSNR (dB)'],
  [['8', '4,608', 'ref.', '32.32'],
   ['7', '4,032', '12.5%', '31.63'],
   ['6', '3,456', '25.0%', '30.99'],
   ['5', '2,880', '37.5%', '30.63'],
   ['4', '2,304', '50.0%', '30.16']],
  [800, 1400, 1200, 1400]));
cols.push(figure('gray_psnr_vs_snr.png'));
cols.push(caption(2, 'Grayscale PSNR versus channel SNR for c = 8 to 4. Wider c gives '
  + 'higher PSNR; all curves degrade gracefully.'));

cols.push(subheading('C. Quantization Analysis'));
cols.push(body(
  'To send the symbols as bits, each symbol value is quantized with N bits over the '
  + 'range that covers the symbol distribution. Table II reports the effect on the '
  + 'chosen c = 4 model. Seven bits per symbol is near lossless, costing only 0.03 dB, '
  + 'and gives a data-transfer footprint of k x 7 / 8 = 2,016 bytes per image. Below '
  + 'five bits the quality falls quickly. Fig. 3 shows the same trend.'));
cols.push(tableCaption('II', 'Quantization of the c = 4 model (channel at 19 dB SNR).'));
cols.push(makeTable(
  ['Bits', 'Bytes', 'PSNR (dB)', 'Loss (dB)'],
  [['8', '2,304', '30.15', '0.01'],
   ['7', '2,016', '30.13', '0.03'],
   ['6', '1,728', '30.04', '0.12'],
   ['5', '1,440', '29.71', '0.45'],
   ['4', '1,152', '28.49', '1.67'],
   ['3', '864', '24.88', '5.28']],
  [900, 1300, 1300, 1300]));
cols.push(figure('gray_quant_bits.png'));
cols.push(caption(3, 'Reconstruction PSNR of the c = 4 model versus bits per symbol. '
  + 'Quality is flat down to seven bits.'));

cols.push(subheading('D. Comparison with a Digital Baseline'));
cols.push(body(
  'We compare grayscale Deep JSCC against a digital baseline of JPEG at quality 73 '
  + 'followed by uncoded BPSK over the same AWGN channel. Fig. 4 shows the result. '
  + 'The digital baseline shows a sharp cliff: it fails completely below about 10 dB '
  + 'and reaches roughly 33 dB above 13 dB. Deep JSCC degrades gracefully and stays '
  + 'usable across the whole range. At 19 dB the digital baseline is slightly higher '
  + 'in PSNR, but the c = 4 model transmits fewer bytes (2,016) than the average JPEG '
  + 'file (about 2,317 bytes) while avoiding the cliff.'));
cols.push(figure('gray_vs_jpeg.png'));
cols.push(caption(4, 'Grayscale Deep JSCC versus JPEG plus BPSK over AWGN. The digital '
  + 'baseline fails below its cliff; Deep JSCC does not.'));

// VII. SDR Demonstration
cols.push(heading('VII. Software-Defined-Radio Demonstration'));
cols.push(body(
  'We implement the transmitter and receiver as two functions: one maps an image to '
  + 'a bitstream (encoder, quantization, bit packing), and the other maps a bitstream '
  + 'back to an image. Bits cannot be placed on the antenna directly; they are first '
  + 'mapped to constellation symbols using QPSK, and then shaped into a smooth '
  + 'waveform by a root-raised-cosine filter before the radio transmits them. The '
  + 'receiver applies the matched filter, recovers timing, demodulates back to bits, '
  + 'and reconstructs the image. Fig. 5 shows the full pipeline.'));
cols.push(figure('sdr_pipeline.png'));
cols.push(caption(5, 'Full Deep JSCC over USRP pipeline. Green blocks are built, '
  + 'orange blocks are the digital modem to add, blue blocks are the radio hardware.'));
cols.push(body(
  'Fig. 6 shows an image to bits to image round trip through the software pipeline. '
  + 'At 19 dB SNR the reconstruction is about 30 dB, matching the c = 4 model, and the '
  + 'picture is visually close to the original. Over-the-air transmission on a USRP is '
  + 'left as future work; the digital modem (QPSK plus pulse shaping) and '
  + 'synchronization are the next components to add.'));
cols.push(figure('sdr_roundtrip.png'));
cols.push(caption(6, 'Round trip through the software pipeline: input, decoded at '
  + '19 dB SNR, and decoded over a clean link.'));

// VIII. Discussion and Limitations
cols.push(heading('VIII. Discussion and Limitations'));
cols.push(body(
  'We discuss the trade-off between transmitted data and reconstruction quality, and '
  + 'we note that grayscale PSNR is not directly comparable to colour PSNR because a '
  + 'grayscale image is an easier reconstruction target. We also state the current '
  + 'scope of the SDR demonstration.'));
cols.push(note('[Expand as the results are finalized.]'));

// IX. Conclusion and Future Work
cols.push(heading('IX. Conclusion and Future Work'));
cols.push(body(
  'We summarize the grayscale Deep JSCC system, the bandwidth savings obtained by '
  + 'reducing the bottleneck, and the SDR proof of concept. Over-the-air testing on a '
  + 'USRP is identified as the main direction for future work.'));

// References
cols.push(heading('References'));
cols.push(refItem(
  '[1] E. Bourtsoulatze, D. Burth Kurka, and D. Gunduz, "Deep Joint Source-Channel '
  + 'Coding for Wireless Image Transmission," IEEE Transactions on Cognitive '
  + 'Communications and Networking, 2019.'));
cols.push(refItem('[2] [Add references as needed.]'));

// ── assemble ────────────────────────────────────────────────────────────────────
const doc = new Document({
  styles: { default: { document: { run: { font: 'Times New Roman', size: 20 } } } },
  sections: [
    { properties: { page: PAGE }, children: front },
    {
      properties: {
        type: SectionType.CONTINUOUS,
        page: PAGE,
        column: { count: 2, space: 480 },
      },
      children: cols,
    },
  ],
});

const out = path.join(__dirname, 'TOR_Paper_Draft.docx');
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log('Wrote', out);
});
