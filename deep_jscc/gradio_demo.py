"""Interactive web demo for the grayscale Deep-JSCC project.

Transmit an image through the learned joint source-channel code and watch it
survive a noisy channel. Reuses the transmitter/receiver functions from
sdr_demo.py.

Run:   python gradio_demo.py         (opens http://127.0.0.1:7860)
Share: set SHARE=True in launch() for a temporary public link.
Deploy: push this file + checkpoints/ + *.py to a Hugging Face Space (CPU is fine).
"""
import os, math
import numpy as np
import gradio as gr

from sdr_demo import get_model, encode_to_bits, decode_from_bits, _rgb_to_gray, NBITS

HERE = os.path.dirname(os.path.abspath(__file__))
CS = [8, 7, 6, 5, 4]

# preload the five grayscale models so switching is instant
for c in CS:
    get_model('gray', c)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse < 1e-9 else 10 * math.log10(255.0 ** 2 / mse)


def transmit(image, c, snr_db, clean_link):
    """Encode -> (channel) -> decode, and report the numbers."""
    if image is None:
        return None, None, "Please upload or select an image first."

    model = get_model('gray', int(c))
    gray_in = _rgb_to_gray(np.asarray(image)) if np.asarray(image).ndim == 3 \
        else np.asarray(image)

    bits = encode_to_bits(image, model=model)
    snr = 'clean' if clean_link else float(snr_db)
    recon = decode_from_bits(bits, snr_db=snr, model=model)

    # match sizes for PSNR (encoder resizes to 96x96)
    from PIL import Image as PILImage
    gin = np.array(PILImage.fromarray(gray_in).resize((96, 96)))
    psnr = _psnr(gin, recon)

    k = int(c) * 24 * 24
    n_bits = len(bits)
    saved = 100 * (1 - k / (8 * 24 * 24))
    report = (
        f"### Transmission report\n"
        f"| Quantity | Value |\n|---|---|\n"
        f"| Bottleneck width c | **{int(c)}** |\n"
        f"| Channel symbols (k) | {k:,} |\n"
        f"| Data saved vs c=8 | **{saved:.1f}%** |\n"
        f"| Bitstream sent | {n_bits:,} bits ({n_bits // 8:,} bytes) |\n"
        f"| Channel | {'clean link' if clean_link else f'AWGN, SNR = {snr_db:.0f} dB'} |\n"
        f"| Reconstruction PSNR | **{psnr:.2f} dB** |\n"
    )
    # upscale for nicer display
    gin_disp = np.array(PILImage.fromarray(gin).resize((240, 240), PILImage.NEAREST))
    rec_disp = np.array(PILImage.fromarray(recon).resize((240, 240), PILImage.NEAREST))
    return gin_disp, rec_disp, report


# ── examples from STL-10 ────────────────────────────────────────────────────────
EX_DIR = os.path.join(HERE, 'demo_examples')
example_files = []
if os.path.isdir(EX_DIR):
    example_files = [os.path.join(EX_DIR, f) for f in sorted(os.listdir(EX_DIR))
                     if f.lower().endswith(('.png', '.jpg'))]

INTRO = """
# Deep JSCC: Wireless Image Transmission Demo

This demo sends a **grayscale image** through a learned **joint source-channel
code** over a simulated noisy wireless channel. Unlike JPEG plus a digital radio
(which fails suddenly when the channel gets bad), Deep JSCC **degrades
gracefully**.

**Try it:** pick an image, choose the bottleneck width **c** (smaller c = less
data sent) and the channel **SNR** (lower = noisier), then press *Transmit*.
Watch how quality changes with the channel and with how much data you send.
"""

# light blue-and-white theme
LIGHT_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#eef4fb",
    block_background_fill="#ffffff",
    block_border_color="#d5e3f2",
    block_title_text_color="#1b3a5b",
    body_text_color="#1b3a5b",
    button_primary_background_fill="#2f6fb2",
    button_primary_background_fill_hover="#255c96",
    button_primary_text_color="#ffffff",
)

# force light mode even if the browser is set to dark (runs before Gradio renders)
FORCE_LIGHT_HEAD = """
<script>
(function () {
  var u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.href);
  }
})();
</script>
"""

with gr.Blocks(title="Deep JSCC Image Transmission",
               theme=LIGHT_THEME, head=FORCE_LIGHT_HEAD) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Image to send", type="numpy", height=240)
            if example_files:
                gr.Examples(examples=[[f] for f in example_files], inputs=[inp],
                            label="Example images (STL-10)")
            c_sel = gr.Dropdown(CS, value=4, label="Bottleneck width c "
                                "(smaller = less data)")
            snr = gr.Slider(0, 25, value=19, step=1, label="Channel SNR (dB)")
            clean = gr.Checkbox(False, label="Clean link (no channel noise)")
            btn = gr.Button("Transmit", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                out_in = gr.Image(label="Sent (grayscale)", height=240)
                out_rec = gr.Image(label="Received (reconstructed)", height=240)
            report = gr.Markdown()

    btn.click(transmit, [inp, c_sel, snr, clean], [out_in, out_rec, report])

    gr.Markdown(
        "Model: grayscale Deep JSCC on STL-10, trained at SNR 19 dB. "
        "Encoder and decoder use real-valued weights; the transmitted symbols are "
        f"quantized to {NBITS} bits each. Based on Bourtsoulatze et al., 2019.")


if __name__ == '__main__':
    demo.launch()
