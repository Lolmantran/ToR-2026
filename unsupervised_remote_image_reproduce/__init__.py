"""
Cycle-CNN – Unsupervised Remote Sensing Image Super-Resolution
Wang et al., IGARSS 2019

Package layout
--------------
models.py       — G1, G2, ResBlock, SubpixelConv, SRCNN
losses.py       — cycle_loss, identity_loss, total_loss
data_loader.py  — UnpairedSRDataset, SyntheticSRDataset, get_dataloaders
utils.py        — checkpointing, metrics, visualisation
train.py        — full training loop (CLI entry-point)
evaluate.py     — Table-1 evaluation (CLI entry-point)
"""
