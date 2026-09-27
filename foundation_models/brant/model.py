"""BRANT wrappers (official 505.68M-parameter release, pretrained on intracranial EEG).

Configuration of the released weights (BRANT paper, Sec. 3.2): temporal encoder 12 layers and
spatial (channel) encoder 5 layers, model dimension 2048, FFN 3072, 16 heads; patches of 1500
samples (6 s at 250 Hz), 15 patches per sequence, 8 frequency bands.

Input (following the official downstream preprocessing, which linearly interpolates every segment
to 15 x 1500 samples): each (C, 3000) window is linearly interpolated to 22500 samples and split
into 15 non-overlapping patches of 1500 samples per channel. For every patch and channel the
frequency input is the log absolute band power, log(sum of the PSD within the band), in the eight
BRANT bands (theta 4-8, alpha 8-13, beta 13-30, gamma1 30-50, gamma2 50-70, gamma3 70-90,
gamma4 90-110, gamma5 110-128 Hz), with the PSD computed at BRANT's 250 Hz sampling rate; the
model turns these into band weights with a softmax.

Classification (Appendix C.9): the temporal encoder models each channel over its 15 patches,
the channel encoder models the channels at each patch position, and the resulting (B, C, 15, 2048)
representations are averaged over the patch and channel dimensions and mapped to class logits by
a linear layer. CTC: the same input path; the 15 patch positions are the CTC time axis, and the
channel representations are concatenated at each position before the linear CTC head.

Weights: time_encoder.pt and channel_encoder.pt from the official release
(https://github.com/yzz673/Brant); set BRANT_CKPT_DIR to the folder that contains them.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from brant.brant_encoders import ChannelEncoder, TimeEncoder
from common import ctc, protocol

NAME = "BRANT"
SIZES = ["base"]  # single released checkpoint

SEQ_LEN = 15
PATCH_LEN = 1500
FS = 250.0
D_MODEL = 2048
DIM_FEEDFORWARD = 3072
TIME_LAYERS = 12
CHANNEL_LAYERS = 5
N_HEADS = 16
BANDS = [(4, 8), (8, 13), (13, 30), (30, 50), (50, 70), (70, 90), (90, 110), (110, 128)]

# Batch size (Appendix C.8): 32 when memory permits, otherwise 4-16, set by the number of channels.
# The same value is used for LP, FFT and scratch.
BATCH_SIZE_TIERS = ((8, 16), (16, 8), (None, 4))
CTC_BATCH_SIZE = {"base": protocol.batch_size(ctc.N_CHANNELS, BATCH_SIZE_TIERS)}


def batch_size(model_size, dataset_name, n_channels):
    return protocol.batch_size(n_channels, BATCH_SIZE_TIERS)


def _unwrap_ddp(state):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def band_log_power(patches):
    """(..., PATCH_LEN) -> (..., 8) log absolute band power from the periodogram at FS."""
    spec = torch.fft.rfft(patches.float(), dim=-1)
    psd = spec.real.square() + spec.imag.square()
    psd = psd / (FS * patches.shape[-1])
    psd[..., 1:] = psd[..., 1:] * 2  # one-sided spectrum
    freqs = torch.fft.rfftfreq(patches.shape[-1], d=1.0 / FS).to(patches.device)
    powers = [psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) for lo, hi in BANDS]
    return torch.log(torch.stack(powers, dim=-1) + 1e-10)


class _BrantBackbone(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.encoder_t = TimeEncoder(in_dim=PATCH_LEN, d_model=D_MODEL, dim_feedforward=DIM_FEEDFORWARD,
                                     seq_len=SEQ_LEN, n_layer=TIME_LAYERS, nhead=N_HEADS, band_num=len(BANDS),
                                     project_mode="linear", learnable_mask=True)
        self.encoder_ch = ChannelEncoder(out_dim=PATCH_LEN, d_model=D_MODEL, dim_feedforward=DIM_FEEDFORWARD,
                                         n_layer=CHANNEL_LAYERS, nhead=N_HEADS)
        if init == "pretrained":
            ckpt_dir = Path(os.environ.get("BRANT_CKPT_DIR", "."))
            for module, name in [(self.encoder_t, "time_encoder.pt"), (self.encoder_ch, "channel_encoder.pt")]:
                state = torch.load(ckpt_dir / name, map_location="cpu", weights_only=False)
                module.load_state_dict(_unwrap_ddp(state), strict=True)
        self.d_model = D_MODEL

    def patches(self, x):
        """(B, C, T) -> per-channel, per-patch representations (B, C, 15, d_model)."""
        b, c, _ = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.interpolate(x.float(), size=SEQ_LEN * PATCH_LEN, mode="linear", align_corners=True)
            x = x.view(b, c, SEQ_LEN, PATCH_LEN)
            power = band_log_power(x)  # (B, C, 15, 8)
        h = self.encoder_t(mask=None, data=x, power=power, need_mask=False)  # (B*C, 15, d)
        h = h.reshape(b, c, SEQ_LEN, D_MODEL).transpose(1, 2).reshape(b * SEQ_LEN, c, D_MODEL)
        h, _ = self.encoder_ch(h)  # attention across channels at each patch position
        return h.reshape(b, SEQ_LEN, c, D_MODEL).transpose(1, 2)


class BrantClassifier(nn.Module):
    def __init__(self, n_classes, init="pretrained"):
        super().__init__()
        self.backbone = _BrantBackbone(init)
        self.head = nn.Linear(D_MODEL, n_classes)

    def forward(self, x):
        return self.head(self.backbone.patches(x).mean(dim=(1, 2)))


class BrantCTC(nn.Module):
    def __init__(self, init="pretrained"):
        super().__init__()
        self.backbone = _BrantBackbone(init)
        self.ctc_head = ctc.make_ctc_head(ctc.N_CHANNELS * D_MODEL)

    def forward(self, x):
        h = self.backbone.patches(x)  # (B, C, 15, d_model)
        return self.ctc_head(h.permute(0, 2, 1, 3).flatten(2))  # (B, 15, vocab)


def build_classifier(model_size, n_channels, n_classes, init="pretrained"):
    return BrantClassifier(n_classes, init)


def build_ctc(model_size, init="pretrained"):
    return BrantCTC(init)
