"""Chronos-2 (small / base) wrappers.

Classification (Appendix C.9): the EMG channels of a window are treated as related variates of
one grouped multivariate series (shared group ID per window, different IDs across windows). Each
channel goes through Chronos-2's native normalization, patching (patch length 16) and input
embedding, followed by its temporal and group-attention encoder. The context-patch
representations are averaged over patches and then over channels, and mapped to class logits by
a linear layer. The [REG] token and the forecast-patch positions are not used.

CTC: the same input path; context-patch representations (188 patches for 3000 samples) are
concatenated across the 32 channels at each patch position and fed to the linear CTC head.
"""
import torch
import torch.nn as nn
from chronos import Chronos2Model
from chronos.chronos2.config import Chronos2CoreConfig

from common import ctc, protocol

NAME = "Chronos-2"
SIZES = ["small", "base"]
MODEL_IDS = {"small": "autogluon/chronos-2-small", "base": "autogluon/chronos-2"}

# Batch size (Appendix C.8): 32 when memory permits, otherwise 4-16, set by the number of channels.
# The same value is used for LP, FFT and scratch.
BATCH_SIZE_TIERS = {"small": ((24, 32), (48, 16), (96, 8), (None, 4)),
                    "base": ((6, 32), (12, 16), (24, 8), (None, 4))}
CTC_BATCH_SIZE = {s: protocol.batch_size(ctc.N_CHANNELS, BATCH_SIZE_TIERS[s]) for s in BATCH_SIZE_TIERS}


def batch_size(model_size, dataset_name, n_channels):
    return protocol.batch_size(n_channels, BATCH_SIZE_TIERS[model_size])


def _load_encoder(model_size, init):
    if init == "random":
        config = Chronos2CoreConfig.from_pretrained(MODEL_IDS[model_size])
        return Chronos2Model(config)
    return Chronos2Model.from_pretrained(MODEL_IDS[model_size])


class _Chronos2Backbone(nn.Module):
    def __init__(self, model_size, init):
        super().__init__()
        self.encoder = _load_encoder(model_size, init)
        self.d_model = self.encoder.model_dim

    def context_patches(self, x):
        """(B, C, T) -> per-channel context-patch representations (B, C, P, d_model)."""
        b, c, t = x.shape
        if t > self.encoder.chronos_config.context_length:
            raise ValueError(f"window length {t} exceeds the Chronos-2 context length")
        group_ids = torch.arange(b, device=x.device).repeat_interleave(c)
        outputs, _, _, n_ctx = self.encoder.encode(context=x.reshape(b * c, t), group_ids=group_ids,
                                                   num_output_patches=1)
        h = outputs.last_hidden_state[:, :n_ctx]  # drop [REG] and forecast positions
        return h.reshape(b, c, n_ctx, self.d_model)


class Chronos2Classifier(nn.Module):
    def __init__(self, model_size, n_channels, n_classes, init="pretrained"):
        super().__init__()
        self.backbone = _Chronos2Backbone(model_size, init)
        self.head = nn.Linear(self.backbone.d_model, n_classes)

    def forward(self, x):
        h = self.backbone.context_patches(x).mean(dim=2).mean(dim=1)  # patches, then channels
        return self.head(h)


class Chronos2CTC(nn.Module):
    def __init__(self, model_size, init="pretrained"):
        super().__init__()
        self.backbone = _Chronos2Backbone(model_size, init)
        self.ctc_head = ctc.make_ctc_head(ctc.N_CHANNELS * self.backbone.d_model)

    def forward(self, x):
        h = self.backbone.context_patches(x)  # (B, C, P, d_model)
        return self.ctc_head(h.permute(0, 2, 1, 3).flatten(2))  # (B, P, vocab)


def build_classifier(model_size, n_channels, n_classes, init="pretrained"):
    return Chronos2Classifier(model_size, n_channels, n_classes, init)


def build_ctc(model_size, init="pretrained"):
    return Chronos2CTC(model_size, init)
