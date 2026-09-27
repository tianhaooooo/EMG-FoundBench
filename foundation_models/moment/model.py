"""MOMENT-1 (small / base) wrappers.

Classification (Appendix C.9): the EMG window is passed directly to MOMENT's native multichannel
input; each channel is normalized, patchified (patch length 8) and encoded by the shared T5
encoder. Patch representations are averaged over time, concatenated across channels and mapped
to class logits by a linear layer (momentfm classification head, reduction="concat").

CTC: the same input path; per-patch embeddings (375 patches for 3000 samples) are concatenated
across the 32 channels at each patch position and fed to the linear CTC head.
"""
import torch.nn as nn
from momentfm import MOMENTPipeline

from common import ctc

NAME = "MOMENT"
SIZES = ["small", "base"]
MODEL_IDS = {"small": "AutonLab/MOMENT-1-small", "base": "AutonLab/MOMENT-1-base"}

# Batch size (Appendix C.8): 32 when memory permits, otherwise 4-16.
# The same value is used for LP, FFT and scratch so paired comparisons share one batch size.
DEFAULT_BATCH_SIZE = {"small": 32, "base": 8}
BATCH_SIZE_OVERRIDE = {
    "small": {"CapgMyo": 4, "HDEMG": 8, "GRABMyo": 16},
    "base": {"CapgMyo": 4, "HDEMG": 4, "GRABMyo": 4},
}
CTC_BATCH_SIZE = {"small": 16, "base": 4}


def batch_size(model_size, dataset_name, n_channels):
    return BATCH_SIZE_OVERRIDE[model_size].get(dataset_name, DEFAULT_BATCH_SIZE[model_size])


def _reinitialize(module):
    """Randomly re-initialize every parameter of an already-constructed model."""
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        elif isinstance(getattr(m, "weight", None), nn.Parameter) and "norm" in type(m).__name__.lower():
            nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)


def _load(model_size, task_name, **kwargs):
    pipeline = MOMENTPipeline.from_pretrained(
        MODEL_IDS[model_size],
        model_kwargs={"task_name": task_name, "freeze_embedder": False, "freeze_encoder": False,
                      "freeze_head": False, **kwargs},
    )
    pipeline.init()
    return pipeline


class MomentClassifier(nn.Module):
    def __init__(self, model_size, n_channels, n_classes, init="pretrained"):
        super().__init__()
        self.moment = _load(model_size, "classification", n_channels=n_channels, num_class=n_classes)
        if init == "random":
            _reinitialize(self.moment)

    @property
    def head(self):
        return self.moment.head

    def forward(self, x):
        return self.moment(x_enc=x).logits


class MomentCTC(nn.Module):
    def __init__(self, model_size, init="pretrained"):
        super().__init__()
        self.moment = _load(model_size, "embedding")
        if init == "random":
            _reinitialize(self.moment)
        self.ctc_head = ctc.make_ctc_head(ctc.N_CHANNELS * self.moment.config.d_model)

    def forward(self, x):
        h = self.moment.embed(x_enc=x, reduction="none").embeddings  # (B, C, P, d_model)
        return self.ctc_head(h.permute(0, 2, 1, 3).flatten(2))  # (B, P, vocab)


def build_classifier(model_size, n_channels, n_classes, init="pretrained"):
    return MomentClassifier(model_size, n_channels, n_classes, init)


def build_ctc(model_size, init="pretrained"):
    return MomentCTC(model_size, init)
