"""Lag-Llama wrappers.

Lag-Llama is a univariate decoder-only forecaster whose tokens are lag features. Classification
(Appendix C.9):

    (B, C, T) --channel-independent--> (B*C, T)
      -> native robust scaling of each series
      -> native lag-feature construction: with a maximum lag of 1092, the first 1092 samples serve
         only as lag history and the remaining T - 1092 samples become tokens (1908 for T = 3000);
         84 lag values + 2 scale features + 6 time features (set to zero) per token
      -> token embedding -> 8 transformer blocks -> final RMSNorm -> (B*C, T - 1092, 144)
      -> mean over tokens, then over channels -> Linear(144, n_classes)

The distribution head (param_proj) is not used. CTC: the same input path; token representations
are concatenated across the 32 channels at each token position and fed to the linear CTC head.

Requires the upstream Lag-Llama code (https://github.com/time-series-foundation-models/lag-llama,
commit df7531a), which is not pip-installable: set LAG_LLAMA_SRC to the root of a checkout
(the directory that contains lag_llama/ and gluon_utils/). Pretrained weights: lag-llama.ckpt from
the Hugging Face repo time-series-foundation-models/Lag-Llama.
"""
import os
import sys

import torch
import torch.nn as nn

if os.environ.get("LAG_LLAMA_SRC"):
    sys.path.insert(0, os.environ["LAG_LLAMA_SRC"])
from lag_llama.model.module import LagLlamaModel  # noqa: E402

from common import ctc, protocol  # noqa: E402

NAME = "Lag-Llama"
SIZES = ["base"]  # single released checkpoint
HF_REPO = "time-series-foundation-models/Lag-Llama"
N_TIME_FEATURES = 6

# Batch size (Appendix C.8): 32 when memory permits, otherwise 4-16, set by the number of channels.
# The same value is used for LP, FFT and scratch.
BATCH_SIZE_TIERS = ((4, 32), (8, 16), (16, 8), (None, 4))
CTC_BATCH_SIZE = {"base": protocol.batch_size(ctc.N_CHANNELS, BATCH_SIZE_TIERS)}


def batch_size(model_size, dataset_name, n_channels):
    return protocol.batch_size(n_channels, BATCH_SIZE_TIERS)


def _load_checkpoint():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, "lag-llama.ckpt")
    # Lightning checkpoint whose hyper-parameters contain a pickled gluonts distribution object.
    return torch.load(path, map_location="cpu", weights_only=False)


class _LagLlamaBackbone(nn.Module):
    def __init__(self, init):
        super().__init__()
        ckpt = _load_checkpoint()
        self.model = LagLlamaModel(**ckpt["hyper_parameters"]["model_kwargs"])  # architecture only
        if init == "pretrained":
            state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            self.model.load_state_dict(state, strict=True)
        self.max_lag = max(self.model.lags_seq)
        self.d_model = self.model.transformer.wte.out_features

    def tokens(self, x):
        """(B, C, T) -> token representations (B, C, T - max_lag, d_model)."""
        b, c, t = x.shape
        if t <= self.max_lag:
            raise ValueError(f"window length {t} must exceed the maximum lag {self.max_lag}")
        series = x.reshape(b * c, t).float()
        n = series.shape[0]
        with torch.autocast(device_type=x.device.type, enabled=False):
            inputs, _, _ = self.model.prepare_input(
                past_target=series,
                past_observed_values=torch.ones_like(series),
                past_time_feat=series.new_zeros(n, t, N_TIME_FEATURES),
                future_time_feat=series.new_zeros(n, 1, N_TIME_FEATURES),
                future_target=None,
            )
        h = self.model.transformer.wte(inputs)
        for block in self.model.transformer.h:
            h = block(h, False)
        h = self.model.transformer.ln_f(h)
        return h.reshape(b, c, h.shape[1], self.d_model)


class LagLlamaClassifier(nn.Module):
    def __init__(self, n_classes, init="pretrained"):
        super().__init__()
        self.backbone = _LagLlamaBackbone(init)
        self.head = nn.Linear(self.backbone.d_model, n_classes)

    def forward(self, x):
        h = self.backbone.tokens(x).mean(dim=2).mean(dim=1)  # tokens, then channels
        return self.head(h)


class LagLlamaCTC(nn.Module):
    def __init__(self, init="pretrained"):
        super().__init__()
        self.backbone = _LagLlamaBackbone(init)
        self.ctc_head = ctc.make_ctc_head(ctc.N_CHANNELS * self.backbone.d_model)

    def forward(self, x):
        h = self.backbone.tokens(x)  # (B, C, P, d_model)
        return self.ctc_head(h.permute(0, 2, 1, 3).flatten(2))  # (B, P, vocab)


def build_classifier(model_size, n_channels, n_classes, init="pretrained"):
    return LagLlamaClassifier(n_classes, init)


def build_ctc(model_size, init="pretrained"):
    return LagLlamaCTC(init)
