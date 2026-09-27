"""TimesFM-2.5-200M wrappers.

TimesFM is a univariate decoder-only forecaster. Classification (Appendix C.9) reuses its input
tokenizer and 20-layer transformer and drops the forecasting heads:

    (B, C, T) --channel-independent--> (B*C, T)
      -> front-pad to a multiple of the patch length 32 (padded points masked, as in TimesFM's
         own forecast path)
      -> TimesFM running per-patch normalization (same statistics as TimesFM's decode)
      -> tokenizer -> 20 transformer layers -> (B*C, P, 1280)
      -> mean over valid patches, then over channels -> Linear(1280, n_classes)

CTC: the same input path; patch representations (94 patches for 3000 samples) are concatenated
across the 32 channels at each patch position and fed to the linear CTC head.

Requires the TimesFM 2.5 torch implementation (https://github.com/google-research/timesfm);
pretrained weights are google/timesfm-2.5-200m-pytorch.
"""
import torch
import torch.nn as nn
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
from timesfm.torch.normalization import RMSNorm
from timesfm.torch.util import revin

from common import ctc, protocol

NAME = "TimesFM"
SIZES = ["base"]  # TimesFM-2.5-200M; the only released size
HF_REPO = "google/timesfm-2.5-200m-pytorch"

# Batch size (Appendix C.8): 32 when memory permits, otherwise 4-16, set by the number of channels.
# The same value is used for LP, FFT and scratch.
BATCH_SIZE_TIERS = ((12, 32), (24, 16), (48, 8), (None, 4))
CTC_BATCH_SIZE = {"base": protocol.batch_size(ctc.N_CHANNELS, BATCH_SIZE_TIERS)}


def batch_size(model_size, dataset_name, n_channels):
    return protocol.batch_size(n_channels, BATCH_SIZE_TIERS)


def _init_scratch(module):
    """Random initialization. The torch port stores RMSNorm `scale` as a direct multiplier that is
    zero at construction, which would make every transformer block output (and receive) exactly
    zero; it is therefore set to one. Linear layers use Xavier-uniform weights and zero biases.
    PerDimScale keeps its zero initialization (softplus(0) * 1.4427 = 1, standard scaling)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, RMSNorm):
            nn.init.ones_(m.scale)


class _TimesFMBackbone(nn.Module):
    def __init__(self, init):
        super().__init__()
        core = TimesFM_2p5_200M_torch_module()
        if init == "pretrained":
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            core.load_state_dict(load_file(hf_hub_download(HF_REPO, "model.safetensors")), strict=True)
        else:
            _init_scratch(core)
        self.patch_len = core.p
        self.tokenizer = core.tokenizer
        self.stacked_xf = core.stacked_xf
        self.d_model = core.md

    def _normalize(self, x):
        """(N, T) -> normalized patches (N, P, patch_len) and padding mask (N, P, patch_len)."""
        n, t = x.shape
        pad = (-t) % self.patch_len
        mask = torch.zeros_like(x, dtype=torch.bool)
        if pad:
            x = torch.cat([x.new_zeros(n, pad), x], dim=1)
            mask = torch.cat([torch.ones(n, pad, dtype=torch.bool, device=x.device), mask], dim=1)
        px = x.view(n, -1, self.patch_len)
        pm = mask.view(n, -1, self.patch_len)
        legit = (~pm).to(x.dtype)
        cnt = torch.cumsum(legit.sum(-1), dim=1)
        s1 = torch.cumsum((px * legit).sum(-1), dim=1)
        s2 = torch.cumsum((px.square() * legit).sum(-1), dim=1)
        cnt_safe = torch.where(cnt == 0, torch.ones_like(cnt), cnt)
        mu = torch.where(cnt == 0, torch.zeros_like(s1), s1 / cnt_safe)
        var = torch.where(cnt == 0, torch.zeros_like(s2), s2 / cnt_safe - mu.square())
        sigma = torch.sqrt(torch.clamp(var, min=0.0))
        normed = revin(px, mu, sigma, reverse=False)
        return torch.where(pm, torch.zeros_like(normed), normed), pm

    def patches(self, x):
        """(B, C, T) -> patch representations (B, C, P, d_model) and valid-patch mask (B, C, P)."""
        b, c, t = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            normed, pm = self._normalize(x.reshape(b * c, t).float())
        h = self.tokenizer(torch.cat([normed, pm.to(normed.dtype)], dim=-1))
        patch_mask = pm[..., -1]
        for layer in self.stacked_xf:
            h, _ = layer(h, patch_mask, None)
        p = h.shape[1]
        return h.reshape(b, c, p, self.d_model), (~patch_mask).reshape(b, c, p)


class TimesFMClassifier(nn.Module):
    def __init__(self, n_classes, init="pretrained"):
        super().__init__()
        self.backbone = _TimesFMBackbone(init)
        self.head = nn.Linear(self.backbone.d_model, n_classes)
        # Zero-initialized head so that training starts at loss = ln(n_classes) for both
        # initializations (the pooled features of a randomly initialized backbone are large).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h, valid = self.backbone.patches(x)
        valid = valid.unsqueeze(-1).to(h.dtype)
        pooled = (h * valid).sum(2) / valid.sum(2).clamp_min(1.0)  # mean over valid patches
        return self.head(pooled.mean(dim=1).float())  # mean over channels


class TimesFMCTC(nn.Module):
    def __init__(self, init="pretrained"):
        super().__init__()
        self.backbone = _TimesFMBackbone(init)
        self.ctc_head = ctc.make_ctc_head(ctc.N_CHANNELS * self.backbone.d_model)

    def forward(self, x):
        h, _ = self.backbone.patches(x)  # (B, C, P, d_model)
        return self.ctc_head(h.permute(0, 2, 1, 3).flatten(2))  # (B, P, vocab)


def build_classifier(model_size, n_channels, n_classes, init="pretrained"):
    return TimesFMClassifier(n_classes, init)


def build_ctc(model_size, init="pretrained"):
    return TimesFMCTC(init)
