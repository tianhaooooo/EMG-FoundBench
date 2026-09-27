"""UniTS wrappers (official x128 pretrained checkpoint: d_model 128, 3 blocks, 8 heads, patch 16).

Classification (Appendix C.9): the EMG window is given to UniTS through its native variate
dimension (input layout (B, T, C)). UniTS tokenizes each variate (instance normalization, right
zero-padding to a multiple of 16, non-overlapping patches of 16 samples: 188 patches for 3000
samples), adds its native prompt tokens and classification token, and runs its sequence- and
variate-attention blocks. The classification head's pooled representation is taken before the
category-token matching step (one vector per channel), averaged across channels and mapped to
class logits by a linear layer. The forecasting head, mask tokens and category tokens are unused.

The EMG task has no pretrained prompt/classification tokens; they are randomly initialized for
both pretrained and scratch models. All other backbone tensors are loaded from the checkpoint.

CTC: the same input path; the per-channel patch tokens after the backbone (prompt and
classification tokens removed) are concatenated across the 32 channels at each patch position
and fed to the linear CTC head.
"""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from common import ctc
from units.units_upstream import Model

NAME = "UniTS"
SIZES = ["base"]  # official x128 checkpoint
CKPT_URL = "https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x128_pretrain_checkpoint.pth"
CKPT_SHA256 = "35d17336c8857d2bacefe403fe7f21ec33ffae9f67d497d569e6af0cb38b5474"
ARCH = dict(d_model=128, e_layers=3, n_heads=8, patch_len=16, stride=16, prompt_num=10, dropout=0.1)
TASK = "emg"
NEW_TASK_TOKENS = ("prompt_tokens.", "cls_tokens.")

MAX_BATCH_SIZE = 32
CTC_BATCH_SIZE = {"base": MAX_BATCH_SIZE}


def batch_size(model_size, dataset_name, n_channels):
    return MAX_BATCH_SIZE


def _checkpoint_path():
    path = Path(os.environ.get("UNITS_CKPT", Path.home() / ".cache" / "units" / Path(CKPT_URL).name))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(CKPT_URL, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CKPT_SHA256:
        raise ValueError(f"checksum mismatch for {path}")
    return path


def _build_backbone(n_channels, n_classes, init):
    config = [(TASK, dict(dataset=TASK, task_name="classification", enc_in=n_channels, num_class=n_classes))]
    model = Model(SimpleNamespace(**ARCH), config)
    model.forecast_head = nn.Identity()
    model.prompt2forecat = nn.Identity()
    model.mask_tokens = nn.ParameterDict()
    model.category_tokens = nn.ParameterDict()
    if init == "pretrained":
        # The release stores argparse / NumPy metadata next to the weights, so it needs the full
        # unpickler; the file is verified against the official checksum first.
        raw = torch.load(_checkpoint_path(), map_location="cpu", weights_only=False)["student"]
        raw = {k[len("module."):] if k.startswith("module.") else k: v for k, v in raw.items()}
        current = model.state_dict()
        required = {k for k in current if not k.startswith(NEW_TASK_TOKENS)}
        missing = sorted(required - raw.keys())
        mismatched = sorted(k for k in required & raw.keys() if raw[k].shape != current[k].shape)
        if missing or mismatched:
            raise RuntimeError(f"incompatible checkpoint: missing={missing}, shape mismatch={mismatched}")
        model.load_state_dict({k: raw[k] for k in required}, strict=False)
    return model


def _encode(model, x):
    """(B, C, T) -> backbone tokens (B, C, prompts + P + 1, d) and the number of patches P."""
    tokens, _, _, n_vars, _ = model.tokenize(x.transpose(1, 2))
    n_patches = tokens.shape[-2]
    tokens = model.prepare_prompt(tokens, n_vars, model.prompt_tokens[TASK], model.cls_tokens[TASK], 1,
                                  task_name="classification")
    return model.backbone(tokens, model.prompt_num, n_patches), n_patches


class UniTSClassifier(nn.Module):
    def __init__(self, n_channels, n_classes, init="pretrained"):
        super().__init__()
        self.backbone = _build_backbone(n_channels, n_classes, init)
        self.head = nn.Linear(ARCH["d_model"], n_classes)

    def forward(self, x):
        tokens, _ = _encode(self.backbone, x)
        pooled = self.backbone.cls_head(tokens, return_feature=True)  # (B, C, 1, d), before category matching
        return self.head(pooled.mean(dim=1).squeeze(1))


class UniTSCTC(nn.Module):
    def __init__(self, init="pretrained"):
        super().__init__()
        self.backbone = _build_backbone(ctc.N_CHANNELS, 1, init)
        self.ctc_head = ctc.make_ctc_head(ctc.N_CHANNELS * ARCH["d_model"])

    def forward(self, x):
        tokens, n_patches = _encode(self.backbone, x)
        start = self.backbone.prompt_num
        h = tokens[:, :, start:start + n_patches]  # (B, C, P, d)
        return self.ctc_head(h.permute(0, 2, 1, 3).flatten(2))  # (B, P, vocab)


def build_classifier(model_size, n_channels, n_classes, init="pretrained"):
    return UniTSClassifier(n_channels, n_classes, init)


def build_ctc(model_size, init="pretrained"):
    return UniTSCTC(init)
