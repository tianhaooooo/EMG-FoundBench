"""PatchTST wrappers (Hugging Face transformers implementation).

Pretrained checkpoint: ibm/patchtst-etth1-forecasting (patch length 12, stride 12, d_model 128,
3 layers, 16 heads, BatchNorm, per-channel standard scaling, [CLS] token, sin-cos positions).

Classification (Appendix C.9): the input is transposed to the time-major layout expected by
PatchTST; each channel is patchified (250 patches for 3000 samples) and encoded as an independent
variable stream by the shared transformer. Patch representations ([CLS] excluded) are averaged
over the patch and channel dimensions and mapped to class logits by a linear layer.

The model is built for a 3000-sample context. All pretrained tensors are loaded except the
sin-cos position table, which is not learned (the released table equals the deterministic table
generated for its 512-sample context) and is therefore generated natively for 250 patches.

CTC: the same input path; patch representations are concatenated across the 32 channels at each
patch position and fed to the linear CTC head.
"""
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel

from common import ctc

NAME = "PatchTST"
SIZES = ["base"]  # single released checkpoint
HF_REPO = "ibm/patchtst-etth1-forecasting"
WINDOW_LEN = 3000
POSITION_TABLE = "encoder.positional_encoder.position_enc"

MAX_BATCH_SIZE = 32
CTC_BATCH_SIZE = {"base": MAX_BATCH_SIZE}


def batch_size(model_size, dataset_name, n_channels):
    return MAX_BATCH_SIZE


class _PatchTSTBackbone(nn.Module):
    def __init__(self, n_channels, init):
        super().__init__()
        config = PatchTSTConfig.from_pretrained(HF_REPO)
        config.context_length = WINDOW_LEN
        config.num_input_channels = n_channels
        self.model = PatchTSTModel(config)  # random initialization
        if init == "pretrained":
            state = PatchTSTModel.from_pretrained(HF_REPO).state_dict()
            state = {k: v for k, v in state.items() if k != POSITION_TABLE}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing != [POSITION_TABLE] or unexpected:
                raise RuntimeError(f"unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.use_cls_token = config.use_cls_token
        self.d_model = config.d_model

    def patches(self, x):
        """(B, C, T) -> patch representations (B, C, P, d_model), [CLS] removed."""
        h = self.model(past_values=x.transpose(1, 2)).last_hidden_state
        return h[:, :, 1:] if self.use_cls_token else h


class PatchTSTClassifier(nn.Module):
    def __init__(self, n_channels, n_classes, init="pretrained"):
        super().__init__()
        self.backbone = _PatchTSTBackbone(n_channels, init)
        self.head = nn.Linear(self.backbone.d_model, n_classes)

    def forward(self, x):
        return self.head(self.backbone.patches(x).mean(dim=(1, 2)))


class PatchTSTCTC(nn.Module):
    def __init__(self, init="pretrained"):
        super().__init__()
        self.backbone = _PatchTSTBackbone(ctc.N_CHANNELS, init)
        self.ctc_head = ctc.make_ctc_head(ctc.N_CHANNELS * self.backbone.d_model)

    def forward(self, x):
        h = self.backbone.patches(x)  # (B, C, P, d_model)
        return self.ctc_head(h.permute(0, 2, 1, 3).flatten(2))  # (B, P, vocab)


def build_classifier(model_size, n_channels, n_classes, init="pretrained"):
    return PatchTSTClassifier(n_channels, n_classes, init)


def build_ctc(model_size, init="pretrained"):
    return PatchTSTCTC(init)
