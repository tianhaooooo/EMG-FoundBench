"""BiLSTM conventional supervised reference (Appendix C.10), randomly initialized.

    (B, C, T) -> channel-wise depthwise Conv1d (kernel = stride = T // 150; 150 steps for T = 3000)
      -> linear projection of the C channels to 128 at each time step
      -> 2 stacked bidirectional LSTM layers (hidden 256 per direction, dropout 0.4)
      -> final hidden states of the forward and backward directions of the top layer (512)
      -> linear classification layer

The model has no pretrained checkpoint, so it is evaluated only under the scratch, held-out-user,
LOSO and five-shot protocols (not linear probing or pretrained fine-tuning).
"""
import torch
import torch.nn as nn

NAME = "BiLSTM"
SIZES = ["base"]
PRETRAINED = False
TARGET_STEPS = 150
PROJ_DIM = 128
HIDDEN = 256
DROPOUT = 0.4
WINDOW_LEN = 3000
MAX_BATCH_SIZE = 32


def batch_size(model_size, dataset_name, n_channels):
    return MAX_BATCH_SIZE


class BiLSTMClassifier(nn.Module):
    def __init__(self, n_channels, n_classes, seq_len=WINDOW_LEN):
        super().__init__()
        stride = max(1, seq_len // TARGET_STEPS)
        self.frontend = nn.Conv1d(n_channels, n_channels, kernel_size=stride, stride=stride, groups=n_channels)
        self.input_proj = nn.Linear(n_channels, PROJ_DIM)
        self.lstm = nn.LSTM(PROJ_DIM, HIDDEN, num_layers=2, batch_first=True, bidirectional=True, dropout=DROPOUT)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(2 * HIDDEN, n_classes)

    def forward(self, x):
        z = self.input_proj(self.frontend(x).transpose(1, 2))  # (B, ~150, 128)
        _, (h_n, _) = self.lstm(z)  # h_n: (num_layers * 2, B, HIDDEN)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # top layer, forward and backward
        return self.head(self.dropout(h))


def build_classifier(model_size, n_channels, n_classes, init="random"):
    return BiLSTMClassifier(n_channels, n_classes)


def build_ctc(model_size, init="random"):
    raise NotImplementedError("the BiLSTM reference is not evaluated on emg2qwerty")
