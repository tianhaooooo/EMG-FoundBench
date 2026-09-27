"""Training / evaluation protocol shared by every model family (Appendix C.8).

  * AdamW, betas=(0.9, 0.95), weight decay 0.01, eps=1e-7, constant learning rate
  * learning rate 1e-5 (linear probing) or 6e-5 (full fine-tuning and scratch)
  * up to 50 epochs, validation-based early stopping on macro-F1 (patience 10),
    best-validation checkpoint used for test evaluation
  * bfloat16 mixed precision
  * batch size 32 when memory permits, otherwise reduced to 4-16
  * five-shot: head-only adaptation, lr 5e-4, 30 epochs, final epoch, seeds 42-46

Model wrappers (see <model>/model.py) are nn.Modules whose forward(x) maps a (B, C, T) window
batch to class logits and which expose the linear classification layer as `.head`.
"""
import contextlib
import gc
import random

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-7
WEIGHT_DECAY = 0.01
LR = {"lp": 1e-5, "fft": 6e-5, "scratch": 6e-5}
MAX_EPOCHS = 50
PATIENCE = 10

FEWSHOT_SEEDS = [42, 43, 44, 45, 46]
FEWSHOT_N_PER_CLASS = 5
FEWSHOT_EPOCHS = 30
FEWSHOT_LR = 5e-4


def batch_size(n_channels, tiers):
    """Batch size for a window with `n_channels` channels from ((max_channels, batch_size), ...)."""
    for max_channels, bs in tiers:
        if max_channels is None or n_channels <= max_channels:
            return bs
    raise ValueError("tiers must end with (None, batch_size)")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(params, lr):
    return torch.optim.AdamW(params, lr=lr, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)


def autocast():
    if DEVICE.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def classification_metrics(labels, preds):
    """Accuracy and macro-averaged F1 / precision / recall over the classes present in labels or preds."""
    classes = np.union1d(labels, preds)
    tp = np.array([np.sum((preds == c) & (labels == c)) for c in classes], dtype=float)
    pred_n = np.array([np.sum(preds == c) for c in classes], dtype=float)
    true_n = np.array([np.sum(labels == c) for c in classes], dtype=float)
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    precision, recall = div(tp, pred_n), div(tp, true_n)
    f1 = div(2 * tp, pred_n + true_n)
    return {"accuracy": float(np.mean(labels == preds)), "f1": float(f1.mean()),
            "precision": float(precision.mean()), "recall": float(recall.mean())}


def configure_trainable(model, head_only):
    """head_only=True: only the classification head is updated (linear probing, five-shot)."""
    head_params = {id(p) for p in model.head.parameters()}
    for p in model.parameters():
        p.requires_grad = (not head_only) or id(p) in head_params


def set_train_mode(model, head_only):
    """A frozen backbone stays in eval mode, so dropout and normalization statistics are unchanged."""
    if head_only:
        model.eval()
        model.head.train()
    else:
        model.train()


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        with autocast():
            logits = model(x.to(DEVICE))
        preds.append(logits.float().argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return classification_metrics(np.concatenate(labels), np.concatenate(preds))


def _train_one_epoch(model, loader, optimizer, head_only):
    criterion = nn.CrossEntropyLoss()
    set_train_mode(model, head_only)
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
        loss = criterion(logits.float(), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def train_with_early_stopping(model, train_loader, val_loader, mode, ckpt_path,
                              max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Train in mode 'lp' | 'fft' | 'scratch' with early stopping on validation macro-F1,
    then restore the best-validation checkpoint. Returns the best validation macro-F1."""
    head_only = mode == "lp"
    configure_trainable(model, head_only)
    optimizer = make_optimizer([p for p in model.parameters() if p.requires_grad], LR[mode])
    best_f1, bad_epochs = -1.0, 0
    for epoch in range(1, max_epochs + 1):
        loss = _train_one_epoch(model, train_loader, optimizer, head_only)
        val = evaluate(model, val_loader)
        print(f"  epoch {epoch:3d}/{max_epochs}  loss={loss:.4f}  "
              f"val_acc={val['accuracy']:.4f}  val_f1={val['f1']:.4f}", flush=True)
        if val["f1"] > best_f1:
            best_f1, bad_epochs = val["f1"], 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stopping at epoch {epoch}")
                break
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return best_f1


# ----------------------------------------------------------------------------
# Five-shot adaptation
# ----------------------------------------------------------------------------
def sample_support_indices(train_y, n_classes, n_per_class, seed):
    """Deterministic given (train_y, n_classes, n_per_class, seed); identical across model families."""
    rng = np.random.RandomState(seed)
    support = []
    for cls in range(n_classes):
        cls_idx = np.where(train_y == cls)[0]
        if len(cls_idx) == 0:
            continue
        support.extend(rng.choice(cls_idx, min(n_per_class, len(cls_idx)), replace=False).tolist())
    return np.array(sorted(support))


def adapt_head(model, support_loader, seed, epochs=FEWSHOT_EPOCHS, lr=FEWSHOT_LR):
    """Head-only adaptation for a fixed budget; no early stopping, the final epoch is kept."""
    torch.manual_seed(seed)
    configure_trainable(model, head_only=True)
    optimizer = make_optimizer([p for p in model.parameters() if p.requires_grad], lr)
    for _ in range(epochs):
        _train_one_epoch(model, support_loader, optimizer, head_only=True)
