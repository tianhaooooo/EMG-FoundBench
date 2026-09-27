"""emg2qwerty keystroke decoding with CTC, shared by every model family (Appendix C.9).

Each 32 x 3000 window is passed to the backbone through its native input interface, without an
additional convolutional or recurrent front-end. The backbone output is arranged as a temporal
sequence, and the channel-wise representations are concatenated at each timestep before a
single linear CTC head (32 * d_model -> 71: 70 characters + blank). The CTC input length is the
temporal resolution of the backbone output.

The whole model is fine-tuned with the full fine-tuning recipe of Appendix C.8 (AdamW, lr 6e-5,
up to 50 epochs, early stopping with patience 10 on validation CER, ties broken by validation
CTC loss); the best validation checkpoint is used for testing. Following CTC fine-tuning practice
in speech recognition (wav2vec 2.0; Deep Speech 2), during the first epoch the backbone is frozen
and only the CTC head is trained, the learning rate is warmed up linearly to 6e-5, and training
windows are ordered by target length; the complete model is then fine-tuned with shuffled
windows. Gradients are clipped to norm 1.0. Loss: nn.CTCLoss(blank=0, zero_infinity=True, mean
reduction). Decoding is greedy. CER is computed per user on the concatenation of that user's
decoded windows (temporal order) and averaged over users.

Model wrappers used here map (B, 32, 3000) to per-timestep logits (B, T, 71).
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from common import protocol

N_CHANNELS = 32
VOCAB = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-"
VOCAB_SIZE = len(VOCAB) + 1  # + blank
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
IDX2CHAR = {i + 1: c for i, c in enumerate(VOCAB)}
BLANK = 0
BLANK_BIAS_INIT = -2.0
HEAD_ONLY_EPOCHS = 1
GRAD_CLIP_NORM = 1.0


def make_ctc_head(in_dim):
    head = nn.Linear(in_dim, VOCAB_SIZE)
    with torch.no_grad():
        head.bias[BLANK] = BLANK_BIAS_INIT
    return head


def text_to_indices(text):
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def greedy_decode(logits):
    ids = logits.argmax(dim=-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != BLANK and i != prev:
            out.append(IDX2CHAR.get(i, ""))
        prev = i
    return "".join(out)


def _levenshtein(a, b):
    """Edit distance between a and b (bit-parallel algorithm of Myers / Hyyroe)."""
    m = len(a)
    if m == 0:
        return len(b)
    peq = {}
    for i, c in enumerate(a):
        peq[c] = peq.get(c, 0) | (1 << i)
    full, high = (1 << m) - 1, 1 << (m - 1)
    pv, mv, score = full, 0, m
    for c in b:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & full) ^ pv) | eq
        ph = mv | (~(xh | pv) & full)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        pv = mh | (~(xv | ph) & full)
        mv = ph & xv
    return score


def cer(ref, hyp):
    """Levenshtein distance between ref and hyp divided by len(ref)."""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(ref, hyp) / len(ref)


class EMGQwertyDataset(Dataset):
    """Memory-mapped per-user windows: <user>/{split}_emg.npy (N, 32, 3000), {split}_labels.npy (N,)."""

    def __init__(self, user_dirs, split):
        self.arrays, self.texts, self.index, self.user_ids = [], [], [], []
        for user_idx, user_dir in enumerate(user_dirs):
            arr = np.load(user_dir / f"{split}_emg.npy", mmap_mode="r")
            labels = np.load(user_dir / f"{split}_labels.npy", allow_pickle=True)
            self.arrays.append(arr)
            self.texts.extend(labels.tolist())
            self.index.extend((user_idx, i) for i in range(len(arr)))
            self.user_ids.extend([user_idx] * len(arr))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        a, i = self.index[idx]
        return np.array(self.arrays[a][i], dtype=np.float32), self.texts[idx]


def collate(batch):
    emgs, texts = zip(*batch)
    return torch.from_numpy(np.stack(emgs)), list(texts)


def ctc_loss(logits, texts, loss_fn):
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
    targets = [text_to_indices(t) for t in texts]
    flat = torch.tensor([c for t in targets for c in t], dtype=torch.long, device=logits.device)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long, device=logits.device)
    input_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device)
    return loss_fn(log_probs, flat, input_lengths, target_lengths)


@torch.no_grad()
def evaluate(model, loader, user_ids, loss_fn):
    """Per-user CER on the concatenated decoded windows, mean CTC loss, and a few decoded samples."""
    model.eval()
    refs, hyps, losses = [], [], []
    for x, texts in loader:
        with protocol.autocast():
            logits = model(x.to(protocol.DEVICE))
        losses.append(ctc_loss(logits, texts, loss_fn).item())
        logits = logits.float().cpu()
        refs.extend("".join(c for c in t if c in CHAR2IDX) for t in texts)
        hyps.extend(greedy_decode(logits[b]) for b in range(len(texts)))
    user_ref, user_hyp = defaultdict(str), defaultdict(str)
    for uid, r, h in zip(user_ids, refs, hyps):
        user_ref[uid] += r
        user_hyp[uid] += h
    score = float(np.mean([cer(user_ref[u], user_hyp[u]) for u in user_ref]))
    return score, float(np.mean(losses)), list(zip(refs[:3], hyps[:3]))


def run(model, model_name, data_root, n_users, batch_size, out_dir, num_workers=4, log_every=100,
        init_ckpt=None, start_epoch=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    user_dirs = sorted(Path(data_root).glob("user_*"))[:n_users]
    ds = {s: EMGQwertyDataset(user_dirs, s) for s in ("train", "val", "test")}
    print(f"{len(user_dirs)} users | " + " ".join(f"{s}={len(d)}" for s, d in ds.items()))
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"), drop_last=(s == "train"),
                             collate_fn=collate, num_workers=num_workers, pin_memory=True)
               for s, d in ds.items()}
    by_length = sorted(range(len(ds["train"])), key=lambda i: len(text_to_indices(ds["train"].texts[i])))
    sorted_train = DataLoader(ds["train"], batch_size=batch_size, sampler=by_length, drop_last=True,
                              collate_fn=collate, num_workers=num_workers, pin_memory=True)

    model = model.to(protocol.DEVICE)
    if init_ckpt:
        model.load_state_dict(torch.load(init_ckpt, map_location=protocol.DEVICE, weights_only=True))
    lr = protocol.LR["fft"]
    optimizer = protocol.make_optimizer(model.parameters(), lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} CTC | lr={lr} batch_size={batch_size} params={n_params:,}")

    loss_fn = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    ckpt = out_dir / f"{model_name}_ctc_best.pt"
    best, bad_epochs, history = (float("inf"), float("inf")), 0, []
    if init_ckpt:
        best = evaluate(model, loaders["val"], ds["val"].user_ids, loss_fn)[:2]
        torch.save(model.state_dict(), ckpt)
        print(f"resumed from {init_ckpt} at epoch {start_epoch} | val CER {best[0]:.4f} | val loss {best[1]:.4f}",
              flush=True)
    head_params = list(model.ctc_head.parameters())
    for epoch in range(start_epoch, protocol.MAX_EPOCHS + 1):
        head_only = epoch <= HEAD_ONLY_EPOCHS
        for p in model.parameters():
            p.requires_grad = not head_only
        for p in head_params:
            p.requires_grad = True
        if head_only:  # frozen backbone kept in eval mode, as in linear probing
            model.eval()
            model.ctc_head.train()
        else:
            model.train()
        train_loader = sorted_train if head_only else loaders["train"]
        trainable = [p for p in model.parameters() if p.requires_grad]
        start, running = time.time(), []
        for i, (x, texts) in enumerate(train_loader):
            if epoch == 1:  # linear warm-up over the first epoch
                for g in optimizer.param_groups:
                    g["lr"] = lr * (i + 1) / len(train_loader)
            optimizer.zero_grad()
            with protocol.autocast():
                logits = model(x.to(protocol.DEVICE))
            loss = ctc_loss(logits, texts, loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            optimizer.step()
            running.append(loss.item())
            if (i + 1) % log_every == 0:
                print(f"  epoch {epoch} step {i + 1}/{len(train_loader)} loss={np.mean(running):.4f}"
                      f"{' (head only)' if head_only else ''}", flush=True)
                running = []
        val_cer, val_loss, samples = evaluate(model, loaders["val"], ds["val"].user_ids, loss_fn)
        history.append({"epoch": epoch, "val_cer": val_cer, "val_loss": val_loss})
        print(f"epoch {epoch}/{protocol.MAX_EPOCHS} done in {time.time() - start:.0f}s | "
              f"val CER {val_cer:.4f} | val loss {val_loss:.4f}", flush=True)
        for ref, hyp in samples:
            print(f"    ref={ref!r}  hyp={hyp!r}", flush=True)
        if (val_cer, val_loss) < best:
            best, bad_epochs = (val_cer, val_loss), 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= protocol.PATIENCE:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=protocol.DEVICE, weights_only=True))
    test_cer, test_loss, _ = evaluate(model, loaders["test"], ds["test"].user_ids, loss_fn)
    result = {"model": model_name, "n_users": len(user_dirs), "lr": lr, "batch_size": batch_size,
              "best_val_cer": best[0], "test_cer": test_cer, "history": history}
    print(f"test CER {test_cer:.4f} (best val CER {best[0]:.4f})")
    (out_dir / f"{model_name}_emg2qwerty.json").write_text(json.dumps(result, indent=2))
    return result
