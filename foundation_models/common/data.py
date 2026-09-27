"""Data loading shared by every model family.

Benchmark layout (output of data_processing/build_benchmark.py):
    <data_dir>/<subject>/{train,val,test}_emg.npy   (N, C, 3000) float32
    <data_dir>/<subject>/{train,val,test}_labels.npy (N,)
  or, for datasets with a predefined partition,
    <data_dir>/pool_A/<subject>/...  and  <data_dir>/pool_B/<subject>/...

LOSO layout (output of data_processing/build_loso.py):
    <data_dir>/<subject>/{emg,labels}.npy, optionally <data_dir>/loso_cohort.json
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

POOL_A_FRACTION = 0.9
LOSO_COHORT_SIZE = 10


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


class WindowDataset(Dataset):
    """Memory-mapped windows from several per-subject arrays of shape (N, C, T).

    Windows with fewer than `n_channels` channels are zero-padded along the channel axis.
    Raw labels are mapped to contiguous indices with `label_map`.
    """

    def __init__(self, arrays, labels, n_channels, label_map):
        self.arrays = arrays
        self.n_channels = n_channels
        self.index = [(a, i) for a, arr in enumerate(arrays) for i in range(len(arr))]
        flat = np.concatenate(labels) if labels else np.array([], dtype=np.int64)
        self.labels = np.array([label_map[v] for v in flat], dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        a, i = self.index[idx]
        x = np.array(self.arrays[a][i], dtype=np.float32)  # copy out of the read-only memory map
        if x.shape[0] < self.n_channels:
            x = np.pad(x, ((0, self.n_channels - x.shape[0]), (0, 0)))
        return torch.from_numpy(x), torch.tensor(self.labels[idx])


# ----------------------------------------------------------------------------
# Pool A / Pool B protocol
# ----------------------------------------------------------------------------
def split_pools(data_dir):
    """Return (pool_a_dirs, pool_b_dirs).

    Uses explicit pool_A/ and pool_B/ sub-folders when present; otherwise sorts the subject
    folders and assigns the first 90% to Pool A and the remaining subjects to Pool B.
    """
    data_dir = Path(data_dir)
    if (data_dir / "pool_A").is_dir() and (data_dir / "pool_B").is_dir():
        pool_a = sorted((d for d in (data_dir / "pool_A").iterdir() if d.is_dir()), key=lambda d: natural_key(d.name))
        pool_b = sorted((d for d in (data_dir / "pool_B").iterdir() if d.is_dir()), key=lambda d: natural_key(d.name))
        return pool_a, pool_b
    subjects = sorted((d for d in data_dir.iterdir() if d.is_dir()), key=lambda d: natural_key(d.name))
    if not subjects:
        raise ValueError(f"No subject folders found in {data_dir}")
    n_a = max(1, int(len(subjects) * POOL_A_FRACTION))
    return subjects[:n_a], subjects[n_a:]


def _load_split(dirs, split):
    arrays, labels = [], []
    for d in dirs:
        xp, yp = d / f"{split}_emg.npy", d / f"{split}_labels.npy"
        if xp.exists() and yp.exists():
            arrays.append(np.load(xp, mmap_mode="r"))
            labels.append(np.load(yp))
    return arrays, labels


def load_pool_data(data_dir):
    """Load Pool A train/val/test and Pool B train/test with one global label mapping."""
    pool_a, pool_b = split_pools(data_dir)
    raw = {
        "a_train": _load_split(pool_a, "train"),
        "a_val": _load_split(pool_a, "val"),
        "a_test": _load_split(pool_a, "test"),
        "b_train": _load_split(pool_b, "train"),
        "b_test": _load_split(pool_b, "test"),
    }
    if not raw["a_train"][0]:
        raise ValueError(f"No Pool A training data found in {data_dir}")
    all_arrays = [arr for arrays, _ in raw.values() for arr in arrays]
    n_channels = max(arr.shape[1] for arr in all_arrays)
    classes = np.unique(np.concatenate([np.concatenate(lbls) for _, lbls in raw.values() if lbls]))
    label_map = {c: i for i, c in enumerate(classes)}
    splits = {k: (WindowDataset(a, l, n_channels, label_map) if a else None) for k, (a, l) in raw.items()}
    return {
        "splits": splits,
        "n_channels": n_channels,
        "seq_len": all_arrays[0].shape[2],
        "n_classes": len(classes),
        "n_pool_a": len(pool_a),
        "n_pool_b": len(pool_b),
    }


# ----------------------------------------------------------------------------
# Cohort-restricted LOSO protocol
# ----------------------------------------------------------------------------
def loso_cohort(data_dir):
    """Subjects of the LOSO evaluation cohort: all subjects if at most 10, otherwise the first 10
    in canonical subject-ID order (loso_cohort.json written by build_loso.py takes precedence)."""
    data_dir = Path(data_dir)
    cohort_file = data_dir / "loso_cohort.json"
    if cohort_file.exists():
        return json.loads(cohort_file.read_text())["cohort"]
    subjects = sorted((d.name for d in data_dir.iterdir() if (d / "emg.npy").exists()), key=natural_key)
    return subjects[:LOSO_COHORT_SIZE]


def load_loso_data(data_dir, train_subjects, val_subject, test_subject):
    """The complete set of annotated classes is kept (no gesture subset), as in the Pool A/B protocol."""
    root = Path(data_dir)

    def load(subject_list):
        arrays = [np.load(root / s / "emg.npy", mmap_mode="r") for s in subject_list]
        labels = [np.load(root / s / "labels.npy") for s in subject_list]
        return arrays, labels

    parts = {"train": load(train_subjects), "val": load([val_subject]), "test": load([test_subject])}
    classes = np.unique(np.concatenate([np.concatenate(l) for _, l in parts.values()])).tolist()
    label_map = {c: i for i, c in enumerate(classes)}
    n_channels = max(arr.shape[1] for arrays, _ in parts.values() for arr in arrays)
    datasets = {k: WindowDataset(a, l, n_channels, label_map) for k, (a, l) in parts.items()}
    return datasets, n_channels, len(classes)
