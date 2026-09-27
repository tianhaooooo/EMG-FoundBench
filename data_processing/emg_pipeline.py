from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.signal as sig

TARGET_FS = 1000
WINDOW_SAMPLES = 3 * TARGET_FS
BANDPASS_HZ = (20.0, 450.0)
FILTER_ORDER = 4
SUB_1KHZ_CUTOFF_FRACTION = 0.9
NOTCH_HZ = (50.0, 60.0)
NOTCH_Q = 30.0
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
POOL_A_FRACTION = 0.9
LOSO_COHORT_SIZE = 10
NO_TRIAL = -1
EXCLUDED = -1


@dataclass
class Recording:
    emg: np.ndarray
    fs: float
    labels: object
    trials: object = None


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def _bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    sos = sig.butter(FILTER_ORDER, [low, high], btype="bandpass", fs=fs, output="sos")
    return sig.sosfiltfilt(sos, x, axis=0)


def _resample(x: np.ndarray, fs: float) -> np.ndarray:
    ratio = Fraction(TARGET_FS, int(round(fs)))
    if ratio == 1:
        return x
    return sig.resample_poly(x, ratio.numerator, ratio.denominator, axis=0)


def condition(emg: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(emg, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("recording contains non-finite samples; the loader must handle them")
    if fs >= TARGET_FS:
        x = _bandpass(_resample(x, fs), TARGET_FS, *BANDPASS_HZ)
    else:
        high = min(BANDPASS_HZ[1], SUB_1KHZ_CUTOFF_FRACTION * fs / 2)
        x = _resample(_bandpass(x, fs, BANDPASS_HZ[0], high), fs)
    for f0 in NOTCH_HZ:
        if f0 < fs / 2:
            b, a = sig.iirnotch(f0, NOTCH_Q, fs=TARGET_FS)
            x = sig.filtfilt(b, a, x, axis=0)
    return x.astype(np.float32)


def resample_index(n_native: int, fs: float, n_out: int) -> np.ndarray:
    return np.minimum((np.arange(n_out) * (fs / TARGET_FS)).astype(np.int64), n_native - 1)


def zscore(window_ct: np.ndarray) -> np.ndarray:
    m = window_ct.mean(axis=1, keepdims=True)
    s = window_ct.std(axis=1, keepdims=True)
    s[s < 1e-8] = 1.0
    return ((window_ct - m) / s).astype(np.float32)


def _plain(v):
    if isinstance(v, bytes):
        return v.decode()
    return v.item() if isinstance(v, np.generic) else v


def _per_sample(values, n_native: int, idx: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if len(values) != n_native:
        raise ValueError(f"per-sample array has {len(values)} entries for {n_native} samples")
    return values[idx]


def _labels_on_grid(raw, n_native: int, idx: np.ndarray, class_index: dict) -> np.ndarray:
    if np.ndim(raw) == 0:
        return np.full(len(idx), class_index.get(_plain(raw), EXCLUDED), dtype=np.int64)
    uniq, inv = np.unique(_per_sample(raw, n_native, idx), return_inverse=True)
    lut = np.array([class_index.get(_plain(u), EXCLUDED) for u in uniq], dtype=np.int64)
    return lut[inv.reshape(-1)]


def _trials_on_grid(trials, n_native: int, idx: np.ndarray):
    if trials is None:
        return None
    if np.ndim(trials) == 0:
        return np.full(len(idx), int(trials), dtype=np.int64)
    return _per_sample(trials, n_native, idx).astype(np.int64)


def split_counts(n_units: int) -> tuple[int, int, int]:
    if n_units < 3:
        return n_units, 0, 0
    n_train = min(int(round(n_units * SPLIT_FRACTIONS[0])), n_units - 2)
    n_val = (n_units - n_train) // 2
    return n_train, n_val, n_units - n_train - n_val


def _assign_trials(recs: list[dict]) -> None:
    trials = []
    for ri, r in enumerate(recs):
        if r["trials"] is None:
            continue
        trl, lab = r["trials"], r["labels"]
        for t in np.unique(trl[trl != NO_TRIAL]):
            mask = trl == t
            valid = lab[mask]
            valid = valid[valid != EXCLUDED]
            if valid.size == 0:
                continue
            vals, cnt = np.unique(valid, return_counts=True)
            trials.append(dict(rec=ri, trial=t, label=int(vals[cnt.argmax()]),
                               pure=bool(cnt.max() == valid.size), first=int(mask.argmax())))
    if not trials:
        return
    stratified = all(t["pure"] for t in trials)
    groups: dict[int, list] = {}
    for t in trials:
        groups.setdefault(t["label"] if stratified else 0, []).append(t)
    for group in groups.values():
        group.sort(key=lambda t: (t["rec"], t["first"]))
        n_tr, n_va, _ = split_counts(len(group))
        for i, t in enumerate(group):
            sp = 0 if i < n_tr else (1 if i < n_tr + n_va else 2)
            r = recs[t["rec"]]
            r["split"][r["trials"] == t["trial"]] = sp


def _assign_blocks(r: dict) -> None:
    n = len(r["labels"])
    b1 = int(round(n * SPLIT_FRACTIONS[0]))
    b2 = int(round(n * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1])))
    r["split"][:b1], r["split"][b1:b2], r["split"][b2:] = 0, 1, 2


def _majority(lab: np.ndarray):
    vals, cnt = np.unique(lab, return_counts=True)
    k = cnt.argmax()
    if vals[k] == EXCLUDED or 2 * cnt[k] <= len(lab):
        return None
    return int(vals[k])


def stretch_to_window(seg_tc: np.ndarray) -> np.ndarray:
    ratio = Fraction(WINDOW_SAMPLES, len(seg_tc))
    y = sig.resample_poly(seg_tc, ratio.numerator, ratio.denominator, axis=0)
    return y[:WINDOW_SAMPLES]


def reflect_pad_to_window(seg_tc: np.ndarray) -> np.ndarray:
    pad = WINDOW_SAMPLES - len(seg_tc)
    return np.pad(seg_tc, ((pad // 2, pad - pad // 2), (0, 0)), mode="reflect")


SHORT_TRIAL_HANDLERS = {"stretch": stretch_to_window, "reflect_pad": reflect_pad_to_window}


def _windows(r: dict, out: dict, short_trial_handling) -> None:
    split, lab, trl, x = r["split"], r["labels"], r["trials"], r["emg"]
    change = np.diff(split) != 0
    if trl is not None:
        change |= np.diff(trl) != 0
    bounds = np.flatnonzero(change) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(split)]):
        if split[s] < 0:
            continue
        xs, ys = out[SPLITS[split[s]]]
        if short_trial_handling and trl is not None and e - s < WINDOW_SAMPLES:
            y = _majority(lab[s:e])
            if y is not None:
                xs.append(zscore(SHORT_TRIAL_HANDLERS[short_trial_handling](x[s:e]).T))
                ys.append(y)
            continue
        for w0 in range(s, e - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
            y = _majority(lab[w0:w0 + WINDOW_SAMPLES])
            if y is not None:
                xs.append(zscore(x[w0:w0 + WINDOW_SAMPLES].T))
                ys.append(y)


def process_subject(recordings: list[Recording], classes: list,
                    short_trial_handling: str | None = None) -> dict:
    class_index = {_plain(c): i for i, c in enumerate(classes)}
    recs = []
    for rec in recordings:
        n_native = len(rec.emg)
        x = condition(rec.emg, rec.fs)
        rec.emg = None
        idx = resample_index(n_native, rec.fs, len(x))
        recs.append(dict(emg=x, labels=_labels_on_grid(rec.labels, n_native, idx, class_index),
                         trials=_trials_on_grid(rec.trials, n_native, idx),
                         split=np.full(len(x), -1, dtype=np.int64)))
    n_channels = {r["emg"].shape[1] for r in recs}
    if len(n_channels) > 1:
        raise ValueError(f"inconsistent channel counts within subject: {sorted(n_channels)}")

    _assign_trials(recs)
    for r in recs:
        if r["trials"] is None:
            _assign_blocks(r)

    out = {s: ([], []) for s in SPLITS}
    for r in recs:
        _windows(r, out, short_trial_handling)
        r["emg"] = None
    c = next(iter(n_channels)) if n_channels else 0
    return {s: (np.stack(xs) if xs else np.zeros((0, c, WINDOW_SAMPLES), np.float32),
                np.asarray(ys, dtype=np.int64)) for s, (xs, ys) in out.items()}


def save_subject(out_dir: Path, subject: str, result: dict) -> dict:
    d = Path(out_dir) / subject
    d.mkdir(parents=True, exist_ok=True)
    counts = {}
    for sp, (x, y) in result.items():
        np.save(d / f"{sp}_emg.npy", x)
        np.save(d / f"{sp}_labels.npy", y)
        counts[sp] = int(len(y))
    return counts


def assign_pools(subjects: list[str]) -> dict:
    subjects = sorted(subjects, key=natural_key)
    n_a = max(1, int(len(subjects) * POOL_A_FRACTION))
    return {"pool_A": subjects[:n_a], "pool_B": subjects[n_a:]}


def pipeline_config() -> dict:
    return dict(target_fs=TARGET_FS, window_samples=WINDOW_SAMPLES, window_overlap=0,
                resampling="polyphase", bandpass_hz=list(BANDPASS_HZ), filter_order=FILTER_ORDER,
                notch_hz=list(NOTCH_HZ), notch_q=NOTCH_Q,
                normalization="per-channel per-window z-score",
                split_fractions=list(SPLIT_FRACTIONS),
                pool_A_fraction=POOL_A_FRACTION)


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
