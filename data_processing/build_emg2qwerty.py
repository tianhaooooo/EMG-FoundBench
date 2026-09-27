#!/usr/bin/env python3
import argparse
import os
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emg_pipeline import (SPLIT_FRACTIONS, SPLITS, WINDOW_SAMPLES, condition,
                          pipeline_config, resample_index, write_json, zscore)

NATIVE_FS = 2000
DEFAULT_OUT_ROOT = Path(os.environ.get("EMG_OUT_ROOT", Path(__file__).resolve().parent / "processed")) / "benchmark"


def read_session(path: Path):
    with h5py.File(path, "r") as f:
        g = f["emg2qwerty"]
        ts = g["timeseries"][...]
        emg = np.concatenate([ts["emg_left"], ts["emg_right"]], axis=1).astype(np.float64)
        user = g.attrs.get("user", "unknown")
        user = user.decode() if isinstance(user, bytes) else str(user)
        keys = g.attrs.get("keystrokes", "[]")
        keys = json.loads(keys) if isinstance(keys, (str, bytes)) else list(keys)
        return emg, ts["time"].astype(np.float64), keys, user


def key_text(keys) -> str:
    text = ""
    for k in keys:
        key = k.get("key", "")
        if key == "Key.space":
            text += " "
        elif not key.startswith("Key."):
            text += key
    return text.strip()


def process_session(path: Path):
    emg, time, keys, user = read_session(path)
    x = condition(emg, NATIVE_FS)
    t = time[resample_index(len(time), NATIVE_FS, len(x))]
    n = len(x)
    b1 = int(round(n * SPLIT_FRACTIONS[0]))
    b2 = int(round(n * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1])))
    out = {s: ([], []) for s in SPLITS}
    for sp, (s, e) in zip(SPLITS, ((0, b1), (b1, b2), (b2, n))):
        for w0 in range(s, e - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
            t0, t1 = t[w0], t[w0 + WINDOW_SAMPLES - 1]
            text = key_text(k for k in keys if t0 <= k.get("start", -np.inf) < t1)
            if text:
                out[sp][0].append(zscore(x[w0:w0 + WINDOW_SAMPLES].T))
                out[sp][1].append(text)
    return user, float(time[0]), out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", type=Path, required=True, help="folder with emg2qwerty *.hdf5 sessions")
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--max_sessions", type=int, default=None)
    args = p.parse_args()

    sessions = sorted(args.raw_dir.glob("*.hdf5"))[:args.max_sessions]
    per_user = defaultdict(list)
    for path in sessions:
        user, start, out = process_session(path)
        per_user[user].append((start, out))
        print(f"  {path.name}: user {user} " + str({s: len(v[1]) for s, v in out.items()}), flush=True)

    out_dir = args.out_root / "emg2qwerty"
    counts = {}
    for user, items in per_user.items():
        d = out_dir / f"user_{user}"
        d.mkdir(parents=True, exist_ok=True)
        counts[f"user_{user}"] = {}
        for sp in SPLITS:
            xs = [w for _, out in sorted(items, key=lambda it: it[0]) for w in out[sp][0]]
            ys = [y for _, out in sorted(items, key=lambda it: it[0]) for y in out[sp][1]]
            np.save(d / f"{sp}_emg.npy", np.stack(xs) if xs else np.zeros((0, 32, WINDOW_SAMPLES), np.float32))
            np.save(d / f"{sp}_labels.npy", np.array(ys, dtype=object))
            counts[f"user_{user}"][sp] = len(ys)

    write_json(out_dir / "dataset_info.json", dict(
        dataset="emg2qwerty", native_fs=NATIVE_FS, pipeline=pipeline_config(),
        split="contiguous 70/15/15 temporal blocks per session",
        target="characters of keypresses with onset inside the window; Key.space -> ' ', "
               "other special keys (backspace, enter, shift, ...) omitted; empty windows dropped",
        users=counts))
    print(f"done: {len(counts)} users")


if __name__ == "__main__":
    main()
