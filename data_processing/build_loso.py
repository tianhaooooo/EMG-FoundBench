#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emg_pipeline import LOSO_COHORT_SIZE, SPLITS, natural_key, write_json

DEFAULT_BENCH_ROOT = Path(os.environ.get("EMG_OUT_ROOT", Path(__file__).resolve().parent / "processed")) / "benchmark"
DEFAULT_LOSO_ROOT = Path(os.environ.get("EMG_OUT_ROOT", Path(__file__).resolve().parent / "processed")) / "loso"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    p.add_argument("--loso_root", type=Path, default=DEFAULT_LOSO_ROOT)
    args = p.parse_args()

    src = args.bench_root / args.dataset
    dst = args.loso_root / args.dataset
    subjects = sorted((d.name for d in src.iterdir() if (d / "train_emg.npy").exists()), key=natural_key)
    counts = {}
    for s in subjects:
        xs = [np.load(src / s / f"{sp}_emg.npy") for sp in SPLITS]
        ys = [np.load(src / s / f"{sp}_labels.npy", allow_pickle=True) for sp in SPLITS]
        (dst / s).mkdir(parents=True, exist_ok=True)
        np.save(dst / s / "emg.npy", np.concatenate(xs))
        np.save(dst / s / "labels.npy", np.concatenate(ys))
        counts[s] = int(sum(len(y) for y in ys))
        print(f"  {s}: {counts[s]} windows", flush=True)

    write_json(dst / "loso_cohort.json", dict(cohort=subjects[:LOSO_COHORT_SIZE],
                                              rule="first 10 subjects in canonical id order",
                                              windows_per_subject=counts,
                                              source=str(src)))
    print(f"done: {len(subjects)} subjects, cohort = {subjects[:LOSO_COHORT_SIZE]}")


if __name__ == "__main__":
    main()
