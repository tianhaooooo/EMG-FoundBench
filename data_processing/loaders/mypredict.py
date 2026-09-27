import re

import h5py
import numpy as np

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

NATIVE_FS = 1000
GRID = "EMG_Right_MA"
NAMED = {
    "MP1": [f"EMG_{side}_{m}" for side in ("Left", "Right")
            for m in ("BF", "GM", "Gmax", "Gmed", "RF", "ST", "TA", "VL")],
    "MP2": [],
    "MP3": [f"EMG_Left_{m}" for m in ("AM", "BF", "GM", "RF", "ST", "TA", "VL")]
           + [f"EMG_Right_{m}" for m in ("AM", "BF", "GM", "Gmax", "RF", "ST", "TA", "VL")],
}
USE_GRID = {"MP1": False, "MP2": True, "MP3": True}


class MyPredict(DatasetLoader):
    classes = list(range(10))

    def __init__(self, group: str = "MP1"):
        self.group = group
        self.name = {"MP1": "MyPredict", "MP2": "MyPredict2", "MP3": "MyPredict3"}[group]
        self.root = RAW_ROOT / "MyPredict"

    def subjects(self):
        return sorted((f.stem for f in self.root.glob(f"{self.group}*.hdf5")), key=natural_key)

    def _emg(self, t):
        parts = [np.stack([t[c][...].astype(np.float64).reshape(-1) for c in NAMED[self.group]], axis=1)] \
            if NAMED[self.group] else []
        if USE_GRID[self.group]:
            grid = t[GRID][...].astype(np.float64)
            parts.append(grid.reshape(len(grid), -1))
        n = min(len(p) for p in parts)
        return np.concatenate([p[:n] for p in parts], axis=1).astype(np.float32)

    def load(self, subject):
        recs = []
        with h5py.File(self.root / f"{subject}.hdf5", "r") as h:
            days = sorted((k for k in h if k.startswith("Day_")), key=natural_key)
            for di, day in enumerate(days):
                for name in sorted((k for k in h[day] if k.startswith("Trial_")), key=natural_key):
                    t = h[day][name]
                    emg = self._emg(t)
                    fs = 1.0 / np.median(np.diff(t["Time"][...].reshape(-1)))
                    if abs(fs - NATIVE_FS) > 1.0:
                        raise ValueError(f"{subject}/{day}/{name}: sampling rate {fs:.1f} Hz, expected {NATIVE_FS}")
                    label = np.floor(t["Label"][...].reshape(-1) + 1e-6).astype(np.int64)
                    n = min(len(emg), len(label))
                    emg, label = emg[:n], label[:n]
                    bad = ~np.isfinite(emg).all(axis=1)
                    if bad.any():
                        emg = np.where(np.isfinite(emg), emg, 0.0)
                        label = np.where(bad, -1, label)
                    recs.append(Recording(emg=emg, fs=NATIVE_FS, labels=label,
                                          trials=di * 1000 + int(re.search(r"\d+", name).group())))
        return recs
