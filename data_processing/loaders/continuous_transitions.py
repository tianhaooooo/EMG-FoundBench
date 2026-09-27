import re

import h5py
import numpy as np

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

NATIVE_FS = 2000
LABEL_DELAY = 1000
RAW_DIRS = [RAW_ROOT / "ContinuousTransitions_raw", RAW_ROOT / "[EMG] Shri Continuous Dataset"]


class ContinuousTransitions(DatasetLoader):
    name = "ContinuousTransitions"
    classes = list(range(7))

    def __init__(self):
        from pathlib import Path
        self._files = {}
        for d in RAW_DIRS:
            for f in Path(d).glob("P*.hdf5"):
                self._files.setdefault("s" + str(int(re.search(r"P(\d+)", f.stem).group(1))), f)

    def subjects(self):
        return sorted(self._files, key=natural_key)

    def load(self, subject):
        recs = []
        with h5py.File(self._files[subject], "r") as h:
            for session in ("ramp", "continuous"):
                g = h[session]["emg"]
                emg = g["signal"][()].astype(np.float64)
                labels = g["prompt"][()].astype(np.int64) - 1
                if session == "continuous":
                    labels = np.r_[np.full(LABEL_DELAY, labels[0]), labels[:-LABEL_DELAY]]
                trials = g["trial"][()].astype(np.int64)
                recs.append(Recording(emg=emg, fs=NATIVE_FS, labels=labels, trials=trials))
        return recs
