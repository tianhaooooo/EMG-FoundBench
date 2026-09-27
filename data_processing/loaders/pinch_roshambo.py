import re

import numpy as np

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

FILE_RE = re.compile(r"subject(\d+)_session(\d+)_emg\.npy")


class _MyoDataset(DatasetLoader):
    folder = ""

    def __init__(self):
        self.root = RAW_ROOT / "PINCH&ROSHAMBO" / self.folder
        self._files = {}
        for f in self.root.glob("*_emg.npy"):
            subj, ses = map(int, FILE_RE.match(f.name).groups())
            self._files.setdefault(f"s{subj}", []).append((ses, f))

    def subjects(self):
        return sorted(self._files, key=natural_key)

    def load(self, subject):
        recs = []
        for _, f in sorted(self._files[subject]):
            emg = np.load(f).astype(np.float64)
            ann = np.load(f.with_name(f.name.replace("_emg.npy", "_ann.npy")), allow_pickle=True)
            ann = np.array([a.decode() if isinstance(a, bytes) else str(a) for a in ann])
            n = min(len(emg), len(ann))
            recs.append(Recording(emg=emg[:n], fs=200, labels=ann[:n], trials=None))
        return recs



class Pinch(_MyoDataset):
    name = "Pinch"
    folder = "Pinch"
    classes = ["Pinch1", "Pinch2", "Pinch3", "Pinch4"]


class Roshambo(_MyoDataset):
    name = "Roshambo"
    folder = "Roshambo"
    classes = ["rock", "paper", "scissor"]
