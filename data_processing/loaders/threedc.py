import re

import numpy as np

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

FILE_RE = re.compile(r"3dc_EMG_gesture_(\d+)_(\d+)\.txt")


class ThreeDC(DatasetLoader):
    name = "3DC"
    classes = list(range(11))

    def __init__(self):
        self.root = RAW_ROOT / "3DC_raw"

    def subjects(self):
        return sorted((f"s{int(d.name[len('Participant'):])}" for d in self.root.glob("Participant*")
                       if d.is_dir()), key=natural_key)

    def load(self, subject):
        base = self.root / f"Participant{subject[1:]}"
        items = []
        for fi, folder in enumerate(("train", "test")):
            for f in (base / folder / "EMG").glob("3dc_EMG_gesture_*.txt"):
                rep, cls = map(int, FILE_RE.match(f.name).groups())
                items.append((cls, fi, rep, f))
        return [Recording(emg=np.loadtxt(f, delimiter=",", dtype=np.float64), fs=1000,
                          labels=cls, trials=fi * 10 + rep)
                for cls, fi, rep, f in sorted(items)]
