import re

import numpy as np
import scipy.io as sio

from emg_pipeline import NO_TRIAL, Recording, natural_key
from . import RAW_ROOT, DatasetLoader

PER_EXERCISE_OFFSETS = {"E1": 0, "E2": 12, "E3": 29}
CONFIG = {
    "DB1": dict(fs=100, offsets=PER_EXERCISE_OFFSETS, classes=list(range(1, 53))),
    "DB2": dict(fs=2000, offsets=None, classes=list(range(1, 50))),
    "DB3": dict(fs=2000, offsets=None, classes=list(range(1, 50))),
    "DB4": dict(fs=2000, offsets=PER_EXERCISE_OFFSETS, classes=list(range(1, 53))),
    "DB5": dict(fs=200, offsets=PER_EXERCISE_OFFSETS, classes=list(range(1, 53))),
    "DB6": dict(fs=2000, offsets=None, classes=[1, 3, 4, 6, 9, 10, 11], drop_channels=[8, 9]),
    "DB7": dict(fs=2000, offsets=None, classes=list(range(1, 41))),
}
SUBJECT_RE = re.compile(r"^S(\d+)_", re.IGNORECASE)


class NinaPro(DatasetLoader):
    def __init__(self, db: str):
        self.db = db
        self.cfg = CONFIG[db]
        self.name = f"NinaPro_{db}"
        self.classes = self.cfg["classes"]
        self.root = RAW_ROOT / f"NinaPro_{db}"
        self._files = {}
        for f in self.root.rglob("*.mat"):
            if "__MACOSX" in f.parts or f.name.startswith("._"):
                continue
            m = SUBJECT_RE.match(f.name)
            if m:
                self._files.setdefault(f"s{int(m.group(1))}", []).append(f)

    def subjects(self):
        return sorted(self._files, key=natural_key)

    def _exercise(self, fname: str) -> str:
        tokens = fname[:-4].upper().split("_")
        for ex in ("E1", "E2", "E3"):
            if ex in tokens:
                return ex
        raise ValueError(f"cannot determine exercise from {fname}")

    def load(self, subject):
        recs = []
        for f in sorted(self._files[subject], key=lambda p: natural_key(p.name)):
            m = sio.loadmat(f)
            emg = m["emg"].astype(np.float64)
            lab = m["restimulus"].reshape(-1).astype(np.int64)
            rep = m["rerepetition"].reshape(-1).astype(np.int64)
            n = min(len(emg), len(lab), len(rep))
            emg, lab, rep = emg[:n], lab[:n], rep[:n]
            if self.cfg.get("drop_channels"):
                emg = np.delete(emg, self.cfg["drop_channels"], axis=1)
            if self.cfg["offsets"]:
                lab = np.where(lab > 0, lab + self.cfg["offsets"][self._exercise(f.name)], 0)
            trials = np.where((lab > 0) & (rep > 0), lab * 1000 + rep, NO_TRIAL)
            recs.append(Recording(emg=emg, fs=self.cfg["fs"], labels=lab, trials=trials))
        return recs
