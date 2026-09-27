import re

import numpy as np
import wfdb

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

FILE_RE = re.compile(r"session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)")


class GRABMyo(DatasetLoader):
    name = "GRABMyo"
    classes = list(range(1, 18))

    def __init__(self, sessions=("Session1",)):
        self.root = RAW_ROOT / "GRABMyo" / "1.1.0"
        self.sessions = sessions

    def subjects(self):
        dirs = {d.name for s in self.sessions for d in (self.root / s).glob("session*_participant*")}
        return sorted(("s" + d.split("participant")[1] for d in dirs), key=natural_key)

    def load(self, subject):
        pid = int(subject[1:])
        records = []
        for s in self.sessions:
            for hea in (self.root / s / f"{s.lower()}_participant{pid}").glob("*.hea"):
                ses, _, gesture, trial = map(int, FILE_RE.search(hea.stem).groups())
                records.append((ses, gesture, trial, hea))
        recs = []
        for ses, gesture, trial, hea in sorted(records):
            r = wfdb.rdrecord(str(hea.with_suffix("")))
            keep = [i for i, name in enumerate(r.sig_name) if not name.startswith("U")]
            recs.append(Recording(emg=r.p_signal[:, keep], fs=r.fs, labels=gesture,
                                  trials=ses * 100 + trial))
        return recs
