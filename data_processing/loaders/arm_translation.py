import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

NATIVE_FS = 2000
RAW_DIR = RAW_ROOT / "arm_trans_raw" / "data"
BLOCK_RE = re.compile(r"participant(\d+)_day(\d+)_block(\d+)")


class ArmTranslation(DatasetLoader):
    name = "ArmTranslation"
    classes = [1, 2, 3, 4, 5, 6]

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir = Path(raw_dir)

    def subjects(self):
        return sorted((f"s{int(d.name.split('_')[1])}" for d in self.raw_dir.glob("participant_*")
                       if d.is_dir()), key=natural_key)

    def load(self, subject):
        base = self.raw_dir / f"participant_{subject[1:]}"
        blocks = sorted((tuple(map(int, BLOCK_RE.search(d.name).groups()[1:])), d)
                        for d in base.iterdir() if d.is_dir() and BLOCK_RE.search(d.name))
        recs = []
        for (day, block), d in blocks:
            trials = pd.read_csv(d / "trials.csv")
            with h5py.File(d / "emg_data.hdf5", "r") as h:
                for i, row in trials.iterrows():
                    emg = h[str(i)][...].astype(np.float64).T
                    recs.append(Recording(emg=emg, fs=NATIVE_FS, labels=int(row["grasp"]),
                                          trials=(day * 10 + block) * 1000 + i))
        return recs
