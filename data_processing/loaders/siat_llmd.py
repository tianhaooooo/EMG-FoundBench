import re

import numpy as np
import pandas as pd

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

NATIVE_FS = 1920
ACTIVITIES = ["DNS", "HS", "KLCL", "KLFT", "LLB", "LLF", "LLS", "LUGB", "LUGF", "SITDN",
              "STC", "STDUP", "TO", "TPTO", "UPS", "WAK"]


class SIATLLMD(DatasetLoader):
    name = "SIAT-LLMD"
    classes = ACTIVITIES

    def __init__(self):
        self.root = RAW_ROOT / "SIAT_LLMD" / "SIAT_LLMD20230404"

    def subjects(self):
        return sorted((f"s{int(d.name[3:])}" for d in self.root.glob("Sub*") if d.is_dir()),
                      key=natural_key)

    def load(self, subject):
        data_dir = self.root / f"Sub{int(subject[1:]):02d}" / "Data"
        recs = []
        for f in sorted(data_dir.glob("*_Data.csv")):
            activity = re.match(r"Sub\d+_(.+)_Data\.csv", f.name).group(1)
            df = pd.read_csv(f)
            fs = 1.0 / np.median(np.diff(df["Time"].to_numpy()))
            if abs(fs - NATIVE_FS) > 1.0:
                raise ValueError(f"{f.name}: sampling rate {fs:.1f} Hz, expected {NATIVE_FS}")
            emg = df[[c for c in df.columns if c.startswith("sEMG:")]].to_numpy(np.float64)
            recs.append(Recording(emg=emg, fs=NATIVE_FS, labels=activity, trials=None))
        return recs
