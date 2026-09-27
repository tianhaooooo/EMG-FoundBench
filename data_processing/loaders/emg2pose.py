from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

NATIVE_FS = 2000
RAW_DIR = RAW_ROOT / "emg2pose_raw" / "emg2pose_data"
METADATA_FALLBACK = RAW_ROOT / "emg2pose_metadata.csv"


class EMG2Pose(DatasetLoader):
    name = "emg2pose"

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir = Path(raw_dir)
        meta_path = self.raw_dir / "metadata.csv"
        self.meta = pd.read_csv(meta_path if meta_path.exists() else METADATA_FALLBACK)
        self.classes = sorted(self.meta["stage"].unique())

    def subjects(self):
        return sorted(self.meta["user"].astype(str).unique(), key=natural_key)

    def load(self, subject):
        rows = self.meta[self.meta["user"].astype(str) == subject].sort_values(["start", "side"])
        recs = []
        for _, row in rows.iterrows():
            path = self.raw_dir / f"{row['filename']}.hdf5"
            if not path.exists():
                raise FileNotFoundError(f"{path} (extract emg2pose_dataset.tar first)")
            with h5py.File(path, "r") as h:
                emg = h["emg2pose"]["timeseries"]["emg"].astype(np.float64)
            recs.append(Recording(emg=emg, fs=NATIVE_FS, labels=row["stage"], trials=None))
        return recs
