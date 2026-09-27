import numpy as np
import pandas as pd
import scipy.io as sio

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

NATIVE_FS = 2400


class MendeleyKnee(DatasetLoader):
    name = "MendeleyKnee"
    classes = [0, 1]

    def __init__(self):
        root = RAW_ROOT / "Mendeley_Knee_EMG" / "Data"
        rows = sio.loadmat(root / "Raw_Data.mat")["DATA_EXPORT_SORTED"]
        meta = pd.read_excel(root / "Anonymized_Data_Identifiers.xlsx")
        if len(meta) != rows.shape[0] or any(
                int(np.squeeze(rows[i][0])) != int(meta["Subject ID"].iloc[i]) for i in range(len(meta))):
            raise ValueError("Excel identifiers are not row-aligned with Raw_Data.mat")
        self._rows = {}
        for i in range(rows.shape[0]):
            subject = f"s{int(np.squeeze(rows[i][0]))}"
            label = int(meta["Knee Injury History"].iloc[i])
            self._rows.setdefault(subject, []).append((label, np.asarray(rows[i][2], dtype=np.float64)))

    def subjects(self):
        return sorted(self._rows, key=natural_key)

    def load(self, subject):
        rows = self._rows[subject]
        n = min(len(emg) for _, emg in rows)
        emg = np.concatenate([e[:n] for _, e in rows], axis=1)
        return [Recording(emg=emg, fs=NATIVE_FS, labels=rows[0][0], trials=None)]
