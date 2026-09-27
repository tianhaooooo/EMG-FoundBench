import numpy as np
import scipy.io as sio

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader

NATIVE_FS = 2000
CATEGORIES = ["Level_Ground", "Ramp_Ambulation", "Stairs_Ambulation"]
EMG_FIELDS = ("RightLeg_EMG", "RightFoot_EMG")


def _cycles(node):
    if isinstance(node, np.ndarray) and node.dtype == object:
        for item in node.flat:
            yield from _cycles(item)
    elif hasattr(node, "_fieldnames"):
        field = next((f for f in EMG_FIELDS if f in node._fieldnames), None)
        if field:
            yield np.asarray(getattr(node, field), dtype=np.float64)
        else:
            for f in node._fieldnames:
                yield from _cycles(getattr(node, f))


class HDEMG(DatasetLoader):
    name = "HD-EMG"
    classes = CATEGORIES
    short_trial_handling = "reflect_pad"

    def __init__(self):
        self.root = RAW_ROOT / "HDEMG"

    def subjects(self):
        return sorted((d.name for d in self.root.glob("P*") if d.is_dir()), key=natural_key)

    def load(self, subject):
        m = sio.loadmat(self.root / subject / subject / f"{subject}.mat",
                        struct_as_record=False, squeeze_me=True)
        cycles = m[subject].RightFoot_GaitCycle_Data
        recs, trial = [], 0
        for cat in CATEGORIES:
            for emg in _cycles(getattr(cycles, cat)):
                recs.append(Recording(emg=emg, fs=NATIVE_FS, labels=cat, trials=trial))
                trial += 1
        return recs
