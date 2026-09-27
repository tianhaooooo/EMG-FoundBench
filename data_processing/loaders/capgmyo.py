import io
import re
import zipfile

import numpy as np
import scipy.io as sio

from emg_pipeline import Recording, natural_key
from . import RAW_ROOT, DatasetLoader


class CapgMyo(DatasetLoader):
    name = "CapgMyo"
    classes = list(range(1, 9))
    short_trial_handling = "stretch"

    def __init__(self):
        self.zip_root = RAW_ROOT / "CapgMyo_DBa"
        self.dir_root = RAW_ROOT / "CapgMyo_DBa_extracted"

    def _sources(self):
        def sid(name):
            return "s" + str(int(re.search(r"dba-s(\d+)", name).group(1)))
        src = {}
        for z in self.zip_root.glob("dba-s*.zip"):
            src[sid(z.stem)] = z
        for d in self.dir_root.glob("dba-s*"):
            src.setdefault(sid(d.name), d)
        return src

    def subjects(self):
        return sorted(self._sources(), key=natural_key)

    def _mats(self, src):
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                for n in sorted(x for x in z.namelist() if x.endswith(".mat")):
                    yield sio.loadmat(io.BytesIO(z.read(n)))
        else:
            for f in sorted(src.rglob("*.mat")):
                yield sio.loadmat(f)

    def load(self, subject):
        items = []
        for m in self._mats(self._sources()[subject]):
            items.append((int(m["gesture"].item()), int(m["trial"].item()),
                          m["data"].astype(np.float64)))
        return [Recording(emg=x, fs=1000, labels=g, trials=g * 100 + t)
                for g, t, x in sorted(items, key=lambda it: (it[0], it[1]))]
