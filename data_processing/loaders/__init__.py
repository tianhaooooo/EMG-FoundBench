import os
from pathlib import Path

RAW_ROOT = Path(os.environ.get("EMG_RAW_ROOT", Path(__file__).resolve().parents[1] / "raw_data"))


class DatasetLoader:
    name: str = ""
    classes: list = []
    short_trial_handling: str | None = None

    def subjects(self) -> list[str]:
        raise NotImplementedError

    def load(self, subject: str) -> list:
        raise NotImplementedError



def registry() -> dict:
    from .arm_translation import ArmTranslation
    from .capgmyo import CapgMyo
    from .continuous_transitions import ContinuousTransitions
    from .emg2pose import EMG2Pose
    from .grabmyo import GRABMyo
    from .hdemg import HDEMG
    from .mendeley_knee import MendeleyKnee
    from .mypredict import MyPredict
    from .ninapro import NinaPro
    from .pinch_roshambo import Pinch, Roshambo
    from .siat_llmd import SIATLLMD
    from .threedc import ThreeDC

    loaders = {f"NinaPro_{db}": (lambda db=db: NinaPro(db))
               for db in ("DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7")}
    loaders.update({
        "GRABMyo": GRABMyo, "CapgMyo": CapgMyo, "Pinch": Pinch, "Roshambo": Roshambo,
        "3DC": ThreeDC, "HD-EMG": HDEMG, "SIAT-LLMD": SIATLLMD, "MendeleyKnee": MendeleyKnee,
        "MyPredict": MyPredict, "MyPredict2": (lambda: MyPredict("MP2")),
        "MyPredict3": (lambda: MyPredict("MP3")), "ContinuousTransitions": ContinuousTransitions,
        "emg2pose": EMG2Pose, "ArmTranslation": ArmTranslation,
    })
    return loaders
