"""Registry of model families. Each family module provides NAME, SIZES, build_classifier(),
build_ctc(), batch_size() and CTC_BATCH_SIZE."""
import importlib

FAMILIES = {"moment": "moment.model", "chronos2": "chronos2.model", "timesfm": "timesfm25.model",
            "lagllama": "lagllama.model", "patchtst": "patchtst.model",
            "brant": "brant.model", "units": "units.model",
            "bilstm": "bilstm.model"}
ALL_SIZES = ["small", "base"]


def get_family(name, model_size):
    family = importlib.import_module(FAMILIES[name])
    if model_size not in family.SIZES:
        raise ValueError(f"{name} is available in sizes {family.SIZES}, got {model_size!r}")
    return family


def display_name(family, model_size):
    if len(family.SIZES) == 1:
        return family.NAME  # e.g. TimesFM
    return f"{family.NAME}-{model_size[0].upper()}"  # e.g. MOMENT-S, Chronos-2-B
