"""Running experiments: calibration nodes, hardware runners, instrument drivers,
data saving, and simulated data for dry runs.

Names are imported lazily, so ``import qqea.experiments.hp83732b`` (which
``qqea.schedules.single_qubit`` does) does not pull in the calibration nodes,
which themselves import ``qqea.schedules``.
"""

_EXPORTS = {
    "CalibrationNodes": "qqea.experiments.calibration_nodes",
    "HP83732B": "qqea.experiments.hp83732b",
    "set_datadir_from_env": "qqea.experiments.datadir",
    "save_retrieve_acquisition_dataset": "qqea.experiments.saving",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
