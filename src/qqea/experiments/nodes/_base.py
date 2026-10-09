"""Shared machinery for the calibration nodes: the constructor, the batched sweep
helper, saving of raw runs, the call history, and state-restoring helpers.

The node contract: a node returns ``(ok, value)`` and does not raise on a bad
fit. ``ok`` is False when the result should not be trusted; qubit parameters
are only written when ``ok`` is True and ``update`` is True.
"""

import contextlib
import functools
import warnings
from datetime import datetime

from qcodes import ManualParameter
from quantify_scheduler.gettables import ScheduleGettable

from qqea.experiments.saving import save_retrieve_acquisition_dataset


def _qubit_names(target):
    """Names of the qubit (or qubits) a node was called on."""
    if target is None:
        return []
    if hasattr(target, "name"):
        return [target.name]
    try:
        return [q.name for q in target]
    except (TypeError, AttributeError):
        return []


def node(method):
    """Record each call of a node method in ``self.history``.

    One entry per call: ``{'time', 'node', 'qubits', 'ok', 'value', 'tuids'}``,
    plus ``'error'`` when the node raised (the exception is re-raised).
    ``tuids`` are the datasets the call wrote, in order.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("qubit", kwargs.get("qubits"))
        outer_tuids = self._tuids
        self._tuids = []
        entry = {
            "time": datetime.now(),
            "node": method.__name__,
            "qubits": _qubit_names(target),
        }
        try:
            ok, value = method(self, *args, **kwargs)
        except Exception as exc:
            entry.update(ok=False, value=None, tuids=self._tuids,
                         error=f"{type(exc).__name__}: {exc}")
            self.history.append(entry)
            raise
        finally:
            tuids = self._tuids
            self._tuids = outer_tuids + tuids
        entry.update(ok=ok, value=value, tuids=tuids)
        self.history.append(entry)
        return ok, value

    return wrapper


def batched_parameter(name, unit="", label=None, batch_size=None):
    """A batched ``ManualParameter`` to sweep with ``ScheduleGettable``."""
    kwargs = {"unit": unit}
    if label is not None:
        kwargs["label"] = label
    param = ManualParameter(name=name, **kwargs)
    param.batched = True
    if batch_size is not None:
        param.batch_size = batch_size
    return param


class NodeBase:
    """Holds the runtime instruments shared by every node."""

    def __init__(self, quantum_device, meas_ctrl, acq_delay, instrument_coordinator=None):
        self.quantum_device = quantum_device
        self.meas_ctrl = meas_ctrl
        self.acq_delay = acq_delay
        self.instrument_coordinator = instrument_coordinator
        self.history = []
        self._tuids = []

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def _run_sweep(self, schedule_function, schedule_kwargs, settables, setpoints, label,
                   repetitions, grid=False, soft_avg=None, **gettable_kwargs):
        """Run a batched ``ScheduleGettable`` sweep through ``meas_ctrl``.

        ``setpoints`` is one array (``meas_ctrl.setpoints``) or, with
        ``grid=True``, a list of arrays (``meas_ctrl.setpoints_grid``).
        ``gettable_kwargs`` go to ``ScheduleGettable`` (``real_imag``,
        ``num_channels``, ``max_batch_size``); ``batched`` is always True.
        Returns the dataset and records its TUID for the history.
        """
        gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=schedule_function,
            schedule_kwargs=schedule_kwargs,
            batched=True,
            **gettable_kwargs,
        )
        self.meas_ctrl.gettables(gettable)
        self.meas_ctrl.settables(settables)
        if grid:
            self.meas_ctrl.setpoints_grid(setpoints)
        else:
            self.meas_ctrl.setpoints(setpoints)
        self.quantum_device.cfg_sched_repetitions(repetitions)

        run_kwargs = {} if soft_avg is None else {"soft_avg": soft_avg}
        dataset = self.meas_ctrl.run(label, **run_kwargs)
        tuid = dataset.attrs.get("tuid")
        if tuid is not None:
            self._tuids.append(tuid)
        return dataset

    def _require_instrument_coordinator(self, node_name):
        if self.instrument_coordinator is None:
            raise RuntimeError(
                f"{node_name} requires instrument_coordinator to be set on "
                "CalibrationNodes -- pass instrument_coordinator=ic to the constructor."
            )

    def _save_raw(self, dataset, name, **attrs):
        """Save a raw ``retrieve_acquisition()`` dataset to the data directory.

        ``attrs`` (typically the swept setpoints) are stored on the saved copy.
        A failed save only warns: the measurement already happened and the
        node can still use the data. Returns the TUID, or None.
        """
        to_save = dataset.copy()
        to_save.attrs.update(attrs)
        try:
            tuid = save_retrieve_acquisition_dataset(to_save, name)
        except Exception as exc:  # never lose a calibration over a save
            warnings.warn(f"Could not save {name!r}: {type(exc).__name__}: {exc}")
            return None
        self._tuids.append(tuid)
        return tuid

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _quiet(self):
        """Silence ``meas_ctrl`` progress output, restoring the previous setting."""
        prev = self.meas_ctrl.verbose()
        self.meas_ctrl.verbose(False)
        try:
            yield
        finally:
            self.meas_ctrl.verbose(prev)

    @staticmethod
    def _assign_acq_channels(qubits):
        """Map ``qubits[i]`` to acquisition channel ``i``.

        Left in place afterwards on purpose: later multi-qubit runs in the
        notebook (e.g. GST) rely on the same mapping.
        """
        for i, qubit in enumerate(qubits):
            qubit.measure.acq_channel(i)
