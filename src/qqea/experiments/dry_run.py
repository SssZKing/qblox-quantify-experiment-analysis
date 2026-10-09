"""Dry runs: exercise the calibration nodes on a dummy Qblox cluster.

For use in the ``qblox_dev`` env only, never against the live setup. In a dry
run the schedules are compiled and uploaded to a qblox-instruments dummy
cluster, so compilation, ``meas_ctrl`` and the analysis all run; only the
acquired data is fake. Example::

    from qqea.experiments import dry_run

    cluster = dry_run.dummy_cluster(hardware_cfg)      # instead of Cluster(..., identifier=ip)
    ...  # build ic, meas_ctrl, quantum_device and the qubits as usual
    nodes = CalibrationNodes(quantum_device, meas_ctrl, ACQ_DELAY, instrument_coordinator=ic)

    with dry_run.simulate(dry_run.readout_data()):
        ok, fid = nodes.readout_calibration(qubit1)

Without :func:`simulate` the dummy cluster returns empty data, which still
checks that every node compiles, runs, and fails cleanly with ``(False, None)``.
"""

import contextlib

import numpy as np

from qqea.experiments.simulated_data import (
    ErrorModel,
    get_simulated_rb_data,
    get_simulated_readout_data,
    gate_strings_from_schedule,
)

# hardware_description instrument_type -> qblox_instruments.ClusterType member
_MODULE_TYPES = {
    "QCM": "CLUSTER_QCM",
    "QCM_RF": "CLUSTER_QCM_RF",
    "QRM": "CLUSTER_QRM",
    "QRM_RF": "CLUSTER_QRM_RF",
    "QTM": "CLUSTER_QTM",
}


def dummy_cluster(hardware_cfg, name=None):
    """A qblox-instruments dummy ``Cluster`` with the modules of ``hardware_cfg``.

    ``hardware_cfg`` is the quantify hardware compilation config; its
    ``hardware_description`` names the cluster and the instrument type in each
    slot. ``name`` defaults to the cluster's name in that description, so the
    rest of the config works unchanged.
    """
    from qblox_instruments import Cluster, ClusterType

    clusters = {
        key: value for key, value in hardware_cfg["hardware_description"].items()
        if value.get("instrument_type") == "Cluster"
    }
    if len(clusters) != 1:
        raise ValueError(f"Expected one Cluster in hardware_description, found {len(clusters)}.")
    cluster_name, description = next(iter(clusters.items()))
    dummy_cfg = {
        int(slot): getattr(ClusterType, _MODULE_TYPES[module["instrument_type"]])
        for slot, module in description["modules"].items()
    }
    return Cluster(name=name or cluster_name, dummy_cfg=dummy_cfg)


@contextlib.contextmanager
def simulate(data_function):
    """Replace ``ScheduleGettable.get`` data with ``data_function(gettable, real_data)``.

    The real ``get`` still runs first (on the dummy cluster), so compilation
    and upload are exercised; ``data_function`` returns ``[I0, Q0, I1, Q1, ...]``
    and is converted to magnitude/phase (degrees) when the gettable was made
    with ``real_imag=False``. The original ``get`` is restored on exit.
    """
    from quantify_scheduler.gettables import ScheduleGettable

    original_get = ScheduleGettable.get

    def get(self):
        data = data_function(self, original_get(self))
        if getattr(self, "real_imag", True):
            return tuple(data)
        out = []
        for i_data, q_data in zip(data[0::2], data[1::2]):
            z = np.asarray(i_data) + 1j * np.asarray(q_data)
            out.extend([np.abs(z), np.angle(z, deg=True)])
        return tuple(out)

    ScheduleGettable.get = get
    try:
        yield
    finally:
        ScheduleGettable.get = original_get


def readout_data(states_key="prepared_states", **blob_kwargs):
    """Data function for the readout calibration nodes: one IQ blob per prepared state.

    ``blob_kwargs`` go to :func:`get_simulated_readout_data` (centroids, noise).
    For ``multiplexed_readout_calibration`` every channel gets the same blobs.
    """
    def data_function(gettable, real_data):
        states = np.asarray(gettable.schedule_kwargs[states_key]())
        num_channels = (getattr(gettable, "num_channels", None)
                        or getattr(gettable, "_num_channels", None) or 1)
        data = []
        for _ in range(num_channels):
            data.extend(get_simulated_readout_data(states, **blob_kwargs))
        return data

    return data_function


def rb_data(error_model=None, num_cal_points=0):
    """Data function for ``randomized_benchmarking``, from the compiled gate sequence.

    ``num_cal_points`` = 0 matches ``randomized_benchmarking_schedule``, which
    has no calibration points; check this if that schedule changes.
    """
    error_model = error_model or ErrorModel(depolarizing_error=0.002)

    def data_function(gettable, real_data):
        compiled = gettable.compiled_schedule
        return get_simulated_rb_data(
            error_model=error_model,
            gate_strings=gate_strings_from_schedule(compiled),
            lengths=np.asarray(gettable.schedule_kwargs["lengths"]()),
            repetitions=compiled.repetitions,
            num_cal_points=num_cal_points,
        )

    return data_function
