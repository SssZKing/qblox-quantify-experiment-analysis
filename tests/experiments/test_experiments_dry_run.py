"""Dry run on a qblox-instruments dummy cluster: schedules compile, run and save.

Needs the full stack (quantify-scheduler, qblox-instruments), so it runs in the
qblox_dev env only and never touches the real cluster:

    python -m pytest tests/experiments/test_experiments_dry_run.py
"""

import numpy as np
import pytest

pytest.importorskip("qblox_instruments")

from qcodes.instrument import Instrument  # noqa: E402
from quantify_core.data.handling import set_datadir  # noqa: E402
from quantify_core.measurement import MeasurementControl  # noqa: E402
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice  # noqa: E402
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement  # noqa: E402
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator  # noqa: E402
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent  # noqa: E402

from qqea.experiments import dry_run  # noqa: E402
from qqea.experiments.calibration_nodes import CalibrationNodes  # noqa: E402

# The SNL315 layout (slots 2 and 4 QCM-RF, slot 7 QRM-RF), one qubit.
HARDWARE_CFG = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "2": {"instrument_type": "QCM_RF"},
                "4": {"instrument_type": "QCM_RF"},
                "7": {"instrument_type": "QRM_RF"},
            },
        },
    },
    "hardware_options": {
        "modulation_frequencies": {
            "qubit1:res-qubit1.ro": {"lo_freq": 7.1e9},
            "qubit1:mw-qubit1.01": {"lo_freq": 4.43e9},
        },
    },
    "connectivity": {
        "graph": [
            ("cluster0.module2.complex_output_0", ["qubit1:mw"]),
            ("cluster0.module7.complex_output_0", ["qubit1:res"]),
            ("cluster0.module7.digital_output_0", ["qubit1:switch"]),
        ]
    },
}


@pytest.fixture
def setup(tmp_path):
    set_datadir(str(tmp_path))
    cluster = dry_run.dummy_cluster(HARDWARE_CFG)
    ic = InstrumentCoordinator("ic")
    ic.add_component(ClusterComponent(cluster))
    meas_ctrl = MeasurementControl("meas_ctrl")
    quantum_device = QuantumDevice("quantum_device")
    quantum_device.hardware_config(HARDWARE_CFG)
    quantum_device.instr_instrument_coordinator(ic.name)

    qubit = BasicTransmonElement("qubit1")
    quantum_device.add_element(qubit)
    qubit.clock_freqs.readout(7.145e9)
    qubit.clock_freqs.f01(4.5e9)
    qubit.rxy.amp180(0.3)
    qubit.measure.pulse_amp(0.1)
    qubit.measure.pulse_duration(2e-6)
    qubit.measure.integration_time(1.9e-6)
    qubit.measure.acq_delay(216e-9)

    nodes = CalibrationNodes(quantum_device, meas_ctrl, 216e-9, instrument_coordinator=ic)
    yield nodes, qubit, tmp_path
    Instrument.close_all()


def test_readout_calibration_on_simulated_blobs(setup):
    nodes, qubit, _ = setup
    with dry_run.simulate(dry_run.readout_data(g_centroid=0.01 + 0.01j, e_centroid=0.03 + 0.03j,
                                               measurement_noise=0.003)):
        ok, fidelity = nodes.readout_calibration(qubit, num_points=600)
    assert ok and fidelity > 0.95
    assert np.isfinite(qubit.measure.acq_threshold())
    assert nodes.history[-1]["tuids"]


def test_drag_runs_on_the_dummy_cluster_and_saves_raw_data(setup):
    nodes, qubit, datadir = setup
    motzoi_before = qubit.rxy.motzoi()
    ok, value = nodes.drag_calibration(qubit, motzoi_list=np.linspace(-0.1, 0.1, 5), repetitions=10)
    # the dummy cluster returns no real signal, so the fit result is meaningless;
    # what matters is that it ran, kept the (ok, value) contract and saved the data
    assert isinstance(ok, bool)
    if not ok:
        assert qubit.rxy.motzoi() == motzoi_before
    tuids = nodes.history[-1]["tuids"]
    assert len(tuids) == 1
    assert list(datadir.glob(f"*/{tuids[0]}*/snapshot.json"))
