"""CalibrationNodes plumbing with fake instruments: no cluster, no data directory.

Run in the qblox_dev env (never qblox_env):  python -m pytest tests/experiments
"""

import numpy as np
import pytest

from qqea.experiments.calibration_nodes import CalibrationNodes
from qqea.experiments.nodes import _base, iswap_stark, single_qubit


class FakeParam:
    def __init__(self, value=None):
        self.value = value

    def __call__(self, *args):
        if args:
            self.value = args[0]
            return None
        return self.value


class Namespace:
    def __init__(self, **params):
        for key, value in params.items():
            setattr(self, key, FakeParam(value))


class FakeQubit:
    def __init__(self, name="qubit1"):
        self.name = name
        self.rxy = Namespace(motzoi=0.03, amp180=0.4, duration=100e-9)
        self.measure = Namespace(acq_channel=0, pulse_duration=2e-6)
        self.clock_freqs = Namespace(f01=4.5e9, readout=7.1e9)


class FakeDataset:
    def __init__(self, tuid):
        self.tuid = tuid
        self.attrs = {"tuid": tuid}


class FakeMeasCtrl:
    """Records what a node asked for; ``on_run`` runs inside ``run``."""

    def __init__(self, on_run=None):
        self.verbose = FakeParam(True)
        self.on_run = on_run
        self.runs = []
        self.calls = []

    def gettables(self, g):
        self.calls.append(("gettables", g))

    def settables(self, s):
        self.calls.append(("settables", s))

    def setpoints(self, s):
        self.calls.append(("setpoints", s))

    def setpoints_grid(self, s):
        self.calls.append(("setpoints_grid", s))

    def run(self, label, **kwargs):
        if self.on_run is not None:
            self.on_run()
        self.runs.append((label, kwargs))
        return FakeDataset(f"20261009-000000-000-{len(self.runs):06d}")


class FakeDevice:
    def __init__(self):
        self.cfg_sched_repetitions = FakeParam(1)


@pytest.fixture
def fake_gettable(monkeypatch):
    made = []

    def gettable(device, **kwargs):
        made.append(kwargs)
        return kwargs

    monkeypatch.setattr(_base, "ScheduleGettable", gettable)
    return made


def make_nodes(meas_ctrl=None, **kwargs):
    return CalibrationNodes(FakeDevice(), meas_ctrl or FakeMeasCtrl(), 216e-9, **kwargs)


def test_every_old_node_is_still_a_method():
    old_nodes = [
        "resonator_calibration", "ramsey_chevron_calibration", "rabi_calibration",
        "rabi_amplification_calibration", "readout_calibration", "multiplexed_readout_calibration",
        "T1_calibration", "T2_echo_calibration", "multi_qubit_t1_and_t2", "drag_calibration",
        "ro_freq_optimization", "ro_amp_optimization", "ro_len_optimization",
        "ro_weight_calibration", "allxy_calibration", "randomized_benchmarking",
        "stark_phase_difference_calibration", "stark_phase_sum_calibration",
        "stark_phase_quadrature_calibration", "run_stark_phase_calibration", "run_roadmap",
    ]
    for name in old_nodes:
        assert callable(getattr(CalibrationNodes, name)), name


def test_ramsey_chevron_runs_with_motzoi_zero_and_restores_it(monkeypatch, fake_gettable):
    qubit = FakeQubit()
    seen = []
    meas_ctrl = FakeMeasCtrl(on_run=lambda: seen.append(qubit.rxy.motzoi()))
    nodes = make_nodes(meas_ctrl)
    monkeypatch.setattr(single_qubit, "ramsey_chevron_fit", lambda tuid, q, **kw: 4.501e9)

    ok, f01 = nodes.ramsey_chevron_calibration(qubit)

    assert (ok, f01) == (True, 4.501e9)
    assert seen == [0]
    assert qubit.rxy.motzoi() == 0.03
    assert qubit.clock_freqs.f01() == 4.501e9
    assert fake_gettable[0]["max_batch_size"] == np.arange(0, 20e-6, 0.2e-6).size
    assert [name for name, _ in meas_ctrl.calls] == ["gettables", "settables", "setpoints_grid"]


def test_ramsey_chevron_restores_motzoi_when_the_run_fails(monkeypatch, fake_gettable):
    qubit = FakeQubit()

    def fail():
        raise ConnectionError("cluster went away")

    nodes = make_nodes(FakeMeasCtrl(on_run=fail))
    with pytest.raises(ConnectionError):
        nodes.ramsey_chevron_calibration(qubit)
    assert qubit.rxy.motzoi() == 0.03
    entry = nodes.history[-1]
    assert entry["node"] == "ramsey_chevron_calibration"
    assert entry["ok"] is False and "ConnectionError" in entry["error"]


def test_allxy_restores_verbose(monkeypatch, fake_gettable):
    class Analysis:
        def __init__(self, **kwargs):
            self.quantities_of_interest = {"deviation": 0.01}

        def run(self):
            return self

    monkeypatch.setattr(single_qubit, "AllXYAnalysis", Analysis)
    verbose_during = []
    meas_ctrl = FakeMeasCtrl()
    meas_ctrl.on_run = lambda: verbose_during.append(meas_ctrl.verbose())
    nodes = make_nodes(meas_ctrl)

    assert nodes.allxy_calibration(FakeQubit()) == (True, 0.01)
    assert verbose_during == [False]
    assert meas_ctrl.verbose() is True


def test_history_records_value_and_tuids(monkeypatch, fake_gettable):
    monkeypatch.setattr(single_qubit, "ramsey_chevron_fit", lambda tuid, q, **kw: float("nan"))
    nodes = make_nodes()
    assert nodes.ramsey_chevron_calibration(FakeQubit("qubit2")) == (False, None)
    entry = nodes.history[-1]
    assert entry["qubits"] == ["qubit2"]
    assert entry["ok"] is False and entry["value"] is None
    assert entry["tuids"] == ["20261009-000000-000-000001"]


def test_drag_saves_raw_data_and_updates_motzoi(monkeypatch):
    saved = []
    monkeypatch.setattr(single_qubit, "DRAG_calibration_sched", lambda **kw: "raw")
    monkeypatch.setattr(single_qubit, "DRAG_fit", lambda ds, motzoi, plot: 0.012)

    class Raw:
        attrs = {}

        def copy(self):
            return self

    monkeypatch.setattr(single_qubit, "DRAG_calibration_sched", lambda **kw: Raw())
    monkeypatch.setattr(_base, "save_retrieve_acquisition_dataset",
                        lambda ds, name: saved.append((name, dict(ds.attrs))) or "TUID-1")
    nodes = make_nodes(instrument_coordinator=object())
    qubit = FakeQubit()

    assert nodes.drag_calibration(qubit) == (True, 0.012)
    assert qubit.rxy.motzoi() == 0.012
    assert saved[0][0] == "qubit1 DRAG calibration"
    assert len(saved[0][1]["motzoi_list"]) == 41
    assert nodes.history[-1]["tuids"] == ["TUID-1"]


def test_a_failed_save_only_warns(monkeypatch):
    class Raw:
        attrs = {}

        def copy(self):
            return self

    def broken_save(ds, name):
        raise OSError("disk full")

    monkeypatch.setattr(single_qubit, "DRAG_calibration_sched", lambda **kw: Raw())
    monkeypatch.setattr(single_qubit, "DRAG_fit", lambda ds, motzoi, plot: 0.0)
    monkeypatch.setattr(_base, "save_retrieve_acquisition_dataset", broken_save)
    nodes = make_nodes(instrument_coordinator=object())
    with pytest.warns(UserWarning, match="disk full"):
        assert nodes.drag_calibration(FakeQubit()) == (True, 0.0)


def test_drag_without_instrument_coordinator_raises():
    with pytest.raises(RuntimeError, match="instrument_coordinator"):
        make_nodes().drag_calibration(FakeQubit())


@pytest.mark.parametrize(
    "null, delta",
    [(4.0, 20.0), (-7.5, 20.0), (25.0, 20.0), (-31.0, 20.0)],
)
def test_bracket_null_recovers_the_null(null, delta):
    m_plus, m_minus = abs(delta - null), abs(-delta - null)
    assert iswap_stark.IswapStarkNodes._bracket_null(m_plus, m_minus, delta) == pytest.approx(null)


def test_bracket_null_rejects_an_inconsistent_pair():
    assert iswap_stark.IswapStarkNodes._bracket_null(5.0, 5.0, 20.0) is None


def test_run_roadmap_retries_then_stops_on_fail():
    nodes = make_nodes()
    calls = []

    def flaky(qubit, **kwargs):
        calls.append(qubit.name)
        return False, None

    nodes.flaky = flaky
    nodes.never_reached = lambda qubit: pytest.fail("ran after a failed step")
    log = nodes.run_roadmap([("Flaky", "flaky", {}), ("Next", "never_reached", {})],
                            [FakeQubit("q1"), FakeQubit("q2")], retries=1)
    assert calls == ["q1", "q1", "q2", "q2"]
    assert [(r["qubit"], r["ok"], r["attempts"]) for r in log] == [("q1", False, 2), ("q2", False, 2)]
    assert nodes.meas_ctrl.verbose() is True


def test_a_fit_that_raises_on_bad_data_is_a_failed_node(monkeypatch):
    # e.g. curve_fit on the NaN data a dummy cluster returns
    class Raw:
        attrs = {}

        def copy(self):
            return self

    def nan_fit(ds, motzoi, plot):
        raise ValueError("array must not contain infs or NaNs")

    monkeypatch.setattr(single_qubit, "DRAG_calibration_sched", lambda **kw: Raw())
    monkeypatch.setattr(single_qubit, "DRAG_fit", nan_fit)
    monkeypatch.setattr(_base, "save_retrieve_acquisition_dataset", lambda ds, name: "TUID-2")
    nodes = make_nodes(instrument_coordinator=object())
    qubit = FakeQubit()

    assert nodes.drag_calibration(qubit) == (False, None)
    assert qubit.rxy.motzoi() == 0.03
    assert nodes.history[-1]["ok"] is False and "error" not in nodes.history[-1]
