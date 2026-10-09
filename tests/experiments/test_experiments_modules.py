"""simulated_data, runners re-exports and the data directory helper.

Run in the qblox_dev env (never qblox_env):  python -m pytest tests/experiments
"""

import numpy as np
import pytest

from qqea.experiments import datadir, runners
from qqea.experiments.simulated_data import (
    ErrorModel,
    get_simulated_rb_data,
    get_simulated_readout_data,
)

X = np.array([[0, 1], [1, 0]], dtype=complex)
GROUND = np.array([[1, 0], [0, 0]], dtype=complex)


def test_error_model_applies_the_gate_once():
    # the old version applied the unitary four times, so X came back as identity
    rho = ErrorModel().apply(GROUND, X)
    assert np.real(rho[1, 1]) == pytest.approx(1.0)


def test_error_model_errors():
    assert np.real(ErrorModel(decay_error=0.1).apply(GROUND, X)[1, 1]) == pytest.approx(0.9)
    assert np.real(ErrorModel(depolarizing_error=0.2).apply(GROUND, X)[1, 1]) == pytest.approx(0.9)
    # a Z phase error does not change populations
    assert np.real(ErrorModel(coherent_phase_error=0.3).apply(GROUND, X)[1, 1]) == pytest.approx(1.0)


def test_error_model_reset_zeroes_every_rate():
    model = ErrorModel(0.1, 0.2, 0.3, 0.4)
    model.reset()
    assert (model.coherent_phase_error, model.incoherent_phase_error,
            model.decay_error, model.depolarizing_error) == (0, 0, 0, 0)


def test_rb_data_decays_and_keeps_the_calibration_points():
    gate_strings = [[X] * n + [X] * (n % 2) for n in (2, 20, 200)]  # identity overall
    i_data, q_data = get_simulated_rb_data(
        ErrorModel(depolarizing_error=0.01), gate_strings, lengths=[2, 20, 200, "cal0", "cal1"],
    )
    assert len(i_data) == 5
    assert i_data[0] < i_data[1] < i_data[2] < 0.5
    assert (i_data[3], q_data[3], i_data[4], q_data[4]) == (0, 0, 1, 5)


def test_rb_data_without_calibration_points_and_with_shots():
    np.random.seed(0)
    i_data, _ = get_simulated_rb_data(
        ErrorModel(), [[X], [X, X]], lengths=[1, 2], repetitions=200, num_cal_points=0,
    )
    assert list(i_data) == [1.0, 0.0]


def test_readout_data_follows_the_prepared_states():
    np.random.seed(1)
    states = np.array([0, 1] * 500)
    i_data, q_data = get_simulated_readout_data(states, g_centroid=0j, e_centroid=1 + 1j,
                                                measurement_noise=0.01)
    assert np.mean(i_data[states == 1]) == pytest.approx(1, abs=0.01)
    assert np.mean(q_data[states == 0]) == pytest.approx(0, abs=0.01)


def test_runners_moved_out_of_schedules_without_aliases():
    from qqea.schedules import single_qubit

    for name in ["readout_freq_optimization", "readout_amp_optimization",
                 "readout_len_optimization", "readout_weight_optimization",
                 "DRAG_calibration_sched"]:
        assert callable(getattr(runners, name))
        assert not hasattr(single_qubit, name)
        assert not hasattr(runners, name + "_TWPA")
        assert not hasattr(single_qubit, name + "_TWPA")


def test_set_datadir_from_env(monkeypatch, tmp_path):
    import quantify_core.data.handling as dh

    chosen = []
    monkeypatch.setattr(dh, "set_datadir", chosen.append)
    monkeypatch.setenv("QBLOX_DATADIR", str(tmp_path))
    assert datadir.set_datadir_from_env() == tmp_path
    assert chosen == [str(tmp_path)]

    monkeypatch.delenv("QBLOX_DATADIR")
    with pytest.raises(RuntimeError, match="QBLOX_DATADIR"):
        datadir.set_datadir_from_env()

    monkeypatch.setenv("QBLOX_DATADIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        datadir.set_datadir_from_env()
