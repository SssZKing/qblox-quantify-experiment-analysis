"""Fits from qqea.fitting run on synthetic data with a known answer.

Run in the qblox_dev env (never qblox_env):  python -m pytest tests
"""

import matplotlib

matplotlib.use("Agg")  # no windows from fits that plot

import numpy as np
import pytest
import xarray as xr

from qqea.fitting import fits
from qqea.fitting.models import DoubleLorentzianModel, double_lorentzian_func
from qqea.fitting.readout import (
    IQ_blob_SNR,
    RO_amp_fit,
    rotate_real_imag,
    two_state_discriminator,
)
from qqea.fitting.single_qubit import DRAG_fit, rabi_amplification_fit, t1_and_t2_fit
from qqea.fitting.units import P2Z, Z2P, W2dBm, Vp2dBm, dBm2Vp, dBm2W

RNG_SEED = 1234


def _blobs(rng, n, separation, sigma=1.0):
    g = rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)
    e = separation + rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)
    return g, e


def test_unit_round_trips():
    assert dBm2Vp(Vp2dBm(0.1)) == pytest.approx(0.1)
    assert dBm2W(W2dBm(1e-3)) == pytest.approx(1e-3)
    assert W2dBm(1e-3) == pytest.approx(0.0)
    assert Z2P(P2Z(0.3)) == pytest.approx(0.3)


def test_rotate_real_imag_puts_signal_on_real_axis():
    signal = np.linspace(0, 1, 50)
    angle = 0.7
    z = signal * np.exp(1j * angle)
    rot_real, rot_imag, fitted_angle = rotate_real_imag(z.real, z.imag)
    assert np.allclose(rot_imag, 0, atol=1e-12)
    assert np.allclose(np.abs(rot_real), signal)


def test_two_state_discriminator_fidelity():
    rng = np.random.default_rng(RNG_SEED)
    g, e = _blobs(rng, 20000, separation=4.0)
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
        g.real, g.imag, e.real, e.imag, b_print=False, b_plot=False)
    # 4 sigma apart: each state misassigned with probability Phi(-2) = 2.3%
    assert fidelity == pytest.approx(97.7, abs=0.5)
    assert gg + ge == pytest.approx(1)
    assert eg + ee == pytest.approx(1)


def test_iq_blob_snr():
    rng = np.random.default_rng(RNG_SEED)
    g, e = _blobs(rng, 20000, separation=4.0)
    # |dZ|^2 / (2 var) with var = sigma^2 = 1 per quadrature
    assert IQ_blob_SNR(g, e) == pytest.approx(8.0, rel=0.05)


def _ro_sweep_dataset(rng, setpoints, separation_of, shots=500):
    columns = []
    for s in setpoints:
        g, e = _blobs(rng, shots, separation_of(s))
        columns += [g, e]
    return xr.Dataset({"y0": (("shot", "acq"), np.column_stack(columns))})


def test_ro_amp_fit_finds_peak():
    rng = np.random.default_rng(RNG_SEED)
    amps = np.linspace(0, 1, 21)
    ds = _ro_sweep_dataset(rng, amps, lambda a: 6 * np.exp(-((a - 0.5) / 0.15) ** 2))
    optimal = RO_amp_fit(ds, amps)
    assert optimal is not None
    assert abs(optimal - 0.5) <= 0.15  # the fidelity plateau near the top is flat


def test_ro_amp_fit_too_few_points_returns_none():
    rng = np.random.default_rng(RNG_SEED)
    amps = np.array([0.2, 0.4])
    ds = _ro_sweep_dataset(rng, amps, lambda a: 4.0)
    assert RO_amp_fit(ds, amps) is None


def _flat_dataset(x0, x1, y0, y1, **attrs):
    return xr.Dataset(
        {"y0": ("dim_0", np.asarray(y0, float)), "y1": ("dim_0", np.asarray(y1, float))},
        coords={"x0": ("dim_0", np.asarray(x0, float)), "x1": ("dim_0", np.asarray(x1, float))},
        attrs=attrs,
    )


def test_t1_and_t2_fit_single_qubit():
    rng = np.random.default_rng(RNG_SEED)
    tau = np.linspace(0, 150e-6, 40)
    t1, t2 = 30e-6, 20e-6
    x0, x1, y0 = [], [], []
    for _ in range(3):  # repeats
        for case, decay in ((1, t1), (2, t2)):
            x0 += list(tau)
            x1 += [case] * tau.size
            y0 += list(0.8 * np.exp(-tau / decay) + 0.1 + rng.normal(0, 0.005, tau.size))
    ds = _flat_dataset(x0, x1, y0, np.full(len(y0), np.nan))
    q0, q1 = t1_and_t2_fit(ds)
    assert np.allclose(q0[0], 30, rtol=0.05)  # microseconds
    assert np.allclose(q0[1], 20, rtol=0.05)
    assert q1 == [[], []]  # y1 is NaN: no second qubit


def test_rabi_amplification_fit_finds_pi_amplitude():
    rng = np.random.default_rng(RNG_SEED)
    amp_pi = 0.4
    amps = np.linspace(0.95, 1.05, 41) * amp_pi
    n_pulses = np.arange(0, 12, 2)
    x0, x1, pop = [], [], []
    for n in n_pulses:  # x1 outer, x0 inner: rows of reshape(ylen, xlen)
        x0 += list(amps)
        x1 += [n] * amps.size
        pop += list(np.sin(n * np.pi * amps / amp_pi / 2) ** 2)
    pop = np.array(pop) + rng.normal(0, 0.002, len(pop))
    ds = _flat_dataset(x0, x1, pop, 0.5 * pop, xlen=amps.size, ylen=n_pulses.size)
    result = rabi_amplification_fit(ds)
    assert result == pytest.approx(amp_pi, rel=2e-3)


def test_drag_fit_finds_zero_crossing():
    rng = np.random.default_rng(RNG_SEED)
    motzoi = np.linspace(-1, 1, 21)
    m0 = 0.23
    diff = (motzoi - m0) * (1 - 0.5j) + rng.normal(0, 0.01, motzoi.size)
    g1 = rng.normal(0, 1, motzoi.size) + 1j * rng.normal(0, 1, motzoi.size)
    g0 = g1 + diff
    interleaved = np.empty(2 * motzoi.size, complex)
    interleaved[0::2], interleaved[1::2] = g0, g1
    ds = xr.Dataset({"y0": ("acq", interleaved)})
    assert DRAG_fit(ds, list(motzoi), plot=False) == pytest.approx(m0, abs=0.02)


def test_lorentzian_width_is_hwhm():
    x0, width, area = 0.3, 0.02, 1.0
    peak = double_lorentzian_func(np.array([x0]), x0, width, area, 10.0, width, 0.0, 0.0)[0]
    half = double_lorentzian_func(np.array([x0 + width]), x0, width, area, 10.0, width, 0.0, 0.0)[0]
    assert peak == pytest.approx(area / (np.pi * width))
    assert half == pytest.approx(peak / 2)


def test_double_lorentzian_model_recovers_peaks():
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 1, 1001)
    truth = dict(x0=0.3, width=0.02, a=0.05, x0_=0.7, width_=0.03, a_=0.08, c=0.1)
    y = double_lorentzian_func(x, **truth) + rng.normal(0, 0.005, x.size)
    model = DoubleLorentzianModel()
    result = model.fit(y, params=model.guess(y, x=x), x=x)
    for name in ("x0", "x0_"):
        assert result.params[name].value == pytest.approx(truth[name], abs=1e-3)
    for name in ("width", "width_"):
        assert result.params[name].value == pytest.approx(truth[name], rel=0.05)


def test_fits_keeps_old_helpers_names():
    for name in ("np", "plt", "curve_fit", "load_dataset", "exp_decay_model", "decay_osci_model",
                 "RabiAnalysis", "ReadoutCalibrationAnalysis", "t1_and_t2_fit", "RO_freq_fit",
                 "rpe_quadrature_fit", "save_retrieve_acquisition_dataset", "BlochSpherePlot"):
        assert hasattr(fits, name), name


def test_rb_analysis_still_importable_from_schedules():
    two_qubit = pytest.importorskip("qqea.schedules.two_qubit")
    from qqea.fitting.rb import RBAnalysis

    assert two_qubit.RBAnalysis is RBAnalysis
