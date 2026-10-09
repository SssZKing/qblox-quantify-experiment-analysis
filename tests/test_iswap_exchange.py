"""qqea.fitting.iswap_exchange on synthetic swap traces with known T1, Tphi and t_iSWAP.

Run in the qblox_dev env (never qblox_env):  python -m pytest tests
"""

import numpy as np
import pytest

from qqea.fitting.iswap_exchange import (
    NS,
    _fft_swap_time,
    US,
    IswapExchangeModel,
    iswap_analytic_error,
    iswap_exchange_fit,
    iswap_exchange_func,
    iswap_exchange_pair_fit,
    iswap_exchange_pair_func,
    iswap_exchange_screen,
    mean_times,
    t_phi,
)

T1, TPHI, T_SWAP, E0, E1 = 30 * US, 15 * US, 492 * NS, 0.03, 0.08
T = np.linspace(0, 60 * US, 601)


def _traces(noise=0.005, seed=1234):
    rng = np.random.default_rng(seed)
    donor = iswap_exchange_func(T, T1, TPHI, T_SWAP, E0, E1)
    acceptor = iswap_exchange_func(T, T1, TPHI, T_SWAP, E0, E1, acceptor=True)
    return donor + rng.normal(0, noise, T.size), acceptor + rng.normal(0, noise, T.size)


def test_pair_func_without_detuning_matches_single_trace_func():
    donor, acceptor = iswap_exchange_pair_func(T, T1, TPHI, T_SWAP, 0.0, E0, E1, E0, E1)
    assert np.allclose(donor, iswap_exchange_func(T, T1, TPHI, T_SWAP, E0, E1))
    assert np.allclose(acceptor, iswap_exchange_func(T, T1, TPHI, T_SWAP, E0, E1, acceptor=True))


@pytest.mark.parametrize("tphi, noise", [(15 * US, 0.005), (2 * US, 0.03)])
def test_fft_guess_finds_swap_time_even_when_swap_dies_early(tphi, noise):
    rng = np.random.default_rng(7)
    donor = iswap_exchange_func(T, T1, tphi, T_SWAP, E0, E1) + rng.normal(0, noise, T.size)
    acceptor = iswap_exchange_func(T, T1, tphi, T_SWAP, E0, E1, acceptor=True) + rng.normal(0, noise, T.size)
    assert _fft_swap_time(T, donor) == pytest.approx(T_SWAP, rel=0.05)
    assert _fft_swap_time(T, donor - acceptor) == pytest.approx(T_SWAP, rel=0.05)


def test_iswap_exchange_fit_recovers_rates():
    donor, _ = _traces()
    _, summary = iswap_exchange_fit(T, donor)  # t_iSWAP guessed from the FFT
    assert summary["t1"] == pytest.approx(T1, rel=0.1)
    assert summary["tphi"] == pytest.approx(TPHI, rel=0.15)
    assert summary["iswap_time"] == pytest.approx(T_SWAP, rel=0.01)
    assert summary["gate_error"] == pytest.approx(0.8 * T_SWAP * (1 / T1 + 1 / TPHI), rel=0.15)


def test_iswap_exchange_model_guess_and_fit():
    donor, _ = _traces()
    model = IswapExchangeModel()
    result = model.fit(donor, params=model.guess(donor, t=T), t=T)
    assert result.params["iswap_time"].value == pytest.approx(T_SWAP, rel=0.01)
    assert 1 / result.params["gamma_1"].value == pytest.approx(T1, rel=0.1)


def test_iswap_exchange_pair_fit_recovers_rates():
    donor, acceptor = _traces()
    _, summary = iswap_exchange_pair_fit(T, donor, acceptor, iswap_time_guess=T_SWAP, delta=0.0)
    assert summary["t1"] == pytest.approx(T1, rel=0.1)
    assert summary["tphi"] == pytest.approx(TPHI, rel=0.15)
    assert summary["iswap_time"] == pytest.approx(T_SWAP, rel=0.01)
    assert summary["e0_d"] == pytest.approx(E0, abs=0.02)


def test_screen_rejects_a_trace_that_does_not_swap():
    donor, acceptor = _traces()
    flat = np.full(T.size, 0.05)
    ok, _ = iswap_exchange_screen(T[None, :], np.stack([donor, flat])[:, None, :],
                                  np.stack([acceptor, flat])[:, None, :])
    assert ok.ravel().tolist() == [True, False]


def test_gate_error_formulas():
    t2 = 40 * US
    tphi = t_phi(T1, t2)
    assert t_phi(T1, 2 * T1) == np.inf
    assert iswap_analytic_error(T1, t2, T_SWAP) == pytest.approx(0.8 * T_SWAP * (1 / T1 + 1 / tphi))
    t1_pair, _ = mean_times((20 * US, 60 * US), (30 * US, 60 * US))
    assert t1_pair == pytest.approx(30 * US)  # harmonic mean of 20 and 60 us


def test_model_listed_with_other_models():
    from qqea.fitting import models

    assert models.IswapExchangeModel is IswapExchangeModel
