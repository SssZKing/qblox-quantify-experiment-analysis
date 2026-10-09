"""Incoherent (T1, T2) error of the iSWAP gate: QuTiP time evolution vs Appendix L.

Two qubits on resonance in the rotating frame, exchange coupling g set by the
iSWAP time (g = pi / (2 * t_iSWAP)), with amplitude damping and white pure
dephasing on each qubit and no flux noise:

    H = g * (a_1^dag a_2 + a_2^dag a_1)
    c_ops = sqrt(1/T1_q) a_q,  sqrt(2/Tphi_q) n_q,    1/T2 = 1/(2 T1) + 1/Tphi

The dephasing operator sqrt(2/Tphi) n decays a qubit's coherence as exp(-t/Tphi),
the same convention as p_phi = (1 - exp(-t/Tphi)) / 2 in Li et al., PRX 14,
041050 (2024), Appendix L.

Two things are computed:

  1. the population exchange curve - one excitation starting in qubit 1,
     P_1(t) and P_2(t) - for a set of (T1, T2);
  2. the average gate fidelity of the iSWAP from the full 16x16 propagator at
     t = t_iSWAP, compared with Appendix L's relaxation (L8) and pure-dephasing
     (L13) terms summed over both qubits - the only channels considered for now:

         1 - F ~ 2/5 * t * sum_q (1/T1_q + 1/Tphi_q)

     and with an idle of the same length, which to first order has the same
     error (the first-order infidelity of a Lindblad term, (d|L|^2 - |Tr L|^2) /
     (d(d+1)) dt, is unchanged by L -> U^dag L U).
  3. a closed form of the exchange curve in T1 and Tphi (`exchange_model`) and a
     fit of it to a measured or simulated trace (`fit_exchange`).

Run: python incoherent_error/iswap_incoherent.py   (inside the qblox_env conda env)
"""

import os

import lmfit
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

US = 1e-6
NS = 1e-9
T_ISWAP = 492 * NS
HERE = os.path.dirname(os.path.abspath(__file__))

_A1 = qt.tensor(qt.destroy(2), qt.qeye(2))
_A2 = qt.tensor(qt.qeye(2), qt.destroy(2))
_N1 = _A1.dag() * _A1
_N2 = _A2.dag() * _A2


def t_phi(t1, t2):
    """Pure-dephasing time from T1 and T2; np.inf when T2 = 2 T1."""
    rate = 1 / t2 - 1 / (2 * t1)
    if rate < -1e-12 / t2:
        raise ValueError("T2 = %.3g must not exceed 2 T1 = %.3g" % (t2, 2 * t1))
    return np.inf if rate <= 0 else 1 / rate


def _pair(x):
    return (x, x) if np.isscalar(x) else tuple(x)


def hamiltonian(iswap_time=T_ISWAP):
    g = np.pi / (2 * iswap_time)
    return g * (_A1.dag() * _A2 + _A2.dag() * _A1)


def collapse_ops(t1, t2):
    """T1 and T2 per qubit; a scalar applies to both qubits."""
    ops = []
    for a, n, t1_q, t2_q in zip((_A1, _A2), (_N1, _N2), _pair(t1), _pair(t2)):
        ops.append(np.sqrt(1 / t1_q) * a)
        tphi_q = t_phi(t1_q, t2_q)
        if np.isfinite(tphi_q):
            ops.append(np.sqrt(2 / tphi_q) * n)
    return ops


def exchange_curve(t, t1, t2, iswap_time=T_ISWAP):
    """Excited-state populations (P_1, P_2) for |10> evolving under the exchange."""
    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
    result = qt.mesolve(hamiltonian(iswap_time), psi0, t, collapse_ops(t1, t2), e_ops=[_N1, _N2])
    return result.expect[0], result.expect[1]


def exchange_envelope(t_max, t1, t2, iswap_time=T_ISWAP):
    """Top and bottom envelope of P_1: its value at even / odd multiples of t_iSWAP.

    On resonance those are the swap maxima and minima, so no peak finding; pure
    dephasing shifts the true extrema slightly, which changes the values by < 4e-4
    for T2 >= 10 us at t_iSWAP = 492 ns. Returns (t_top, top, t_bottom, bottom).
    """
    t = np.arange(int(t_max / iswap_time) + 1) * iswap_time
    p1, _ = exchange_curve(t, t1, t2, iswap_time)
    return t[::2], p1[::2], t[1::2], p1[1::2]


def mean_times(t1, t2):
    """(T1, Tphi) of the pair as seen by `exchange_model`.

    The model depends on the arithmetic mean of the two qubits' *rates*, so the
    pair times are the harmonic means of the two qubits' times:

        1/T1 = (1/T1_1 + 1/T1_2) / 2,    1/Tphi = (1/Tphi_1 + 1/Tphi_2) / 2
    """
    gamma_1 = np.mean([1 / x for x in _pair(t1)])
    gamma_phi = np.mean([1 / t_phi(a, b) for a, b in zip(_pair(t1), _pair(t2))])
    return 1 / gamma_1, (1 / gamma_phi if gamma_phi > 0 else np.inf)


def _z(t, gamma_phi, iswap_time):
    """Damped pseudo-spin z(t) = P_10 - P_01, with the exp(-t/T1) factored out."""
    w0 = np.pi / iswap_time
    w = np.sqrt(np.maximum(w0**2 - gamma_phi**2, 1e-30))
    return np.exp(-gamma_phi * t) * (np.cos(w * t) + gamma_phi / w * np.sin(w * t))


def _donor_rates(t, gamma_1, gamma_phi, iswap_time, e0, e1):
    """Donor trace in rates: gamma_1 = 1/T1, gamma_phi = 1/Tphi (pair values)."""
    return e0 + (1 - e0 - e1) * 0.5 * np.exp(-gamma_1 * t) * (1 + _z(t, gamma_phi, iswap_time))


def _acceptor_rates(t, gamma_1, gamma_phi, iswap_time, e0, e1):
    """Acceptor trace in rates: gamma_1 = 1/T1, gamma_phi = 1/Tphi (pair values)."""
    return e0 + (1 - e0 - e1) * 0.5 * np.exp(-gamma_1 * t) * (1 - _z(t, gamma_phi, iswap_time))


def exchange_model(t, t1, tphi, iswap_time=T_ISWAP, e0=0.0, e1=0.0, acceptor=False):
    """Closed form of the readout trace of the donor (the qubit prepared in |1>).

    With `acceptor=True`, the trace of the other qubit (prepared in |0>).
    The derivation below calls the donor qubit 1 and the start state |10>.

    In the one-excitation manifold {|10>, |01>} write P = P_10 + P_01 and the
    pseudo-spin z = P_10 - P_01, y = 2 Im rho_{10,01}. Relaxation (either qubit)
    empties the manifold into |00> and never feeds back; pure dephasing only
    decays the coherence y; the exchange rotates z into y at Omega = 2g = pi/t_iSWAP:

        dP/dt = -gamma_1 P
        dz/dt = -gamma_1 z - Omega y
        dy/dt = -(gamma_1 + 2 gamma_phi) y + Omega z

    with gamma_1 = 1/T1 and gamma_phi = 1/Tphi (per qubit; the |10>-|01> coherence
    decays at 1/Tphi_1 + 1/Tphi_2 = 2 gamma_phi). With z(0) = 1, y(0) = 0 this is a
    damped oscillator, and the donor and acceptor populations are (P +- z) / 2:

        P_donor(t)    = 1/2 exp(-t/T1) [1 + exp(-t/Tphi) (cos(w t) + sin(w t) / (w Tphi))]
        P_acceptor(t) = 1/2 exp(-t/T1) [1 - exp(-t/Tphi) (cos(w t) + sin(w t) / (w Tphi))]
        w = sqrt((pi/t_iSWAP)^2 - 1/Tphi^2)

    Readout: y(t) = e0 + (1 - e0 - e1) P(t), e0 = P(read 1 | 0), e1 = P(read 0 | 1)
    of the qubit being read.

    T1 moves only the midline and envelope together (exp(-t/T1)); Tphi only damps
    the oscillation about the midline - so a fit separates them. For unequal
    qubits, T1 and Tphi here are the pair values of `mean_times`: the arithmetic
    mean of the two qubits' rates, i.e. the harmonic mean of their times,
    1/T1 = (1/T1_1 + 1/T1_2) / 2 and 1/Tphi = (1/Tphi_1 + 1/Tphi_2) / 2. Unequal
    T1 adds a term of relative size |1/T1_1 - 1/T1_2| t_iSWAP / (2 pi), e.g. 3e-3
    for 40 vs 15 us, and a swap trace cannot tell which qubit has which rate.
    `tphi` may be np.inf (no pure dephasing).
    """
    f = _acceptor_rates if acceptor else _donor_rates
    return f(np.asarray(t, float), 1 / t1, 1 / tphi, iswap_time, e0, e1)


def fit_exchange(t, y, t1_guess=30 * US, tphi_guess=50 * US, iswap_time_guess=T_ISWAP,
                 e0_guess=0.02, e1_guess=0.1, e0=None, e1=None, acceptor=False):
    """Fit `exchange_model` to a donor (or, `acceptor=True`, acceptor) trace.

    Fits the rates, so Tphi = inf is allowed. Passing `e0` / `e1` holds that
    readout error fixed (e.g. at the single-shot calibration, e0 = 1 - F_g,
    e1 = 1 - F_e); otherwise it is fitted, starting from `e0_guess` / `e1_guess`,
    and then also absorbs state-preparation error and thermal population.

    Returns (lmfit result, summary dict). The summary has the pair T1, Tphi (the
    harmonic means of the two qubits' times, see `mean_times`; with
    standard errors where the fit gives them), t_iSWAP, e0, e1, and the (L8)+(L13)
    linear gate error these imply,

        1 - F ~ 2/5 t_iSWAP sum_q (1/T1_q + 1/Tphi_q) = 4/5 t_iSWAP (1/T1 + 1/Tphi),

    which assumes the dephasing noise on the two qubits is independent: noise
    common to both cancels in the exchange (it only sees the frequency difference)
    but still costs gate fidelity.
    """
    model = lmfit.Model(_acceptor_rates if acceptor else _donor_rates)
    params = model.make_params(gamma_1=1 / t1_guess, gamma_phi=1 / tphi_guess,
                               iswap_time=iswap_time_guess, e0=e0_guess, e1=e1_guess)
    params["gamma_1"].set(min=0)
    params["gamma_phi"].set(min=0)
    params["iswap_time"].set(min=0.5 * iswap_time_guess, max=2 * iswap_time_guess)
    for name, fixed in (("e0", e0), ("e1", e1)):
        if fixed is None:
            params[name].set(min=0, max=0.5)
        else:
            params[name].set(value=fixed, vary=False)
    result = model.fit(np.asarray(y, float), params, t=np.asarray(t, float))

    p = result.params

    def inverse(name):
        rate, err = p[name].value, p[name].stderr
        time = 1 / rate if rate > 0 else np.inf
        return time, (err / rate**2 if (err is not None and rate > 0) else np.nan)

    t1, t1_err = inverse("gamma_1")
    tphi, tphi_err = inverse("gamma_phi")
    iswap_time = p["iswap_time"].value
    summary = {
        "t1": t1, "t1_err": t1_err,
        "tphi": tphi, "tphi_err": tphi_err,
        "gamma_phi": p["gamma_phi"].value, "gamma_phi_err": p["gamma_phi"].stderr,
        "iswap_time": iswap_time, "e0": p["e0"].value, "e1": p["e1"].value,
        "gate_error": 0.8 * iswap_time * (p["gamma_1"].value + p["gamma_phi"].value),
    }
    return result, summary


def _z_detuned(t, gamma_phi, omega, delta):
    """Pseudo-spin z(t) with detuning, relaxation factored out (z(0) = 1).

    Bloch equations of H = delta/2 (n_d - n_a) + g (a_d^dag a_a + h.c.), with
    Omega = 2g and the |10>-|01> coherence (x, y) damped at 2 gamma_phi:

        dx/dt = -2 gamma_phi x - delta y
        dy/dt = delta x - 2 gamma_phi y - Omega z
        dz/dt = Omega y

    Solved through the eigen-decomposition of the 3x3 generator. For delta = 0 it
    reduces to the damped oscillator of `exchange_model`.
    """
    g2 = 2 * gamma_phi
    m = np.array([[-g2, -delta, 0.0], [delta, -g2, -omega], [0.0, omega, 0.0]])
    lam, vec = np.linalg.eig(m)
    coef = np.linalg.solve(vec, np.array([0.0, 0.0, 1.0]))
    return np.real(np.exp(np.outer(np.asarray(t, float), lam)) @ (vec[2] * coef))


def exchange_model_pair(t, t1, tphi, iswap_time, delta=0.0, e0_d=0.0, e1_d=0.0, e0_a=0.0, e1_a=0.0):
    """Donor and acceptor readout traces with a detuning `delta` (rad/s) from the resonance.

    `iswap_time` is the resonant swap time pi/Omega; with detuning the swap runs at
    sqrt(Omega^2 + delta^2) and transfers only Omega^2 / (Omega^2 + delta^2) of the
    excitation. T1, Tphi are the pair values of `mean_times`. Returns (y_donor, y_acceptor).
    """
    z = _z_detuned(t, 1 / tphi, np.pi / iswap_time, delta)
    envelope = 0.5 * np.exp(-np.asarray(t, float) / t1)
    return (e0_d + (1 - e0_d - e1_d) * envelope * (1 + z),
            e0_a + (1 - e0_a - e1_a) * envelope * (1 - z))


def fit_exchange_pair(t, y_donor, y_acceptor, t1_guess=None, tphi_guess=None, iswap_time_guess=None,
                      readout=None, delta=None):
    """Joint fit of donor and acceptor traces with `exchange_model_pair`.

    Shared: 1/T1, 1/Tphi, Omega = pi/t_iSWAP and the detuning |delta|; each qubit
    has its own readout errors e0, e1 (fitted - they also absorb preparation error
    and any population lost when the pump turns on). The swap frequency is guessed
    from the FFT of donor - acceptor, which has no slow background; the fit is
    started at two detunings and the better result kept (the cost is flat in delta
    at delta = 0).

    `readout` fixes any of e0_d, e1_d, e0_a, e1_a to a given value, e.g.
    {"e0_d": 0.03, "e0_a": 0.03} for the calibrated ground-state errors; the rest stay free.

    `delta` (rad/s) fixes the detuning instead of fitting it, e.g. delta=0 when the pump
    sits on the chevron resonance. A swap trace hardly constrains delta: on the 2026-10-03
    sweeps the free fit returned 0 - 400 kHz between repetitions at one power (chevrons:
    resonance known to +-20 kHz) with chi2 within 1 % of delta = 0, and the scatter went
    into Tphi (bumps at 13 and 18 dBm). None (default) fits it from two starts.

    Returns (lmfit MinimizerResult, summary dict) with t1, tphi, iswap_time
    (resonant), delta (rad/s), their standard errors, the readout errors, redchi and
    the (L8)+(L13) gate error 4/5 t_iSWAP (1/T1 + 1/Tphi).
    """
    t = np.asarray(t, float)
    y_donor, y_acceptor = np.asarray(y_donor, float), np.asarray(y_acceptor, float)
    diff = (y_donor - y_acceptor) - np.mean(y_donor - y_acceptor)
    n_fft = 16 * t.size
    spectrum = np.abs(np.fft.rfft(diff * np.hanning(t.size), n_fft))
    freqs = np.fft.rfftfreq(n_fft, t[1] - t[0])
    omega_eff = 2 * np.pi * freqs[1 + np.argmax(spectrum[1:])]
    if iswap_time_guess is not None:
        omega_eff = np.pi / iswap_time_guess
    t1_guess = t[-1] / 2 if t1_guess is None else t1_guess
    tphi_guess = t[-1] if tphi_guess is None else tphi_guess

    def residual(params):
        v = params.valuesdict()
        z = _z_detuned(t, v["gamma_phi"], v["omega"], v["delta"])
        envelope = 0.5 * np.exp(-v["gamma_1"] * t)
        model_d = v["e0_d"] + (1 - v["e0_d"] - v["e1_d"]) * envelope * (1 + z)
        model_a = v["e0_a"] + (1 - v["e0_a"] - v["e1_a"]) * envelope * (1 - z)
        return np.concatenate([model_d - y_donor, model_a - y_acceptor])

    best = None
    for delta_frac in ((0.05, 0.4) if delta is None else (0.0,)):
        params = lmfit.Parameters()
        params.add("gamma_1", 1 / t1_guess, min=0)
        params.add("gamma_phi", 1 / tphi_guess, min=0)
        params.add("omega", omega_eff * np.sqrt(1 - delta_frac**2), min=0.3 * omega_eff, max=1.5 * omega_eff)
        if delta is None:
            params.add("delta", delta_frac * omega_eff, min=0, max=omega_eff)
        else:
            params.add("delta", float(delta), vary=False)
        for name in ("e0_d", "e1_d", "e0_a", "e1_a"):
            if readout is not None and name in readout:
                params.add(name, readout[name], vary=False)
            else:
                params.add(name, 0.05, min=0, max=0.5)
        result = lmfit.minimize(residual, params)
        if best is None or result.chisqr < best.chisqr:
            best = result

    p = best.params

    def err(name):
        return p[name].stderr if p[name].stderr is not None else np.nan

    g1, gphi, omega = p["gamma_1"].value, p["gamma_phi"].value, p["omega"].value
    summary = {
        "t1": 1 / g1, "t1_err": err("gamma_1") / g1**2,
        "tphi": 1 / gphi if gphi > 0 else np.inf,
        "tphi_err": err("gamma_phi") / gphi**2 if gphi > 0 else np.nan,
        "gamma_1": g1, "gamma_phi": gphi,
        "iswap_time": np.pi / omega, "iswap_time_err": np.pi * err("omega") / omega**2,
        "delta": p["delta"].value, "delta_err": err("delta"),
        "e0_d": p["e0_d"].value, "e1_d": p["e1_d"].value,
        "e0_a": p["e0_a"].value, "e1_a": p["e1_a"].value,
        "redchi": best.redchi,
        "gate_error": 0.8 * (np.pi / omega) * (g1 + gphi),
    }
    return best, summary


# ----------------------------------------------------------------------------- power sweeps
TRACE_KEYS = ("gamma_1", "gamma_phi", "iswap_time", "delta", "redchi",
              "e0_d", "e1_d", "e0_a", "e1_a", "t1_err", "tphi_err", "gate_error")


def screen_traces(t, donor, acceptor, prep_min=0.5, acceptor_max=0.3, corr_min=0.5, env_min=0.1):
    """Model-free check of every trace before fitting.

    `donor`, `acceptor` have shape (repetition, power, duration), `t` (power, duration).
    A trace passes when
      * the preparation worked: donor(t=0) > prep_min and acceptor(t=0) < acceptor_max
        (100 shots: a good start reads 0.75 - 0.9 and 0 - 0.05);
      * it swaps: donor - acceptor correlates with the median of all repetitions at that
        power by more than corr_min, over the durations where the median still oscillates
        (smoothed |median| > env_min). The median is not pulled by a few failed runs.
    Returns (ok, metrics) with ok of shape (repetition, power) and metrics holding
    donor0, acceptor0 and corr of the same shape.
    """
    diff = donor - acceptor
    median = np.median(diff, axis=0)
    corr = np.zeros(diff.shape[:2])
    for k in range(diff.shape[1]):
        envelope = np.convolve(np.abs(median[k]), np.ones(9) / 9, "same")
        window = envelope > env_min
        for r in range(diff.shape[0]):
            corr[r, k] = np.corrcoef(diff[r, k][window], median[k][window])[0, 1]
    metrics = {"donor0": donor[:, :, 0], "acceptor0": acceptor[:, :, 0], "corr": corr}
    ok = (metrics["donor0"] > prep_min) & (metrics["acceptor0"] < acceptor_max) & (corr > corr_min)
    return ok, metrics


def _fit_trace_job(args):
    t, y_donor, y_acceptor, iswap_time_guess, readout, delta = args
    try:
        _, f = fit_exchange_pair(t, y_donor, y_acceptor, iswap_time_guess=iswap_time_guess, readout=readout,
                                 delta=delta)
        return [f[key] for key in TRACE_KEYS]
    except Exception:
        return [np.nan] * len(TRACE_KEYS)


def fit_traces(t, donor, acceptor, ok, iswap_time_guess, workers=8, readout=None, delta=None):
    """`fit_exchange_pair` on every trace with ok = True, in parallel processes.

    The swap frequency is seeded with `iswap_time_guess` (one per power, e.g. the chevron
    t_swap): at high power the swap dies within ~5 % of the window and an FFT guess locks
    onto the slow decay. `readout` fixes readout errors: {name: array (repetition,)} with a
    value per repetition (e.g. from the readout calibration before it) for any of e0_d, e1_d,
    e0_a, e1_a. `delta` is passed to `fit_exchange_pair` (None: fitted; a number: fixed,
    rad/s). Returns {key: array (repetition, power)} for TRACE_KEYS, NaN where ok is False
    or the fit raised.
    """
    from concurrent.futures import ProcessPoolExecutor

    index = [(r, k) for r in range(donor.shape[0]) for k in range(donor.shape[1]) if ok[r, k]]
    jobs = [(t[k], donor[r, k], acceptor[r, k], iswap_time_guess[k],
             None if readout is None else {name: float(v[r]) for name, v in readout.items()}, delta)
            for r, k in index]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_fit_trace_job, jobs, chunksize=10))
    else:
        rows = [_fit_trace_job(job) for job in jobs]
    out = {key: np.full(donor.shape[:2], np.nan) for key in TRACE_KEYS}
    for (r, k), row in zip(index, rows):
        for key, value in zip(TRACE_KEYS, row):
            out[key][r, k] = value
    return out


def readout_fidelities(ds):
    """(F_g, F_e) per channel of a Multiplexed Readout Calibration dataset, from the raw IQ.

    Each channel's shots are projected onto the line through the |0> and |1> means and
    the threshold that maximises F_g + F_e is used (the quantify analysis folder of this
    run holds only the last channel's result, since the second analysis overwrites it).
    """
    states = ds.x0.values
    out = []
    for i in range(len(ds.data_vars) // 2):
        z = ds["y%d" % (2 * i)].values + 1j * ds["y%d" % (2 * i + 1)].values
        m0, m1 = z[states == 0].mean(), z[states == 1].mean()
        proj = np.real((z - m0) * np.conj(m1 - m0)) / abs(m1 - m0)
        p0, p1 = np.sort(proj[states == 0]), np.sort(proj[states == 1])
        cut = np.sort(proj)
        f_g = np.searchsorted(p0, cut, side="left") / p0.size
        f_e = 1 - np.searchsorted(p1, cut, side="left") / p1.size
        best = np.argmax(f_g + f_e)
        out.append((f_g[best], f_e[best]))
    return out


def fit_chevron(t, freqs, y_donor, y_acceptor, iswap_time_guess, f0_guess=None):
    """Fit one pump-power slice of a chevron with the detuned model of `exchange_model_pair`.

    `y_donor`, `y_acceptor` have shape (frequency, duration). Shared over the whole slice:
    the resonance f0 (the detuning at pump frequency f is 2 pi (f - f0)), Omega, 1/T1,
    1/Tphi and the four readout errors. Returns a dict with f0, f0_err (Hz), iswap_time,
    tphi, t1 and redchi.
    """
    t = np.asarray(t, float)

    def residual(params):
        v = params.valuesdict()
        envelope = 0.5 * np.exp(-v["gamma_1"] * t)
        out = []
        for f, yd, ya in zip(freqs, y_donor, y_acceptor):
            z = _z_detuned(t, v["gamma_phi"], v["omega"], 2 * np.pi * (f - v["f0"]))
            out.append(v["e0_d"] + (1 - v["e0_d"] - v["e1_d"]) * envelope * (1 + z) - yd)
            out.append(v["e0_a"] + (1 - v["e0_a"] - v["e1_a"]) * envelope * (1 - z) - ya)
        return np.concatenate(out)

    params = lmfit.Parameters()
    params.add("f0", freqs[len(freqs) // 2] if f0_guess is None else f0_guess)
    params.add("omega", np.pi / iswap_time_guess, min=0)
    params.add("gamma_1", 1e5, min=0)
    params.add("gamma_phi", 1e6, min=0)
    for name in ("e0_d", "e1_d", "e0_a", "e1_a"):
        params.add(name, 0.05, min=0, max=0.5)
    result = lmfit.minimize(residual, params)
    p = result.params
    return {"f0": p["f0"].value, "f0_err": p["f0"].stderr if p["f0"].stderr is not None else np.nan,
            "iswap_time": np.pi / p["omega"].value, "t1": 1 / p["gamma_1"].value,
            "tphi": 1 / p["gamma_phi"].value if p["gamma_phi"].value > 0 else np.inf,
            "redchi": result.redchi}


def windowed_swap_frequency(t, diff, iswap_time, periods=3, min_amplitude=0.03):
    """Swap frequency in consecutive windows of `periods` swap periods (2 t_iSWAP each).

    Fits a damped cosine to donor - acceptor in each window and stops once the oscillation
    amplitude falls below `min_amplitude`. A constant result means a fixed exchange rate and
    resonance during the pump; a drift shows them changing while the pump is on.
    Returns (window centres, frequencies) in s and Hz.
    """
    from scipy.optimize import curve_fit

    def damped_cos(x, a, f, phase, gamma, c):
        return a * np.exp(-gamma * x) * np.cos(2 * np.pi * f * x + phase) + c

    width = periods * 2 * iswap_time
    centres, freqs = [], []
    for j in range(int(t[-1] / width) + 1):
        m = (t >= j * width) & (t < (j + 1) * width)
        if m.sum() < 10:
            break
        try:
            p, _ = curve_fit(damped_cos, t[m] - j * width, diff[m],
                             p0=(0.3, 1 / (2 * iswap_time), 0, 1e5, 0), maxfev=20000)
        except RuntimeError:
            break
        if abs(p[0]) * np.exp(-p[3] * width / 2) < min_amplitude:
            break
        centres.append((j + 0.5) * width)
        freqs.append(abs(p[1]))
    return np.array(centres), np.array(freqs)


def fit_growing_loss(t, y, e0, t_min=0.6 * US):
    """Single-qubit decay under the pump with a loss that builds up while the pump is on.

        y = e0 + a exp(-t/T1 - (t/tau_h)^2),   fitted for t >= t_min

    i.e. a relaxation rate 1/T1 + 2 t / tau_h^2 that grows linearly with pump duration
    (as a slowly heating bath would give); tau_h = inf means no build-up. The floor e0 is
    fixed (e.g. the calibrated 1 - F_g). Returns t1, tau_h and the ratio of the chi2 of a
    plain exponential to that of this model (> 1: the build-up term is needed).
    """
    from scipy.optimize import curve_fit

    m = np.asarray(t) >= t_min
    x, v = np.asarray(t)[m], np.asarray(y)[m]

    def grow(x, a, g1, gh):
        return e0 + a * np.exp(-g1 * x - (gh * x) ** 2)

    def plain(x, a, g1):
        return e0 + a * np.exp(-g1 * x)

    p, _ = curve_fit(grow, x, v, p0=(0.85, 1 / (x[-1] / 3), 1 / x[-1]),
                     bounds=([0, 0, 0], [2, 1e8, 1e8]), maxfev=20000)
    q, _ = curve_fit(plain, x, v, p0=(0.85, 1 / (x[-1] / 3)), maxfev=20000)
    ratio = np.sum((plain(x, *q) - v) ** 2) / np.sum((grow(x, *p) - v) ** 2)
    return {"t1": 1 / p[1] if p[1] > 0 else np.inf,
            "tau_h": 1 / p[2] if p[2] > 1 else np.inf,          # < 1 /s: no measurable build-up
            "chi2_ratio": ratio}


# ----------------------------------------------------------------------------- flux noise (Appendix L1)
def echo_gaussian_profile(tau, y, sigma, grid):
    """Profile chi2 of the Gaussian (1/f) echo rate, Li et al. Appendix L1:

        P(t) = A + B exp(-Gamma_exp t - (Gamma_phi^E t)^2)

    For each Gamma_phi^E on `grid` (same time unit as `tau`, inverse) A, B, Gamma_exp are
    refitted. `sigma` is the per-point noise. Returns (best, upper95, reduced chi2): the
    one-sided 95 % bound is where chi2 - chi2_min exceeds 2.71, with chi2 rescaled by the
    reduced chi2 when that is above 1.
    """
    from scipy.optimize import curve_fit

    chi = []
    for gp in grid:
        def f(t, a, b, ge):
            return a + b * np.exp(-ge * t - (gp * t) ** 2)
        p, _ = curve_fit(f, tau, y, p0=(np.mean(y[-5:]), y[0] - np.mean(y[-5:]), 1 / tau[-1] * 3),
                         sigma=sigma, maxfev=20000)
        chi.append(np.sum(((f(tau, *p) - y) / sigma) ** 2))
    chi = np.array(chi)
    best = grid[np.argmin(chi)]
    red = chi.min() / (tau.size - 4)
    dchi = (chi - chi.min()) / max(red, 1.0)
    above = (grid >= best) & (dchi > 2.71)
    return best, (grid[np.argmax(above)] if above.any() else np.nan), red


def flux_noise_iswap_error(sqrt_a, dfdphi, iswap_time=T_ISWAP, t_meas=3600.0, n=4000, seed=0):
    """iSWAP error from quasi-static 1/f flux noise shared by both qubits (Appendix L1 for an iSWAP).

    S_Phi = A / |omega| with sqrt(A) = `sqrt_a` (in Phi0). Over a gate of length t the noise
    below ~1/t is static; averaged over a measurement of length `t_meas` it gives a flux offset
    with variance 2 A ln(1 / (omega_ir t)), omega_ir = 2 pi / t_meas -- the variance for which a
    free-induction decay is exp(-(Gamma_R t)^2), Gamma_R = sqrt(A |ln(omega_ir t)|) |d omega/d Phi|.
    The same offset dPhi shifts both qubits (one SNAIL loop), by 2 pi dfdphi[q] dPhi, with
    `dfdphi` in Hz per Phi0. Each sample gives the unitary of
        H = sum_q delta_q n_q + g (a1^dag a2 + h.c.),  g = pi / (2 t_iSWAP),
    and the error is 1 - F_ave against the ideal iSWAP, averaged over `n` samples.
    Returns (mean error, rms flux offset in Phi0).
    """
    from scipy.linalg import expm

    rng = np.random.default_rng(seed)
    big_l = abs(np.log(2 * np.pi / t_meas * iswap_time))
    sigma_phi = np.sqrt(2 * sqrt_a**2 * big_l)
    g = np.pi / (2 * iswap_time)
    n1, n2 = np.diag([0, 0, 1, 1]), np.diag([0, 1, 0, 1])          # |q1 q2>: 00, 01, 10, 11
    hop = np.zeros((4, 4)); hop[1, 2] = hop[2, 1] = 1
    ideal = expm(-1j * g * hop * iswap_time)
    err = []
    for dphi in rng.normal(0, sigma_phi, n):
        d1, d2 = (2 * np.pi * f * dphi for f in dfdphi)
        u = expm(-1j * (d1 * n1 + d2 * n2 + g * hop) * iswap_time)
        err.append(1 - (abs(np.trace(ideal.conj().T @ u)) ** 2 + 4) / 20)
    return float(np.mean(err)), sigma_phi


def gate_error(t1, t2, iswap_time=T_ISWAP, idle=False):
    """1 - average gate fidelity of the iSWAP (or an idle of the same length)."""
    h = 0 * _N1 if idle else hamiltonian(iswap_time)
    noisy = qt.propagator(h, iswap_time, collapse_ops(t1, t2))
    ideal = qt.to_super((-1j * h * iswap_time).expm())
    return 1 - qt.average_gate_fidelity(ideal.dag() * noisy)


def channel_errors(t1, t2, t=T_ISWAP):
    """Per-qubit Kraus probabilities and the (L8), (L13) errors they give.

        p1    = 1 - exp(-t/T1)            (L8):   F = (3 - p1 + 2 sqrt(1 - p1)) / 5
        p_phi = (1 - exp(-t/Tphi)) / 2    (L13):  F = 1 - 4 p_phi / 5

    and the error of each channel is 1 - F.

    TYPO IN THE PAPER, Eq. (L13): it is printed as

        1 - F = 1 - 4 p_phi / 5              (as printed - wrong)

    but 1 - 4 p_phi / 5 is F itself, so the infidelity is

        1 - F = 1 - (1 - 4 p_phi / 5) = 4 p_phi / 5      (correct)

    Check: (L2) with the Kraus operators (L9)-(L10) gives
    F = [4 + 16 (1 - p_phi)] / 20 = 1 - 4 p_phi / 5, and only 4 p_phi / 5 agrees
    with the paper's own approximation ~ 2/5 t/Tphi (p_phi ~ t / (2 Tphi) is small,
    so the printed form would give an error close to 1). (L8) is printed correctly.

    Returns one dict per qubit with p1, p_phi, the exact errors (err_relax,
    err_dephase) and their linear forms 2/5 t/T1 and 2/5 t/Tphi.
    """
    rows = []
    for t1_q, t2_q in zip(_pair(t1), _pair(t2)):
        tphi_q = t_phi(t1_q, t2_q)
        p1 = 1 - np.exp(-t / t1_q)
        p_phi = (1 - np.exp(-t / tphi_q)) / 2
        f_relax = (3 - p1 + 2 * np.sqrt(1 - p1)) / 5        # (L8)  F
        f_dephase = 1 - 4 * p_phi / 5                       # (L13) F - the paper prints this as 1 - F (typo)
        rows.append({
            "p1": p1,
            "p_phi": p_phi,
            "err_relax": 1 - f_relax,
            "err_dephase": 1 - f_dephase,                   # = 4 p_phi / 5
            "lin_relax": 0.4 * t / t1_q,
            "lin_dephase": 0.4 * t / tphi_q,
        })
    return rows


def analytic_error(t1, t2, t=T_ISWAP, exact=False):
    """(L8) + (L13) summed over both qubits, linear in t; `exact` uses p1, p_phi unexpanded."""
    keys = ("err_relax", "err_dephase") if exact else ("lin_relax", "lin_dephase")
    return sum(row[k] for row in channel_errors(t1, t2, t) for k in keys)


def main():
    t1_values = np.array([10, 20, 40, 80]) * US
    t2_over_t1 = [2.0, 1.5, 1.0, 0.5]          # 2.0 is the T1 limit (no pure dephasing)

    print("iSWAP time %.0f ns, g/2pi = %.4f MHz, both qubits identical\n"
          % (T_ISWAP / NS, 1 / (4 * T_ISWAP) / 1e6))
    print("%6s %6s %8s | %9s | %10s %10s %10s %10s | %6s"
          % ("T1", "T2", "Tphi", "P2(tsw)", "1-F sim", "1-F idle", "linear", "exact", "sim/lin"))
    print("%6s %6s %8s | %9s | %10s %10s %10s %10s |"
          % ("us", "us", "us", "", "", "", "", ""))
    print("-" * 94)
    table = []
    t_gate = np.linspace(0, T_ISWAP, 201)
    for t1 in t1_values:
        for r in t2_over_t1:
            t2 = r * t1
            tphi = t_phi(t1, t2)
            _, p2 = exchange_curve(t_gate, t1, t2)
            e_sim = gate_error(t1, t2)
            e_idle = gate_error(t1, t2, idle=True)
            e_lin = analytic_error(t1, t2)
            e_exact = analytic_error(t1, t2, exact=True)
            table.append((t1, t2, e_sim, e_lin))
            print("%6.0f %6.0f %8s | %9.5f | %10.3e %10.3e %10.3e %10.3e | %6.3f" % (
                t1 / US, t2 / US, "inf" if np.isinf(tphi) else "%.1f" % (tphi / US),
                p2[-1], e_sim, e_idle, e_lin, e_exact, e_sim / e_lin))

    print("\nunequal qubits, T1 = (40, 20) us, T2 = (30, 15) us:")
    t1, t2 = (40 * US, 20 * US), (30 * US, 15 * US)
    print("   1-F sim %.3e   idle %.3e   linear %.3e"
          % (gate_error(t1, t2), gate_error(t1, t2, idle=True), analytic_error(t1, t2)))

    # --- figure: exchange curves and gate error ---------------------------------
    colors = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834"]
    t_max = 80 * US
    t_dense = np.arange(0, t_max + 0.5 * NS, 1 * NS)       # 1 ns steps, drawn faintly
    legend_title = dict(title="solid top, dashed bottom envelope", title_fontsize=8)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    for c, t1 in zip(colors, t1_values):
        p1, _ = exchange_curve(t_dense, t1, 2 * t1)
        ax.plot(t_dense / US, p1, color=c, lw=0.3, alpha=0.25)
        t_top, top, t_bot, bot = exchange_envelope(t_max, t1, 2 * t1)
        ax.plot(t_top / US, top, color=c, label="T1 = %.0f us" % (t1 / US))
        ax.plot(t_bot / US, bot, color=c, ls="--")
    ax.set(title="T1 only (T2 = 2 T1): P(Q1 excited)", xlabel="time (us)", ylabel="population")
    ax.legend(loc="upper right", **legend_title)

    ax = axes[0, 1]
    t1 = 40 * US
    for c, t2 in zip(colors, np.array([80, 40, 20, 10]) * US):
        p1, _ = exchange_curve(t_dense, t1, t2)
        ax.plot(t_dense / US, p1, color=c, lw=0.3, alpha=0.25)
        t_top, top, t_bot, bot = exchange_envelope(t_max, t1, t2)
        ax.plot(t_top / US, top, color=c, label="T2 = %.0f us" % (t2 / US))
        ax.plot(t_bot / US, bot, color=c, ls="--")
    ax.set(title="T1 = 40 us, varying T2: P(Q1 excited)", xlabel="time (us)", ylabel="population")
    ax.legend(loc="upper right", **legend_title)

    ax = axes[1, 0]
    t_short = np.linspace(0, 4 * T_ISWAP, 801)
    for c, t2 in zip(colors, np.array([80, 40, 20, 10]) * US):
        p1, p2 = exchange_curve(t_short, t1, t2)
        ax.plot(t_short / NS, p1, color=c, label="T2 = %.0f us" % (t2 / US))
        ax.plot(t_short / NS, p2, color=c, ls="--")
    for k in (1, 3):
        ax.axvline(k * T_ISWAP / NS, color="0.5", lw=0.6)
    ax.set(title="first swaps at T1 = 40 us (solid Q1, dashed Q2)",
           xlabel="time (ns)", ylabel="population")
    ax.legend(loc="center right")

    ax = axes[1, 1]
    tab = np.array(table)
    for c, r in zip(colors, t2_over_t1):
        sel = np.isclose(tab[:, 1] / tab[:, 0], r)
        ax.loglog(tab[sel, 0] / US, tab[sel, 2], "o", color=c, label="T2 = %.1f T1" % r)
        ax.loglog(tab[sel, 0] / US, tab[sel, 3], "-", color=c, lw=1)
    ax.set(title="iSWAP error at %.0f ns: QuTiP (dots) vs (L8)+(L13) (lines)" % (T_ISWAP / NS),
           xlabel="T1 (us)", ylabel="1 - F_ave")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "iswap_incoherent_sim.png"), dpi=150)
    print("\nsaved iswap_incoherent_sim.png")


if __name__ == "__main__":
    main()
