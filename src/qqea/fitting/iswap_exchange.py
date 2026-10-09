"""Closed-form fits of iSWAP population exchange in terms of T1, Tphi and t_iSWAP.

Ported from the incoherent-error study (``iswap_incoherent.py``). Two qubits on
resonance exchange one excitation at Omega = pi / t_iSWAP while each relaxes
(1/T1) and dephases (1/Tphi). In the one-excitation manifold this is a damped
oscillator with a closed form (see :func:`iswap_exchange_func`), so the fit parameters
are the physical rates themselves, plus the readout errors e0 = P(read 1 | 0) and
e1 = P(read 0 | 1) of the qubit being read:

- :class:`IswapExchangeModel` / :func:`iswap_exchange_fit`: one trace (donor or acceptor).
- :func:`iswap_exchange_pair_fit`: donor and acceptor jointly, optional detuning.
- :func:`iswap_exchange_chevron_fit`: a pump-frequency x duration slice with a shared resonance.
- :func:`iswap_exchange_screen` / :func:`iswap_exchange_sweep_fit`: power sweeps with many repetitions.
- :func:`iswap_channel_errors` / :func:`iswap_analytic_error`: the (L8)+(L13) gate error of
  Li et al., PRX 14, 041050 (2024), Appendix L, and :func:`iswap_flux_noise_error`
  and :func:`echo_gaussian_profile_fit` for its flux-noise part (Appendix L1).

All times are in seconds and rates in 1/s; ``US`` and ``NS`` are unit helpers.
The QuTiP time evolution used to validate the closed form stays in the study.
"""

import lmfit
import numpy as np

US = 1e-6
NS = 1e-9


# ----------------------------------------------------------------------------- pair times
def t_phi(t1, t2):
    """Pure-dephasing time from T1 and T2; np.inf when T2 = 2 T1."""
    rate = 1 / t2 - 1 / (2 * t1)
    if rate < -1e-12 / t2:
        raise ValueError("T2 = %.3g must not exceed 2 T1 = %.3g" % (t2, 2 * t1))
    return np.inf if rate <= 0 else 1 / rate


def _pair(x):
    return (x, x) if np.isscalar(x) else tuple(x)


def mean_times(t1, t2):
    """(T1, Tphi) of the pair as seen by `iswap_exchange_func`.

    The model depends on the arithmetic mean of the two qubits' *rates*, so the
    pair times are the harmonic means of the two qubits' times:

        1/T1 = (1/T1_1 + 1/T1_2) / 2,    1/Tphi = (1/Tphi_1 + 1/Tphi_2) / 2
    """
    gamma_1 = np.mean([1 / x for x in _pair(t1)])
    gamma_phi = np.mean([1 / t_phi(a, b) for a, b in zip(_pair(t1), _pair(t2))])
    return 1 / gamma_1, (1 / gamma_phi if gamma_phi > 0 else np.inf)


# ----------------------------------------------------------------------------- exchange model, one trace
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


def iswap_exchange_func(t, t1, tphi, iswap_time, e0=0.0, e1=0.0, acceptor=False):
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


class IswapExchangeModel(lmfit.model.Model):
    """lmfit model of one swap trace: :func:`iswap_exchange_func` written in rates.

    Parameters are ``gamma_1`` = 1/T1 and ``gamma_phi`` = 1/Tphi (pair values, see
    :func:`mean_times`), ``iswap_time`` and the readout errors ``e0``, ``e1``. Fitting the
    rates allows Tphi = inf. ``acceptor=True`` models the qubit prepared in |0>.
    """

    def __init__(self, *args, acceptor=False, **kwargs):
        # pass in the defining equation so the user doesn't have to later.
        super().__init__(_acceptor_rates if acceptor else _donor_rates, *args, **kwargs)

        self.set_param_hint("gamma_1", min=0)
        self.set_param_hint("gamma_phi", min=0)
        self.set_param_hint("e0", min=0, max=0.5)
        self.set_param_hint("e1", min=0, max=0.5)

    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        """Starting values: t_iSWAP from the trace's FFT (bounded to 0.5 - 2x), T1 = 30 us,
        Tphi = 50 us, e0 = 0.02, e1 = 0.1. Keywords named like a parameter override it."""
        t = kws.get("t")
        if t is None:
            raise ValueError(
                'Time variable "t" must be specified in order to guess parameters'
            )

        iswap_time = kws.get("iswap_time", _fft_swap_time(t, data))
        self.set_param_hint("iswap_time", value=iswap_time, min=0.5 * iswap_time, max=2 * iswap_time)
        self.set_param_hint("gamma_1", value=1 / (30 * US), min=0)
        self.set_param_hint("gamma_phi", value=1 / (50 * US), min=0)
        self.set_param_hint("e0", value=0.02, min=0, max=0.5)
        self.set_param_hint("e1", value=0.1, min=0, max=0.5)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)


def _fft_swap_time(t, y):
    """t_iSWAP guessed from a swap trace: half the period of its strongest FFT peak.

    A cubic is subtracted first so the T1 midline does not dominate the low bins, and
    peaks below 3 cycles per window are ignored. The taper only falls off towards the
    end of the window: a symmetric Hann window would suppress the first few swaps,
    which are all that is left when the swap dies early (high pump power).
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    span = t[-1] - t[0]
    x = (t - t[0]) / span
    residual = y - np.polyval(np.polyfit(x, y, 3), x)
    taper = np.hanning(2 * t.size)[t.size:]
    n_fft = 16 * t.size
    spectrum = np.abs(np.fft.rfft(residual * taper, n_fft))
    freqs = np.fft.rfftfreq(n_fft, t[1] - t[0])
    spectrum[freqs < 3 / span] = 0
    return 0.5 / freqs[np.argmax(spectrum)]


def iswap_exchange_fit(t, y, t1_guess=30 * US, tphi_guess=50 * US, iswap_time_guess=None,
                       e0_guess=0.02, e1_guess=0.1, e0=None, e1=None, acceptor=False):
    """Fit `iswap_exchange_func` to a donor (or, `acceptor=True`, acceptor) trace.

    Fits the rates, so Tphi = inf is allowed. Passing `e0` / `e1` holds that
    readout error fixed (e.g. at the single-shot calibration, e0 = 1 - F_g,
    e1 = 1 - F_e); otherwise it is fitted, starting from `e0_guess` / `e1_guess`,
    and then also absorbs state-preparation error and thermal population.
    With `iswap_time_guess=None` the swap time is guessed from the trace's FFT.

    Returns (lmfit result, summary dict). The summary has the pair T1, Tphi (the
    harmonic means of the two qubits' times, see `mean_times`; with
    standard errors where the fit gives them), t_iSWAP, e0, e1, and the (L8)+(L13)
    linear gate error these imply,

        1 - F ~ 2/5 t_iSWAP sum_q (1/T1_q + 1/Tphi_q) = 4/5 t_iSWAP (1/T1 + 1/Tphi),

    which assumes the dephasing noise on the two qubits is independent: noise
    common to both cancels in the exchange (it only sees the frequency difference)
    but still costs gate fidelity.
    """
    if iswap_time_guess is None:
        iswap_time_guess = _fft_swap_time(t, y)
    model = IswapExchangeModel(acceptor=acceptor)
    params = model.make_params(gamma_1=1 / t1_guess, gamma_phi=1 / tphi_guess,
                               iswap_time=iswap_time_guess, e0=e0_guess, e1=e1_guess)
    params["iswap_time"].set(min=0.5 * iswap_time_guess, max=2 * iswap_time_guess)
    for name, fixed in (("e0", e0), ("e1", e1)):
        if fixed is not None:
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


# ----------------------------------------------------------------------------- donor + acceptor with detuning
def _z_detuned(t, gamma_phi, omega, delta):
    """Pseudo-spin z(t) with detuning, relaxation factored out (z(0) = 1).

    Bloch equations of H = delta/2 (n_d - n_a) + g (a_d^dag a_a + h.c.), with
    Omega = 2g and the |10>-|01> coherence (x, y) damped at 2 gamma_phi:

        dx/dt = -2 gamma_phi x - delta y
        dy/dt = delta x - 2 gamma_phi y - Omega z
        dz/dt = Omega y

    Solved through the eigen-decomposition of the 3x3 generator. For delta = 0 it
    reduces to the damped oscillator of `iswap_exchange_func`.
    """
    g2 = 2 * gamma_phi
    m = np.array([[-g2, -delta, 0.0], [delta, -g2, -omega], [0.0, omega, 0.0]])
    lam, vec = np.linalg.eig(m)
    coef = np.linalg.solve(vec, np.array([0.0, 0.0, 1.0]))
    return np.real(np.exp(np.outer(np.asarray(t, float), lam)) @ (vec[2] * coef))


def iswap_exchange_pair_func(t, t1, tphi, iswap_time, delta=0.0, e0_d=0.0, e1_d=0.0, e0_a=0.0, e1_a=0.0):
    """Donor and acceptor readout traces with a detuning `delta` (rad/s) from the resonance.

    `iswap_time` is the resonant swap time pi/Omega; with detuning the swap runs at
    sqrt(Omega^2 + delta^2) and transfers only Omega^2 / (Omega^2 + delta^2) of the
    excitation. T1, Tphi are the pair values of `mean_times`. Returns (y_donor, y_acceptor).
    """
    z = _z_detuned(t, 1 / tphi, np.pi / iswap_time, delta)
    envelope = 0.5 * np.exp(-np.asarray(t, float) / t1)
    return (e0_d + (1 - e0_d - e1_d) * envelope * (1 + z),
            e0_a + (1 - e0_a - e1_a) * envelope * (1 - z))


def iswap_exchange_pair_fit(t, y_donor, y_acceptor, t1_guess=None, tphi_guess=None,
                            iswap_time_guess=None, readout=None, delta=None):
    """Joint fit of donor and acceptor traces with `iswap_exchange_pair_func`.

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
    omega_eff = np.pi / _fft_swap_time(t, y_donor - y_acceptor)
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


def iswap_exchange_chevron_fit(t, freqs, y_donor, y_acceptor, iswap_time_guess, f0_guess=None):
    """Fit one pump-power slice of a chevron with the detuned model of `iswap_exchange_pair_func`.

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


def iswap_windowed_frequency_fit(t, diff, iswap_time, periods=3, min_amplitude=0.03):
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


def growing_loss_fit(t, y, e0, t_min=0.6 * US):
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


# ----------------------------------------------------------------------------- power sweeps
TRACE_KEYS = ("gamma_1", "gamma_phi", "iswap_time", "delta", "redchi",
              "e0_d", "e1_d", "e0_a", "e1_a", "t1_err", "tphi_err", "gate_error")


def iswap_exchange_screen(t, donor, acceptor, prep_min=0.5, acceptor_max=0.3, corr_min=0.5, env_min=0.1):
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
        _, f = iswap_exchange_pair_fit(t, y_donor, y_acceptor, iswap_time_guess=iswap_time_guess, readout=readout,
                                 delta=delta)
        return [f[key] for key in TRACE_KEYS]
    except Exception:
        return [np.nan] * len(TRACE_KEYS)


def iswap_exchange_sweep_fit(t, donor, acceptor, ok, iswap_time_guess, workers=8, readout=None, delta=None):
    """`iswap_exchange_pair_fit` on every trace with ok = True, in parallel processes.

    The swap frequency is seeded with `iswap_time_guess` (one per power, e.g. the chevron
    t_swap): at high power the swap dies within ~5 % of the window, where the FFT guess
    from the trace itself is least reliable. `readout` fixes readout errors: {name: array (repetition,)} with a
    value per repetition (e.g. from the readout calibration before it) for any of e0_d, e1_d,
    e0_a, e1_a. `delta` is passed to `iswap_exchange_pair_fit` (None: fitted; a number: fixed,
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


# ----------------------------------------------------------------------------- gate error (Appendix L)
def iswap_channel_errors(t1, t2, t):
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


def iswap_analytic_error(t1, t2, t, exact=False):
    """(L8) + (L13) summed over both qubits, linear in t; `exact` uses p1, p_phi unexpanded."""
    keys = ("err_relax", "err_dephase") if exact else ("lin_relax", "lin_dephase")
    return sum(row[k] for row in iswap_channel_errors(t1, t2, t) for k in keys)


def echo_gaussian_profile_fit(tau, y, sigma, grid):
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


def iswap_flux_noise_error(sqrt_a, dfdphi, iswap_time, t_meas=3600.0, n=4000, seed=0):
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
