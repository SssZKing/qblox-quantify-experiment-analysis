"""lmfit models: beating Ramsey, double/triple Lorentzian and damped swap oscillation.

Also holds shared instances of quantify-core's models (``exp_decay_model``,
``decay_osci_model``, ``lorentzian_model``, ``resonator_model``) that the fits
and notebooks reuse.
"""

import lmfit
import numpy as np
from numpy.typing import NDArray
from quantify_core.analysis.fitting_models import (
    DecayOscillationModel,
    ExpDecayModel,
    LorentzianModel,
    ResonatorModel,
)
from scipy.signal import hilbert

resonator_model = ResonatorModel()
lorentzian_model = LorentzianModel()
exp_decay_model = ExpDecayModel()
decay_osci_model = DecayOscillationModel()

def two_tone_decay_func(
    t: float,
    tau: float,
    frequency0: float,
    phase0: float,
    amplitude0: float,
    frequency1: float,
    phase1: float,
    amplitude1: float,
    offset: float,
):
    
    oscillation0 = amplitude0 * (np.cos(2 * np.pi * frequency0 * t + phase0))
    oscillation1 = amplitude1 * (np.cos(2 * np.pi * frequency1 * t + phase1))
    oscillation = oscillation0 + oscillation1
    exp_decay = np.exp(-(t / tau))
    osc_decay = oscillation * exp_decay + offset
    return osc_decay

def fft_2freq_phase_guess(data: np.ndarray, t: np.ndarray):
    """
    Guess for a cosine fit using FFT, only works for evenly spaced points.

    Parameters
    ----------
    data:
        Input data to FFT
    t:
        Independent variable (e.g. time)

    Returns
    -------
    freq_guess:
        Guess for the 2 frequencies of the cosine function
    ph_guess:
        Guess for the phase of the cosine function
    """

    # Only first half of array is used, because the second half contains the
    # negative frequecy components, and we want a positive frequency.
    power = np.fft.fft(data)[: len(data) // 2]
    freq = np.fft.fftfreq(len(data), t[1] - t[0])[: len(power)]
    power[0] = 0  # Removes DC component from fourier transform

    # Use absolute value of complex valued spectrum
    abs_power = np.abs(power)
    
    idx = np.flip(np.argsort(abs_power)[-2:])
    freq_guess0 = freq[idx[0]]
    freq_guess1 = freq[idx[1]]

    # the condition data == max(data) can have several solutions
    #               (for example when discretization is visible)
    # to prevent errors we pick the first solution
    ph_guess0 = 2 * np.pi - (2 * np.pi * t[data == max(data)] * freq_guess0)[0]
    ph_guess1 = 2 * np.pi - (2 * np.pi * t[data == max(data)] * freq_guess1)[0]

    return [freq_guess0, freq_guess1], [ph_guess0, ph_guess1]

class BeatingDecayOscillationModel(lmfit.model.Model):
    r"""
    Model for a beating decaying oscillation which decays to a point with 0 offset from
    the centre of the of the oscillation (as in a Ramsey experiment, for example).

    ``frequency_max`` (Hz) and ``tau_max`` (s) bound both frequencies and the decay
    time; the defaults (500 kHz, 500 us) suit a Ramsey trace in seconds.
    """

    # pylint: disable=empty-docstring
    # pylint: disable=abstract-method
    # pylint: disable=too-few-public-methods


    def __init__(self, *args, frequency_max=500e3, tau_max=500e-6, **kwargs):
        # pass in the defining equation so the user doesn't have to later.
        super().__init__(two_tone_decay_func, *args, **kwargs)

        # Enforce oscillation frequency is positive
        self.set_param_hint("frequency0", min=0, max=frequency_max)
        self.set_param_hint("frequency1", min=0, max=frequency_max)
        # Enforce amplitude is positive
        self.set_param_hint("amplitude0", min=0)
        self.set_param_hint("amplitude1", min=0)
        # Enforce decay time is positive
        self.set_param_hint("tau", min=0, max=tau_max)

    # pylint: disable=missing-function-docstring

    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        t = kws.get("t")
        if t is None:
            raise ValueError(
                'Time variable "t" must be specified in order to guess parameters'
            )

        amp_guess = abs(max(data) - min(data)) / 2  # amp is positive by convention
        exp_offs_guess = np.mean(data)
        tau_guess = 2 / 3 * np.max(t)

        (freq_guess, phase_guess) = fft_2freq_phase_guess(data, t)

        self.set_param_hint("frequency0", value=freq_guess[0], min=0)
        self.set_param_hint("frequency1", value=freq_guess[1], min=0)
        self.set_param_hint("amplitude0", value=amp_guess, min=0)
        self.set_param_hint("amplitude1", value=amp_guess, min=0)
        self.set_param_hint("offset", value=exp_offs_guess)
        self.set_param_hint("phase0", value=phase_guess[0])
        self.set_param_hint("phase1", value=phase_guess[1])
        self.set_param_hint("tau", value=tau_guess, min=0)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

def double_lorentzian_func(
    x: float,
    x0: float,
    width: float,
    a: float,
    x0_: float,
    width_: float,
    a_: float,
    c: float,
) -> float:
    """Sum of two Lorentzian peaks on an offset ``c``.

    Each peak is ``a * width / (pi * ((x - x0)**2 + width**2))``: ``width`` is the
    half width at half maximum and ``a`` the peak area, so the height is
    ``a / (pi * width)``.
    """
    return (
        a * width / (np.pi * ((x - x0) ** 2 + width**2))
        + a_ * width_ / (np.pi * ((x - x0_) ** 2 + width_**2))
        + c
    )
    
class DoubleLorentzianModel(lmfit.model.Model):
    def __init__(self, *args, **kwargs) -> None:
        # pass in the defining equation so the user doesn't have to later.
        super().__init__(double_lorentzian_func, *args, **kwargs)

        self.set_param_hint("x0", vary=True)
        self.set_param_hint("a", vary=True)
        self.set_param_hint("c", vary=True)
        self.set_param_hint("width", vary=True)
        self.set_param_hint("x0_", vary=True)
        self.set_param_hint("a_", vary=True)
        self.set_param_hint("width_", vary=True)

    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        """Guess some initial values for the model based on the data."""
        x = kws.get("x")

        if x is None:
            return None  # type: ignore

        # Guess that the resonance is where the function takes its maximal
        # value
        midpoint = int(data.size/2)
        x0_guess = x[np.argmax(data[:midpoint])]
        x0_guess_ = x[midpoint+np.argmax(data[midpoint:])]
        self.set_param_hint("x0", value=x0_guess)
        self.set_param_hint("x0_", value=x0_guess_)

        # assume the user isn't trying to fit just a small part of a resonance curve.
        xmin = x.min()
        xmax = x.max()
        width_max = xmax - xmin

        delta_x = np.diff(x)  # assume f is sorted
        min_delta_x = delta_x[delta_x > 0].min()
        # assume data actually samples the resonance reasonably
        width_min = min_delta_x
        width_guess = np.sqrt(width_min * width_max)  # geometric mean, why not?
        self.set_param_hint("width", value=width_guess)
        self.set_param_hint("width_", value=width_guess)

        # The guess for the vertical offset is the mean absolute value of the data
        c_guess = np.mean(data)
        self.set_param_hint("c", value=c_guess)

        # Calculate A_guess from difference between the peak and the backround level
        a_guess = np.pi * width_guess * (np.max(data) - c_guess)
        self.set_param_hint("a", value=a_guess)
        self.set_param_hint("a_", value=a_guess)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

def triple_lorentzian_func(
    x: float,
    x0: float,
    width: float,
    a: float,
    x0_: float,
    width_: float,
    a_: float,
    x0__: float,
    width__: float,
    a__: float,
    c: float,
) -> float:
    """Sum of three Lorentzian peaks on an offset ``c``; same peak form as
    :func:`double_lorentzian_func` (``width`` = HWHM, ``a`` = area)."""
    return (
        a * width / (np.pi * ((x - x0) ** 2 + width**2))
        + a_ * width_ / (np.pi * ((x - x0_) ** 2 + width_**2))
        + a__ * width__ / (np.pi * ((x - x0__) ** 2 + width__**2))
        + c
    )

class TripleLorentzianModel(lmfit.model.Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(triple_lorentzian_func, *args, **kwargs)

        self.set_param_hint("x0", vary=True)
        self.set_param_hint("a", vary=True)
        self.set_param_hint("c", vary=True)
        self.set_param_hint("width", vary=True)
        self.set_param_hint("x0_", vary=True)
        self.set_param_hint("a_", vary=True)
        self.set_param_hint("width_", vary=True)
        self.set_param_hint("x0__", vary=True)
        self.set_param_hint("a__", vary=True)
        self.set_param_hint("width__", vary=True)

    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        """Guess some initial values for the model based on the data."""
        x = kws.get("x")

        if x is None:
            return None  # type: ignore

        third = int(data.size / 3)
        x0_guess = x[np.argmax(data[:third])]
        x0_guess_ = x[third + np.argmax(data[third:2 * third])]
        x0_guess__ = x[2 * third + np.argmax(data[2 * third:])]
        self.set_param_hint("x0", value=x0_guess)
        self.set_param_hint("x0_", value=x0_guess_)
        self.set_param_hint("x0__", value=x0_guess__)

        xmin = x.min()
        xmax = x.max()
        width_max = xmax - xmin

        delta_x = np.diff(x)
        min_delta_x = delta_x[delta_x > 0].min()
        width_min = min_delta_x
        width_guess = np.sqrt(width_min * width_max)
        self.set_param_hint("width", value=width_guess)
        self.set_param_hint("width_", value=width_guess)
        self.set_param_hint("width__", value=width_guess)

        c_guess = np.mean(data)
        self.set_param_hint("c", value=c_guess)

        a_guess = np.pi * width_guess * (np.max(data) - c_guess)
        self.set_param_hint("a", value=a_guess)
        self.set_param_hint("a_", value=a_guess)
        self.set_param_hint("a__", value=a_guess)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

def swap_decay_func(
    t: float,
    tau_osc: float,
    frequency: float,
    phase: float,
    amplitude: float,
    offset_i: float,
    offset_f: float,
    offset_tau: float,
):
    # Exponentially damped oscillation riding on an offset that relaxes from
    # offset_i -> offset_f with its own time constant offset_tau (decoupled from tau_osc).
    oscillation = amplitude * np.exp(-(t / tau_osc)) * np.cos(2 * np.pi * frequency * t + phase)
    offset_curve = offset_f + (offset_i - offset_f) * np.exp(-(t / offset_tau))
    return oscillation + offset_curve

def _moving_average(y: NDArray, window: int) -> NDArray:
    """Edge-padded moving average, so the output keeps the length of the input."""
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 else len(y) - 1)
    window = max(window, 3)
    pad = window // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(window) / window, mode="valid")

def _frequency_candidates(data: NDArray, t: NDArray, n_peaks: int = 3) -> list:
    """Dominant oscillation frequencies of a linearly detrended, windowed FFT.

    The spectrum is zero padded and each peak position is refined by parabolic
    interpolation, so the guess is not quantised to the FFT bin spacing. Bins
    below one full period per acquisition window and above 0.45/dt are ignored:
    neither can describe a swap oscillation at this sampling rate.
    """
    n = len(data)
    dt = float(np.median(np.diff(t)))
    span = float(t[-1] - t[0])
    detrended = data - np.polyval(np.polyfit(t, data, 1), t)

    nfft = int(2 ** np.ceil(np.log2(max(8 * n, 64))))
    magnitude = np.abs(np.fft.rfft(detrended * np.hanning(n), nfft))
    freqs = np.fft.rfftfreq(nfft, dt)
    df = freqs[1] - freqs[0]
    magnitude[freqs < 1.0 / span] = 0.0
    magnitude[freqs > 0.45 / dt] = 0.0

    # blank +-1.5 spectral resolution elements around each peak found, so the
    # candidates are distinct peaks instead of one peak sampled three times.
    exclusion = int(round(1.5 / (span * df))) + 2
    candidates = []
    remaining = magnitude.copy()
    for _ in range(n_peaks):
        k = int(np.argmax(remaining))
        if remaining[k] <= 0:
            break
        if 0 < k < len(magnitude) - 1:
            left, peak, right = magnitude[k - 1], magnitude[k], magnitude[k + 1]
            curvature = left - 2 * peak + right
            shift = np.clip(0.5 * (left - right) / curvature, -0.5, 0.5) if curvature != 0 else 0.0
        else:
            shift = 0.0
        candidates.append(float(freqs[k] + shift * df))
        remaining[max(0, k - exclusion):k + exclusion + 1] = 0.0

    return candidates or [1.0 / span]

def _swap_linear_refine(data: NDArray, t: NDArray, frequency: float, tau_osc: float, offset_tau: float):
    """Exact least squares for the parameters the model is linear in.

    With the frequency and both time constants held fixed, amplitude, phase,
    offset_i and offset_f follow from a linear solve rather than from heuristics,
    which is what removes the need to hand-tune starting values per dataset.
    """
    envelope = np.exp(-t / tau_osc)
    design = np.column_stack(
        [
            envelope * np.cos(2 * np.pi * frequency * t),
            envelope * np.sin(2 * np.pi * frequency * t),
            np.ones_like(t),
            np.exp(-t / offset_tau),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, data, rcond=None)
    residual = float(np.sum((data - design @ coefficients) ** 2))
    cos_coeff, sin_coeff, offset_f, offset_step = coefficients
    values = {
        "amplitude": float(np.hypot(cos_coeff, sin_coeff)),
        "phase": float(np.arctan2(-sin_coeff, cos_coeff)),
        "offset_f": float(offset_f),
        "offset_i": float(offset_f + offset_step),
    }
    return values, residual

class SwapDecayModel(lmfit.model.Model):
    def __init__(self, *args, **kwargs) -> None:
        # pass in the defining equation so the user doesn't have to later.
        super().__init__(swap_decay_func, *args, **kwargs)

        # Enforce oscillation frequency is positive
        self.set_param_hint("frequency", min=0)
        # Enforce amplitude is positive
        self.set_param_hint("amplitude", min=0)
        # Enforce both decay times are positive
        self.set_param_hint("tau_osc", min=0)
        self.set_param_hint("offset_tau", min=0)
        # phase is left unbounded: pinning it to [0, 2 pi) stalls the fit whenever
        # the true phase sits on the boundary. guess() wraps it into [-pi, pi).
        self.set_param_hint("phase", min=-np.inf, max=np.inf)

    def guess(self, data, **kws) -> lmfit.parameter.Parameters:
        """Starting values for a damped swap oscillation on a relaxing offset.

        For each of the strongest few spectral peaks: the baseline (moving average
        over one period) gives offset_i, offset_f and offset_tau, the analytic
        signal of the residual gives tau_osc from a log-linear envelope fit, and a
        linear solve gives amplitude and phase. The candidate with the smallest
        residual wins, which keeps the fit away from the local minima that
        previously forced per-dataset tweaking of tau_osc and phase.
        """
        t = kws.get("t")
        if t is None:
            raise ValueError(
                'Time variable "t" must be specified in order to guess parameters'
            )

        data = np.asarray(data, dtype=float)
        t = np.asarray(t, dtype=float)
        # estimate on a time axis starting at 0, then shift the result back
        t_start = float(t[0])
        t_rel = t - t_start
        span = float(t_rel[-1]) or 1.0
        dt = float(np.median(np.diff(t_rel)))

        best_residual, best = np.inf, None
        for frequency in _frequency_candidates(data, t_rel):
            period = 1.0 / frequency
            window = max(int(round(period / dt)), 3)
            baseline = _moving_average(data, window)

            # offset relaxation: log-linear fit of the baseline towards its final value
            offset_final = float(np.mean(baseline[-window:]))
            relaxation = baseline - offset_final
            offset_tau = span / 2
            if abs(relaxation[0]) > 1e-12:
                same_sign = np.sign(relaxation) == np.sign(relaxation[0])
                usable = same_sign & (np.abs(relaxation) > 0.15 * abs(relaxation[0]))
                if usable.sum() > 5:
                    slope = np.polyfit(t_rel[usable], np.log(np.abs(relaxation[usable])), 1)[0]
                    if slope < 0:
                        offset_tau = float(np.clip(-1.0 / slope, dt, 10 * span))

            # oscillation decay: log-linear fit of the analytic-signal envelope
            envelope = _moving_average(np.abs(hilbert(data - baseline)), window)
            envelope = np.clip(envelope, 1e-12, None)
            tau_osc = span / 2
            usable = envelope > 0.1 * envelope.max()
            if usable.sum() > 5:
                slope = np.polyfit(t_rel[usable], np.log(envelope[usable]), 1)[0]
                if slope < 0:
                    tau_osc = float(np.clip(-1.0 / slope, period, 10 * span))

            values, residual = _swap_linear_refine(data, t_rel, frequency, tau_osc, offset_tau)
            if residual < best_residual:
                values.update(frequency=frequency, tau_osc=tau_osc, offset_tau=offset_tau)
                best_residual, best = residual, values

        # put the guess back on the raw time axis (a no-op when t starts at 0)
        if t_start:
            decay = np.exp(np.clip(t_start / best["tau_osc"], -50, 50))
            offset_decay = np.exp(np.clip(t_start / best["offset_tau"], -50, 50))
            best["amplitude"] *= decay
            best["phase"] -= 2 * np.pi * best["frequency"] * t_start
            best["offset_i"] = best["offset_f"] + (best["offset_i"] - best["offset_f"]) * offset_decay
        best["phase"] = float((best["phase"] + np.pi) % (2 * np.pi) - np.pi)

        self.set_param_hint("tau_osc", value=best["tau_osc"], min=0)
        self.set_param_hint("frequency", value=best["frequency"], min=0.5 / span, max=0.5 / dt)
        self.set_param_hint("phase", value=best["phase"], min=-np.inf, max=np.inf)
        self.set_param_hint("amplitude", value=best["amplitude"], min=0)
        self.set_param_hint("offset_i", value=best["offset_i"])
        self.set_param_hint("offset_f", value=best["offset_f"])
        self.set_param_hint("offset_tau", value=best["offset_tau"], min=0)

        params = self.make_params()
        return lmfit.models.update_param_vals(params, self.prefix, **kws)

def swap_metrics(fit_result, t) -> dict:
    """Derived iSWAP numbers and envelope values from a SwapDecayModel fit.

    The model's envelopes are

        top(t)    = offset(t) + amplitude * exp(-t / tau_osc)
        bottom(t) = offset(t) - amplitude * exp(-t / tau_osc)

    with offset(t) relaxing from offset_i to offset_f over offset_tau, so the
    bottom envelope is a difference of two exponentials and has no time constant
    of its own; its start and end values inside the fitted window are reported
    instead. offset_f extrapolates to t -> inf and is not meaningful when
    offset_tau exceeds the measurement window.

    "resolved" is False when the oscillation is sampled at fewer than 4 points
    per period or the fitted contrast is under 5% of the signal range, i.e. when
    the extracted gate time should not be trusted.
    """
    t = np.asarray(t, dtype=float)
    params = fit_result.params
    frequency = params["frequency"].value
    tau_osc = params["tau_osc"].value
    period = 1.0 / frequency
    iswap_time = period / 2

    offset = params["offset_f"].value + (
        params["offset_i"].value - params["offset_f"].value
    ) * np.exp(-t / params["offset_tau"].value)
    half_contrast = params["amplitude"].value * np.exp(-t / tau_osc)
    top = offset + half_contrast
    bottom = offset - half_contrast

    samples_per_period = period / float(np.median(np.diff(t)))
    resolved = bool(
        params["amplitude"].value > 0.05 * float(np.ptp(fit_result.data))
        and samples_per_period >= 4
    )

    return {
        "frequency": frequency,
        "period": period,
        "iswap_time": iswap_time,
        "tau_osc": tau_osc,
        "offset_tau": params["offset_tau"].value,
        "efficiency": float(np.exp(-iswap_time / tau_osc)),
        "rsquared": fit_result.rsquared,
        "samples_per_period": samples_per_period,
        "resolved": resolved,
        "top": top,
        "bottom": bottom,
        "contrast": 2 * half_contrast,
        "bottom_start": float(bottom[0]),
        "bottom_final": float(bottom[-1]),
        "contrast_start": float(2 * half_contrast[0]),
        "contrast_final": float(2 * half_contrast[-1]),
    }
