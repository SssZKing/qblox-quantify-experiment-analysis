"""lmfit models: beating Ramsey and double/triple Lorentzian.

The iSWAP population-exchange model, ``IswapExchangeModel``, and its fits live in
:mod:`qqea.fitting.iswap_exchange`; the model is re-exported here.

Also holds shared instances of quantify-core's models (``exp_decay_model``,
``decay_osci_model``, ``lorentzian_model``, ``resonator_model``) that the fits
and notebooks reuse.
"""

import lmfit
import numpy as np
from quantify_core.analysis.fitting_models import (
    DecayOscillationModel,
    ExpDecayModel,
    LorentzianModel,
    ResonatorModel,
)

from qqea.fitting.iswap_exchange import IswapExchangeModel  # noqa: F401  (re-exported with the other models)

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
