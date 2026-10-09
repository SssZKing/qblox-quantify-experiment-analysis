"""Simulated data for dry runs, adapted from the quantify-scheduler tutorials.

The generators return what ``ScheduleGettable.get()`` would, so they can stand
in for hardware data; :mod:`qqea.experiments.dry_run` wires them into the
calibration nodes.

Fixed relative to the tutorial copy:

- ``ErrorModel.apply`` applied the gate unitary once per error step (four
  times per gate); it now applies it once, then the errors.
- ``ErrorModel.reset`` wrote to an unused ``errors`` dict; it now zeroes the
  error rates.
- ``get_simulated_rb_data`` always dropped the last two lengths as calibration
  points; that is now the ``num_cal_points`` argument (default 2, as before).
"""

from itertools import groupby
from typing import List, Optional, Union

import numpy as np

_GROUND = np.array([[1, 0], [0, 0]], dtype=complex)


def _dagger(matrix: np.ndarray) -> np.ndarray:
    return np.conj(matrix).T


def _phase_gate(phase: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * phase)]])


class ErrorModel:
    """
    Single-qubit error model acting on density matrices, applied after each gate.

    Parameters
    ----------
    coherent_phase_error: float
        Applies a Z rotation with angle coherent_phase_error.
    incoherent_phase_error: float
        Applies a Z rotation with an angle picked from a normal distribution with
        standard deviation incoherent_phase_error (a new angle on every call).
    decay_error: float
        Moves the state by a fraction decay_error towards the ground state.
    depolarizing_error: float
        Moves the state by a fraction depolarizing_error towards the center of
        the Bloch sphere.
    """

    def __init__(
        self,
        coherent_phase_error: float = 0,
        incoherent_phase_error: float = 0,
        decay_error: float = 0,
        depolarizing_error: float = 0,
    ) -> None:
        self.coherent_phase_error = coherent_phase_error
        self.incoherent_phase_error = incoherent_phase_error
        self.decay_error = decay_error
        self.depolarizing_error = depolarizing_error

    @staticmethod
    def _quantum_channel_of_a_gate(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
        r"""Return the channel U \rho U^\dagger."""
        return unitary @ rho @ _dagger(unitary)

    @staticmethod
    def _decay_channel(rho: np.ndarray, error_rate: float = 0.1) -> np.ndarray:
        """Decay into |0><0| with probability ``error_rate``."""
        ground_state = np.zeros(rho.shape)
        ground_state[0, 0] = 1
        return (1 - error_rate) * rho + error_rate * ground_state

    @staticmethod
    def _depolarizing_channel(rho: np.ndarray, error_rate: float = 0.1) -> np.ndarray:
        """Blend the state with the maximally mixed state."""
        d = rho.shape[0]
        return (1 - error_rate) * rho + error_rate * np.eye(d) / d

    def apply(
        self,
        input_state: np.ndarray = _GROUND,
        unitary: np.ndarray = np.eye(2),
    ) -> np.ndarray:
        """Apply the gate ``unitary`` to ``input_state``, then the error channels."""
        rho = self._quantum_channel_of_a_gate(input_state, unitary)
        if self.coherent_phase_error:
            rho = self._quantum_channel_of_a_gate(rho, _phase_gate(self.coherent_phase_error))
        if self.incoherent_phase_error:
            angle = np.random.normal(scale=self.incoherent_phase_error)
            rho = self._quantum_channel_of_a_gate(rho, _phase_gate(angle))
        rho = self._decay_channel(rho, self.decay_error)
        rho = self._depolarizing_channel(rho, self.depolarizing_error)
        return rho

    def reset(self) -> None:
        """Reset all error rates to 0."""
        self.coherent_phase_error = 0
        self.incoherent_phase_error = 0
        self.decay_error = 0
        self.depolarizing_error = 0


# IQ points of |0> and |1> in the simulated RB data
RB_GROUND_IQ = 0
RB_EXCITED_IQ = 1 + 5.0j


def get_simulated_rb_data(
    error_model: Optional[ErrorModel] = None,
    gate_strings: Optional[list] = None,
    input_state: np.ndarray = _GROUND,
    lengths: Union[list, np.ndarray] = (1, 2, 3),
    repetitions: Optional[int] = None,
    num_cal_points: int = 2,
) -> List[np.ndarray]:
    """Return single-qubit simulated RB data as ``[I, Q]``.

    One point per gate string for the first ``len(lengths) - num_cal_points``
    lengths, followed by the ground- and excited-state calibration points
    (``num_cal_points`` = 2) or nothing (0). With ``repetitions`` > 1 each point
    is the mean of that many single shots, otherwise the exact |1> population.
    """
    if error_model is None:
        error_model = ErrorModel()
    if gate_strings is None:
        gate_strings = [[np.eye(2)]]
    if num_cal_points not in (0, 2):
        raise ValueError("num_cal_points must be 0 or 2.")

    def _apply_gate_string(gate_string):
        rho = input_state
        for gate in gate_string:
            rho = error_model.apply(input_state=rho, unitary=gate)
        return rho

    n_sequences = len(lengths) - num_cal_points
    shots = repetitions is not None and repetitions > 1
    excited_population = []
    for idx in range(n_sequences):
        if not shots:
            excited_population.append(np.real(_apply_gate_string(gate_strings[idx])[1, 1]))
        elif error_model.incoherent_phase_error:
            # the incoherent phase is drawn per shot, so each shot is its own run
            p1 = [np.real(_apply_gate_string(gate_strings[idx])[1, 1]) for _ in range(repetitions)]
            excited_population.append(np.mean(np.random.rand(repetitions) < p1))
        else:
            p1 = np.clip(np.real(_apply_gate_string(gate_strings[idx])[1, 1]), 0, 1)
            excited_population.append(np.random.binomial(repetitions, p1) / repetitions)

    measurements = [
        (1 - p) * RB_GROUND_IQ + p * RB_EXCITED_IQ for p in excited_population
    ]
    if num_cal_points:
        measurements = measurements + [RB_GROUND_IQ, RB_EXCITED_IQ]
    measurements = np.asarray(measurements, dtype=complex)
    return [np.real(measurements), np.imag(measurements)]


def gate_strings_from_schedule(compiled_schedule) -> list:
    """Split the gate unitaries of a compiled schedule at each reset.

    Uses quantify-scheduler's private ``_walk_schedule``, so it may need
    updating when quantify-scheduler is upgraded.
    """
    from quantify_scheduler.schedules._visualization.circuit_diagram import _walk_schedule

    gate_string = []
    for _, operation in _walk_schedule(compiled_schedule):
        if "gate_library" in str(type(operation)):
            gate_string.append(operation["gate_info"]["unitary"])
    # Split at resets, which have unitary=None
    return [list(group) for key, group in groupby(gate_string, key=lambda x: x is not None) if key]


def simulated_rb_get(self):
    """Replacement for ``ScheduleGettable.get()`` returning simulated RB data.

    Kept for the quantify tutorial pattern (``ScheduleGettable.old_get`` holds the
    real ``get`` and ``self.error_model`` is set on the gettable). New code should
    use :func:`qqea.experiments.dry_run.simulate_rb`.
    """
    self.old_get()
    return get_simulated_rb_data(
        error_model=self.error_model,
        gate_strings=gate_strings_from_schedule(self._compiled_schedule),
        input_state=_GROUND,
        lengths=self.schedule_kwargs["lengths"](),
        repetitions=self.compiled_schedule.repetitions,
    )


def get_simulated_ssro_data(
    num_shots: int,
    g_centroid: complex = 10e-3 + 10e-3j,
    e_centroid: complex = 30e-3 + 30e-3j,
    measurement_noise: float = 0.01,
) -> List[np.ndarray]:
    """Mock single-shot readout data: ``num_shots`` interleaved g/e pairs as ``[I, Q]``."""
    def blob(centroid):
        return (np.random.normal(centroid.real, measurement_noise, num_shots)
                + 1j * np.random.normal(centroid.imag, measurement_noise, num_shots))

    result = np.empty(2 * num_shots, dtype=complex)
    result[0::2] = blob(g_centroid)
    result[1::2] = blob(e_centroid)
    return [np.real(result), np.imag(result)]


def get_simulated_readout_data(
    prepared_states,
    g_centroid: complex = 10e-3 + 10e-3j,
    e_centroid: complex = 30e-3 + 30e-3j,
    measurement_noise: float = 0.01,
) -> List[np.ndarray]:
    """Mock single shots for an arbitrary sequence of prepared states (0 or 1), as ``[I, Q]``."""
    states = np.asarray(prepared_states)
    centroids = np.where(states == 1, e_centroid, g_centroid)
    shots = (centroids.real + np.random.normal(0, measurement_noise, states.shape)
             + 1j * (centroids.imag + np.random.normal(0, measurement_noise, states.shape)))
    return [np.real(shots), np.imag(shots)]


def get_simulated_binary_ssro_data(
    num_shots: int, p_ground: float = 0.95, p_excited: float = 0.95
) -> List[np.ndarray]:
    """Mock discriminated single-shot readout data: interleaved g/e outcomes."""
    result = np.empty(2 * num_shots, dtype=float)
    result[0::2] = np.random.binomial(n=1, p=1 - p_ground, size=num_shots)
    result[1::2] = np.random.binomial(n=1, p=p_excited, size=num_shots)
    return [result, np.full_like(result, np.nan)]


def get_simulated_res_spec_data(
    fr: float = 7.902e9,
    q_l: float = 25e3,
    q_e: float = 32e3,
    a: float = 14,
    theta: float = 0.3,
    phi_v: float = 1.39e-6,
    phi_0: float = 4.6,
    alpha: float = 30,
    f: Union[list, np.ndarray, None] = None,
    noise: float = 0.05,
) -> tuple:
    """Simulated resonator spectroscopy as ``(magnitude, phase in degrees)``."""
    from quantify_core.analysis.fitting_models import hanger_func_complex_SI

    if f is None:
        span = 20e6
        f = np.linspace(fr - span / 2, fr + span / 2, 501)
    s21 = hanger_func_complex_SI(
        f, fr, q_l, q_e, a, theta, phi_v, phi_0, alpha
    ) + a * noise * np.random.rand(len(f))
    return (np.abs(s21), 180 / np.pi * np.angle(s21))


def get_simulated_tof_data() -> List[np.ndarray]:
    """Mock data for a time of flight measurement."""
    tof_t_setpoints = np.arange(1000)
    y = (
        np.heaviside(tof_t_setpoints - 200, 0.5)
        - np.heaviside(tof_t_setpoints - tof_t_setpoints.size * 0.7, 0.5)
    ) * 30e-3
    y += np.random.normal(loc=0.0, scale=1e-3, size=y.size)
    return [y, np.zeros_like(y)]
