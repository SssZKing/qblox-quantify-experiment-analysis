from itertools import groupby
from typing import Union

import numpy as np

from quantify_core.analysis.fitting_models import hanger_func_complex_SI
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.schedules._visualization.circuit_diagram import _walk_schedule


class ErrorModel:
    """
    A class for error models that act on quantum states.

    Parameters
    ----------
    coherent_phase_error: float
        Applies a rotation with angle coherent_phase_error.
    incoherent_phase_error: float
        Applies a rotation with an angle picked from a normal distribution with
        standard deviation incoherent_phase_error.
    decay_error: float
        Moves the state by a fraction decay_error towards the ground state.
    depolarizing_error: float
        Moves the state by a fraction depolarizing_error towards the center of
        the Bloch sphere.

    Attributes
    ----------
    errors : dict
        Key, value pairs of error_type, error_rate for the different error models.

    Methods
    -------
    apply(input_state: np.ndarray = np.array([[1, 0], [0, 0]]), unitary: np.ndarray = np.eye(2)) -> np.ndarray:
        Applies the specified error models to a quantum state.

    """

    def __init__(
        self,
        coherent_phase_error: float = 0,
        incoherent_phase_error: float = 0,
        decay_error: float = 0,
        depolarizing_error: float = 0,
    ) -> None:
        """Add errors as parameters of self."""
        self.coherent_phase_error = coherent_phase_error
        self.incoherent_phase_error = incoherent_phase_error
        self.decay_error = decay_error
        self.depolarizing_error = depolarizing_error

    def _quantum_channel_of_a_gate(
        self,
        rho: np.ndarray,
        unitary: np.ndarray,
    ) -> np.ndarray:
        r"""Return the channel U \rho U^\dagger."""
        return unitary @ rho @ np.asarray(np.matrix(unitary).H)

    def _decay_channel(self, rho: np.ndarray, error_rate: float = 0.1) -> np.ndarray:
        """Decay into |0><0| with probability p."""
        d = rho.shape
        ground_state = np.zeros(d)
        ground_state[0, 0] = 1
        return (1 - error_rate) * rho + error_rate * ground_state

    def _depolarizing_channel(self, rho: np.ndarray, error_rate: float = 0.1) -> np.ndarray:
        """
        Depolarize the channel with noise parameter p.

        The depolarizing channel blends the input state with the maximally mixed
        state.
        """
        d = rho.shape[0]  # Dimension (should be 2 for a single qubit)
        identity = np.eye(d)
        return (1 - error_rate) * rho + error_rate * identity / d

    def apply(
        self,
        input_state: np.ndarray = np.array([[1, 0], [0, 0]]),
        unitary: np.ndarray = np.eye(2),
    ) -> np.ndarray:
        """Apply the error models."""
        output_state = input_state
        # All error channels commute with each other, order is irrelevant
        # Apply the coherent_phase_error
        rotation_matrix = np.array([[1, 0], [0, np.exp(1j * self.coherent_phase_error)]])
        output_state = self._quantum_channel_of_a_gate(output_state, rotation_matrix @ unitary)
        # Apply the incoherent_phase_error
        rotation_matrix = np.array(
            [
                [1, 0],
                [0, np.exp(1j * np.random.normal(scale=self.incoherent_phase_error))],
            ]
        )
        output_state = self._quantum_channel_of_a_gate(output_state, rotation_matrix @ unitary)
        # Apply the decay error
        output_state = self._decay_channel(
            self._quantum_channel_of_a_gate(output_state, unitary), self.decay_error
        )
        # Apply the depolarizing_error
        output_state = self._depolarizing_channel(
            self._quantum_channel_of_a_gate(output_state, unitary), self.depolarizing_error
        )
        return output_state

    def reset(
        self,
    ) -> None:
        """Reset all error rates to 0."""
        self.errors = {
            "coherent_phase_error": 0,
            "incoherent_phase_error": 0,
            "decay_error": 0,
            "depolarizing_error": 0,
        }


def get_simulated_rb_data(
    error_model: ErrorModel = ErrorModel(),
    gate_strings: list = [[np.eye(2)]],
    input_state: np.ndarray = np.array([[1, 0], [0, 0]]),
    lengths: np.ndarray = [1, 2, 3],
    repetitions: Union[int, None] = None,
) -> np.ndarray:
    """Return single-qubit simulated RB data."""

    def _apply_gate_string(
        gate_string: list,
        input_state: np.ndarray = np.array([[1, 0], [0, 0]]),
    ) -> np.ndarray:
        output_state = input_state
        for gate in gate_string:
            output_state = error_model.apply(
                input_state=output_state,
                unitary=gate,
            )
        return output_state

    def _measure(rho: np.ndarray = np.eye(2)) -> int:
        """Return 0 or 1 with the probabilities on the diagonal."""
        return int(np.random.rand() < rho[1, 1])

    def _avg_measure(rho: np.ndarray = np.eye(2)) -> int:
        """Return the avg probabilities on the diagonal."""
        return rho[1, 1]

    measured_states = []
    for length_idx, _ in enumerate(lengths[:-2]):
        single_shots = []
        if repetitions is not None and repetitions > 1:
            # TODO: use binomial to speed this up
            for _ in range(repetitions):
                output_state = _apply_gate_string(gate_strings[length_idx], input_state)
                single_shots.append(_measure(output_state))
            measured_states.append(np.mean(single_shots))
        else:
            output_state = _apply_gate_string(gate_strings[length_idx], input_state)
            measured_states.append(_avg_measure(output_state))

    # Define IQ values for ground state and excited state
    ground_state = 0
    excited_state = 1 + 5.0j

    # compute the IQ values for the measurement
    measurements = [
        (1 - measured_state) * ground_state + measured_state * excited_state
        for measured_state in measured_states
    ]
    # Assign the last of the x and y measurements
    # to ground and excited state IQ values
    measurements = np.concatenate([measurements, [ground_state], [excited_state]])

    return [np.real(measurements), np.imag(measurements)]


def simulated_rb_get(self: ScheduleGettable):
    """Replace the output of ScheduleGettable.get() with simulated data."""
    self.old_get()
    # Use quantify_scheduler.schedules._visualization.circuit_diagram
    # ._walk_schedule to extract the full gate string
    gate_string = []
    for _, operation in _walk_schedule(self._compiled_schedule):
        if "gate_library" in str(type(operation)):
            gate_string.append(operation["gate_info"]["unitary"])
    # Split at resets, which have unitary=None
    gate_strings = [
        list(group) for key, group in groupby(gate_string, key=lambda x: x is not None) if key
    ]
    # TODO: generalize rho_ground_state to two-qubit case
    rho_ground_state = np.array([[1, 0], [0, 0]])
    return get_simulated_rb_data(
        error_model=self.error_model,
        gate_strings=gate_strings,
        input_state=rho_ground_state,
        lengths=self.schedule_kwargs["lengths"](),
        repetitions=self.compiled_schedule.repetitions,
    )


def get_simulated_ssro_data(
    num_shots: int,
    g_centroid: complex = 10e-3 + 10e-3j,
    e_centroid: complex = 30e-3 + 30e-3j,
    measurement_noise: float = 0.01,
) -> list[np.array]:
    """Generate mock data for an SSRO experiment."""
    real_part1 = np.random.normal(loc=g_centroid.real, scale=measurement_noise, size=num_shots)
    imag_part1 = np.random.normal(loc=g_centroid.imag, scale=measurement_noise, size=num_shots)
    real_part2 = np.random.normal(loc=e_centroid.real, scale=measurement_noise, size=num_shots)
    imag_part2 = np.random.normal(loc=e_centroid.imag, scale=measurement_noise, size=num_shots)

    ground_population = real_part1 + imag_part1 * 1j
    excited_population = real_part2 + imag_part2 * 1j

    result: list[complex] = list()
    for a, b in zip(ground_population, excited_population):
        result.append(a)
        result.append(b)

    result = np.asarray(result, dtype=complex)
    return [np.real(result), np.imag(result)]


def get_simulated_binary_ssro_data(
    num_shots: int, p_ground: float = 0.95, p_excited: float = 0.95
) -> list[np.array]:
    """Generate mock data for a discriminated SSRO experiment."""
    ground_state_measurements = np.random.binomial(n=1, p=1 - p_ground, size=num_shots)
    excited_state_measurements = np.random.binomial(n=1, p=p_excited, size=num_shots)

    result: list[complex] = list()
    for a, b in zip(ground_state_measurements, excited_state_measurements):
        result.append(a)
        result.append(b)

    result = np.asarray(result, dtype=float)
    return [result, np.zeros_like(result) * float("nan")]


def get_simulated_res_spec_data(
    fr: float = 7.902e9,
    q_l: float = 25e3,
    q_e: float = 32e3,
    a: float = 14,
    theta: float = 0.3,
    phi_v: float = 1.39e-6,
    phi_0: float = 4.6,
    alpha: float = 30,
    f: Union[list, np.ndarray] = None,
    noise: float = 0.05,
) -> list[np.ndarray]:
    """Generate simulated data for resonator spectroscopy."""
    if f is None:
        range = 20e6
        n_points = 501
        f = np.linspace(fr - range / 2, fr + range / 2, n_points)
    s21 = hanger_func_complex_SI(
        f, fr, q_l, q_e, a, theta, phi_v, phi_0, alpha
    ) + a * noise * np.random.rand(len(f))
    return (np.abs(s21), 180 / np.pi * np.angle(s21))


def get_simulated_tof_data() -> list[np.array]:
    """Generate mock data for a time of flight measurement."""
    tof_t_setpoints = np.arange(1000)
    y = (
        np.heaviside(tof_t_setpoints - 200, 0.5)
        - np.heaviside(tof_t_setpoints - tof_t_setpoints.size * 0.7, 0.5)
    ) * 30e-3
    y += np.random.normal(loc=0.0, scale=1e-3, size=y.size)
    return [y, np.zeros_like(y)]
