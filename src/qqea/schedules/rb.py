"""Randomized benchmarking schedules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from quantify_scheduler import Schedule
from quantify_scheduler.backends.qblox.constants import MIN_TIME_BETWEEN_OPERATIONS
from quantify_scheduler.operations import IdlePulse, Reset, ResetClockPhase, X

from qqea.schedules.clifford.clifford_group import (
    common_cliffords,
)
from qqea.schedules.clifford.randomized_benchmarking import (
    randomized_benchmarking_sequence,
)
from qqea.schedules.gates import index_to_operation
from qqea.schedules.twpa import max_readout_duration, measure_with_twpa

if TYPE_CHECKING:
    from quantify_scheduler.device_under_test.transmon_element import (
        BasicTransmonElement,
    )


def randomized_benchmarking_schedule(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    lengths: Iterable[int],
    seeds: Iterable[int],
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
) -> Schedule:
    """
    Generate a randomized benchmarking schedule.

    All Clifford gates in the schedule are decomposed into products
    of the following unitary operations:

        {'iSWAP', 'I', 'Rx(pi)', 'Rx(pi/2)', 'Ry(pi)', 'Ry(pi/2)', 'Rx(-pi/2)', 'Ry(-pi/2)'}

    Parameters
    ----------
    qubit_specifier
        String or iterable of strings specifying which qubits to conduct the
        experiment on. If one name is specified, then single qubit randomized
        benchmarking is performed. If two names are specified, then two-qubit
        randomized benchmarking is performed.
    lengths
        Array of non-negative integers specifying how many Cliffords
        to apply before each recovery and measurement. If lengths is of size M
        then there will be M recoveries and M measurements in the schedule.
    desired_net_clifford_index
        Optional index specifying what the net Clifford gate should be. If None
        is specified, then no recovery Clifford is calculated. The default index
        is 0, which corresponds to the identity gate. For a map of common Clifford
        gates to Clifford indices, please see: two_qubit_clifford_group.common_cliffords
    seeds
        Optional random seeds to use for all lengths m. If a seed is None,
        then a new seed will be used for each length m. Values can be any integer
        between 0 and 2**32 - 1 inclusive.
    repetitions
        Optional positive integer specifying the amount of times the
        Schedule will be repeated. This corresponds to the number of averages
        for each measurement.

    """
    # ---- Error handling and argument parsing ----#
    lengths = np.asarray(lengths, dtype=int)

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    readout_duration = max_readout_duration(qubits)

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit randomized benchmarking supported.")

    # ---- Build RB schedule ----#
    sched = Schedule(
        "Randomized benchmarking on " + " and ".join(qubit_names), repetitions=repetitions
    )

    # two-qubit RB needs buffer time for phase corrections on drive lines
    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    # seeds and lengths both have length len(seed_setpoints)*len(length_setpoints)
    # or max_batch_size, whichever is smaller. If seed_setpoints is [1,2,3] and
    # length_setpoints is [4,5], then seeds will be [1,2,3,1,2,3] and lengths will
    # be [4,5,4,5,4,5]. This is why we iterate up to [:-2] for both. # FIXME: this seems fishy
    for acq_idx, (seed, m) in enumerate(zip(seeds[:-2], lengths[:-2])):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

        # m-sized random sample of the single/two qubit Clifford group
        rb_sequence_m = randomized_benchmarking_sequence(
            m, number_of_qubits=n, seed=seed, desired_net_cl=desired_net_clifford_index
        )

        for m_i, clifford_gate_idx in enumerate(rb_sequence_m):
            gate_sched = index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx)
            if gate_sched is not None:
                sched.add(gate_sched)

        # TODO: project before measuring if desired_net_clifford_index is not
        # common_cliffords["I"]
        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=acq_idx,
        )

    # Calibration points measured by preparing ground and excited states.
    sched.add(Reset(*qubit_names), label="Reset Cal 0")
    measure_with_twpa(
        sched,
        *qubit_names,
        readout_duration=readout_duration,
        acq_index=acq_idx + 1,
        label="Calibration 0",
    )
    
    reset_cal_1 = sched.add(Reset(*qubit_names), label="Reset Cal 1")
    for qubit_name in qubit_names:
        sched.add(X(qubit_name), ref_op=reset_cal_1, rel_time=0)
    measure_with_twpa(
        sched,
        *qubit_names,
        readout_duration=readout_duration,
        acq_index=acq_idx + 2,
        label="Calibration 1",
    )

    sched.add(IdlePulse(duration=4e-9))
    return sched


def simultaneous_randomized_benchmarking_schedule(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    lengths: Iterable[int],
    seeds: Iterable[int],
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
) -> Schedule:
    """
    Generate a simultaneous single-qubit randomized benchmarking schedule.

    All Clifford gates in the schedule are decomposed into products
    of the following unitary operations:

        {'I', 'Rx(pi)', 'Rx(pi/2)', 'Ry(pi)', 'Ry(pi/2)', 'Rx(-pi/2)', 'Ry(-pi/2)'}

    Parameters
    ----------
    qubit_specifier
        String or iterable of strings specifying which qubits to conduct the
        experiment on. If one name is specified, then single qubit randomized
        benchmarking is performed. If two names are specified, then independent
        single-qubit RB sequences are run simultaneously on both qubits.
    lengths
        Array of non-negative integers specifying how many Cliffords
        to apply before each recovery and measurement. If lengths is of size M
        then there will be M recoveries and M measurements in the schedule.
    desired_net_clifford_index
        Optional index specifying what the net Clifford gate should be. If None
        is specified, then no recovery Clifford is calculated. The default index
        is 0, which corresponds to the identity gate. For a map of common Clifford
        gates to Clifford indices, please see: two_qubit_clifford_group.common_cliffords
    seeds
        Optional random seeds to use for all lengths m. Each entry can either be
        a single integer seed (expanded deterministically to one distinct seed per
        qubit) or an iterable with one seed per qubit. If a seed is None, then a
        new seed will be used by the sequence generator.
    repetitions
        Optional positive integer specifying the amount of times the
        Schedule will be repeated. This corresponds to the number of averages
        for each measurement.

    """
    # ---- Error handling and argument parsing ----#
    lengths = np.asarray(lengths, dtype=int)

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    readout_duration = max_readout_duration(qubits)

    n = len(qubit_names)
    if n != 2:
        raise ValueError("Simultaneous randomized benchmarking should have two qubits.")

    # ---- Build RB schedule ----#
    sched = Schedule(
        "Randomized benchmarking on " + " and ".join(qubit_names), repetitions=repetitions
    )

    # two-qubit operation timing margin can still be useful for simultaneous 1Q RB
    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 0e-9][n - 1]

    # seeds and lengths both have length len(seed_setpoints)*len(length_setpoints)
    # or max_batch_size, whichever is smaller. If seed_setpoints is [1,2,3] and
    # length_setpoints is [4,5], then seeds will be [1,2,3,1,2,3] and lengths will
    # be [4,5,4,5,4,5]. This is why we iterate up to [:-2] for both. # FIXME: this seems fishy
    for acq_idx, (seed, m) in enumerate(zip(seeds[:-2], lengths[:-2])):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

        # m-sized random sample of the single-qubit Clifford group
        rb_sequence_m1 = randomized_benchmarking_sequence(
            m, number_of_qubits=1, seed=seed, desired_net_cl=desired_net_clifford_index
        )
        rb_sequence_m2 = randomized_benchmarking_sequence(
            m, number_of_qubits=1, seed=seed-1 if seed > 0 else seed+1, desired_net_cl=desired_net_clifford_index
        )

        for m_i, (clifford_gate_idx1, clifford_gate_idx2) in enumerate(zip(rb_sequence_m1, rb_sequence_m2)):
            two_qubit_clifford_gate_idx = clifford_gate_idx1 + 24 * clifford_gate_idx2
            gate_sched = index_to_operation(qubit_names, operation_buffer_time, two_qubit_clifford_gate_idx)
            if gate_sched is not None:
                sched.add(gate_sched)

        # TODO: project before measuring if desired_net_clifford_index is not
        # common_cliffords["I"]
        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=acq_idx,
        )

    # Calibration points measured by preparing ground and excited states.
    sched.add(Reset(*qubit_names), label="Reset Cal 0")
    measure_with_twpa(
        sched,
        *qubit_names,
        readout_duration=readout_duration,
        acq_index=acq_idx + 1,
        label="Calibration 0",
    )
    
    reset_cal_1 = sched.add(Reset(*qubit_names), label="Reset Cal 1")
    for qubit_name in qubit_names:
        sched.add(X(qubit_name), ref_op=reset_cal_1, rel_time=0)
    measure_with_twpa(
        sched,
        *qubit_names,
        readout_duration=readout_duration,
        acq_index=acq_idx + 2,
        label="Calibration 1",
    )

    sched.add(IdlePulse(duration=4e-9))
    return sched
