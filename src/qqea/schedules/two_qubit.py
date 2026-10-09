"""Utility functions for executing Schedules on Qblox hardware."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from qqea.schedules.clifford.clifford_group import (
    SingleQubitClifford,
    TwoQubitClifford,
    common_cliffords,
)
from qqea.schedules.clifford.randomized_benchmarking import randomized_benchmarking_sequence
from quantify_core.analysis.single_qubit_timedomain import SingleQubitTimedomainAnalysis
from quantify_core.visualization.mpl_plotting import (
    set_suptitle_from_dataset,
    set_xlabel,
    set_ylabel,
)
from quantify_scheduler import Schedule
from quantify_scheduler.backends.qblox.constants import MIN_TIME_BETWEEN_OPERATIONS
from quantify_scheduler.operations import X90, Y90, IdlePulse, Measure, Reset, Rxy, Rz, X, Y, MarkerPulse, ResetClockPhase, ShiftClockPhase

if TYPE_CHECKING:
    from collections.abc import Iterable

    from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
    from xarray import Dataset

from qqea.schedules.single_qubit import (TWPA_DELAY, TWPA_RINGUP, TWPA_TAIL)
# iSWAP_DELAY = 38e-9
iSWAP_DELAY = 120e-9
# Extra idle after every iSWAP (and sqrt-iSWAP) before the next operation, on top of the
# iSWAP_DELAY that waits for the RF to switch off. Measured 2026-10-02: a pump pulse changes
# the sum Stark phase of the NEXT iSWAP with a ~20 ns time constant -- back-to-back iSWAPs
# (26 ns RF gap) carry ~28 deg less sum phase per pair than iSWAPs >= ~100 ns apart, and a
# train of them shows a ~25 deg one-time phase. 0 keeps the original timing; ~100e-9 makes
# every iSWAP see the same (isolated) context. Read at call time, so it can be changed live.
ISWAP_SETTLE = 0.0
# Extra idle inserted only between CONSECUTIVE iSWAPs (no single-qubit layer between them) in
# pyGSTi circuits, so that every iSWAP in GST sees the isolated context the Stark phases are
# calibrated in. 1Q gate -> iSWAP and iSWAP -> 1Q gate need no extra time (checked 2026-10-02
# with a software iSWAP_DELAY scan: >= 15 ns margin before, >= 40 ns after). Measured memory of
# a pump pulse on the next iSWAP: 24 deg/pair with tau ~10 ns plus 10 deg/pair with tau ~80 ns;
# 100 ns leaves ~2.9 deg/pair (~1.4 deg/gate).
ISWAP_CONSECUTIVE_GAP = 100e-9
# Edge iswap.pulse_amp (dBm) at or below which the pump counts as off: the idle reference of
# process tomography runs the iSWAP schedule with pulse_amp(-100), and only that may skip the
# virtual-Z corrections (process_tomography_schedule raises otherwise).
PUMP_OFF_DBM = -60.0

# Individual per-qubit AC-Stark phase correction (dissertation part 3), applied on every
# iSWAP occurrence. Unlike parts 1+2 (_detuning_residual), this isn't derived from calibrated
# f01/pump frequency -- it's a single constant per qubit, measured directly by experiment.
#
# The live values are the iSWAP edge parameters qubit1_qubit2.iswap.ac_stark_phase_q1/q2
# (read by _ac_stark_phases, saved in every dataset snapshot). These two module constants
# are only the fallback for an edge that does not have those parameters.
#
# They are consumed in exactly one place: _iswap_rz_corrections, which folds them into the
# same per-qubit phase register as parts 1+2 so the iSWAP's frame exchange propagates them.
# Do NOT reintroduce them as a bare constant Rz beside the gate -- that is correct only for
# the first iSWAP of a block and leaves floor(n/2)*(Q0 - Q1) of differential Z error after
# that (see _iswap_rz_corrections). Every scheduling path already routes through
# _iswap_rz_corrections, so the Clifford (RB/RPE/QPT) and pyGSTi-circuit (GST) builders stay
# in sync by construction. Note the corollary: they now apply only when phase correction is
# enabled at all, i.e. when quantum_device / correction_phase is supplied.
AC_STARK_PHASE_Q0 = 98.5 # 77.86 # 60.74 #-5.15-6.15-0.07 #-26.36-13.05-6.15-6.31 #-92.73 - 16.17
AC_STARK_PHASE_Q1 = -106.5 #-56.32 #-42.97 # 3.61-2.00+4.69 #34.11+9.88+4.16+1.22 #140.72 - 14.61 + 2.29

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
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL

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
        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx))
        marker_pulse = sched.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit_names[0] + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
        )

    # Calibration points measured by preparing ground and excited states.
    sched.add(Reset(*qubit_names), label="Reset Cal 0")
    ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx + 1), label="Calibration 0")
    marker_pulse = sched.add(
        MarkerPulse(
            duration=marker_duration,
            port=qubit_names[0] + ":switch",
        ),
        ref_op=ro_pulse,
        ref_pt="start",
        rel_time=TWPA_DELAY-TWPA_RINGUP,
    )
    
    reset_cal_1 = sched.add(Reset(*qubit_names), label="Reset Cal 1")
    for qubit_name in qubit_names:
        sched.add(X(qubit_name), ref_op=reset_cal_1, rel_time=0)
    ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx + 2), label="Calibration 1")
    marker_pulse = sched.add(
        MarkerPulse(
            duration=marker_duration,
            port=qubit_names[0] + ":switch",
        ),
        ref_op=ro_pulse,
        ref_pt="start",
        rel_time=TWPA_DELAY-TWPA_RINGUP,
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
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL

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
        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx))
        marker_pulse = sched.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit_names[0] + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
        )

    # Calibration points measured by preparing ground and excited states.
    sched.add(Reset(*qubit_names), label="Reset Cal 0")
    ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx + 1), label="Calibration 0")
    marker_pulse = sched.add(
        MarkerPulse(
            duration=marker_duration,
            port=qubit_names[0] + ":switch",
        ),
        ref_op=ro_pulse,
        ref_pt="start",
        rel_time=TWPA_DELAY-TWPA_RINGUP,
    )
    
    reset_cal_1 = sched.add(Reset(*qubit_names), label="Reset Cal 1")
    for qubit_name in qubit_names:
        sched.add(X(qubit_name), ref_op=reset_cal_1, rel_time=0)
    ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx + 2), label="Calibration 1")
    marker_pulse = sched.add(
        MarkerPulse(
            duration=marker_duration,
            port=qubit_names[0] + ":switch",
        ),
        ref_op=ro_pulse,
        ref_pt="start",
        rel_time=TWPA_DELAY-TWPA_RINGUP,
    )

    sched.add(IdlePulse(duration=4e-9))
    return sched

def iswap_pulsed_pump_schedule(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    lengths: Iterable[int],
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
    iswap_spacing: float = 0.0,
) -> Schedule:
    """
    Pulsed pump iSWAP repetitive schedule.

    ``iswap_spacing`` (seconds, default 0) inserts an idle between consecutive
    iSWAPs. With 0 the iSWAPs run back to back (26 ns RF gap), a context in
    which a pump pulse changes the next iSWAP (2026-10-02: ~28 deg of sum phase
    per pair, fading with ~10 ns and ~80 ns time constants), so the duration
    calibrated here may differ from the one isolated iSWAPs need.
    """
    # ---- Error handling and argument parsing ----#
    lengths = np.asarray(lengths, dtype=int)

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit randomized benchmarking supported.")

    # ---- Build RB schedule ----#
    sched = Schedule(
        "Pulsed pump iSWAP", repetitions=repetitions
    )

    # two-qubit RB needs buffer time for phase corrections on drive lines
    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    for acq_idx, m in enumerate(lengths):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

        sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=72))

        for m_i in range(m):
            if m_i > 0 and iswap_spacing > 0:
                sched.add(IdlePulse(duration=iswap_spacing))
            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))

        # TODO: project before measuring if desired_net_clifford_index is not
        # common_cliffords["I"]
        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol))
        marker_pulse = sched.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit_names[0] + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
        )

    sched.add(IdlePulse(duration=4e-9))
    return sched


def iswap_tail_1q_schedule(
    qubit_specifier: Iterable[BasicTransmonElement],
    lengths: Iterable[int],
    gap: float = 0.0,
    pump: bool = True,
    reference_duration: float | None = None,
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "ThresholdedAcquisition",
) -> Schedule:
    """
    Does the pump's tail after an iSWAP disturb the single-qubit gate that follows?

    From |00>, repeat ``m`` times: iSWAP, idle ``gap``, X180 on both qubits
    (Clifford 75), then measure. Flipping both qubits together keeps the state in
    {|00>, |11>}, both eigenstates of the iSWAP, so the iSWAP only acts as a pump
    pulse and never moves an excitation. The X180 pairs also echo away any static Z
    phase (Stark shift, qubit detuning during idles), so what accumulates with
    ``m`` is the pulse error the pump tail causes in the X180s -- e.g. an axis tilt
    from the qubits still being Stark-shifted while they are driven.

    ``gap`` (seconds, 4 ns grid) is added on top of the default 16 ns buffer
    between the iSWAP block's RF-off and the X180 pulses. ``pump=False`` replaces
    every iSWAP block with an idle of ``reference_duration`` (26 ns lead + pulse
    duration), giving the same timing without any pump: the 1Q error floor.
    """
    lengths = np.asarray(lengths, dtype=int)
    qubits = list(qubit_specifier)
    qubit_names = [qubit.name for qubit in qubits]
    if len(qubit_names) != 2:
        raise ValueError("iswap_tail_1q_schedule requires two qubits.")
    if not pump and reference_duration is None:
        raise ValueError("pump=False needs reference_duration (26 ns + iSWAP pulse duration).")
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL
    operation_buffer_time = MIN_TIME_BETWEEN_OPERATIONS * 4e-9

    sched = Schedule("iSWAP tail on 1Q gates", repetitions=repetitions)
    for acq_idx, m in enumerate(lengths):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")
        for _ in range(m):
            if pump:
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))
            else:
                sched.add(IdlePulse(duration=reference_duration))
            if gap > 0:
                sched.add(IdlePulse(duration=gap))
            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=75))
        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol))
        sched.add(
            MarkerPulse(duration=marker_duration, port=qubit_names[0] + ":switch"),
            ref_op=ro_pulse, ref_pt="start", rel_time=TWPA_DELAY - TWPA_RINGUP,
        )
    sched.add(IdlePulse(duration=4e-9))
    return sched

def pump_probe_ramsey_schedule(
    qubit_specifier: Iterable[BasicTransmonElement],
    delays: Iterable[float],
    probe: int = 0,
    pump_duration: float = 2e-6,
    window: float = 200e-9,
    final_axis: str = "x",
    pump: bool = True,
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "ThresholdedAcquisition",
) -> Schedule:
    """
    Frequency of one qubit after a pump pulse, without any iSWAP in the probe.

    Per delay: reset, a bare pump pulse of ``pump_duration`` fired on ``|00>``
    (an iSWAP eigenstate, so nothing is exchanged), an idle ``delay`` from the
    pump's RF-off, then a Ramsey on qubit ``probe``: X90, free evolution for
    ``window``, closing X90 (``final_axis="x"``) or Y90 (``"y"``), measure.
    The two closing axes are the cos/sin quadratures of the phase accumulated
    in the window. ``pump=False`` replaces the pump pulse with an idle of the
    same length (pump-off reference with identical timing); the phase
    difference on - off over ``2 pi window`` is the mean qubit frequency shift
    in ``[delay, delay + window]`` after the X90.

    The pump pulse is placed like the iSWAP's own marker (RF lands 26 ns after
    the reset's ResetClockPhase), and the delay counts from its RF-off.
    ``delays`` and ``window`` are in seconds on the 4 ns grid.
    """
    delays = np.asarray(delays, dtype=float)
    qubits = list(qubit_specifier)
    qubit_names = [qubit.name for qubit in qubits]
    if len(qubit_names) != 2:
        raise ValueError("pump_probe_ramsey_schedule requires the two qubits of the iSWAP edge.")
    if final_axis not in ("x", "y"):
        raise ValueError("final_axis must be 'x' or 'y'.")
    for x in list(delays) + [window, pump_duration]:
        if x < 0 or abs(round(x / 4e-9) * 4e-9 - x) > 1e-12:
            raise ValueError("delays, window and pump_duration must be >= 0 and on the 4 ns grid.")
    q = qubit_names[probe]
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL
    sched = Schedule("Pump-probe Ramsey on " + q, repetitions=repetitions)
    for acq_idx, delay in enumerate(delays):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")
        if pump:
            sched.add(MarkerPulse(duration=pump_duration, port="snail:switch"), rel_time=26e-9 - iSWAP_DELAY)
            sched.add(IdlePulse(duration=iSWAP_DELAY))
        else:
            sched.add(IdlePulse(duration=26e-9 + pump_duration))
        if delay > 0:
            sched.add(IdlePulse(duration=delay))
        sched.add(Rxy(theta=90, phi=0, qubit=q))
        if window > 0:
            sched.add(IdlePulse(duration=window))
        sched.add(Rxy(theta=90, phi=0 if final_axis == "x" else 90, qubit=q))
        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol))
        sched.add(
            MarkerPulse(duration=marker_duration, port=qubit_names[0] + ":switch"),
            ref_op=ro_pulse, ref_pt="start", rel_time=TWPA_DELAY - TWPA_RINGUP,
        )
    sched.add(IdlePulse(duration=4e-9))
    return sched


def pump_probe_swap_schedule(
    qubit_specifier: Iterable[BasicTransmonElement],
    durations: Iterable[float],
    pre_duration: float = 2e-6,
    gap: float = 0.0,
    pre: bool = True,
    excite: int = 1,
    excite_before_pre: bool = False,
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "ThresholdedAcquisition",
) -> Schedule:
    """
    Exchange (chevron) after a pump pre-pulse: does a preceding pump pulse change the swap?

    Per duration: reset, a bare pump pulse of ``pre_duration`` on ``|00>`` (an
    iSWAP eigenstate, nothing is exchanged) or, with ``pre=False``, an idle of the
    same length; an idle ``gap``; X180 on qubit ``excite``; a probe pump pulse of
    the given duration (0 = none); measure both qubits. The pump frequency and
    power are the generator's own (set outside, e.g. as a meas_ctrl settable), so
    a frequency x duration grid gives a chevron with and without the pre-pulse.
    Pump pulses are placed like the iSWAP's marker (RF lands 26 ns after the
    previous operation). Times in seconds on the 4 ns grid.

    ``excite_before_pre=True`` applies the X180 before the pre-pulse instead of
    after the gap. With a pre-pulse of one iSWAP length the excitation then sits
    on the other qubit when the probe starts, and the gap can be as short as the
    back-to-back iSWAP spacing (the probe's RF lands ``gap`` + 26 ns after the
    pre-pulse's RF-off), which probes the fast part of the tail.
    """
    durations = np.asarray(durations, dtype=float)
    qubits = list(qubit_specifier)
    qubit_names = [qubit.name for qubit in qubits]
    if len(qubit_names) != 2:
        raise ValueError("pump_probe_swap_schedule requires the two qubits of the iSWAP edge.")
    for x in list(durations) + [pre_duration, gap]:
        if x < 0 or abs(round(x / 4e-9) * 4e-9 - x) > 1e-12:
            raise ValueError("durations, pre_duration and gap must be >= 0 and on the 4 ns grid.")
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL
    sched = Schedule("Pump-probe swap", repetitions=repetitions)

    def _pump(duration):
        sched.add(MarkerPulse(duration=duration, port="snail:switch"), rel_time=26e-9 - iSWAP_DELAY)
        sched.add(IdlePulse(duration=iSWAP_DELAY))

    for acq_idx, dur in enumerate(durations):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")
        if excite_before_pre:
            sched.add(Rxy(theta=180, phi=0, qubit=qubit_names[excite]))
        if pre_duration > 0:
            if pre:
                _pump(pre_duration)
            else:
                sched.add(IdlePulse(duration=26e-9 + pre_duration))
        if gap > 0:
            sched.add(IdlePulse(duration=gap))
        if not excite_before_pre:
            sched.add(Rxy(theta=180, phi=0, qubit=qubit_names[excite]))
        if dur > 0:
            _pump(dur)
        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol))
        sched.add(
            MarkerPulse(duration=marker_duration, port=qubit_names[0] + ":switch"),
            ref_op=ro_pulse, ref_pt="start", rel_time=TWPA_DELAY - TWPA_RINGUP,
        )
    sched.add(IdlePulse(duration=4e-9))
    return sched


def iswap_virtual_z_schedule(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    lengths: Iterable[int],
    angle: Iterable[float],
    gate_idx: int,
    pause_duration: float = 0,
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
) -> Schedule:

    # ---- Error handling and argument parsing ---- #
    lengths = np.asarray(lengths, dtype=int)

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit iSWAP phase tracking is supported.")

    sched = Schedule("iSWAP Phase Tracking", repetitions=repetitions)

    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    for acq_idx, m in enumerate(lengths):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

        # for _ in range(10):
        #     sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))

        sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=gate_idx))

        for m_i, clifford_gate_idx in enumerate(range(m)):
            gate_sched = index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760)
            if gate_sched is not None:
                # sched.add(IdlePulse(duration=pause_duration))
                sched.add(gate_sched)
            # if m_i == 0:
            #     sched.add(ShiftClockPhase(clock=f"{qubit_names[0]}.01", phase_shift=27))
            #     sched.add(ShiftClockPhase(clock=f"{qubit_names[1]}.01", phase_shift=27))
            #     sched.add(IdlePulse(duration=4e-9))

        sched.add(ShiftClockPhase(clock=f"{qubit_names[0]}.01", phase_shift=angle))
        sched.add(ShiftClockPhase(clock=f"{qubit_names[1]}.01", phase_shift=angle))

        sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=gate_idx))

        # sched.add(IdlePulse(duration=400e-9))

        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol))
        marker_pulse = sched.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit_names[0] + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
        )

    sched.add(IdlePulse(duration=4e-9))
    return sched


# --------------------------------------------------------------------------
# iSWAP virtual-Z phase compensation (dissertation section 4.3.2)
# --------------------------------------------------------------------------

def _detuning_residual(qubits, quantum_device) -> float:
    """
    ``Delta - Delta_tilde`` in Hz from the qubits' calibrated f01 and the
    iSWAP edge's own pump frequency.

    ``Delta_tilde`` is read straight off the device config, so it always
    matches the frequency the gate is actually pumped at -- retuning the
    gate updates the correction automatically.

    ``Delta_tilde`` must carry the same sign as ``Delta = f01_q0 - f01_q1``
    while the configured pump frequency is a positive magnitude, hence the
    ``copysign``. Note that ``copysign(abs(Delta) - pump_freq, Delta)``
    would be wrong -- it discards the inner sign when the pump sits above
    the bare detuning.
    """
    edges = quantum_device.generate_device_config().edges
    edge_name = f"{qubits[0].name}_{qubits[1].name}"
    if edge_name not in edges:
        edge_name = f"{qubits[1].name}_{qubits[0].name}"
    if edge_name not in edges:
        raise ValueError(
            f"No iSWAP edge found for {qubits[0].name}/{qubits[1].name} "
            f"in the device config (have: {sorted(edges)})."
        )
    pump_freq = edges[edge_name]["iSWAP"].factory_kwargs["pulse_frequency"]

    detuning = qubits[0].clock_freqs.f01() - qubits[1].clock_freqs.f01()
    return detuning - math.copysign(pump_freq, detuning)


def _iswap_pump_amp(qubits, quantum_device) -> float:
    """The iSWAP edge's configured pump power ``pulse_amp`` (dBm), read from the
    device config like ``_detuning_residual`` reads the pump frequency."""
    edges = quantum_device.generate_device_config().edges
    edge_name = f"{qubits[0].name}_{qubits[1].name}"
    if edge_name not in edges:
        edge_name = f"{qubits[1].name}_{qubits[0].name}"
    if edge_name not in edges:
        raise ValueError(
            f"No iSWAP edge found for {qubits[0].name}/{qubits[1].name} "
            f"in the device config (have: {sorted(edges)})."
        )
    return float(edges[edge_name]["iSWAP"].factory_kwargs["pulse_amp"])


def _ac_stark_phases(qubits, quantum_device) -> tuple[float, float]:
    """
    Per-qubit AC-Stark phases ``(phi_AC for qubits[0], phi_AC for qubits[1])``
    in degrees, read from the iSWAP edge's ``iswap.ac_stark_phase_q1`` /
    ``iswap.ac_stark_phase_q2`` (named after the device's qubit1 / qubit2:
    q1 = the edge's parent element, q2 = its child), and returned in
    ``qubits`` order.

    Keeping them on the edge puts them in every dataset snapshot next to the
    pump frequency they were calibrated with. If the edge has no such
    parameters (an older ``CompositeiSWAPEdge``), falls back to the
    module-level ``AC_STARK_PHASE_Q0``/``AC_STARK_PHASE_Q1`` with a warning.
    """
    for edge_name, swapped in ((f"{qubits[0].name}_{qubits[1].name}", False),
                               (f"{qubits[1].name}_{qubits[0].name}", True)):
        try:
            edge = quantum_device.get_edge(edge_name)
        except KeyError:
            continue
        iswap_params = edge.iswap.parameters
        if "ac_stark_phase_q1" in iswap_params and "ac_stark_phase_q2" in iswap_params:
            phases = (iswap_params["ac_stark_phase_q1"](), iswap_params["ac_stark_phase_q2"]())
            return phases[::-1] if swapped else phases
        break
    print(
        "Warning: iSWAP edge has no ac_stark_phase_q1/q2 parameters -- using the "
        f"module constants AC_STARK_PHASE_Q0/Q1 = ({AC_STARK_PHASE_Q0}, {AC_STARK_PHASE_Q1})."
    )
    return (AC_STARK_PHASE_Q0, AC_STARK_PHASE_Q1)


def _iswap_rz_corrections(
    t_list: list[float], detuning_residual: float,
    ac_stark: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """
    Turn a chronological list of iSWAP times (seconds since the phase reset)
    into the ``(theta_q0, theta_q1)`` ``Rz`` pair to insert after each one.

    Maintains a correction register ``c_q`` per qubit -- the phase that must
    be added to that qubit's NCO so its frame matches the actual excitation
    phase -- which each iSWAP transforms as::

        c_q0' = c_q1 - (Delta - Delta_tilde) * t + phi_AC_q0
        c_q1' = c_q0 + (Delta - Delta_tilde) * t + phi_AC_q1

    The gate actually inserted is the delta ``c_q' - c_q``. Registers start
    at zero, so the caller must pass one block's worth of times at a time
    (each block begins at its own ``ResetClockPhase``).

    ``phi_AC`` (dissertation part 3, ``AC_STARK_PHASE_Q0/Q1``) MUST enter here,
    inside the register, and not as a bare constant ``Rz`` next to the gate.
    Both forms agree on the first iSWAP of a block, but from the second on they
    diverge: the register's ``c_q0 <-> c_q1`` exchange means the correction the
    excitation carries follows it onto the other qubit, so after two iSWAPs each
    qubit is owed ``phi_AC_q0 + phi_AC_q1``, not ``2*phi_AC_q0`` / ``2*phi_AC_q1``.
    Applying the constant outside the register leaves a residual differential
    error of ``floor(n/2) * (phi_AC_q0 - phi_AC_q1)``, i.e. a *different* Z frame
    on the 1st and 2nd occurrence of the gate within one circuit. That is not a
    per-gate constant, so nothing downstream can absorb it. GST on
    20260903-1832 measured exactly this: -130/+113 deg of equal-and-opposite Z
    between its 1-iSWAP and 2-iSWAP circuits, against -+126.55 deg predicted
    from the (-92.73, +140.72) constants then in use, and a 2*DeltaLogL 8x
    worse than the same experiment with the constants zeroed (20260903-2029).

    Parameters
    ----------
    ac_stark
        ``(phi_AC_q0, phi_AC_q1)`` in degrees. Every schedule builder passes
        the edge's values (``_ac_stark_phases``). ``None`` falls back to the
        module-level ``AC_STARK_PHASE_Q0``/``AC_STARK_PHASE_Q1``.

    Notes
    -----
    There is deliberately no ``pump_phase``/``phi_0`` argument. It used to enter
    antisymmetrically (``+phi_0`` on q0, ``-phi_0`` on q1), which makes it a
    strict special case of ``ac_stark``: ``pump_phase=x`` is bit-for-bit
    ``ac_stark=(phi_AC_q0 + x, phi_AC_q1 - x)`` (verified to 1e-13 over eight
    iSWAPs). It was never independently measurable either -- only the pair
    (``theta_1``, ``theta_2``) is observed, and per arXiv 2604.27080 an
    equal-and-opposite Z on the two frames is exactly a shift of the drive
    phase by twice the amount, so ``phi_p`` is pure gauge. The physical knob is
    the pump generator's own phase (R&S SGS100A ``.phase()``); whatever it is
    set to shows up in the measured ``AC_STARK_PHASE_Q0``/``Q1`` and is
    corrected from there.
    """
    if ac_stark is None:
        ac_stark = (AC_STARK_PHASE_Q0, AC_STARK_PHASE_Q1)
    ac_q0, ac_q1 = ac_stark

    corrections = []
    c_q0, c_q1 = 0.0, 0.0
    for t in t_list:
        residual = 360.0 * detuning_residual * t
        new_c_q0 = c_q1 - residual + ac_q0
        new_c_q1 = c_q0 + residual + ac_q1
        corrections.append((new_c_q0 - c_q0, new_c_q1 - c_q1))
        c_q0, c_q1 = new_c_q0, new_c_q1
    return corrections


def compute_iswap_t1(sched: Schedule, quantum_device) -> list[float]:
    """
    Compile ``sched`` and return, for every iSWAP occurrence, the elapsed
    time in seconds since the most recent preceding ``ResetClockPhase`` in
    the schedule, in chronological order.

    ``ResetClockPhase`` (not ``Reset`` itself) is the actual "phase = 0"
    reference point: ``Reset`` has a long thermalization duration, and
    ``ResetClockPhase`` is scheduled at its *end* (``ref_pt="end"``), so
    using ``Reset``'s own (start-time) abs_time would incorrectly include
    that whole thermalization wait in ``t1``.

    iSWAP gates compile down to a ``MarkerPulse`` on the ``"snail:switch"``
    port (see ``CompositeiSWAPEdge.generate_edge_config``), which is used
    here as the unique fingerprint to find each occurrence in the compiled
    timing table.
    """
    from quantify_scheduler.backends import SerialCompiler

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )
    table = compiled_schedule.timing_table.data.sort_values("abs_time")

    iswap_times = table.loc[table["port"] == "snail:switch", "abs_time"].tolist()

    # Only the qubit (".01") clock resets start a block. ``Measure`` also
    # emits ResetClockPhase on the readout (".ro") clocks, and counting those
    # as block starts makes the shortest ".01" -> ".ro" gap look like the
    # block period.
    is_qubit_reset = table["operation"].str.match(r"ResetClockPhase\(clock='[^']*\.01'")
    reset_times = sorted(table.loc[is_qubit_reset, "abs_time"].tolist())
    if not reset_times:
        raise ValueError(
            "No ResetClockPhase on a '.01' clock found in the compiled timing table."
        )

    # The marker edge is NOT when the gate acts. It is issued ``iSWAP_DELAY``
    # early to compensate the pump's electrical delay, so the pump RF -- and
    # therefore the swap -- lands at ``marker + iSWAP_DELAY``. That is the
    # reference used here, for both the elapsed time and for deciding which
    # block a gate belongs to.
    #
    # Using the raw marker edge breaks down as soon as ``iSWAP_DELAY`` exceeds
    # whatever sits between the ResetClockPhase and the gate: the marker then
    # fires *before* its own block's reset, the gate gets anchored to the
    # previous block, and ``t`` picks up a whole Reset thermalisation.
    t1_list = []
    for t_marker in iswap_times:
        t_rf = t_marker + iSWAP_DELAY
        preceding_resets = [t for t in reset_times if t <= t_rf]
        t_ref = max(preceding_resets) if preceding_resets else 0.0
        t1_list.append(t_rf - t_ref)

    # Every gate must fall inside its own block. Anything at or beyond the
    # block period means the anchoring above went wrong and the corrections
    # would be silently garbage.
    block_gaps = [b - a for a, b in zip(reset_times, reset_times[1:]) if b > a]
    if block_gaps:
        block_period = min(block_gaps)
        overrun = [t for t in t1_list if t >= block_period]
        if overrun:
            raise ValueError(
                f"{len(overrun)} iSWAP(s) resolved to t >= the {block_period*1e6:.1f} us "
                f"block period (max {max(overrun)*1e6:.1f} us) -- they have been "
                "anchored to the wrong ResetClockPhase."
            )
    if any(t < 0 for t in t1_list):
        raise ValueError(
            "Negative iSWAP times: a gate resolved to before its own "
            "ResetClockPhase. Check iSWAP_DELAY against the schedule layout."
        )
    return t1_list


# --------------------------------------------------------------------------
# RPE schedules
# --------------------------------------------------------------------------

def iswap_RPE_f(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    lengths: Iterable[int],
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "ThresholdedAcquisition",
    quantum_device=None,
    final_axis: str = "x",
) -> Schedule:
    """
    Insert virtual-Z corrections around each iSWAP, following the phase
    tracking of dissertation section 4.3.2.

    Maintaining a correction register ``c_q`` per qubit, an iSWAP occurring
    at time ``t`` after the ``ResetClockPhase`` transforms it as::

        c_a' = c_b - (Delta - Delta_tilde) * t + phi_AC_a
        c_b' = c_a + (Delta - Delta_tilde) * t + phi_AC_b

    and the ``Rz`` actually inserted is the delta ``c_q' - c_q``. This one
    rule contains part 1 (the ``c_a <-> c_b`` swap, eq. 55), part 2 (the
    ``(Delta - Delta_tilde) * t`` residual, eq. 57) and part 3 (the constant
    per-qubit AC-Stark offset ``AC_STARK_PHASE_Q0``/``Q1``). Part 3 must ride
    inside the register like this rather than sit beside the gate as a bare
    constant ``Rz`` -- see ``_iswap_rz_corrections``.

    Note that the time variable is ``t`` measured from the phase reset, not
    the gap since the previous iSWAP: the mismatch is a frame offset set by
    total elapsed phase, not a quantity that accumulates per gate.

    Nothing has to be supplied for the residual. ``Delta_tilde`` is read off
    the iSWAP edge in the device config and combined with the qubits'
    calibrated f01 (see ``_detuning_residual``), so retuning the gate or
    recalibrating the qubits updates the correction on the next call.

    That residual -- the part of the detuning the pump does *not* already
    cancel, set by the AC Stark shift during pumping -- is on the ~100 kHz
    scale, not the full ~260 MHz bare detuning: the pump imprints
    ``Delta_tilde * t`` on the transferred excitation for free (eq. 53's
    off-diagonal ``e^{i phi_s}``). The qubits' own f01 reproducibility
    (observed to drift by >100 kHz within a single calibration run) is
    comparable to the residual itself, so a small leftover offset is
    expected.

    Parameters
    ----------
    quantum_device
        Required to enable the correction -- an upfront compile pass reads
        the real iSWAP times out of the schedule's own timing table. If
        ``None`` (default), no correction is applied and no compile is done.
        Requires two qubits.

    Notes
    -----
    Part 1 alone is a no-op in this schedule: ``index_to_operation`` maps
    every Clifford to a physical ``Rxy``, so there are no virtual-Z gates
    and ``c_a = c_b = 0`` until part 2 or ``AC_STARK_PHASE_Q0``/``Q1`` seeds
    the register.

    The pump's own phase is not an argument here: it is set on the generator
    (R&S SGS100A ``.phase()``) and lands in the measured
    ``AC_STARK_PHASE_Q0``/``Q1``, which ``_iswap_rz_corrections`` already
    carries. That is well posed because dissertation part 4 is handled in
    hardware -- the pump LO is derived by mixing the two qubit generators, so
    ``abs(G_a - G_b) = G_s`` holds and their phase drift cancels. The pump's
    phase is therefore stable relative to the NCO resets from block to block,
    and a single constant per-qubit offset is meaningful across the whole
    sweep.
    """
    # ---- Error handling and argument parsing ---- #
    lengths = np.asarray(lengths, dtype=int)
    # Closing pi/2 on q0: "x" = X90 (Clifford 16, default), "y" = Y90
    # (Clifford 21). The two are the cos/sin quadratures of the same phase.
    final_clifford = {"x": 16, "y": 21}[final_axis]

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit iSWAP phase tracking is supported.")

    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    def _build(iswap_phases: list[tuple[float, float]] | None) -> Schedule:
        """Build the RPE schedule. If ``iswap_phases`` is given, insert an
        Rz correction ``(theta_q0, theta_q1)`` right after each iSWAP
        occurrence, consumed in chronological order."""
        phase_iter = iter(iswap_phases) if iswap_phases is not None else None
        sched = Schedule("RPE_f", repetitions=repetitions)

        for acq_idx, m in enumerate(lengths):
            qubit_reset = sched.add(Reset(*qubit_names))
            for qubit_name in qubit_names:
                sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=13))

            for m_i, clifford_gate_idx in enumerate(range(m)):
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=6))
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))
                if phase_iter is not None:
                    theta_q0, theta_q1 = next(phase_iter)
                    sched.add(Rz(qubit=qubit_names[0], theta=theta_q0))
                    sched.add(Rz(qubit=qubit_names[1], theta=theta_q1))
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=144))
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))
                if phase_iter is not None:
                    theta_q0, theta_q1 = next(phase_iter)
                    sched.add(Rz(qubit=qubit_names[0], theta=theta_q0))
                    sched.add(Rz(qubit=qubit_names[1], theta=theta_q1))

            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=final_clifford))

            ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol))
            sched.add(
                MarkerPulse(
                    duration=marker_duration,
                    port=qubit_names[0] + ":switch",
                ),
                ref_op=ro_pulse,
                ref_pt="start",
                rel_time=TWPA_DELAY-TWPA_RINGUP,
            )

        sched.add(IdlePulse(duration=4e-9))
        return sched

    # ---- Pass 1: build without corrections, purely to resolve timing ---- #
    sched = _build(iswap_phases=None)

    if quantum_device is None:
        return sched

    if n != 2:
        raise ValueError("iSWAP phase correction requires two qubits.")

    detuning_residual = _detuning_residual(qubits, quantum_device)

    ac_stark = _ac_stark_phases(qubits, quantum_device)

    # Every "snail:switch" pulse in this schedule is a full iSWAP -- two per
    # m_i iteration -- so the whole list is corrected. Registers reset at
    # each block, whose boundaries follow from ``lengths``.
    t1_list = compute_iswap_t1(sched, quantum_device)
    expected = 2 * int(np.sum(lengths))
    if len(t1_list) != expected:
        raise ValueError(
            f"Expected {expected} 'snail:switch' pulses in the timing table, found {len(t1_list)}."
        )

    t_iter = iter(t1_list)
    iswap_phases = []
    for m in lengths:
        block = [next(t_iter) for _ in range(2 * m)]
        iswap_phases.extend(
            _iswap_rz_corrections(block, detuning_residual, ac_stark)
        )

    # ---- Pass 2: rebuild with the computed corrections applied ---- #
    return _build(iswap_phases=iswap_phases)


def iswap_RPE_theta_sum(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    lengths: Iterable[int],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "ThresholdedAcquisition",
    quantum_device=None,
    warmup_iswaps: int = 0,
    final_axis: str = "x",
    train_padding: float = 0.0,
    iswap_spacing: float = 0.0,
) -> Schedule:
    """
    Amplify ``theta_sum = theta_1 + theta_2`` alone, free of ``phi_zz`` and
    ``phi_p``.

    ``final_axis`` picks the closing pi/2 on q0: ``"x"`` (X90, Clifford 16,
    the default) or ``"y"`` (Y90, Clifford 21). The two runs are the cos and
    sin quadratures of the same phase, so together they give the per-pair
    phase with its sign and the one-time phase at m = 0, which a single
    projection cannot (arXiv 2604.27080 uses the same cos/sin pairs).

    ``train_padding`` (seconds, multiple of 4 ns) inserts an idle between the
    ``mX90`` and the first iSWAP and again between the last iSWAP and the
    closing pi/2 -- to test whether the pump's turn-on/off reaches into the
    neighbouring single-qubit pulses.

    ``iswap_spacing`` (seconds, multiple of 4 ns) inserts an idle between
    consecutive iSWAPs of the train (default 0: back to back, 26 ns RF gap).
    Used to test whether the per-gate phase depends on how close the pump
    pulses are. The idle adds qubit-detuning phase ``360 (df1 + df2) *
    spacing`` per pair, which the caller has to account for.

    Sequence: ``mX90`` on q0, ``m`` x (iSWAP, iSWAP) back to back, ``X90`` on
    q0, measure. The state stays in ``(|00> + |10>)/sqrt(2)``: one iSWAP pair
    returns the excitation to q0 carrying ``-exp(i (theta_1 + theta_2))``
    (Eq. 8), so ``|11>`` is never populated (no ``phi_zz``) and the pair's two
    ``phi_p`` contributions cancel. P(q0) oscillates at
    ``180 deg - |theta_sum|`` per pair.

    This is not one of the paper's Fig. 5 sequences: there ``phi_zz ~ 0`` is
    assumed, so sequence d (``iswap_RPE_d``) is read as ``theta_sum``. With a
    non-zero ``phi_zz`` sequence d gives ``theta_sum + phi_zz``; this sequence
    separates the two.

    Every iSWAP is corrected exactly as in ``iswap_RPE_f`` (see there for the
    register rule); pass ``quantum_device`` to enable it.

    ``warmup_iswaps`` bare iSWAPs are fired on ``|00>`` right after each
    block's ``ResetClockPhase``, before the ``mX90``. ``|00>`` is an iSWAP
    eigenstate, so the measured state is unchanged; they only stop the first
    iSWAP of the train from also being the first pump pulse of the block.
    They get no ``Rz`` and are skipped by position in the correction pass.
    """
    # ---- Error handling and argument parsing ---- #
    lengths = np.asarray(lengths, dtype=int)
    warmup_iswaps = int(warmup_iswaps)
    if warmup_iswaps < 0:
        raise ValueError("warmup_iswaps must be >= 0.")
    final_clifford = {"x": 16, "y": 21}[final_axis]
    if train_padding < 0 or iswap_spacing < 0:
        raise ValueError("train_padding and iswap_spacing must be >= 0.")

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL

    n = len(qubit_names)
    if n != 2:
        raise ValueError("iswap_RPE_theta_sum requires two qubits.")

    operation_buffer_time = MIN_TIME_BETWEEN_OPERATIONS * 4e-9

    def _build(iswap_phases: list[tuple[float, float]] | None) -> Schedule:
        """Build the schedule. If ``iswap_phases`` is given, insert an Rz
        correction ``(theta_q0, theta_q1)`` right after each iSWAP, consumed
        in chronological order."""
        phase_iter = iter(iswap_phases) if iswap_phases is not None else None
        sched = Schedule("RPE_theta_sum", repetitions=repetitions)

        for acq_idx, m in enumerate(lengths):
            qubit_reset = sched.add(Reset(*qubit_names))
            for qubit_name in qubit_names:
                sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

            for _ in range(warmup_iswaps):
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))

            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=13))
            if train_padding > 0:
                sched.add(IdlePulse(duration=train_padding))

            for k in range(2 * m):
                if k > 0 and iswap_spacing > 0:
                    sched.add(IdlePulse(duration=iswap_spacing))
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))
                if phase_iter is not None:
                    theta_q0, theta_q1 = next(phase_iter)
                    sched.add(Rz(qubit=qubit_names[0], theta=theta_q0))
                    sched.add(Rz(qubit=qubit_names[1], theta=theta_q1))

            if train_padding > 0:
                sched.add(IdlePulse(duration=train_padding))
            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=final_clifford))

            ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol))
            sched.add(
                MarkerPulse(
                    duration=marker_duration,
                    port=qubit_names[0] + ":switch",
                ),
                ref_op=ro_pulse,
                ref_pt="start",
                rel_time=TWPA_DELAY-TWPA_RINGUP,
            )

        sched.add(IdlePulse(duration=4e-9))
        return sched

    # ---- Pass 1: build without corrections, purely to resolve timing ---- #
    sched = _build(iswap_phases=None)

    if quantum_device is None:
        return sched

    detuning_residual = _detuning_residual(qubits, quantum_device)

    ac_stark = _ac_stark_phases(qubits, quantum_device)

    t1_list = compute_iswap_t1(sched, quantum_device)
    expected = 2 * int(np.sum(lengths)) + warmup_iswaps * len(lengths)
    if len(t1_list) != expected:
        raise ValueError(
            f"Expected {expected} 'snail:switch' pulses in the timing table "
            f"({len(lengths)} blocks of {warmup_iswaps} warm-up + 2m iSWAPs), found {len(t1_list)}."
        )

    t_iter = iter(t1_list)
    iswap_phases = []
    for m in lengths:
        for _ in range(warmup_iswaps):
            next(t_iter)  # warm-up on |00>: no excitation, nothing to track
        block = [next(t_iter) for _ in range(2 * m)]
        iswap_phases.extend(
            _iswap_rz_corrections(block, detuning_residual, ac_stark)
        )

    # ---- Pass 2: rebuild with the computed corrections applied ---- #
    return _build(iswap_phases=iswap_phases)


def iswap_RPE_d(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    lengths: Iterable[int],
    sqrt_iswap_duration: float,
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "ThresholdedAcquisition",
    quantum_device=None,
) -> Schedule:
    """
    Insert virtual-Z corrections around each full iSWAP, following the phase
    tracking of dissertation section 4.3.2. See ``iswap_RPE_f`` for the
    derivation of the correction rule and the meaning of each parameter.

    Only the ``m`` *full* iSWAPs in each block are compensated. The block's
    two bare ``sqrt_iswap_duration`` marker pulses -- one before and one
    after -- are deliberately skipped:

    - the leading one runs while both registers are still zero (the Clifford
      decomposition contains no virtual-Z gates), so there is nothing to
      exchange;
    - the trailing one is followed only by the measurement, so no later gate
      needs a corrected frame, and the full iSWAPs' corrections have already
      been applied before it;
    - a software phase exchange is in any case not valid for a *partial*
      swap (eq. 56) -- only the pump-phase route works for the iSWAP family.

    Parameters
    ----------
    quantum_device
        Required to enable the correction. If ``None`` (default), no
        correction is applied and no compile pass is done. The pump
        frequency is read from its iSWAP edge config, so it needs no
        separate argument. The pump's own phase offset is set on the
        generator and absorbed into ``AC_STARK_PHASE_Q0``/``Q1``.
    """
    # ---- Error handling and argument parsing ---- #
    lengths = np.asarray(lengths, dtype=int)

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit iSWAP phase tracking is supported.")

    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    def _build(iswap_phases: list[tuple[float, float]] | None) -> Schedule:
        """Build the RPE schedule. If ``iswap_phases`` is given, insert an
        Rz correction ``(theta_q0, theta_q1)`` right after each full iSWAP
        occurrence, consumed in chronological order."""
        phase_iter = iter(iswap_phases) if iswap_phases is not None else None
        sched = Schedule("RPE_d", repetitions=repetitions)

        for acq_idx, m in enumerate(lengths):
            qubit_reset = sched.add(Reset(*qubit_names))
            for qubit_name in qubit_names:
                sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=3))
            sched.add(MarkerPulse(duration=sqrt_iswap_duration, port="snail:switch"), rel_time=26e-9-iSWAP_DELAY)
            sched.add(IdlePulse(duration=iSWAP_DELAY + ISWAP_SETTLE))
            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=3))

            for m_i, clifford_gate_idx in enumerate(range(m)):
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))
                if phase_iter is not None:
                    theta_q0, theta_q1 = next(phase_iter)
                    sched.add(Rz(qubit=qubit_names[0], theta=theta_q0))
                    sched.add(Rz(qubit=qubit_names[1], theta=theta_q1))

            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=3))
            sched.add(MarkerPulse(duration=sqrt_iswap_duration, port="snail:switch"), rel_time=26e-9-iSWAP_DELAY)
            sched.add(IdlePulse(duration=iSWAP_DELAY + ISWAP_SETTLE))

            ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol))
            marker_pulse = sched.add(
                MarkerPulse(
                    duration=marker_duration,
                    port=qubit_names[0] + ":switch",
                ),
                ref_op=ro_pulse,
                ref_pt="start",
                rel_time=TWPA_DELAY-TWPA_RINGUP,
            )

        sched.add(IdlePulse(duration=4e-9))
        return sched

    # ---- Pass 1: build without corrections, purely to resolve timing ---- #
    sched = _build(iswap_phases=None)

    if quantum_device is None:
        return sched

    if n != 2:
        raise ValueError("iSWAP phase correction requires two qubits.")

    detuning_residual = _detuning_residual(qubits, quantum_device)

    ac_stark = _ac_stark_phases(qubits, quantum_device)

    # Each block fires on "snail:switch" m + 2 times: the leading sqrt-iSWAP,
    # the m full iSWAPs, then the trailing sqrt-iSWAP. Skip the two bare
    # sqrt-iSWAPs by position and correct only the full ones.
    t1_list = compute_iswap_t1(sched, quantum_device)
    expected = int(np.sum(lengths)) + 2 * len(lengths)
    if len(t1_list) != expected:
        raise ValueError(
            f"Expected {expected} 'snail:switch' pulses in the timing table "
            f"({len(lengths)} blocks of sqrt-iSWAP + m iSWAPs + sqrt-iSWAP), "
            f"found {len(t1_list)}."
        )

    t_iter = iter(t1_list)
    iswap_phases = []
    for m in lengths:
        next(t_iter)  # leading sqrt-iSWAP: registers still zero, nothing to exchange
        block = [next(t_iter) for _ in range(m)]
        next(t_iter)  # trailing sqrt-iSWAP: only the measurement follows it
        iswap_phases.extend(
            _iswap_rz_corrections(block, detuning_residual, ac_stark)
        )

    # ---- Pass 2: rebuild with the computed corrections applied ---- #
    return _build(iswap_phases=iswap_phases)

def iswap_delay_schedule(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    gate_idx: int,
    delays: Iterable[float],
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
) -> Schedule:

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit iSWAP phase tracking is supported.")

    sched = Schedule("iSWAP Delay Calibration", repetitions=repetitions)

    # operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    for acq_idx, delay in enumerate(delays):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

        first_pulse = sched.add(index_to_operation(qubit_names, 0, clifford_gate_idx=gate_idx))

        gate_sched = index_to_operation(qubit_names, 0, clifford_gate_idx=5760)
        swap_pulse = sched.add(gate_sched, ref_op=first_pulse, ref_pt="end", rel_time=delay-120e-9)

        # sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=gate_idx))
        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol),
            ref_op=swap_pulse, ref_pt="end", rel_time=0)
        marker_pulse = sched.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit_names[0] + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
        )

    sched.add(IdlePulse(duration=4e-9))
    return sched


from quantify_scheduler.operations import Operation

class iSWAP(Operation):
    r"""
    iSWAP gate, a common entangling gate.

    This operation can be represented by the following unitary:

    .. math::

        \mathrm{iSWAP}  = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & i & 0 \\
            0 & i & 0 & 0 \\
            0 & 0 & 0 & 1 \\ \end{bmatrix}

    Parameters
    ----------
    qC
        The control device element.
    qT
        The target device element
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, qC: str, qT: str, **device_overrides) -> None:
        device_element_control, device_element_target = qC, qT
        plot_func = "quantify_scheduler.schedules._visualization.circuit_diagram.cz"
        super().__init__(f"iSWAP ({device_element_control}, {device_element_target})")
        self.data.update(
            {
                "name": f"iSWAP ({device_element_control}, {device_element_target})",
                "gate_info": {
                    "unitary": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]),
                    "tex": r"iSWAP",
                    "plot_func": plot_func,
                    "device_elements": [device_element_control, device_element_target],
                    "symmetric": True,
                    "operation_type": "iSWAP",
                    "device_overrides": device_overrides,
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        gate_info = self.data["gate_info"]
        device_element_control = gate_info["device_elements"][0]
        device_element_target = gate_info["device_elements"][1]

        return (
            f"{self.__class__.__name__}(qC='{device_element_control}',qT='{device_element_target}')"
        )

def index_to_operation(
    qubit_names: list[str],
    operation_buffer_time: float,
    clifford_gate_idx: int,
    idle_duration: float | None = None,
) -> Schedule | None:
    """
    Convert a Clifford gate index to a Quantify Schedule of physical operations.

    This function takes a list of qubit names, a buffer time between operations, and a Clifford gate index.
    It determines the appropriate Clifford class (single or two-qubit), obtains the gate decomposition for the
    specified Clifford index, and maps each gate in the decomposition to a Quantify operation using a predefined
    mapping. The resulting operations are assembled into a Quantify Schedule, with appropriate timing and referencing
    for single- and two-qubit gates. If the decomposition results in no physical operations, None is returned.

    Parameters
    ----------
    qubit_names : list[str]
        List of qubit names. Length 1 for single-qubit, 2 for two-qubit Clifford gates.
    operation_buffer_time : float
        Buffer time (in seconds) to insert between operations in the schedule.
    clifford_gate_idx : int
        Index of the Clifford gate to decompose and schedule.

    Returns
    -------
    Schedule | None
        A Quantify Schedule object containing the physical operations for the Clifford gate,
        or None if the decomposition results in no operations.

    Raises
    ------
    NotImplementedError
        If the number of qubits is not 1 or 2.

    """
    if len(qubit_names) == 1:
        clifford_class = SingleQubitClifford
    elif len(qubit_names) == 2:  # noqa: PLR2004
        clifford_class = TwoQubitClifford
    else:
        raise NotImplementedError
    # ---- PycQED mappings ----#
    # map the pycqed qubit names to the ones used in quantify
    pycqed_qubit_map = {f"q{idx}": name for idx, name in enumerate(qubit_names)}
    # pycqed returns RB sequences as a list of strings. Map those to quantify operations
    pycqed_operation_map = {
        "I": lambda q: None,  # noqa: ARG005
        # "I": lambda q: IdlePulse(duration=100e-9),
        "X180": lambda q: X(pycqed_qubit_map[q[0]]),
        "X90": lambda q: X90(pycqed_qubit_map[q[0]]),
        "Y180": lambda q: Y(pycqed_qubit_map[q[0]]),
        "Y90": lambda q: Y90(pycqed_qubit_map[q[0]]),
        "mX90": lambda q: Rxy(qubit=pycqed_qubit_map[q[0]], phi=0.0, theta=-90.0),
        "mY90": lambda q: Rxy(qubit=pycqed_qubit_map[q[0]], phi=90.0, theta=-90.0),
        "iSWAP": lambda q: iSWAP(qC=pycqed_qubit_map[q[0]], qT=pycqed_qubit_map[q[1]]),
    }
    cl_decomp = clifford_class(clifford_gate_idx).gate_decomposition()
    gate_sched = Schedule()
    ref_op = gate_sched.add(IdlePulse(0.0))
    ref_ops = [ref_op, ref_op]

    for qubits, gates in cl_decomp:
        subsched = Schedule()
        subsched.add(IdlePulse(0.0))
        for gate in gates:
            op = pycqed_operation_map[gate](qubits)
            if op is not None:
                if gate == 'iSWAP':
                    subsched.add(op, rel_time=26e-9-iSWAP_DELAY)
                    # The AC-Stark phase (part 3) is NOT applied here as a bare
                    # constant Rz: it is carried in _iswap_rz_corrections'
                    # register so that the iSWAP's c_q0 <-> c_q1 exchange
                    # propagates it, and reaches the schedule as part of the
                    # correction pair the caller inserts after this block. See
                    # _iswap_rz_corrections for why the constant form is wrong
                    # from the second iSWAP of a block onwards.
                    subsched.add(IdlePulse(duration=iSWAP_DELAY + ISWAP_SETTLE))
                else:
                    subsched.add(op, rel_time=operation_buffer_time)
        if len(subsched.operations) == 1:
            # no gates added, only the initial IdlePulse
            continue
        if qubits == ("q0",):
            schedulable = gate_sched.add(subsched, ref_op=ref_ops[0])
            ref_ops[0] = schedulable
        elif qubits == ("q1",):
            # FIXME: this relies on the fact that single qubit Clifford are ALWAYS defined for both, and ALWAYS in the order q0, q1
            schedulable = gate_sched.add(subsched, ref_op=ref_ops[0], ref_pt="start")
            ref_ops[1] = schedulable
        elif qubits in [("q0", "q1"), ("q1", "q0")]:
            schedulable = gate_sched.add(subsched, ref_op=ref_ops[0])
            # Synchronise against q1's chain too -- the add() above only
            # references q0's. rel_time is 0, NOT operation_buffer_time: the
            # buffer is already applied inside subsched (rel_time on the op
            # itself), so passing it here again stacks on top of it and puts
            # 2 x operation_buffer_time in front of the gate.
            schedulable.add_timing_constraint(0, ref_ops[1])
            ref_ops = [schedulable, schedulable]
    # ``operations`` is a hash-keyed dict that always holds the anchor
    # IdlePulse(0.0), so it can never be empty -- count schedulables instead.
    # One schedulable means only that anchor was added, i.e. every entry in
    # the decomposition was skipped and the Clifford is the identity.
    if len(gate_sched.schedulables) <= 1:
        if idle_duration is None:
            return None
        # The whole Clifford is the identity, so nothing was emitted. Return a
        # real idle of one gate slot instead, so the operation still occupies
        # time and every sequence of the same gate count has the same duration.
        #
        # Note this deliberately keys off the *whole* decomposition being
        # empty, not off individual "I" entries. Those also appear as
        # structural padding around a two-qubit gate (e.g. Clifford 5760 is
        # [(q0,I),(q1,I),(q0q1,iSWAP),(q0,I),(q1,I)]), and timing them would
        # insert two spurious idle slots around every iSWAP.
        idle_sched = Schedule()
        idle_sched.add(IdlePulse(duration=idle_duration))
        return idle_sched
    return gate_sched


# RBAnalysis moved to qqea.fitting.rb; re-exported here so old imports keep working.
from qqea.fitting.rb import RBAnalysis  # noqa: E402,F401


######################################################################################################
# Quantum State Tomography Schedule
######################################################################################################

def _tomo_setup(qubit_specifier, operation_state_idx, warmup_iswaps=0, warmup_gap=0.0,
                warmup_duration=None):
    """
    Shared argument parsing for ``tomography_pre_build`` and
    ``tomography_schedule``.

    Both must derive the schedule's structure identically, otherwise a
    pre-computed correction list would not line up with the gates it is
    consumed against. Mirrors ``_qpt_setup``.
    """
    warmup_iswaps = int(warmup_iswaps)
    if warmup_iswaps < 0:
        raise ValueError("warmup_iswaps must be >= 0.")
    warmup_gap = float(warmup_gap)
    if warmup_gap < 0 or abs(round(warmup_gap / 4e-9) * 4e-9 - warmup_gap) > 1e-12:
        raise ValueError("warmup_gap must be >= 0 and on the 4 ns grid.")
    if warmup_duration is not None:
        warmup_duration = float(warmup_duration)
        if warmup_duration <= 0 or abs(round(warmup_duration / 4e-9) * 4e-9 - warmup_duration) > 1e-12:
            raise ValueError("warmup_duration must be > 0 and on the 4 ns grid.")

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit tomography supported.")
    if n == 1 and warmup_iswaps:
        raise ValueError("warmup_iswaps requires two qubits.")

    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL
    # two-qubit RB needs buffer time for phase corrections on drive lines
    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    # What one single-qubit tomography rotation occupies: its leading buffer
    # plus the pulse. Used to pad an all-identity rotation (index 0 below) so
    # every experiment in the set has the same duration.
    single_qubit_block = operation_buffer_time + max(qubit.rxy.duration() for qubit in qubits)

    if n == 1:
        tomo_sequence = [15, 16, 0] # index for [mY90, X90, I]
    else:
        tomo_sequence = [360, 384, 0, 15, 375, 399, 16, 376, 400]

    # Only the gate under test can carry an iSWAP; every tomography rotation
    # in tomo_sequence decomposes to single-qubit Rxy. It is applied once per
    # block, so the total is simply one per rotation.
    n_iswap_per_op = 0
    if n == 2:
        n_iswap_per_op = sum(
            gates.count("iSWAP")
            for _, gates in TwoQubitClifford(operation_state_idx).gate_decomposition()
        )

    return {
        "qubits": qubits,
        "qubit_names": qubit_names,
        "n": n,
        "marker_duration": marker_duration,
        "operation_buffer_time": operation_buffer_time,
        "single_qubit_block": single_qubit_block,
        "tomo_sequence": tomo_sequence,
        "n_iswap_per_op": n_iswap_per_op,
        # Corrected iSWAPs only -- the warm-up iSWAPs carry no correction.
        "n_iswap_total": len(tomo_sequence) * n_iswap_per_op,
        "warmup_iswaps": warmup_iswaps,
        "warmup_gap": warmup_gap,
        "warmup_duration": warmup_duration,
    }


def _tomo_build(cfg, operation_state_idx, acq_protocol, bin_mode, repetitions,
                iswap_phases, pad_identity=True):
    """
    Build the state-tomography schedule from a ``_tomo_setup`` dict.
    Mirrors ``_qpt_build``.

    The gate under test runs first in every block (after any warm-up iSWAPs),
    so its ``t`` -- and hence its correction -- is the same for every
    rotation. The tomography rotation follows the gate, so ``pad_identity``
    does not affect ``t``; it keeps the time before readout equal instead, so
    T1 decay does not differ across the rotation set.

    ``cfg["warmup_iswaps"]`` bare iSWAPs (Clifford 5760) are fired right after
    the ``ResetClockPhase``, while both qubits are still in ``|00>``. ``|00>``
    is an eigenstate of the iSWAP, so they do not change the prepared state;
    they only make sure the gate under test is not the first pump pulse after
    a long pump-off period. They get no ``Rz``: with no excitation there is no
    phase to track, and leaving the frames untouched means the correction
    register still starts at zero for the gate under test.

    ``cfg["warmup_duration"]`` replaces each warm-up iSWAP with a bare pump
    pulse (``MarkerPulse`` on ``snail:switch``) of that length, timed like the
    iSWAP's own marker. ``cfg["warmup_gap"]`` is an idle between the warm-up
    pulses and the gate under test -- also with no warm-up pulses, so a
    pump-off reference at the same gate time is ``warmup_iswaps=0`` with the
    same gap. Together they make a pump-probe measurement of how a pump pulse
    affects the next iSWAP ``warmup_gap`` later.
    """
    qubit_names = cfg["qubit_names"]
    operation_buffer_time = cfg["operation_buffer_time"]
    single_qubit_block = cfg["single_qubit_block"]

    phase_iter = iter(iswap_phases) if iswap_phases is not None else None

    sched = Schedule(
        "Tomography on " + " and ".join(qubit_names), repetitions=repetitions
    )

    for acq_idx, tomo_idx in enumerate(cfg["tomo_sequence"]):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

        for _ in range(cfg["warmup_iswaps"]):
            if cfg["warmup_duration"] is None:
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))
            else:
                # same placement as the iSWAP in index_to_operation: RF lands 26 ns after
                # the previous operation, followed by the RF-off wait
                sched.add(MarkerPulse(duration=cfg["warmup_duration"], port="snail:switch"),
                          rel_time=26e-9 - iSWAP_DELAY)
                sched.add(IdlePulse(duration=iSWAP_DELAY + ISWAP_SETTLE))
        if cfg["warmup_gap"] > 0:
            sched.add(IdlePulse(duration=cfg["warmup_gap"]))

        operation_sched = index_to_operation(qubit_names, operation_buffer_time, operation_state_idx)
        if operation_sched is not None:
            sched.add(operation_sched)
            if phase_iter is not None:
                theta_q0, theta_q1 = next(phase_iter)
                sched.add(Rz(qubit=qubit_names[0], theta=theta_q0))
                sched.add(Rz(qubit=qubit_names[1], theta=theta_q1))

        gate_sched = index_to_operation(
            qubit_names,
            operation_buffer_time,
            tomo_idx,
            idle_duration=single_qubit_block if pad_identity else None,
        )
        if gate_sched is not None:
            sched.add(gate_sched)

        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol, bin_mode=bin_mode))
        sched.add(
            MarkerPulse(
                duration=cfg["marker_duration"],
                port=qubit_names[0] + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
        )

    sched.add(IdlePulse(duration=4e-9))
    return sched


def tomography_pre_build(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    operation_state_idx: int,
    quantum_device,
    repetitions: int = 1,
    acq_protocol: Literal['SSBIntegrationComplex', 'ThresholdedAcquisition'] | str = 'ThresholdedAcquisition',
    bin_mode: Literal['average', 'append'] | str = 'append',
    pad_identity: bool = True,
    warmup_iswaps: int = 0,
    warmup_gap: float = 0.0,
    warmup_duration: float | None = None,
) -> np.ndarray:
    """
    Compute the iSWAP ``Rz`` corrections for state tomography without
    touching hardware, for use as ``correction_phase`` in
    ``tomography_schedule``. Mirrors ``process_tomography_pre_build``.

    The correction rule is the same as ``iswap_RPE_f`` -- see that function
    for the derivation and ``_detuning_residual`` for how the pump frequency
    is obtained. The gate under test is applied once per block and the
    correction registers restart at each block's ``ResetClockPhase``, so each
    block stands alone. Warm-up iSWAPs (see ``_tomo_build``) are skipped by
    position and get no correction; they only delay the gate under test, and
    its later ``t`` is picked up from the timing table.

    The structural arguments -- ``qubit_specifier``, ``operation_state_idx``,
    ``pad_identity``, ``warmup_iswaps``, ``warmup_gap``, ``warmup_duration``
    -- MUST match those passed to ``tomography_schedule``.

    Returns
    -------
    np.ndarray
        Shape ``(n_iswap, 2)`` of ``(theta_q0, theta_q1)`` in degrees, one row
        per iSWAP in chronological order. Shape ``(0, 2)`` if the schedule
        contains no iSWAP.

    Notes
    -----
    The result depends on the qubits' calibrated ``clock_freqs.f01()`` and on
    the iSWAP edge's pump frequency, so it goes stale when either is
    recalibrated -- re-run this after retuning.
    """
    cfg = _tomo_setup(qubit_specifier, operation_state_idx, warmup_iswaps, warmup_gap, warmup_duration)

    if cfg["n"] != 2 or cfg["n_iswap_per_op"] == 0:
        return np.empty((0, 2))
    if cfg["n_iswap_per_op"] > 1:
        raise NotImplementedError(
            f"Clifford {operation_state_idx} decomposes to {cfg['n_iswap_per_op']} iSWAPs. "
            "The correction is inserted after the whole gate subschedule, which is "
            "only equivalent for a single iSWAP; interleaving would require changing "
            "index_to_operation."
        )

    sched = _tomo_build(cfg, operation_state_idx, acq_protocol, bin_mode,
                        repetitions, iswap_phases=None, pad_identity=pad_identity)

    detuning_residual = _detuning_residual(cfg["qubits"], quantum_device)

    ac_stark = _ac_stark_phases(cfg["qubits"], quantum_device)
    t1_list = compute_iswap_t1(sched, quantum_device)

    n_warmup = cfg["warmup_iswaps"]
    per_block = n_warmup + cfg["n_iswap_per_op"]
    expected = len(cfg["tomo_sequence"]) * per_block
    if len(t1_list) != expected:
        raise ValueError(
            f"Expected {expected} 'snail:switch' pulses in the timing table "
            f"({len(cfg['tomo_sequence'])} rotations x ({n_warmup} warm-up + "
            f"{cfg['n_iswap_per_op']}) iSWAP), found {len(t1_list)}."
        )

    # Each block is n_warmup warm-up iSWAPs followed by the gate under test;
    # keep only the latter.
    t_gate = t1_list[n_warmup::per_block]

    return np.array(
        [_iswap_rz_corrections([t], detuning_residual, ac_stark)[0] for t in t_gate]
    )

def tomography_schedule(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    operation_state_idx: int,
    repetitions: int = 1,
    instrument_coordinator=None,
    acq_protocol: Literal['SSBIntegrationComplex', 'ThresholdedAcquisition'] | str = 'ThresholdedAcquisition',
    bin_mode: Literal['average', 'append'] | str = 'append',
    quantum_device=None,
    correction_phase: np.ndarray | None = None,
    pad_identity: bool = True,
    warmup_iswaps: int = 0,
    warmup_gap: float = 0.0,
    warmup_duration: float | None = None,
):
    """
    Run state tomography of the state prepared by ``operation_state_idx`` and
    return the raw acquisition. Mirrors ``process_tomography_schedule``.

    This compiles and executes internally, so both ``quantum_device`` and
    ``instrument_coordinator`` must be supplied.

    iSWAP phase tracking
    --------------------
    Pass ``correction_phase`` from ``tomography_pre_build`` to insert the
    virtual-Z corrections. The corrections are computed there rather than
    here so that this function builds and compiles the schedule exactly once.

    ``correction_phase=None`` runs uncorrected -- right for single-qubit
    tomography, a gate with no iSWAP, or an A/B against the corrected result.
    A warning is printed if the schedule contains iSWAPs and no correction was
    supplied, since that is silent otherwise.

    Parameters
    ----------
    instrument_coordinator
        Required -- raises if not given. Not defaulted to a module-level
        global, so that importing this module does not depend on instruments
        having been constructed first.
    quantum_device
        Required -- raises if not given.
    correction_phase
        Shape ``(n_iswap, 2)`` array of ``(theta_q0, theta_q1)`` in degrees,
        as returned by ``tomography_pre_build`` called with matching
        structural arguments.
    pad_identity
        Replace an all-identity tomography rotation with an idle of the same
        length, so every rotation in the set waits the same time before
        readout. Must match the value passed to ``tomography_pre_build``.
    warmup_iswaps
        Number of bare iSWAPs fired on ``|00>`` before the gate under test, so
        it is not the first pump pulse of the block (see ``_tomo_build``).
        Must match the value passed to ``tomography_pre_build``.
    warmup_gap
        Idle (s, 4 ns grid) between the warm-up pulses and the gate under test
        -- the delay of a pump-probe measurement. Applied also when
        ``warmup_iswaps`` is 0, which gives the pump-off reference at the same
        gate time. Must match ``tomography_pre_build``.
    warmup_duration
        If given (s, 4 ns grid), each warm-up iSWAP is replaced by a bare pump
        pulse of this length. Must match ``tomography_pre_build``.
    """
    from quantify_scheduler.backends import SerialCompiler

    if quantum_device is None:
        raise RuntimeError(
            "tomography_schedule requires a quantum_device (pass quantum_device=... )."
        )
    if instrument_coordinator is None:
        raise RuntimeError(
            "tomography_schedule requires an instrument_coordinator "
            "(pass instrument_coordinator=... )."
        )

    cfg = _tomo_setup(qubit_specifier, operation_state_idx, warmup_iswaps, warmup_gap, warmup_duration)

    expected = cfg["n_iswap_total"]
    if correction_phase is not None:
        correction_phase = list(correction_phase)
        if len(correction_phase) != expected:
            raise ValueError(
                f"correction_phase has {len(correction_phase)} entries but this "
                f"schedule contains {expected} iSWAP(s). It must come from "
                "tomography_pre_build() called with the same qubit_specifier / "
                "operation_state_idx."
            )
    elif expected:
        print(
            f"Warning: schedule contains {expected} iSWAP(s) but correction_phase "
            "is None -- running without virtual-Z phase correction. Use "
            "tomography_pre_build() to compute it."
        )

    sched = _tomo_build(cfg, operation_state_idx, acq_protocol, bin_mode,
                        repetitions, iswap_phases=correction_phase,
                        pad_identity=pad_identity)

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(schedule=sched, config=quantum_device.generate_compilation_config())

    instrument_coordinator.prepare(compiled_schedule)
    instrument_coordinator.start()
    instrument_coordinator.wait_done(timeout_sec=60)

    return instrument_coordinator.retrieve_acquisition()


######################################################################################################
# Quantum Process Tomography Schedule
######################################################################################################

def _qpt_setup(qubit_specifier, operation_state_idx, experiment_list, identity_reference):
    """
    Shared argument parsing for ``process_tomography_pre_build`` and
    ``process_tomography_schedule``.

    Both must derive the schedule's structure identically, otherwise a
    pre-computed correction list would not line up with the gates it is
    consumed against.
    """
    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit tomography supported.")

    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL
    # two-qubit RB needs buffer time for phase corrections on drive lines
    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    # What one single-qubit tomography rotation occupies: its leading buffer
    # plus the pulse. Used to pad an all-identity rotation so every experiment
    # in the set has the same duration (see ``pad_identity``).
    single_qubit_block = operation_buffer_time + max(qubit.rxy.duration() for qubit in qubits)

    if n == 1:
        tomo_sequence = [0, 3, 21, 15, 13, 16] # index for [I, X180, Y90, mY90, mX90, X90]
    else:
        tomo_sequence = [
            0, 72, 504, 360, 312, 384,
            3, 75, 507, 363, 315, 387,
            21, 93, 525, 381, 333, 405,
            15, 87, 519, 375, 327, 399,
            13, 85, 517, 373, 325, 397,
            16, 88, 520, 376, 328, 400
        ]

    # Materialised because the schedule may be built more than once -- a bare
    # iterator would be exhausted by the first pass.
    experiment_list = np.asarray(experiment_list)
    if identity_reference:
        experiment_list = np.repeat(experiment_list, 2)

    # Only the gate under test can carry an iSWAP; every tomography pre/post
    # rotation decomposes to single-qubit Rxy.
    n_iswap_per_op = 0
    if n == 2:
        n_iswap_per_op = sum(
            gates.count("iSWAP")
            for _, gates in TwoQubitClifford(operation_state_idx).gate_decomposition()
        )

    # Number of blocks that actually contain the gate under test: with
    # identity_reference the list is doubled and only even acq_idx get it.
    n_op_blocks = sum(
        1
        for acq_idx in range(len(experiment_list))
        if not identity_reference or acq_idx % 2 == 0
    )

    return {
        "qubits": qubits,
        "qubit_names": qubit_names,
        "n": n,
        "marker_duration": marker_duration,
        "operation_buffer_time": operation_buffer_time,
        "tomo_sequence": tomo_sequence,
        "single_qubit_block": single_qubit_block,
        "experiment_list": experiment_list,
        "n_iswap_per_op": n_iswap_per_op,
        "n_op_blocks": n_op_blocks,
        "n_iswap_total": n_op_blocks * n_iswap_per_op,
    }


def _qpt_build(cfg, operation_state_idx, identity_reference, acq_protocol,
               bin_mode, repetitions, iswap_phases, pad_identity=True):
    """
    Build the process-tomography schedule from a ``_qpt_setup`` dict.

    If ``iswap_phases`` is given, insert an ``Rz`` correction
    ``(theta_q0, theta_q1)`` right after the gate under test, consumed in
    chronological order.

    With ``pad_identity`` an all-identity tomography rotation is replaced by
    an ``IdlePulse`` of the same length as a real rotation. ``"I"`` maps to
    ``None`` in ``pycqed_operation_map``, so such a rotation would otherwise
    occupy zero time and those experiments would run shorter than the rest.
    """
    qubit_names = cfg["qubit_names"]
    tomo_sequence = cfg["tomo_sequence"]
    operation_buffer_time = cfg["operation_buffer_time"]
    single_qubit_block = cfg["single_qubit_block"]

    def _add_rotation(sched, clifford_idx):
        """Add one tomography rotation, padding a pure identity to match."""
        rotation = index_to_operation(
            qubit_names,
            operation_buffer_time,
            clifford_idx,
            idle_duration=single_qubit_block if pad_identity else None,
        )
        if rotation is not None:
            sched.add(rotation)

    phase_iter = iter(iswap_phases) if iswap_phases is not None else None
    sched = Schedule(
        "QPT idx " + str(operation_state_idx) + " on " + " and ".join(qubit_names),
        repetitions=repetitions,
    )

    for acq_idx, experiment_idx in enumerate(cfg["experiment_list"]):
        prep_idx = experiment_idx // len(tomo_sequence)
        meas_idx = experiment_idx % len(tomo_sequence)

        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

        _add_rotation(sched, tomo_sequence[prep_idx])

        if not identity_reference or acq_idx % 2 == 0:
            operation_sched = index_to_operation(qubit_names, operation_buffer_time, operation_state_idx)
            if operation_sched is not None:
                sched.add(operation_sched)
                if phase_iter is not None:
                    theta_q0, theta_q1 = next(phase_iter)
                    sched.add(Rz(qubit=qubit_names[0], theta=theta_q0))
                    sched.add(Rz(qubit=qubit_names[1], theta=theta_q1))

        _add_rotation(sched, tomo_sequence[meas_idx])

        ro_pulse = sched.add(Measure(*qubit_names, acq_index=acq_idx, acq_protocol=acq_protocol, bin_mode=bin_mode))
        sched.add(
            MarkerPulse(
                duration=cfg["marker_duration"],
                port=qubit_names[0] + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
        )

    sched.add(IdlePulse(duration=4e-9))
    return sched


def process_tomography_pre_build(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    operation_state_idx: int,
    experiment_list: Iterable,
    quantum_device,
    repetitions: int = 1,
    identity_reference: bool = False,
    acq_protocol: Literal['SSBIntegrationComplex', 'ThresholdedAcquisition'] | str = 'ThresholdedAcquisition',
    bin_mode: Literal['average', 'append'] | str = 'append',
    pad_identity: bool = True,
) -> np.ndarray:
    """
    Compute the iSWAP ``Rz`` corrections for a tomography experiment list
    without touching hardware, for use as ``correction_phase`` in
    ``process_tomography_schedule``.

    This exists so the correction's compile pass is paid once rather than on
    every run: ``process_tomography_schedule`` then builds and compiles a
    single schedule instead of two.

    The correction rule is the same as ``iswap_RPE_f`` -- see that function
    for the derivation and ``_detuning_residual`` for how the pump frequency
    is obtained. Only the gate under test carries an iSWAP (every tomography
    pre/post rotation is single-qubit), and the correction registers restart
    at each block's ``ResetClockPhase``, so each block stands alone.

    The structural arguments -- ``qubit_specifier``, ``operation_state_idx``,
    ``experiment_list``, ``identity_reference`` -- MUST match those passed to
    ``process_tomography_schedule``, otherwise the corrections will not line
    up with the gates. ``process_tomography_schedule`` checks the length as a
    guard, but a same-length mismatch cannot be detected.

    Returns
    -------
    list[tuple[float, float]]
        One ``(theta_q0, theta_q1)`` pair per iSWAP, in chronological order.
        Empty if the schedule contains no iSWAP.

    Notes
    -----
    The result depends on the qubits' calibrated ``clock_freqs.f01()`` and on
    the iSWAP edge's pump frequency, so it goes stale when either is
    recalibrated -- re-run this after retuning.
    """
    cfg = _qpt_setup(qubit_specifier, operation_state_idx, experiment_list, identity_reference)

    if cfg["n"] != 2 or cfg["n_iswap_per_op"] == 0:
        return np.empty((0, 2))
    if cfg["n_iswap_per_op"] > 1:
        raise NotImplementedError(
            f"Clifford {operation_state_idx} decomposes to {cfg['n_iswap_per_op']} iSWAPs. "
            "The correction is inserted after the whole gate subschedule, which is "
            "only equivalent for a single iSWAP; interleaving would require changing "
            "index_to_operation."
        )

    sched = _qpt_build(cfg, operation_state_idx, identity_reference, acq_protocol,
                       bin_mode, repetitions, iswap_phases=None,
                       pad_identity=pad_identity)

    detuning_residual = _detuning_residual(cfg["qubits"], quantum_device)

    ac_stark = _ac_stark_phases(cfg["qubits"], quantum_device)
    t1_list = compute_iswap_t1(sched, quantum_device)

    expected = cfg["n_iswap_total"]
    if len(t1_list) != expected:
        raise ValueError(
            f"Expected {expected} 'snail:switch' pulses in the timing table "
            f"({cfg['n_op_blocks']} blocks x {cfg['n_iswap_per_op']} iSWAP), "
            f"found {len(t1_list)}."
        )

    return np.array(
        [_iswap_rz_corrections([t], detuning_residual, ac_stark)[0] for t in t1_list]
    )


def process_tomography_schedule(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    operation_state_idx: int,
    experiment_list: Iterable,
    repetitions: int = 1,
    identity_reference: bool = False,
    instrument_coordinator=None,
    acq_protocol: Literal['SSBIntegrationComplex', 'ThresholdedAcquisition'] | str = 'ThresholdedAcquisition',
    bin_mode: Literal['average', 'append'] | str = 'append',
    quantum_device=None,
    correction_phase: np.ndarray | None = None,
    pad_identity: bool = True,
):
    """
    Run process tomography for the gate ``operation_state_idx`` and return the
    raw acquisition.

    This compiles and executes internally, so both ``quantum_device`` and
    ``instrument_coordinator`` must be supplied.

    iSWAP phase tracking
    --------------------
    Pass ``correction_phase`` from ``process_tomography_pre_build`` to insert
    the virtual-Z corrections. The corrections are computed there rather than
    here so that this function builds and compiles the schedule exactly once;
    computing them inline would double the compile cost of every run.

    ``correction_phase=None`` is allowed only when the schedule has no iSWAP
    (single-qubit tomography, a gate with no iSWAP) or for the idle reference:
    the same schedule with the pump off, i.e. the edge's ``iswap.pulse_amp``
    at or below ``PUMP_OFF_DBM``. Otherwise it raises -- an uncorrected iSWAP
    misses the AC-Stark phases and the detuning-residual frame tracking.

    Parameters
    ----------
    instrument_coordinator
        Required -- raises if not given. Not defaulted to a module-level
        global, so that importing this module does not depend on instruments
        having been constructed first.
    quantum_device
        Required -- raises if not given.
    correction_phase
        One ``(theta_q0, theta_q1)`` pair per iSWAP, in chronological order,
        as returned by ``process_tomography_pre_build`` called with matching
        structural arguments. Required when the schedule contains iSWAPs and
        the pump is on.
    """
    from quantify_scheduler.backends import SerialCompiler

    if quantum_device is None:
        raise RuntimeError(
            "process_tomography_schedule requires a quantum_device "
            "(pass quantum_device=... )."
        )
    if instrument_coordinator is None:
        raise RuntimeError(
            "process_tomography_schedule requires an instrument_coordinator "
            "(pass instrument_coordinator=... )."
        )

    cfg = _qpt_setup(qubit_specifier, operation_state_idx, experiment_list, identity_reference)

    expected = cfg["n_iswap_total"]
    if correction_phase is not None:
        correction_phase = list(correction_phase)
        if len(correction_phase) != expected:
            raise ValueError(
                f"correction_phase has {len(correction_phase)} entries but this "
                f"schedule contains {expected} iSWAP(s). It must come from "
                "process_tomography_pre_build called with the same "
                "qubit_specifier / operation_state_idx / experiment_list / "
                "identity_reference."
            )
    elif expected:
        # Without correction_phase the gate carries neither the AC-Stark phases nor the
        # detuning-residual frame tracking (2026-10-06: an uncorrected QPT showed
        # theta_sum ~28 deg and an arbitrary theta_d ~65 deg after a nulling calibration).
        # Only the idle reference -- the same schedule with the pump off -- may run without it.
        pump_amp = _iswap_pump_amp(cfg["qubits"], quantum_device)
        if pump_amp > PUMP_OFF_DBM:
            raise ValueError(
                f"Schedule contains {expected} iSWAP(s) but correction_phase is None, so the "
                "gate would run without the AC-Stark and detuning-residual virtual-Z corrections. "
                "Pass correction_phase from process_tomography_pre_build(). Only an idle "
                "reference with the pump off may run uncorrected: set the edge's "
                f"iswap.pulse_amp <= {PUMP_OFF_DBM:g} dBm (and the generator to match; it is "
                f"{pump_amp:g} dBm now)."
            )
        print(
            f"Idle reference: {expected} iSWAP slot(s) with the pump off "
            f"(iswap.pulse_amp = {pump_amp:g} dBm), running without virtual-Z phase correction."
        )

    sched = _qpt_build(cfg, operation_state_idx, identity_reference, acq_protocol,
                       bin_mode, repetitions, iswap_phases=correction_phase,
                       pad_identity=pad_identity)

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(schedule=sched, config=quantum_device.generate_compilation_config())

    instrument_coordinator.prepare(compiled_schedule)
    instrument_coordinator.start()
    instrument_coordinator.wait_done(timeout_sec=300)

    return instrument_coordinator.retrieve_acquisition()


######################################################################################################
# Helper functions for Gate Set Tomography
######################################################################################################

import pygsti

def pygsti_circuit_to_gate_decomposition(circuit, max_qubits=2):
    """
    Convert a pyGSTi Circuit into a list of (qubits, gates) tuples, similar to
    TwoQubitClifford(...).gate_decomposition().

    For 2-qubit circuits, each 1-qubit layer is completed into a full Clifford
    step by inserting an explicit idle on the other qubit.

    Parameters
    ----------
    circuit : pygsti.circuits.Circuit
        Input pyGSTi circuit.
    max_qubits : int, optional
        Maximum supported qubits in the circuit (default: 2).

    Returns
    -------
    list[tuple[tuple[str, ...], list[str]]]
        Example: [(('q0',), ['X90']), (('q1',), ['I']), (('q0', 'q1'), ['CNOT'])]
    """

    gate_name_map = {
        # 1Q idle
        'Gi': 'I',

        # 1Q pi rotations
        'Gx': 'X180',
        'Gy': 'Y180',
        'Gz': 'Z180',

        # 1Q +pi/2 rotations
        'Gxpi2': 'X90',
        'Gypi2': 'Y90',
        'Gzpi2': 'Z90',

        # 1Q -pi/2 rotations
        'Gxmpi2': 'mX90',
        'Gympi2': 'mY90',
        'Gzmpi2': 'mZ90',

        # 2Q gates
        'Gcnot': 'CNOT',
        'Gcphase': 'CZ',
        'Gcz': 'CZ',
        'Giswap': 'iSWAP'
    }

    line_labels = tuple(lbl for lbl in getattr(circuit, 'line_labels', ()) if lbl != '*')
    if line_labels and len(line_labels) > max_qubits:
        raise ValueError(f"This helper supports up to {max_qubits} qubits, got {len(line_labels)}.")

    def _normalize_qubits(raw_qubits):
        if raw_qubits is None:
            return ()
        if isinstance(raw_qubits, (int, np.integer, str)):
            return (raw_qubits,)
        return tuple(raw_qubits)

    def _to_q_label(qubit):
        s = str(qubit)
        return s if s.startswith('q') else f"q{s}"

    def _format_qubits(qubits):
        return tuple(_to_q_label(q) for q in qubits)

    def _parse_string_label(label_str):
        # Example: 'Gcnot:0:1' or 'Gxpi2:0'
        parts = label_str.split(':')
        name = parts[0]
        qubits = tuple(p for p in parts[1:] if p != '')
        return name, qubits

    def _is_empty_token(op):
        if isinstance(op, (list, tuple)) and len(op) == 0:
            return True
        if isinstance(op, str) and op.strip() in {'[]', '()', '{}'}:
            return True
        return False

    def _extract_name_qubits(op):
        # pyGSTi Label-like
        if hasattr(op, 'name'):
            name = op.name
            qubits = _normalize_qubits(getattr(op, 'sslbls', None))
            return name, qubits

        # tuple-like, e.g. ('Gxpi2',0) or ('Gcnot',0,1)
        if isinstance(op, tuple) and len(op) > 0 and isinstance(op[0], str):
            name = op[0]
            qubits = tuple(op[1:])
            return name, qubits

        # string-like, e.g. 'Gxpi2:0'
        if isinstance(op, str):
            return _parse_string_label(op)

        # fallback
        s = str(op)
        return _parse_string_label(s)

    decomposition = []
    line_q_labels = tuple(_to_q_label(lbl) for lbl in line_labels)

    for layer in circuit:
        components = getattr(layer, 'components', None)
        if components is not None:
            primitive_ops = tuple(components)
        elif _is_empty_token(layer):
            primitive_ops = ()
        else:
            primitive_ops = (layer,)

        layer_entries = []
        active_q_labels = set()

        for op in primitive_ops:
            if _is_empty_token(op):
                continue

            name, qubits = _extract_name_qubits(op)

            # For 1Q circuits, unlabeled single-qubit ops can be assigned to the only line.
            if len(qubits) == 0 and len(line_labels) == 1:
                qubits = (line_labels[0],)

            if len(qubits) > max_qubits:
                raise ValueError(f"Gate '{name}' acts on {len(qubits)} qubits; only <= {max_qubits} supported.")

            out_name = gate_name_map.get(name, name.lstrip('G'))
            q_tuple = _format_qubits(qubits)
            layer_entries.append((q_tuple, [out_name]))
            for q in q_tuple:
                active_q_labels.add(q)

        # Fill explicit idle for missing qubit(s) on layers without a 2Q gate.
        has_2q_gate = any(len(qs) == 2 for qs, _ in layer_entries)
        if not has_2q_gate and len(line_q_labels) >= 1:
            for q in line_q_labels:
                if q not in active_q_labels:
                    layer_entries.append(((q,), ['I']))

            # Keep deterministic order q0 then q1
            layer_entries.sort(key=lambda entry: (len(entry[0]) != 1, entry[0]))

        decomposition.extend(layer_entries)

    return decomposition

def pygsti_circuit_to_operation(
    qubit_names: list[str],
    operation_buffer_time: float,
    idle_duration: float,
    pygsti_circuit,
    iswap_phase_iter=None,
 ) -> Schedule | None:
    """
    Convert a pyGSTi circuit into a Quantify Schedule of physical operations.

    This function maps the output of ``pygsti_circuit_to_gate_decomposition`` to Quantify
    operations and assembles them into a schedule with timing constraints for
    single- and two-qubit gates. If the decomposition results in no physical operations,
    ``None`` is returned.

    Parameters
    ----------
    qubit_names : list[str]
        List of qubit names. Length 1 for single-qubit, 2 for two-qubit circuits.
    operation_buffer_time : float
        Buffer time (in seconds) to insert between operations in the schedule.
    idle_duration : float
        Pulse duration used for the "I" (idle) gate. Same derivation as process
        tomography's ``single_qubit_block`` -- ``max(qubit.rxy.duration() for
        qubit in qubits)`` -- computed by the caller (see ``_gst_setup``) rather
        than passed in as a free-standing, externally-configurable argument.
        ``operation_buffer_time`` is NOT folded in here, unlike
        ``single_qubit_block``: it is already applied separately below via
        ``rel_time=operation_buffer_time`` for every gate, "I" included, so
        adding it into ``idle_duration`` too would double it for idles only.
    pygsti_circuit : pygsti.circuits.Circuit
        pyGSTi circuit to decompose and schedule.
    iswap_phase_iter : Iterator[tuple[float, float]] | None, optional
        If given, a shared iterator (created once, outside the per-circuit
        loop, by the caller) yielding one ``(theta_q0, theta_q1)`` virtual-Z
        correction pair per iSWAP, in chronological order across the whole
        experiment. Consumed one pair per iSWAP encountered here, mirroring
        ``_qpt_build``'s ``phase_iter``/``next(phase_iter)`` pattern.

    Returns
    -------
    Schedule | None
        A Quantify Schedule object containing the physical operations for the circuit,
        or None if the decomposition results in no operations.

    Raises
    ------
    NotImplementedError
        If the number of qubits is not 1 or 2.

    """
    if len(qubit_names) not in (1, 2):
        raise NotImplementedError
    # ---- PycQED mappings ----#
    # map the pycqed qubit names to the ones used in quantify
    pycqed_qubit_map = {f"q{idx}": name for idx, name in enumerate(qubit_names)}
    # pycqed returns RB sequences as a list of strings. Map those to quantify operations
    pycqed_operation_map = {
        "I": lambda q: IdlePulse(duration=idle_duration),  # noqa: ARG005
        "X180": lambda q: X(pycqed_qubit_map[q[0]]),
        "X90": lambda q: X90(pycqed_qubit_map[q[0]]),
        "Y180": lambda q: Y(pycqed_qubit_map[q[0]]),
        "Y90": lambda q: Y90(pycqed_qubit_map[q[0]]),
        "mX90": lambda q: Rxy(qubit=pycqed_qubit_map[q[0]], phi=0.0, theta=-90.0),
        "mY90": lambda q: Rxy(qubit=pycqed_qubit_map[q[0]], phi=90.0, theta=-90.0),
        "iSWAP": lambda q: iSWAP(qC=pycqed_qubit_map[q[0]], qT=pycqed_qubit_map[q[1]]),
    }
    gate_decomp = pygsti_circuit_to_gate_decomposition(pygsti_circuit)
    # gate_decomp = TwoQubitClifford(6565).gate_decomposition()
    gate_sched = Schedule()
    ref_op = gate_sched.add(IdlePulse(0.0))
    ref_ops = [ref_op, ref_op]
    # True while the last emitted layer was an iSWAP with no single-qubit layer since.
    prev_layer_iswap = False

    for qubits, gates in gate_decomp:
        if len(qubits) == 0:
            if len(qubit_names) == 1:
                qubits = ("q0",)
            else:
                raise ValueError(
                    "Encountered gate decomposition entry with empty qubit tuple "
                    "for a multi-qubit circuit."
                )

        subsched = Schedule()
        subsched.add(IdlePulse(0.0))
        for gate in gates:
            if gate not in pycqed_operation_map:
                raise ValueError(f"Unsupported gate in decomposition: {gate}")
            op = pycqed_operation_map[gate](qubits)
            if op is not None:
                if gate == 'iSWAP':
                    if prev_layer_iswap and ISWAP_CONSECUTIVE_GAP > 0:
                        subsched.add(IdlePulse(duration=ISWAP_CONSECUTIVE_GAP))
                    subsched.add(op, rel_time=26e-9-iSWAP_DELAY)
                    subsched.add(IdlePulse(duration=iSWAP_DELAY + ISWAP_SETTLE))
                    # Address the pair by qubit_names, NOT by qubits[0]/[1]: the
                    # latter follows the gate label's argument order, so a
                    # "Giswap:1:0" would put the correction on the wrong qubits.
                    # _iswap_rz_corrections returns (theta_q0, theta_q1) indexed
                    # by physical qubit, and it already carries the part-3
                    # AC-Stark phase inside its swap register -- there is
                    # deliberately no separate constant AC_STARK Rz here.
                    if iswap_phase_iter is not None:
                        theta_q0, theta_q1 = next(iswap_phase_iter)
                        subsched.add(Rz(qubit=qubit_names[0], theta=theta_q0))
                        subsched.add(Rz(qubit=qubit_names[1], theta=theta_q1))
                else:
                    subsched.add(op, rel_time=operation_buffer_time)
        if len(subsched.operations) == 1:
            # no gates added, only the initial IdlePulse
            continue
        # Any single-qubit layer (an explicit "I" idle included) separates two iSWAPs by
        # at least one gate slot, so only back-to-back iSWAP layers get the extra gap.
        prev_layer_iswap = len(qubits) == 2 and "iSWAP" in gates
        if qubits == ("q0",):
            schedulable = gate_sched.add(subsched, ref_op=ref_ops[0])
            ref_ops[0] = schedulable
        elif qubits == ("q1",):
            # FIXME: this relies on the fact that single qubit Clifford are ALWAYS defined for both, and ALWAYS in the order q0, q1
            schedulable = gate_sched.add(subsched, ref_op=ref_ops[0], ref_pt="start")
            ref_ops[1] = schedulable
        elif qubits in [("q0", "q1"), ("q1", "q0")]:
            schedulable = gate_sched.add(subsched, ref_op=ref_ops[0])
            # Synchronise against q1's chain too -- the add() above only
            # references q0's. rel_time is 0, NOT operation_buffer_time: the
            # buffer is already applied inside subsched (rel_time on the op
            # itself), so passing it here again stacks on top of it and puts
            # 2 x operation_buffer_time in front of the gate.
            schedulable.add_timing_constraint(0, ref_ops[1])
            ref_ops = [schedulable, schedulable]
    if len(gate_sched.operations) == 1:
        return None
    return gate_sched

def _gst_setup(qubit_specifier, experiment_list):
    """
    Shared argument parsing for ``GST_pre_build`` and ``GST_schedule``.

    Both must derive the schedule's structure identically, otherwise a
    pre-computed correction list would not line up with the gates it is
    consumed against. Mirrors ``_qpt_setup``.
    """
    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit GST supported.")

    marker_duration = max(qubit.measure.pulse_duration() for qubit in qubits) + TWPA_RINGUP + TWPA_TAIL
    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    # "I"-gate pulse duration -- same derivation as process tomography's
    # single_qubit_block (max(qubit.rxy.duration() for qubit in qubits)), but
    # without its operation_buffer_time addend: pygsti_circuit_to_operation
    # already applies that separately, the same way, to every gate including "I".
    idle_duration = max(qubit.rxy.duration() for qubit in qubits)

    experiment_list = list(experiment_list)
    n_iswap_per_circuit = [
        sum(gates.count("iSWAP") for _, gates in pygsti_circuit_to_gate_decomposition(circuit))
        for circuit in experiment_list
    ]

    return {
        "qubits": qubits,
        "qubit_names": qubit_names,
        "n": n,
        "marker_duration": marker_duration,
        "operation_buffer_time": operation_buffer_time,
        "idle_duration": idle_duration,
        "experiment_list": experiment_list,
        "n_iswap_per_circuit": n_iswap_per_circuit,
        "n_iswap_total": sum(n_iswap_per_circuit),
    }


def _gst_build(cfg, acq_protocol, bin_mode, repetitions, iswap_phases=None):
    """
    Build the GST schedule from a ``_gst_setup`` dict. Mirrors ``_qpt_build``.

    If ``iswap_phases`` is given, insert the ``Rz`` correction pair right
    after each iSWAP, consumed in chronological order across the whole
    ``experiment_list`` (spanning circuit boundaries) via a single shared
    iterator -- ``pygsti_circuit_to_operation`` itself does not reset
    anything at a circuit boundary, so this relies on ``iswap_phases``
    already being arranged as one entry per iSWAP in schedule order.
    """
    qubit_names = cfg["qubit_names"]
    operation_buffer_time = cfg["operation_buffer_time"]
    idle_duration = cfg["idle_duration"]

    phase_iter = iter(iswap_phases) if iswap_phases is not None else None

    sched = Schedule(
        "GST on " + " and ".join(qubit_names), repetitions=repetitions
    )

    for acq_idx, experiment_circuit in enumerate(cfg["experiment_list"]):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")

        operation_sched = pygsti_circuit_to_operation(
            qubit_names=qubit_names,
            operation_buffer_time=operation_buffer_time,
            idle_duration=idle_duration,
            pygsti_circuit=experiment_circuit,
            iswap_phase_iter=phase_iter,
        )
        if operation_sched is not None:
            sched.add(operation_sched)

        ro_pulse = sched.add(
            Measure(
                *qubit_names,
                acq_index=acq_idx,
                acq_protocol=acq_protocol,
                bin_mode=bin_mode,
            )
        )
        sched.add(
            MarkerPulse(
                duration=cfg["marker_duration"],
                port=qubit_names[0] + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY - TWPA_RINGUP,
        )

    sched.add(IdlePulse(duration=4e-9))
    return sched


def GST_pre_build(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    experiment_list: Iterable,
    quantum_device,
    repetitions: int = 1,
    acq_protocol: Literal['SSBIntegrationComplex', 'ThresholdedAcquisition'] | str = 'ThresholdedAcquisition',
    bin_mode: Literal['average', 'append'] | str = 'append',
) -> np.ndarray:
    """
    Compute the iSWAP ``Rz`` corrections for a GST experiment list without
    touching hardware, for use as ``correction_phase`` in ``GST_schedule``.
    Mirrors ``process_tomography_pre_build``.

    This exists so the correction's compile pass is paid once rather than on
    every run: ``GST_schedule`` then builds and compiles a single schedule
    instead of two.

    Unlike process tomography (at most one iSWAP per block), a single GST
    circuit can contain several consecutive ``Giswap`` gates (e.g. a repeated
    germ). ``_iswap_rz_corrections``'s phase register must accumulate across
    all iSWAPs WITHIN one circuit and only reset at the next circuit's own
    ``ResetClockPhase``, so the flat, chronological list from
    ``compute_iswap_t1`` (which spans the whole multi-circuit schedule) is
    split back into per-circuit chunks -- sized from each circuit's own
    iSWAP count, via ``pygsti_circuit_to_gate_decomposition`` -- and
    ``_iswap_rz_corrections`` is run once per chunk, restarting its register
    at zero for every circuit.

    The structural arguments -- ``qubit_specifier``, ``experiment_list`` --
    MUST match those passed to ``GST_schedule``, otherwise the corrections
    will not line up with the gates.

    Returns
    -------
    np.ndarray
        One ``(theta_q0, theta_q1)`` pair per iSWAP, in chronological order
        across the whole ``experiment_list``. Empty if the circuits contain
        no iSWAP.

    Notes
    -----
    The result depends on the qubits' calibrated ``clock_freqs.f01()`` and on
    the iSWAP edge's pump frequency, so it goes stale when either is
    recalibrated -- re-run this after retuning.
    """
    cfg = _gst_setup(qubit_specifier, experiment_list)

    if cfg["n"] != 2 or cfg["n_iswap_total"] == 0:
        return np.empty((0, 2))

    sched = _gst_build(cfg, acq_protocol, bin_mode, repetitions, iswap_phases=None)

    detuning_residual = _detuning_residual(cfg["qubits"], quantum_device)

    ac_stark = _ac_stark_phases(cfg["qubits"], quantum_device)
    t1_list = compute_iswap_t1(sched, quantum_device)

    expected = cfg["n_iswap_total"]
    if len(t1_list) != expected:
        raise ValueError(
            f"Expected {expected} 'snail:switch' pulses in the timing table "
            f"(sum of per-circuit iSWAP counts across {len(cfg['n_iswap_per_circuit'])} "
            f"circuits), found {len(t1_list)}."
        )

    corrections = []
    pos = 0
    for n_iswap in cfg["n_iswap_per_circuit"]:
        if n_iswap == 0:
            continue
        block_t1 = t1_list[pos:pos + n_iswap]
        corrections.extend(_iswap_rz_corrections(block_t1, detuning_residual, ac_stark))
        pos += n_iswap

    return np.array(corrections)


def GST_schedule(
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    experiment_list: Iterable,
    repetitions: int = 1,
    instrument_coordinator=None,
    quantum_device=None,
    correction_phase: np.ndarray | None = None,
    acq_protocol: Literal['SSBIntegrationComplex', 'ThresholdedAcquisition'] | str = 'ThresholdedAcquisition',
    bin_mode: Literal['average', 'append'] | str = 'append',
):
    """
    Run GST for ``experiment_list`` and return the raw acquisition. Mirrors
    ``process_tomography_schedule``.

    This compiles and executes internally, so both ``quantum_device`` and
    ``instrument_coordinator`` must be supplied.

    iSWAP phase tracking
    --------------------
    Pass ``correction_phase`` from ``GST_pre_build`` to insert the virtual-Z
    corrections. The corrections are computed there rather than here so that
    this function builds and compiles the schedule exactly once; computing
    them inline would double the compile cost of every run.

    ``correction_phase=None`` runs uncorrected. That is the right choice for
    single-qubit GST, for a gate set with no iSWAP, or for an A/B against the
    corrected result -- but a warning is printed if the circuits contain
    iSWAPs and no correction was supplied, since that is silent otherwise.

    Parameters
    ----------
    instrument_coordinator
        Required -- raises if not given. Not defaulted to a module-level
        global, so that importing this module does not depend on instruments
        having been constructed first.
    quantum_device
        Required -- raises if not given.
    correction_phase
        One ``(theta_q0, theta_q1)`` pair per iSWAP, in chronological order
        across the whole ``experiment_list``, as returned by ``GST_pre_build``
        called with matching structural arguments.
    """
    from quantify_scheduler.backends import SerialCompiler

    if quantum_device is None:
        raise RuntimeError("GST_schedule requires a quantum_device (pass quantum_device=... ).")
    if instrument_coordinator is None:
        raise RuntimeError("GST_schedule requires an instrument_coordinator (pass instrument_coordinator=... ).")

    cfg = _gst_setup(qubit_specifier, experiment_list)

    expected = cfg["n_iswap_total"]
    if correction_phase is not None:
        correction_phase = list(correction_phase)
        if len(correction_phase) != expected:
            raise ValueError(
                f"correction_phase has {len(correction_phase)} entries but this "
                f"experiment_list contains {expected} iSWAP(s). It must come from "
                "GST_pre_build() called with the same qubit_specifier / experiment_list."
            )
    elif expected:
        print(
            f"Warning: experiment_list contains {expected} iSWAP(s) but correction_phase "
            "is None -- running without virtual-Z phase correction. Use "
            "GST_pre_build() to compute it."
        )

    sched = _gst_build(cfg, acq_protocol, bin_mode, repetitions, iswap_phases=correction_phase)

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )

    instrument_coordinator.prepare(compiled_schedule)
    instrument_coordinator.start()
    instrument_coordinator.wait_done(timeout_sec=300)

    return instrument_coordinator.retrieve_acquisition()