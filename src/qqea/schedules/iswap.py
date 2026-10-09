"""iSWAP characterisation schedules: pulsed pump, pump-probe, phase tracking and RPE."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
from quantify_scheduler import Schedule
from quantify_scheduler.backends.qblox.constants import MIN_TIME_BETWEEN_OPERATIONS
from quantify_scheduler.operations import (
    IdlePulse,
    MarkerPulse,
    Reset,
    ResetClockPhase,
    Rxy,
    Rz,
    ShiftClockPhase,
)

from qqea.schedules import gates as _gates
from qqea.schedules.clifford.clifford_group import (
    common_cliffords,
)
from qqea.schedules.gates import index_to_operation
from qqea.schedules.phase import (
    _ac_stark_phases,
    _detuning_residual,
    _iswap_rz_corrections,
    compute_iswap_t1,
)
from qqea.schedules.twpa import max_readout_duration, measure_with_twpa

if TYPE_CHECKING:
    from quantify_scheduler.device_under_test.transmon_element import (
        BasicTransmonElement,
    )


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
    readout_duration = max_readout_duration(qubits)

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
        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=acq_idx,
            acq_protocol=acq_protocol,
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
    readout_duration = max_readout_duration(qubits)
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
        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=acq_idx,
            acq_protocol=acq_protocol,
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
    readout_duration = max_readout_duration(qubits)
    sched = Schedule("Pump-probe Ramsey on " + q, repetitions=repetitions)
    for acq_idx, delay in enumerate(delays):
        qubit_reset = sched.add(Reset(*qubit_names))
        for qubit_name in qubit_names:
            sched.add(ResetClockPhase(clock=f"{qubit_name}.01"), ref_op=qubit_reset, ref_pt="end")
        if pump:
            sched.add(MarkerPulse(duration=pump_duration, port="snail:switch"), rel_time=26e-9 - _gates.iSWAP_DELAY)
            sched.add(IdlePulse(duration=_gates.iSWAP_DELAY))
        else:
            sched.add(IdlePulse(duration=26e-9 + pump_duration))
        if delay > 0:
            sched.add(IdlePulse(duration=delay))
        sched.add(Rxy(theta=90, phi=0, qubit=q))
        if window > 0:
            sched.add(IdlePulse(duration=window))
        sched.add(Rxy(theta=90, phi=0 if final_axis == "x" else 90, qubit=q))
        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=acq_idx,
            acq_protocol=acq_protocol,
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
    readout_duration = max_readout_duration(qubits)
    sched = Schedule("Pump-probe swap", repetitions=repetitions)

    def _pump(duration):
        sched.add(MarkerPulse(duration=duration, port="snail:switch"), rel_time=26e-9 - _gates.iSWAP_DELAY)
        sched.add(IdlePulse(duration=_gates.iSWAP_DELAY))

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
        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=acq_idx,
            acq_protocol=acq_protocol,
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
    readout_duration = max_readout_duration(qubits)

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

        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=acq_idx,
            acq_protocol=acq_protocol,
        )

    sched.add(IdlePulse(duration=4e-9))
    return sched


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
    readout_duration = max_readout_duration(qubits)

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

            measure_with_twpa(
                sched,
                *qubit_names,
                readout_duration=readout_duration,
                acq_index=acq_idx,
                acq_protocol=acq_protocol,
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
    readout_duration = max_readout_duration(qubits)

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

            measure_with_twpa(
                sched,
                *qubit_names,
                readout_duration=readout_duration,
                acq_index=acq_idx,
                acq_protocol=acq_protocol,
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
    readout_duration = max_readout_duration(qubits)

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
            sched.add(MarkerPulse(duration=sqrt_iswap_duration, port="snail:switch"), rel_time=26e-9-_gates.iSWAP_DELAY)
            sched.add(IdlePulse(duration=_gates.iSWAP_DELAY + _gates.ISWAP_SETTLE))
            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=3))

            for m_i, clifford_gate_idx in enumerate(range(m)):
                sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=5760))
                if phase_iter is not None:
                    theta_q0, theta_q1 = next(phase_iter)
                    sched.add(Rz(qubit=qubit_names[0], theta=theta_q0))
                    sched.add(Rz(qubit=qubit_names[1], theta=theta_q1))

            sched.add(index_to_operation(qubit_names, operation_buffer_time, clifford_gate_idx=3))
            sched.add(MarkerPulse(duration=sqrt_iswap_duration, port="snail:switch"), rel_time=26e-9-_gates.iSWAP_DELAY)
            sched.add(IdlePulse(duration=_gates.iSWAP_DELAY + _gates.ISWAP_SETTLE))

            measure_with_twpa(
                sched,
                *qubit_names,
                readout_duration=readout_duration,
                acq_index=acq_idx,
                acq_protocol=acq_protocol,
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
    readout_duration = max_readout_duration(qubits)

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
        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=acq_idx,
            acq_protocol=acq_protocol,
            ref_op=swap_pulse,
            ref_pt="end",
            rel_time=0,
        )

    sched.add(IdlePulse(duration=4e-9))
    return sched
