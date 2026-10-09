"""State and process tomography schedules."""

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
    Rz,
)

from qqea.schedules import gates as _gates
from qqea.schedules import phase as _phase
from qqea.schedules.clifford.clifford_group import (
    TwoQubitClifford,
)
from qqea.schedules.gates import index_to_operation
from qqea.schedules.phase import (
    _ac_stark_phases,
    _detuning_residual,
    _iswap_pump_amp,
    _iswap_rz_corrections,
    compute_iswap_t1,
)
from qqea.schedules.twpa import max_readout_duration, measure_with_twpa

if TYPE_CHECKING:
    from quantify_scheduler.device_under_test.transmon_element import (
        BasicTransmonElement,
    )

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

    readout_duration = max_readout_duration(qubits)
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
        "readout_duration": readout_duration,
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
                          rel_time=26e-9 - _gates.iSWAP_DELAY)
                sched.add(IdlePulse(duration=_gates.iSWAP_DELAY + _gates.ISWAP_SETTLE))
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

        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=cfg["readout_duration"],
            acq_index=acq_idx,
            acq_protocol=acq_protocol,
            bin_mode=bin_mode,
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

    readout_duration = max_readout_duration(qubits)
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
        "readout_duration": readout_duration,
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

        measure_with_twpa(
            sched,
            *qubit_names,
            readout_duration=cfg["readout_duration"],
            acq_index=acq_idx,
            acq_protocol=acq_protocol,
            bin_mode=bin_mode,
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
        if pump_amp > _phase.PUMP_OFF_DBM:
            raise ValueError(
                f"Schedule contains {expected} iSWAP(s) but correction_phase is None, so the "
                "gate would run without the AC-Stark and detuning-residual virtual-Z corrections. "
                "Pass correction_phase from process_tomography_pre_build(). Only an idle "
                "reference with the pump off may run uncorrected: set the edge's "
                f"iswap.pulse_amp <= {_phase.PUMP_OFF_DBM:g} dBm (and the generator to match; it is "
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
