"""Gate set tomography schedules from pyGSTi circuits."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
from quantify_scheduler import Schedule
from quantify_scheduler.backends.qblox.constants import MIN_TIME_BETWEEN_OPERATIONS
from quantify_scheduler.operations import (
    X90,
    Y90,
    IdlePulse,
    Reset,
    ResetClockPhase,
    Rxy,
    Rz,
    X,
    Y,
)

from qqea.schedules import gates as _gates
from qqea.schedules.gates import iSWAP
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
                    if prev_layer_iswap and _gates.ISWAP_CONSECUTIVE_GAP > 0:
                        subsched.add(IdlePulse(duration=_gates.ISWAP_CONSECUTIVE_GAP))
                    subsched.add(op, rel_time=26e-9-_gates.iSWAP_DELAY)
                    subsched.add(IdlePulse(duration=_gates.iSWAP_DELAY + _gates.ISWAP_SETTLE))
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

    readout_duration = max_readout_duration(qubits)
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
        "readout_duration": readout_duration,
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
