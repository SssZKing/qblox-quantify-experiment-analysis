"""iSWAP virtual-Z phase compensation (dissertation section 4.3.2)."""

from __future__ import annotations

import math

from quantify_scheduler import Schedule

from qqea.schedules import gates as _gates

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
        t_rf = t_marker + _gates.iSWAP_DELAY
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
