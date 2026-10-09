"""The iSWAP gate and the Clifford-to-operation mapping shared by all two-qubit builders."""

from __future__ import annotations

import numpy as np
from quantify_scheduler import Schedule
from quantify_scheduler.operations import X90, Y90, IdlePulse, Operation, Rxy, X, Y

from qqea.schedules.clifford.clifford_group import (
    SingleQubitClifford,
    TwoQubitClifford,
)

# iSWAP_DELAY = 38e-9
iSWAP_DELAY = 120e-9


# Extra idle after every iSWAP (and sqrt-iSWAP) before the next operation, on top of the
# iSWAP_DELAY that waits for the RF to switch off. Measured 2026-10-02: a pump pulse changes
# the sum Stark phase of the NEXT iSWAP with a ~20 ns time constant -- back-to-back iSWAPs
# (26 ns RF gap) carry ~28 deg less sum phase per pair than iSWAPs >= ~100 ns apart, and a
# train of them shows a ~25 deg one-time phase. 0 keeps the original timing; ~100e-9 makes
# every iSWAP see the same (isolated) context. Read at call time, so it can be changed live:
# set qqea.schedules.gates.ISWAP_SETTLE (setting the name re-exported by two_qubit has no effect).
ISWAP_SETTLE = 0.0


# Extra idle inserted only between CONSECUTIVE iSWAPs (no single-qubit layer between them) in
# pyGSTi circuits, so that every iSWAP in GST sees the isolated context the Stark phases are
# calibrated in. 1Q gate -> iSWAP and iSWAP -> 1Q gate need no extra time (checked 2026-10-02
# with a software iSWAP_DELAY scan: >= 15 ns margin before, >= 40 ns after). Measured memory of
# a pump pulse on the next iSWAP: 24 deg/pair with tau ~10 ns plus 10 deg/pair with tau ~80 ns;
# 100 ns leaves ~2.9 deg/pair (~1.4 deg/gate).
# Read at call time; change it live through qqea.schedules.gates.ISWAP_CONSECUTIVE_GAP.
ISWAP_CONSECUTIVE_GAP = 100e-9


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
