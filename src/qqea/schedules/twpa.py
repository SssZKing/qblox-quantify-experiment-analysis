"""TWPA pump triggering for readout.

The TWPA pump is gated by a marker on ``<qubit>:switch``. It has to be on before
the readout pulse reaches the TWPA, so the marker starts ``TWPA_RINGUP`` (plus
``TWPA_DELAY``) before the readout and stays on ``TWPA_TAIL`` after it::

    marker start    = readout start + TWPA_DELAY - TWPA_RINGUP
    marker duration = readout duration + TWPA_RINGUP + TWPA_TAIL

Every schedule builder adds its readout through :func:`measure_with_twpa` (gate
level ``Measure``) or :func:`add_twpa_marker` (pulse level readout), so the timing
lives in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantify_scheduler.operations.gate_library import Measure
from quantify_scheduler.operations.pulse_library import MarkerPulse

if TYPE_CHECKING:
    from quantify_scheduler.schedules.schedule import Schedulable, Schedule

TWPA_DELAY = -36e-9
TWPA_RINGUP = 960e-9
TWPA_TAIL = 40e-9

__all__ = [
    "TWPA_DELAY",
    "TWPA_RINGUP",
    "TWPA_TAIL",
    "max_readout_duration",
    "add_twpa_marker",
    "measure_with_twpa",
]


def max_readout_duration(qubits) -> float:
    """Longest ``measure.pulse_duration()`` of one element or an iterable of them."""
    if hasattr(qubits, "name"):
        qubits = [qubits]
    return max(qubit.measure.pulse_duration() for qubit in qubits)


def add_twpa_marker(
    schedule: Schedule,
    ref_op: Schedulable,
    switch_port: str,
    readout_duration: float,
    *,
    delay: float | None = None,
    ref_pt: str = "start",
    label: str | None = None,
) -> Schedulable:
    """Add the TWPA pump marker for a readout that starts at ``ref_op``.

    Parameters
    ----------
    schedule
        Schedule to add the marker to.
    ref_op
        The readout (``Measure``, readout pulse or acquisition) the pump covers.
    switch_port
        Marker port of the TWPA switch, e.g. ``"qubit1:switch"``.
    readout_duration
        Length of the readout the pump has to cover, in seconds.
    delay
        Pump delay in seconds. Defaults to ``TWPA_DELAY``.
    ref_pt
        Point of ``ref_op`` the readout starts at (default ``"start"``).
    label
        Optional label of the marker.
    """
    if delay is None:
        delay = TWPA_DELAY
    return schedule.add(
        MarkerPulse(
            duration=readout_duration + TWPA_RINGUP + TWPA_TAIL,
            port=switch_port,
        ),
        ref_op=ref_op,
        ref_pt=ref_pt,
        rel_time=delay - TWPA_RINGUP,
        label=label,
    )


def measure_with_twpa(
    schedule: Schedule,
    *qubits: str,
    readout_duration: float,
    rel_time: float = 0,
    ref_op: Schedulable | str | None = None,
    ref_pt: str | None = None,
    label: str | None = None,
    marker_label: str | None = None,
    **measure_kwargs,
) -> Schedulable:
    """Add ``Measure(*qubits, **measure_kwargs)`` and the TWPA pump marker for it.

    The marker goes on the first qubit's ``<qubit>:switch`` port and is added right
    after the ``Measure``, so the next operation added without a ``ref_op`` follows
    the marker, as the hand-written blocks this replaces did.

    Parameters
    ----------
    schedule
        Schedule to add the readout to.
    *qubits
        Element names to measure. Multiplexed readout passes several.
    readout_duration
        Readout length the pump covers, normally :func:`max_readout_duration` of the
        measured elements.
    rel_time, ref_op, ref_pt, label
        Passed to ``schedule.add`` for the ``Measure``.
    marker_label
        Optional label of the marker.
    **measure_kwargs
        Passed to ``Measure`` (``acq_index``, ``acq_protocol``, ``bin_mode``, ...).

    Returns
    -------
    Schedulable
        The ``Measure``.
    """
    ro_pulse = schedule.add(
        Measure(*qubits, **measure_kwargs),
        rel_time=rel_time,
        ref_op=ref_op,
        ref_pt=ref_pt,
        label=label,
    )
    add_twpa_marker(
        schedule, ro_pulse, qubits[0] + ":switch", readout_duration, label=marker_label
    )
    return ro_pulse
