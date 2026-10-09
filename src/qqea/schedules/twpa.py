"""TWPA pump triggering for readout.

The TWPA pump is gated by a marker on ``<qubit>:switch``. It has to be on before
the readout pulse reaches the TWPA, so the marker starts ``TWPA_RINGUP`` (plus
``TWPA_DELAY``) before the readout and stays on ``TWPA_TAIL`` after it::

    marker start    = readout start + TWPA_DELAY - TWPA_RINGUP
    marker duration = readout duration + TWPA_RINGUP + TWPA_TAIL

Every schedule builder adds its readout through :func:`measure_with_twpa` (gate
level ``Measure``) or :func:`add_twpa_marker` (pulse level readout), so the timing
lives in one place.

Whether a qubit's readout pumps the TWPA is a per-qubit setting, the element
parameter ``twpa_pump``::

    from qqea.schedules.twpa import set_twpa_pump
    set_twpa_pump(qubit1, True)    # readouts of qubit1 trigger the pump
    set_twpa_pump(qubit2, False)   # qubit2 is read out without it

``set_twpa_pump`` adds the parameter to the element the first time, so it shows
up in every dataset snapshot. A qubit that never had it set pumps the TWPA, as
all readouts did before the setting existed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qcodes.instrument import Instrument
from qcodes.parameters import ManualParameter
from qcodes.validators import Bool
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
    "set_twpa_pump",
    "twpa_pump_enabled",
    "max_readout_duration",
    "add_twpa_marker",
    "measure_with_twpa",
]


def set_twpa_pump(element, enabled: bool = True) -> None:
    """Set whether readouts of ``element`` trigger the TWPA pump.

    Adds a ``twpa_pump`` parameter to the element the first time it is called.
    """
    if "twpa_pump" not in element.parameters:
        element.add_parameter(
            "twpa_pump",
            parameter_class=ManualParameter,
            initial_value=True,
            vals=Bool(),
            docstring="Whether readouts of this element trigger the TWPA pump "
            "(marker on <element>:switch). See qqea.schedules.twpa.",
        )
    element.twpa_pump(enabled)


def twpa_pump_enabled(qubit) -> bool:
    """Whether readouts of ``qubit`` (an element or its name) trigger the TWPA pump.

    True unless the element's ``twpa_pump`` parameter is set to False. A name with
    no element behind it (e.g. a bare port prefix) also counts as pumped.
    """
    if isinstance(qubit, str):
        try:
            qubit = Instrument.find_instrument(qubit)
        except KeyError:
            return True
    param = qubit.parameters.get("twpa_pump")
    return True if param is None else bool(param())


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

    Returns
    -------
    Schedulable or None
        The marker, or None if the element behind ``switch_port`` has its TWPA
        pump turned off (:func:`set_twpa_pump`).
    """
    if not twpa_pump_enabled(switch_port.split(":")[0]):
        return None
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

    The marker goes on the ``<qubit>:switch`` port of the first measured qubit whose
    TWPA pump is on (:func:`set_twpa_pump`); with none on, no marker is added. It is
    added right after the ``Measure``, so the next operation added without a
    ``ref_op`` follows the marker, as the hand-written blocks this replaces did.

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
    pumped = [q for q in qubits if twpa_pump_enabled(q)]
    if pumped:
        add_twpa_marker(
            schedule, ro_pulse, pumped[0] + ":switch", readout_duration, label=marker_label
        )
    return ro_pulse
