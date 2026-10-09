"""Calibration nodes extracted from the SNL315 notebook.

Each measurement/calibration routine ("node") is a method of
:class:`CalibrationNodes`. The shared runtime instruments
(``quantum_device``, ``meas_ctrl``, ``instrument_coordinator``) and constants
(``ACQ_DELAY``) are held on the instance, so the notebook only needs to
construct the object once::

    from qqea.experiments.calibration_nodes import CalibrationNodes
    nodes = CalibrationNodes(quantum_device, meas_ctrl, ACQ_DELAY, instrument_coordinator=ic)
    ok, fr = nodes.resonator_calibration(qubit1, amp=..., duration=...)

``instrument_coordinator`` is only needed by the nodes that drive the
instrument coordinator directly instead of going through ``meas_ctrl``
(``drag_calibration`` and the ``ro_*_optimization``/``ro_weight_calibration``
nodes); it can be omitted if you never call those.

The ``qubit`` object stays a per-call argument since it changes between nodes.

Every node returns ``(ok, value)`` and does not raise on a bad fit. Each call
is also appended to ``nodes.history`` (time, node, qubits, ok, value, and the
TUIDs of the datasets it wrote).

The methods are grouped by topic in :mod:`qqea.experiments.nodes`:
``single_qubit``, ``readout``, ``iswap_stark`` and ``roadmap``.
"""

from qqea.experiments.nodes.iswap_stark import IswapStarkNodes
from qqea.experiments.nodes.readout import ReadoutNodes
from qqea.experiments.nodes.roadmap import RoadmapNodes
from qqea.experiments.nodes.single_qubit import SingleQubitNodes

__all__ = ["CalibrationNodes"]


class CalibrationNodes(SingleQubitNodes, ReadoutNodes, IswapStarkNodes, RoadmapNodes):
    """Container for calibration nodes sharing the same runtime instruments.

    Parameters
    ----------
    quantum_device : QuantumDevice
    meas_ctrl : MeasurementControl
    acq_delay : float
        Acquisition delay (time of flight) in seconds, used by resonator spectroscopy.
    instrument_coordinator : InstrumentCoordinator, optional
        Needed only by the nodes that bypass ``meas_ctrl``.
    """
