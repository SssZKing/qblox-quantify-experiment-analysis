"""Hardware runners: build a schedule, compile it and run it on the instrument
coordinator directly, returning the raw ``retrieve_acquisition()`` data.

These bypass ``meas_ctrl``/``ScheduleGettable`` because the fits need the raw
complex acquisitions (``BinMode.APPEND``, Trace). Readouts go through
:func:`qqea.schedules.twpa.measure_with_twpa`, so each qubit's ``twpa_pump``
setting decides whether the pump is gated on. They moved here from
``qqea.schedules.single_qubit``. The readout runs dropped their old ``_TWPA``
suffix, with no aliases: ``readout_freq_optimization_TWPA`` is now
``readout_freq_optimization``, and likewise for amp, len and weight. Each runner is split into a ``*_schedule`` builder,
which only builds the schedule, and the run function, which keeps the old
signature and return value.

``CalibrationNodes`` saves what these return with
:func:`qqea.experiments.saving.save_retrieve_acquisition_dataset`; calling a
runner directly saves nothing, as before.
"""

from typing import Union

import numpy as np
from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.gate_library import Reset, Rxy, X
from quantify_scheduler.operations.pulse_library import IdlePulse, SetClockFrequency
from quantify_scheduler.schedules.schedule import Schedule

from qqea.schedules.twpa import measure_with_twpa

__all__ = [
    "readout_freq_optimization_schedule",
    "readout_amp_optimization_schedule",
    "readout_len_optimization_schedule",
    "readout_weight_optimization_schedule",
    "drag_calibration_schedule",
    "run_compiled",
    "readout_freq_optimization",
    "readout_amp_optimization",
    "readout_len_optimization",
    "readout_weight_optimization",
    "DRAG_calibration_sched",
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _as_1d(values) -> np.ndarray:
    """``values`` as a 1-D array; a scalar becomes a one-element array."""
    arr = np.asarray(values)
    return arr.reshape(arr.shape or (1,))


def run_compiled(schedule, quantum_device, instrument_coordinator, timeout_sec):
    """Compile ``schedule`` for ``quantum_device``, run it and return the acquisitions."""
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )
    instrument_coordinator.prepare(compiled_schedule)
    instrument_coordinator.start()
    instrument_coordinator.wait_done(timeout_sec=timeout_sec)
    return instrument_coordinator.retrieve_acquisition()


def _add_ge_pair(schedule, qubit_name, acq_index, readout_duration, **measure_kwargs):
    """Measure the qubit in |g> (acq_index) and then in |e> (acq_index + 1),
    with the TWPA pump gated on if the qubit's ``twpa_pump`` setting allows."""
    for excited in (False, True):
        schedule.add(Reset(qubit_name))
        if excited:
            schedule.add(X(qubit=qubit_name))
        measure_with_twpa(
            schedule,
            qubit_name,
            readout_duration=readout_duration,
            marker_label=f"TWPA_mark {acq_index + excited}",
            acq_index=acq_index + excited,
            acq_protocol="SSBIntegrationComplex",
            bin_mode=BinMode.APPEND,
            **measure_kwargs,
        )


# ----------------------------------------------------------------------------
# schedule builders
# ----------------------------------------------------------------------------
def readout_freq_optimization_schedule(qubit, frequencies, repetitions=1) -> Schedule:
    """|g>/|e> readout pairs, one per readout frequency (acq_index 2i and 2i+1)."""
    readout_duration = qubit.measure.pulse_duration()
    name = qubit.name
    schedule = Schedule("Readout frequency optimization", repetitions)
    for i, freq in enumerate(_as_1d(frequencies)):
        schedule.add(SetClockFrequency(clock=f"{name}.ro", clock_freq_new=freq))
        _add_ge_pair(schedule, name, 2 * i, readout_duration)
    schedule.add(IdlePulse(duration=4e-9), label="end")
    return schedule


def readout_amp_optimization_schedule(qubit, amps, repetitions=1) -> Schedule:
    """|g>/|e> readout pairs, one per readout pulse amplitude."""
    readout_duration = qubit.measure.pulse_duration()
    name = qubit.name
    schedule = Schedule("Readout amplitude optimization", repetitions)
    for i, amp in enumerate(_as_1d(amps)):
        _add_ge_pair(schedule, name, 2 * i, readout_duration, pulse_amp=amp)
    schedule.add(IdlePulse(duration=4e-9), label="end")
    return schedule


def readout_len_optimization_schedule(qubit, lens, repetitions=1) -> Schedule:
    """|g>/|e> readout pairs, one per readout duration (pulse and integration together)."""
    name = qubit.name
    schedule = Schedule("Readout length optimization", repetitions)
    for i, length in enumerate(_as_1d(lens)):
        _add_ge_pair(
            schedule, name, 2 * i, length,
            pulse_duration=length, integration_time=length,
        )
    schedule.add(IdlePulse(duration=4e-9), label="end")
    return schedule


def readout_weight_optimization_schedule(qubit, excited, repetitions=1) -> Schedule:
    """One averaged Trace acquisition of the qubit in |g> (or |e> if ``excited``)."""
    name = qubit.name
    schedule = Schedule("Readout weight optimization", repetitions)
    schedule.add(Reset(name))
    if excited:
        schedule.add(X(qubit=name))
    measure_with_twpa(
        schedule, name, readout_duration=qubit.measure.pulse_duration(),
        acq_channel=0, acq_protocol="Trace", bin_mode=BinMode.AVERAGE,
    )
    schedule.add(IdlePulse(duration=4e-9), label="end")
    return schedule


# The two AllXY elements DRAG_fit compares: equally sensitive to the DRAG
# phase error, with opposite sign.
_DRAG_PAIRS = (
    ((180, 0), (90, 90)),
    ((180, 90), (90, 0)),
)


def drag_calibration_schedule(qubit, motzoi_list, repetitions=1) -> Schedule:
    """For each motzoi, the two DRAG-sensitive AllXY elements (acq_index 2i and 2i+1)."""
    readout_duration = qubit.measure.pulse_duration()
    name = qubit.name
    schedule = Schedule("DRAG calibration", repetitions)
    for i, m in enumerate(_as_1d(motzoi_list)):
        for k, ((th0, phi0), (th1, phi1)) in enumerate(_DRAG_PAIRS):
            schedule.add(Reset(name))
            schedule.add(Rxy(qubit=name, theta=th0, phi=phi0, motzoi=m))
            schedule.add(Rxy(qubit=name, theta=th1, phi=phi1, motzoi=m))
            measure_with_twpa(schedule, name, readout_duration=readout_duration,
                              marker_label=f"TWPA_mark {2 * i + k}", acq_index=2 * i + k)
    schedule.add(IdlePulse(duration=4e-9), label="end")
    return schedule


# ----------------------------------------------------------------------------
# runners (same signatures and return values as before the move)
# ----------------------------------------------------------------------------
def readout_freq_optimization(
    qubit,
    frequencies: Union[np.ndarray, float],
    instrument_coordinator,
    quantum_device,
    repetitions: int = 1,
):
    schedule = readout_freq_optimization_schedule(qubit, frequencies, repetitions)
    return run_compiled(schedule, quantum_device, instrument_coordinator, timeout_sec=90)


def readout_amp_optimization(
    qubit,
    amps: Union[np.ndarray, float],
    instrument_coordinator,
    quantum_device,
    repetitions: int = 1,
):
    schedule = readout_amp_optimization_schedule(qubit, amps, repetitions)
    return run_compiled(schedule, quantum_device, instrument_coordinator, timeout_sec=90)


def readout_len_optimization(
    qubit,
    lens: Union[np.ndarray, float],
    instrument_coordinator,
    quantum_device,
    repetitions: int = 1,
):
    schedule = readout_len_optimization_schedule(qubit, lens, repetitions)
    return run_compiled(schedule, quantum_device, instrument_coordinator, timeout_sec=90)


def readout_weight_optimization(
    qubit,
    instrument_coordinator,
    quantum_device,
    repetitions: int = 1,
):
    """Return ``(g_trace, e_trace)``, two separate averaged Trace runs."""
    g_trace, e_trace = (
        run_compiled(
            readout_weight_optimization_schedule(qubit, excited, repetitions),
            quantum_device, instrument_coordinator, timeout_sec=30,
        )
        for excited in (False, True)
    )
    return g_trace, e_trace


def DRAG_calibration_sched(
    qubit,
    motzoi_list: Union[np.ndarray, float],
    instrument_coordinator,
    quantum_device,
    repetitions: int = 1,
):
    schedule = drag_calibration_schedule(qubit, motzoi_list, repetitions)
    return run_compiled(schedule, quantum_device, instrument_coordinator, timeout_sec=60)
