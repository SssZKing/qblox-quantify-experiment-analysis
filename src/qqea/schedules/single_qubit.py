from __future__ import annotations

from typing import Optional, List, Literal, Union

import numpy as np

from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex, Trace, NumericalWeightedIntegration
from quantify_scheduler.operations.gate_library import X90, Measure, Reset, Rxy, X, Y
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    SetClockFrequency,
    ResetClockPhase,
    SquarePulse,
    MarkerPulse,
    DRAGPulse,
    VoltageOffset,
)
from quantify_scheduler.qblox.operations import ConditionalReset
from quantify_scheduler.backends.qblox.operations.rf_switch_toggle import RFSwitchToggle
from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.operations.shared_native_library import SpectroscopyOperation
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.schedule import Schedule

from qqea.schedules.twpa import (
    TWPA_DELAY,
    TWPA_RINGUP,
    TWPA_TAIL,
    add_twpa_marker,
    max_readout_duration,
    measure_with_twpa,
)

STARK_LEAD = 10e-6  # Stark tone turns on this long before the end of Reset

# e-f guassian pulse
# E2F_DURATION=400e-9
# E2F_G_AMP=0.22474349565127685

# e-f guassian pulse Q1
E2F_DURATION = 400e-9
E2F_G_AMP = 0.28971234624362635

# e-f guassian pulse Q2
# E2F_DURATION = 200e-9
# E2F_G_AMP = 0.38808960669708015

# # e-f square pulse
# E2F_DURATION=400e-9
# E2F_G_AMP=0.07048438715164222

RESET_CHANNEL_0 = 4
RESET_CHANNEL_1 = 5

def time_of_flight_calibration(
    port: str,
    clock: str,
    pulse_duration: float,
    pulse_amp: float,
    acq_duration: float,
    time_of_flight: float,
    repetitions: int = 1,
    TWPA_delay: float=0,
    pulse_type: Literal[
        "Square", "DRAG"
    ] = "Square",
) -> Schedule:

    schedule = Schedule("Time of flight")
    schedule.add(IdlePulse(duration=1e-6))

    if pulse_type=="Square":
        spec_pulse = schedule.add(
            SquarePulse(
                duration=pulse_duration,
                amp=pulse_amp,
                port=port,
                clock=clock,
            ),
        )
    elif pulse_type=="DRAG":
        spec_pulse = schedule.add(
            DRAGPulse(
                G_amp=pulse_amp,
                D_amp=0,
                duration=pulse_duration,
                phase=0,
                port=port,
                clock=clock,
            ),
        )
        
    schedule.add(
        Trace(
            duration=acq_duration,
            port=port,
            clock=clock,
            acq_channel=0,
            acq_index=0,
        ),
        ref_pt="start",
        rel_time=time_of_flight
    )

    add_twpa_marker(schedule, spec_pulse, port[:-3] + "switch", pulse_duration, delay=TWPA_delay)

    schedule.add(IdlePulse(duration=4e-9))

    return schedule

def sweep_optimal_TWPA(
    TWPA_pump,
    TWPA_freq: float,
    TWPA_power: float,
    pulse_amp: float,
    pulse_duration: float,
    frequencies: np.ndarray,
    acquisition_delay: float,
    integration_time: float,
    port: str,
    clock: str,
    init_duration: float = 10e-6,
    repetitions: int = 1,
    port_out: Optional[str] = None,
) -> Schedule:
    
    schedule = Schedule("Sweep power and frequency for TWPA", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequencies.flat[0]))

    if port_out is None:
        port_out = port

    TWPA_pump.frequency(TWPA_freq)
    TWPA_pump.power(TWPA_power)
    
    for i, freq in enumerate(frequencies):
        schedule.add(IdlePulse(duration=init_duration), label=f"buffer {i}")

        schedule.add(
            SetClockFrequency(clock=clock, clock_freq_new=freq),
            label=f"set_freq {i} ({clock} {freq:e} Hz)",
        )

        spec_pulse = schedule.add(
            SquarePulse(
                duration=pulse_duration,
                amp=pulse_amp,
                port=port_out,
                clock=clock,
            ),
            label=f"spec_pulse {i})",
        )

        schedule.add(
            SSBIntegrationComplex(
                duration=integration_time,
                port=port,
                clock=clock,
                acq_index=i,
                acq_channel=0,
                bin_mode=BinMode.AVERAGE,
            ),
            ref_op=spec_pulse,
            ref_pt="start",
            rel_time=acquisition_delay,
            label=f"acquisition {i})",
        )

        add_twpa_marker(
            schedule,
            spec_pulse,
            port[:-3] + "switch",
            integration_time,
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule

def heterodyne_spec_sched_nco(
    pulse_amp: float,
    pulse_duration: float,
    frequencies: np.ndarray,
    acquisition_delay: float,
    integration_time: float,
    port: str,
    clock: str,
    init_duration: float = 10e-6,
    repetitions: int = 1,
    port_out: Optional[str] = None,
) -> Schedule:
    
    schedule = Schedule("Fast heterodyne spectroscopy (NCO sweep)(TWPA)", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequencies.flat[0]))

    if port_out is None:
        port_out = port

    for i, freq in enumerate(frequencies):
        schedule.add(IdlePulse(duration=init_duration), label=f"buffer {i}")

        schedule.add(
            SetClockFrequency(clock=clock, clock_freq_new=freq),
            label=f"set_freq {i} ({clock} {freq:e} Hz)",
        )
        
        spec_pulse = schedule.add(
            SquarePulse(
                duration=pulse_duration,
                amp=pulse_amp,
                port=port_out,
                clock=clock,
            ),
            label=f"spec_pulse {i})",
        )

        schedule.add(
            SSBIntegrationComplex(
                duration=integration_time,
                port=port,
                clock=clock,
                acq_index=i,
                acq_channel=0,
                bin_mode=BinMode.AVERAGE,
            ),
            ref_op=spec_pulse,
            ref_pt="start",
            rel_time=acquisition_delay,
            label=f"acquisition {i})",
        )

        add_twpa_marker(
            schedule,
            spec_pulse,
            port[:-3] + "switch",
            integration_time,
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule

# def multiplexed_heterodyne_spec_sched_nco(
#     pulse_amp: float,
#     pulse_duration: float,
#     frequencies_qubit1: np.ndarray,
#     frequencies_qubit2: np.ndarray,
#     acquisition_delay: float,
#     integration_time: float,
#     ports,
#     clocks,
#     init_duration: float = 10e-6,
#     repetitions: int = 1,
#     port_out: Optional[str] = None,
# ) -> Schedule:
    
#     schedule = Schedule("Multiplexed heterodyne spectroscopy (NCO sweep)(TWPA)", repetitions)

#     if port_out is None:
#             port_out = ports
        
#     for acq_channel, clock in enumerate(clocks):
#         if acq_channel == 0:
#             frequencies = frequencies_qubit1
#         if acq_channel == 1:
#             frequencies = frequencies_qubit2

#         schedule.add_resource(ClockResource(name=clock, freq=frequencies.flat[0]))

#         for i, freq in enumerate(frequencies):
#             schedule.add(IdlePulse(duration=init_duration))

#             schedule.add(
#                 SetClockFrequency(clock=clocks[acq_channel], clock_freq_new=freq),
#             )

#             spec_pulse = schedule.add(
#                 SquarePulse(
#                     duration=pulse_duration,
#                     amp=pulse_amp,
#                     port=port_out[acq_channel],
#                     clock=clocks[acq_channel],
#                 ),
#             )

#             schedule.add(
#                 SSBIntegrationComplex(
#                     duration=integration_time,
#                     port=ports[acq_channel],
#                     clock=clocks[acq_channel],
#                     acq_index=i,
#                     acq_channel=acq_channel,
#                     bin_mode=BinMode.AVERAGE,
#                 ),
#                 ref_op=spec_pulse,
#                 ref_pt="start",
#                 rel_time=acquisition_delay,
#             )

#             if acq_channel == 0:
#                 marker_pulse = schedule.add(
#                     MarkerPulse(
#                         duration=integration_time+TWPA_RINGUP+TWPA_TAIL, 
#                         port=ports[0][:-3] + "switch",
#                     ),
#                     ref_op=spec_pulse,
#                     ref_pt="start",
#                     rel_time=TWPA_DELAY-TWPA_RINGUP,
#                 )
    
#                 schedule.add(IdlePulse(duration=4e-9))
        
#     return schedule

def multiplexed_heterodyne_spec_sched_nco(
    pulse_amp: float,
    pulse_duration: float,
    frequencies_qubit1: np.ndarray,
    frequencies_qubit2: np.ndarray,
    frequencies_idx: np.ndarray,
    acquisition_delay: float,
    integration_time: float,
    ports,
    clocks,
    init_duration: float = 10e-6,
    repetitions: int = 1,
    port_out: Optional[str] = None,
) -> Schedule:
    
    schedule = Schedule("Multiplexed heterodyne spectroscopy (NCO sweep)(TWPA)", repetitions)

    if port_out is None:
            port_out = ports
        
    for acq_channel, clock in enumerate(clocks):
        if acq_channel == 0:
            frequencies = frequencies_qubit1
        if acq_channel == 1:
            frequencies = frequencies_qubit2

        schedule.add_resource(ClockResource(name=clock, freq=frequencies.flat[0]))

    for i, idx in enumerate(frequencies_idx):
        prep = schedule.add(IdlePulse(duration=init_duration))

        for acq_channel, _ in enumerate(clocks):
            if acq_channel == 0:
                frequencies = frequencies_qubit1
            if acq_channel == 1:
                frequencies = frequencies_qubit2

            schedule.add(
                SetClockFrequency(clock=clocks[acq_channel], clock_freq_new=frequencies[idx]),
                ref_op=prep,
                ref_pt="end",
                rel_time=-100e-9
            )

            spec_pulse = schedule.add(
                SquarePulse(
                    duration=pulse_duration,
                    amp=pulse_amp,
                    port=port_out[acq_channel],
                    clock=clocks[acq_channel],
                ),
                ref_op=prep,
                ref_pt="end",
            )

            schedule.add(
                SSBIntegrationComplex(
                    duration=integration_time,
                    port=ports[acq_channel],
                    clock=clocks[acq_channel],
                    acq_index=i,
                    acq_channel=acq_channel,
                    bin_mode=BinMode.AVERAGE,
                ),
                ref_op=prep,
                ref_pt="end",
                rel_time=acquisition_delay,
            )

            if acq_channel == 0:
                add_twpa_marker(schedule, spec_pulse, ports[0][:-3] + "switch", integration_time)
    
                schedule.add(IdlePulse(duration=4e-9))
        
    return schedule

def two_tone_spec_sched_nco(
    spec_pulse_amp: float,
    spec_pulse_duration: float,
    spec_pulse_port: str,
    spec_pulse_clock: str,
    spec_pulse_frequencies: np.ndarray,
    ro_pulse_amp: float,
    ro_pulse_duration: float,
    ro_pulse_delay: float,
    ro_pulse_port: str,
    ro_pulse_clock: str,
    ro_pulse_frequency: float,
    ro_acquisition_delay: float,
    ro_integration_time: float,
    init_duration: float,
    repetitions: int = 1,
) -> Schedule:
    
    schedule = Schedule("Fast two-tone spectroscopy (NCO sweep)(TWPA)", repetitions)
    schedule.add_resources(
        [
            ClockResource(name=spec_pulse_clock, freq=spec_pulse_frequencies.flat[0]),
            ClockResource(name=ro_pulse_clock, freq=ro_pulse_frequency),
        ]
    )

    for i, spec_pulse_freq in enumerate(spec_pulse_frequencies):
        schedule.add(IdlePulse(duration=init_duration), label=f"buffer {i}")

        schedule.add(
            SetClockFrequency(clock=spec_pulse_clock, clock_freq_new=spec_pulse_freq),
            label=f"set_freq {i} ({spec_pulse_clock} {spec_pulse_freq:e} Hz)",
        )

        spec_pulse = schedule.add(
            SquarePulse(
                duration=spec_pulse_duration,
                amp=spec_pulse_amp,
                port=spec_pulse_port,
                clock=spec_pulse_clock,
            ),
            label=f"spec_pulse {i}",
        )
        
        ro_pulse = schedule.add(
            SquarePulse(
                duration=ro_pulse_duration,
                amp=ro_pulse_amp,
                port=ro_pulse_port,
                clock=ro_pulse_clock,
            ),
            ref_op=spec_pulse,
            ref_pt="end",
            rel_time=ro_pulse_delay,
            label=f"readout_pulse {i}",
        )

        schedule.add(
            SSBIntegrationComplex(
                duration=ro_integration_time,
                port=ro_pulse_port,
                clock=ro_pulse_clock,
                acq_index=i,
                acq_channel=0,
                bin_mode=BinMode.AVERAGE,
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=ro_acquisition_delay,
            label=f"acquisition {i}",
        )

        add_twpa_marker(
            schedule,
            ro_pulse,
            ro_pulse_port[:-3] + "switch",
            ro_integration_time,
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def multiplexed_two_tone_spec_sched_nco(
    qubits: list[any],
    pulse_amp: float,
    pulse_duration: float,
    frequencies_qubit1: np.ndarray,
    frequencies_qubit2: np.ndarray,
    frequencies_idx: np.ndarray,
    ro_pulse_delay: float,
    init_duration: float = 300e-6,
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubits[0].measure.pulse_duration()
    acquisition_delay = qubits[0].measure.acq_delay()
    qubits = [qubit.name for qubit in qubits]
    spec_ports = [f"{qubit}:mw" for qubit in qubits]
    spec_clocks = [f"{qubit}.01" for qubit in qubits]
    ro_ports = [f"{qubit}:res" for qubit in qubits]
    ro_clocks = [f"{qubit}.ro" for qubit in qubits]
    
    schedule = Schedule("Multiplexed qubit spectroscopy (TWPA)", repetitions)
        
    for acq_channel, spec_clock in enumerate(spec_clocks):
        if acq_channel == 0:
            frequencies = frequencies_qubit1
        if acq_channel == 1:
            frequencies = frequencies_qubit2
            
        schedule.add_resource(ClockResource(name=spec_clock, freq=frequencies.flat[0]))

    for i, idx in enumerate(frequencies_idx):
        prep = schedule.add(IdlePulse(duration=init_duration))

        for acq_channel, _ in enumerate(spec_clocks):
            if acq_channel == 0:
                frequencies = frequencies_qubit1
            if acq_channel == 1:
                frequencies = frequencies_qubit2

            schedule.add(
                SetClockFrequency(clock=spec_clocks[acq_channel], clock_freq_new=frequencies[idx]),
                ref_op=prep,
                ref_pt="end",
                rel_time=-100e-9
            )

            spec_pulse = schedule.add(
                SquarePulse(
                    duration=pulse_duration,
                    amp=pulse_amp,
                    port=spec_ports[acq_channel],
                    clock=spec_clocks[acq_channel],
                ),
                ref_op=prep,
                ref_pt="end",
            )

            ro_pulse = schedule.add(
                Measure(
                    qubits[acq_channel], 
                    acq_index=i, 
                    acq_channel=acq_channel,
                ),
                ref_op=spec_pulse,
                ref_pt="end",
                rel_time=ro_pulse_delay,
            )

            if acq_channel == 0:
                add_twpa_marker(
                    schedule,
                    ro_pulse,
                    qubits[acq_channel] + ":switch",
                    readout_duration,
                )
    
                schedule.add(IdlePulse(duration=4e-9))
        
    return schedule

def rabi_sched(
    pulse_amp: Union[np.ndarray, float],
    pulse_duration: Union[np.ndarray, float],
    frequency: float,
    qubit: any,
    port: str = None,
    clock: str = None,
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    # ensure pulse_amplitude and pulse_duration are iterable.
    amps = np.asarray(pulse_amp)
    amps = amps.reshape(amps.shape or (1,))
    durations = np.asarray(pulse_duration)
    durations = durations.reshape(durations.shape or (1,))

    # either the shapes of the amp and duration must match or one of
    # them must be a constant floating point value.
    if len(amps) == 1:
        amps = np.ones(np.shape(durations)) * amps
    elif len(durations) == 1:
        durations = np.ones(np.shape(amps)) * durations
    elif len(durations) != len(amps):
        raise ValueError(
            f"Shapes of pulse_amplitude ({pulse_amp.shape}) and "
            f"pulse_duration ({pulse_duration.shape}) are incompatible."
        )

    if port is None:
        port = f"{qubit}:mw"
    if clock is None:
        clock = f"{qubit}.01"

    schedule = Schedule("Rabi(TWPA)", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequency))

    for i, (amp, duration) in enumerate(zip(amps, durations)):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        DRAG = schedule.add(
            DRAGPulse(
                duration=duration,
                G_amp=amp,
                D_amp=0,
                port=port,
                clock=clock,
                phase=0,
                # sigma=duration/4,
            ),
            label=f"Rabi_pulse {i}",
        )

        # N.B. acq_channel is not specified
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def rabi_amplification(
    pulse_amp: Union[np.ndarray, float],
    pulse_duration: Union[np.ndarray, float],
    pi_number: np.ndarray,
    frequency: float,
    qubit: any,
    port: str = None,
    clock: str = None,
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    # ensure pulse_amplitude and pulse_duration are iterable.
    amps = np.asarray(pulse_amp)
    amps = amps.reshape(amps.shape or (1,))
    durations = np.asarray(pulse_duration)
    durations = durations.reshape(durations.shape or (1,))

    # either the shapes of the amp and duration must match or one of
    # them must be a constant floating point value.
    if len(amps) == 1:
        amps = np.ones(np.shape(durations)) * amps
    elif len(durations) == 1:
        durations = np.ones(np.shape(amps)) * durations
    elif len(durations) != len(amps):
        raise ValueError(
            f"Shapes of pulse_amplitude ({pulse_amp.shape}) and "
            f"pulse_duration ({pulse_duration.shape}) are incompatible."
        )

    if port is None:
        port = f"{qubit}:mw"
    if clock is None:
        clock = f"{qubit}.01"

    schedule = Schedule("Rabi Error Amplification(TWPA)", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequency))

    for i, (amp, duration) in enumerate(zip(amps, durations)):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        for n in range(pi_number):
            DRAG = schedule.add(
                DRAGPulse(
                    duration=duration,
                    G_amp=amp,
                    D_amp=0,
                    port=port,
                    clock=clock,
                    phase=0,
                    # sigma=duration/4,
                ),
                # label=f"Rabi_pulse {i*n}",
            )

        # N.B. acq_channel is not specified
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def ramsey_sched(
    times: Union[np.ndarray, float],
    qubit: any,
    artificial_detuning: float = 0,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("Ramsey(TWPA)", repetitions)

    if isinstance(times, float):
        times = [times]

    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        x90 = schedule.add(X90(qubit))

        # schedule.add(MarkerPulse(
        #         duration=tau,
        #         port="snail:switch",
        #     ), ref_op=x90, ref_pt="end", rel_time=-60e-9)

        # the phase of the second pi/2 phase progresses to propagate
        # recovery_phase = np.rad2deg(2 * np.pi * artificial_detuning * tau)
        # schedule.add(
        #     Rxy(theta=90, phi=recovery_phase, qubit=qubit), ref_op=x90, ref_pt="end", rel_time=tau
        # )

        # the phase of the second pi/2 phase progresses to propagate
        recovery_phase = np.rad2deg(2 * np.pi * artificial_detuning * tau)
        schedule.add(
            Rxy(theta=90, phi=recovery_phase, qubit=qubit), ref_pt="start", rel_time=tau
        )
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            acq_protocol=acq_protocol,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def add_stark_tone(
    schedule: Schedule,
    qubit: str,
    reset_op,
    ro_op,
    stark_amp: float,
    stark_freq: float,
    label: Optional[str] = None,
) -> None:
    """Add a constant Stark tone from ``STARK_LEAD`` before the end of ``reset_op``
    until the end of ``ro_op``.

    The tone is a sequencer offset on ``<qubit>:mw`` with clock ``<qubit>.stark``
    (needs the ``<qubit>:mw-<qubit>.stark`` entry in the hardware config), so it
    does not use waveform memory and anything played on ``<qubit>:mw-<qubit>.01``
    in between adds on top of it (keep the summed amplitude below 1). The reset
    duration must be >= ``STARK_LEAD``. Call once per shot after ``ro_op`` is added;
    the ``<qubit>.stark`` clock resource is added to ``schedule`` if missing.
    """
    clock = qubit + ".stark"
    if clock not in schedule.resources:
        schedule.add_resource(ClockResource(name=clock, freq=stark_freq))

    schedule.add(
        VoltageOffset(
            offset_path_I=np.real(stark_amp),
            offset_path_Q=np.imag(stark_amp),
            port=qubit + ":mw",
            clock=clock,
        ),
        ref_op=reset_op,
        ref_pt="end",
        rel_time=-STARK_LEAD,
        label=None if label is None else f"Stark on {label}",
    )
    # as in long_square_pulse: the offset is cleared 4 ns early and the last 4 ns are a
    # normal pulse, so the tone can also end at the very end of the schedule
    stark_off = schedule.add(
        VoltageOffset(offset_path_I=0.0, offset_path_Q=0.0, port=qubit + ":mw", clock=clock),
        ref_op=ro_op,
        ref_pt="end",
        rel_time=-4e-9,
        label=None if label is None else f"Stark off {label}",
    )
    schedule.add(
        SquarePulse(amp=stark_amp, duration=4e-9, port=qubit + ":mw", clock=clock),
        ref_op=stark_off,
        ref_pt="start",
        label=None if label is None else f"Stark tail {label}",
    )

def stark_ramsey_sched(
    times: Union[np.ndarray, float],
    qubit: any,
    stark_amp: float,
    stark_freq: float,
    artificial_detuning: float = 0,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
    repetitions: int = 1,
) -> Schedule:
    """Ramsey with a constant off-resonant Stark tone on for the whole sequence.

    The tone is played on ``<qubit>:mw`` with clock ``<qubit>.stark`` (needs the
    ``<qubit>:mw-<qubit>.stark`` entry in the hardware config). It turns on
    ``STARK_LEAD`` before the first X90, inside the reset wait, so the Stark shift
    is in steady state, stays on under both pi/2 pulses (outputs add, so keep
    ``stark_amp + rxy.amp180`` below 1) and turns off when the readout finishes.
    ``times`` are start-to-start delays as in ``ramsey_sched``.
    """

    readout_duration = qubit.measure.pulse_duration()
    if qubit.reset.duration() < STARK_LEAD:
        raise ValueError(
            f"reset.duration ({qubit.reset.duration()}) must be >= STARK_LEAD ({STARK_LEAD})"
        )
    qubit = qubit.name

    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("Stark Ramsey(TWPA)", repetitions)

    for i, tau in enumerate(times):
        reset = schedule.add(Reset(qubit), label=f"Reset {i}")
        x90 = schedule.add(X90(qubit))

        # the phase of the second pi/2 phase progresses to propagate
        recovery_phase = np.rad2deg(2 * np.pi * artificial_detuning * tau)
        schedule.add(
            Rxy(theta=90, phi=recovery_phase, qubit=qubit), ref_op=x90, ref_pt="start", rel_time=tau
        )
        ro_pulse = schedule.add(Measure(qubit, acq_index=i, acq_protocol=acq_protocol), label=f"Measurement {i}")

        add_stark_tone(
            schedule, qubit, reset, ro_pulse, stark_amp, stark_freq, label=f"{i}"
        )

        add_twpa_marker(
            schedule,
            ro_pulse,
            qubit + ":switch",
            readout_duration,
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def coupled_ramsey_sched(
    times: Union[np.ndarray, float],
    qubit: BasicTransmonElement,
    qubit_c: BasicTransmonElement,
    condition: bool = True,
    artificial_detuning: float = 0,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    qubit_c = qubit_c.name
    
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("Coupled Ramsey(TWPA)", repetitions)

    if isinstance(times, float):
        times = [times]

    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        if condition:
            schedule.add(X(qubit_c))
        pi_2 = schedule.add(X90(qubit))

        # the phase of the second pi/2 phase progresses to propagate
        recovery_phase = np.rad2deg(2 * np.pi * artificial_detuning * tau)
        schedule.add(
            Rxy(theta=90, phi=recovery_phase, qubit=qubit), ref_pt="start", rel_time=tau
        )

        # # test active AC-Stark shift on SNAIL to reduce ZZ
        # schedule.add(
        #     MarkerPulse(
        #         duration=tau + 38e-9,
        #         port="snail:switch",
        #     ),
        #     ref_op=pi_2,
        #     ref_pt="start",
        #     rel_time=-38e-9,
        # )

        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            acq_protocol=acq_protocol,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule

def coupled_phase_ramsey_sched(
    phases: Union[np.ndarray, float],
    qubit: BasicTransmonElement,
    qubit_c: BasicTransmonElement,
    condition: bool = True,
    tau: float = 2e-6,
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    qubit_c = qubit_c.name
    
    # ensure phases is an iterable when passing floats.
    phases = np.asarray(phases)
    phases = phases.reshape(phases.shape or (1,))

    schedule = Schedule("Coupled Ramsey(TWPA)", repetitions)

    if isinstance(phases, float):
        phases = [phases]

    for i, phase in enumerate(phases):
        reset = schedule.add(Reset(qubit), label=f"Reset {i}")
        if condition:
            schedule.add(X(qubit_c), ref_op=reset, ref_pt="end")
        pi_2 = schedule.add(X90(qubit), ref_op=reset, ref_pt="end")

        # the phase of the second pi/2 phase progresses to propagate
        schedule.add(
            Rxy(theta=90, phi=phase, qubit=qubit), ref_pt="end", rel_time=tau
        )

        # test active AC-Stark shift on SNAIL to reduce ZZ
        schedule.add(
            MarkerPulse(
                duration=tau + 138e-9,
                port="snail:switch",
            ),
            ref_op=pi_2,
            ref_pt="start",
            rel_time=-38e-9,
        )

        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule

def echo_sched(
    times: Union[np.ndarray, float],
    qubit: any,
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
) -> Schedule:
    
    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name

    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("Echo(TWPA)", repetitions)
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X90(qubit))
        schedule.add(X(qubit), ref_pt="end", rel_time=tau / 2)
        schedule.add(X90(qubit), ref_pt="end", rel_time=tau / 2)
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_protocol=acq_protocol,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    return schedule

def t1_sched(
    times: Union[np.ndarray, float],
    qubit: any,
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
) -> Schedule:
    
    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("T1(TWPA)", repetitions)
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X(qubit), label=f"pi {i}")

        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            acq_protocol=acq_protocol,
            ref_pt="end",
            rel_time=tau,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    return schedule

def t1_and_t2(
    times: Union[np.ndarray, float],
    qubit: any,
    case: Union[np.ndarray, int],
    artificial_detuning: float = 0,
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
) -> Schedule:
    
    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("T1 and T2(TWPA)", repetitions)
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")

        if case == 1:
            schedule.add(X(qubit), label=f"pi {i}")
            ro_pulse = schedule.add(
                Measure(qubit, acq_index=i, acq_protocol=acq_protocol),
                ref_pt="end",
                rel_time=tau,
                label=f"Measurement {i}",
            )

        elif case == 2:
            schedule.add(X90(qubit))
            schedule.add(X(qubit), ref_pt="end", rel_time=tau / 2)
            schedule.add(X90(qubit), ref_pt="end", rel_time=tau / 2)
            ro_pulse = schedule.add(Measure(qubit, acq_index=i, acq_protocol=acq_protocol), label=f"Measurement {i}")

        elif case == 3:
            schedule.add(X90(qubit), label=f"pi/2 {i}")
            # the phase of the second pi/2 pulse progresses to propagate
            recovery_phase = np.rad2deg(2 * np.pi * artificial_detuning * tau)
            schedule.add(
                Rxy(theta=90, phi=recovery_phase, qubit=qubit),
                ref_pt="start",
                rel_time=tau,
                label=f"recovery pi/2 {i}",
            )
            ro_pulse = schedule.add(Measure(qubit, acq_index=i, acq_protocol=acq_protocol), label=f"Measurement {i}")

        add_twpa_marker(
            schedule,
            ro_pulse,
            qubit + ":switch",
            readout_duration,
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    return schedule

def multi_qubit_t1_and_t2(
    times: Union[np.ndarray, float],
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    case: Union[np.ndarray, int],
    artificial_detuning: float = 0,
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
    
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("Multi Qubit T1 and T2(TWPA)", repetitions)
    for i, tau in enumerate(times):
        for qubit in qubit_names:
            schedule.add(Reset(qubit))

            if case == 1:
                schedule.add(X(qubit))
                ro_pulse = schedule.add(
                    Measure(qubit, acq_index=i, acq_protocol=acq_protocol),
                    ref_pt="end",
                    rel_time=tau,
                )

            elif case == 2:
                schedule.add(X90(qubit))
                schedule.add(X(qubit), ref_pt="end", rel_time=tau / 2)
                schedule.add(X90(qubit), ref_pt="end", rel_time=tau / 2)
                ro_pulse = schedule.add(Measure(qubit, acq_index=i, acq_protocol=acq_protocol))

            elif case == 3:
                schedule.add(X90(qubit))
                # the phase of the second pi/2 pulse progresses to propagate
                recovery_phase = np.rad2deg(2 * np.pi * artificial_detuning * tau)
                schedule.add(
                    Rxy(theta=90, phi=recovery_phase, qubit=qubit),
                    ref_pt="start",
                    rel_time=tau,
                )
                ro_pulse = schedule.add(Measure(qubit, acq_index=i, acq_protocol=acq_protocol))

            add_twpa_marker(schedule, ro_pulse, qubit + ":switch", readout_duration)

            schedule.add(IdlePulse(duration=4e-9))
    return schedule
    
def multiplexed_readout_calibration_sched(
    qubits: List[any],
    prepared_states: List[int],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
) -> Schedule:

    readout_duration = max_readout_duration(qubits)

    qubit_names = [q.name for q in qubits]
    schedule = Schedule(f"Multiplexed readout calibration (TWPA)", repetitions)

    for i, prep_state in enumerate(prepared_states):
        reset = schedule.add(Reset(*qubit_names))
        if prep_state == 0:
            pass
        elif prep_state == 1:
            for qubit_name in qubit_names:
                schedule.add(X(qubit_name), ref_op=reset, rel_time=0)
        else:
            raise ValueError(f"Prepared state ({prep_state}) must be either 0 or 1.")
            
        measure_with_twpa(
            schedule,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=i,
            bin_mode=BinMode.APPEND,
            acq_protocol=acq_protocol,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    return schedule
    
def readout_calibration_sched(
    qubit: any,
    prepared_states: List[int],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    schedule = Schedule(f"Readout calibration {qubit}, {prepared_states}(TWPA)", repetitions)

    for i, prep_state in enumerate(prepared_states):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        if prep_state == 0:
            pass
        elif prep_state == 1:
            DRAG = schedule.add(Rxy(qubit=qubit, theta=180, phi=0))

        elif prep_state == 2:
            schedule.add(X(qubit))
            DRAG = schedule.add(
                DRAGPulse(
                    duration=E2F_DURATION,
                    G_amp=E2F_G_AMP,
                    D_amp=0,
                    port=f"{qubit}:fs",
                    clock=f"{qubit}.12",
                    phase=0,
                    # sigma=duration/4,
                ),
                label=f"f_state_Pi {i}",
            )
            
        else:
            raise ValueError(f"Prepared state ({prep_state}) must be either 0, 1 or 2.")
        ro_pulse = schedule.add(
            Measure(
                qubit, acq_index=i, bin_mode=BinMode.APPEND, acq_protocol=acq_protocol
            ),
            label=f"Measurement {i}",
        )

        schedule.add(ResetClockPhase(clock=f"{qubit}.ro"), ref_op=ro_pulse, ref_pt="start", rel_time=-4e-9)

        add_twpa_marker(
            schedule,
            ro_pulse,
            qubit + ":switch",
            readout_duration,
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    return schedule

def readout_freq_optimization_TWPA(
    qubit: BasicTransmonElement,
    frequencies: Union[np.ndarray, float], 
    instrument_coordinator, 
    quantum_device,
    repetitions: int = 1,
):
    marker_duration = qubit.measure.pulse_duration()+TWPA_RINGUP+TWPA_TAIL
    qubit = qubit.name
    
    frequency = np.asarray(frequencies)
    frequency = frequency.reshape(frequencies.shape or (1,))

    schedule = Schedule("Readout frequency optimization (TWPA)", repetitions)

    for i, freq in enumerate(frequencies):
        schedule.add(SetClockFrequency(clock=f"{qubit}.ro", clock_freq_new=freq))
        schedule.add(Reset(qubit))
        ro_pulse = schedule.add(Measure(qubit, acq_index=i*2, acq_protocol="SSBIntegrationComplex", bin_mode=BinMode.APPEND))

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
            label=f"TWPA_mark {i*2})",
        )

        schedule.add(Reset(qubit))
        schedule.add(X(qubit=qubit))
        ro_pulse = schedule.add(Measure(qubit, acq_index=i*2+1, acq_protocol="SSBIntegrationComplex", bin_mode=BinMode.APPEND))

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
            label=f"TWPA_mark {i*2+1})",
        )

    schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())

    instrument_coordinator.prepare(compiled_schedule)
    instrument_coordinator.start()
    instrument_coordinator.wait_done(timeout_sec=90)

    return instrument_coordinator.retrieve_acquisition()

def readout_amp_optimization_TWPA(
    qubit: BasicTransmonElement,
    amps: Union[np.ndarray, float], 
    instrument_coordinator, 
    quantum_device,
    repetitions: int = 1,
):
    marker_duration = qubit.measure.pulse_duration()+TWPA_RINGUP+TWPA_TAIL
    qubit = qubit.name

    schedule = Schedule("Readout amplitude optimization (TWPA)", repetitions)

    for i, amp in enumerate(amps):
        schedule.add(Reset(qubit))
        ro_pulse = schedule.add(Measure(qubit, pulse_amp=amp, acq_index=i*2, acq_protocol="SSBIntegrationComplex", bin_mode=BinMode.APPEND))

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
            label=f"TWPA_mark {i*2})",
        )

        schedule.add(Reset(qubit))
        schedule.add(X(qubit=qubit))
        ro_pulse = schedule.add(Measure(qubit, pulse_amp=amp, acq_index=i*2+1, acq_protocol="SSBIntegrationComplex", bin_mode=BinMode.APPEND))

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
            label=f"TWPA_mark {i*2+1})",
        )

    schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())

    instrument_coordinator.prepare(compiled_schedule)
    instrument_coordinator.start()
    instrument_coordinator.wait_done(timeout_sec=90)

    return instrument_coordinator.retrieve_acquisition()

def readout_len_optimization_TWPA(
    qubit: BasicTransmonElement,
    lens: Union[np.ndarray, float],
    instrument_coordinator,
    quantum_device,
    repetitions: int = 1,
):
    qubit = qubit.name

    schedule = Schedule("Readout length optimization (TWPA)", repetitions)

    for i, length in enumerate(lens):
        marker_duration = length + TWPA_RINGUP + TWPA_TAIL

        schedule.add(Reset(qubit))
        ro_pulse = schedule.add(Measure(qubit, pulse_duration=length, integration_time=length, acq_index=i*2, acq_protocol="SSBIntegrationComplex", bin_mode=BinMode.APPEND))

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
            label=f"TWPA_mark {i*2})",
        )

        schedule.add(Reset(qubit))
        schedule.add(X(qubit=qubit))
        ro_pulse = schedule.add(Measure(qubit, pulse_duration=length, integration_time=length, acq_index=i*2+1, acq_protocol="SSBIntegrationComplex", bin_mode=BinMode.APPEND))

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
            label=f"TWPA_mark {i*2+1})",
        )

    schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())

    instrument_coordinator.prepare(compiled_schedule)
    instrument_coordinator.start()
    instrument_coordinator.wait_done(timeout_sec=90)

    return instrument_coordinator.retrieve_acquisition()

def readout_weight_optimization_TWPA(
    qubit: BasicTransmonElement,
    instrument_coordinator, 
    quantum_device,
    repetitions: int = 1,
):
    marker_duration = qubit.measure.pulse_duration()+TWPA_RINGUP+TWPA_TAIL
    qubit = qubit.name


    for i in range(2):
        schedule = Schedule("Readout weight optimization (TWPA)", repetitions)

        schedule.add(Reset(qubit))
        if i:
            schedule.add(X(qubit=qubit))
        ro_pulse = schedule.add(Measure(qubit, acq_channel=0, acq_protocol="Trace", bin_mode=BinMode.AVERAGE))

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end")
        compiler = SerialCompiler(name="compiler")
        compiled_schedule = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())

        instrument_coordinator.prepare(compiled_schedule)
        instrument_coordinator.start()
        instrument_coordinator.wait_done(timeout_sec=30)

        if i == 0:
            g_trace = instrument_coordinator.retrieve_acquisition()
        elif i == 1:
            e_trace = instrument_coordinator.retrieve_acquisition()

    return g_trace, e_trace

def allxy_sched(
    qubit: any,
    element_select_idx: Union[np.ndarray, int] = np.arange(21),
    repetitions: int = 1,
) -> Schedule:
    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    element_idxs = np.asarray(element_select_idx)
    element_idxs = element_idxs.reshape(element_idxs.shape or (1,))

    # all combinations of Idle, X90, Y90, X180 and Y180 gates that are part of
    # the AllXY experiment
    allxy_combinations = [
        [(0, 0), (0, 0)],
        [(180, 0), (180, 0)],
        [(180, 90), (180, 90)],
        [(180, 0), (180, 90)],
        [(180, 90), (180, 0)],
        [(90, 0), (0, 0)],
        [(90, 90), (0, 0)],
        [(90, 0), (90, 90)],
        [(90, 90), (90, 0)],
        [(90, 0), (180, 90)],
        [(90, 90), (180, 0)],
        [(180, 0), (90, 90)],
        [(180, 90), (90, 0)],
        [(90, 0), (180, 0)],
        [(180, 0), (90, 0)],
        [(90, 90), (180, 90)],
        [(180, 90), (90, 90)],
        [(180, 0), (0, 0)],
        [(180, 90), (0, 0)],
        [(90, 0), (90, 0)],
        [(90, 90), (90, 90)],
    ]
    schedule = Schedule("AllXY(TWPA)", repetitions)

    for i, elt_idx in enumerate(element_idxs):
        # check index valid
        if elt_idx > len(allxy_combinations) or elt_idx < 0:
            raise ValueError(
                f"Invalid index selected: {elt_idx}. "
                "Index must be in range 0 to 21 inclusive."
            )

        ((th0, phi0), (th1, phi1)) = allxy_combinations[elt_idx]

        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(Rxy(qubit=qubit, theta=th0, phi=phi0))
        schedule.add(Rxy(qubit=qubit, theta=th1, phi=phi1))
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    return schedule

def DRAG_calibration_sched(
    qubit: BasicTransmonElement,
    motzoi_list: Union[np.ndarray, float], 
    instrument_coordinator, 
    quantum_device,
    repetitions: int = 1,
):
    marker_duration = qubit.measure.pulse_duration()+TWPA_RINGUP+TWPA_TAIL
    qubit = qubit.name
    
    motzoi = np.asarray(motzoi_list)
    motzoi = motzoi.reshape(motzoi.shape or (1,))

    allxy_combinations = [
        [(180, 0), (90, 90)],
        [(180, 90), (90, 0)],
    ]
    schedule = Schedule("DRAG calibration (TWPA)", repetitions)

    for i, m in enumerate(motzoi):
        ((th0, phi0), (th1, phi1)) = allxy_combinations[0]

        schedule.add(Reset(qubit))
        schedule.add(Rxy(qubit=qubit, theta=th0, phi=phi0, motzoi=m))
        schedule.add(Rxy(qubit=qubit, theta=th1, phi=phi1, motzoi=m))
        ro_pulse = schedule.add(Measure(qubit, acq_index=i*2))

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
            label=f"TWPA_mark {i*2})",
        )

        ((th0, phi0), (th1, phi1)) = allxy_combinations[1]

        schedule.add(Reset(qubit))
        schedule.add(Rxy(qubit=qubit, theta=th0, phi=phi0, motzoi=m))
        schedule.add(Rxy(qubit=qubit, theta=th1, phi=phi1, motzoi=m))
        ro_pulse = schedule.add(Measure(qubit, acq_index=i*2+1))

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=marker_duration,
                port=qubit + ":switch",
            ),
            ref_op=ro_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
            label=f"TWPA_mark {i*2+1})",
        )

    schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(schedule=schedule, config=quantum_device.generate_compilation_config())

    instrument_coordinator.prepare(compiled_schedule)
    instrument_coordinator.start()
    instrument_coordinator.wait_done(timeout_sec=60)

    return instrument_coordinator.retrieve_acquisition()
    

def dressed_e_cavity(
    pulse_amp: float,
    pulse_duration: float,
    frequencies: np.ndarray,
    acquisition_delay: float,
    integration_time: float,
    qubit: str,
    port: str,
    clock: str,
    init_duration: float = 10e-6,
    repetitions: int = 1,
    port_out: Optional[str] = None,
) -> Schedule:
    
    schedule = Schedule("Dressed e Cavity(TWPA)", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequencies.flat[0]))

    if port_out is None:
        port_out = port

    for i, freq in enumerate(frequencies):
        schedule.add(IdlePulse(duration=init_duration), label=f"buffer {i}")

        schedule.add(Reset(qubit))
        schedule.add(X(qubit))

        schedule.add(
            SetClockFrequency(clock=clock, clock_freq_new=freq),
            label=f"set_freq {i} ({clock} {freq:e} Hz)",
        )

        spec_pulse = schedule.add(
            SquarePulse(
                duration=pulse_duration,
                amp=pulse_amp,
                port=port_out,
                clock=clock,
            ),
            label=f"spec_pulse {i})",
        )

        ro_pulse = schedule.add(
            SSBIntegrationComplex(
                duration=integration_time,
                port=port,
                clock=clock,
                acq_index=i,
                acq_channel=0,
                bin_mode=BinMode.AVERAGE,
            ),
            ref_op=spec_pulse,
            ref_pt="start",
            rel_time=acquisition_delay,
            label=f"acquisition {i})",
        )

        marker_pulse = schedule.add(
            MarkerPulse(
                duration=integration_time+acquisition_delay, 
                port=port[:-3] + "switch",
            ),
            ref_op=spec_pulse,
            ref_pt="start",
            rel_time=TWPA_DELAY-TWPA_RINGUP,
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def f_state_spec_sched_nco(
    spec_pulse_amp: float,
    spec_pulse_duration: float,
    spec_pulse_port: str,
    spec_pulse_clock: str,
    spec_pulse_frequencies: np.ndarray,
    qubit: any,
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    schedule = Schedule("f state spectroscopy (NCO sweep)(TWPA)", repetitions)
    schedule.add_resources([
        ClockResource(name=spec_pulse_clock, freq=spec_pulse_frequencies.flat[0]),
    ])

    for i, spec_pulse_freq in enumerate(spec_pulse_frequencies):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X(qubit), label=f"pi {i}")

        schedule.add(
            SetClockFrequency(clock=spec_pulse_clock, clock_freq_new=spec_pulse_freq),
            label=f"set_freq {i} ({spec_pulse_clock} {spec_pulse_freq:e} Hz)",
        )
        
        spec_pulse = schedule.add(
            SquarePulse(
                duration=spec_pulse_duration,
                amp=spec_pulse_amp,
                port=spec_pulse_port,
                clock=spec_pulse_clock,
            ),
            label=f"spec_pulse {i}",
        )
        
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def f_state_rabi_sched(
    pulse_amp: Union[np.ndarray, float],
    pulse_duration: Union[np.ndarray, float],
    frequency: float,
    qubit: any,
    port: str = None,
    clock: str = None,
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    # ensure pulse_amplitude and pulse_duration are iterable.
    amps = np.asarray(pulse_amp)
    amps = amps.reshape(amps.shape or (1,))
    durations = np.asarray(pulse_duration)
    durations = durations.reshape(durations.shape or (1,))

    # either the shapes of the amp and duration must match or one of
    # them must be a constant floating point value.
    if len(amps) == 1:
        amps = np.ones(np.shape(durations)) * amps
    elif len(durations) == 1:
        durations = np.ones(np.shape(amps)) * durations
    elif len(durations) != len(amps):
        raise ValueError(
            f"Shapes of pulse_amplitude ({pulse_amp.shape}) and "
            f"pulse_duration ({pulse_duration.shape}) are incompatible."
        )

    if port is None:
        port = f"{qubit}:fs"
    if clock is None:
        clock = f"{qubit}.12"

    schedule = Schedule("f state Rabi(TWPA)", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequency))

    for i, (amp, duration) in enumerate(zip(amps, durations)):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        schedule.add(X(qubit), label=f"Pi {i}")
        
        DRAG = schedule.add(
            DRAGPulse(
                duration=duration,
                G_amp=amp,
                D_amp=0,
                port=port,
                clock=clock,
                phase=0,
                # sigma=duration/4,
            ),
            label=f"Rabi_pulse {i}",
        )
        # square_pulse = schedule.add(
        #     SquarePulse(
        #         duration=duration,
        #         amp=amp,
        #         port=port,
        #         clock=clock,
        #     ),
        #     label=f"Spuare_pulse {i}",
        # )
        
        # N.B. acq_channel is not specified
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

# def f_state_ramsey_sched_TWPA(
#     times: Union[np.ndarray, float],
#     qubit: any,
#     artificial_detuning: float = 0,
#     repetitions: int = 1,
# ) -> Schedule:

#     marker_duration = qubit.measure.pulse_duration()+TWPA_RINGUP+TWPA_TAIL
#     qubit = qubit.name
    
#     # ensure times is an iterable when passing floats.
#     times = np.asarray(times)
#     times = times.reshape(times.shape or (1,))

#     schedule = Schedule("Ramsey(TWPA)", repetitions)

#     if isinstance(times, float):
#         times = [times]

#     for i, tau in enumerate(times):
#         schedule.add(Reset(qubit), label=f"Reset {i}")
#         schedule.add(X90(qubit))

#         # the phase of the second pi/2 phase progresses to propagate
#         recovery_phase = np.rad2deg(2 * np.pi * artificial_detuning * tau)
#         schedule.add(
#             Rxy(theta=90, phi=recovery_phase, qubit=qubit), ref_pt="start", rel_time=tau
#         )
#         ro_pulse = schedule.add(Measure(qubit, acq_index=i), label=f"Measurement {i}")

#         marker_pulse = schedule.add(
#             MarkerPulse(
#                 duration=marker_duration,
#                 port=qubit + ":switch",
#             ),
#             ref_op=ro_pulse,
#             ref_pt="start",
#             rel_time=TWPA_DELAY-TWPA_RINGUP,
#             label=f"TWPA_mark {i})",
#         )

#         schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
#     return schedule

def f_state_cavity(
    pulse_amp: float,
    pulse_duration: float,
    frequencies: np.ndarray,
    acquisition_delay: float,
    integration_time: float,
    qubit: str,
    port: str,
    clock: str,
    init_duration: float = 10e-6,
    repetitions: int = 1,
    port_out: Optional[str] = None,
) -> Schedule:
    
    schedule = Schedule("Dressed f Cavity(TWPA)", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequencies.flat[0]))

    if port_out is None:
        port_out = port

    for i, freq in enumerate(frequencies):
        schedule.add(IdlePulse(duration=init_duration), label=f"buffer {i}")

        schedule.add(Reset(qubit))
        schedule.add(X(qubit))
        DRAG = schedule.add(
            DRAGPulse(
                duration=E2F_DURATION,
                G_amp=E2F_G_AMP,
                D_amp=0,
                port=f"{qubit}:fs",
                clock=f"{qubit}.12",
                phase=0,
                # sigma=duration/4,
            ),
            label=f"Rabi_pulse {i}",
        )

        schedule.add(
            SetClockFrequency(clock=clock, clock_freq_new=freq),
            label=f"set_freq {i} ({clock} {freq:e} Hz)",
        )

        spec_pulse = schedule.add(
            SquarePulse(
                duration=pulse_duration,
                amp=pulse_amp,
                port=port_out,
                clock=clock,
            ),
            label=f"spec_pulse {i})",
        )

        ro_pulse = schedule.add(
            SSBIntegrationComplex(
                duration=integration_time,
                port=port,
                clock=clock,
                acq_index=i,
                acq_channel=0,
                bin_mode=BinMode.AVERAGE,
            ),
            ref_op=spec_pulse,
            ref_pt="start",
            rel_time=acquisition_delay,
            label=f"acquisition {i})",
        )

        add_twpa_marker(
            schedule,
            spec_pulse,
            port[:-3] + "switch",
            integration_time,
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def f_state_t1(
    times: Union[np.ndarray, float],
    qubit: any,
    case: Union[np.ndarray, int],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
) -> Schedule:
    
    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("f-state T1", repetitions)
    for i, tau in enumerate(times):
        schedule.add(Reset(qubit), label=f"Reset {i}")

        if case == 1:
            schedule.add(X(qubit), label=f"pi {i}")

        elif case == 2:
            schedule.add(X(qubit), label=f"pi {i}")
            DRAG = schedule.add(
                DRAGPulse(
                    duration=E2F_DURATION,
                    G_amp=E2F_G_AMP,
                    D_amp=0,
                    port=f"{qubit}:fs",
                    clock=f"{qubit}.12",
                    phase=0,
                    # sigma=duration/4,
                ),
            )

        schedule.add(IdlePulse(duration=tau), label=f"wait {i}")

        if case == 1:
            schedule.add(X(qubit))
        if case == 2:
            DRAG = schedule.add(
                DRAGPulse(
                    duration=E2F_DURATION,
                    G_amp=E2F_G_AMP,
                    D_amp=0,
                    port=f"{qubit}:fs",
                    clock=f"{qubit}.12",
                    phase=0,
                    # sigma=duration/4,
                ),
            )
            schedule.add(X(qubit))

        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            acq_protocol=acq_protocol,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
    return schedule

def multiplex_IQ(
    qubit: any,
    prepared_states: List[int],
    multiplexing_freq: List[float],
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    acquisition_delay = qubit.measure.acq_delay()
    pulse_duration = qubit.measure.pulse_duration()
    pulse_amp = qubit.measure.pulse_amp()
    qubit = qubit.name
    
    schedule = Schedule(f"Meltipexed Readout calibration {qubit}, {prepared_states}(TWPA)", repetitions)
    
    for acq_channel, dres_state in enumerate(["dres_g", "dres_e", "dres_f"]):
        schedule.add_resource(ClockResource(name=qubit+"."+dres_state, freq=multiplexing_freq[acq_channel]))
    
    for i, prep_state in enumerate(prepared_states):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        if prep_state == 0:
            pass
        elif prep_state == 1:
            DRAG = schedule.add(Rxy(qubit=qubit, theta=180, phi=0))

        elif prep_state == 2:
            schedule.add(X(qubit))
            DRAG = schedule.add(
                DRAGPulse(
                    duration=E2F_DURATION,
                    G_amp=E2F_G_AMP,
                    D_amp=0,
                    port=f"{qubit}:fs",
                    clock=f"{qubit}.12",
                    phase=0,
                ),
                label=f"f_state_Pi {i}",
            )
            
        else:
            raise ValueError(f"Prepared state ({prep_state}) must be either 0, 1 or 2.")
        
        prep = schedule.add(IdlePulse(duration=4e-9), label=f"prep {i}")

        for acq_channel, dres_state in enumerate(["dres_g", "dres_e", "dres_f"]):            
            schedule.add(
                SquarePulse(
                    duration=pulse_duration,
                    amp=pulse_amp,
                    port=f"{qubit}:res",
                    clock=qubit+"."+dres_state,
                ),
                ref_op=prep,
                ref_pt="end",
            )
            
            schedule.add(
                SSBIntegrationComplex(
                    duration=pulse_duration,
                    port=f"{qubit}:res",
                    clock=qubit+"."+dres_state,
                    acq_index=i,
                    acq_channel=acq_channel,
                    bin_mode=BinMode.APPEND,
                ),
                ref_op=prep,
                ref_pt="end",
                rel_time=acquisition_delay,
            )

        add_twpa_marker(
            schedule,
            prep,
            qubit + ":switch",
            readout_duration,
            ref_pt="end",
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule

def RO_raw_trace(
    qubit: any,
    trace_time: float,
    prepared_states: List[int],
    multiplexing_freq: List[float],
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    acquisition_delay = qubit.measure.acq_delay()
    pulse_duration = qubit.measure.pulse_duration()
    pulse_amp = qubit.measure.pulse_amp()
    qubit = qubit.name
    
    schedule = Schedule(f"Meltipexed RO Trace {qubit}, {prepared_states}(TWPA)", repetitions)
    
    for acq_channel, dres_state in enumerate(["dres_g", "dres_e", "dres_f"]):
        schedule.add_resource(ClockResource(name=qubit+"."+dres_state, freq=multiplexing_freq[acq_channel]))
    
    for i, prep_state in enumerate(prepared_states):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        if prep_state == 0:
            pass
        elif prep_state == 1:
            DRAG = schedule.add(Rxy(qubit=qubit, theta=180, phi=0))

        elif prep_state == 2:
            schedule.add(X(qubit))
            DRAG = schedule.add(
                DRAGPulse(
                    duration=E2F_DURATION,
                    G_amp=E2F_G_AMP,
                    D_amp=0,
                    port=f"{qubit}:fs",
                    clock=f"{qubit}.12",
                    phase=0,
                ),
                label=f"f_state_Pi {i}",
            )
            
        else:
            raise ValueError(f"Prepared state ({prep_state}) must be either 0, 1 or 2.")

        prep = schedule.add(IdlePulse(duration=4e-9), label=f"prep {i}")

        for acq_channel, dres_state in enumerate(["dres_g", "dres_e", "dres_f"]):            
            schedule.add(
                SquarePulse(
                    duration=pulse_duration,
                    amp=pulse_amp,
                    port=f"{qubit}:res",
                    clock=qubit+"."+dres_state,
                ),
                ref_op=prep,
                ref_pt="end",
            )
            
        schedule.add(
            Trace(
                duration=pulse_duration,
                port=f"{qubit}:res",
                clock=qubit+"."+dres_state,
                acq_index=i,
                acq_channel=0,
                bin_mode=BinMode.AVERAGE,
            ),
            ref_op=prep,
            ref_pt="end",
            rel_time=acquisition_delay,
        )

        add_twpa_marker(
            schedule,
            prep,
            qubit + ":switch",
            readout_duration,
            ref_pt="end",
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule

def rabi_population(
    qubit: any,
    case: int,
    angles: Union[np.ndarray, float],
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition"
    ] = "SSBIntegrationComplex",
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name
    
    schedule = Schedule(f"Rabi Population Measurement(TWPA)", repetitions)

    for i, angle in enumerate(angles):
        schedule.add(Reset(qubit), label=f"Reset {i}")

        if case == 1:
            DRAG = schedule.add(
                DRAGPulse(
                    duration=E2F_DURATION,
                    G_amp=E2F_G_AMP * angle/180,
                    D_amp=0,
                    port=f"{qubit}:fs",
                    clock=f"{qubit}.12",
                    phase=0,
                ),
                label=f"Theta Pulse {i}",
            )
            # square_pulse = schedule.add(
            #     SquarePulse(
            #         duration=E2F_DURATION,
            #         amp=E2F_G_AMP * angle/180,
            #         port=f"{qubit}:fs",
            #         clock=f"{qubit}.12",
            #     ),
            #     label=f"Spuare_pulse {i}",
            # )
            schedule.add(X(qubit))

        elif case == 2:
            schedule.add(X(qubit))
            DRAG = schedule.add(
                DRAGPulse(
                    duration=E2F_DURATION,
                    G_amp=E2F_G_AMP * angle/180,
                    D_amp=0,
                    port=f"{qubit}:fs",
                    clock=f"{qubit}.12",
                    phase=0,
                ),
                label=f"Theta Pulse {i}",
            )
            # square_pulse = schedule.add(
            #     SquarePulse(
            #         duration=E2F_DURATION,
            #         amp=E2F_G_AMP * angle/180,
            #         port=f"{qubit}:fs",
            #         clock=f"{qubit}.12",
            #     ),
            #     label=f"Spuare_pulse {i}",
            # )
            schedule.add(X(qubit))
            
        else:
            raise ValueError(f"Sequence case ({case}) must be either 1 or 2.")
            
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule

def SNAIL_swap_sched(
    pulse_frequency: Union[np.ndarray, float],
    pulse_amp: Union[np.ndarray, float],
    pulse_duration: Union[np.ndarray, float],
    snail_drive: any,
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    qubit_e: BasicTransmonElement | Iterable[BasicTransmonElement],
    swap_type: Literal['iSWAP', 'bSWAP'],
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
    repetitions: int = 1,
) -> Schedule:

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    readout_duration = max_readout_duration(qubits)

    qubit_e_names = [qubit.name for qubit in qubit_e]

    snail_drive.frequency(pulse_frequency)
    
    # ensure pulse_amplitude and pulse_duration are iterable.
    amps = np.asarray(pulse_amp)
    amps = amps.reshape(amps.shape or (1,))
    durations = np.asarray(pulse_duration)
    durations = durations.reshape(durations.shape or (1,))

    # either the shapes of the amp and duration must match or one of
    # them must be a constant floating point value.
    if len(amps) == 1:
        amps = np.ones(np.shape(durations)) * amps
    elif len(durations) == 1:
        durations = np.ones(np.shape(amps)) * durations
    elif len(durations) != len(amps):
        raise ValueError(
            f"Shapes of pulse_amplitude ({pulse_amp.shape}) and "
            f"pulse_duration ({pulse_duration.shape}) are incompatible."
        )

    schedule = Schedule("SNAIL swap", repetitions)

    for i, (amp, duration) in enumerate(zip(amps, durations)):
        snail_drive.power(amp)
        reset = schedule.add(Reset(*qubit_names), label=f"Reset {i}")
        if swap_type == 'iSWAP':
            # schedule.add(X(qubit_e_names[0]), label=f"pi {i}", ref_op=reset, ref_pt="end")
            # schedule.add(X(qubit_e_names[1]), ref_op=reset, ref_pt="end")
            for qubit_e_name in qubit_e_names:
                schedule.add(X(qubit_e_name), ref_op=reset, ref_pt="end")
        
        schedule.add(MarkerPulse(
                duration=duration,
                port="snail:switch",
            ),)
        
        schedule.add(IdlePulse(duration=100e-9))
        
        measure_with_twpa(
            schedule,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=i,
            acq_protocol=acq_protocol,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    # # Calibration points measured by preparing ground and excited states.
    # schedule.add(Reset(qubit), label="Reset Cal 0")
    # ro_pulse = schedule.add(Measure(qubit, acq_index=i + 1), label="Calibration 0")
    # marker_pulse = schedule.add(
    #     MarkerPulse(
    #         duration=8.4e-06,
    #         port=qubit + ":switch",
    #     ),
    #     ref_op=ro_pulse,
    #     ref_pt="start",
    #     rel_time=-9.96e-07,
    # )
    
    # reset_cal_1 = schedule.add(Reset(qubit), label="Reset Cal 1")
    # schedule.add(X(qubit), ref_op=reset_cal_1, rel_time=0)
    # ro_pulse = schedule.add(Measure(qubit, acq_index=i + 2), label="Calibration 1")
    # marker_pulse = schedule.add(
    #     MarkerPulse(
    #         duration=8.4e-06,
    #         port=qubit + ":switch",
    #     ),
    #     ref_op=ro_pulse,
    #     ref_pt="start",
    #     rel_time=-9.96e-07,
    # )
    # schedule.add(IdlePulse(duration=4e-9))

    return schedule

def SNAIL_spec_sched(
    spec_pulse_amp: float,
    spec_pulse_duration: float,
    spec_pulse_frequencies: np.ndarray,
    snail_drive: any,
    qubit: any,
    init_duration: float,
    repetitions: int = 1,
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name

    schedule = Schedule("signal generator spectroscopy (TWPA)", repetitions)

    for i, spec_pulse_freq in enumerate(spec_pulse_frequencies):
        snail_drive.frequency(spec_pulse_freq)
        snail_drive.power(spec_pulse_amp)
        
        schedule.add(IdlePulse(duration=init_duration), label=f"buffer {i}")

        schedule.add(MarkerPulse(
                duration=spec_pulse_duration,
                port="snail:switch",
            ),)
        
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule
    
# def SNAIL_spec_sched_VNA(
#     spec_pulse_amp: float,
#     spec_pulse_duration: float,
#     spec_pulse_frequencies: np.ndarray,
#     snail_drive: any,
#     ro_vna: any,
#     qubit: any,
#     init_duration: float,
#     repetitions: int = 1,
# ) -> Schedule:
    
#     marker_duration = qubit.measure.pulse_duration()+TWPA_RINGUP+TWPA_TAIL
#     qubit = qubit.name
    
#     snail_drive.pulsemod_state('ON')
#     snail_drive.pulsemod_source('EXT')
#     snail_drive.pulsemod_trig_mode('EXT')
#     snail_drive.status('ON')

#     schedule = Schedule("signal generator spectroscopy VNA", repetitions)

#     for i, spec_pulse_freq in enumerate([spec_pulse_frequencies]):
#         snail_drive.frequency(spec_pulse_freq)
#         snail_drive.power(spec_pulse_amp)
        
#         schedule.add(IdlePulse(duration=init_duration), label=f"buffer {i}")

#         schedule.add(MarkerPulse(
#                 duration=spec_pulse_duration,
#                 port="snail:switch",
#             ),)

#         ro_vna.traces.tr1.run_sweep()
        
#         ro_pulse = schedule.add(Measure(qubit, acq_index=i), label=f"Measurement {i}")

#         marker_pulse = schedule.add(
#             MarkerPulse(
#                 duration=marker_duration,
#                 port=qubit + ":switch",
#             ),
#             ref_op=ro_pulse,
#             ref_pt="start",
#             rel_time=TWPA_DELAY-TWPA_RINGUP,
#             label=f"TWPA_mark {i})",
#         )

#         schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

#     return schedule

def pump_heterodyne_spec_sched_nco(
    pump_frequency: Union[np.ndarray, float],
    pump_amp: Union[np.ndarray, float],
    pump_duration: Union[np.ndarray, float],
    snail_drive: any,
    pulse_amp: float,
    pulse_duration: float,
    frequencies: np.ndarray,
    acquisition_delay: float,
    integration_time: float,
    port: str,
    clock: str,
    init_duration: float = 10e-6,
    repetitions: int = 1,
    port_out: Optional[str] = None,
) -> Schedule:

    snail_drive.frequency(pump_frequency)

    # ensure pulse_amplitude and pulse_duration are iterable.
    amps = np.asarray(pump_amp)
    amps = amps.reshape(amps.shape or (1,))
    durations = np.asarray(pump_duration)
    durations = durations.reshape(durations.shape or (1,))

    # either the shapes of the amp and duration must match or one of
    # them must be a constant floating point value.
    if len(amps) == 1:
        amps = np.ones(np.shape(durations)) * amps
    elif len(durations) == 1:
        durations = np.ones(np.shape(amps)) * durations
    elif len(durations) != len(amps):
        raise ValueError(
            f"Shapes of pulse_amplitude ({pulse_amp.shape}) and "
            f"pulse_duration ({pulse_duration.shape}) are incompatible."
        )
    
    schedule = Schedule("SNAIL pump heterodyne spectroscopy (NCO sweep)(TWPA)", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequencies.flat[0]))

    if port_out is None:
        port_out = port

    for i, (amp, duration) in enumerate(zip(amps, durations)):
        snail_drive.power(amp)
        for i, freq in enumerate(frequencies):
            schedule.add(IdlePulse(duration=init_duration), label=f"buffer {i}")

            # schedule.add(MarkerPulse(
            #     duration=duration,
            #     port="snail:switch",
            # ),)

            schedule.add(MarkerPulse(
                duration=2e-6,
                port="snail:switch",
            ),)

            schedule.add(IdlePulse(duration=300e-9))

            schedule.add(X(clock[:6]), label=f"pi {i}")

            schedule.add(IdlePulse(duration=duration))
    
            schedule.add(
                SetClockFrequency(clock=clock, clock_freq_new=freq),
                label=f"set_freq {i} ({clock} {freq:e} Hz)",
            )
    
            spec_pulse = schedule.add(
                SquarePulse(
                    duration=pulse_duration,
                    amp=pulse_amp,
                    port=port_out,
                    clock=clock,
                ),
                label=f"spec_pulse {i})",
            )
    
            schedule.add(
                SSBIntegrationComplex(
                    duration=integration_time,
                    port=port,
                    clock=clock,
                    acq_index=i,
                    acq_channel=0,
                    bin_mode=BinMode.AVERAGE,
                ),
                ref_op=spec_pulse,
                ref_pt="start",
                rel_time=acquisition_delay,
                label=f"acquisition {i})",
            )
    
            add_twpa_marker(
                schedule,
                spec_pulse,
                port[:-3] + "switch",
                integration_time,
                label=f"TWPA_mark {i})",
            )
    
            schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule

def pump_t1_sched(
    times: Union[np.ndarray, float],
    snail_drive: any,
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    qubit_e: BasicTransmonElement | Iterable[BasicTransmonElement],
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
    repetitions: int = 1,
) -> Schedule:

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    readout_duration = max_readout_duration(qubits)

    qubit_e_names = [qubit.name for qubit in qubit_e]
    
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))
    
    schedule = Schedule("Pump T1", repetitions)

    for i, tau in enumerate(times):
        reset = schedule.add(Reset(*qubit_names), label=f"Reset {i}")

        for qubit_e in qubit_e_names:
            schedule.add(X(qubit_e), ref_op=reset, ref_pt="end")
        
        schedule.add(MarkerPulse(
                duration=tau,
                port="snail:switch",
            ), rel_time=-38e-9)
        
        schedule.add(IdlePulse(duration=38e-9))
        
        measure_with_twpa(
            schedule,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=i,
            acq_protocol=acq_protocol,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def pump_ramsey_sched(
    times: Union[np.ndarray, float],
    snail_drive: any,
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    qubit_e: BasicTransmonElement | Iterable[BasicTransmonElement],
    artificial_detuning: float = 0,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
    repetitions: int = 1,
) -> Schedule:

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    readout_duration = max_readout_duration(qubits)

    qubit_e_names = [qubit.name for qubit in qubit_e]
    
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))
    
    schedule = Schedule("Pump Ramsey", repetitions)

    for i, tau in enumerate(times):
        reset = schedule.add(Reset(*qubit_names), label=f"Reset {i}")

        for qubit_e in qubit_e_names:
            schedule.add(X90(qubit_e), ref_op=reset, ref_pt="end")
        
        schedule.add(MarkerPulse(
                duration=tau,
                port="snail:switch",
            ), ref_op=reset, ref_pt="end", rel_time=-38e-9)
        
        schedule.add(IdlePulse(duration=38e-9))
        
        # the phase of the second pi/2 phase progresses to propagate
        recovery_phase = np.rad2deg(2 * np.pi * artificial_detuning * tau)
        for qubit_e in qubit_e_names:
            schedule.add(
                Rxy(theta=90, phi=recovery_phase, qubit=qubit_e), ref_op=reset, ref_pt="end", rel_time=tau
            )

        measure_with_twpa(
            schedule,
            *qubit_names,
            readout_duration=readout_duration,
            acq_index=i,
            acq_protocol=acq_protocol,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def pump_t1_and_t2_sched(
    times: Union[np.ndarray, float],
    snail_drive: any,
    qubit_specifier: BasicTransmonElement | Iterable[BasicTransmonElement],
    qubit_e: BasicTransmonElement | Iterable[BasicTransmonElement],
    case: Union[np.ndarray, int],
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
    repetitions: int = 1,
) -> Schedule:

    if hasattr(qubit_specifier, "name"):
        qubits = [qubit_specifier]
    else:
        qubits = [q for q in qubit_specifier]

    qubit_names = [qubit.name for qubit in qubits]
    readout_duration = max_readout_duration(qubits)

    qubit_e_names = [qubit.name for qubit in qubit_e]
    
    # ensure times is an iterable when passing floats.
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))
    
    schedule = Schedule("Pump T1 and T2", repetitions)

    for i, tau in enumerate(times):
        reset = schedule.add(Reset(*qubit_names), label=f"Reset {i}")
        
        if case == 1:
            for qubit_e in qubit_e_names:
                schedule.add(X(qubit_e), ref_op=reset, ref_pt="end")

            schedule.add(MarkerPulse(
                    duration=tau,
                    port="snail:switch",
                ), rel_time=-38e-9)
            
            schedule.add(IdlePulse(duration=38e-9))

            ro_pulse = schedule.add(Measure(*qubit_names, acq_index=i, acq_protocol=acq_protocol), label=f"Measurement {i}")


        elif case == 2:
            schedule.add(X90(qubit))

            schedule.add(MarkerPulse(
                    duration=tau / 2,
                    port="snail:switch",
                ), rel_time=-38e-9)
            schedule.add(IdlePulse(duration=38e-9))

            schedule.add(X(qubit))

            schedule.add(MarkerPulse(
                    duration=tau / 2,
                    port="snail:switch",
                ), rel_time=-38e-9)
            schedule.add(IdlePulse(duration=38e-9))

            schedule.add(X90(qubit))

            ro_pulse = schedule.add(Measure(qubit, acq_index=i, acq_protocol=acq_protocol), label=f"Measurement {i}")

        add_twpa_marker(
            schedule,
            ro_pulse,
            qubit_names[0] + ":switch",
            readout_duration,
            label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")

    return schedule

def pump_RPM(
    qubit: any,
    case: int,
    angles: Union[np.ndarray, float],
    pump_frequency: Union[np.ndarray, float],
    pump_amp: Union[np.ndarray, float],
    pump_duration: Union[np.ndarray, float],
    snail_drive: any,
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition"
    ] = "SSBIntegrationComplex",
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    qubit = qubit.name

    snail_drive.frequency(pump_frequency)
    snail_drive.power(pump_amp)
    
    schedule = Schedule(f"Rabi Population Measurement(TWPA)", repetitions)

    for i, angle in enumerate(angles):
        schedule.add(Reset(qubit), label=f"Reset {i}")

        schedule.add(MarkerPulse(
            duration=pump_duration,
            port="snail:switch",
        ),)

        schedule.add(IdlePulse(duration=100e-9))

        if case == 1:
            DRAG = schedule.add(
                DRAGPulse(
                    duration=E2F_DURATION,
                    G_amp=E2F_G_AMP * angle/180,
                    D_amp=0,
                    port=f"{qubit}:fs",
                    clock=f"{qubit}.12",
                    phase=0,
                ),
                label=f"Theta Pulse {i}",
            )
            schedule.add(X(qubit))

        elif case == 2:
            schedule.add(X(qubit))
            DRAG = schedule.add(
                DRAGPulse(
                    duration=E2F_DURATION,
                    G_amp=E2F_G_AMP * angle/180,
                    D_amp=0,
                    port=f"{qubit}:fs",
                    clock=f"{qubit}.12",
                    phase=0,
                ),
                label=f"Theta Pulse {i}",
            )
            schedule.add(X(qubit))
            
        else:
            raise ValueError(f"Sequence case ({case}) must be either 1 or 2.")
            
        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule

def cavity_charging(
    qubit: BasicTransmonElement,
    pulse_amp: Union[np.ndarray, float],
    frequency: float,
    port: str,
    clock: str,
    init_duration: float = 15e-6,
    repetitions: int = 1,
    acq_protocol: Literal[
        "SSBIntegrationComplex", "ThresholdedAcquisition", "NumericalSeparatedWeightedIntegration"
    ] = "SSBIntegrationComplex",
) -> Schedule:

    readout_duration = qubit.measure.pulse_duration()
    pi_duration = qubit.rxy.duration()
    qubit = qubit.name
    
    schedule = Schedule("Cavity Charging", repetitions)
    schedule.add_resource(ClockResource(name=clock, freq=frequency))

    for i, amp in enumerate(pulse_amp):
        schedule.add(Reset(qubit), label=f"Reset {i}")
        
        charging_pulse = schedule.add(
            SquarePulse(
                duration=pi_duration + init_duration,
                amp=amp,
                port=port,
                clock=clock,
            ),
            label=f"charging_pulse {i})",
        )

        schedule.add(X(qubit), ref_op=charging_pulse, ref_pt="start", rel_time=init_duration, label=f"pi {i}")
        # schedule.add(X(qubit), ref_op=charging_pulse, ref_pt="end", rel_time=200e-9, label=f"pi {i}")

        measure_with_twpa(
            schedule,
            qubit,
            readout_duration=readout_duration,
            acq_index=i,
            acq_protocol=acq_protocol,
            ref_pt="end",
            rel_time=5e-6,
            label=f"Measurement {i}",
            marker_label=f"TWPA_mark {i})",
        )

        schedule.add(IdlePulse(duration=4e-9), label=f"end {i}")
        
    return schedule
