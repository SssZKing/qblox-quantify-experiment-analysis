"""Check that a refactor of qqea.schedules leaves every compiled schedule unchanged.

Builds each schedule builder in ``qqea.schedules`` twice, once from a baseline
commit and once from the working tree, compiles both to device level with the
same mock device, and compares the timing tables (start time, duration, port,
clock and operation of every pulse and acquisition).

No instruments are touched: pump generators and the instrument coordinator are
replaced by fakes, and compilation stops before the hardware backend. Nothing is
written outside a temporary folder.

Run it from the repo root, in an env that has the pinned quantify stack
(``qblox_dev``)::

    python tools/compare_schedules.py
    python tools/compare_schedules.py --base 1d19c85 --only rabi,t1
    python tools/compare_schedules.py --snapshot path/to/snapshot.json

``--base`` defaults to ``1d19c85``, the commit that moved the modules into
``qqea`` unchanged. ``--snapshot`` takes the device parameters from a quantify
dataset snapshot instead of the built-in mock values.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import traceback

sys.dont_write_bytecode = True

DEFAULT_BASE = "1d19c85"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Mock device
# --------------------------------------------------------------------------

ELEMENT_PARAMS = {
    "qubit1": {
        "clock_freqs": {"f01": 5.00e9, "f12": 4.80e9, "readout": 7.00e9},
        "rxy": {"amp180": 0.30, "duration": 40e-9},
        "measure": {
            "pulse_amp": 0.10,
            "pulse_duration": 2.0e-6,
            "acq_delay": 120e-9,
            "integration_time": 1.8e-6,
            "acq_channel": 0,
        },
        "reset": {"duration": 200e-6},
    },
    "qubit2": {
        "clock_freqs": {"f01": 5.26e9, "f12": 5.05e9, "readout": 7.10e9},
        "rxy": {"amp180": 0.32, "duration": 40e-9},
        "measure": {
            "pulse_amp": 0.12,
            "pulse_duration": 1.6e-6,
            "acq_delay": 120e-9,
            "integration_time": 1.4e-6,
            "acq_channel": 1,
        },
        "reset": {"duration": 200e-6},
    },
}
EDGE_PARAMS = {
    "pulse_frequency": 259.8e6,
    "pulse_amp": 5.0,
    "pulse_duration": 400e-9,
    "ac_stark_phase_q1": 98.5,
    "ac_stark_phase_q2": -106.5,
}


def _edge_class():
    """The package's iSWAP edge, or the SNL315 notebook version for a baseline that predates it."""
    try:
        from qqea.schedules.edges import CompositeiSWAPEdge

        return CompositeiSWAPEdge
    except ImportError:
        pass

    from qcodes.instrument import InstrumentChannel
    from qcodes.instrument.parameter import ManualParameter
    from quantify_scheduler.backends.graph_compilation import OperationCompilationConfig
    from quantify_scheduler.device_under_test.edge import Edge
    from quantify_scheduler.helpers.validators import Numbers

    def composite_marker_pulse(pulse_frequency, pulse_amp, pulse_duration):
        from quantify_scheduler.operations.pulse_library import MarkerPulse

        return MarkerPulse(duration=pulse_duration, port="snail:switch")

    class iSWAPChannel(InstrumentChannel):
        def __init__(self, parent, name, **kwargs):
            super().__init__(parent=parent, name=name)
            for p in EDGE_PARAMS:
                self.add_parameter(
                    p,
                    parameter_class=ManualParameter,
                    initial_value=kwargs.get(p, 0.0),
                    vals=Numbers(allow_nan=True),
                )

    class CompositeiSWAPEdge(Edge):
        def __init__(self, parent_element_name, child_element_name, **kwargs):
            iswap_data = kwargs.pop("iswap", {})
            super().__init__(
                parent_element_name=parent_element_name,
                child_element_name=child_element_name,
                **kwargs,
            )
            self.add_submodule("iswap", iSWAPChannel(parent=self, name="iswap", **iswap_data))

        def generate_edge_config(self):
            return {
                self.name: {
                    "iSWAP": OperationCompilationConfig(
                        factory_func=composite_marker_pulse,
                        factory_kwargs={
                            "pulse_frequency": self.iswap.pulse_frequency(),
                            "pulse_amp": self.iswap.pulse_amp(),
                            "pulse_duration": self.iswap.pulse_duration(),
                        },
                    )
                }
            }

    return CompositeiSWAPEdge


def _apply_snapshot(instruments, path):
    with open(path) as f:
        snap = json.load(f)
    snap = snap.get("instruments", snap)
    for name, inst in instruments.items():
        if name not in snap:
            continue
        for sub_name, sub in snap[name].get("submodules", {}).items():
            if not hasattr(inst, sub_name):
                continue
            submodule = getattr(inst, sub_name)
            for p_name, p in sub.get("parameters", {}).items():
                value = p.get("value")
                if value is None or p_name not in submodule.parameters:
                    continue
                try:
                    submodule.parameters[p_name](value)
                except Exception:  # noqa: BLE001  read-only or invalid in this version
                    pass


def build_device(snapshot=None):
    from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
    from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

    qd = QuantumDevice("compare_device")
    elements = {}
    for name, params in ELEMENT_PARAMS.items():
        q = BasicTransmonElement(name)
        for sub, values in params.items():
            for p, v in values.items():
                try:
                    getattr(q, sub).parameters[p](v)
                except Exception as exc:  # noqa: BLE001  same on both sides, so only warn
                    print(f"  warning: could not set {name}.{sub}.{p}: {exc}")
        qd.add_element(q)
        elements[name] = q
    edge = _edge_class()(parent_element_name="qubit1", child_element_name="qubit2")
    for p, v in EDGE_PARAMS.items():
        edge.iswap.parameters[p](v)
    qd.add_edge(edge)
    if snapshot:
        _apply_snapshot({**elements, edge.name: edge}, snapshot)
    return qd, elements["qubit1"], elements["qubit2"]


class FakeSource:
    """Stands in for a signal generator; records every call."""

    def __init__(self, name, log):
        self._name = name
        self._log = log

    def __getattr__(self, attr):
        def call(*args, **kwargs):
            self._log.append([self._name, attr, [_jsonable(a) for a in args], sorted(kwargs)])

        return call


class FakeCoordinator:
    """Stands in for the InstrumentCoordinator; keeps the compiled schedules."""

    def __init__(self):
        self.compiled = []

    def prepare(self, compiled_schedule):
        self.compiled.append(compiled_schedule)

    def start(self):
        pass

    def wait_done(self, timeout_sec=None):
        pass

    def retrieve_acquisition(self):
        return None


# --------------------------------------------------------------------------
# Cases: (case name, module, function, kwargs builder)
# --------------------------------------------------------------------------


def cases():
    import numpy as np

    lin = np.linspace
    times = np.array([100e-9, 1.1e-6, 2.1e-6])
    ro_freqs = lin(7.000e9, 7.002e9, 3)
    spec = dict(acquisition_delay=120e-9, integration_time=1.8e-6, port="qubit1:res", clock="qubit1.ro")

    sq, tq = "single_qubit", "two_qubit"
    c = [
        ("time_of_flight", sq, "time_of_flight_calibration", lambda e: dict(
            port="qubit1:res", clock="qubit1.ro", pulse_duration=1e-6, pulse_amp=0.1,
            acq_duration=2e-6, time_of_flight=200e-9, TWPA_delay=-36e-9)),
        ("time_of_flight_drag", sq, "time_of_flight_calibration", lambda e: dict(
            port="qubit1:res", clock="qubit1.ro", pulse_duration=1e-6, pulse_amp=0.1,
            acq_duration=2e-6, time_of_flight=200e-9, pulse_type="DRAG")),
        ("sweep_optimal", sq, "sweep_optimal_TWPA", lambda e: dict(
            TWPA_pump=e["pump"], TWPA_freq=8e9, TWPA_power=-5.0, pulse_amp=0.1,
            pulse_duration=2e-6, frequencies=ro_freqs, **spec)),
        ("heterodyne", sq, "heterodyne_spec_sched_nco_TWPA", lambda e: dict(
            pulse_amp=0.1, pulse_duration=2e-6, frequencies=ro_freqs, **spec)),
        ("multiplexed_heterodyne", sq, "multiplexed_heterodyne_spec_sched_nco_TWPA", lambda e: dict(
            pulse_amp=0.1, pulse_duration=2e-6, frequencies_qubit1=ro_freqs,
            frequencies_qubit2=lin(7.100e9, 7.102e9, 3), frequencies_idx=np.arange(3),
            acquisition_delay=120e-9, integration_time=1.8e-6,
            ports=["qubit1:res", "qubit2:res"], clocks=["qubit1.ro", "qubit2.ro"])),
        ("two_tone", sq, "two_tone_spec_sched_nco_TWPA", lambda e: dict(
            spec_pulse_amp=0.05, spec_pulse_duration=10e-6, spec_pulse_port="qubit1:mw",
            spec_pulse_clock="qubit1.01", spec_pulse_frequencies=lin(4.99e9, 5.01e9, 3),
            ro_pulse_amp=0.1, ro_pulse_duration=2e-6, ro_pulse_delay=200e-9,
            ro_pulse_port="qubit1:res", ro_pulse_clock="qubit1.ro", ro_pulse_frequency=7e9,
            ro_acquisition_delay=120e-9, ro_integration_time=1.8e-6, init_duration=10e-6)),
        ("multiplexed_two_tone", sq, "multiplexed_two_tone_spec_sched_nco_TWPA", lambda e: dict(
            qubits=[e["q1"], e["q2"]], pulse_amp=0.05, pulse_duration=10e-6,
            frequencies_qubit1=lin(4.99e9, 5.01e9, 3), frequencies_qubit2=lin(5.25e9, 5.27e9, 3),
            frequencies_idx=np.arange(3), ro_pulse_delay=200e-9)),
        ("rabi", sq, "rabi_sched_TWPA", lambda e: dict(
            pulse_amp=lin(0, 0.5, 3), pulse_duration=40e-9, frequency=5e9, qubit=e["q1"])),
        ("rabi_amplification", sq, "rabi_amplification_TWPA", lambda e: dict(
            pulse_amp=lin(0.25, 0.35, 3), pulse_duration=40e-9, pi_number=3, frequency=5e9,
            qubit=e["q1"])),
        ("ramsey", sq, "ramsey_sched_TWPA", lambda e: dict(
            times=times, qubit=e["q1"], artificial_detuning=1e6)),
        ("stark_ramsey", sq, "stark_ramsey_sched_TWPA", lambda e: dict(
            times=times, qubit=e["q1"], stark_amp=0.1, stark_freq=5.05e9)),
        ("coupled_ramsey", sq, "coupled_ramsey_sched_TWPA", lambda e: dict(
            times=times, qubit=e["q1"], qubit_c=e["q2"])),
        ("coupled_phase_ramsey", sq, "coupled_phase_ramsey_sched_TWPA", lambda e: dict(
            phases=lin(0, 360, 3), qubit=e["q1"], qubit_c=e["q2"])),
        ("echo", sq, "echo_sched_TWPA", lambda e: dict(times=times, qubit=e["q1"])),
        ("t1", sq, "t1_sched_TWPA", lambda e: dict(times=times, qubit=e["q1"])),
    ]
    for case in (1, 2, 3):
        c += [
            (f"t1_and_t2_case{case}", sq, "t1_and_t2_TWPA", lambda e, case=case: dict(
                times=times, qubit=e["q1"], case=case, artificial_detuning=1e6)),
            (f"multi_qubit_t1_and_t2_case{case}", sq, "multi_qubit_t1_and_t2_TWPA",
             lambda e, case=case: dict(times=times, qubit_specifier=[e["q1"], e["q2"]], case=case)),
        ]
    c += [
        ("multiplexed_readout_calibration", sq, "multiplexed_readout_calibration_sched_TWPA",
         lambda e: dict(qubits=[e["q1"], e["q2"]], prepared_states=[0, 1])),
        ("readout_calibration", sq, "readout_calibration_sched_TWPA", lambda e: dict(
            qubit=e["q1"], prepared_states=[0, 1, 2])),
        ("readout_freq_optimization", sq, "readout_freq_optimization_TWPA", lambda e: dict(
            qubit=e["q1"], frequencies=lin(7.000e9, 7.001e9, 2), instrument_coordinator=e["ic"],
            quantum_device=e["qd"])),
        ("readout_amp_optimization", sq, "readout_amp_optimization_TWPA", lambda e: dict(
            qubit=e["q1"], amps=[0.1, 0.2], instrument_coordinator=e["ic"], quantum_device=e["qd"])),
        ("readout_len_optimization", sq, "readout_len_optimization_TWPA", lambda e: dict(
            qubit=e["q1"], lens=[1e-6, 2e-6], instrument_coordinator=e["ic"], quantum_device=e["qd"])),
        ("readout_weight_optimization", sq, "readout_weight_optimization_TWPA", lambda e: dict(
            qubit=e["q1"], instrument_coordinator=e["ic"], quantum_device=e["qd"])),
        ("allxy", sq, "allxy_sched_TWPA", lambda e: dict(qubit=e["q1"])),
        ("drag_calibration", sq, "DRAG_calibration_sched", lambda e: dict(
            qubit=e["q1"], motzoi_list=lin(-0.1, 0.1, 3), instrument_coordinator=e["ic"],
            quantum_device=e["qd"])),
        ("dressed_e_cavity", sq, "dressed_e_cavity_TWPA", lambda e: dict(
            pulse_amp=0.1, pulse_duration=2e-6, frequencies=ro_freqs, qubit="qubit1", **spec)),
        ("f_state_spec", sq, "f_state_spec_sched_nco_TWPA", lambda e: dict(
            spec_pulse_amp=0.05, spec_pulse_duration=10e-6, spec_pulse_port="qubit1:fs",
            spec_pulse_clock="qubit1.12", spec_pulse_frequencies=lin(4.79e9, 4.81e9, 3),
            qubit=e["q1"])),
        ("f_state_rabi", sq, "f_state_rabi_sched_TWPA", lambda e: dict(
            pulse_amp=lin(0, 0.5, 3), pulse_duration=400e-9, frequency=4.8e9, qubit=e["q1"])),
        ("f_state_cavity", sq, "f_state_cavity_TWPA", lambda e: dict(
            pulse_amp=0.1, pulse_duration=2e-6, frequencies=ro_freqs, qubit="qubit1", **spec)),
        ("f_state_t1_case1", sq, "f_state_t1_TWPA", lambda e: dict(times=times, qubit=e["q1"], case=1)),
        ("f_state_t1_case2", sq, "f_state_t1_TWPA", lambda e: dict(times=times, qubit=e["q1"], case=2)),
        ("multiplex_IQ", sq, "multiplex_IQ_TWPA", lambda e: dict(
            qubit=e["q1"], prepared_states=[0, 1, 2], multiplexing_freq=[7.000e9, 7.001e9, 7.002e9])),
        ("RO_raw_trace", sq, "RO_raw_trace_TWPA", lambda e: dict(
            qubit=e["q1"], trace_time=2e-6, prepared_states=[0, 1, 2],
            multiplexing_freq=[7.000e9, 7.001e9, 7.002e9])),
        ("rabi_population_case1", sq, "rabi_population_TWPA", lambda e: dict(
            qubit=e["q1"], case=1, angles=lin(0, 180, 3))),
        ("rabi_population_case2", sq, "rabi_population_TWPA", lambda e: dict(
            qubit=e["q1"], case=2, angles=lin(0, 180, 3))),
        ("snail_swap", sq, "SNAIL_swap_sched_TWPA", lambda e: dict(
            pulse_frequency=6e9, pulse_amp=np.array([-10.0, -5.0]), pulse_duration=200e-9,
            snail_drive=e["snail"], qubit_specifier=[e["q1"], e["q2"]], qubit_e=[e["q1"]],
            swap_type="iSWAP")),
        ("snail_spec", sq, "SNAIL_spec_sched_TWPA", lambda e: dict(
            spec_pulse_amp=-10.0, spec_pulse_duration=2e-6, spec_pulse_frequencies=lin(6e9, 6.01e9, 3),
            snail_drive=e["snail"], qubit=e["q1"], init_duration=10e-6)),
        ("pump_heterodyne", sq, "pump_heterodyne_spec_sched_nco_TWPA", lambda e: dict(
            pump_frequency=6e9, pump_amp=-10.0, pump_duration=200e-9, snail_drive=e["snail"],
            pulse_amp=0.1, pulse_duration=2e-6, frequencies=ro_freqs, **spec)),
        ("pump_t1", sq, "pump_t1_sched_TWPA", lambda e: dict(
            times=times, snail_drive=e["snail"], qubit_specifier=[e["q1"], e["q2"]], qubit_e=[e["q1"]])),
        ("pump_ramsey", sq, "pump_ramsey_sched_TWPA", lambda e: dict(
            times=times, snail_drive=e["snail"], qubit_specifier=[e["q1"], e["q2"]], qubit_e=[e["q1"]])),
        ("pump_t1_and_t2_case1", sq, "pump_t1_and_t2_sched_TWPA", lambda e: dict(
            times=times, snail_drive=e["snail"], qubit_specifier=[e["q1"], e["q2"]],
            qubit_e=[e["q1"]], case=1)),
        ("pump_RPM", sq, "pump_RPM_TWPA", lambda e: dict(
            qubit=e["q1"], case=1, angles=lin(0, 180, 3), pump_frequency=6e9, pump_amp=-10.0,
            pump_duration=200e-9, snail_drive=e["snail"])),
        ("cavity_charging", sq, "cavity_charging_TWPA", lambda e: dict(
            qubit=e["q1"], pulse_amp=[0.1, 0.2], frequency=7e9, port="qubit1:res", clock="qubit1.ro")),
    ]

    pair = lambda e: [e["q1"], e["q2"]]  # noqa: E731
    c += [
        ("rb_1q", tq, "randomized_benchmarking_schedule", lambda e: dict(
            qubit_specifier=e["q1"], lengths=[1, 3, 1, 1], seeds=[11, 12, 0, 0])),
        ("rb_2q", tq, "randomized_benchmarking_schedule", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 2, 1, 1], seeds=[11, 12, 0, 0])),
        ("simultaneous_rb", tq, "simultaneous_randomized_benchmarking_schedule", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 3, 1, 1], seeds=[11, 12, 0, 0])),
        ("iswap_pulsed_pump", tq, "iswap_pulsed_pump_schedule", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 3])),
        ("iswap_pulsed_pump_spaced", tq, "iswap_pulsed_pump_schedule", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 3], iswap_spacing=100e-9)),
        ("iswap_tail_1q", tq, "iswap_tail_1q_schedule", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 2], gap=40e-9)),
        ("pump_probe_ramsey", tq, "pump_probe_ramsey_schedule", lambda e: dict(
            qubit_specifier=pair(e), delays=[0.0, 40e-9])),
        ("pump_probe_swap", tq, "pump_probe_swap_schedule", lambda e: dict(
            qubit_specifier=pair(e), durations=[0.0, 200e-9])),
        ("iswap_virtual_z", tq, "iswap_virtual_z_schedule", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 2], angle=10.0, gate_idx=16)),
        ("iswap_RPE_f", tq, "iswap_RPE_f", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 2], quantum_device=e["qd"])),
        ("iswap_RPE_f_y", tq, "iswap_RPE_f", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 2], quantum_device=e["qd"], final_axis="y")),
        ("iswap_RPE_theta_sum", tq, "iswap_RPE_theta_sum", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 2], quantum_device=e["qd"])),
        ("iswap_RPE_d", tq, "iswap_RPE_d", lambda e: dict(
            qubit_specifier=pair(e), lengths=[1, 2], sqrt_iswap_duration=200e-9,
            quantum_device=e["qd"])),
        ("iswap_delay", tq, "iswap_delay_schedule", lambda e: dict(
            qubit_specifier=pair(e), gate_idx=16, delays=[0.0, 40e-9])),
        ("tomography_pre_build", tq, "tomography_pre_build", lambda e: dict(
            qubit_specifier=pair(e), operation_state_idx=5760, quantum_device=e["qd"])),
        ("tomography", tq, "tomography_schedule", lambda e: dict(
            qubit_specifier=pair(e), operation_state_idx=5760, instrument_coordinator=e["ic"],
            quantum_device=e["qd"], correction_phase=e["mod"].tomography_pre_build(
                pair(e), 5760, e["qd"]))),
        ("tomography_1q", tq, "tomography_schedule", lambda e: dict(
            qubit_specifier=e["q1"], operation_state_idx=16, instrument_coordinator=e["ic"],
            quantum_device=e["qd"])),
        ("process_tomography_pre_build", tq, "process_tomography_pre_build", lambda e: dict(
            qubit_specifier=pair(e), operation_state_idx=5760, experiment_list=[0, 7, 40],
            quantum_device=e["qd"])),
        ("process_tomography", tq, "process_tomography_schedule", lambda e: dict(
            qubit_specifier=pair(e), operation_state_idx=5760, experiment_list=[0, 7, 40],
            instrument_coordinator=e["ic"], quantum_device=e["qd"],
            correction_phase=e["mod"].process_tomography_pre_build(
                pair(e), 5760, [0, 7, 40], e["qd"]))),
        ("gst_pre_build", tq, "GST_pre_build", lambda e: dict(
            qubit_specifier=pair(e), experiment_list=_gst_circuits(), quantum_device=e["qd"])),
        ("gst", tq, "GST_schedule", lambda e: dict(
            qubit_specifier=pair(e), experiment_list=_gst_circuits(), instrument_coordinator=e["ic"],
            quantum_device=e["qd"], correction_phase=e["mod"].GST_pre_build(
                pair(e), _gst_circuits(), e["qd"]))),
    ]
    return c


def _gst_circuits():
    from pygsti.circuits import Circuit

    return [
        Circuit([("Gxpi2", 0), ("Gypi2", 1)], line_labels=(0, 1)),
        Circuit([("Gxpi2", 0), ("Giswap", 0, 1), ("Gympi2", 1)], line_labels=(0, 1)),
        Circuit([("Giswap", 0, 1), ("Giswap", 0, 1), ("Gx", 0)], line_labels=(0, 1)),
    ]


# --------------------------------------------------------------------------
# Dump one side
# --------------------------------------------------------------------------


def _jsonable(x):
    import numpy as np

    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x if isinstance(x, (int, float, str, bool, type(None))) else repr(x)


def _rows(schedule):
    table = schedule.timing_table.data
    rows = []
    for r in table.itertuples(index=False):
        rows.append([
            round(float(r.abs_time) * 1e12, 3),  # ps
            round(float(r.duration) * 1e12, 3),
            str(r.port),
            str(r.clock),
            bool(r.is_acquisition),
            str(r.operation),
        ])
    rows.sort(key=lambda row: (row[0], row[2] or "", row[3] or "", row[5]))
    return rows


def dump(src, out, snapshot, only):
    sys.path.insert(0, src)
    import importlib

    import qqea

    if not os.path.abspath(qqea.__file__).startswith(os.path.abspath(src)):
        raise SystemExit(f"qqea imported from {qqea.__file__}, not from {src}")

    from quantify_scheduler.backends import SerialCompiler
    from quantify_scheduler.schedules.schedule import ScheduleBase

    results = {}
    for name, module, func, make_kwargs in cases():
        if only and not any(o in name for o in only):
            continue
        qd, q1, q2 = build_device(snapshot)
        log = []
        mod = importlib.import_module(f"qqea.schedules.{module}")
        env = {
            "qd": qd, "q1": q1, "q2": q2, "ic": FakeCoordinator(), "mod": mod,
            "pump": FakeSource("TWPA_pump", log), "snail": FakeSource("snail_drive", log),
        }
        entry = {"function": f"{module}.{func}"}
        try:
            value = getattr(mod, func)(**make_kwargs(env))
            schedules = list(env["ic"].compiled)
            if isinstance(value, ScheduleBase):
                compiled = SerialCompiler(name="compare").compile(
                    schedule=value, config=qd.generate_compilation_config()
                )
                schedules.append(compiled)
                entry["value"] = None
            else:
                entry["value"] = _jsonable(value)
            entry["schedules"] = [_rows(s) for s in schedules]
            entry["calls"] = log
            entry["status"] = "ok"
        except Exception as exc:  # noqa: BLE001  recorded and compared
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc(limit=-3)
        finally:
            qd.close_all()
        results[name] = entry
        print(f"  {name}: {entry['status']}", flush=True)

    with open(out, "w") as f:
        json.dump(results, f)


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------


def _close(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def _diff_rows(a, b, limit=6):
    lines = []
    sa = {json.dumps(r) for r in a}
    sb = {json.dumps(r) for r in b}
    for r in sorted(sa - sb)[:limit]:
        lines.append(f"      - {r}")
    for r in sorted(sb - sa)[:limit]:
        lines.append(f"      + {r}")
    return lines


def compare(base, new):
    report, n_same, n_diff = [], 0, 0
    for name in new:
        b, n = base.get(name), new[name]
        if b is None:
            report.append(f"NEW        {name}")
            continue
        if b["status"] == "error" or n["status"] == "error":
            if b["status"] == n["status"] == "error" and b["error"] == n["error"]:
                report.append(f"same error {name}: {n['error']}")
                n_same += 1
            else:
                report.append(f"DIFFERENT  {name}: base {b.get('error', 'ok')} / new {n.get('error', 'ok')}")
                if n["status"] == "error":
                    report.append(n["traceback"])
                n_diff += 1
            continue
        problems = []
        if len(b["schedules"]) != len(n["schedules"]):
            problems.append(f"    {len(b['schedules'])} vs {len(n['schedules'])} compiled schedules")
        else:
            for i, (rb, rn) in enumerate(zip(b["schedules"], n["schedules"])):
                if rb != rn:
                    problems.append(f"    schedule {i}: {len(rb)} vs {len(rn)} rows")
                    problems += _diff_rows(rb, rn)
        if not _close(b["value"], n["value"]):
            problems.append(f"    return value {b['value']} vs {n['value']}")
        if b["calls"] != n["calls"]:
            problems.append(f"    instrument calls {b['calls']} vs {n['calls']}")
        if problems:
            report.append(f"DIFFERENT  {name}")
            report += problems
            n_diff += 1
        else:
            rows = sum(len(s) for s in n["schedules"])
            report.append(f"same       {name} ({rows} pulses/acquisitions)")
            n_same += 1
    report.append(f"\n{n_same} same, {n_diff} different")
    return "\n".join(report), n_diff


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _extract(ref, dest):
    data = subprocess.run(
        ["git", "-C", REPO, "archive", ref, "src"], check=True, stdout=subprocess.PIPE
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest)
    return os.path.join(dest, "src")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")
    d = sub.add_parser("dump", help="internal: dump one side")
    d.add_argument("--src", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--snapshot")
    d.add_argument("--only", default="")
    parser.add_argument("--base", default=DEFAULT_BASE, help="git ref of the baseline (default %(default)s)")
    parser.add_argument("--snapshot", help="quantify snapshot.json to take device parameters from")
    parser.add_argument("--only", default="", help="comma-separated substrings of case names")
    parser.add_argument("--report", help="also write the report to this file")
    args = parser.parse_args()
    only = [o for o in args.only.split(",") if o]

    if args.cmd == "dump":
        dump(args.src, args.out, args.snapshot, only)
        return

    with tempfile.TemporaryDirectory(prefix="qqea_compare_") as tmp:
        sides = {"base": _extract(args.base, os.path.join(tmp, "base")),
                 "new": os.path.join(REPO, "src")}
        dumps = {}
        for side, src in sides.items():
            out = os.path.join(tmp, f"{side}.json")
            print(f"Building schedules from {side} ({args.base if side == 'base' else 'working tree'})")
            cmd = [sys.executable, os.path.abspath(__file__), "dump", "--src", src, "--out", out]
            if args.snapshot:
                cmd += ["--snapshot", args.snapshot]
            if only:
                cmd += ["--only", ",".join(only)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
            subprocess.run(cmd, check=True, cwd=tmp, env=env)
            with open(out) as f:
                dumps[side] = json.load(f)

    report, n_diff = compare(dumps["base"], dumps["new"])
    print("\n" + report)
    if args.report:
        with open(args.report, "w") as f:
            f.write(report + "\n")
    sys.exit(1 if n_diff else 0)


if __name__ == "__main__":
    main()
