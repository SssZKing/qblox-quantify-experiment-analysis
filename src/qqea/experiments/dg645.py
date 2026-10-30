from typing import TYPE_CHECKING

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.validators import Enum, Ints, Numbers

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter


class DG645(VisaInstrument):
    """
    QCoDeS driver for the Stanford Research Systems DG645 digital delay generator.

    Ethernet raw socket: ``DG645("DG645_1", "TCPIP0::172.16.0.10::5025::SOCKET")``.

    Delay channels: T0, T1, A, B, C, D, E, F, G, H. Outputs: T0, AB, CD, EF, GH
    (output AB is high from the A delay to the B delay, etc.).

    ``delay_<ch>`` is the delay in seconds of channel <ch> relative to its reference
    channel ``ref_<ch>``; setting one keeps the other.

    Output levels (``amplitude_<out>``, ``offset_<out>``) are calibrated for a 50 Ohm
    load; into a high-impedance input the voltage is about twice the setting.

    Replaces qcodes_contrib_drivers' DG645 (v0.22.0), which fails in __init__
    (undefined ``k`` in a label), sends BURC when setting the burst period, and leaves
    the *CAL? / *TST? replies unread.
    """

    default_terminator = "\r\n"

    CHANNELS = {"T0": 0, "T1": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6, "F": 7, "G": 8, "H": 9}
    OUTPUTS = {"T0": 0, "AB": 1, "CD": 2, "EF": 3, "GH": 4}
    TRIGGER_SOURCES = {
        "internal": 0,
        "ext_rising": 1,
        "ext_falling": 2,
        "single_ext_rising": 3,
        "single_ext_falling": 4,
        "single": 5,
        "line": 6,
    }

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name, address, **kwargs)

        # trigger
        self.trigger_source: Parameter = self.add_parameter(
            name="trigger_source",
            label="Trigger source",
            get_cmd="TSRC?",
            set_cmd="TSRC {}",
            val_mapping=self.TRIGGER_SOURCES,
        )
        """Parameter trigger_source"""
        self.trigger_level: Parameter = self.add_parameter(
            name="trigger_level",
            label="Trigger threshold",
            unit="V",
            get_cmd="TLVL?",
            set_cmd="TLVL {}",
            get_parser=float,
            vals=Numbers(min_value=-3.5, max_value=3.5),
        )
        """Parameter trigger_level"""
        self.trigger_rate: Parameter = self.add_parameter(
            name="trigger_rate",
            label="Internal trigger rate",
            unit="Hz",
            get_cmd="TRAT?",
            set_cmd="TRAT {}",
            get_parser=float,
            vals=Numbers(min_value=100e-6, max_value=10e6),
        )
        """Parameter trigger_rate"""
        self.trigger_holdoff: Parameter = self.add_parameter(
            name="trigger_holdoff",
            label="Trigger holdoff",
            unit="s",
            get_cmd="HOLD?",
            set_cmd="HOLD {}",
            get_parser=float,
            vals=Numbers(min_value=0, max_value=2000),
        )
        """Parameter trigger_holdoff"""
        self.advanced_trigger: Parameter = self.add_parameter(
            name="advanced_trigger",
            label="Advanced triggering",
            get_cmd="ADVT?",
            set_cmd="ADVT {}",
            val_mapping={"OFF": 0, "ON": 1},
        )
        """Parameter advanced_trigger"""

        # burst
        self.burst_mode: Parameter = self.add_parameter(
            name="burst_mode",
            label="Burst mode",
            get_cmd="BURM?",
            set_cmd="BURM {}",
            val_mapping={"OFF": 0, "ON": 1},
        )
        """Parameter burst_mode"""
        self.burst_count: Parameter = self.add_parameter(
            name="burst_count",
            label="Burst count",
            get_cmd="BURC?",
            set_cmd="BURC {}",
            get_parser=int,
            vals=Ints(min_value=1),
        )
        """Parameter burst_count"""
        self.burst_delay: Parameter = self.add_parameter(
            name="burst_delay",
            label="Burst delay",
            unit="s",
            get_cmd="BURD?",
            set_cmd="BURD {}",
            get_parser=float,
            vals=Numbers(min_value=0, max_value=2000),
        )
        """Parameter burst_delay"""
        self.burst_period: Parameter = self.add_parameter(
            name="burst_period",
            label="Burst period",
            unit="s",
            get_cmd="BURP?",
            set_cmd="BURP {}",
            get_parser=float,
            vals=Numbers(min_value=100e-9, max_value=2000 - 10e-9),
        )
        """Parameter burst_period"""
        self.burst_T0_config: Parameter = self.add_parameter(
            name="burst_T0_config",
            label="Burst T0 configuration",
            get_cmd="BURT?",
            set_cmd="BURT {}",
            val_mapping={"every_delay_cycle": 0, "first_delay_cycle": 1},
        )
        """Parameter burst_T0_config"""

        # delay channels A..H: delay (s) relative to a reference channel
        for ch in ("A", "B", "C", "D", "E", "F", "G", "H"):
            self.add_parameter(
                name=f"delay_{ch}",
                label=f"{ch} delay",
                unit="s",
                get_cmd=lambda ch=ch: self._get_delay(ch)[1],
                set_cmd=lambda value, ch=ch: self.set_delay(ch, self._get_delay(ch)[0], value),
                vals=Numbers(min_value=-2000, max_value=2000),
            )
            self.add_parameter(
                name=f"ref_{ch}",
                label=f"{ch} delay reference",
                get_cmd=lambda ch=ch: self._get_delay(ch)[0],
                set_cmd=lambda ref, ch=ch: self.set_delay(ch, ref, self._get_delay(ch)[1]),
                vals=Enum(*(c for c in self.CHANNELS if c not in ("T1", ch))),
            )

        # outputs T0, AB, CD, EF, GH
        for out, idx in self.OUTPUTS.items():
            self.add_parameter(
                name=f"amplitude_{out}",
                label=f"Output {out} amplitude (50 Ohm)",
                unit="V",
                get_cmd=f"LAMP?{idx}",
                set_cmd=f"LAMP {idx},{{}}",
                get_parser=float,
                vals=Numbers(min_value=0.5, max_value=5.0),
            )
            self.add_parameter(
                name=f"offset_{out}",
                label=f"Output {out} offset (50 Ohm)",
                unit="V",
                get_cmd=f"LOFF?{idx}",
                set_cmd=f"LOFF {idx},{{}}",
                get_parser=float,
                vals=Numbers(min_value=-2.0, max_value=2.0),
            )
            self.add_parameter(
                name=f"polarity_{out}",
                label=f"Output {out} polarity",
                get_cmd=f"LPOL?{idx}",
                set_cmd=f"LPOL {idx},{{}}",
                val_mapping={"NEG": 0, "POS": 1},
            )
            if idx > 0:
                self.add_parameter(
                    name=f"prescale_{out}",
                    label=f"Output {out} prescale factor",
                    get_cmd=f"PRES?{idx}",
                    set_cmd=f"PRES {idx},{{}}",
                    get_parser=int,
                    vals=Ints(min_value=1),
                )
                self.add_parameter(
                    name=f"prescale_phase_{out}",
                    label=f"Output {out} prescale phase",
                    get_cmd=f"PHAS?{idx}",
                    set_cmd=f"PHAS {idx},{{}}",
                    get_parser=int,
                    vals=Ints(min_value=0),
                )
        self.trigger_prescale: Parameter = self.add_parameter(
            name="trigger_prescale",
            label="Trigger prescale factor",
            get_cmd="PRES?0",
            set_cmd="PRES 0,{}",
            get_parser=int,
            vals=Ints(min_value=1),
        )
        """Parameter trigger_prescale"""

        self.add_function("reset", call_cmd="*RST")
        self.add_function("clear_status", call_cmd="*CLS")
        self.add_function("trigger", call_cmd="*TRG")

        self.connect_message()

    # delays
    def _get_delay(self, channel: str) -> tuple:
        """(reference channel name, delay in s) of ``channel``."""
        ref, delay = self.ask(f"DLAY?{self.CHANNELS[channel]}").split(",")
        names = {v: k for k, v in self.CHANNELS.items()}
        return names[int(ref)], float(delay)

    def set_delay(self, channel: str, reference: str, delay: float) -> None:
        """Set ``channel`` to ``delay`` seconds after ``reference`` (e.g. set_delay("B", "A", 500e-9))."""
        self.write(f"DLAY {self.CHANNELS[channel]},{self.CHANNELS[reference]},{delay:.12e}")

    def set_pulse(self, output: str = "AB", start: float = 0.0, width: float = 100e-9, reference: str = "T0",
                  amplitude: float = None, offset: float = None, polarity: str = "POS") -> None:
        """One output pulse: first channel of ``output`` at ``start`` after ``reference``,
        second channel ``width`` after the first. Amplitude/offset are 50 Ohm values."""
        first, second = output[0], output[1]
        self.set_delay(first, reference, start)
        self.set_delay(second, first, width)
        if amplitude is not None:
            self.parameters[f"amplitude_{output}"](amplitude)
        if offset is not None:
            self.parameters[f"offset_{output}"](offset)
        self.parameters[f"polarity_{output}"](polarity)

    # status
    def last_error(self) -> int:
        """Last error code (0 = no error); reading clears it."""
        return int(self.ask("LERR?"))

    def errors(self, max_count: int = 20) -> list:
        """All pending error codes."""
        out = []
        for _ in range(max_count):
            e = self.last_error()
            if e == 0:
                break
            out.append(e)
        return out

    def operation_complete(self) -> bool:
        return self.ask("*OPC?").strip() == "1"

    def self_test(self) -> int:
        """Run the self test; returns 0 on success (reads the reply, unlike the contrib driver)."""
        return int(self.ask("*TST?"))

    def self_calibrate(self) -> int:
        """Run the auto-calibration; returns 0 on success."""
        return int(self.ask("*CAL?"))

    def save_settings(self, location: int) -> None:
        """Save the instrument settings to location 1-9."""
        self.write(f"*SAV {int(location)}")

    def recall_settings(self, location: int) -> None:
        """Recall the instrument settings from location 0-9 (0 = factory defaults)."""
        self.write(f"*RCL {int(location)}")

    def local(self) -> None:
        self.write("LCAL")

    def remote(self) -> None:
        self.write("REMT")


MAX_SWITCH_DRIVE_V = 2.0     # 50-Ohm amplitude limit for outputs driving the HMC-C019 Vctl (~4 V into high-Z); never higher


ALL_OUTPUTS = ("AB", "CD", "EF", "GH")


def configure_pump_gate(dg: DG645, outputs=ALL_OUTPUTS, start: float = 0.0, width: float = 500e-9,
                        trigger_level: float = 1.0) -> None:
    """Set ``outputs`` (default all four: the rear-panel AB+CD+EF+GH port ORs them, and the front AB/CD may drive the
    two Vctl lines directly -- either way every output must carry the same pulse) to one identical pulse
    ``start`` .. ``start + width`` after T0, 2.0 V amplitude / 0 V offset / positive, external rising trigger.
    Burst mode off. Reads the levels back and raises if they are not the safe values."""
    dg.burst_mode("OFF") if "burst_mode" in dg.parameters else None
    for out in outputs:
        idx = dg.OUTPUTS[out]
        dg.write(f"LOFF {idx},0")
        dg.write(f"LAMP {idx},{MAX_SWITCH_DRIVE_V}")
        dg.write(f"LPOL {idx},1")
        dg.set_delay(out[0], "T0", start)
        dg.set_delay(out[1], out[0], width)
    dg.write(f"TLVL {trigger_level}")
    dg.write("TSRC 1")
    check_pump_gate_levels(dg, outputs)


def check_pump_gate_levels(dg: DG645, outputs=ALL_OUTPUTS) -> None:
    """Raise unless every output in ``outputs`` is at <= 2.0 V amplitude and 0 V offset."""
    for out in outputs:
        idx = dg.OUTPUTS[out]
        amp, off = float(dg.ask(f"LAMP?{idx}")), float(dg.ask(f"LOFF?{idx}"))
        if amp > MAX_SWITCH_DRIVE_V + 1e-6 or abs(off) > 1e-6:
            raise RuntimeError(f"DG645 {out}: amplitude {amp} V / offset {off} V -- switch drive must be <= "
                               f"{MAX_SWITCH_DRIVE_V} V, 0 V offset")


class DG645PumpWidth:
    """Callable settable for meas_ctrl: pump-gate length on the DG645 (seconds).

    Setting it moves the second channel of each output in ``outputs`` (B, D, F, H) to ``width`` after the first, so all
    outputs -- and the rear AB+CD+EF+GH OR port -- carry one identical pulse. Width 0 gives no pulse. Levels are checked before
    every set (<= 2.0 V, 0 V offset). Use as ``meas_ctrl.settables(pump_width.parameter)``.
    """

    def __init__(self, dg: DG645, outputs=ALL_OUTPUTS, name: str = "dg645_pump_width"):
        from qcodes.parameters import Parameter
        from qcodes.validators import Numbers as _Num
        self.dg, self.outputs = dg, tuple(outputs)
        self.parameter = Parameter(name, label="DG645 pump length", unit="s", get_cmd=self._get, set_cmd=self._set,
                                   vals=_Num(0, 200e-6))

    def _get(self) -> float:
        out = self.outputs[0]
        return self.dg._get_delay(out[1])[1]

    def _set(self, width: float) -> None:
        check_pump_gate_levels(self.dg, self.outputs)
        for out in self.outputs:
            self.dg.set_delay(out[1], out[0], float(width))
        errs = self.dg.errors()
        if errs:
            raise RuntimeError(f"DG645 errors after setting pump width {width}: {errs}")


def configure_pump_train(dg: DG645, n_pulses: int = 4, width: float = 500e-9, gap: float = 16e-9, start: float = 0.0,
                         trigger_level: float = 1.0) -> float:
    """Back-to-back pump train on the rear AB+CD+EF+GH OR port: pulse k (k = 0..n_pulses-1) on output ALL_OUTPUTS[k] at
    ``start + k (width + gap)`` .. ``+ width`` after T0. Outputs beyond ``n_pulses`` duplicate pulse 0 exactly, so the OR port
    carries n_pulses pulses and nothing else. 2.0 V / 0 V offset / positive, external rising trigger, levels checked.
    Returns the train length (first rising to last falling edge)."""
    n_pulses = int(n_pulses)
    if not 1 <= n_pulses <= len(ALL_OUTPUTS):
        raise ValueError(f"n_pulses must be 1..{len(ALL_OUTPUTS)}, got {n_pulses}")
    dg.burst_mode("OFF") if "burst_mode" in dg.parameters else None
    for k, out in enumerate(ALL_OUTPUTS):
        idx = dg.OUTPUTS[out]
        dg.write(f"LOFF {idx},0")
        dg.write(f"LAMP {idx},{MAX_SWITCH_DRIVE_V}")
        dg.write(f"LPOL {idx},1")
        j = k if k < n_pulses else 0
        dg.set_delay(out[0], "T0", start + j * (width + gap))
        dg.set_delay(out[1], out[0], width)
        dg.ask("*OPC?")          # handshake per output: back-to-back writes overran the input buffer (errors 171, 110)
    dg.write(f"TLVL {trigger_level}")
    dg.write("TSRC 1")
    check_pump_gate_levels(dg)
    errs = dg.errors()
    if errs:
        raise RuntimeError(f"DG645 errors after configuring the pump train: {errs}")
    return n_pulses * width + (n_pulses - 1) * gap


class DG645PumpTrain:
    """meas_ctrl settables for a back-to-back pump train (see ``configure_pump_train``): ``.n`` (number of pulses 1..4),
    ``.gap`` (s) and ``.width`` (s) each reprogram the whole train; ``.length`` (get only) is the current train length, to be
    passed as ``pump_duration`` to the lab code's DG645-gated SNAIL swap schedule (``TWPA_schedule.SNAIL_swap_dg645_sched_TWPA``,
    not part of qqea). ``start`` (s, fixed) delays the whole train after
    the trigger: use it to pad the trigger-to-chip latency up to the 4 ns Qblox grid."""

    def __init__(self, dg: DG645, n: int = 4, width: float = 500e-9, gap: float = 16e-9, name: str = "dg645_train", start: float = 0.0):
        from qcodes.parameters import Parameter
        from qcodes.validators import Numbers as _Num, Ints as _Ints
        self.dg, self._n, self._width, self._gap, self._start = dg, int(n), float(width), float(gap), float(start)
        self.n = Parameter(name + "_n", label="pump pulses", unit="", get_cmd=lambda: self._n, set_cmd=self._set_n, vals=_Ints(1, 4))
        self.gap = Parameter(name + "_gap", label="pump gap", unit="s", get_cmd=lambda: self._gap, set_cmd=self._set_gap, vals=_Num(0, 2e-6))
        self.width = Parameter(name + "_width", label="pump pulse width", unit="s", get_cmd=lambda: self._width, set_cmd=self._set_width,
                               vals=_Num(0, 2e-6))
        self.length = Parameter(name + "_length", label="pump train length", unit="s", get_cmd=self._length, set_cmd=False)
        self.apply()

    def _length(self) -> float:
        return self._n * self._width + (self._n - 1) * self._gap

    def apply(self) -> float:
        return configure_pump_train(self.dg, self._n, self._width, self._gap, start=self._start)

    def _set_n(self, n):
        self._n = int(n); self.apply()

    def _set_gap(self, gap):
        self._gap = float(gap); self.apply()

    def _set_width(self, width):
        self._width = float(width); self.apply()
