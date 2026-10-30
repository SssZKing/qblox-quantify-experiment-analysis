"""QCoDeS driver for the HP (Agilent) 83731B/83732B synthesized signal generator.

Commands follow the "HP 83731A/32A and HP 83731B/32B Synthesized Signal Generators Programming Guide" (SCPI). Notes from it that
matter for using the generator as a pulsed pump:

* ALC: internal levelling is always active; ``alc_source`` only chooses the detector (internal, external diode or external power
  meter). There is no ALC on/off or bandwidth command on this instrument (``POW:ALC[:STAT]`` / ``:BWID`` give -113).
* Pulse modulation with internal levelling is NOT levelled pulse by pulse: on every power or frequency change (and the first time
  pulse modulation is enabled) the output briefly switches to CW (~30 ms) to sample the level, and the pulses are then played with the
  stored "vernier" level DAC (``level_vernier_dac`` = ``DIAG:IBUS? 23``). The CW-to-pulse level accuracy drifts by about
  0.07 dB/C with the instrument temperature; the manual recommends recalibrating (re-sending the power) when the ambient changes.
* ``average_power_protection`` (``POW:PROT:STAT``, needs option 1E1) switches in 110 dB during that CW calibration to protect
  average-power-sensitive loads; it wears the step attenuator and adds 70 ms to every power/frequency change.
* ``questionable_condition`` (``STAT:QUES:COND?``) is live: bit 3 output power unlevelled/uncalibrated, bit 4 reference oven cold
  (option 1E5), bit 5 frequency unlocked, bit 7 modulation uncalibrated, bit 8 any of 3/5/7. The ``power_unleveled`` etc.
  parameters decode it.
* The instrument has no temperature readback.
* Toggling ``PULM:STAT`` alone does NOT re-run the level calibration. Seen 2026-10-28: after ``PULM:STAT OFF`` the CW output
  was effectively off (TWPA unpumped, readout background -33 dB, same as RF off), and switching back ON stored an invalid level
  (vernier 16711 instead of ~3700, error +610 "Track and hold failed, level is invalid", QUES still 0). Re-sending FREQ/POW fixed
  both at once. ``pulsemod_state`` therefore re-sends the power after every change (``relevel()``).

The undocumented-for-users ``DIAG:IBUS <n>,<value>`` write forms (special pulse modulation mode, writing the vernier) are deliberately
not exposed.
"""
from typing import TYPE_CHECKING

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.validators import Enum, Numbers

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

_ON_OFF = {"OFF": 0, "ON": 1}


def _int(s: str) -> int:
    return int(float(s))


class HP83732B(VisaInstrument):
    """
    QCoDeS driver for the HP 83731B/83732B synthesized signal generator (HP is now Agilent/Keysight).
    In the SNL315 setup it is the TWPA pump, with its output gated by the cluster's marker through
    external pulse modulation.

    ``status`` switches the RF output; ``rf_output`` is the same parameter under the name some
    notebook cells use, and ``on()``/``off()`` are shortcuts for it.
    """

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name, address, **kwargs)

        # ---- carrier ----
        self.frequency: Parameter = self.add_parameter(
            name="frequency", label="Frequency", unit="Hz",
            get_cmd="FREQ?", set_cmd="FREQ {}", get_parser=float,
            vals=Numbers(min_value=10e6, max_value=20e9),
        )
        """CW frequency (FREQ[:CW])."""
        self.power: Parameter = self.add_parameter(
            name="power", label="Power", unit="dBm",
            get_cmd="POW?", set_cmd="POW {}", get_parser=float,
            vals=Numbers(min_value=-110, max_value=30),
        )
        """Output level (POW[:LEV]). In pulse mode every change triggers the ~30 ms CW level calibration."""
        self.status: Parameter = self.add_parameter(
            name="status", label="Status",
            get_cmd="OUTP:STAT?", set_cmd="OUTP:STAT {}", val_mapping=_ON_OFF,
        )
        """RF output on/off (OUTP[:STAT])."""
        self.rf_output: Parameter = self.add_parameter(
            name="rf_output", label="RF output",
            get_cmd="OUTP:STAT?", set_cmd="OUTP:STAT {}", val_mapping=_ON_OFF,
            snapshot_exclude=True,  # same setting as ``status``
        )
        """Alias of ``status``."""

        # ---- pulse modulation ----
        self.pulsemod_state: Parameter = self.add_parameter(
            name="pulsemod_state", label="PulseMod State",
            get_cmd="PULM:STAT?", set_cmd=self._set_pulsemod_state, val_mapping=_ON_OFF,
        )
        """Pulse modulation on/off (PULM:STAT), followed by ``relevel()`` so the track-and-hold gets a valid level."""
        self.pulsemod_source: Parameter = self.add_parameter(
            name="pulsemod_source", label="PulseMod Source",
            get_cmd="PULM:SOUR?", set_cmd="PULM:SOUR {}", vals=Enum("INT", "EXT"),
        )
        """Pulse modulation source, INT or EXT (PULM:SOUR)."""
        self.pulsemod_ext_polarity: Parameter = self.add_parameter(
            name="pulsemod_ext_polarity", label="PulseMod External Polarity",
            get_cmd="PULM:EXT:POL?", set_cmd="PULM:EXT:POL {}", vals=Enum("NORM", "INV"),
        )
        """External pulse input polarity, NORM or INV (PULM:EXT:POL)."""

        # ---- automatic level control ----
        self.alc_source: Parameter = self.add_parameter(
            name="alc_source", label="ALC Source",
            get_cmd="POW:ALC:SOUR?", set_cmd="POW:ALC:SOUR {}", vals=Enum("INT", "DIOD", "PMET"),
        )
        """Levelling detector: INT (internal), DIOD (external diode), PMET (external power meter) (POW:ALC:SOUR)."""
        self.alc_pmeter_level: Parameter = self.add_parameter(
            name="alc_pmeter_level", label="ALC Power-Meter Reading", unit="dBm",
            get_cmd="POW:ALC:PMET?", set_cmd="POW:ALC:PMET {}", get_parser=float,
            vals=Numbers(min_value=-120, max_value=30),
        )
        """Initial external power-meter reading for power-meter levelling (POW:ALC:PMET)."""
        self.level_vernier_dac: Parameter = self.add_parameter(
            name="level_vernier_dac", label="Level Vernier DAC",
            get_cmd="DIAG:IBUS? 23", set_cmd=False, get_parser=_int,
        )
        """Read-only: the level vernier DAC the output (and pulses) are played with (DIAG:IBUS? 23, from the manual's pulsed-power
        pre-calibration program)."""

        # ---- RF output protection / attenuator ----
        self.average_power_protection: Parameter = self.add_parameter(
            name="average_power_protection", label="Average Power Inhibit",
            get_cmd="POW:PROT:STAT?", set_cmd="POW:PROT:STAT {}", val_mapping=_ON_OFF,
        )
        """Average-power inhibit during the pulse-mode CW calibration (POW:PROT:STAT, option 1E1). Wears the step attenuator."""
        self.attenuation_auto: Parameter = self.add_parameter(
            name="attenuation_auto", label="Attenuator Auto Coupling",
            get_cmd="POW:ATT:AUTO?", set_cmd="POW:ATT:AUTO {}", val_mapping=_ON_OFF,
        )
        """Step attenuator coupled to the power level (POW:ATT:AUTO)."""
        self.output_protection: Parameter = self.add_parameter(
            name="output_protection", label="Output Protection",
            get_cmd="OUTP:PROT?", set_cmd="OUTP:PROT {}", val_mapping=_ON_OFF,
        )
        """RF output (reverse power) protection (OUTP:PROT[:STAT])."""
        self.output_impedance: Parameter = self.add_parameter(
            name="output_impedance", label="Output Impedance", unit="Ohm",
            get_cmd="OUTP:IMP?", set_cmd=False, get_parser=float,
        )
        """Read-only output impedance (OUTP:IMP?)."""

        # ---- other modulations (read back so a stray AM/FM/PM is noticed) ----
        for mod in ("AM", "FM", "PM"):
            self.add_parameter(
                name=f"{mod.lower()}_state", label=f"{mod} State",
                get_cmd=f"{mod}:STAT?", set_cmd=f"{mod}:STAT {{}}", val_mapping=_ON_OFF,
            )

        # ---- instrument information / status ----
        self.reference_source: Parameter = self.add_parameter(
            name="reference_source", label="Reference Source",
            get_cmd="ROSC:SOUR?", set_cmd=False, get_parser=str.strip,
        )
        """Read-only frequency reference in use: INT, EXT or NONE (ROSC:SOUR?; selected automatically by the rear input)."""
        self.options: Parameter = self.add_parameter(
            name="options", label="Options", get_cmd="*OPT?", set_cmd=False, get_parser=str.strip,
        )
        """Installed options (*OPT?)."""
        self.scpi_version: Parameter = self.add_parameter(
            name="scpi_version", label="SCPI Version", get_cmd="SYST:VERS?", set_cmd=False, get_parser=str.strip,
        )
        """SCPI version (SYST:VERS?)."""
        self.questionable_condition: Parameter = self.add_parameter(
            name="questionable_condition", label="Questionable Condition",
            get_cmd="STAT:QUES:COND?", set_cmd=False, get_parser=_int,
        )
        """Live questionable-condition register (STAT:QUES:COND?): bit 3 power unlevelled, 4 oven cold, 5 frequency unlocked,
        7 modulation uncalibrated, 8 any of 3/5/7."""
        self.operation_condition: Parameter = self.add_parameter(
            name="operation_condition", label="Operation Condition",
            get_cmd="STAT:OPER:COND?", set_cmd=False, get_parser=_int,
        )
        """Live operation-condition register (STAT:OPER:COND?): bit 0 YIG calibration running, 1 hardware settling,
        7 level-correct routine running."""
        for pname, bit, label in (("power_unleveled", 3, "Power Unleveled"), ("reference_oven_cold", 4, "Reference Oven Cold"),
                                  ("frequency_unlocked", 5, "Frequency Unlocked"),
                                  ("modulation_uncalibrated", 7, "Modulation Uncalibrated")):
            self.add_parameter(
                name=pname, label=label, set_cmd=False,
                get_cmd=lambda b=bit: bool(self.questionable_condition() >> b & 1),
            )

        self.add_function("reset", call_cmd="*RST")
        self.connect_message()

    def on(self) -> None:
        """Turn the RF output on."""
        self.status("ON")

    def off(self) -> None:
        """Turn the RF output off."""
        self.status("OFF")

    def _set_pulsemod_state(self, value: int) -> None:
        self.write(f"PULM:STAT {value}")
        self.relevel()

    def relevel(self) -> None:
        """Re-send the current power, which makes the instrument re-run its level calibration (~30 ms CW in pulse mode)."""
        self.write(f"POW {float(self.ask('POW?'))}")

    def get_errors(self, max_n: int = 20) -> list:
        """Read the error queue until it reports 0 (SYST:ERR?)."""
        out = []
        for _ in range(max_n):
            e = self.ask("SYST:ERR?").strip()
            out.append(e)
            if e.lstrip("+").startswith("0"):
                break
        return out

    def health(self) -> dict:
        """One read-only snapshot of the levelling-relevant state (for monitoring a pulsed pump)."""
        q = self.questionable_condition()
        return {"frequency": self.frequency(), "power": self.power(), "status": self.status(), "pulsemod": self.pulsemod_state(),
                "alc_source": self.alc_source(), "vernier_dac": self.level_vernier_dac(), "questionable": q,
                "power_unleveled": bool(q >> 3 & 1), "oven_cold": bool(q >> 4 & 1), "frequency_unlocked": bool(q >> 5 & 1),
                "operation": self.operation_condition(), "reference": self.reference_source()}
