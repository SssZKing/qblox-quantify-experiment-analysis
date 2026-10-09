from typing import TYPE_CHECKING

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.parameters import create_on_off_val_mapping
from qcodes.validators import Enum, Numbers

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

class HP83732B(VisaInstrument):
    """
    QCoDeS driver for the HP 83732B synthesized signal generator (HP is now
    Keysight, formerly Agilent). In the SNL315 setup it is the TWPA pump, with
    its output gated by the cluster's marker through external pulse modulation.

    ``status`` switches the RF output; ``rf_output`` is the same parameter
    under the name some notebook cells use, and ``on()``/``off()`` are
    shortcuts for it.
    """

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name, address, **kwargs)
        # general commands
        self.frequency: Parameter = self.add_parameter(
            name="frequency",
            label="Frequency",
            unit="Hz",
            get_cmd="FREQ?",
            set_cmd="FREQ {}",
            get_parser=float,
            vals=Numbers(min_value=10e6, max_value=20e9),
        )
        """Parameter frequency"""
        self.power: Parameter = self.add_parameter(
            name="power",
            label="Power",
            unit="dBm",
            get_cmd="POW?",
            set_cmd="POW {}",
            get_parser=float,
            vals=Numbers(min_value=-110, max_value=30),
        )
        """Parameter power"""
        self.status: Parameter = self.add_parameter(
            name="status",
            label="Status",
            get_cmd="OUTP:STAT?",
            set_cmd="OUTP:STAT {}",
            val_mapping={"OFF": 0, "ON": 1},
        )
        """Parameter status"""
        self.rf_output: Parameter = self.add_parameter(
            name="rf_output",
            label="RF output",
            get_cmd="OUTP:STAT?",
            set_cmd="OUTP:STAT {}",
            val_mapping={"OFF": 0, "ON": 1},
            snapshot_exclude=True,  # same setting as ``status``
        )
        """Alias of ``status``"""
        self.pulsemod_state: Parameter = self.add_parameter(
            name="pulsemod_state",
            label="PulseMod State",
            get_cmd="PULM:STAT?",
            set_cmd="PULM:STAT {}",
            val_mapping={"OFF": 0, "ON": 1},
        )
        """Parameter pulsemod_state"""
        self.pulsemod_source: Parameter = self.add_parameter(
            name="pulsemod_source",
            label="PulseMod Source",
            get_cmd="PULM:SOUR?",
            set_cmd="PULM:SOUR {}",
            vals=Enum("INT", "EXT"),
        )
        """Parameter pulsemod_source"""
        self.add_function("reset", call_cmd="*RST")

        self.connect_message()

    def on(self) -> None:
        """Turn the RF output on."""
        self.status("ON")

    def off(self) -> None:
        """Turn the RF output off."""
        self.status("OFF")
