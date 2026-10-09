"""Fit the 2026-09-30 iSWAP QPT series and the idle (pump off) series after it, as far as they have
been saved (only fits not cached yet).

    python run_qpt_series.py
"""
import os
import time
import warnings

warnings.filterwarnings("ignore")
from quantify_core.data.handling import set_datadir

# quantify data directory; set QBLOX_DATADIR to override (default: SNL315/CD1)
set_datadir(os.environ.get(
    "QBLOX_DATADIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CD1")))
import qpt_series as qser  # noqa: E402

START, STOP, NAME = "20260930-191734", "20260930-2250", "S0930"   # first readout calibration; qubits recalibrated from 22:50
IDLE_START, IDLE_NAME = "20260930-230856", "S0930idle"   # readout calibration before the first idle run

if __name__ == "__main__":
    t0 = time.time()
    for s in (qser.Series(START, NAME, stop=STOP), qser.Series(IDLE_START, IDLE_NAME, gate="idle")):
        print(f"{s.name}: {len(s)} QPT runs in {len(s.blocks)} blocks, {len(s.t1t2)} T1/T2 runs", flush=True)
        s.fit()
    print(f"done in {time.time() - t0:.0f} s", flush=True)
