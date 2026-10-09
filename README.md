# qblox-quantify-experiment-analysis

Experiment schedules, calibration nodes and analysis tools for two-qubit transmon experiments on a Qblox cluster with quantify (device SNL315). It covers randomized benchmarking, state and process tomography, gate set tomography, and iSWAP incoherent-error studies.

## Layout

```
src/snl_qblox/                    installable package
    TWPA_schedule.py              quantify-scheduler schedules (with TWPA pump handling)
    calibration_nodes.py          CalibrationNodes: resonator/qubit/gate calibration routines
    helpers.py                    fitting and plotting helpers for quantify datasets
    analyzer.py                   extra lmfit models (beating decay, multi-Lorentzian, swap decay)
    tomography_tools.py           state/process tomography, MLE, pyGSTi helpers
    simulated_data.py             simulated datasets for dry runs
    HP83732B.py                   qcodes driver for the HP 83732B signal generator
    pycqed_randomized_benchmarking/
                                  Clifford groups and RB, adapted from PycQED (MIT), extended
                                  for tomography, GST and iSWAP phase calibration
SNL315/
    SNL315.ipynb                  main measurement notebook
    SNL315_plot.ipynb             analysis and paper figures
    bell_state/                   Bell-state tomography analysis
    process_tomography/           iSWAP process tomography analysis
    gst/                          standalone GST analysis and reports
    incoherent_error/             iSWAP incoherent error (T1/T2) simulation and fits
```

## Install

Into the existing conda environment, without changing any installed packages:

```
conda activate qblox_env
pip install -e . --no-deps
```

On a fresh environment, `pip install -e .` pulls in the dependencies listed in `pyproject.toml`.

Then import from the package, for example:

```python
from snl_qblox.helpers import *
from snl_qblox.calibration_nodes import CalibrationNodes
from snl_qblox.pycqed_randomized_benchmarking.utils import randomized_benchmarking_schedule
```

## Data

Raw data is not stored in this repository. Notebooks and scripts read the quantify data directory from the `QBLOX_DATADIR` environment variable, and fall back to `SNL315/CD1`. The Bell-state CSVs are expected in `SNL315/bell_state/`.

`SNL315/process_tomography/run_qpt_series.py` imports `qpt_series`, which is not included yet.
