# qblox-quantify-experiment-analysis

Python modules for running two-qubit transmon experiments on a Qblox cluster with quantify and analysing the results. They cover schedules, calibration nodes, fitting helpers, randomized benchmarking, state and process tomography, and gate set tomography.

## Contents

```
src/
    TWPA_schedule.py              quantify-scheduler schedules (with TWPA pump handling)
    calibration_nodes.py          CalibrationNodes: resonator/qubit/gate calibration routines
    helpers.py                    fitting and plotting helpers for quantify datasets
    analyzer.py                   extra lmfit models (beating decay, multi-Lorentzian, swap decay)
    tomography_tools.py           state/process tomography, MLE, pyGSTi helpers (+ PMatrix2.pkl)
    simulated_data.py             simulated datasets for dry runs
    HP83732B.py                   qcodes driver for the HP 83732B signal generator
    pycqed_randomized_benchmarking/
                                  Clifford groups and RB, adapted from PycQED (MIT), extended
                                  for tomography, GST and iSWAP phase calibration
```

## Install

Recommended: create a separate env that copies the lab PC `qblox_env` exactly (Python 3.9.18, Windows) and installs this repo into it. `qblox_env` itself is not touched.

```
conda env create -f environment.yml
conda activate qblox_dev
```

Or, inside an existing env that already has the packages, install only these modules:

```
pip install -e . --no-deps
```

Use the editable install (`-e`), because `tomography_tools` loads `PMatrix2.pkl` from the file next to it.

The dependency versions in `pyproject.toml` are pinned to the ones in `qblox_env` (quantify-scheduler 0.23.1, quantify-core 0.9.1, qblox-instruments 0.16.0, qcodes 0.46.0, pygsti 0.9.14.3, qutip 5.0.2, numpy 1.26.4).

## Usage

The import names are the same as before, so notebook import lines don't change:

```python
from helpers import *
from TWPA_schedule import *
from calibration_nodes import CalibrationNodes
from tomography_tools import *
from pycqed_randomized_benchmarking.utils import RBAnalysis, randomized_benchmarking_schedule
import pycqed_randomized_benchmarking.utils as rbu
```

Once the package is installed, the `sys.path.insert(0, 'C:/Users/1238Tech/Documents/Python-Packages')` line is no longer needed. If you leave that line in, the old Python-Packages folder takes precedence over the installed copy.
