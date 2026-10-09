# qblox-quantify-experiment-analysis

Python modules for running two-qubit transmon experiments on a Qblox cluster with quantify and analysing the results. They cover schedules, calibration nodes, fitting helpers, randomized benchmarking, state and process tomography, and gate set tomography.

## Packages

One package, `qqea` (short for qblox-quantify-experiment-analysis), with four subpackages split by role:

```
src/qqea/
    schedules/                    building experiment schedules
        single_qubit.py           spectroscopy, Rabi, Ramsey, T1/T2, readout, DRAG, f-state, SNAIL/pump
        two_qubit.py              re-exports the two-qubit modules below, plus RBAnalysis
        gates.py                  iSWAP gate, Clifford index to operations, iSWAP timing settings
        phase.py                  iSWAP virtual-Z phase corrections
        rb.py                     randomized benchmarking and simultaneous RB
        iswap.py                  iSWAP characterisation: pulsed pump, pump-probe, phase tracking, RPE
        tomography.py             state and process tomography schedules
        gst.py                    gate set tomography schedules from pyGSTi circuits
        twpa.py                   TWPA pump marker for readout, per-qubit on/off switch
        edges.py                  CompositeiSWAPEdge, the iSWAP edge for the quantum device
        clifford/                 Clifford group and RB sequences, adapted from PycQED (MIT)
    experiments/                  running experiments
        calibration_nodes.py      CalibrationNodes: resonator/qubit/gate calibration routines
        nodes/                    the node methods by topic: single_qubit, readout, iswap_stark, roadmap
        runners.py                DRAG and readout-optimization runs that drive the cluster directly
        saving.py                 save raw retrieve_acquisition() data as a quantify dataset
        datadir.py                set_datadir_from_env(): data directory from QBLOX_DATADIR
        hp83732b.py               qcodes driver for the HP 83732B signal generator (TWPA pump)
        simulated_data.py         simulated datasets for dry runs
        dry_run.py                run the nodes on a dummy cluster (qblox_dev only)
    fitting/                      analysis: fitting and plotting
        readout.py                g/e discrimination, readout freq/amp/duration optimization, weights
        single_qubit.py           Ramsey chevron, conditional Ramsey, Rabi amplification, T1/T2, DRAG
        two_qubit.py              iSWAP chevron, theta_p, RPE
        rb.py                     rb_simple_fit and RBAnalysis
        models.py                 extra lmfit models (beating decay, multi-Lorentzian, iSWAP exchange)
        iswap_exchange.py         iSWAP population exchange in T1, Tphi, t_iSWAP; iSWAP gate-error budget
        plotting.py, units.py, dataset_info.py, saving.py
        fits.py                   the old helpers namespace, re-exporting all of the above
    tomography/                   analysis: tomography
        state.py                  two-qubit density matrices: linear inversion, MLE
        process.py                process tomography: linear inversion, CPTP MLE, SPAM correction, fidelities
        iswap.py                  5-parameter iSWAP model and angle extraction
        gst.py                    Quantify GST datasets to pyGSTi
        plotting.py               3D chi-matrix plots
        tomography.py             the old flat tomography_tools namespace
```

Imports from `qqea.schedules.two_qubit` keep working; it re-exports every two-qubit builder.

Schedule builders no longer end in `_TWPA` (`rabi_sched_TWPA` is now `rabi_sched`), and there are no aliases under the old names. Some names now match builders in `quantify_scheduler.schedules`, so prefer `from qqea.schedules import single_qubit as sq` over star imports.

Whether a qubit's readout triggers the TWPA pump is a per-qubit setting. Every qubit pumps unless told otherwise:

```python
from qqea.schedules.twpa import set_twpa_pump
set_twpa_pump(qubit2, False)
```

The iSWAP edge now ships in the package: `from qqea.schedules.edges import CompositeiSWAPEdge`.

Old imports map to new ones as follows:

| Old import | New import |
| --- | --- |
| `TWPA_schedule` | `qqea.schedules.single_qubit` |
| `pycqed_randomized_benchmarking.utils` | `qqea.schedules.two_qubit` |
| `pycqed_randomized_benchmarking.clifford_group` | `qqea.schedules.clifford.clifford_group` |
| `calibration_nodes` | `qqea.experiments.calibration_nodes` |
| `HP83732B` | `qqea.experiments.hp83732b` |
| `simulated_data` | `qqea.experiments.simulated_data` |
| `helpers` | `qqea.fitting.fits` |
| `analyzer` | `qqea.fitting.models` |
| `tomography_tools` | `qqea.tomography` (or `qqea.tomography.tomography` for `import *` code) |

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

The dependency versions in `pyproject.toml` are pinned to the ones in `qblox_env` (quantify-scheduler 0.23.1, quantify-core 0.9.1, qblox-instruments 0.16.0, qcodes 0.46.0, pygsti 0.9.14.3, qutip 5.0.2, numpy 1.26.4).

## License

MIT, see [LICENSE](LICENSE). `qqea/schedules/clifford/` is adapted from [PycQED](https://github.com/DiCarloLab-Delft/PycQED_py3) and keeps its own MIT notice in `src/qqea/schedules/clifford/LICENSE.txt`.

## Usage

```python
from qqea.schedules.single_qubit import rabi_sched
from qqea.schedules.two_qubit import randomized_benchmarking_schedule
from qqea.experiments.calibration_nodes import CalibrationNodes
from qqea.fitting.fits import t1_and_t2_fit
from qqea.tomography import calculate_density_matrix, mle_chi_from_p00
```

## Tests

```
pip install pytest
pytest
```

Run them in `qblox_dev` after `pip install -e . --no-deps` (they need qutip, scipy, xarray, and pyGSTi/quantify-core for the reference copy of the old modules).

The old `Python-Packages` folder and the notebooks that put it on `sys.path` are not affected.

## Tests

In `qblox_dev` (never `qblox_env`):

```
pip install pytest
python -m pytest tests
```
