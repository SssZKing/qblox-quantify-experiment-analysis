"""Flat namespace of the old ``helpers`` module, kept for backward compatibility.

The code now lives in topic modules; import from those in new code:

- :mod:`qqea.fitting.readout`: g/e discrimination, readout optimization, IQ rotation
- :mod:`qqea.fitting.single_qubit`: Ramsey, Rabi amplification, T1/T2, DRAG
- :mod:`qqea.fitting.two_qubit`: iSWAP chevron, theta_p, RPE
- :mod:`qqea.fitting.rb`: randomized benchmarking
- :mod:`qqea.fitting.models`: lmfit models and shared model instances
- :mod:`qqea.fitting.plotting`, :mod:`qqea.fitting.units`, :mod:`qqea.fitting.dataset_info`

``from qqea.fitting.fits import *`` still provides every name ``helpers`` did
(including ``np``, ``plt``, ``curve_fit`` and the quantify analysis classes), except
the unused ``Cluster``/``ClusterType`` from qblox-instruments.
"""

# Third-party names the old module exposed through ``import *``; callers such as
# calibration_nodes and the SNL315 notebook rely on some of them.
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from matplotlib.ticker import FuncFormatter
from quantify_core.analysis.base_analysis import Basic2DAnalysis
from quantify_core.analysis.cosine_analysis import CosineAnalysis
from quantify_core.analysis.fitting_models import (
    DecayOscillationModel,
    ExpDecayModel,
    LorentzianModel,
    ResonatorModel,
)
from quantify_core.analysis.readout_calibration_analysis import ReadoutCalibrationAnalysis
from quantify_core.analysis.single_qubit_timedomain import (
    AllXYAnalysis,
    EchoAnalysis,
    RabiAnalysis,
    RamseyAnalysis,
    T1Analysis,
)
from quantify_core.analysis.spectroscopy_analysis import (
    QubitSpectroscopyAnalysis,
    ResonatorSpectroscopyAnalysis,
)
from quantify_core.data import handling as dh
from quantify_core.data.handling import (
    default_datadir,
    get_datadir,
    load_dataset,
    load_snapshot,
    set_datadir,
)
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter

from qqea.fitting.dataset_info import dataset_current, dataset_power, experiment_duration
from qqea.fitting.models import (
    decay_osci_model,
    exp_decay_model,
    lorentzian_model,
    resonator_model,
)
from qqea.fitting.plotting import BlochSpherePlot, bra_tex, cavity_map, create_ntwk, plot_IQ
from qqea.fitting.rb import rb_simple_fit
from qqea.fitting.readout import (
    IQ_blob_SNR,
    RO_amp_fit,
    RO_freq_fit,
    RO_len_fit,
    readout_weight_extractor,
    rotate_IQ_blob,
    rotate_real_imag,
    two_state_discriminator,
)
from qqea.fitting.saving import save_retrieve_acquisition_dataset
from qqea.fitting.single_qubit import (
    DRAG_fit,
    conditional_ramsey_fit,
    linear,
    rabi_amplification_fit,
    ramsey_chevron_fit,
    t1_and_t2_fit,
    v_line,
)
from qqea.fitting.two_qubit import (
    iswap_chevron_fit,
    iswap_theta_p_fit,
    rpe_quadrature_fit,
    rpe_rate_fit,
)
from qqea.fitting.units import P2Z, Z2P, W2dBm, Vp2dBm, dBm2Vp, dBm2W
