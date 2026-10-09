"""The old flat ``tomography_tools`` namespace, for code written against it.

``from qqea.tomography.tomography import *`` gives the same names ``from tomography_tools import *``
did, including the third-party names that module happened to export (np, plt, qt, pygsti,
load_dataset, least_squares, ...), so notebooks that relied on them keep working. Unlike
``qqea.tomography`` this imports matplotlib and pyGSTi straight away. New code should import from
``qqea.tomography`` and its modules instead.
"""
# ruff: noqa: F401
import itertools
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize
from scipy.linalg import fractional_matrix_power, sqrtm, solve
from scipy.optimize import least_squares, minimize, minimize_scalar
import xarray as xr
import qutip as qt
from qutip import matrix_histogram
import pygsti
from quantify_core.data.handling import load_dataset

from ._paulis import *  # noqa: F401,F403
from ._paulis import I, sigma_x, sigma_y, sigma_z  # noqa: F401
from .state import *  # noqa: F401,F403
from .state import (  # noqa: F401
    STOKES_PLOT_ORDER, _project_eigs_to_simplex,
)
from .process import *  # noqa: F401,F403
from .process import (  # noqa: F401
    PMATRIX_PATH, PMatrix, op_label, _CHOI_DIMS, _Y_PARITY, _tomo_rotations, _mle_forward_matrix,
    _linear_init, _choi_from_t, _t_from_choi, _spam_forward_matrix,
)
from .iswap import *  # noqa: F401,F403
from .iswap import _wrap  # noqa: F401
from .gst import *  # noqa: F401,F403
from .gst import _gst_ds_counts  # noqa: F401
from .plotting import *  # noqa: F401,F403
from .plotting import titlefont, axisfont, ticksfont  # noqa: F401
