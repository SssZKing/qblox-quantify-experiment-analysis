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

##################################################
### Quantum State Tomography (QST) ###
##################################################

# Derive the density matrix
I = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Density matrix Pauli operator basis
II = np.kron(I, I)
IX = np.kron(I, sigma_x)
IY = np.kron(I, sigma_y)
IZ = np.kron(I, sigma_z)
XI = np.kron(sigma_x, I)
XX = np.kron(sigma_x, sigma_x)
XY = np.kron(sigma_x, sigma_y)
XZ = np.kron(sigma_x, sigma_z)
YI = np.kron(sigma_y, I)
YX = np.kron(sigma_y, sigma_x)
YY = np.kron(sigma_y, sigma_y)
YZ = np.kron(sigma_y, sigma_z)
ZI = np.kron(sigma_z, I)
ZX = np.kron(sigma_z, sigma_x)
ZY = np.kron(sigma_z, sigma_y)
ZZ = np.kron(sigma_z, sigma_z)

def calculate_density_matrix(dataset, confusion_matrix=None):
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    q1_state_append = dataset[list(dataset.data_vars)[0]].values
    q2_state_append = dataset[list(dataset.data_vars)[1]].values
    
    p11 = q1_state_append * q2_state_append
    p10 = q1_state_append * (1-q2_state_append)
    p01 = (1-q1_state_append) * q2_state_append
    p00 = (1-q1_state_append) * (1-q2_state_append)
    
    p11 = p11.mean(axis=0)
    p10 = p10.mean(axis=0)
    p01 = p01.mean(axis=0)
    p00 = p00.mean(axis=0)

    p11 = np.insert(p11, 6, p11[3])
    p11 = np.append(p11, p11[7])
    p11 = np.append(p11, p11[2])
    p11 = np.append(p11, p11[:3])

    p10 = np.insert(p10, 6, p10[3])
    p10 = np.append(p10, p10[7])
    p10 = np.append(p10, p10[2])
    p10 = np.append(p10, p10[:3])

    p01 = np.insert(p01, 6, p01[3])
    p01 = np.append(p01, p01[7])
    p01 = np.append(p01, p01[2])
    p01 = np.append(p01, p01[:3])

    p00 = np.insert(p00, 6, p00[3])
    p00 = np.append(p00, p00[7])
    p00 = np.append(p00, p00[2])
    p00 = np.append(p00, p00[:3])

    if confusion_matrix is not None:
        p00, p01, p10, p11 = np.linalg.inv(confusion_matrix) @ [p00, p01, p10, p11]
        
    stokes = np.zeros(15)
    
    for i in range(1,16):
        if i in [1,2,3]:
            stokes[i-1] = p00[i-1] + p10[i-1] - p01[i-1] - p11[i-1]
        elif i in [4, 8, 12]:
            stokes[i-1] = p00[i-1] + p01[i-1] - p10[i-1] - p11[i-1]
        elif i in [5, 6, 7, 9, 10, 11, 13, 14, 15]:
            stokes[i-1] = p00[i-1] + p11[i-1] - p10[i-1] - p01[i-1]
    
    rho = 0.25 * (
        II
        + stokes[0] * IX
        + stokes[1] * IY
        + stokes[2] * IZ
        + stokes[3] * XI
        + stokes[4] * XX
        + stokes[5] * XY
        + stokes[6] * XZ
        + stokes[7] * YI
        + stokes[8] * YX
        + stokes[9] * YY
        + stokes[10] * YZ
        + stokes[11] * ZI
        + stokes[12] * ZX
        + stokes[13] * ZY
        + stokes[14] * ZZ
    )

    return rho, stokes

def extract_population_phase(single_rho, label, coherence_eps=1e-10):
    p0 = float(np.real(single_rho[0, 0]))
    p1 = float(np.real(single_rho[1, 1]))

    coherence = single_rho[0, 1]
    coherence_mag = float(np.abs(coherence))
    phase_rad = float(np.angle(coherence)) if coherence_mag > coherence_eps else float("nan")
    phase_deg = float(np.degrees(phase_rad)) if coherence_mag > coherence_eps else float("nan")

    print(f"{label} population: P(|0>)={p0:.4f}, P(|1>)={p1:.4f}")
    print(f"{label} phase from rho[0,1]: {phase_rad:.4f} rad ({phase_deg:.2f} deg), |rho[0,1]|={coherence_mag:.4f}")
    print("-" * 70)


def compute_stokes(rho):
    # Calculate the 15 expected Stokes parameters from the current density matrix
    # Assumes IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ are defined 2-qubit Pauli matrices
    paulis = [
        IX, IY, IZ, 
        XI, XX, XY, XZ, 
        YI, YX, YY, YZ, 
        ZI, ZX, ZY, ZZ
    ]
    return np.array([np.real(np.trace(np.dot(rho, P))) for P in paulis])

# Reorder a 15-element Stokes vector from the standard Pauli order
# (IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ)
# into the grouped bar-plot order: single-Q2 (I*), single-Q1 (*I), then the two-qubit terms.
STOKES_PLOT_ORDER = [2, 0, 1, 11, 3, 7, 14, 6, 10, 12, 4, 8, 13, 5, 9]

def rearrange_stokes(stokes):
    return np.asarray(stokes)[STOKES_PLOT_ORDER]

def mle_residuals(t, measured_stokes):
    # Reconstruct density matrix from parameters t
    T = t2T(t)
    rho = T2rho(T)
    
    # Calculate expected stokes
    expected_stokes = compute_stokes(rho)
    
    # Return the difference as residuals
    return expected_stokes - measured_stokes
# Helper functions for Maximum Likelihood Estimation (MLE) of quantum density matrices.
# These functions parameterize a physical density matrix using the Cholesky decomposition: rho = T * T^dagger
# where T is a complex lower triangular matrix that can be unrolled into a flat sequence 't' for optimizers.

def t2T(t):
    # Reconstructs the lower triangular matrix T from a 1D parameter array t.
    d = int(np.sqrt(len(t)))
    idx = 0
    cur_length = d
    T = np.zeros([d, d])
    for j in range(int(d)):
        # Construct the real parts of the j-th subdiagonal
        T = T + 1 * np.diag(t[np.arange(idx, idx + cur_length)], -j)
        idx = idx + cur_length
        if j > 0:
            # Construct the imaginary parts of the j-th subdiagonal (for off-diagonals only)
            T = T + 1j * np.diag(t[np.arange(idx, idx + cur_length)], -j)
            idx = idx + cur_length
        cur_length -= 1
    return T

def T2rho(T, normalize=True):
    # Reconstructs the density matrix from the Cholesky decomposition matrix T.
    rho = np.dot(T, T.conj().T)  # T * T^dagger
    if normalize:
        rho = rho / np.trace(rho)  # Normalize to ensure Tr(rho) = 1
    return rho

def rho2T(rho):
    # Extracts the Cholesky decomposition matrix T from a given density matrix rho.
    def _make_positive(rho_in):
        # Projects rho to the nearest physical (positive semi-definite) state if non-physical
        d, v = np.linalg.eig(rho_in)
        rho = np.zeros(rho_in.shape)
        for j in range(len(d)):
            # Force all eigenvalues to be positive
            rho = rho + np.abs(d[j]) * np.outer(v[:, j], v[:, j].conj().transpose())
        rho = (rho + rho.conj().transpose()) / 2.0 # Ensure strictly Hermitian
        return rho

    def _psd_cholesky(rho, eps=1e-10):
        # Add a tiny identity matrix to ensure numerical stability during Cholesky
        eps_I = np.eye(rho.shape[0]) * eps
        return np.linalg.cholesky(eps_I + rho)

    # Check if there are any negative eigenvalues indicating a non-physical density matrix
    if not np.all(np.linalg.eigvals(rho) > 0):
        new_rho = _make_positive(rho)
        return _psd_cholesky(new_rho)

    return np.linalg.cholesky(rho)

def T2t(T):
    # Flattens the lower triangular Cholesky matrix T into a real 1D array t
    d = len(T)
    idx = 0
    cur_length = d
    t = np.zeros(d**2)
    for j in range(d):
        # Extract the real parts of the subdiagonal
        t[np.arange(idx, idx + cur_length)] = np.real(np.diag(T, -j))
        idx = idx + cur_length
        if j > 0:
            # Extract the imaginary parts of the subdiagonal
            t[np.arange(idx, idx + cur_length)] = np.imag(np.diag(T, -j))
            idx = idx + cur_length
        cur_length -= 1
    return t
    
##################################################
### Quantum Process Tomography (QPT) ###
##################################################

PMATRIX_PATH = Path(__file__).with_name("PMatrix2.pkl")  # shipped with the package

with open(PMATRIX_PATH, "rb") as file:
    PMatrix = pickle.load(file)

def func_c2(i: int, j: int, k: int):
    # Constants c2[i, j, k] such that
    # func_E2[i] = sum_{j,k} func_c2[i,j,k] (func_F2[j] x func_F2[k]) |0>|0><0|<0| (func_F2[j]^{\dagger} x func_F2[k]^{\dagger})
    if (i, j, k) == (0, 0, 0):
        return 1
    elif (i, j, k) == (0, 0, 1):
        return 1
    elif (i, j, k) == (0, 1, 0):
        return 1
    elif (i, j, k) == (0, 1, 1):
        return 1
    elif (i, j, k) == (1, 0, 0):
        return -(1 + 1j)
    elif (i, j, k) == (1, 0, 1):
        return -(1 + 1j)
    elif (i, j, k) == (1, 0, 2):
        return 2
    elif (i, j, k) == (1, 0, 4):
        return 1j
    elif (i, j, k) == (1, 0, 5):
        return 1j
    elif (i, j, k) == (1, 1, 0):
        return -(1 + 1j)
    elif (i, j, k) == (1, 1, 1):
        return -(1 + 1j)
    elif (i, j, k) == (1, 1, 2):
        return 2
    elif (i, j, k) == (1, 1, 4):
        return 1j
    elif (i, j, k) == (1, 1, 5):
        return 1j
    elif (i, j, k) == (2, 0, 4):
        return 1
    elif (i, j, k) == (2, 0, 5):
        return -1
    elif (i, j, k) == (2, 1, 4):
        return 1
    elif (i, j, k) == (2, 1, 5):
        return -1
    elif (i, j, k) == (3, 0, 0):
        return 1
    elif (i, j, k) == (3, 0, 1):
        return -1
    elif (i, j, k) == (3, 1, 0):
        return 1
    elif (i, j, k) == (3, 1, 1):
        return -1
    elif (i, j, k) == (4, 0, 0):
        return -(1 + 1j)
    elif (i, j, k) == (4, 0, 1):
        return -(1 + 1j)
    elif (i, j, k) == (4, 1, 0):
        return -(1 + 1j)
    elif (i, j, k) == (4, 1, 1):
        return -(1 + 1j)
    elif (i, j, k) == (4, 2, 0):
        return 2
    elif (i, j, k) == (4, 2, 1):
        return 2
    elif (i, j, k) == (4, 4, 0):
        return 1j
    elif (i, j, k) == (4, 4, 1):
        return 1j
    elif (i, j, k) == (4, 5, 0):
        return 1j
    elif (i, j, k) == (4, 5, 1):
        return 1j
    elif (i, j, k) == (5, 0, 0):
        return (1 + 1j) ** 2
    elif (i, j, k) == (5, 0, 1):
        return (1 + 1j) ** 2
    elif (i, j, k) == (5, 1, 0):
        return (1 + 1j) ** 2
    elif (i, j, k) == (5, 1, 1):
        return (1 + 1j) ** 2
    elif (i, j, k) == (5, 0, 2):
        return -2 * (1 + 1j)
    elif (i, j, k) == (5, 1, 2):
        return -2 * (1 + 1j)
    elif (i, j, k) == (5, 2, 0):
        return -2 * (1 + 1j)
    elif (i, j, k) == (5, 2, 1):
        return -2 * (1 + 1j)
    elif (i, j, k) == (5, 0, 4):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 0, 5):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 1, 4):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 1, 5):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 4, 0):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 4, 1):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 5, 0):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 5, 1):
        return -1j * (1 + 1j)
    elif (i, j, k) == (5, 2, 2):
        return 4
    elif (i, j, k) == (5, 2, 4):
        return 2 * 1j
    elif (i, j, k) == (5, 2, 5):
        return 2 * 1j
    elif (i, j, k) == (5, 4, 2):
        return 2 * 1j
    elif (i, j, k) == (5, 5, 2):
        return 2 * 1j
    elif (i, j, k) == (5, 4, 4):
        return -1
    elif (i, j, k) == (5, 4, 5):
        return -1
    elif (i, j, k) == (5, 5, 4):
        return -1
    elif (i, j, k) == (5, 5, 5):
        return -1
    elif (i, j, k) == (6, 0, 4):
        return -(1 + 1j)
    elif (i, j, k) == (6, 1, 4):
        return -(1 + 1j)
    elif (i, j, k) == (6, 0, 5):
        return 1 + 1j
    elif (i, j, k) == (6, 1, 5):
        return 1 + 1j
    elif (i, j, k) == (6, 2, 4):
        return 2
    elif (i, j, k) == (6, 2, 5):
        return -2
    elif (i, j, k) == (6, 4, 4):
        return 1j
    elif (i, j, k) == (6, 5, 4):
        return 1j
    elif (i, j, k) == (6, 4, 5):
        return -1j
    elif (i, j, k) == (6, 5, 5):
        return -1j
    elif (i, j, k) == (7, 0, 0):
        return -(1 + 1j)
    elif (i, j, k) == (7, 1, 0):
        return -(1 + 1j)
    elif (i, j, k) == (7, 0, 1):
        return 1 + 1j
    elif (i, j, k) == (7, 1, 1):
        return 1 + 1j
    elif (i, j, k) == (7, 2, 0):
        return 2
    elif (i, j, k) == (7, 2, 1):
        return -2
    elif (i, j, k) == (7, 4, 0):
        return 1j
    elif (i, j, k) == (7, 5, 0):
        return 1j
    elif (i, j, k) == (7, 4, 1):
        return -1j
    elif (i, j, k) == (7, 5, 1):
        return -1j
    elif (i, j, k) == (8, 4, 0):
        return 1
    elif (i, j, k) == (8, 4, 1):
        return 1
    elif (i, j, k) == (8, 5, 0):
        return -1
    elif (i, j, k) == (8, 5, 1):
        return -1
    elif (i, j, k) == (9, 4, 0):
        return -(1 + 1j)
    elif (i, j, k) == (9, 4, 1):
        return -(1 + 1j)
    elif (i, j, k) == (9, 5, 0):
        return 1 + 1j
    elif (i, j, k) == (9, 5, 1):
        return 1 + 1j
    elif (i, j, k) == (9, 4, 4):
        return 1j
    elif (i, j, k) == (9, 4, 5):
        return 1j
    elif (i, j, k) == (9, 5, 4):
        return -1j
    elif (i, j, k) == (9, 5, 5):
        return -1j
    elif (i, j, k) == (9, 4, 2):
        return 2
    elif (i, j, k) == (9, 5, 2):
        return -2
    elif (i, j, k) == (10, 4, 4):
        return 1
    elif (i, j, k) == (10, 5, 5):
        return 1
    elif (i, j, k) == (10, 4, 5):
        return -1
    elif (i, j, k) == (10, 5, 4):
        return -1
    elif (i, j, k) == (11, 4, 0):
        return 1
    elif (i, j, k) == (11, 5, 1):
        return 1
    elif (i, j, k) == (11, 4, 1):
        return -1
    elif (i, j, k) == (11, 5, 0):
        return -1
    elif (i, j, k) == (12, 0, 0):
        return 1
    elif (i, j, k) == (12, 0, 1):
        return 1
    elif (i, j, k) == (12, 1, 0):
        return -1
    elif (i, j, k) == (12, 1, 1):
        return -1
    elif (i, j, k) == (13, 0, 0):
        return -(1 + 1j)
    elif (i, j, k) == (13, 0, 1):
        return -(1 + 1j)
    elif (i, j, k) == (13, 1, 0):
        return 1 + 1j
    elif (i, j, k) == (13, 1, 1):
        return 1 + 1j
    elif (i, j, k) == (13, 0, 2):
        return 2
    elif (i, j, k) == (13, 1, 2):
        return -2
    elif (i, j, k) == (13, 0, 4):
        return 1j
    elif (i, j, k) == (13, 0, 5):
        return 1j
    elif (i, j, k) == (13, 1, 4):
        return -1j
    elif (i, j, k) == (13, 1, 5):
        return -1j
    elif (i, j, k) == (14, 0, 4):
        return 1
    elif (i, j, k) == (14, 1, 5):
        return 1
    elif (i, j, k) == (14, 0, 5):
        return -1
    elif (i, j, k) == (14, 1, 4):
        return -1
    elif (i, j, k) == (15, 0, 0):
        return 1
    elif (i, j, k) == (15, 1, 1):
        return 1
    elif (i, j, k) == (15, 0, 1):
        return -1
    elif (i, j, k) == (15, 1, 0):
        return -1
    else:
        return 0


op_label = [["$I$", "$X$", "$Y$", "$Z$"] for i in range(2)]

def qpt_plot_cmap(chi, lbls_list, title='', cmap=plt.cm.bwr, figsize=(8, 6), threshold=None):
    """Like qt.qpt_plot_combined but with a user-chosen (cyclic) colormap."""
    xlabels = ["".join(p) for p in itertools.product(*lbls_list)]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    matrix_histogram(chi, xlabels, xlabels, bar_style='abs', color_style='phase',
                     cmap=cmap, options={'threshold': threshold}, ax=ax)
    ax.set_title(title)
    return fig, ax

def map_from_bloch_state_to_pauli_basis2(q, n, arr):

    if arr.shape != (6, 6, 6, 6):
        raise ValueError("Input array must be 6x6x6x6")

    if (q not in range(0, 16)) or (n not in range(0, 16)):
        raise ValueError("Input indices must be between 0 and 15, inclusive")

    result = 0

    for i in range(0, 6):  # first qubit prepare
        for j in range(0, 6):  # second qubit prepare
            for k in range(0, 6):  # first qubit measure
                for l in range(0, 6):  # second qubit measure
                    result += func_c2(q, i, j) * np.conj(func_c2(n, k, l)) * arr[i][j][k][l]

    return result


titlefont = {"color": "black", "weight": "normal", "size": 10}
axisfont = {"color": "black", "weight": "normal", "size": 8}
ticksfont = {"color": "black", "weight": "normal", "size": 8}

def plot_process_tomography2(chi_vector, save_file: str = None):

    plt.rcParams["text.usetex"] = False

    ticksfont = {"color": "black", "weight": "normal", "size": 4}

    cmap = colormaps.get_cmap("viridis")

    # set up figure
    fig = plt.figure(figsize=(6, 4), dpi=200, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # coordinates
    r = np.arange(16)
    _x, _y = np.meshgrid(r, r)
    x, y = _x.ravel(), _y.ravel()

    values0 = np.abs(chi_vector)
    values1 = np.angle(chi_vector)

    top = values0
    bottom = np.zeros_like(top)

    width = depth = 0.7

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    colors = cmap(norm(values1))

    xy_ticks_labels = [
        r"$II$",
        r"$IX$",
        r"$IY$",
        r"$IZ$",
        r"$XI$",
        r"$XX$",
        r"$XY$",
        r"$XZ$",
        r"$YI$",
        r"$YX$",
        r"$YY$",
        r"$YZ$",
        r"$ZI$",
        r"$ZX$",
        r"$ZY$",
        r"$ZZ$",
    ]

    ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors)
    ax.set_xticks(r + 0.5, labels=xy_ticks_labels, fontdict=ticksfont)
    ax.set_yticks(r + 0.5, labels=xy_ticks_labels, fontdict=ticksfont)
    ax.set_zticks([0, (max(top) / 2).round(2), max(top).round(2)])
    ax.set_xlabel("Prepared", fontdict=axisfont)
    ax.set_ylabel("Measured", fontdict=axisfont)
    ax.set_title(r"$\chi$ matrix", fontdict=titlefont)
    ax.view_init(20, -60, 0)

    sc = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_ticks(
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        labels=[r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"],
    )

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")

    plt.show()

def get_spam_error_superoperator(measured_identity_ptm):
    """
    Calculates the SPAM error superoperator (E_SPAM) from the measured Identity gate QPT.
    In the absence of gate error on the identity operation, the measured PTM is purely SPAM.
    """
    return np.array(measured_identity_ptm)

def get_spam_corrected_gate(measured_gate_ptm, spam_error_superoperator, p=0.5):
    """
    Calculates the SPAM-corrected gate superoperator (PTM representation).
    
    Parameters:
    - measured_gate_ptm: The uncorrected PTM of the measured gate.
    - spam_error_superoperator: The E_SPAM superoperator (usually measured identity).
    - p: The fraction of SPAM error attributed to state preparation (default p=1/2).
         Measurement error fraction is (1-p).
    """
    E_spam = np.array(spam_error_superoperator)
    R_meas = np.array(measured_gate_ptm)
    
    # Calculate fractional inverse matrix powers for Preparation (P) and Measurement (M)
    # P = E_spam^p  => P^-1 = E_spam^(-p)
    # M = E_spam^(1-p) => M^-1 = E_spam^(-(1-p))
    P_inv = fractional_matrix_power(E_spam, -p)
    M_inv = fractional_matrix_power(E_spam, -(1 - p))
    
    # R_corrected = M^-1 * R_meas * P^-1
    R_corrected = M_inv @ R_meas @ P_inv
    
    # Fractional matrix power might introduce highly negligible imaginary parts due to float precision
    if np.isrealobj(E_spam) and np.isrealobj(R_meas):
        R_corrected = np.real(R_corrected)
        
    return R_corrected

def QPT_fidelity(chi_measured, chi_ideal):
    """
    Calculate the gate fidelity and process tomography using the Uhlmann fidelity formula.
    FP = | Tr( sqrt( sqrt(chi_ideal) * chi_measured * sqrt(chi_ideal) ) ) |^2
    FG = (d * FP + 1) / (d + 1)
    
    Parameters:
    - chi_measured: The actual (or SPAM-corrected) chi matrix of the gate.
    - chi_ideal: The ideal target chi matrix of the gate.
    """
    # Determine the dimension d of the Hilbert space
    # Chi dimension is d^2 x d^2
    dim_sq = chi_measured.shape[0]
    d = int(np.sqrt(dim_sq))
    
    # Calculate sqrt of the target ideal chi matrix
    sqrt_chi_g = sqrtm(chi_ideal)
    
    # Inner product: sqrt(chi_ideal) * chi_measured * sqrt(chi_ideal)
    inner = sqrt_chi_g @ chi_measured @ sqrt_chi_g
    
    # Square root of the inner product
    sqrt_inner = sqrtm(inner)
    
    # Process fidelity is the square of the absolute trace
    F_pro = np.abs(np.trace(sqrt_inner))**2
    
    # Convert Process Fidelity to Average Gate Fidelity
    F_avg = (d * F_pro + 1) / (d + 1)
    
    return F_pro, F_avg

_wrap = lambda x: (x + np.pi) % (2*np.pi) - np.pi

def iSWAP_RPE(a):
    """5-parameter iSWAP (paper Eq. 8): a = [theta_p, phi_p, theta_1, theta_2, phi_zz]."""
    tp, pp, t1, t2, pzz = a
    c, s = np.cos(tp), np.sin(tp)
    U = np.array([[1, 0, 0, 0],
        [0, np.exp(1j*t2)*c,        1j*np.exp(1j*(t2+pp))*s, 0],
        [0, 1j*np.exp(1j*(t1-pp))*s, np.exp(1j*t1)*c,        0],
        [0, 0, 0, np.exp(1j*(t1+t2+pzz))]], dtype=complex)
    return qt.Qobj(U, dims=[[2, 2], [2, 2]])

# --- Pauli basis (qutip 'chi' ordering: P_i = P[i//4] (x) P[i%4], P=[I,X,Y,Z]) ---
_P1  = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
_P16 = [qt.tensor(a, b) for a in _P1 for b in _P1]
_PL  = [a + b for a in "IXYZ" for b in "IXYZ"]
_wrap = lambda x: (x + np.pi) % (2*np.pi) - np.pi

def as_chi(M):
    """Wrap a 16x16 array / Qobj as a chi-superrep Qobj."""
    A = M.full() if isinstance(M, qt.Qobj) else np.asarray(M)
    return qt.Qobj(A, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='chi')

def _project_eigs_to_simplex(w, T):
    """Euclidean projection of real eigenvalues w onto {lambda >= 0, sum(lambda) = T}
    (Duchi et al. 2008): shift every eigenvalue down by the same additive constant mu,
    then clip at 0. Unlike clip-then-uniformly-rescale, this only erodes the small/negative
    eigenvalues and leaves a dominant eigenvalue close to T essentially untouched."""
    w_sorted = np.sort(w)[::-1]
    css = np.cumsum(w_sorted) - T
    idx = np.arange(1, len(w) + 1)
    cond = w_sorted - css / idx > 0
    rho = idx[cond][-1]
    mu = css[cond][-1] / rho
    return np.clip(w - mu, 0, None)

def project_to_cp_chi(chi):
    """Project a (possibly non-physical) chi matrix onto the nearest completely-positive,
    trace-normalized process: Hermitize its Choi matrix, then find the Frobenius-closest PSD
    matrix with trace exactly d (the correct trace-preserving normalization for a d-dim
    system, e.g. d=4 for 2 qubits -- NOT the chi-superrep's Tr(chi)=d^2=16 convention) via
    _project_eigs_to_simplex. This removes the negative-eigenvalue artifact that a raw/
    inverted SPAM correction (e.g. get_spam_corrected_gate with p=0) can introduce -- which is
    what makes fidelity formulas that assume a valid density-matrix-like input (e.g.
    proc_fid_to_unitary's <c|chi|c>) report values above 1 / infidelities below 0 -- without
    the over-correction a naive clip-and-globally-rescale projection causes: that approach
    shrinks *every* eigenvalue (including an already-good, near-unitary dominant one) by the
    same ratio just because a few small eigenvalues were negative, which artificially
    depolarizes the whole process and can crater the fidelity far more than the actual
    unphysicality warrants.
    """
    choi_in = qt.to_choi(as_chi(chi))
    choi = choi_in.full()
    choi = (choi + choi.conj().T) / 2
    d = int(round(np.sqrt(choi.shape[0])))
    w, v = np.linalg.eigh(choi)
    w_proj = _project_eigs_to_simplex(w, d)
    choi_cp = (v * w_proj) @ v.conj().T
    choi_cp_qobj = qt.Qobj(choi_cp, dims=choi_in.dims, superrep='choi')
    return qt.to_chi(choi_cp_qobj)

##################################################
### Maximum-likelihood QPT (CPTP, from raw counts)
##################################################
# Unlike project_to_cp_chi (which repairs an already-inverted chi), this fits a physical
# process directly to the measured 00-outcome counts. Same Cholesky trick as the density-
# matrix MLE above (rho = T T^dag -> here Choi J = T T^dag), so positivity holds by
# construction rather than by clipping, and trace preservation is imposed exactly.

_MLE_CACHE = {}
_CHOI_DIMS = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]


def _tomo_rotations():
    """The 6 single-qubit tomography rotations, in the notebook's tomo_sequence order
    [I, X180, Y90, mY90, mX90, X90] (indices [0, 3, 21, 15, 13, 16])."""
    def Rx(t):
        return np.array([[np.cos(t/2), -1j*np.sin(t/2)], [-1j*np.sin(t/2), np.cos(t/2)]])

    def Ry(t):
        return np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])

    return [np.eye(2), Rx(np.pi), Ry(np.pi/2), Ry(-np.pi/2), Rx(-np.pi/2), Rx(np.pi/2)]


def _mle_forward_matrix():
    """Cached (1296, 256) matrix BC with  p00 = real(BC @ choi.ravel()).

    Encodes p00[i,j,k,l] = Tr[E_kl Lambda(rho_ij)] for
        rho_ij = (F_i (x) F_j) |00><00| (F_i (x) F_j)^dag        (prep)
        E_kl   = (F_k (x) F_l)^dag |00><00| (F_k (x) F_l)        ("00" effect)
    where Lambda is the channel qutip itself associates with the Choi matrix (qt.to_super of
    it, column-stacked: vec(Lambda(rho)) = S vec(rho), and Tr[E X] = vec(E^T) . vec(X)).

    Built from qutip's own Choi -> superoperator map, so it cannot disagree with to_chi /
    to_choi / process_fidelity. The previous construction pushed qutip's chi through the
    textbook formula Lambda(rho) = sum chi_mn P_m rho P_n, but qutip 5's chi is that of the
    TRANSPOSED Kraus operators (to_chi(to_super(U)) = 16 c(U^T) c(U^T)^dag), so the fit
    returned the channel with every Kraus operator transposed -- invisible for symmetric
    gates (iSWAP, diagonal phases) but wrong for e.g. Z(Q1).iSWAP, and it turned the TP
    constraint into a unitality constraint on the true process. Checked: a Lindblad
    simulation of a non-unital, asymmetric gate is reproduced to 2e-16 (previously 0.28).

    Built lazily: assembling it costs ~2 s, too slow to pay on every module import.
    """
    if "BC" not in _MLE_CACHE:
        F = _tomo_rotations()
        rho00 = np.zeros((4, 4)); rho00[0, 0] = 1.0
        Fs = [np.kron(F[a], F[b]) for a in range(6) for b in range(6)]
        R = np.array([(Fab @ rho00 @ Fab.conj().T).ravel(order='F') for Fab in Fs]).T     # (16, 36)
        E = np.array([(Fab.conj().T @ rho00 @ Fab).T.ravel(order='F') for Fab in Fs])     # (36, 16)

        BC = np.empty((1296, 256), dtype=complex)
        for k in range(256):
            e = np.zeros(256); e[k] = 1.0
            S = qt.to_super(qt.Qobj(e.reshape(16, 16), dims=_CHOI_DIMS, superrep='choi')).full()
            BC[:, k] = (E @ S @ R).T.ravel()          # row = prep * 36 + meas
        _MLE_CACHE["BC"] = BC
    return _MLE_CACHE["BC"]


# (-1)^(number of Y factors) of each two-qubit Pauli, in _P16 order. c(U^T) = _Y_PARITY * c(U),
# so D chi D (D = diag(_Y_PARITY)) swaps between the chi of a channel and of its transpose.
_Y_PARITY = np.array([(-1) ** (a.count("Y")) for a in [x + y for x in "IXYZ" for y in "IXYZ"]])


def _linear_init(p_obs):
    """Linear-inversion starting point for the MLE, in qutip's chi convention.

    solve(PMatrix, ...) assumes the textbook chi, which qutip reads as the Kraus-transposed
    channel; D chi D maps it back (see _mle_forward_matrix)."""
    chi = chi_from_p00_linear(p_obs).full()
    return qt.Qobj(_Y_PARITY[:, None] * chi * _Y_PARITY[None, :], dims=_CHOI_DIMS, superrep='chi')


def _choi_from_t(t, n=16):
    """J = T T^dag (T lower-triangular complex, so J >= 0 by construction), then rescaled to
    satisfy trace preservation exactly: with M = Tr_out(J), the sandwich
    J -> (M^-1/2 (x) I) J (M^-1/2 (x) I) gives Tr_out = I. In qutip's Choi convention the TP
    condition is einsum('ikjk->ij', J.reshape(4,4,4,4)) == I (confirmed with a non-unital
    amplitude-damping map, which distinguishes it from the other index pairing)."""
    idx = np.tril_indices(n)
    ntri = len(idx[0])
    T = np.zeros((n, n), dtype=complex)
    T[idx] = t[:ntri] + 1j * t[ntri:]
    J = T @ T.conj().T
    M = np.einsum('ikjk->ij', J.reshape(4, 4, 4, 4))
    w, v = np.linalg.eigh((M + M.conj().T) / 2)
    Minv_sqrt = v @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ v.conj().T
    Q = np.kron(Minv_sqrt, np.eye(4))
    return Q @ J @ Q.conj().T


def _t_from_choi(J, n=16):
    w, v = np.linalg.eigh((J + J.conj().T) / 2)
    Jp = (v * np.maximum(w, 1e-9)) @ v.conj().T
    T = np.linalg.cholesky(Jp + 1e-9 * np.eye(n))
    idx = np.tril_indices(n)
    return np.concatenate([T[idx].real, T[idx].imag])


def chi_from_p00_linear(p00):
    """The existing linear-inversion estimate, as a qutip chi Qobj -- i.e. exactly
    solve(PMatrix, ...) on p00, wrapped with the 16x normalization the notebook applies."""
    pr = np.asarray(p00, dtype=float).reshape(6, 6, 6, 6)
    mv = np.array([map_from_bloch_state_to_pauli_basis2(v // 16, v % 16, pr) for v in range(256)])
    return qt.Qobj(16 * solve(PMatrix, mv).reshape(16, 16), dims=_CHOI_DIMS, superrep='chi')


def mle_chi_from_p00(p00, n_shots, chi_init=None, maxiter=2000, verbose=False, return_result=False,
                     forward=None):
    """Maximum-likelihood CPTP process estimate from measured 00-outcome frequencies.

    Maximizes the binomial log-likelihood sum_s [n_s log p_s + (N - n_s) log(1 - p_s)] over
    processes parameterized as Choi J = T T^dag with Tr_out(J) = I, so the estimate is
    CP *and* TP by construction. This differs from linear inversion + project_to_cp_chi in
    two ways that both matter: the estimate can never leave the physical set (rather than
    being dragged back to its boundary afterwards), and each setting is weighted by its own
    shot noise instead of every Choi entry being treated as equally reliable.

    Parameters
    ----------
    p00 : array, (6,6,6,6) or (1296,)
        Measured P(outcome = 00) per (prep, meas) setting -- the notebook's `p00`.
    n_shots : int
        TOTAL shots behind each setting. When p00 is averaged over several datasets this is
        the sum over them (e.g. 10 datasets x 400 shots -> 4000), not the per-dataset count;
        it sets the weight of the likelihood against the CPTP constraint.
    chi_init : qutip Qobj, optional
        Starting point. Defaults to the linear-inversion estimate, projected onto the
        physical set to give the Cholesky factorization something valid to start from.
    forward : (1296, 256) array, optional
        Forward matrix p00 = real(forward @ choi.ravel()) to fit with instead of
        _mle_forward_matrix(), e.g. one whose "00" effect includes the measured readout error
        (F^dag M00 F with M00 = P(read 00 | basis state)). mle_chi_spam_corrected takes the same.

    Returns
    -------
    chi : qutip Qobj, superrep='chi', Tr = 16, guaranteed CPTP.
    result : scipy OptimizeResult, only if return_result=True.

    Notes
    -----
    On simulated data with a known gate (3 gate types x {500, 4000} shots x 3 seeds), this
    lands 9-12x closer to the true Choi matrix in Frobenius norm than linear inversion +
    project_to_cp_chi, recovering F_pro ~ 0.999 where the projection route gives 0.945-0.982.
    Takes ~5 s on simulated data, ~25 s on real 10-dataset data.

    MLE estimates sit on the boundary of the physical set (typically rank-deficient), which
    is a known source of bias -- see Blume-Kohout's "hedged MLE" if you need error bars.
    """
    BC = _mle_forward_matrix() if forward is None else forward
    p_obs = np.asarray(p00, dtype=float).ravel()
    if p_obs.size != 1296:
        raise ValueError(f"p00 must have 1296 entries (6^4), got {p_obs.size}.")
    n1 = p_obs * n_shots
    n0 = (1.0 - p_obs) * n_shots

    if chi_init is None:
        chi_init = _linear_init(p_obs)
    J0 = qt.to_choi(chi_init).full()
    t0 = _t_from_choi(J0)

    def nll(t):
        p = np.real(BC @ _choi_from_t(t).ravel())
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -np.sum(n1 * np.log(p) + n0 * np.log1p(-p))

    res = minimize(nll, t0, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 10**7, "disp": verbose})
    chi = qt.to_chi(qt.Qobj(_choi_from_t(res.x), dims=_CHOI_DIMS, superrep="choi"))
    return (chi, res) if return_result else chi


def _spam_forward_matrix(ref_chi, p=0.5, forward=None):
    """Linear map L with p00 = real(L @ vec(J_G)), for J_G the Choi of the SPAM-CORRECTED
    gate G -- i.e. the forward model with SPAM folded in.

    Inverts get_spam_corrected_gate's convention exactly:
        R_corrected = E^-(1-p) R_meas E^-p    <=>    R_meas = E^(1-p) R_corrected E^p
    so predicting counts from G means pushing E^(1-p) G E^p through the ordinary forward
    model. E is taken from ref_chi (the identity-reference process).

    Built column-by-column through qutip's own to_super/to_choi so no vec-ordering
    convention can drift from the rest of this module. Costs ~7 s; not cached because it
    depends on ref_chi.
    """
    BC = _mle_forward_matrix() if forward is None else forward
    E = qt.to_super(ref_chi).full()
    A = fractional_matrix_power(E, 1.0 - p)
    B = fractional_matrix_power(E, p)

    L = np.empty((1296, 256), dtype=complex)
    for k in range(256):
        e = np.zeros(256); e[k] = 1.0
        J_basis = qt.Qobj(e.reshape(16, 16), dims=_CHOI_DIMS, superrep='choi')
        R_meas = A @ qt.to_super(J_basis).full() @ B
        J_meas = qt.to_choi(qt.Qobj(R_meas, dims=_CHOI_DIMS, superrep='super')).full()
        L[:, k] = BC @ J_meas.ravel()
    return L


def mle_chi_spam_corrected(p00, ref_chi, n_shots, p=0.5, chi_init=None,
                           maxiter=2000, verbose=False, return_result=False, forward=None):
    """MLE of the SPAM-CORRECTED gate, fitted directly against the raw 00-outcome counts.

    Replaces the whole  MLE -> get_spam_corrected_gate -> project_to_cp_chi  chain. Rather
    than reconstructing the measured process and then inverting SPAM out of it (an inversion
    that breaks complete positivity even when its inputs are exactly physical), this folds
    SPAM into the forward model and fits the corrected gate itself under the CPTP
    constraint. The result is SPAM-corrected AND physical by construction, so no projection
    step is needed at all.

    Parameters
    ----------
    p00 : array, (6,6,6,6) or (1296,)
        Measured P(outcome = 00) for the GATE (not the reference).
    ref_chi : qutip Qobj
        The identity-reference process, as a chi Qobj -- e.g. mle_chi_from_p00(ref_p00, n).
        Held FIXED: this is not a joint gate+SPAM fit, so error in the reference estimate
        propagates into the gate. A genuinely self-consistent treatment is GST.
    n_shots : int
        Total shots behind each setting (summed over datasets).
    p : float
        Fraction of SPAM attributed to state preparation, matching
        get_spam_corrected_gate's `p`. 0.5 splits it evenly (Blume-Kohout et al.,
        arXiv:2412.16293, recommend the even split).

    Returns
    -------
    chi : qutip Qobj, superrep='chi', Tr = 16, guaranteed CPTP and SPAM-corrected.

    Notes
    -----
    On simulated data with a known gate and known 6% depolarizing SPAM, this lands ~1.5x
    closer to the true Choi matrix than MLE -> SPAM-invert -> project (0.022 vs 0.032 in
    Frobenius norm) while being exactly CP and TP, where the invert-then-project route still
    leaves TP violated. Note the un-projected invert route reports a *higher* fidelity than
    the truth-closest estimate -- fidelity computed on a non-physical chi is not bounded by
    1 and should not be used to rank these.
    """
    L = _spam_forward_matrix(ref_chi, p=p, forward=forward)
    p_obs = np.asarray(p00, dtype=float).ravel()
    if p_obs.size != 1296:
        raise ValueError(f"p00 must have 1296 entries (6^4), got {p_obs.size}.")
    n1 = p_obs * n_shots
    n0 = (1.0 - p_obs) * n_shots

    if chi_init is None:
        chi_init = _linear_init(p_obs)
    t0 = _t_from_choi(qt.to_choi(chi_init).full())

    def nll(t):
        pr = np.real(L @ _choi_from_t(t).ravel())
        pr = np.clip(pr, 1e-12, 1.0 - 1e-12)
        return -np.sum(n1 * np.log(pr) + n0 * np.log1p(-pr))

    res = minimize(nll, t0, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 10**7, "disp": verbose})
    chi = qt.to_chi(qt.Qobj(_choi_from_t(res.x), dims=_CHOI_DIMS, superrep="choi"))
    return (chi, res) if return_result else chi


def closest_unitary(chi):
    """Nearest EXACTLY-unitary matrix to a (possibly non-physical) chi: dominant Choi
    eigenvector, then SVD polar projection (U,_,Vh = svd(K); U@Vh) forces ||U^dag U - I||=0,
    discarding any magnitude/non-unitarity information in the raw reconstruction."""
    ch = qt.to_choi(as_chi(chi)).full(); ch = (ch + ch.conj().T)/2
    w, v = np.linalg.eigh(ch)
    K = v[:, np.argmax(w)].reshape(4, 4, order='F')
    if abs(K[0, 0]) > 1e-9:
        K = K/(K[0, 0]/abs(K[0, 0]))          # fix global phase
    U, _, Vh = np.linalg.svd(K)               # polar -> nearest unitary
    return qt.Qobj(U @ Vh, dims=[[2, 2], [2, 2]]), w

def chi_to_unitary(chi):
    """RAW dominant-Choi-eigenvector reconstruction (matches RPE_sim.ipynb's chi_to_unitary
    exactly): only the global phase is fixed, NO polar projection. On noisy/non-physical chi
    this is generally NOT exactly unitary -- use this for a true unfitted/unprojected
    comparison; use closest_unitary() when you need a guaranteed-unitary matrix."""
    choi = qt.to_choi(as_chi(chi)).full()
    choi = (choi + choi.conj().T) / 2
    w, V = np.linalg.eigh(choi)
    K = (V[:, np.argmax(w)] * np.sqrt(max(w.max(), 0.0))).reshape(4, 4, order='F')
    return qt.Qobj(K / (K[0, 0] / abs(K[0, 0])), dims=[[2, 2], [2, 2]])

def unitary_pauli_coeffs(U):
    """c_P = Tr(P.dag @ U)/4 in the qutip chi Pauli ordering."""
    M = U.full() if isinstance(U, qt.Qobj) else np.asarray(U)
    return np.array([np.trace(P.full().conj().T @ M)/4 for P in _P16])

def proc_fid_to_unitary(chi, U):
    """Process (entanglement) fidelity of measured chi to unitary U: <c| chi_norm |c>.

    qutip's chi of U is 16 c(U^T) c(U^T)^dag (see _mle_forward_matrix), so the overlap is taken
    with c(U^T); this equals qt.process_fidelity(chi, U). Using c(U) instead measures the
    fidelity to U^T, which differs whenever U is not symmetric."""
    M = as_chi(chi).full(); M = (M + M.conj().T)/2; M = M/np.trace(M).real
    c = _Y_PARITY * unitary_pauli_coeffs(U)
    return float(np.real(np.vdot(c, M @ c)))

def extract_angles(U, tol=1e-2, phi_p=0):
    """Observable angles (theta_p, theta_s, theta_d, phi_zz, theta_sum) always;
    individuals (theta_1, theta_2, phi_p) either from an assumed phi_p -- via the off-diagonal
    phases a12 = theta_2+phi_p+pi/2, a21 = theta_1-phi_p+pi/2, inverted as
    theta_1 = a21-pi/2+phi_p, theta_2 = a12-pi/2-phi_p -- or, with phi_p=None, read off the
    diagonal when it survives (i.e. not a full iSWAP). The assumed branch also resolves a full
    iSWAP, where the diagonal vanishes and the tol-based branch cannot separate
    theta_1/theta_2/phi_p.

    Parameters
    ----------
    phi_p : float | None
        phi_p in RADIANS to assume (default 0). None instead reads phi_p off the data via the
        tol-based branch, falling back to NaN individuals when the diagonal has vanished.

    Note that phi_p is only a free *choice* where it is unobservable -- at a full iSWAP
    (theta_p = pi/2) the diagonal vanishes and only theta_sum and theta_d are fixed by the
    data, so phi_p just gauges how theta_sum splits into theta_1/theta_2. Away from that,
    the diagonal does determine phi_p, and forcing a different value here returns
    theta_1/theta_2 that no longer reproduce U -- prefer phi_p=None there.
    """
    M = U.full() if isinstance(U, qt.Qobj) else np.asarray(U)
    a12, a21 = np.angle(M[1, 2]), np.angle(M[2, 1])
    mag_off  = 0.5*(abs(M[1, 2]) + abs(M[2, 1]))
    mag_diag = 0.5*(abs(M[1, 1]) + abs(M[2, 2]))
    out = dict(
        theta_p   = np.arctan2(mag_off, mag_diag),
        theta_sum = _wrap(a12 + a21 - np.pi),     # theta_1 + theta_2      (phi_p-independent)
        theta_s   = _wrap(np.angle(M[3, 3])),     # theta_1+theta_2+phi_zz (seq d)
        theta_d   = _wrap(a21 - a12),             # (theta_1-theta_2)-2phi_p (seq f)
    )
    out['phi_zz'] = _wrap(out['theta_s'] - out['theta_sum'])
    if phi_p is not None:
        out['theta_1'] = _wrap(a21 - np.pi/2 + phi_p)
        out['theta_2'] = _wrap(a12 - np.pi/2 - phi_p)
        out['phi_p']   = _wrap(phi_p)
    elif abs(M[1, 1]) > tol and abs(M[2, 2]) > tol:
        out['theta_1'] = _wrap(np.angle(M[2, 2]))
        out['theta_2'] = _wrap(np.angle(M[1, 1]))
        out['phi_p']   = _wrap(a12 - np.angle(M[1, 1]) - np.pi/2)
        # theta_sum/phi_zz above came from the off-diagonals, which scale as sin(theta_p) and
        # vanish as the gate approaches the identity -- the same degeneracy that sends this
        # branch to the diagonal in the first place. Whenever the diagonal is the better
        # conditioned estimator, phi_zz must be rebuilt from it too, or it silently inherits
        # the degeneracy that theta_1/theta_2 just escaped. Measured on the five 20260910
        # idle-reference runs (theta_p = 0.18 deg): off-diagonal route gives phi_zz scattering
        # +48 to -16 deg run to run, diagonal route gives -80.26 +- 0.09 deg, which is the
        # passive ZZ over the 518 ns window to 0.5% of the independently measured 432.5 kHz.
        out['theta_sum'] = _wrap(out['theta_1'] + out['theta_2'])
        out['phi_zz']    = _wrap(out['theta_s'] - out['theta_sum'])
    else:
        out['theta_1'] = out['theta_2'] = out['phi_p'] = np.nan
    return out

def fit_iswap_rpe_to_chi(chi, p0=None, phi_p=0):
    """Least-squares fit of iSWAP_RPE params to a measured chi (in chi-superrep space).
    phi_p is passed straight to extract_angles -- it only reinterprets the fitted U's angles,
    it does not constrain the fit itself."""
    M = as_chi(chi).full(); M = (M + M.conj().T)/2; M = M/np.trace(M).real
    def resid(p):
        cm = qt.to_chi(qt.to_super(iSWAP_RPE(p))).full(); cm = cm/np.trace(cm).real
        d = (M - cm).ravel()
        return np.concatenate([d.real, d.imag])
    if p0 is None:
        p0 = [np.pi/2, 0.0, 0.0, 0.0, 0.0]
    r = least_squares(resid, p0)
    U = iSWAP_RPE(r.x)
    angles = extract_angles(U, phi_p=phi_p)
    return angles, proc_fid_to_unitary(chi, U), r

def deg(d):
    return {k: (np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else np.rad2deg(v))
            for k, v in d.items()}

##################################################
### Gate Set Tomography (GST) ###
##################################################


def ds_from_dataset_txt(dataset_txt):
    lines = [ln.strip() for ln in dataset_txt.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("dataset_txt is empty")

    header = lines[0]
    if not header.startswith("## Columns ="):
        raise ValueError("First line must start with '## Columns ='")

    col_part = header.split("=", 1)[1].strip()
    outcome_labels = [tok.strip().split()[0] for tok in col_part.split(",")]
    if not outcome_labels:
        raise ValueError("No outcome labels found in header")

    ds = pygsti.data.DataSet(outcome_labels=outcome_labels)

    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 1 + len(outcome_labels):
            raise ValueError(
                f"Bad row: expected {1 + len(outcome_labels)} columns, got {len(parts)} in '{ln}'"
            )
        circuit_label = parts[0]
        counts = [int(x) for x in parts[1:]]
        count_dict = {out: cnt for out, cnt in zip(outcome_labels, counts)}
        ds.add_count_dict(circuit_label, count_dict)

    ds.done_adding_data()
    return ds

def gst_ds_datasets_homogeneity(ds_list):
    """chi^2/dof for "did these repeat GST runs measure the same device?".

    Repeat runs of one GST experiment can be pooled only if they are i.i.d. samples of the
    same gate set. This is a chi^2 test of homogeneity on the per-circuit outcome counts:
    each circuit contributes (n_runs - 1) * 3 degrees of freedom, and the pooled outcome
    frequencies supply the expected counts.

    Returns (chi2, dof, ratio). ratio ~= 1 means the runs are statistically identical and
    pooling is sound. ratio >> 1 means the device moved between runs -- pooling then averages
    over different gate sets, and since an average of unitaries is not unitary, the drift is
    converted into apparent depolarization: the fit reports a lower fidelity with the loss
    showing up in the non-unitary column, indistinguishable from real decoherence.

    Measured on the five 20260909/10 runs: ratio 2.76, with the pairwise total-variation
    distance growing monotonically with time separation (0.016 adjacent, 0.024 across
    4.4 hours, against a 0.011 shot-noise floor) -- i.e. genuine drift, not scatter.
    """
    C = np.stack([_gst_ds_counts(d) for d in ds_list])       # (n_runs, n_circuits, 4)
    N = C.sum(axis=2)
    tot = C.sum(axis=0)
    p = tot / np.maximum(tot.sum(axis=1, keepdims=True), 1)
    exp = p[None, :, :] * N[:, :, None]
    chi2 = float(np.where(exp > 5, (C - exp) ** 2 / np.maximum(exp, 1e-9), 0.0).sum())
    dof = (len(ds_list) - 1) * 3 * C.shape[1]
    return chi2, dof, chi2 / dof


def _gst_ds_counts(gst_ds):
    """(n_circuits, 4) outcome counts in 00, 01, 10, 11 order."""
    y0 = np.rint(np.asarray(gst_ds["y0"].values)).astype(int)
    y1 = np.rint(np.asarray(gst_ds["y1"].values)).astype(int)
    c = np.empty((y0.shape[1], 4), dtype=int)
    c[:, 0] = ((y0 == 0) & (y1 == 0)).sum(axis=0)
    c[:, 1] = ((y0 == 0) & (y1 == 1)).sum(axis=0)
    c[:, 2] = ((y0 == 1) & (y1 == 0)).sum(axis=0)
    c[:, 3] = ((y0 == 1) & (y1 == 1)).sum(axis=0)
    return c


def concat_gst_datasets(ds_list, check_homogeneity=True, warn_ratio=1.5):
    """Stack repeat GST runs along the shot axis into one dataset.

    Combining happens at the SHOT level, before any counting, so the result feeds
    ``gst_ds_to_pygsti_dataset_txt`` unchanged and the pooled counts are exactly the sum of
    the parts. All runs must share the same circuit list in the same order -- the acq_index
    axis is positional, so a differing experiment_list would silently mix circuits.

    Pooling is a statistics-vs-systematics trade: n runs cut the shot-noise floor by sqrt(n)
    but fold any drift between them into the fit as apparent decoherence. Unless you
    specifically want the time-averaged gate, prefer fitting each run separately and
    averaging the extracted parameters, using the run-to-run spread as the error bar.
    ``check_homogeneity`` warns when the runs fail that test; see
    ``gst_ds_datasets_homogeneity``.
    """

    ds_list = list(ds_list)
    if not ds_list:
        raise ValueError("ds_list is empty.")
    shapes = {np.asarray(d["y0"].values).shape[1] for d in ds_list}
    if len(shapes) != 1:
        raise ValueError(f"datasets cover different circuit counts: {sorted(shapes)}")

    if check_homogeneity and len(ds_list) > 1:
        chi2, dof, ratio = gst_ds_datasets_homogeneity(ds_list)
        if ratio > warn_ratio:
            print(
                f"Warning: these {len(ds_list)} runs are not statistically identical "
                f"(homogeneity chi2/dof = {ratio:.2f}). Pooling will convert the drift "
                f"between them into apparent decoherence. Consider fitting each run "
                f"separately and averaging the parameters instead."
            )

    data = {
        v: (("repetition", "acq_index_0"),
            np.concatenate([np.asarray(d[v].values) for d in ds_list], axis=0))
        for v in ("y0", "y1")
    }
    return xr.Dataset(data)


def gst_ds_to_pygsti_dataset_txt(
    gst_ds,
    listOfExperiments,
    mode="auto",
    single_qubit_var="y0",
):
    """Convert a Quantify GST dataset to pyGSTi dataset.txt text.

    - First column is a compact pyGSTi circuit string, e.g. ({})Gxpi2:0Gxpi2:0
    - No file is written; this function only returns dataset_txt.

    Example usage
    dataset_txt = gst_ds_to_pygsti_dataset_txt(gst_ds, listOfExperiments, mode="auto")
    print("\n".join(dataset_txt.splitlines()[:6]))
    """
    data_vars = list(gst_ds.data_vars)
    data_var_set = set(data_vars)

    if mode == "auto":
        mode = "2q" if {"y0", "y1"}.issubset(data_var_set) else "1q"

    n_exp = len(listOfExperiments)

    def _extract_shots(var_name, exp_idx):
        if var_name not in gst_ds.data_vars:
            raise KeyError(f"'{var_name}' not found in gst_ds.data_vars: {list(gst_ds.data_vars)}")
        arr = np.asarray(gst_ds[var_name].values)
        if arr.ndim == 0:
            raise ValueError(f"{var_name} is scalar; expected shot/experiment dimensions.")
        if arr.ndim == 1:
            if n_exp != 1:
                raise ValueError(
                    f"{var_name} is 1D but listOfExperiments has {n_exp} experiments."
                )
            return arr
        if arr.shape[1] == n_exp:
            return arr[:, exp_idx]
        if arr.shape[0] == n_exp:
            return arr[exp_idx, :]
        raise ValueError(
            f"Cannot align {var_name} shape {arr.shape} with {n_exp} experiments."
        )

    def _circuit_to_dataset_label(circuit):
        # Prefer pyGSTi's compact "Circuit(({})Gxpi2:0...@(0))" style when available.
        s = repr(circuit)
        if s.startswith("Circuit(") and s.endswith(")"):
            s = s[len("Circuit("):-1]
        # # Remove trailing qubit-location annotation "@(0)" / "@(0,1)" for dataset text.
        # at_idx = s.rfind("@(")
        # if at_idx != -1 and s.endswith(")"):
        #     s = s[:at_idx]
        return s

    if mode == "1q":
        if single_qubit_var not in data_var_set:
            if len(data_vars) == 1:
                single_qubit_var = data_vars[0]
            else:
                raise KeyError(
                    f"single_qubit_var '{single_qubit_var}' missing; available vars: {data_vars}"
                )
        header = "## Columns = 0 count, 1 count"
    elif mode == "2q":
        if not {"y0", "y1"}.issubset(data_var_set):
            raise KeyError(f"2q mode requires y0 and y1; available vars: {data_vars}")
        header = "## Columns = 00 count, 01 count, 10 count, 11 count"
    else:
        raise ValueError("mode must be 'auto', '1q', or '2q'.")

    lines = [header]
    for exp_idx, circuit in enumerate(listOfExperiments):
        cstr = _circuit_to_dataset_label(circuit)

        if mode == "1q":
            y = _extract_shots(single_qubit_var, exp_idx)
            y = np.rint(np.asarray(y)).astype(int).ravel()
            if not np.isin(y, [0, 1]).all():
                raise ValueError(f"{single_qubit_var} contains non-binary values for experiment {exp_idx}.")
            c1 = int(np.sum(y == 1))
            c0 = int(np.sum(y == 0))
            lines.append(f"{cstr} {c0} {c1}")
            continue

        y0 = _extract_shots("y0", exp_idx)
        y1 = _extract_shots("y1", exp_idx)
        y0 = np.rint(np.asarray(y0)).astype(int).ravel()
        y1 = np.rint(np.asarray(y1)).astype(int).ravel()
        nshots = min(y0.size, y1.size)
        if nshots == 0:
            raise ValueError(f"No shots found for experiment {exp_idx}.")
        y0 = y0[:nshots]
        y1 = y1[:nshots]
        if (not np.isin(y0, [0, 1]).all()) or (not np.isin(y1, [0, 1]).all()):
            raise ValueError(f"y0/y1 contain non-binary values for experiment {exp_idx}.")

        c00 = int(np.sum((y0 == 0) & (y1 == 0)))
        c01 = int(np.sum((y0 == 0) & (y1 == 1)))
        c10 = int(np.sum((y0 == 1) & (y1 == 0)))
        c11 = int(np.sum((y0 == 1) & (y1 == 1)))
        lines.append(f"{cstr} {c00} {c01} {c10} {c11}")

    return "\n".join(lines)

# generated_germs = [
#     pygsti.circuits.Circuit('Gi:0@(0,1)'),
#     pygsti.circuits.Circuit('Gi:1@(0,1)'),
#     pygsti.circuits.Circuit('Gxpi2:0@(0,1)'),
#     pygsti.circuits.Circuit('Gxpi2:1@(0,1)'),
#     pygsti.circuits.Circuit('Gypi2:0@(0,1)'),
#     pygsti.circuits.Circuit('Gypi2:1@(0,1)'),
#     pygsti.circuits.Circuit('Giswap:0:1@(0,1)'),
#     pygsti.circuits.Circuit('Gxpi2:0Gxpi2:1Gypi2:0Gxpi2:0Giswap:0:1Giswap:0:1@(0,1)'),
#     pygsti.circuits.Circuit('Gxpi2:0Gxpi2:1Gxpi2:1Giswap:0:1Giswap:0:1@(0,1)'),
#     pygsti.circuits.Circuit('Gxpi2:0Gxpi2:0Gxpi2:1Gypi2:0Gxpi2:0Gypi2:1@(0,1)'),
#     pygsti.circuits.Circuit('Gypi2:0Gypi2:0Gypi2:1Giswap:0:1Gypi2:1Giswap:0:1@(0,1)'),
#     pygsti.circuits.Circuit('Gi:0Giswap:0:1Gxpi2:0Gxpi2:1Gxpi2:1Gypi2:1@(0,1)'),
#     pygsti.circuits.Circuit('Gxpi2:0Giswap:0:1Gxpi2:0Giswap:0:1Gxpi2:1Gxpi2:1@(0,1)'),
#     pygsti.circuits.Circuit('Gxpi2:0Gxpi2:1Giswap:0:1Gypi2:0Gypi2:0@(0,1)'),
#     pygsti.circuits.Circuit('Gxpi2:1Gypi2:1Gypi2:1Gypi2:1Gxpi2:1Giswap:0:1@(0,1)'),
#     pygsti.circuits.Circuit('Gxpi2:0Gypi2:1Giswap:0:1Gxpi2:1Gypi2:0Giswap:0:1@(0,1)'),
# ]

# # Use standard-model 2Q fiducials for this alternate 2Q gate model
# prep_fiducials = smq2Q_XYICNOT.prep_fiducials()
# meas_fiducials = smq2Q_XYICNOT.meas_fiducials()
# print(f"prep fiducials: {len(prep_fiducials)}")
# print(f"meas fiducials: {len(meas_fiducials)}")

# # Memory-safe germ search for the iSWAP model
# generated_germs = gsel.find_germs(
#     iswap_mdl,
#     seed=1234,
#     randomize=False,
#     num_gs_copies=1,
#     algorithm='greedy',
#     mode='single-Jac',
#     mem_limit=2 * 1024**3,
#     toss_random_frac=0.98,
#     force='singletons',
#     verbosity=2,
#  )

# print(f"Generated germs: {len(generated_germs)}")
# is_complete_infl_generated = gsel.test_germ_set_infl(iswap_mdl, generated_germs)
# is_complete_finitel_generated = gsel.test_germ_set_finitel(iswap_mdl, generated_germs, length=4)
# print(f"Generated set infl completeness: {is_complete_infl_generated}")
# print(f"Generated set finite-L completeness (L=4): {is_complete_finitel_generated}")