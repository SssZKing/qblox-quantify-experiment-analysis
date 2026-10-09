"""Two-qubit state tomography: linear inversion and maximum-likelihood density matrices.

Conventions: basis |Q1 Q2> = 00, 01, 10, 11 with Q1 = the dataset's first data variable (``y0``)
and the first Kronecker factor. Stokes vectors hold the 15 expectation values <P> of the
non-identity two-qubit Paulis in the order IX, IY, IZ, XI, XX, ..., ZZ (``STOKES_LABELS``).
"""
import numpy as np
from scipy.optimize import least_squares

from ._paulis import PAULI_2Q, PAULI_LABELS, II

STOKES_LABELS = PAULI_LABELS[1:]
STOKES_PAULIS = PAULI_2Q[1:]

# The state tomography schedule records 9 measurement settings, named by the basis each qubit
# is read out in (Q1 first): ZX, ZY, ZZ, XZ, XX, XY, YZ, YX, YY. For each Stokes parameter, the
# setting it is read from, and the signs of the outcomes 00, 01, 10, 11 in its estimator.
N_STATE_SETTINGS = 9
_STOKES_SETTING = np.array([0, 1, 2, 3, 4, 5, 3, 6, 7, 8, 6, 2, 0, 1, 2])
_Q2_PARITY = [1, -1, 1, -1]
_Q1_PARITY = [1, 1, -1, -1]
_JOINT_PARITY = [1, -1, -1, 1]
_STOKES_SIGNS = np.array(
    [_Q2_PARITY] * 3                       # IX IY IZ
    + [_Q1_PARITY] + [_JOINT_PARITY] * 3   # XI XX XY XZ
    + [_Q1_PARITY] + [_JOINT_PARITY] * 3   # YI YX YY YZ
    + [_Q1_PARITY] + [_JOINT_PARITY] * 3   # ZI ZX ZY ZZ
)

# Reorder a 15-element Stokes vector from the standard Pauli order
# (IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ)
# into the grouped bar-plot order: single-Q2 (I*), single-Q1 (*I), then the two-qubit terms.
STOKES_PLOT_ORDER = [2, 0, 1, 11, 3, 7, 14, 6, 10, 12, 4, 8, 13, 5, 9]


def joint_probabilities(dataset):
    """(4, n_settings) array of the joint outcome probabilities [P00, P01, P10, P11] per
    acquisition index, from the thresholded single-shot data of a two-qubit dataset."""
    q1 = dataset[list(dataset.data_vars)[0]].values
    q2 = dataset[list(dataset.data_vars)[1]].values
    return np.array([
        ((1 - q1) * (1 - q2)).mean(axis=0),
        ((1 - q1) * q2).mean(axis=0),
        (q1 * (1 - q2)).mean(axis=0),
        (q1 * q2).mean(axis=0),
    ])


def calculate_density_matrix(dataset, confusion_matrix=None):
    """Linear-inversion density matrix of a two-qubit state tomography run.

    Parameters
    ----------
    dataset : xarray.Dataset or str
        The tomography dataset (or its tuid), with 9 acquisition indices in the setting order
        described at the top of this module.
    confusion_matrix : (4, 4) array, optional
        Readout confusion matrix, P(read row | prepared column) in 00, 01, 10, 11 order; its
        inverse is applied to the outcome probabilities of every setting.

    Returns
    -------
    rho : (4, 4) complex array, not necessarily positive semidefinite.
    stokes : (15,) array of Pauli expectation values in ``STOKES_LABELS`` order.
    """
    if isinstance(dataset, str):
        from quantify_core.data.handling import load_dataset
        dataset = load_dataset(dataset)
    probs = joint_probabilities(dataset)
    if probs.shape[1] != N_STATE_SETTINGS:
        raise ValueError(
            f"expected {N_STATE_SETTINGS} tomography settings, got {probs.shape[1]} acquisition indices"
        )
    probs = probs[:, _STOKES_SETTING]
    if confusion_matrix is not None:
        probs = np.linalg.inv(confusion_matrix) @ probs
    stokes = np.einsum("sk,ks->s", _STOKES_SIGNS, probs)
    return rho_from_stokes(stokes), stokes


def rho_from_stokes(stokes):
    """rho = (II + sum_P <P> P) / 4 from a 15-element Stokes vector."""
    return 0.25 * (II + np.tensordot(stokes, np.array(STOKES_PAULIS), axes=1))


def compute_stokes(rho):
    """The 15 Stokes parameters Tr(rho P) of a two-qubit density matrix."""
    return np.array([np.real(np.trace(np.dot(rho, P))) for P in STOKES_PAULIS])


def rearrange_stokes(stokes):
    return np.asarray(stokes)[STOKES_PLOT_ORDER]


def extract_population_phase(single_rho, label, coherence_eps=1e-10, verbose=True):
    """Populations and the phase of rho[0, 1] of a single-qubit density matrix.

    Prints them (as before) unless ``verbose=False``, and returns a dict with keys
    p0, p1, coherence (|rho[0, 1]|), phase_rad and phase_deg. The phase is NaN when the
    coherence is below ``coherence_eps``.
    """
    rho = single_rho.full() if hasattr(single_rho, "full") else np.asarray(single_rho)
    p0 = float(np.real(rho[0, 0]))
    p1 = float(np.real(rho[1, 1]))

    coherence = rho[0, 1]
    coherence_mag = float(np.abs(coherence))
    phase_rad = float(np.angle(coherence)) if coherence_mag > coherence_eps else float("nan")
    phase_deg = float(np.degrees(phase_rad))

    if verbose:
        print(f"{label} population: P(|0>)={p0:.4f}, P(|1>)={p1:.4f}")
        print(f"{label} phase from rho[0,1]: {phase_rad:.4f} rad ({phase_deg:.2f} deg), |rho[0,1]|={coherence_mag:.4f}")
        print("-" * 70)
    return {"p0": p0, "p1": p1, "coherence": coherence_mag, "phase_rad": phase_rad, "phase_deg": phase_deg}


##################################################
### Maximum-likelihood density matrices
##################################################
# A physical density matrix is parameterized by its Cholesky factor, rho = T T^dag / Tr(T T^dag),
# with T complex lower-triangular and unrolled into a real vector t for the optimizer:
# the real parts of each subdiagonal, then (below the diagonal) their imaginary parts.

def t2T(t):
    """Lower-triangular Cholesky factor T from the real parameter vector t (length d^2)."""
    d = int(np.sqrt(len(t)))
    idx = 0
    cur_length = d
    T = np.zeros([d, d])
    for j in range(d):
        # real parts of the j-th subdiagonal
        T = T + np.diag(t[np.arange(idx, idx + cur_length)], -j)
        idx = idx + cur_length
        if j > 0:
            # imaginary parts of the j-th subdiagonal (off-diagonals only)
            T = T + 1j * np.diag(t[np.arange(idx, idx + cur_length)], -j)
            idx = idx + cur_length
        cur_length -= 1
    return T


def T2t(T):
    """Inverse of t2T: unroll a lower-triangular T into the real vector t."""
    d = len(T)
    idx = 0
    cur_length = d
    t = np.zeros(d**2)
    for j in range(d):
        t[np.arange(idx, idx + cur_length)] = np.real(np.diag(T, -j))
        idx = idx + cur_length
        if j > 0:
            t[np.arange(idx, idx + cur_length)] = np.imag(np.diag(T, -j))
            idx = idx + cur_length
        cur_length -= 1
    return t


def T2rho(T, normalize=True):
    """rho = T T^dag, normalized to unit trace unless ``normalize=False``."""
    rho = np.dot(T, T.conj().T)
    if normalize:
        rho = rho / np.trace(rho)
    return rho


def rho2T(rho):
    """Cholesky factor of rho. A non-physical rho (any eigenvalue <= 0) is first made positive
    by replacing its eigenvalues with their absolute values; this is only meant to give the MLE
    a valid starting point."""
    def _make_positive(rho_in):
        d, v = np.linalg.eig(rho_in)
        rho = np.zeros(rho_in.shape)
        for j in range(len(d)):
            rho = rho + np.abs(d[j]) * np.outer(v[:, j], v[:, j].conj().transpose())
        return (rho + rho.conj().transpose()) / 2.0

    def _psd_cholesky(rho, eps=1e-10):
        # a tiny multiple of the identity keeps the Cholesky decomposition numerically stable
        return np.linalg.cholesky(np.eye(rho.shape[0]) * eps + rho)

    if not np.all(np.linalg.eigvals(rho) > 0):
        return _psd_cholesky(_make_positive(rho))
    return np.linalg.cholesky(rho)


def mle_residuals(t, measured_stokes):
    """Residuals between the Stokes vector of the state t parameterizes and the measured one."""
    return compute_stokes(T2rho(t2T(t))) - measured_stokes


def mle_rho(rho, max_nfev=3000):
    """Least-squares maximum-likelihood state for a linear-inversion rho, as done in the
    notebooks: start from rho's Cholesky factor and fit the Stokes vector with
    ``mle_residuals``. ``project_rho`` gives the same state in closed form."""
    t0 = T2t(rho2T(rho))
    res = least_squares(mle_residuals, t0, args=(compute_stokes(rho),), max_nfev=max_nfev)
    return T2rho(t2T(res.x))


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


def project_rho(rho):
    """Closed form of ``mle_rho``: the fit minimizes the Stokes (= Frobenius) distance to a
    physical state, which is an eigenvalue projection onto the simplex. Agrees with mle_rho to
    ~1e-5 and is about 1000x faster."""
    w, v = np.linalg.eigh((rho + rho.conj().T) / 2)
    return (v * _project_eigs_to_simplex(w, 1.0)) @ v.conj().T
