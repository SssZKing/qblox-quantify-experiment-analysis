"""Two-qubit quantum process tomography (QPT).

Linear inversion, maximum-likelihood CPTP reconstruction (with and without SPAM correction),
projection onto physical processes, and fidelities.

Conventions
-----------
- Basis |Q1 Q2> = 00, 01, 10, 11, with Q1 = dataset ``y0`` = first Kronecker factor.
- The six single-qubit tomography rotations are, in order, [I, X180, Y90, mY90, mX90, X90]
  (``_tomo_rotations``), the schedules' ``tomo_sequence`` [0, 3, 21, 15, 13, 16].
- ``p00`` is the measured probability of outcome 00 per setting, laid out as
  (prep_Q1, prep_Q2, meas_Q1, meas_Q2), i.e. shape (6, 6, 6, 6) or flattened to 1296.
- chi matrices are qutip ``superrep='chi'`` Qobjs normalized to Tr = 16 (d^2). qutip 5's chi
  is that of the TRANSPOSED Kraus operators: to_chi(to_super(U)) = 16 c(U^T) c(U^T)^dag.
- Superoperators are column-stacked (vec(A X B) = (B^T (x) A) vec(X)), as in qutip.
"""
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.linalg import fractional_matrix_power, solve, sqrtm
from scipy.optimize import minimize

from ._paulis import PAULI_2Q, Y_PARITY
from .state import _project_eigs_to_simplex

N_ROT = 6
N_SETTINGS = N_ROT ** 4
CHI_DIMS = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]
_CHOI_DIMS = CHI_DIMS
_Y_PARITY = Y_PARITY

op_label = [["$I$", "$X$", "$Y$", "$Z$"] for i in range(2)]


##################################################
### Linear inversion
##################################################

# Linear-inversion matrix: the data vector built by map_from_bloch_state_to_pauli_basis2 equals
# PMatrix @ chi.ravel() / 16. Entries are 0, +-4 and +-4j (stored as complex64); this is the
# former PMatrix2.pkl, saved as plain .npy so loading it cannot execute code.
PMATRIX_PATH = Path(__file__).with_name("pmatrix.npy")
PMatrix = np.load(PMATRIX_PATH, allow_pickle=False)

# Single-qubit inversion operators written in the six prepared states (_tomo_rotations applied
# to |0>): row a holds the coefficients of operator a over the six preparations.
_C1 = np.array([
    [1, 1, 0, 0, 0, 0],
    [-(1 + 1j), -(1 + 1j), 2, 0, 1j, 1j],
    [0, 0, 0, 0, 1, -1],
    [1, -1, 0, 0, 0, 0],
])
# Two-qubit operator i = 4 a + b is the product of single-qubit operators a (Q1) and b (Q2), so
# c2[i, j, k] = _C1[a, j] * _C1[b, k] for preparation j on Q1 and k on Q2.
_C2 = np.einsum("aj,bk->abjk", _C1, _C1).reshape(16, N_ROT, N_ROT)


def func_c2(i: int, j: int, k: int):
    """Constants c2[i, j, k] such that
    E2[i] = sum_{j,k} c2[i,j,k] (F2[j] x F2[k]) |0>|0><0|<0| (F2[j]^dag x F2[k]^dag),
    i.e. inversion operator i in terms of the 36 prepared product states. Zero outside
    0 <= i < 16, 0 <= j, k < 6."""
    if 0 <= i < 16 and 0 <= j < N_ROT and 0 <= k < N_ROT:
        return _C2[i, j, k]
    return 0


def map_from_bloch_state_to_pauli_basis2(q, n, arr):
    """Element (q, n) of the linear-inversion data: sum_{ijkl} c2[q,i,j] conj(c2[n,k,l]) arr[i,j,k,l]
    for arr = p00 with shape (6, 6, 6, 6)."""
    arr = np.asarray(arr)
    if arr.shape != (6, 6, 6, 6):
        raise ValueError("Input array must be 6x6x6x6")
    if (q not in range(0, 16)) or (n not in range(0, 16)):
        raise ValueError("Input indices must be between 0 and 15, inclusive")
    return _C2[q].ravel() @ arr.reshape(36, 36) @ _C2[n].ravel().conj()


def _inversion_data(p00):
    """All 256 elements of map_from_bloch_state_to_pauli_basis2 at once, index 16 q + n."""
    pr = np.asarray(p00, dtype=float).reshape(36, 36)
    A = _C2.reshape(16, 36)
    return (A @ pr @ A.conj().T).ravel()


def chi_from_p00_linear(p00):
    """Linear-inversion chi as a qutip Qobj (Tr = 16): solve(PMatrix, data) on p00.

    Note this is the textbook chi, which qutip reads as the Kraus-transposed channel (see the
    module docstring); _linear_init converts it for the MLE."""
    mv = _inversion_data(p00)
    return qt.Qobj(16 * solve(PMatrix, mv).reshape(16, 16), dims=CHI_DIMS, superrep='chi')


def as_chi(M):
    """Wrap a 16x16 array / Qobj as a chi-superrep Qobj."""
    A = M.full() if isinstance(M, qt.Qobj) else np.asarray(M)
    return qt.Qobj(A, dims=CHI_DIMS, superrep='chi')


##################################################
### SPAM correction and physical projection
##################################################

def get_spam_corrected_gate(measured_gate_ptm, spam_error_superoperator, p=0.5):
    """SPAM-corrected gate superoperator: R_corrected = E^-(1-p) R_meas E^-p.

    Parameters
    ----------
    measured_gate_ptm : the uncorrected superoperator (or PTM) of the measured gate.
    spam_error_superoperator : E_SPAM, the same representation of the measured identity
        (with no gate, the measured process is pure SPAM).
    p : fraction of the SPAM error attributed to state preparation (default 1/2);
        measurement takes 1 - p.

    The inversion generally breaks complete positivity; mle_chi_spam_corrected avoids that.
    """
    E_spam = np.array(spam_error_superoperator)
    R_meas = np.array(measured_gate_ptm)

    # P = E^p (preparation) and M = E^(1-p) (measurement)
    P_inv = fractional_matrix_power(E_spam, -p)
    M_inv = fractional_matrix_power(E_spam, -(1 - p))
    R_corrected = M_inv @ R_meas @ P_inv

    # fractional_matrix_power can leave negligible imaginary parts on real input
    if np.isrealobj(E_spam) and np.isrealobj(R_meas):
        R_corrected = np.real(R_corrected)
    return R_corrected


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
# matrix MLE (rho = T T^dag -> here Choi J = T T^dag), so positivity holds by construction
# rather than by clipping, and trace preservation is imposed exactly.

_FORWARD_CACHE = {}


def _tomo_rotations():
    """The 6 single-qubit tomography rotations, in the schedules' tomo_sequence order
    [I, X180, Y90, mY90, mX90, X90] (indices [0, 3, 21, 15, 13, 16])."""
    def Rx(t):
        return np.array([[np.cos(t/2), -1j*np.sin(t/2)], [-1j*np.sin(t/2), np.cos(t/2)]])

    def Ry(t):
        return np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])

    return [np.eye(2), Rx(np.pi), Ry(np.pi/2), Ry(-np.pi/2), Rx(-np.pi/2), Rx(np.pi/2)]


def readout_povm(readout):
    """P(read 00 | basis state 00, 01, 10, 11) from measured readout fidelities
    [[F_g, F_e] of Q1, [F_g, F_e] of Q2], or the average over a stack of runs with shape
    (n_runs, 2, 2) (exact for pooled data: p00 is linear in it)."""
    readout = np.asarray(readout, float)
    if readout.ndim == 3:
        return np.mean([readout_povm(r) for r in readout], axis=0)

    def read0(f_g, f_e):
        return np.array([f_g, 1 - f_e])

    return np.kron(read0(*readout[0]), read0(*readout[1]))


def forward_matrix(readout=None):
    """Cached (1296, 256) matrix BC with  p00 = real(BC @ choi.ravel()).

    Encodes p00[i,j,k,l] = Tr[E_kl Lambda(rho_ij)] for
        rho_ij = (F_i (x) F_j) |00><00| (F_i (x) F_j)^dag        (prep)
        E_kl   = (F_k (x) F_l)^dag M00 (F_k (x) F_l)             ("00" effect)
    with M00 = |00><00| for ideal readout (``readout=None``), or diag(readout_povm(readout))
    so that measured readout error does not have to be absorbed by the process. Lambda is the
    channel qutip itself associates with the Choi matrix (qt.to_super of it, column-stacked:
    vec(Lambda(rho)) = S vec(rho), and Tr[E X] = vec(E^T) . vec(X)).

    Built from qutip's own Choi -> superoperator map, so it cannot disagree with to_chi /
    to_choi / process_fidelity. Pushing qutip's chi through the textbook formula
    Lambda(rho) = sum chi_mn P_m rho P_n instead returns the channel with every Kraus operator
    transposed (see the module docstring): invisible for symmetric gates (iSWAP, diagonal
    phases) but wrong for e.g. Z(Q1).iSWAP, and it turns the TP constraint into a unitality
    constraint on the true process. Checked: a Lindblad simulation of a non-unital, asymmetric
    gate is reproduced to 2e-16.

    Built lazily and cached per readout: assembling it costs ~2 s.
    """
    M00 = np.array([1.0, 0.0, 0.0, 0.0]) if readout is None else readout_povm(readout)
    key = M00.tobytes()
    if key not in _FORWARD_CACHE:
        F = _tomo_rotations()
        rho00 = np.zeros((4, 4)); rho00[0, 0] = 1.0
        Fs = [np.kron(F[a], F[b]) for a in range(N_ROT) for b in range(N_ROT)]
        R = np.array([(Fab @ rho00 @ Fab.conj().T).ravel(order='F') for Fab in Fs]).T          # (16, 36)
        E = np.array([(Fab.conj().T @ np.diag(M00) @ Fab).T.ravel(order='F') for Fab in Fs])   # (36, 16)

        BC = np.empty((N_SETTINGS, 256), dtype=complex)
        for k in range(256):
            e = np.zeros(256); e[k] = 1.0
            S = qt.to_super(qt.Qobj(e.reshape(16, 16), dims=CHI_DIMS, superrep='choi')).full()
            BC[:, k] = (E @ S @ R).T.ravel()          # row = prep * 36 + meas
        _FORWARD_CACHE[key] = BC
    return _FORWARD_CACHE[key]


def _mle_forward_matrix():
    """forward_matrix() with ideal readout (kept for existing callers)."""
    return forward_matrix()


def _linear_init(p_obs):
    """Linear-inversion starting point for the MLE, in qutip's chi convention.

    solve(PMatrix, ...) assumes the textbook chi, which qutip reads as the Kraus-transposed
    channel; D chi D maps it back (see forward_matrix)."""
    chi = chi_from_p00_linear(p_obs).full()
    return qt.Qobj(_Y_PARITY[:, None] * chi * _Y_PARITY[None, :], dims=CHI_DIMS, superrep='chi')


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


def _check_p00(p00):
    p_obs = np.asarray(p00, dtype=float).ravel()
    if p_obs.size != N_SETTINGS:
        raise ValueError(f"p00 must have {N_SETTINGS} entries (6^4), got {p_obs.size}.")
    return p_obs


def _mle_fit(L, p_obs, n_shots, chi_init, maxiter, verbose):
    """Maximize the binomial log-likelihood of p_obs under p = real(L @ vec(J)) over CPTP J."""
    n1 = p_obs * n_shots
    n0 = (1.0 - p_obs) * n_shots

    if chi_init is None:
        chi_init = _linear_init(p_obs)
    t0 = _t_from_choi(qt.to_choi(chi_init).full())

    def nll(t):
        p = np.real(L @ _choi_from_t(t).ravel())
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -np.sum(n1 * np.log(p) + n0 * np.log1p(-p))

    res = minimize(nll, t0, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 10**7, "disp": verbose})
    chi = qt.to_chi(qt.Qobj(_choi_from_t(res.x), dims=CHI_DIMS, superrep="choi"))
    return chi, res


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
        Measured P(outcome = 00) per (prep, meas) setting.
    n_shots : int
        TOTAL shots behind each setting. When p00 is averaged over several datasets this is
        the sum over them (e.g. 10 datasets x 400 shots -> 4000), not the per-dataset count;
        it sets the weight of the likelihood against the CPTP constraint.
    chi_init : qutip Qobj, optional
        Starting point. Defaults to the linear-inversion estimate, projected onto the
        physical set to give the Cholesky factorization something valid to start from.
    forward : (1296, 256) array, optional
        Forward matrix to fit with instead of forward_matrix(), e.g. forward_matrix(readout)
        to include the measured readout error. mle_chi_spam_corrected takes the same.

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
    BC = forward_matrix() if forward is None else forward
    p_obs = _check_p00(p00)
    chi, res = _mle_fit(BC, p_obs, n_shots, chi_init, maxiter, verbose)
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
    BC = forward_matrix() if forward is None else forward
    E = qt.to_super(ref_chi).full()
    A = fractional_matrix_power(E, 1.0 - p)
    B = fractional_matrix_power(E, p)

    L = np.empty((N_SETTINGS, 256), dtype=complex)
    for k in range(256):
        e = np.zeros(256); e[k] = 1.0
        J_basis = qt.Qobj(e.reshape(16, 16), dims=CHI_DIMS, superrep='choi')
        R_meas = A @ qt.to_super(J_basis).full() @ B
        J_meas = qt.to_choi(qt.Qobj(R_meas, dims=CHI_DIMS, superrep='super')).full()
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
    p_obs = _check_p00(p00)
    chi, res = _mle_fit(L, p_obs, n_shots, chi_init, maxiter, verbose)
    return (chi, res) if return_result else chi


##################################################
### Unitaries and fidelities
##################################################

def closest_unitary(chi):
    """Nearest EXACTLY-unitary matrix to a (possibly non-physical) chi: dominant Choi
    eigenvector, then SVD polar projection (U,_,Vh = svd(K); U@Vh) forces ||U^dag U - I||=0,
    discarding any magnitude/non-unitarity information in the raw reconstruction.
    Returns (U as a Qobj, Choi eigenvalues)."""
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
    """c_P = Tr(P^dag U)/4 over the two-qubit Paulis, in qutip's chi ordering."""
    M = U.full() if isinstance(U, qt.Qobj) else np.asarray(U)
    return np.array([np.trace(P.conj().T @ M)/4 for P in PAULI_2Q])


def proc_fid_to_unitary(chi, U):
    """Process (entanglement) fidelity of measured chi to unitary U: <c| chi_norm |c>.

    qutip's chi of U is 16 c(U^T) c(U^T)^dag (see the module docstring), so the overlap is
    taken with c(U^T); this equals qt.process_fidelity(chi, U). Using c(U) instead measures
    the fidelity to U^T, which differs whenever U is not symmetric."""
    M = as_chi(chi).full(); M = (M + M.conj().T)/2; M = M/np.trace(M).real
    c = _Y_PARITY * unitary_pauli_coeffs(U)
    return float(np.real(np.vdot(c, M @ c)))


def QPT_fidelity(chi_measured, chi_ideal):
    """Process and average gate fidelity from the Uhlmann formula
        F_pro = | Tr( sqrt( sqrt(chi_ideal) chi_measured sqrt(chi_ideal) ) ) |^2
        F_avg = (d F_pro + 1) / (d + 1).
    Both chi matrices must be normalized to unit trace (qutip's chi divided by 16).
    Returns (F_pro, F_avg)."""
    d = int(np.sqrt(chi_measured.shape[0]))   # chi is d^2 x d^2
    sqrt_chi_g = sqrtm(chi_ideal)
    sqrt_inner = sqrtm(sqrt_chi_g @ chi_measured @ sqrt_chi_g)
    F_pro = np.abs(np.trace(sqrt_inner))**2
    F_avg = (d * F_pro + 1) / (d + 1)
    return F_pro, F_avg


def avg_gate_fidelity(F_pro, d=4):
    """Average gate fidelity from process fidelity, (d F_pro + 1) / (d + 1)."""
    return (d * F_pro + 1) / (d + 1)


def ptm(chi):
    """16x16 Pauli transfer matrix R_ij = Tr[P_i Lambda(P_j)] / 4 (real), rows and columns in
    PAULI_LABELS order. Takes a Qobj in any superrep, or a 16x16 chi array."""
    S = qt.to_super(chi if isinstance(chi, qt.Qobj) else as_chi(chi)).full()
    out = np.array([(S @ P.ravel(order="F")).reshape(4, 4, order="F") for P in PAULI_2Q])
    return np.real(np.einsum("iab,jba->ij", np.array(PAULI_2Q), out)) / 4


def fit_chi2(chi, p00, n_shots, forward=None):
    """Mean squared residual of a reconstruction against the measured p00, in units of
    binomial shot noise (1 = the process describes the data to within shot noise)."""
    BC = forward_matrix() if forward is None else forward
    pred = np.real(BC @ qt.to_choi(chi).full().ravel())
    p = np.asarray(p00, float).ravel()
    return float(np.mean((pred - p) ** 2 / np.clip(p * (1 - p), 1e-4, None) * n_shots))


def reconstruct(p00, ref_p00, n_shots, raw=True, forward=None, p=0.5):
    """Standard reconstruction of a gate measured with an identity reference:
    MLE of the reference (the SPAM map E), MLE of the SPAM-corrected gate against the raw
    counts, optionally the uncorrected MLE of the gate, and the closest unitary of the
    corrected gate.

    Parameters
    ----------
    p00, ref_p00 : (1296,) or (6,6,6,6) measured P(00) of the gate and of the reference.
    n_shots : total shots behind each setting.
    raw : also return the uncorrected MLE under "raw".
    forward : forward matrix for all fits; None is ideal readout, forward_matrix(readout)
        folds the measured readout in.
    p : SPAM split, as in mle_chi_spam_corrected.

    Returns
    -------
    dict with chi Qobjs (Tr = 16) "ref", "corr", optionally "raw", and the unitary "U_fit".
    """
    out = {"ref": mle_chi_from_p00(ref_p00, n_shots, forward=forward)}
    out["corr"] = mle_chi_spam_corrected(p00, out["ref"], n_shots, p=p, forward=forward)
    if raw:
        out["raw"] = mle_chi_from_p00(p00, n_shots, forward=forward)
    out["U_fit"] = closest_unitary(out["corr"])[0]
    return out
