"""The 5-parameter iSWAP model and extraction of its angles from a reconstructed process."""
import numpy as np
import qutip as qt
from scipy.optimize import least_squares

from .process import as_chi, proc_fid_to_unitary


def _wrap(x):
    """Wrap an angle (rad) into [-pi, pi)."""
    return (x + np.pi) % (2*np.pi) - np.pi


def iSWAP_RPE(a):
    """5-parameter iSWAP (paper Eq. 8): a = [theta_p, phi_p, theta_1, theta_2, phi_zz]."""
    tp, pp, t1, t2, pzz = a
    c, s = np.cos(tp), np.sin(tp)
    U = np.array([[1, 0, 0, 0],
        [0, np.exp(1j*t2)*c,        1j*np.exp(1j*(t2+pp))*s, 0],
        [0, 1j*np.exp(1j*(t1-pp))*s, np.exp(1j*t1)*c,        0],
        [0, 0, 0, np.exp(1j*(t1+t2+pzz))]], dtype=complex)
    return qt.Qobj(U, dims=[[2, 2], [2, 2]])


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

    Returns a dict of angles in radians (see ``deg``).
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
    it does not constrain the fit itself.
    Returns (angles dict, process fidelity of chi to the fitted U, least_squares result)."""
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
    """Convert a dict of angles in radians (e.g. from extract_angles) to degrees; None -> NaN."""
    return {k: (np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else np.rad2deg(v))
            for k, v in d.items()}
