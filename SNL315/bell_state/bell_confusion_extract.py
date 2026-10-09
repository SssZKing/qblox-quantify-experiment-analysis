"""Recover the per-run readout confusion matrix used for the Bell-state tomography in
Bell_state_0724_2.csv (SNL315_plot.ipynb section 5.1).

Each CSV row is rho_mle = MLE(linear_inversion(CM^-1 @ p_raw)), where CM was measured right
before the run (construct_confusion_matrix in SNL315.ipynb) but never saved. The linear-inversion
Stokes vector is linear in W = CM^-1, so W follows from a constrained linear least-squares fit
of the raw probabilities to the CSV Stokes vector, passed through the MLE step. Those equations
alone leave some directions of CM free (the single-qubit rows see only 3 settings), and the
MLE projection hides more, so CM is additionally required to be physical (entries in [0, 1],
columns summing to 1). With that constraint the solution comes out prior-independent (checked
per run with two different priors) and rebuilds the CSV rho with the notebook's own
calculate_density_matrix + MLE.

CM convention (same as calculate_density_matrix): CM[measured, prepared], basis |q0 q1> =
00, 01, 10, 11, with q0 = y0 = Q1 and q1 = y1 = Q2. Columns sum to 1.
"""
import csv
import os
import sys

import numpy as np
from scipy.optimize import least_squares

from snl_qblox.tomography_tools import (calculate_density_matrix, compute_stokes, mle_residuals,
                              rho2T, T2t, t2T, T2rho, _project_eigs_to_simplex)
from quantify_core.data.handling import set_datadir, load_dataset

# quantify data directory; set QBLOX_DATADIR to override (default: SNL315/CD1)
DATA_DIR = os.environ.get(
    "QBLOX_DATADIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CD1"))
set_datadir(DATA_DIR)

# CSVs live next to this script (bell_state/); relative names resolve here, not against the cwd.
HERE = os.path.dirname(os.path.abspath(__file__))


def data_path(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


MEAS_CSV = data_path("Bell_state_0724_2.csv")
# Distinct suffix: "<set>_confusion.csv" is the notebook's own saved (measured) CM file.
OUT_CSV = MEAS_CSV.replace(".csv", "_confusion_extracted.csv")

# Setting index feeding each Stokes parameter (IX..ZZ), and which outcome signs it uses; this
# mirrors the reindexing + sign rules inside calculate_density_matrix.
_SETTING = np.array([0, 1, 2, 3, 4, 5, 3, 6, 7, 8, 6, 2, 0, 1, 2])
_SIGNS = np.array([[1, -1, 1, -1]] * 3 + [[1, 1, -1, -1]] + [[1, -1, -1, 1]] * 3
                  + [[1, 1, -1, -1]] + [[1, -1, -1, 1]] * 3 + [[1, 1, -1, -1]]
                  + [[1, -1, -1, 1]] * 3, dtype=float)  # outcome order 00, 01, 10, 11


def raw_probs(tuid):
    """(9, 4) raw joint probabilities [P00, P01, P10, P11] per tomography setting."""
    ds = load_dataset(tuid)
    a = ds[list(ds.data_vars)[0]].values
    c = ds[list(ds.data_vars)[1]].values
    return np.stack([((1 - a) * (1 - c)).mean(0), ((1 - a) * c).mean(0),
                     (a * (1 - c)).mean(0), (a * c).mean(0)], axis=1)


def stokes_lin(W, P):
    """Linear-inversion Stokes vector for corrected probabilities W @ P[setting]."""
    Q = P @ W.T
    return np.einsum("jk,jk->j", _SIGNS, Q[_SETTING])


def rho_from_stokes(s):
    from snl_qblox.tomography_tools import (II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ)
    basis = [IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]
    return 0.25 * (II + sum(si * B for si, B in zip(s, basis)))


def mle_rho(rho):
    """Same MLE as the notebook (SNL315.ipynb, repeated Bell-state cell)."""
    t0 = T2t(rho2T(rho))
    res = least_squares(mle_residuals, t0, args=(compute_stokes(rho),), max_nfev=3000)
    return T2rho(t2T(res.x))


def project_rho(rho):
    """Closed form of mle_rho: the MLE minimises the Stokes (= Frobenius) distance to a
    physical state, i.e. an eigenvalue projection onto the simplex. Agrees with mle_rho
    to ~1e-5 and is ~1000x faster, so it is used inside the fit."""
    w, v = np.linalg.eigh((rho + rho.conj().T) / 2)
    return (v * _project_eigs_to_simplex(w, 1.0)) @ v.conj().T


def cm_to_params(CM):
    """12 free entries: rows 0..2 of each column (the 4th follows from the column sum)."""
    return CM[:3].ravel()


def params_to_cm(x):
    top = x.reshape(3, 4)
    return np.vstack([top, 1 - top.sum(0)])


M1 = lambda fg, fe: np.array([[fg, 1 - fe], [1 - fg, fe]])
PRIORS = {"0.99/0.90": np.kron(M1(0.99, 0.90), M1(0.99, 0.90)),
          "0.96/0.95": np.kron(M1(0.96, 0.95), M1(0.96, 0.95))}


def fit_cm(P, s_target, CM_prior):
    """Physical CM (entries in [0, 1], columns sum to 1) whose linear inversion + MLE
    reproduces s_target. A weak pull toward CM_prior only picks a solution when the data
    leave directions free; the result is checked to be prior-independent."""
    def resid(x):
        CM = params_to_cm(x)
        rho = project_rho(rho_from_stokes(stokes_lin(np.linalg.inv(CM), P)))
        return np.concatenate([(compute_stokes(rho) - s_target) / 1e-3,
                               0.3 * (x - cm_to_params(CM_prior)),
                               1e3 * np.minimum(CM[3], 0)])

    res = least_squares(resid, cm_to_params(CM_prior), bounds=(0, 1))
    return params_to_cm(res.x), np.abs(res.fun[:15]).max() * 1e-3


def marginal_fidelities(CM):
    """Per-qubit assignment fidelities from the 2-qubit CM (averaged over the other qubit)."""
    P = CM  # P[meas, prep], index = 2*q0 + q1
    out = {}
    for q, name in [(0, "Q1"), (1, "Q2")]:
        bit = lambda i: (i >> (1 - q)) & 1
        for s, lab in [(0, "g"), (1, "e")]:
            cols = [i for i in range(4) if bit(i) == s]
            rows = [i for i in range(4) if bit(i) == s]
            out[f"F{lab}_{name}"] = P[np.ix_(rows, cols)].sum(0).mean()
    return out


def load_rows(path):
    rows = []
    with open(data_path(path)) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            vals = [float(x) for x in row[1:]]
            rho = np.array([complex(vals[i], vals[i + 1])
                            for i in range(0, len(vals), 2)]).reshape(4, 4)
            rows.append((row[0], rho))
    return rows


def extract(tuid, rho_csv):
    P = raw_probs(tuid)
    s_csv = compute_stokes(rho_csv)
    fits = [fit_cm(P, s_csv, CM0) for CM0 in PRIORS.values()]
    CM, stokes_err = fits[0]
    prior_spread = np.abs(fits[0][0] - fits[1][0]).max()
    # check with the notebook's own functions (calculate_density_matrix + least-squares MLE)
    rho_chk = mle_rho(calculate_density_matrix(tuid, confusion_matrix=CM)[0])
    chk = np.abs(rho_chk - rho_csv).max()
    return CM, stokes_err, chk, prior_spread


if __name__ == "__main__":
    rows = load_rows(MEAS_CSV)
    if len(sys.argv) > 1:
        rows = rows[: int(sys.argv[1])]
    results = []
    for n, (tuid, rho_csv) in enumerate(rows):
        CM, err, chk, spread = extract(tuid, rho_csv)
        fid = marginal_fidelities(CM)
        results.append((tuid, CM, chk, spread, fid))
        print(f"{n:2d} {tuid}  |drho|={chk:.1e}  prior spread={spread:.3f}  diag="
              + np.array2string(np.diag(CM), precision=3) + "  "
              + "  ".join(f"{k}={v:.3f}" for k, v in fid.items()))

    CMs = np.array([r[1] for r in results])
    print("\nMean CM[measured, prepared] (cols = prepared 00, 01, 10, 11):")
    print(np.round(CMs.mean(0), 4))
    print("Std over runs:")
    print(np.round(CMs.std(0), 4))
    fids = {k: np.array([r[4][k] for r in results]) for k in results[0][4]}
    print("Per-qubit assignment fidelity (mean ± std):",
          ", ".join(f"{k}={v.mean():.4f}±{v.std():.4f}" for k, v in fids.items()))

    # Never overwrite a measured CM file: only replace a file this script wrote itself.
    try:
        with open(OUT_CSV) as f:
            if "max_cm_prior_spread" not in f.readline():
                raise SystemExit(f"{OUT_CSV} exists and was not written by this script; not overwriting.")
    except FileNotFoundError:
        pass
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tuid"] + [f"cm_{i}{j}" for i in range(4) for j in range(4)]
                   + ["max_abs_rho_mismatch", "max_cm_prior_spread"] + list(fids))
        for tuid, CM, chk, spread, fid in results:
            w.writerow([tuid] + list(CM.ravel()) + [chk, spread] + list(fid.values()))
    print(f"\nSaved {OUT_CSV}")
