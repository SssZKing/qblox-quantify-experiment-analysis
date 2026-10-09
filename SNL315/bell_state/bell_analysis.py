"""Analysis layer behind bell_figures.ipynb: load Bell-state tomography sets, pool them, fit a
physical error model, and simulate the fidelity estimators.

Conventions (same as calculate_density_matrix / bell_confusion_extract):
  basis |Q1 Q2> = 00, 01, 10, 11 (Q1 = q0 = dataset y0, Q2 = q1 = y1)
  CM[measured, prepared], columns sum to 1
  target |Psi+> = (|01> + |10>)/sqrt2, so F = <Psi+|rho|Psi+> = (1 + <XX> + <YY> - <ZZ>)/4

Confusion matrices: a set's measured CMs are read from "<set>_confusion.csv" (written by the
SNL315.ipynb loop) and never modified. Sets measured before CMs were saved fall back to CMs
recovered by bell_confusion_extract.fit_cm, cached in "<set>_confusion_extracted.csv".
"""
import csv
import glob
import os
import sys
from datetime import datetime

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bell_confusion_extract as bce
from snl_qblox.tomography_tools import compute_stokes, _project_eigs_to_simplex
from snl_qblox.pycqed_randomized_benchmarking.clifford_group import TwoQubitClifford
from quantify_core.data.handling import load_dataset, load_snapshot

IDEAL = np.array([[0, 0, 0, 0], [0, .5, .5, 0], [0, .5, .5, 0], [0, 0, 0, 0]], dtype=complex)
TOMO_SEQUENCE = [360, 384, 0, 15, 375, 399, 16, 376, 400]
# index_to_operation: 16 ns operation_buffer_time before a single-qubit gate; the iSWAP pump RF
# starts 26 ns after the previous op (rel_time = 26e-9 - iSWAP_DELAY, marker leads by iSWAP_DELAY).
T_BUF_1Q, T_BUF_2Q = 16e-9, 26e-9

# Confusion-matrix shots per column (construct_confusion_matrix repetitions); not stored in
# any dataset, so taken from the notebook loop that produced each set.
CM_SHOTS = {"0723": 500, "0724": 500, "0923": 500, "0924_0": 2000, "0924_1": 2000, "0924_2": 2000}
DEFAULT_CM_SHOTS = 2000   # current notebook loop (construct_confusion_matrix repetitions=2000)


# ----------------------------------------------------------------------------- basics
def fidelity(rho):
    return float(np.real(np.trace(IDEAL @ rho)))


def lin_rho(P, CM):
    """CM-corrected linear-inversion rho (identical to calculate_density_matrix)."""
    return bce.rho_from_stokes(bce.stokes_lin(np.linalg.inv(CM), P))


def mle(rho):
    """Closed form of the notebook's least-squares MLE (agrees to ~1e-5)."""
    return bce.project_rho(rho)


def tuid_time(tuid):
    return datetime.strptime(tuid[:15], "%Y%m%d-%H%M%S")


def list_sets():
    names = [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(HERE, "Bell_state_*.csv"))]
    return sorted(n for n in names if "_confusion" not in n)


def short(name):
    return name.replace("Bell_state_", "")


def cm_shots(name):
    s = short(name)
    return CM_SHOTS.get(s, CM_SHOTS.get(s.split("_")[0], DEFAULT_CM_SHOTS))


# ----------------------------------------------------------------------------- CM files
def _read_cm_file(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return {r[0]: np.array(r[1:17], float).reshape(4, 4) for r in rows[1:]}


def _write_extracted_cache(path, tuids, cms, errs):
    """Write recovered CMs; refuses to touch any file this module/extractor did not write."""
    if os.path.exists(path):
        with open(path) as f:
            if "max_cm_prior_spread" not in f.readline():
                raise RuntimeError(f"{path} was not written by the extractor; not overwriting.")
    fids = [bce.marginal_fidelities(c) for c in cms]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tuid"] + [f"cm_{i}{j}" for i in range(4) for j in range(4)]
                   + ["max_abs_rho_mismatch", "max_cm_prior_spread"] + list(fids[0]))
        for t, c, e, fd in zip(tuids, cms, errs, fids):
            w.writerow([t] + list(c.ravel()) + [e, float("nan")] + list(fd.values()))


def extracted_cms(name, tuids, P, rho_csv, verbose=True):
    """Recovered CMs for a set, from the cache when complete, otherwise fitted and cached."""
    path = os.path.join(HERE, f"{name}_confusion_extracted.csv")
    if os.path.exists(path):
        cache = _read_cm_file(path)
        if all(t in cache for t in tuids):
            return np.array([cache[t] for t in tuids])
    if verbose:
        print(f"  recovering {len(tuids)} confusion matrices for {short(name)} (cached afterwards)...")
    cms, errs = [], []
    for Pk, r in zip(P, rho_csv):
        c, e = bce.fit_cm(Pk, compute_stokes(r), bce.PRIORS["0.99/0.90"])
        cms.append(c)
        errs.append(e)
    _write_extracted_cache(path, tuids, cms, errs)
    return np.array(cms)


# ----------------------------------------------------------------------------- sets
class BellSet:
    """One repeated Bell-tomography set: raw probabilities, CMs and the notebook's MLE rho."""

    def __init__(self, name, verbose=True):
        self.name = name
        self.label = short(name)
        rows = bce.load_rows(name + ".csv")
        self.tuids = [t for t, _ in rows]
        self.rho_csv = np.array([r for _, r in rows])
        self.times = [tuid_time(t) for t in self.tuids]
        self.P = np.array([bce.raw_probs(t) for t in self.tuids])
        self.n_tomo = int(load_dataset(self.tuids[0]).sizes["repetition"])
        self.n_cm = cm_shots(name)
        saved = os.path.join(HERE, f"{name}_confusion.csv")
        if os.path.exists(saved):
            cache = _read_cm_file(saved)
            self.CM = np.array([cache[t] for t in self.tuids])
            self.cm_source = "saved"
        else:
            self.CM = extracted_cms(name, self.tuids, self.P, self.rho_csv, verbose)
            self.cm_source = "extracted"
        self.lin = np.array([lin_rho(p, c) for p, c in zip(self.P, self.CM)])

    def __len__(self):
        return len(self.tuids)

    # --- estimators
    def per_run_F(self):
        """Notebook method: MLE each run, then F (averaging these is biased low)."""
        return np.array([fidelity(mle(r)) for r in self.lin])

    def per_run_F_linear(self):
        return np.array([fidelity(r) for r in self.lin])

    def pooled_rho(self, idx=None):
        L = self.lin if idx is None else self.lin[idx]
        return mle(L.mean(0))

    def pooled_F(self, n_boot=500, seed=0):
        """Average the CM-corrected linear-inversion rho, one MLE, bootstrap over runs."""
        rng = np.random.default_rng(seed)
        n = len(self)
        boot = [fidelity(self.pooled_rho(rng.integers(0, n, n))) for _ in range(n_boot)]
        return fidelity(self.pooled_rho()), float(np.std(boot))

    # --- per-run observables
    def observables(self):
        """Per-run Bell phase / imbalance / coherence (after MLE) and readout marginals."""
        out = {k: [] for k in ["F", "phase_deg", "imbalance", "coherence",
                               "Fg_Q1", "Fe_Q1", "Fg_Q2", "Fe_Q2"]}
        for r, c in zip(self.lin, self.CM):
            rm = mle(r)
            out["F"].append(fidelity(rm))
            out["phase_deg"].append(np.degrees(np.angle(rm[1, 2])))
            out["imbalance"].append(np.real(rm[2, 2] - rm[1, 1]))
            out["coherence"].append(np.abs(rm[1, 2]))
            for k, v in bce.marginal_fidelities(c).items():
                out[k].append(v)
        return {k: np.array(v) for k, v in out.items()}

    def shot_noise_sd(self, n_boot=300, seed=0):
        """Run-to-run sd expected from shot noise alone (tomography + CM shots)."""
        rng = np.random.default_rng(seed)
        Pm, Cm = self.P.mean(0), self.CM.mean(0)
        vals = []
        for _ in range(n_boot):
            Pb = np.array([rng.multinomial(self.n_tomo, p / p.sum()) / self.n_tomo for p in Pm])
            Cb = np.array([rng.multinomial(self.n_cm, c / c.sum()) / self.n_cm for c in Cm.T]).T
            rm = mle(lin_rho(Pb, Cb))
            mf = bce.marginal_fidelities(Cb)
            vals.append(dict(F=fidelity(rm), phase_deg=np.degrees(np.angle(rm[1, 2])),
                             imbalance=np.real(rm[2, 2] - rm[1, 1]), coherence=np.abs(rm[1, 2]), **mf))
        return {k: float(np.std([v[k] for v in vals])) for k in vals[0]}


# ----------------------------------------------------------------------------- snapshot
def snapshot_durations(tuid):
    """(single-qubit rxy duration, sqrt_iSWAP pulse duration) in s."""
    inst = load_snapshot(tuid)["instruments"]
    d = [inst[q]["submodules"]["rxy"]["parameters"]["duration"]["value"] for q in ("qubit1", "qubit2")]
    if not np.isclose(d[0], d[1], rtol=1e-6):
        raise ValueError(f"qubit1/qubit2 rxy durations differ: {d}")
    return d[0], inst["qubit1_qubit2"]["submodules"]["iswap"]["parameters"]["pulse_duration"]["value"]


# ----------------------------------------------------------------------------- readout calibration
def latest_readout_calibration(before_tuid, datadir=None):
    """tuid of the last 'Multiplexed Readout Calibration' saved before ``before_tuid`` (same day)."""
    day = os.path.join(datadir or bce.DATA_DIR, before_tuid[:8])
    cands = sorted(d[:26] for d in os.listdir(day)
                   if "Multiplexed Readout Calibration" in d and d[:26] < before_tuid)
    if not cands:
        raise FileNotFoundError(f"no multiplexed readout calibration before {before_tuid}")
    return cands[-1]


def readout_from_iq(cal_tuid, classify_tuid):
    """Measured single-shot (F_g, F_e) of Q1 and Q2 from the IQ shots of a Multiplexed Readout
    Calibration, classified with the acq_rotation / acq_threshold in force for ``classify_tuid``
    (e.g. the first run of a Bell set). Qblox thresholded acquisition: rotate I + iQ by
    exp(+i rot) and call the shot excited if the real part >= threshold."""
    ds = load_dataset(cal_tuid)
    st = ds.x0.values
    inst = load_snapshot(classify_tuid)["instruments"]
    out = []
    for q, (I, Q) in zip(("qubit1", "qubit2"), ((ds.y0, ds.y1), (ds.y2, ds.y3))):
        m = inst[q]["submodules"]["measure"]["parameters"]
        rot, thr = m["acq_rotation"]["value"], m["acq_threshold"]["value"]
        exc = ((I.values + 1j * Q.values) * np.exp(1j * np.radians(rot))).real >= thr
        out.append((float(1 - exc[st == 0].mean()), float(exc[st == 1].mean())))
    return out


# ----------------------------------------------------------------------------- error model
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1, -1]).astype(complex)
_SM = np.array([[0, 1], [0, 0]], complex)   # |0><1|, |0> = ground
_I2, _I4 = np.eye(2), np.eye(4)
_AX = {"X": _X, "Y": _Y}
GATES = {"I": None, "X180": ("X", np.pi), "X90": ("X", np.pi / 2), "mX90": ("X", -np.pi / 2),
         "Y90": ("Y", np.pi / 2), "mY90": ("Y", -np.pi / 2), "Y180": ("Y", np.pi)}
PARAMS = ["Fg_Q1", "Fe_Q1", "Fg_Q2", "Fe_Q2", "pth_Q1", "pth_Q2", "eps_Xpi", "eps_swap",
          "phi_Q1", "phi_Q2", "eps_tomo_Q1", "eps_tomo_Q2"]
_P0 = np.array([0.99, 0.90, 0.99, 0.90, 0, 0, 0, 0, 0, 0, 0, 0.])
_LB = np.array([0.8, 0.7, 0.8, 0.7, 0, 0, -.3, -.3, -np.pi, -np.pi, -.3, -.3])
_UB = np.array([1, 1, 1, 1, .2, .2, .3, .3, np.pi, np.pi, .3, .3])
IDEAL_PARAMS = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0.])


def _on(A, q):
    return np.kron(A, _I2) if q == 0 else np.kron(_I2, A)


def _tomo_gates(idx):
    dec = dict((qs[0], g[0]) for qs, g in TwoQubitClifford(idx).gate_decomposition())
    return dec["q0"], dec["q1"]


class ErrorModel:
    """Lindblad model of the Bell sequence + tomography + readout.

    Sequence: 16 ns | X180 on Q1 | 26 ns | sqrt_iSWAP (XY exchange, g t = pi/4) | virtual Z
    that makes the Bell coherence real | 16 ns | tomography pulse pair | readout.
    Parameters (PARAMS): per-qubit readout F_g/F_e, thermal population, X180 amplitude error,
    sqrt_iSWAP angle error, Z phase on each qubit, tomography-pulse amplitude errors.
    """

    def __init__(self, T1, T2, t_1q, t_2q):
        self.T1, self.T2, self.t_1q, self.t_2q = list(T1), list(T2), t_1q, t_2q
        cs = []
        for q in range(2):
            g1 = 1 / self.T1[q]
            gp = 1 / self.T2[q] - g1 / 2
            cs.append(np.sqrt(g1) * _on(_SM, q))
            if gp > 0:
                cs.append(np.sqrt(gp / 2) * _on(_Z, q))
        self.c_ops = cs
        self.HXY = (np.pi / 4) / t_2q * (_on(_SM.conj().T, 0) @ _on(_SM, 1) + _on(_SM, 0) @ _on(_SM.conj().T, 1))
        psi = expm(-1j * self.HXY * t_2q) @ np.array([0, 0, 1, 0], complex)   # after X180 on Q1
        self.VZ = _on(np.diag([1, np.exp(1j * np.angle(psi[1] * np.conj(psi[2])))]), 0)
        self.TG = [_tomo_gates(i) for i in TOMO_SEQUENCE]
        self._cache = {}

    def _prop(self, H, t, decoh=True):
        key = (H.tobytes(), t, decoh)
        if key not in self._cache:
            L = -1j * (np.kron(_I4, H) - np.kron(H.T, _I4))
            for C in (self.c_ops if decoh else []):
                CdC = C.conj().T @ C
                L = L + np.kron(C.conj(), C) - 0.5 * np.kron(_I4, CdC) - 0.5 * np.kron(CdC.T, _I4)
            self._cache[key] = expm(L * t)
            if len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def _step(self, r, H, t, decoh=True):
        return (self._prop(H, t, decoh) @ r.reshape(-1, order="F")).reshape(4, 4, order="F")

    def _H1(self, gates, eps):
        H = np.zeros((4, 4), complex)
        for q, g in enumerate(gates):
            s = GATES[g]
            if s:
                H += s[1] * (1 + eps[q]) / self.t_1q / 2 * _on(_AX[s[0]], q)
        return H

    def state(self, p, decoh=True):
        """Bell state right before the tomography pulses."""
        _, _, _, _, p0, p1, ex, dsw, f0, f1, _, _ = p
        r = np.kron(np.diag([1 - p0, p0]), np.diag([1 - p1, p1])).astype(complex)
        r = self._step(r, np.zeros((4, 4)), T_BUF_1Q, decoh)
        r = self._step(r, self._H1(("X180", "I"), (ex, 0)), self.t_1q, decoh)
        r = self._step(r, np.zeros((4, 4)), T_BUF_2Q, decoh)
        r = self._step(r, self.HXY * (1 + dsw), self.t_2q, decoh)
        U = self.VZ @ np.kron(np.diag([1, np.exp(1j * f0)]), np.diag([1, np.exp(1j * f1)]))
        return U @ r @ U.conj().T

    @staticmethod
    def readout_matrix(p):
        M = lambda g, e: np.array([[g, 1 - e], [1 - g, e]])
        return np.kron(M(p[0], p[1]), M(p[2], p[3]))

    def probs(self, p, decoh=True):
        """(9, 4) measured joint probabilities per tomography setting."""
        r = self.state(p, decoh)
        CM = self.readout_matrix(p)
        out = []
        for g in self.TG:
            rr = self._step(r, np.zeros((4, 4)), T_BUF_1Q, decoh)
            rr = self._step(rr, self._H1(g, (p[10], p[11])), self.t_1q, decoh)
            out.append(CM @ np.real(np.diag(rr)))
        return np.array(out)

    def notebook_style_cm(self, p):
        """CM as construct_confusion_matrix measures it: prep X180(s) + padded identity + readout,
        so it contains readout error AND the decay / X180 error before readout."""
        cols = []
        for prep in [("I", "I"), ("I", "X180"), ("X180", "I"), ("X180", "X180")]:
            r = np.diag([1, 0, 0, 0]).astype(complex)
            r = self._step(r, np.zeros((4, 4)), T_BUF_1Q)
            r = self._step(r, self._H1(prep, (p[6], p[6])), self.t_1q)
            r = self._step(r, np.zeros((4, 4)), T_BUF_1Q + self.t_1q)
            cols.append(self.readout_matrix(p) @ np.real(np.diag(r)))
        return np.array(cols).T


def fit_error_model(bset, T1, T2, free=(0, 1, 2, 3, 6, 7, 8, 9, 10, 11)):
    """Fit ErrorModel to the set's mean raw probabilities (independent of any CM).
    Thermal population is fixed at 0 by default: it is degenerate with readout error."""
    t1q, t2q = snapshot_durations(bset.tuids[0])
    model = ErrorModel(T1, T2, t1q, t2q)
    Pm, Pse = bset.P.mean(0), np.maximum(bset.P.std(0) / np.sqrt(len(bset)), 1e-3)
    free = list(free)

    def resid(x):
        p = _P0.copy()
        p[free] = x
        return ((model.probs(p) - Pm) / Pse).ravel()

    r = least_squares(resid, _P0[free], bounds=(_LB[free], _UB[free]))
    p = _P0.copy()
    p[free] = r.x
    dof = r.fun.size - len(free)
    chi2 = float((r.fun ** 2).sum() / dof)
    err = np.full(len(PARAMS), np.nan)
    err[free] = np.sqrt(np.diag(np.linalg.pinv(r.jac.T @ r.jac)) * chi2)
    return model, p, err, chi2


def error_budget(model, p):
    """Fidelity at each stage, and the cost of each fitted gate error on its own."""
    F_T12 = fidelity(model.state(IDEAL_PARAMS))
    costs = {}
    for name, idx in [("X180 amplitude", [6]), ("sqrt_iSWAP angle", [7]), ("Bell phase", [8, 9])]:
        q = IDEAL_PARAMS.copy()
        q[idx] = p[idx]
        costs[name] = fidelity(model.state(q)) - F_T12
    no_tomo = p.copy()
    no_tomo[10:] = 0
    recon = lambda q, CM: fidelity(mle(lin_rho(model.probs(q), CM)))
    costs["tomography pulses (apparent)"] = (recon(p, model.readout_matrix(p))
                                             - recon(no_tomo, model.readout_matrix(no_tomo)))
    return dict(
        F_T1T2=F_T12,
        F_state=fidelity(model.state(no_tomo)),
        F_recon_pure_readout=recon(p, model.readout_matrix(p)),
        F_recon_notebook_cm=recon(p, model.notebook_style_cm(p)),
        costs=costs,
    )


# ----------------------------------------------------------------------------- estimators
def _ideal_tomo_probs(rho, CM):
    Rx = lambda t: np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])
    Ry = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
    G = {"I": np.eye(2), "X90": Rx(np.pi / 2), "mY90": Ry(-np.pi / 2)}
    out = []
    for i in TOMO_SEQUENCE:
        a, b = _tomo_gates(i)
        U = np.kron(G[a], G[b])
        out.append(CM @ np.real(np.diag(U @ rho @ U.conj().T)))
    return np.array(out)


def simulate_estimators(rho_true, CM_true, n_runs, n_tomo, n_cm, reps=100, seed=1):
    """Monte-Carlo of the fidelity estimators on a known state. Returns {estimator: values}."""
    rng = np.random.default_rng(seed)
    Pt = _ideal_tomo_probs(rho_true, CM_true)
    out = {"per-run MLE, averaged": [], "per-run linear, averaged": [], "pooled, one MLE": []}
    for _ in range(reps):
        L = []
        for _ in range(n_runs):
            P = np.array([rng.multinomial(n_tomo, p / p.sum()) / n_tomo for p in Pt])
            C = np.array([rng.multinomial(n_cm, c / c.sum()) / n_cm for c in CM_true.T]).T
            L.append(lin_rho(P, C))
        L = np.array(L)
        out["per-run MLE, averaged"].append(np.mean([fidelity(mle(x)) for x in L]))
        out["per-run linear, averaged"].append(np.mean([fidelity(x) for x in L]))
        out["pooled, one MLE"].append(fidelity(mle(L.mean(0))))
    return {k: np.array(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- T1 / T2
def coherence_repeats(tuid):
    """Per-repeat fits (us, MHz) from helpers.t1_and_t2_fit, in measurement order.
    Returns {"Q1": {"T1": [...], "T2echo": [...], "T2*": [...], "f_ramsey": [...]}, "Q2": {...}};
    failed fits are NaN."""
    from snl_qblox.helpers import t1_and_t2_fit
    q1, q2 = t1_and_t2_fit(tuid, r_squared=0.8)
    keys = ["T1", "T2echo", "T2*", "f_ramsey"]
    return {q: {k: np.asarray(v, float) for k, v in zip(keys, vals)} for q, vals in (("Q1", q1), ("Q2", q2))}


def tls_dips(x, k_sigma=3.0):
    """Boolean mask of the low TLS dips in a set of repeated coherence fits.

    A TLS moving through resonance gives a lower mode (or single low points) under the normal
    spread; the dips are only ever LOW. The k lowest points (k < n/2, so the dips are the
    minority) are flagged when every one lies more than k_sigma standard deviations below the
    mean of the remaining points; the largest such k is used. NaN (failed fits) are not flagged.
    On Gaussian samples without dips this drops a lone lowest point in 12-19 % of sets (n = 20
    / 10), which moves the mean by well under one standard error."""
    x = np.asarray(x, float)
    fin = np.flatnonzero(np.isfinite(x))
    order = fin[np.argsort(x[fin])]
    v = x[order]
    best = 0
    for k in range(1, (len(v) + 1) // 2):
        hi = v[k:]
        if v[k - 1] < hi.mean() - k_sigma * hi.std(ddof=1):
            best = k
    mask = np.zeros(len(x), bool)
    mask[order[:best]] = True
    return mask


# Quantities whose low dips are removed as TLS events. Ramsey T2* is left unfiltered: its low
# repeats come from the qubit frequency jumping (Q2's frequency fluctuates between runs -- see
# process_tomography/qpt_trials.ipynb), which is real dephasing, not a TLS relaxation event.
TLS_FILTERED = ("T1", "T2echo")


def coherence_times(tuid, exclude=None, drop_tls=True):
    """Mean ± standard error (us) per qubit from a Multi Qubit T1 and T2 run.

    Failed fits (NaN) are skipped. With ``drop_tls`` the low TLS dips of T1 and T2 echo
    (TLS_FILTERED) are removed first (tls_dips; see tls_dropped for what was removed); Ramsey
    T2* is always averaged over all repeats. ``exclude`` additionally drops repeats of one qubit
    by index, e.g. ``{"Q2": range(8)}`` for a transient dip; the dropped repeats should be
    justified from coherence_repeats(), not chosen to move the result."""
    reps = coherence_repeats(tuid)
    exclude = {q: set(v) for q, v in (exclude or {}).items()}

    def ms(x, q, k):
        keep = np.array([i not in exclude.get(q, ()) for i in range(len(x))]) & np.isfinite(x)
        if drop_tls and k in TLS_FILTERED:
            keep &= ~tls_dips(np.where(keep, x, np.nan))
        x = x[keep]
        return float(np.mean(x)), float(np.std(x) / np.sqrt(len(x)))

    return {q: {k: ms(reps[q][k], q, k) for k in ("T1", "T2echo", "T2*")} for q in ("Q1", "Q2")}


def tls_dropped(tuid, exclude=None):
    """{qubit: {quantity: values removed as TLS dips}} for a Multi Qubit T1 and T2 run (only the
    TLS_FILTERED quantities; T2* is never filtered, so its entry is always empty)."""
    reps = coherence_repeats(tuid)
    exclude = {q: set(v) for q, v in (exclude or {}).items()}
    out = {}
    for q in ("Q1", "Q2"):
        out[q] = {}
        for k in ("T1", "T2echo", "T2*"):
            x = reps[q][k].copy()
            x[[i for i in exclude.get(q, ()) if i < len(x)]] = np.nan
            out[q][k] = np.sort(x[tls_dips(x)]) if k in TLS_FILTERED else np.array([])
    return out
