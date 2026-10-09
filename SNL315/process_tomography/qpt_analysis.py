"""Analysis layer behind SNL315_plot.ipynb section 5.2: load a repeated iSWAP process-tomography
set, reconstruct it exactly as the notebook does, and simulate the same experiment with
incoherent error only (measured T1, T2) and with the measured readout.

Experiment (pycqed_randomized_benchmarking.utils._qpt_build, Clifford 5760 = one full iSWAP):
    Reset | 16 ns | prep rotation (100 ns) | 26 ns | iSWAP pump pulse | virtual Rz |
    16 ns | meas rotation (100 ns) | readout
36 x 36 settings; with identity_reference every setting is repeated without the iSWAP block
(odd acquisition indices), which gives the SPAM map E. An all-identity rotation is padded to
the same 116 ns, so every setting has the same timing.

Conventions (same as tomography_tools._mle_forward_matrix / bell_analysis):
  basis |Q1 Q2> = 00, 01, 10, 11 (Q1 = q0 = dataset y0 = first Kronecker factor)
  rotations in _tomo_rotations order [I, X180, Y90, mY90, mX90, X90]
  p00 layout (prep_Q1, prep_Q2, meas_Q1, meas_Q2) -> 1296
  superoperators column-stacked (vec(A X B) = (B^T (x) A) vec(X)), which is qutip's convention.
"""
import os
import sys

import numpy as np
import qutip as qt
from scipy.linalg import expm, logm

HERE = os.path.dirname(os.path.abspath(__file__))
BELL = os.path.join(os.path.dirname(HERE), "bell_state")
CACHE = os.path.join(HERE, "cache")    # every cached reconstruction (.npz)
for p in (HERE, BELL):
    if p not in sys.path:
        sys.path.insert(0, p)

import bell_analysis as ba
from snl_qblox.tomography_tools import (closest_unitary, extract_angles, iSWAP_RPE, mle_chi_from_p00,
                              mle_chi_spam_corrected, proc_fid_to_unitary)
from quantify_core.data.handling import load_dataset

T_BUF_1Q = 16e-9     # operation_buffer_time in front of every single-qubit rotation
T_PRE_GATE = 26e-9   # index_to_operation: the iSWAP pump RF starts 26 ns after the previous op
N_ROT = 6
ROTATIONS = [None, ("X", np.pi), ("Y", np.pi / 2), ("Y", -np.pi / 2), ("X", -np.pi / 2), ("X", np.pi / 2)]
U_ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
DIMS = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]


def as_unitary(U):
    return U.full() if isinstance(U, qt.Qobj) else np.asarray(U, complex)


def avg_gate_fidelity(F_pro, d=4):
    return (d * F_pro + 1) / (d + 1)


# ----------------------------------------------------------------------------- data
class QPTSet:
    """One repeated QPT set: per-run raw p00 of the gate and of the identity reference."""

    def __init__(self, tuids):
        self.tuids, P, R = [], [], []
        for t in tuids:
            ds = load_dataset(t)
            if ds.acq_index_0.size != 2 * N_ROT ** 4:
                raise ValueError(f"{t}: expected {2 * N_ROT ** 4} acquisitions (identity_reference).")
            self.tuids.append(ds.attrs["tuid"])
            p00 = ((1 - ds["y0"].values) * (1 - ds["y1"].values)).mean(axis=0)
            P.append(p00[::2])
            R.append(p00[1::2])
        self.P, self.R = np.array(P), np.array(R)
        self.n_rep = int(load_dataset(self.tuids[0]).sizes["repetition"])
        self.times = [ba.tuid_time(t) for t in self.tuids]

    def __len__(self):
        return len(self.tuids)

    def n_shots(self, idx=None):
        """Shots behind each setting of the pooled p00 (the notebook's n_shots)."""
        return self.n_rep * (len(self) if idx is None else len(idx))

    def pooled(self, idx=None):
        """(p00, ref_p00) averaged over runs -- identical to the notebook's p00 / ref_p00."""
        idx = slice(None) if idx is None else list(idx)
        return self.P[idx].mean(0), self.R[idx].mean(0)

    def durations(self):
        """(single-qubit rxy duration, iSWAP pulse duration) in s, from the first run."""
        return ba.snapshot_durations(self.tuids[0])

    def readout(self):
        """Measured (F_g, F_e) of Q1 and Q2 per run: IQ shots of the multiplexed readout
        calibration in force for that run, classified with that run's threshold."""
        out, cache = [], {}
        for t in self.tuids:
            cal = ba.latest_readout_calibration(t)
            if cal not in cache:
                cache[cal] = ba.readout_from_iq(cal, t)
            out.append(cache[cal])
        return np.array(out)          # (n_runs, 2 qubits, [F_g, F_e])


# ----------------------------------------------------------------------------- reconstruction
def reconstruct(p00, ref_p00, n_shots, raw=True, forward=None):
    """The notebook's reconstruction: MLE of the reference (SPAM map E), MLE of the
    SPAM-corrected gate against the raw counts (p = 0.5), and optionally the uncorrected MLE.
    Returns chi Qobjs (qutip normalization, Tr = 16) and the closest unitary of the corrected
    gate, U_fit.

    forward: forward matrix for all three fits; None is the notebook's (ideal "00" effect),
    readout_forward(...) folds the measured readout into it (readout-aware reconstruction)."""
    out = {"ref": mle_chi_from_p00(ref_p00, n_shots, forward=forward)}
    out["corr"] = mle_chi_spam_corrected(p00, out["ref"], n_shots, p=0.5, forward=forward)
    if raw:
        out["raw"] = mle_chi_from_p00(p00, n_shots, forward=forward)
    out["U_fit"] = closest_unitary(out["corr"])[0]
    return out


def readout_povm(readout):
    """P(read 00 | basis state 00, 01, 10, 11) from measured [[F_g, F_e] Q1, [F_g, F_e] Q2], or
    the average over a stack of runs (exact for pooled data: p00 is linear in it)."""
    readout = np.asarray(readout, float)
    if readout.ndim == 3:
        return np.mean([readout_povm(r) for r in readout], axis=0)
    r0 = lambda g, e: np.array([g, 1 - e])
    return np.kron(r0(*readout[0]), r0(*readout[1]))


_FORWARD_CACHE = {}


def readout_forward(readout):
    """Forward matrix like tomography_tools._mle_forward_matrix, but with the measured readout
    in the "00" effect: E_kl = F_kl^dag diag(M00) F_kl, M00 = readout_povm(readout). Readout
    error then no longer has to be absorbed by the process (or by the reference map E)."""
    M00 = readout_povm(readout)
    key = M00.tobytes()
    if key not in _FORWARD_CACHE:
        F = [np.eye(2)] + [expm(-1j * th / 2 * ba._AX[ax]) for ax, th in ROTATIONS[1:]]
        rho00 = np.zeros((4, 4)); rho00[0, 0] = 1
        Fs = [np.kron(F[a], F[b]) for a in range(N_ROT) for b in range(N_ROT)]
        R = np.array([(X @ rho00 @ X.conj().T).ravel(order="F") for X in Fs]).T         # (16, 36)
        E = np.array([(X.conj().T @ np.diag(M00) @ X).T.ravel(order="F") for X in Fs])  # (36, 16)
        BC = np.empty((N_ROT ** 4, 256), complex)
        for k in range(256):
            e = np.zeros(256); e[k] = 1
            S = qt.to_super(qt.Qobj(e.reshape(16, 16), dims=DIMS, superrep="choi")).full()
            BC[:, k] = (E @ S @ R).T.ravel()
        _FORWARD_CACHE[key] = BC
    return _FORWARD_CACHE[key]


def fit_chi2(chi, p00, n_shots, forward=None):
    """Mean squared residual of a fit in units of binomial shot noise (1 = describes the data)."""
    from snl_qblox.tomography_tools import _mle_forward_matrix
    BC = _mle_forward_matrix() if forward is None else forward
    pred = np.real(BC @ qt.to_choi(chi).full().ravel())
    p = np.asarray(p00, float).ravel()
    return float(np.mean((pred - p) ** 2 / np.clip(p * (1 - p), 1e-4, None) * n_shots))


def summarize(rec, U_target=None, phi_p=0.0):
    """Process fidelities of a reconstruction. Incoherent error = distance to the gate's own
    closest unitary (U_fit), so a coherent miscalibration does not count.

    phi_p is passed to extract_angles: 0 for an iSWAP (phi_p unobservable, individual phases
    read from the off-diagonals under that convention); None for an idle (theta_p ~ 0), where
    the off-diagonals vanish and the phases must be read from the diagonal."""
    U = rec["U_fit"] if U_target is None else U_target
    s = {"F_corr": proc_fid_to_unitary(rec["corr"], U),
         "F_ref": proc_fid_to_unitary(rec["ref"], closest_unitary(rec["ref"])[0]),
         "angles_deg": {k: np.degrees(v) for k, v in
                        extract_angles(as_unitary(rec["U_fit"]), phi_p=phi_p).items()}}
    if "raw" in rec:
        s["F_raw"] = proc_fid_to_unitary(rec["raw"], U)
    return s


PAULI_LABELS = [a + b for a in "IXYZ" for b in "IXYZ"]      # Q1 first
_P1 = [np.eye(2), ba._X, ba._Y, ba._Z]
_P16 = [np.kron(a, b) for a in _P1 for b in _P1]


def ptm(chi):
    """16x16 Pauli transfer matrix R_ij = Tr[P_i Lambda(P_j)] / 4 (real), rows/cols PAULI_LABELS."""
    S = qt.to_super(chi).full()
    out = np.array([(S @ P.ravel(order="F")).reshape(4, 4, order="F") for P in _P16])
    return np.real(np.einsum("iab,jba->ij", np.array(_P16), out)) / 4


def save_recon(path, rec):
    np.savez(path, **{k: (v.full()) for k, v in rec.items()})


def load_recon(path):
    d = np.load(path)
    return {k: (qt.Qobj(d[k], dims=[[2, 2], [2, 2]]) if k == "U_fit"
                else qt.Qobj(d[k], dims=DIMS, superrep="chi")) for k in d.files}


def cached_reconstruct(path, p00, ref_p00, n_shots, raw=True, redo=False, forward=None):
    if os.path.exists(path) and not redo:
        return load_recon(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = reconstruct(p00, ref_p00, n_shots, raw, forward)
    save_recon(path, rec)
    return rec


# ----------------------------------------------------------------------------- model
class QPTModel:
    """Lindblad model of the QPT sequence with amplitude damping and pure dephasing on each
    qubit from the measured T1, T2, and no gate errors.

    The iSWAP block is 26 ns of idle and then a constant generator H = i log(U_target) / t_gate
    for the pulse duration, so its decoherence-free unitary is exactly U_target: the ideal
    iSWAP by default; analyze() uses iSWAP_RPE with theta_p = 90 deg, theta_1 = theta_2 =
    phi_p = 0 and the fitted phi_zz (the ZZ phase is the only coherent error). Rotations are
    square pulses of the measured
    length on both qubits simultaneously (an identity is an idle of the same length).
    """

    def __init__(self, T1, T2, t_1q, t_gate, U_target=U_ISWAP, t_pre=T_PRE_GATE):
        self.T1, self.T2 = list(T1), list(T2)
        self.t_1q, self.t_gate, self.t_pre = t_1q, t_gate, t_pre
        self.U_target = as_unitary(U_target)
        self.H_gate = 1j * logm(self.U_target) / t_gate
        self.H_gate = (self.H_gate + self.H_gate.conj().T) / 2
        self._rot = {}

    def c_ops(self, which=("T1_Q1", "Tphi_Q1", "T1_Q2", "Tphi_Q2")):
        cs = []
        for q, name in enumerate(("Q1", "Q2")):
            g1 = 1 / self.T1[q]
            gp = 1 / self.T2[q] - g1 / 2
            if f"T1_{name}" in which:
                cs.append(np.sqrt(g1) * ba._on(ba._SM, q))
            if f"Tphi_{name}" in which and gp > 0:
                cs.append(np.sqrt(gp / 2) * ba._on(ba._Z, q))
        return cs

    @staticmethod
    def liouvillian(H, c_ops):
        I4 = np.eye(4)
        L = -1j * (np.kron(I4, H) - np.kron(H.T, I4))
        for C in c_ops:
            CdC = C.conj().T @ C
            L = L + np.kron(C.conj(), C) - 0.5 * np.kron(I4, CdC) - 0.5 * np.kron(CdC.T, I4)
        return L

    def propagator(self, H, t, c_ops):
        return expm(self.liouvillian(H, c_ops) * t)

    def gate_superop(self, c_ops=None, idle=True):
        """16x16 superoperator of the iSWAP block (26 ns idle + pulse)."""
        c_ops = self.c_ops() if c_ops is None else c_ops
        S = self.propagator(self.H_gate, self.t_gate, c_ops)
        if idle:
            S = S @ self.propagator(np.zeros((4, 4)), self.t_pre, c_ops)
        return S

    def gate_channel(self, c_ops=None, idle=True):
        return qt.Qobj(self.gate_superop(c_ops, idle), dims=DIMS, superrep="super")

    def process_fidelity(self, c_ops=None, idle=True, U=None):
        U = self.U_target if U is None else as_unitary(U)
        chi = qt.to_chi(self.gate_channel(c_ops, idle))
        return proc_fid_to_unitary(chi, U)

    def budget(self):
        """Process infidelity of the iSWAP block from each decay channel on its own."""
        out = {}
        for name in ("T1_Q1", "Tphi_Q1", "T1_Q2", "Tphi_Q2"):
            out[name] = 1 - self.process_fidelity(self.c_ops((name,)))
        out["all"] = 1 - self.process_fidelity()
        return out

    def _rotation_superop(self, a, b, decoh=True):
        """16 ns idle, then rotation a on Q1 and b on Q2 (indices into ROTATIONS)."""
        key = (a, b, decoh)
        if key not in self._rot:
            c_ops = self.c_ops() if decoh else []
            H = np.zeros((4, 4), complex)
            for q, r in enumerate((a, b)):
                if ROTATIONS[r] is not None:
                    ax, th = ROTATIONS[r]
                    H += th / self.t_1q / 2 * ba._on(ba._AX[ax], q)
            self._rot[key] = (self.propagator(H, self.t_1q, c_ops)
                              @ self.propagator(np.zeros((4, 4)), T_BUF_1Q, c_ops))
        return self._rot[key]

    def p00(self, readout, gate=True, decoh=True):
        """(1296,) P(read 00) per setting, in the notebook's p00 layout.

        readout: (2, 2) array [[F_g, F_e] of Q1, [F_g, F_e] of Q2], or a stack of them (one per
        run), in which case p00 is averaged over the runs like the data. gate=False gives the
        identity reference."""
        readout = np.asarray(readout, float)
        if readout.ndim == 3:
            return np.mean([self.p00(r, gate, decoh) for r in readout], axis=0)
        c_ops = self.c_ops() if decoh else []
        G = self.gate_superop(c_ops) if gate else np.eye(16)
        r0 = lambda g, e: np.array([g, 1 - e])            # P(read 0 | state 0, 1)
        m00 = np.kron(r0(*readout[0]), r0(*readout[1]))    # P(read 00 | |ab>)
        diag = np.arange(4) * 5                            # vec indices of the populations
        rho0 = np.zeros(16, complex)
        rho0[0] = 1
        preps = [G @ self._rotation_superop(a, b, decoh) @ rho0
                 for a in range(N_ROT) for b in range(N_ROT)]
        meas = [m00 @ self._rotation_superop(a, b, decoh)[diag] for a in range(N_ROT) for b in range(N_ROT)]
        return np.real(np.array([[m @ r for m in meas] for r in preps])).ravel()


def sample(p, n_shots, rng):
    """Binomial shot noise on a p00 vector."""
    return rng.binomial(n_shots, np.clip(p, 0, 1)) / n_shots


# ----------------------------------------------------------------------------- driver
def analyze(name, tuids, t1t2_tuid, groups=None, redo=False, verbose=True, gate="iswap"):
    """Measured vs simulated fidelities of one QPT set. Reconstructions are cached as
    cache/<name>_<label>.npz next to this module (redo=True recomputes them).

    groups: lists of run indices for the error bar (default: runs sharing a readout
    calibration). The simulation uses the measured T1 and T2 echo of ``t1t2_tuid`` (one tuid,
    or a list whose per-run means are averaged) with TLS dips removed (ba.coherence_times) and
    the measured single-shot readout of every run.

    gate: "iswap" (Clifford 5760 with the pump on) or "idle" (the same schedule with the pump
    off). The simulated gate is iSWAP_RPE with theta_p = 90 deg ("iswap") or 0 ("idle"),
    theta_1 = theta_2 = phi_p = 0, and the fitted phi_zz: the ZZ phase is its only coherent error."""
    if gate not in ("iswap", "idle"):
        raise ValueError(f"gate must be 'iswap' or 'idle', not {gate!r}")
    phi_p = 0.0 if gate == "iswap" else None
    summ = lambda r: summarize(r, phi_p=phi_p)
    say = print if verbose else (lambda *a, **k: None)
    qs = QPTSet(tuids)
    n = qs.n_shots()
    path = lambda label: os.path.join(CACHE, f"{name}_{label}.npz")

    say(f"{name}: {len(qs)} runs x {qs.n_rep} shots, {qs.times[0]:%H:%M}-{qs.times[-1]:%H:%M}")
    p00, ref_p00 = qs.pooled()
    ro = qs.readout()
    fwd = readout_forward(ro)
    # "meas": the notebook's reconstruction; "meas_ra": readout-aware (measured readout in the model)
    meas = cached_reconstruct(path("meas"), p00, ref_p00, n, redo=redo)
    meas_ra = cached_reconstruct(path("meas_ra"), p00, ref_p00, n, redo=redo, forward=fwd)

    if groups is None:
        cals = [ba.latest_readout_calibration(t) for t in qs.tuids]
        groups = [[i for i, c in enumerate(cals) if c == u] for u in dict.fromkeys(cals)]
    group_summaries, group_summaries_ra = [], []
    for g_i, g in enumerate(groups):
        pg, rg = qs.pooled(g)
        rec = cached_reconstruct(path(f"group{g_i}"), pg, rg, qs.n_shots(g), raw=False, redo=redo)
        group_summaries.append(summ(rec))
        rec = cached_reconstruct(path(f"group{g_i}_ra"), pg, rg, qs.n_shots(g), raw=False, redo=redo,
                                 forward=readout_forward(ro[g]))
        group_summaries_ra.append(summ(rec))

    t1t2_tuids = [t1t2_tuid] if isinstance(t1t2_tuid, str) else list(t1t2_tuid)
    coh_runs = {t: ba.coherence_times(t) for t in t1t2_tuids}
    tls = {t: ba.tls_dropped(t) for t in t1t2_tuids}
    # mean of the per-run means; standard error from the per-run standard errors
    coh = {q: {k: (float(np.mean([c[q][k][0] for c in coh_runs.values()])),
                   float(np.sqrt(np.sum([c[q][k][1] ** 2 for c in coh_runs.values()])) / len(coh_runs)))
               for k in ("T1", "T2echo", "T2*")} for q in ("Q1", "Q2")}
    T1 = [coh[q]["T1"][0] * 1e-6 for q in ("Q1", "Q2")]
    T2 = [coh[q]["T2echo"][0] * 1e-6 for q in ("Q1", "Q2")]
    t_1q, t_gate = qs.durations()
    # the simulated gate: a perfect iSWAP (theta_p = 90 deg) or idle (theta_p = 0), theta_1 = theta_2 =
    # phi_p = 0, whose only coherent error is the ZZ phase fitted by the readout-aware reconstruction
    phi_zz = extract_angles(as_unitary(meas_ra["U_fit"]), phi_p=phi_p)["phi_zz"]
    U_sim = iSWAP_RPE([np.pi / 2 if gate == "iswap" else 0.0, 0.0, 0.0, 0.0, phi_zz])
    model = QPTModel(T1, T2, t_1q, t_gate, U_target=U_sim)

    ideal_ro = np.ones((2, 2))
    rng = np.random.default_rng(0)
    p_s, r_s = model.p00(ro), model.p00(ro, gate=False)
    p_n, r_n = sample(p_s, n, rng), sample(r_s, n, rng)
    tag = f"sim_zz{np.degrees(phi_zz):+.2f}"         # the cache name records the simulated gate
    sim_path = lambda label: path(f"{tag}_{label}")
    sims = {
        "sim_decay": cached_reconstruct(sim_path("decay"), model.p00(ideal_ro),
                                        model.p00(ideal_ro, gate=False), n, redo=redo),
        "sim_decay_ro": cached_reconstruct(sim_path("decay_ro"), p_s, r_s, n, redo=redo),
        "sim_decay_ro_shots": cached_reconstruct(sim_path("decay_ro_shots"), p_n, r_n, n, redo=redo),
        "sim_decay_ro_ra": cached_reconstruct(sim_path("decay_ro_ra"), p_s, r_s, n, redo=redo, forward=fwd),
        "sim_decay_ro_shots_ra": cached_reconstruct(sim_path("decay_ro_shots_ra"), p_n, r_n, n, redo=redo,
                                                    forward=fwd),
    }
    sim_data = {"p00": p_s, "ref_p00": r_s, "p00_shots": p_n, "ref_p00_shots": r_n}
    F_groups = np.array([g["F_corr"] for g in group_summaries])
    F_groups_ra = np.array([g["F_corr"] for g in group_summaries_ra])
    return dict(qset=qs, n_shots=n, gate=gate, phi_p=phi_p, coh=coh, coh_runs=coh_runs, tls=tls,
                T1=T1, T2=T2, readout=ro,
                forward=fwd, t_1q=t_1q, t_gate=t_gate, model=model, phi_zz_sim=phi_zz, U_sim=U_sim,
                meas=meas, meas_summary=summ(meas), meas_ra=meas_ra, meas_ra_summary=summ(meas_ra),
                groups=groups, group_summaries=group_summaries, group_summaries_ra=group_summaries_ra,
                F_groups=F_groups, F_groups_ra=F_groups_ra,
                F_model=model.process_fidelity(), budget=model.budget(), sims=sims, sim_data=sim_data,
                sim_summary={k: summ(v) for k, v in sims.items()})


def report(res):
    """Text table of an analyze() result."""
    coh, m = res["coh"], res["meas_summary"]
    runs = res["coh_runs"]
    for t, c in runs.items():
        if len(runs) > 1:
            print(f"T1/T2 run {t}:")
        for q in ("Q1", "Q2"):
            dropped = {k: v for k, v in res["tls"][t][q].items() if len(v)}
            print(q, ", ".join(f"{k} = {v[0]:.1f} ± {v[1]:.1f} us" for k, v in c[q].items()),
                  f"  (TLS dips removed: {', '.join(f'{k} {np.round(v, 1).tolist()}' for k, v in dropped.items())})"
                  if dropped else "")
    if len(runs) > 1:
        print("used (mean of the runs): " + ";  ".join(
            f"{q} " + ", ".join(f"{k} {coh[q][k][0]:.1f}" for k in ("T1", "T2echo")) for q in ("Q1", "Q2")))
    ro = res["readout"].mean(0)
    print(f"readout (mean over runs): Q1 F_g = {ro[0, 0]:.4f}, F_e = {ro[0, 1]:.4f};  "
          f"Q2 F_g = {ro[1, 0]:.4f}, F_e = {ro[1, 1]:.4f}")
    print(f"gate: {res['gate']}; pulses: 1Q {res['t_1q'] * 1e9:.0f} ns, gate window {res['t_gate'] * 1e9:.0f} ns"
          f" (+ {T_PRE_GATE * 1e9:.0f} ns)")
    a = m["angles_deg"]
    print("measured U_fit angles (deg): " + ", ".join(f"{k} {a[k]:.2f}" for k in
                                                     ("theta_p", "theta_sum", "theta_d", "phi_zz")))
    Fg = res["F_groups"]
    err = Fg.std(ddof=1) / np.sqrt(len(Fg))
    s = res["sim_summary"]
    rows = [
        ("simulated, T1 + T2echo (gate block itself)", res["F_model"], None),
        ("measured, SPAM-corrected", m["F_corr"], err),
        ("sim. T1 + T2echo -> tomography -> pipeline", s["sim_decay"]["F_corr"], None),
        ("sim. + measured readout -> pipeline", s["sim_decay_ro"]["F_corr"], None),
        ("sim. + readout + shot noise -> pipeline", s["sim_decay_ro_shots"]["F_corr"], None),
        ("measured, uncorrected", m["F_raw"], None),
        ("sim. + readout -> pipeline, uncorrected", s["sim_decay_ro"]["F_raw"], None),
        ("measured, identity reference (SPAM map)", m["F_ref"], None),
        ("sim. + readout, identity reference", s["sim_decay_ro"]["F_ref"], None),
    ]
    mr, Fr = res["meas_ra_summary"], res["F_groups_ra"]
    rows_ra = [
        ("measured, SPAM-corrected", mr["F_corr"], Fr.std(ddof=1) / np.sqrt(len(Fr))),
        ("sim. + readout + shot noise", s["sim_decay_ro_shots_ra"]["F_corr"], None),
        ("measured, no reference correction", mr["F_raw"], None),
        ("sim. + readout + shot noise, no reference corr.", s["sim_decay_ro_shots_ra"]["F_raw"], None),
        ("measured, identity reference", mr["F_ref"], None),
        ("sim. + readout + shot noise, identity reference", s["sim_decay_ro_shots_ra"]["F_ref"], None),
    ]
    print(f"\n{'process fidelity to the closest unitary':48s}{'F_pro':>8s}{'F_avg':>9s}")
    for lab, F, e in rows:
        print(f"{lab:48s}{F:8.4f}{avg_gate_fidelity(F):9.4f}" + (f"   ± {e:.4f} (sem of {len(Fg)} groups)" if e else ""))
    a = mr["angles_deg"]
    print("\nreadout-aware reconstruction (measured F_g, F_e in the model); U_fit angles (deg): "
          + ", ".join(f"{k} {a[k]:.2f}" for k in ("theta_p", "theta_sum", "theta_d", "phi_zz")))
    for lab, F, e in rows_ra:
        print(f"{lab:48s}{F:8.4f}{avg_gate_fidelity(F):9.4f}" + (f"   ± {e:.4f} (sem of {len(Fr)} groups)" if e else ""))
    print("\nincoherent infidelity of the gate block, each channel alone: "
          + ", ".join(f"{k} {v:.4f}" for k, v in res["budget"].items()))



# ----------------------------------------------------------------------------- estimators across runs
# The pooled estimate (analyze: average P00 over runs, one MLE) is the QPT counterpart of the Bell
# analysis: everything before the MLE is linear -- P00 is averaged over runs, and each run's readout
# enters the forward matrix linearly, so readout_forward(stack of runs) is exact for the pooled data.
# The functions below add the other estimators (one MLE per run, per group), a bootstrap over runs,
# and simulations with a known answer; the many fits run in parallel processes and are cached in
# cache/ next to this module.


def _fit_job(job):
    """Worker: one cached reconstruction. job = dict(path, p00, ref_p00, n_shots, raw, readout);
    readout None = the ideal "00" effect (the SNL315_plot reconstruction)."""
    fwd = None if job.get("readout") is None else readout_forward(job["readout"])
    cached_reconstruct(job["path"], job["p00"], job["ref_p00"], job["n_shots"], raw=job.get("raw", False),
                       forward=fwd)
    return job["path"]


def run_jobs(jobs, workers=20, verbose=True):
    """Run the fits whose cache file is missing, in parallel processes (one BLAS thread each)."""
    todo = [j for j in jobs if not os.path.exists(j["path"])]
    if verbose:
        print(f"{len(jobs)} fits, {len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
    if not todo:
        return
    for d in {os.path.dirname(j["path"]) for j in todo}:
        os.makedirs(d, exist_ok=True)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"                          # inherited by the worker processes
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as ex:
        futs = [ex.submit(_fit_job, j) for j in todo]
        for k, f in enumerate(as_completed(futs), 1):
            f.result()
            if verbose and (k % 20 == 0 or k == len(todo)):
                print(f"  {k}/{len(todo)} fits done", flush=True)


def default_groups(qs):
    """Runs that share a readout calibration (as in analyze)."""
    cals = [ba.latest_readout_calibration(t) for t in qs.tuids]
    return [[i for i, c in enumerate(cals) if c == u] for u in dict.fromkeys(cals)]


def set_jobs(name, qs, ro, groups=None):
    """The data fits analyze() needs: pooled and per-group, both reconstructions."""
    groups = default_groups(qs) if groups is None else groups
    p, r = qs.pooled()
    path = lambda label: os.path.join(CACHE, f"{name}_{label}.npz")
    jobs = [dict(path=path("meas"), p00=p, ref_p00=r, n_shots=qs.n_shots(), raw=True, readout=None),
            dict(path=path("meas_ra"), p00=p, ref_p00=r, n_shots=qs.n_shots(), raw=True, readout=ro)]
    for g_i, g in enumerate(groups):
        pg, rg = qs.pooled(g)
        jobs += [dict(path=path(f"group{g_i}"), p00=pg, ref_p00=rg, n_shots=qs.n_shots(g), readout=None),
                 dict(path=path(f"group{g_i}_ra"), p00=pg, ref_p00=rg, n_shots=qs.n_shots(g), readout=ro[g])]
    return jobs


def per_run_paths(name, n_runs):
    return [os.path.join(CACHE, f"{name}_run{i:03d}_ra.npz") for i in range(n_runs)]


def per_run_jobs(name, qs, ro):
    """One readout-aware, SPAM-corrected fit per run (400 shots per setting)."""
    return [dict(path=p, p00=qs.P[i], ref_p00=qs.R[i], n_shots=qs.n_rep, readout=ro[i])
            for i, p in enumerate(per_run_paths(name, len(qs)))]


def bootstrap_indices(n_runs, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_runs, n_runs) for _ in range(n_boot)]


def bootstrap_paths(name, n_boot):
    return [os.path.join(CACHE, f"{name}_boot{b:02d}_ra.npz") for b in range(n_boot)]


def bootstrap_jobs(name, qs, ro, n_boot=24, seed=0):
    """Pooled readout-aware fits of runs resampled with replacement (bootstrap over runs)."""
    jobs = []
    for p, idx in zip(bootstrap_paths(name, n_boot), bootstrap_indices(len(qs), n_boot, seed)):
        pb, rb = qs.pooled(idx)
        jobs.append(dict(path=p, p00=pb, ref_p00=rb, n_shots=qs.n_shots(idx), readout=ro[idx]))
    return jobs


ESTIMATOR_SHOTS = ((400, 8), (1600, 8), (8000, 4), (40000, 2))


def estimator_sim_paths(label, shots=ESTIMATOR_SHOTS):
    return {n: [os.path.join(CACHE, f"{label}_n{n}_r{k}_ra.npz") for k in range(reps)] for n, reps in shots}


def estimator_sim_jobs(label, model, readout, shots=ESTIMATOR_SHOTS, seed=1):
    """Simulated data with a known answer (model.process_fidelity()) at several shot counts per
    setting -- one run (400), one group (1600), a 20-run set (8000), a 100-run set (40000) --
    each reconstructed readout-aware like the data."""
    p_s, r_s = model.p00(readout), model.p00(readout, gate=False)
    rng = np.random.default_rng(seed)
    jobs = []
    for n, paths in estimator_sim_paths(label, shots).items():
        for p in paths:
            jobs.append(dict(path=p, p00=sample(p_s, n, rng), ref_p00=sample(r_s, n, rng), n_shots=n,
                             readout=readout))
    return jobs


def load_summaries(paths, phi_p=0.0):
    return [summarize(load_recon(p), phi_p=phi_p) for p in paths]


def mean_channel_fidelity(chis):
    """Fidelity (to its own closest unitary) of the average of several channels -- what pooling
    their data measures. Compared with the mean of their own fidelities, the difference is the
    cost of the coherent part changing between them."""
    S = np.mean([qt.to_super(c).full() for c in chis], axis=0)
    chi = qt.to_chi(qt.Qobj(S, dims=DIMS, superrep="super"))
    return proc_fid_to_unitary(chi, closest_unitary(chi)[0])


def reference_z(qs):
    """Per run: rms deviation of the identity-reference P00 from the set median, in units of the
    binomial shot noise of one run (about 1 when the run is consistent with the rest)."""
    med = np.median(qs.R, 0)
    sn = np.sqrt(np.clip(med * (1 - med), 1 / qs.n_rep, None) / qs.n_rep)
    return np.sqrt(np.mean(((qs.R - med) / sn) ** 2, axis=1))


def reference_outliers(qs, z_max=1.5):
    """Runs whose identity reference -- which contains no gate -- is inconsistent with the rest of
    the set (reference_z > z_max): a state-preparation or readout failure during that run (e.g. a
    qubit-frequency excursion). Selecting on the reference cannot select on gate quality."""
    return np.flatnonzero(reference_z(qs) > z_max)
