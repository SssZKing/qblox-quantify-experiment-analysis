"""Simulate two-qubit state tomography of the Bell state (|01> + |10>)/sqrt2 and compare to
the measurement in SNL315_plot.ipynb section 5.1 (Bell_state_0724_2.csv).

Sequence (q0 = Q1 = first tensor factor = dataset y0, q1 = Q2 = y1):
    [16 ns buffer] X180 on PREP_QUBIT
    [26 ns buffer] sqrt_iSWAP (duration from the snapshot)
    [16 ns buffer] tomography pulse pair  (TwoQubitClifford(idx).gate_decomposition())
    readout with assignment fidelities F_g, F_e

Every segment (including buffers) is integrated with a Lindblad master equation with T1/T2 on
both qubits. The sqrt_iSWAP is an XY exchange H = g (s+ s- + s- s+) with g*t = pi/4, followed by
a free virtual-Z frame update that puts the Bell coherence on the real axis (the target used in
the notebook). Reconstruction uses the same linear inversion as calculate_density_matrix.
"""
import csv
import os
import sys

import numpy as np
import qutip as qt
import xarray as xr
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.optimize import least_squares

from snl_qblox.tomography_tools import (calculate_density_matrix, compute_stokes, rearrange_stokes,
                              mle_residuals, rho2T, T2t, t2T, T2rho)
from snl_qblox.pycqed_randomized_benchmarking.clifford_group import TwoQubitClifford
from quantify_core.data.handling import set_datadir, load_snapshot

# quantify data directory; set QBLOX_DATADIR to override (default: SNL315/CD1)
DATA_DIR = os.environ.get(
    "QBLOX_DATADIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CD1"))
set_datadir(DATA_DIR)

# ---------------------------------------------------------------- parameters
T1 = [40e-6, 40e-6]          # s, (q0, q1)
T2 = [40e-6, 40e-6]          # s
F_G, F_E = 0.99, 0.90        # readout fidelity P(read g | g), P(read e | e), both qubits
T_BUF_1Q = 16e-9
T_BUF_2Q = 26e-9
PREP_QUBIT = 0               # qubit that receives the X180
TOMO_SEQUENCE = [360, 384, 0, 15, 375, 399, 16, 376, 400]
N_SHOTS, N_REPEATS = 1000, 49  # same as the measurement, for the shot-noise spread
MEAS_CSV = "Bell_state_0724_2.csv"

# CSVs live next to this script (bell_state/); relative names resolve here, not against the cwd.
HERE = os.path.dirname(os.path.abspath(__file__))


def data_path(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def _snapshot_durations(csv_path):
    """(rxy duration, sqrt_iSWAP duration) from the snapshot of the first measured run."""
    with open(data_path(csv_path)) as f:
        reader = csv.reader(f)
        next(reader)
        tuid = next(reader)[0]
    inst = load_snapshot(tuid)["instruments"]
    durs = [inst[q]["submodules"]["rxy"]["parameters"]["duration"]["value"]
            for q in ("qubit1", "qubit2")]
    if not np.isclose(durs[0], durs[1], rtol=1e-6):
        raise ValueError(f"qubit1/qubit2 rxy durations differ: {durs}")
    t_2q = inst["qubit1_qubit2"]["submodules"]["iswap"]["parameters"]["pulse_duration"]["value"]
    return durs[0], t_2q


# 0724 data: 100 ns single-qubit pulses, 256 ns sqrt_iSWAP
T_1Q, T_SQISWAP = _snapshot_durations(MEAS_CSV)

IDEAL = np.array([[0, 0, 0, 0], [0, .5, .5, 0], [0, .5, .5, 0], [0, 0, 0, 0]], dtype=complex)

# ---------------------------------------------------------------- operators
sm = qt.destroy(2)            # |0><1|, |0> = ground
id2 = qt.qeye(2)
AXES = {"X": qt.sigmax(), "Y": qt.sigmay()}
# gate name -> (axis, angle); rotations are exp(-i angle sigma/2)
GATES = {"I": None, "X180": ("X", np.pi), "X90": ("X", np.pi / 2), "mX90": ("X", -np.pi / 2),
         "Y90": ("Y", np.pi / 2), "mY90": ("Y", -np.pi / 2), "Y180": ("Y", np.pi)}


def on(op, q):
    return qt.tensor(op, id2) if q == 0 else qt.tensor(id2, op)


def collapse_ops(t1, t2):
    c_ops = []
    for q in range(2):
        if t1[q] is not None and np.isfinite(t1[q]):
            c_ops.append(np.sqrt(1 / t1[q]) * on(sm, q))
        if t2[q] is not None and np.isfinite(t2[q]):
            g1 = 1 / t1[q] if np.isfinite(t1[q]) else 0.0
            g_phi = 1 / t2[q] - g1 / 2
            if g_phi > 0:
                c_ops.append(np.sqrt(g_phi / 2) * on(qt.sigmaz(), q))
    return c_ops


def evolve(rho, H, duration, c_ops):
    if duration <= 0:
        return rho
    if H is None:
        H = 0 * qt.tensor(id2, id2)
    res = qt.mesolve(H, rho, np.linspace(0, duration, 21), c_ops, [])
    return res.states[-1]


def single_qubit_H(gates, duration):
    """Square-pulse drive implementing gates = (gate_q0, gate_q1) simultaneously."""
    H = 0 * qt.tensor(id2, id2)
    for q, name in enumerate(gates):
        spec = GATES[name]
        if spec is not None:
            axis, angle = spec
            H += (angle / duration) / 2 * on(AXES[axis], q)
    return H


def tomo_gates(idx):
    dec = dict((qs[0], g[0]) for qs, g in TwoQubitClifford(idx).gate_decomposition())
    return dec["q0"], dec["q1"]


G_XY = (np.pi / 4) / T_SQISWAP
H_XY = G_XY * (on(sm.dag(), 0) * on(sm, 1) + on(sm, 0) * on(sm.dag(), 1))


def frame_phase():
    """Virtual-Z on q0 that makes the ideal post-sqrt_iSWAP coherence rho[01,10] real positive."""
    psi = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    psi = on(qt.sigmax(), PREP_QUBIT) * psi
    psi = (-1j * H_XY * T_SQISWAP).expm() * psi
    r = (psi * psi.dag()).full()
    return np.angle(r[1, 2])


VZ = on(qt.Qobj(np.diag([1, np.exp(1j * frame_phase())])), 0)


def prepared_state(c_ops):
    rho = qt.ket2dm(qt.tensor(qt.basis(2, 0), qt.basis(2, 0)))
    prep = ["I", "I"]
    prep[PREP_QUBIT] = "X180"
    rho = evolve(rho, None, T_BUF_1Q, c_ops)
    rho = evolve(rho, single_qubit_H(prep, T_1Q), T_1Q, c_ops)
    rho = evolve(rho, None, T_BUF_2Q, c_ops)
    rho = evolve(rho, H_XY, T_SQISWAP, c_ops)
    return VZ * rho * VZ.dag()


def joint_probs(t1=T1, t2=T2, f_g=F_G, f_e=F_E):
    """(9, 4) measured probabilities [P00, P01, P10, P11] per tomo setting, plus the
    state right before the tomography pulses."""
    c_ops = collapse_ops(t1, t2)
    rho_bell = prepared_state(c_ops)
    M = np.array([[f_g, 1 - f_e], [1 - f_g, f_e]])  # M[read, true]
    CM = np.kron(M, M)
    probs = []
    for idx in TOMO_SEQUENCE:
        rho = evolve(rho_bell, None, T_BUF_1Q, c_ops)
        rho = evolve(rho, single_qubit_H(tomo_gates(idx), T_1Q), T_1Q, c_ops)
        p_true = np.real(np.diag(rho.full()))
        probs.append(CM @ p_true)
    return np.clip(np.array(probs), 0, None), rho_bell.full(), CM


def sample_dataset(probs, n_shots, rng):
    """Fake quantify dataset (y0 = q0, y1 = q1 single-shot bits) for calculate_density_matrix."""
    y0 = np.zeros((n_shots, 9), dtype=int)
    y1 = np.zeros((n_shots, 9), dtype=int)
    for k, p in enumerate(probs):
        outcome = rng.choice(4, size=n_shots, p=p / p.sum())
        y0[:, k] = outcome // 2
        y1[:, k] = outcome % 2
    return xr.Dataset({"y0": (("repetition", "acq_index_0"), y0),
                       "y1": (("repetition", "acq_index_1"), y1)})


def exact_dataset(probs):
    """Dataset whose shot averages reproduce `probs` to 1e-6 (infinite-shot limit)."""
    n = 10 ** 6
    y0 = np.zeros((n, 9), dtype=np.int8)
    y1 = np.zeros((n, 9), dtype=np.int8)
    for k, p in enumerate(probs):
        counts = np.round(p / p.sum() * n).astype(int)
        counts[-1] = n - counts[:-1].sum()
        outcome = np.repeat(np.arange(4), counts)
        y0[:, k] = outcome // 2
        y1[:, k] = outcome % 2
    return xr.Dataset({"y0": (("repetition", "acq_index_0"), y0),
                       "y1": (("repetition", "acq_index_1"), y1)})


def fidelity(rho):
    s = sqrtm(IDEAL)
    return np.abs(np.trace(sqrtm(s @ rho @ s))) ** 2


def mle(rho):
    t0 = T2t(rho2T(rho))
    res = least_squares(mle_residuals, t0, args=(compute_stokes(rho),), max_nfev=3000)
    return T2rho(t2T(res.x))


def load_measurement():
    tuids, rhos_corr = [], []
    with open(data_path(MEAS_CSV)) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            tuids.append(row[0])
            vals = [float(x) for x in row[1:]]
            rhos_corr.append(np.array([complex(vals[i], vals[i + 1])
                                       for i in range(0, len(vals), 2)]).reshape(4, 4))
    rhos_raw = [calculate_density_matrix(t)[0] for t in tuids]
    return np.array(rhos_raw), np.array(rhos_corr)


def fmt(rho):
    return np.array2string(np.round(rho, 3), precision=3, suppress_small=True)


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # ---- full model
    probs, rho_bell, CM = joint_probs()
    rho_sim_raw = calculate_density_matrix(exact_dataset(probs))[0]
    rho_sim_corr = mle(calculate_density_matrix(exact_dataset(probs), confusion_matrix=CM)[0])
    f_shots = [fidelity(calculate_density_matrix(sample_dataset(probs, N_SHOTS, rng))[0])
               for _ in range(N_REPEATS)]

    # ---- error budget
    budget = {
        "ideal (no T1/T2, perfect RO)": joint_probs(t1=[np.inf] * 2, t2=[np.inf] * 2, f_g=1, f_e=1),
        "T1/T2 only": joint_probs(f_g=1, f_e=1),
        "readout only": joint_probs(t1=[np.inf] * 2, t2=[np.inf] * 2),
        "T1/T2 + readout": (probs, rho_bell, CM),
    }

    # ---- measurement
    meas_raw, meas_corr = load_measurement()
    f_meas_raw = np.array([fidelity(r) for r in meas_raw])
    f_meas_corr = np.array([fidelity(r) for r in meas_corr])
    meas_raw_mean, meas_corr_mean = meas_raw.mean(0), meas_corr.mean(0)

    print(f"From snapshot: single-qubit pulse {T_1Q * 1e9:.0f} ns, sqrt_iSWAP {T_SQISWAP * 1e9:.0f} ns")
    print(f"Frame phase applied after sqrt_iSWAP: {np.degrees(frame_phase()):.1f} deg (virtual Z on q0)")
    print(f"State before tomography pulses (T1/T2 only), F = {fidelity(rho_bell):.4f}\n{fmt(rho_bell)}\n")
    print("Error budget (raw linear-inversion fidelity, infinite shots):")
    for name, (p, _, _) in budget.items():
        print(f"  {name:32s} F = {fidelity(calculate_density_matrix(exact_dataset(p))[0]):.4f}")
    print()
    print(f"Raw rho, simulated (infinite shots):\n{fmt(rho_sim_raw)}")
    print(f"Raw rho, measured (mean of {len(meas_raw)}):\n{fmt(meas_raw_mean)}\n")
    print(f"Readout-corrected + MLE rho, simulated:\n{fmt(rho_sim_corr)}")
    print(f"Corrected rho, measured (from {MEAS_CSV}):\n{fmt(meas_corr_mean)}\n")
    print(f"{'':28s}{'simulated':>20s}{'measured':>20s}")
    print(f"{'raw fidelity':28s}{np.mean(f_shots):>12.4f} ± {np.std(f_shots):.4f}"
          f"{f_meas_raw.mean():>12.4f} ± {f_meas_raw.std():.4f}")
    print(f"{'corrected fidelity':28s}{fidelity(rho_sim_corr):>12.4f} {'':8s}"
          f"{f_meas_corr.mean():>12.4f} ± {f_meas_corr.std():.4f}")

    # ---- Pauli expectation bar chart (same ordering as the notebook)
    labels = ['Z\nI', 'X\nI', 'Y\nI', 'I\nZ', 'I\nX', 'I\nY',
              'Z\nZ', 'Z\nX', 'Z\nY', 'X\nZ', 'X\nX', 'X\nY', 'Y\nZ', 'Y\nX', 'Y\nY']
    x = np.arange(15)
    w = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, (sim, meas, title) in zip(axes, [
            (rho_sim_raw, meas_raw_mean, "Raw (no readout correction)"),
            (rho_sim_corr, meas_corr_mean, "Readout-corrected + MLE")]):
        ax.bar(x - w / 2, rearrange_stokes(compute_stokes(meas)), w, color="black", label="Measured")
        ax.bar(x + w / 2, rearrange_stokes(compute_stokes(sim)), w, color="C1", label="Simulated")
        ax.bar(x, rearrange_stokes(compute_stokes(IDEAL)), 0.9, color="none", edgecolor="gray",
               label="Target", zorder=0)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("Expectation value")
        ax.set_title(title, loc="left")
        ax.legend(loc="lower right", fontsize=8, ncol=3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    fig.tight_layout()
    fig.savefig(data_path("bell_tomo_sim.png"), dpi=200)
    plt.show()
