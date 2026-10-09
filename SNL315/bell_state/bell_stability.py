"""Run-to-run stability of the repeated Bell-state tomography (Bell_state_0923_0..3.csv).

For every run it separates what the readout does from what the Bell state (i.e. the
X180 + sqrt_iSWAP) does:
  readout  : per-qubit F_g / F_e of the confusion matrix used for that run. The CMs were not
             saved, so they are recovered with bell_confusion_extract.fit_cm.
  iSWAP    : population imbalance p10 - p01 (swap angle), Bell phase arg(rho[01,10]) and
             coherence |rho[01,10]|, from the corrected rho and, independent of any CM, from
             the raw data.
Each quantity's run-to-run scatter is compared with a bootstrap of the shot noise
(1000-shot tomography + 500-shot confusion matrix per run, as in the notebook loop). A
ratio well above 1 means real fluctuation.
"""
import csv
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from snl_qblox.tomography_tools import compute_stokes
import bell_confusion_extract as bce

SETS = [bce.data_path(f"Bell_state_0923_{i}.csv") for i in range(4)]
N_TOMO, N_CM = 1000, 500
N_BOOT = 400
IDEAL = np.array([[0, 0, 0, 0], [0, .5, .5, 0], [0, .5, .5, 0], [0, 0, 0, 0]], dtype=complex)

# IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ
iXX, iXY, iYX, iYY = 4, 5, 8, 9


def corrected_rho(P, CM):
    """Notebook pipeline: CM-corrected linear inversion, then MLE (closed form)."""
    return bce.project_rho(bce.rho_from_stokes(bce.stokes_lin(np.linalg.inv(CM), P)))


def metrics(P, CM):
    rho = corrected_rho(P, CM)
    s_raw = bce.stokes_lin(np.eye(4), P)
    p_ii = P[2]  # raw joint probabilities of the no-pulse (Z-basis) setting
    m = {
        "F": np.real(np.trace(IDEAL @ rho)),
        "imbalance": np.real(rho[2, 2] - rho[1, 1]),
        "phase_deg": np.degrees(np.angle(rho[1, 2])),
        "coherence": np.abs(rho[1, 2]),
        "p00+p11": np.real(rho[0, 0] + rho[3, 3]),
        "raw_imbalance": (p_ii[2] - p_ii[1]) / (p_ii[2] + p_ii[1]),
        "raw_phase_deg": np.degrees(np.arctan2(s_raw[iXY] - s_raw[iYX], s_raw[iXX] + s_raw[iYY])),
        "raw_coherence": (s_raw[iXX] + s_raw[iYY]) / 2,
    }
    m.update(bce.marginal_fidelities(CM))
    return m


def tuid_time(tuid):
    return datetime.strptime(tuid[:15], "%Y%m%d-%H%M%S")


def bootstrap(P, CM, rng, n=N_BOOT):
    out = []
    for _ in range(n):
        Pb = np.array([rng.multinomial(N_TOMO, p / p.sum()) / N_TOMO for p in P])
        CMb = np.array([rng.multinomial(N_CM, c / c.sum()) / N_CM for c in CM.T]).T
        out.append(metrics(Pb, CMb))
    return {k: np.array([o[k] for o in out]) for k in out[0]}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    runs = []  # (set, time, P, CM, metrics, csv_F)
    for si, path in enumerate(SETS):
        for tuid, rho_csv in bce.load_rows(path):
            P = bce.raw_probs(tuid)
            CM, err = bce.fit_cm(P, compute_stokes(rho_csv), bce.PRIORS["0.99/0.90"])
            m = metrics(P, CM)
            runs.append((si, tuid_time(tuid), P, CM, m, np.real(np.trace(IDEAL @ rho_csv)), err))
        print(f"loaded {path}")

    keys = list(runs[0][4])
    t0 = min(r[1] for r in runs)
    tmin = np.array([(r[1] - t0).total_seconds() / 60 for r in runs])
    sets = np.array([r[0] for r in runs])
    vals = {k: np.array([r[4][k] for r in runs]) for k in keys}
    print(f"\nCM recovery: max stokes mismatch {max(r[6] for r in runs):.1e}; "
          f"F(rebuilt) vs F(csv) max diff {np.abs(vals['F'] - np.array([r[5] for r in runs])).max():.1e}")

    # shot-noise std per set, from the set's mean raw probabilities and mean CM
    boot_sd = {}
    for si in range(len(SETS)):
        sel = [r for r in runs if r[0] == si]
        b = bootstrap(np.mean([r[2] for r in sel], 0), np.mean([r[3] for r in sel], 0), rng)
        boot_sd[si] = {k: b[k].std() for k in keys}

    print(f"\n{'quantity':16s}" + "".join(f"{'set ' + str(i) + ' mean±sd':>22s}" for i in range(len(SETS)))
          + f"{'sd/shot-noise':>28s}")
    for k in keys:
        row = f"{k:16s}"
        ratios = []
        for si in range(len(SETS)):
            v = vals[k][sets == si]
            row += f"{v.mean():>13.4f} ± {v.std():.4f}"
            ratios.append(v.std() / boot_sd[si][k])
        row += "    " + " ".join(f"{x:5.1f}" for x in ratios)
        print(row)

    # which quantities track the fidelity (Pearson r, within-set, pooled after removing set means)
    print("\nCorrelation with F (within-set, pooled):")
    Fc = vals["F"] - np.array([vals["F"][sets == s].mean() for s in sets])
    for k in keys:
        if k == "F":
            continue
        vc = vals[k] - np.array([vals[k][sets == s].mean() for s in sets])
        print(f"  {k:16s} r = {np.corrcoef(Fc, vc)[0, 1]:+.2f}")

    # time series
    show = ["F", "Fe_Q1", "Fe_Q2", "Fg_Q1", "Fg_Q2", "imbalance", "raw_imbalance",
            "phase_deg", "raw_phase_deg", "coherence"]
    fig, axes = plt.subplots(len(show), 1, figsize=(8, 1.5 * len(show)), sharex=True)
    for ax, k in zip(axes, show):
        for si in range(len(SETS)):
            sel = sets == si
            mu = vals[k][sel].mean()
            ax.fill_between([tmin[sel].min(), tmin[sel].max()], mu - boot_sd[si][k], mu + boot_sd[si][k],
                            color=f"C{si}", alpha=0.2, lw=0)
            ax.plot(tmin[sel], vals[k][sel], "o-", ms=2.5, lw=0.6, color=f"C{si}",
                    label=f"set {si}" if k == show[0] else None)
        ax.set_ylabel(k, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, ncol=4, loc="lower right")
    axes[0].set_title("Bell runs 2026-09-23 (band = mean ± shot-noise sd)", fontsize=9, loc="left")
    axes[-1].set_xlabel(f"minutes after {t0:%H:%M:%S}")
    fig.tight_layout()
    fig.savefig(bce.data_path("bell_stability_0923.png"), dpi=150)
    plt.show()
