"""Compute (and cache) the three repeated iSWAP QPT trials: pooled, per-group, per-run and
bootstrap fits, the incoherent-error simulations, and the estimator simulations.

    python run_qpt_trials.py          (~20-30 min on 20 cores; everything cached afterwards)

The notebook qpt_trials.ipynb then only loads the caches.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import matplotlib

if __name__ == "__main__":
    # only when run as a script: qpt_trials.ipynb imports this module for TRIALS / clean_trials(),
    # and switching the backend on import would replace the notebook's inline backend (no figures)
    matplotlib.use("Agg")
from quantify_core.data.handling import set_datadir

# quantify data directory; set QBLOX_DATADIR to override (default: SNL315/CD1)
DATADIR = os.environ.get(
    "QBLOX_DATADIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CD1"))
set_datadir(DATADIR)
import qpt_analysis as qa  # noqa: E402

T1 = ['20260924-2103', '20260924-2108', '20260924-2113', '20260924-2118', '20260924-2123', '20260924-2128', '20260924-2133', '20260924-2138', '20260924-2143', '20260924-2148', '20260924-2153', '20260924-2158', '20260924-2203', '20260924-2208', '20260924-2213', '20260924-2218', '20260924-2223', '20260924-2228', '20260924-2233', '20260924-2238']
T2 = ['20260925-0336', '20260925-0341', '20260925-0346', '20260925-0351', '20260925-0356', '20260925-0401', '20260925-0406', '20260925-0410', '20260925-0416', '20260925-0421', '20260925-0425', '20260925-0430', '20260925-0436', '20260925-0441', '20260925-0445', '20260925-0450', '20260925-0456', '20260925-0500', '20260925-0505', '20260925-0510']
T3 = ['20260925-0521', '20260925-0526', '20260925-0530', '20260925-0535', '20260925-0541', '20260925-0546', '20260925-0550', '20260925-0555', '20260925-0601', '20260925-0605', '20260925-0610', '20260925-0615', '20260925-0620', '20260925-0625', '20260925-0630', '20260925-0635', '20260925-0640', '20260925-0645', '20260925-0650', '20260925-0655', '20260925-0700', '20260925-0705', '20260925-0710', '20260925-0715', '20260925-0720', '20260925-0725', '20260925-0730', '20260925-0735', '20260925-0740', '20260925-0745', '20260925-0750', '20260925-0754', '20260925-0800', '20260925-0805', '20260925-0809', '20260925-0814', '20260925-0820', '20260925-0824', '20260925-0829', '20260925-0834', '20260925-0839', '20260925-0844', '20260925-0849', '20260925-0854', '20260925-0859', '20260925-0904', '20260925-0909', '20260925-0914', '20260925-0919', '20260925-0924', '20260925-0929', '20260925-0934', '20260925-0939', '20260925-0944', '20260925-0949', '20260925-0954', '20260925-0959', '20260925-1004', '20260925-1009', '20260925-1013', '20260925-1019', '20260925-1024', '20260925-1028', '20260925-1033', '20260925-1039', '20260925-1043', '20260925-1048', '20260925-1053', '20260925-1058', '20260925-1103', '20260925-1108', '20260925-1113', '20260925-1118', '20260925-1123', '20260925-1128', '20260925-1133', '20260925-1138', '20260925-1143', '20260925-1148', '20260925-1153', '20260925-1158', '20260925-1203', '20260925-1208', '20260925-1213', '20260925-1218', '20260925-1223', '20260925-1228', '20260925-1232', '20260925-1238', '20260925-1243', '20260925-1247', '20260925-1252', '20260925-1258', '20260925-1302', '20260925-1307', '20260925-1312', '20260925-1317', '20260925-1322', '20260925-1327', '20260925-1332']

# name: (runs, Multi Qubit T1 and T2 run(s) next to the trial)
TRIALS = {
    "QPT_0924": (T1, "20260924-223804-659-6021ad"),                                   # right after
    "QPT_0925a": (T2, "20260925-051033-600-00ae09"),                                  # right after
    "QPT_0925b": (T3, ["20260925-051033-600-00ae09", "20260925-133224-078-01fc4b"]),  # before, after
}
N_BOOT = 24
ALL = "QPT_all"            # the three trials pooled together (all runs)
ALL_OK = "QPT_all_ok"      # ... without the reference outliers


def clean_trials():
    """{analysis name: (kept runs, T1/T2 runs, excluded run indices)} after removing the runs whose
    identity reference is inconsistent with the rest of their trial (qa.reference_outliers).
    A trial without outliers keeps its name; otherwise the name gets an "_ok" suffix."""
    out = {}
    for name, (tuids, t1t2) in TRIALS.items():
        bad = qa.reference_outliers(qa.QPTSet(tuids))
        kept = [t for i, t in enumerate(tuids) if i not in set(bad)]
        out[name + "_ok" if len(bad) else name] = (kept, t1t2, bad.tolist())
    return out


def _analyze(args):
    name, tuids, t1t2 = args
    set_datadir(DATADIR)
    qa.analyze(name, tuids, t1t2, verbose=False)
    return name


def main():
    from concurrent.futures import ProcessPoolExecutor
    import numpy as np

    t0 = time.time()
    clean = clean_trials()
    for name, (kept, _, bad) in clean.items():
        print(f"{name}: {len(kept)} runs kept, reference outliers excluded: {bad}", flush=True)
    analyses = {name: (tuids, t1t2) for name, (tuids, t1t2) in TRIALS.items()}
    analyses.update({name: (kept, t1t2) for name, (kept, t1t2, _) in clean.items()})

    sets = {}
    jobs = []
    for name, (tuids, _) in analyses.items():
        qs = qa.QPTSet(tuids)
        ro = qs.readout()
        sets[name] = (qs, ro)
        jobs += qa.set_jobs(name, qs, ro) + qa.bootstrap_jobs(name, qs, ro, N_BOOT)
        if name in TRIALS:                   # one fit per run, for the full trials (reused when cleaned)
            jobs += qa.per_run_jobs(name, qs, ro)
    # all three trials pooled into one data set, with and without the outliers
    for all_name, names in ((ALL, list(TRIALS)), (ALL_OK, list(clean))):
        P = np.concatenate([sets[n][0].P for n in names])
        R = np.concatenate([sets[n][0].R for n in names])
        RO = np.concatenate([sets[n][1] for n in names])
        jobs.append(dict(path=os.path.join(qa.CACHE, f"{all_name}_meas_ra.npz"), p00=P.mean(0), ref_p00=R.mean(0),
                         n_shots=sets[names[0]][0].n_rep * len(P), raw=True, readout=RO))
    print(f"[{time.time() - t0:.0f} s] stage 1: data fits", flush=True)
    qa.run_jobs(jobs, workers=20)

    print(f"[{time.time() - t0:.0f} s] stage 2: incoherent-error simulations of each trial", flush=True)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    with ProcessPoolExecutor(max_workers=len(analyses)) as ex:
        for name in ex.map(_analyze, [(n, t, c) for n, (t, c) in analyses.items()]):
            print(f"  {name} done", flush=True)

    print(f"[{time.time() - t0:.0f} s] stage 3: estimator simulations (cleaned trials)", flush=True)
    jobs = []
    for name, (kept, t1t2, _) in clean.items():
        res = qa.analyze(name, kept, t1t2, verbose=False)
        label = f"{name}_est_zz{np.degrees(res['phi_zz_sim']):+.2f}"
        jobs += qa.estimator_sim_jobs(label, res["model"], res["readout"])
    qa.run_jobs(jobs, workers=20)
    print(f"[{time.time() - t0:.0f} s] done", flush=True)


if __name__ == "__main__":
    main()
