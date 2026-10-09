"""A QPT series in blocks: a multiplexed readout calibration, a Multi Qubit T1 and T2 run, then a
block of QPT runs (Clifford 5760, identity reference), repeated -- e.g. 2026-09-30 from 19:17:
RO cal | T1/T2 | 5 x QPT | RO cal | T1/T2 | 5 x QPT | ...

Series(start, name, stop=None, gate="iswap") reads the data directory every time it is built, so a
series that is still running is picked up as far as it has been saved (datasets with start <= tuid <
stop); gate="idle" is the same schedule with the pump off (angles read from the diagonal, phi_p=None;
T1/T2 limit of an idle, theta_p = 0); fit() runs only the fits that are not cached yet
(cache/<name>_*.npz: one per run, one per block -- keyed by its run count, so a growing block is
refitted -- and one for the whole series).

Times: a QPT dataset is saved, and its tuid stamped, when the run ENDS (~4.9 min after it starts);
calibration and T1/T2 tuids are their start times.
"""
import glob
import os
import time

import numpy as np
from quantify_core.data.handling import load_dataset

import qpt_analysis as qa  # also puts bell_state/ on the path
import bell_analysis as ba
from snl_qblox.tomography_tools import iSWAP_RPE

# quantify data directory; set QBLOX_DATADIR to override (default: SNL315/CD1)
DATADIR = os.environ.get(
    "QBLOX_DATADIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CD1"))
QPT_KIND = "QPT idx 5760"
CAL_KIND = "Multiplexed Readout Calibration"
T1T2_KIND = "qubit1 qubit2 T1 and T2"


def _folders(start, kind, stop=None, datadir=DATADIR):
    """{tuid: folder} of the datasets named ``kind`` with start <= tuid < stop (across days)."""
    out = {}
    for day in sorted(glob.glob(os.path.join(datadir, "2" + "[0-9]" * 7))):
        if os.path.basename(day) < start[:8]:
            continue
        for d in os.listdir(day):
            if d[:26] >= start and (stop is None or d[:26] < stop) and d[27:] == kind:
                out[d[:26]] = os.path.join(day, d)
    return dict(sorted(out.items()))


def _finished(folder, settle):
    """A dataset still being written changes; keep it once dataset.hdf5 is `settle` s old."""
    f = os.path.join(folder, "dataset.hdf5")
    return os.path.exists(f) and time.time() - os.path.getmtime(f) > settle


def reference_z(R, n_rep):
    """Per run: rms deviation of the identity-reference P00 from the median of the runs given, in
    units of one run's binomial shot noise (qa.reference_z, for a block of runs)."""
    med = np.median(R, 0)
    sn = np.sqrt(np.clip(med * (1 - med), 1 / n_rep, None) / n_rep)
    return np.sqrt(np.mean(((R - med) / sn) ** 2, axis=1))


class Series:
    def __init__(self, start, name, stop=None, gate="iswap", exclude=(), datadir=DATADIR, settle=60):
        """exclude: QPT tuids (or their first 15 characters) to leave out."""
        if gate not in ("iswap", "idle"):
            raise ValueError(f"gate must be 'iswap' or 'idle', not {gate!r}")
        self.start, self.stop, self.name, self.gate = start, stop, name, gate
        self.phi_p = 0.0 if gate == "iswap" else None      # qa.summarize / extract_angles convention
        qpt = [t for t, f in _folders(start, QPT_KIND, stop, datadir).items()
               if _finished(f, settle) and t not in exclude and t[:15] not in exclude]
        self.cals = list(_folders(start, CAL_KIND, stop, datadir))
        # a T1/T2 run stopped part-way keeps NaN in the points it did not reach: only complete runs
        self.t1t2 = [t for t, f in _folders(start, T1T2_KIND, stop, datadir).items()
                     if _finished(f, settle) and np.isfinite(load_dataset(t).y0.values).all()]
        self.qs = qa.QPTSet(qpt)
        self.tuids = self.qs.tuids
        self.times = self.qs.times                      # end of each QPT run
        self.n_rep = self.qs.n_rep

        # block = the runs after one readout calibration
        cal_of = [max(c for c in self.cals if c < t) for t in self.tuids]
        self.blocks = []
        for c in dict.fromkeys(cal_of):
            idx = [i for i, ci in enumerate(cal_of) if ci == c]
            first, last = self.tuids[idx[0]], self.tuids[idx[-1]]
            before = [t for t in self.t1t2 if c < t < first]
            # the T1/T2 run right after the block (within 15 min of its last run; not a later, unrelated one)
            after = [t for t in self.t1t2 if t > last
                     and (ba.tuid_time(t) - ba.tuid_time(last)).total_seconds() < 15 * 60]
            self.blocks.append(dict(cal=c, idx=idx, tuids=[self.tuids[i] for i in idx],
                                    t1t2_before=before[-1] if before else None,
                                    t1t2_after=after[0] if after else None))
        # measured readout of every run: IQ shots of its block's calibration, its own threshold
        ro, cache = [], {}
        for b in self.blocks:
            for i in b["idx"]:
                if b["cal"] not in cache:
                    cache[b["cal"]] = ba.readout_from_iq(b["cal"], self.tuids[i])
                ro.append(cache[b["cal"]])
        self.readout = np.array(ro)                      # (run, qubit, [F_g, F_e])
        self.cal_readout = {c: np.array(v) for c, v in cache.items()}
        self.ref_z = np.concatenate([reference_z(self.qs.R[b["idx"]], self.n_rep) for b in self.blocks])

    def __len__(self):
        return len(self.tuids)

    # ------------------------------------------------------------------ fits
    def _path(self, label):
        return os.path.join(qa.CACHE, f"{self.name}_{label}_ra.npz")

    def run_path(self, i):
        return self._path(self.tuids[i])

    def block_path(self, b):
        return self._path(f"block_{b['cal']}_n{len(b['idx'])}")

    def all_path(self):
        return self._path(f"all_n{len(self)}")

    def jobs(self):
        qs, ro = self.qs, self.readout
        jobs = [dict(path=self.run_path(i), p00=qs.P[i], ref_p00=qs.R[i], n_shots=self.n_rep, readout=ro[i])
                for i in range(len(self))]
        for b in self.blocks:
            p, r = qs.pooled(b["idx"])
            jobs.append(dict(path=self.block_path(b), p00=p, ref_p00=r, n_shots=qs.n_shots(b["idx"]),
                             readout=ro[b["idx"]]))
        p, r = qs.pooled()
        jobs.append(dict(path=self.all_path(), p00=p, ref_p00=r, n_shots=qs.n_shots(), readout=ro))
        return jobs

    def fit(self, workers=20):
        """Readout-aware, SPAM-corrected fits not cached yet: every run, every block, the series."""
        qa.run_jobs(self.jobs(), workers=workers)

    def run_summaries(self):
        return [qa.summarize(qa.load_recon(self.run_path(i)), phi_p=self.phi_p) for i in range(len(self))]

    def block_summaries(self):
        return [qa.summarize(qa.load_recon(self.block_path(b)), phi_p=self.phi_p) for b in self.blocks]

    def all_summary(self):
        return qa.summarize(qa.load_recon(self.all_path()), phi_p=self.phi_p)

    # ------------------------------------------------------------------ coherence and the T1/T2 limit
    def coherence(self, drop_tls=True):
        """{T1/T2 tuid: ba.coherence_times} for every T1/T2 run of the series."""
        return {t: ba.coherence_times(t, drop_tls=drop_tls) for t in self.t1t2}

    def block_coherence(self, b, coh):
        """T1, T2 echo (us) of a block: mean of the T1/T2 runs right before and right after it."""
        ts = [t for t in (b["t1t2_before"], b["t1t2_after"]) if t is not None]
        return {q: {k: float(np.mean([coh[t][q][k][0] for t in ts])) for k in ("T1", "T2echo")}
                for q in ("Q1", "Q2")}, ts

    def limit(self, c, phi_zz_deg):
        """F_pro of the gate block with only T1, T2 echo (QPTModel), for the gate iSWAP_RPE(90 deg, 0,
        0, 0, phi_zz) -- or theta_p = 0 for an idle -- the same model as qa.analyze."""
        t_1q, t_gate = self.qs.durations()
        U = iSWAP_RPE([np.pi / 2 if self.gate == "iswap" else 0.0, 0.0, 0.0, 0.0, np.radians(phi_zz_deg)])
        T1 = [c[q]["T1"] * 1e-6 for q in ("Q1", "Q2")]
        T2 = [c[q]["T2echo"] * 1e-6 for q in ("Q1", "Q2")]
        return qa.QPTModel(T1, T2, t_1q, t_gate, U_target=U).process_fidelity()
