"""Standalone GST analysis + report generation for the custom iSWAP_RPE model.

Reproduces the "8.2 Two-Qubit iSWAP" section of SNL315.ipynb (model construction, germs,
fiducials, experiment design, dataset conversion, run_stdpractice_gst, HTML report) as a
plain script, so a GST analysis + report can be regenerated for any already-collected
dataset without opening Jupyter.

This only re-analyzes an already-collected dataset by tuid -- it does not talk to any
instrument/hardware. Run from the qblox_env interpreter:

    python run_gst_analysis.py \
        --tuid 20260821-170627 --theta-p-deg 0
    python run_gst_analysis.py \
        --tuid <tuid> --theta-p-deg 90

ALL ANGLES ON THE COMMAND LINE ARE IN DEGREES (--theta-p-deg, --phi-zz-deg).

Two gate settings are supported, each with its own germ set (see
GERM_STRINGS_BY_THETA_P_DEG): theta_p = 0 deg (the "idle reference" gate) and
theta_p = 90 deg (= pi/2, the full iSWAP). --theta-p-deg picks which germ set is used, and
any other angle is rejected rather than silently reusing the wrong set.

Other defaults: phi_zz=-80.75deg, fiducials from smq2Q_XYICNOT, maxLengths=[1,2,4],
modes=['full','TP','Target'].
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


import pygsti  # noqa: E402
from pygsti.modelmembers.povms import UnconstrainedPOVM  # noqa: E402
from pygsti.modelpacks import smq2Q_XYICNOT  # noqa: E402
from quantify_core.data.handling import load_dataset, set_datadir  # noqa: E402
from snl_qblox.tomography_tools import (  # noqa: E402
    concat_gst_datasets,
    ds_from_dataset_txt,
    gst_ds_to_pygsti_dataset_txt,
    iSWAP_RPE,
)

# quantify data directory; set QBLOX_DATADIR to override (default: SNL315/CD1)
DATA_DIR = os.environ.get(
    "QBLOX_DATADIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CD1"))
# reports go next to this script, wherever it is run from
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GST_reports")

# Hardcoded germ sets from SNL315.ipynb, each the exact output of a gsel.find_germs run and
# each verified amplificationally complete (gsel.test_germ_set_infl) for the theta_p it is
# keyed by, at phi_zz=-80.75deg.
#
# Keyed by theta_p in DEGREES because germ sets are NOT transferable across theta_p: the
# search is driven by the target model's Jacobian, so a set found at one theta_p generally
# fails completeness at another (empirically checked -- the 90deg set fails at 45deg and at
# ~5.7deg). Hence one set per supported angle, selected by germs_for_theta_p() below rather
# than a single shared list.
GERM_STRINGS_BY_THETA_P_DEG = {
    # theta_p = pi/2 -- the full iSWAP
    90.0: [
        "Gi:0@(0,1)",
        "Gi:1@(0,1)",
        "Gxpi2:0@(0,1)",
        "Gxpi2:1@(0,1)",
        "Gypi2:0@(0,1)",
        "Gypi2:1@(0,1)",
        "Giswap:0:1@(0,1)",
        "Gxpi2:0Gxpi2:1Gypi2:0Gypi2:0Gypi2:1Gxpi2:1@(0,1)",
        "Gxpi2:0Gxpi2:0Gxpi2:1Gypi2:1Gxpi2:1@(0,1)",
        "Gi:0Gxpi2:0Giswap:0:1Giswap:0:1Giswap:0:1Gxpi2:0@(0,1)",
        "Gxpi2:0Gypi2:0Giswap:0:1Giswap:0:1Giswap:0:1Gypi2:0@(0,1)",
        "Gxpi2:1Gxpi2:1Giswap:0:1Giswap:0:1Giswap:0:1Gypi2:0@(0,1)",
        "Gxpi2:0Gxpi2:0Giswap:0:1Gypi2:1Giswap:0:1Gypi2:1@(0,1)",
        "Gxpi2:1Gypi2:0Giswap:0:1Giswap:0:1Gypi2:0@(0,1)",
        "Gxpi2:0Giswap:0:1Giswap:0:1Gypi2:0Gypi2:1Gypi2:1@(0,1)",
        "Gi:0Gypi2:0Gypi2:0Giswap:0:1Gxpi2:1@(0,1)",
        "Gi:0Gypi2:1Giswap:0:1Gxpi2:0Gxpi2:0@(0,1)",
        "Gxpi2:0Gxpi2:1Gxpi2:1Gypi2:0Giswap:0:1Giswap:0:1@(0,1)",
        "Gxpi2:1Giswap:0:1Giswap:0:1Giswap:0:1Gypi2:0Gypi2:0@(0,1)",
        "Gxpi2:0Giswap:0:1Giswap:0:1Gypi2:1Giswap:0:1Gxpi2:1@(0,1)",
        "Gxpi2:0Gxpi2:1Gypi2:0Gypi2:0@(0,1)",
    ],
    # theta_p = 0 -- the "idle reference" gate
    0.0: [
        "Gi:0@(0,1)",
        "Gi:1@(0,1)",
        "Gxpi2:0@(0,1)",
        "Gxpi2:1@(0,1)",
        "Gypi2:0@(0,1)",
        "Gypi2:1@(0,1)",
        "Giswap:0:1@(0,1)",
        "Gxpi2:0Gxpi2:1Gypi2:0Gypi2:0Gypi2:1Gxpi2:1@(0,1)",
        "Gi:0Gypi2:0Gypi2:0Gxpi2:0Gypi2:1Gypi2:1@(0,1)",
        "Gxpi2:0Gxpi2:0Gxpi2:1Gypi2:1Gypi2:1@(0,1)",
        "Gxpi2:1Gypi2:0Gypi2:1Giswap:0:1Gypi2:1@(0,1)",
        "Gxpi2:0Gypi2:1Giswap:0:1Giswap:0:1Gypi2:1@(0,1)",
        "Gxpi2:0Gypi2:0Giswap:0:1Giswap:0:1Gxpi2:1Gypi2:0@(0,1)",
        "Gxpi2:1Gxpi2:1Gypi2:0Giswap:0:1Giswap:0:1Giswap:0:1@(0,1)",
        "Gxpi2:1Gypi2:0Giswap:0:1Giswap:0:1Gypi2:0@(0,1)",
        "Gxpi2:0Gypi2:1Gxpi2:1Giswap:0:1Giswap:0:1Gxpi2:1@(0,1)",
    ],
}
# Tolerance for matching --theta-p-deg against a key above.
THETA_P_MATCH_TOL_DEG = 1e-6
MAX_LENGTHS = [1, 2, 4]
GATE_NAMES = ["Gi", "Gxpi2", "Gypi2", "Giswap"]
AVAILABILITY = {
    "Gi": [(0,), (1,)],
    "Gxpi2": [(0,), (1,)],
    "Gypi2": [(0,), (1,)],
    "Giswap": [(0, 1)],
}


def germs_for_theta_p(theta_p_deg: float) -> list[str]:
    """Pick the germ set calibrated for this theta_p (degrees).

    Errors rather than falling back to the other set: a germ set found at one theta_p is
    generally not amplificationally complete at another, and running GST against an
    incomplete set yields an unreliable fit rather than an obvious failure.
    """
    for key, germ_strings in GERM_STRINGS_BY_THETA_P_DEG.items():
        if abs(theta_p_deg - key) < THETA_P_MATCH_TOL_DEG:
            return germ_strings
    supported = ", ".join(f"{k:g}" for k in sorted(GERM_STRINGS_BY_THETA_P_DEG))
    raise ValueError(
        f"No germ set for theta_p={theta_p_deg:g} deg. Supported: {supported} deg. "
        "Germ sets are theta_p-specific -- to use another angle, run gsel.find_germs for "
        "that model first and add the result to GERM_STRINGS_BY_THETA_P_DEG."
    )


def build_model(theta_p_deg: float, phi_zz_deg: float, simulator: str = "map"):
    """Build the custom iSWAP_RPE 2-qubit model. Both angles in DEGREES.

    simulator='map' is ~18x faster than 'matrix' for the Jacobian evaluations that dominate
    GST fit time on this model (verified empirically: bulk_dprobs on 300 circuits:
    matrix=3.71s, map=0.21s, with bulk_probs agreeing to machine precision, 4e-16)."""
    custom_iswap_unitary = iSWAP_RPE(
        (np.deg2rad(theta_p_deg), 0, 0, 0, np.deg2rad(phi_zz_deg))
    ).full()
    pspec = pygsti.processors.QubitProcessorSpec(
        num_qubits=2,
        gate_names=GATE_NAMES,
        nonstd_gate_unitaries={"Giswap": custom_iswap_unitary},
        availability=AVAILABILITY,
    )
    mdl = pygsti.models.create_explicit_model(pspec, simulator=simulator, ideal_gate_type="full")

    # Workaround for a pyGSTi 0.9.14.3 bug: set_all_parameterizations('CPTP'/'CPTPLND')
    # crashes for a model whose POVM is a ComputationalBasisPOVM (pyGSTi's own default)
    # with "local variable 'base_items' referenced before assignment" -- a real bug in
    # pygsti/modelmembers/povms/__init__.py's Lindblad-conversion branch, reproduces
    # identically for pyGSTi's own built-in gates too. Swapping to an equivalent
    # UnconstrainedPOVM (verified bit-for-bit identical effects) sidesteps it, and has
    # been verified to have NO effect on 'full'/'full TP'/'TP'/'Target' modes -- it only
    # matters if you add 'CPTP'/'CPTPLND' to --modes later.
    old_povm = mdl.povms["Mdefault"]
    mdl.povms["Mdefault"] = UnconstrainedPOVM(dict(old_povm.items()), old_povm.evotype, old_povm.state_space)
    return mdl


def build_experiment_design(mdl, germ_strings):
    prep_fiducials = smq2Q_XYICNOT.prep_fiducials()
    meas_fiducials = smq2Q_XYICNOT.meas_fiducials()
    germs = [pygsti.circuits.Circuit(s) for s in germ_strings]
    circuits = pygsti.circuits.create_lsgst_circuits(mdl, prep_fiducials, meas_fiducials, germs, MAX_LENGTHS)
    return prep_fiducials, meas_fiducials, germs, circuits


def test_germ_completeness(mdl, germs) -> bool:
    """gsel.test_germ_set_infl() needs sim.dproduct(), which only MatrixForwardSimulator
    implements (MapForwardSimulator deliberately avoids building dense process matrices).
    Run the check against a 'matrix'-simulator copy regardless of what mdl itself uses."""
    from pygsti.algorithms import germselection as gsel

    mdl_matrix = mdl.copy()
    mdl_matrix.sim = "matrix"
    return gsel.test_germ_set_infl(mdl_matrix, germs)


def auto_mem_limit_bytes(fraction: float = 0.7) -> int:
    try:
        import psutil

        available = psutil.virtual_memory().available
        return int(available * fraction)
    except Exception:
        return 4 * 1024**3  # 4GB fallback if psutil isn't available


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tuid",
        default="20260819-154652",
        help="Quantify dataset tuid to analyze. Accepts a comma-separated list of repeat "
        "runs of the SAME experiment design, which are pooled shot-wise before fitting.",
    )
    parser.add_argument(
        "--theta-p-deg",
        type=float,
        default=0.0,
        help="iSWAP_RPE theta_p in DEGREES. Selects the matching germ set, so it must be one "
        "of the calibrated angles: 0 (idle reference) or 90 (full iSWAP).",
    )
    parser.add_argument("--phi-zz-deg", type=float, default=-80.75, help="iSWAP_RPE phi_zz parameter (degrees).")
    parser.add_argument(
        "--modes", default="full,TP,Target", help="Comma-separated run_stdpractice_gst modes."
    )
    parser.add_argument("--report-name", default=None, help="Report directory name (default: derived from tuid).")
    parser.add_argument("--mem-limit-gb", type=float, default=None, help="Override auto-detected memory limit (GB).")
    parser.add_argument(
        "--simulator", default="map", choices=["map", "matrix"], help="Forward simulator (map is ~18x faster here)."
    )
    args = parser.parse_args()

    t_start = time.time()
    set_datadir(DATA_DIR)

    germ_strings = germs_for_theta_p(args.theta_p_deg)

    print(
        f"Building model: theta_p={args.theta_p_deg:g}deg, phi_zz={args.phi_zz_deg}deg, "
        f"simulator={args.simulator}"
    )
    mdl = build_model(args.theta_p_deg, args.phi_zz_deg, simulator=args.simulator)
    print(f"  model parameters: {mdl.num_params}")

    prep_fiducials, meas_fiducials, germs, circuits = build_experiment_design(mdl, germ_strings)
    print(f"  prep fiducials: {len(prep_fiducials)}, meas fiducials: {len(meas_fiducials)}")
    print(f"  germs: {len(germs)}, total circuits: {len(circuits)}")

    if not test_germ_completeness(mdl, germs):
        print(
            f"WARNING: the germ set keyed to theta_p={args.theta_p_deg:g}deg is NOT "
            "amplificationally complete for this model. Results may be unreliable -- "
            "re-run germ selection for this specific (theta_p, phi_zz) before trusting the fit."
        )

    tuids = [t.strip() for t in args.tuid.split(",") if t.strip()]
    print(f"Loading dataset{'s' if len(tuids) > 1 else ''} {', '.join(tuids)} ...")
    if len(tuids) == 1:
        gst_ds = load_dataset(tuids[0])
    else:
        # Pooling happens at the SHOT level, so the counting below is unchanged and the
        # pooled counts are exactly the sum of the parts. concat_gst_datasets runs a
        # homogeneity test and warns if the runs drifted -- pooling then converts that
        # drift into apparent decoherence, since an average of unitaries is not unitary.
        gst_ds = concat_gst_datasets([load_dataset(t) for t in tuids])
    print(f"  sizes: {dict(gst_ds.sizes)}")

    dataset_txt = gst_ds_to_pygsti_dataset_txt(gst_ds, circuits, mode="2q")
    ds = ds_from_dataset_txt(dataset_txt)

    modes = args.modes.split(",")
    mem_limit = int(args.mem_limit_gb * 1024**3) if args.mem_limit_gb else auto_mem_limit_bytes()
    print(f"Running run_stdpractice_gst with modes={modes}, mem_limit={mem_limit / 1024**3:.1f}GB ...")
    t_gst = time.time()
    results = pygsti.run_stdpractice_gst(
        ds,
        mdl,
        prep_fiducials,
        meas_fiducials,
        germs,
        MAX_LENGTHS,
        modes=modes,
        mem_limit=mem_limit,
        verbosity=2,
    )
    print(f"GST fit completed in {time.time() - t_gst:.1f}s")

    for mode in modes:
        if mode == "Target":
            continue
        # run_stdpractice_gst preserves mode strings verbatim as estimate keys (no merging
        # of e.g. 'TP' into 'full TP' happens anywhere in pygsti/protocols/gst.py -- verified).
        if mode in results.estimates:
            est_mdl = results.estimates[mode].models["stdgaugeopt"]
            print(f"2DeltaLogL(estimate[{mode}], data): {pygsti.tools.two_delta_logl(est_mdl, ds)}")
    print(f"2DeltaLogL(ideal target, data): {pygsti.tools.two_delta_logl(mdl, ds)}")

    stem = tuids[0] if len(tuids) == 1 else f"{tuids[0]}+{len(tuids) - 1}more"
    report_name = (
        args.report_name
        or f"{stem}_2Q_iSWAP_theta{args.theta_p_deg:g}deg_phi{args.phi_zz_deg:g}deg"
    )

    report_dir = os.path.join(REPORTS_DIR, report_name).replace("\\", "/")
    print(f"Building HTML report -> {report_dir} ...")
    pygsti.report.construct_standard_report(
        results,
        title=f"Q1Q2 iSWAP_RPE GST Report ({args.tuid})",
        verbosity=1,
    ).write_html(report_dir, auto_open=False, verbosity=1)

    # Save the full Results object (all estimates, all gauge-optimized models) alongside the
    # report, so it can be loaded back directly later -- the HTML report itself is a one-way
    # static rendering (main.html + an offline/ assets bundle) with no embedded model data.
    # Written after write_html (not before) in case write_html ever clears its target dir.
    results_dir = f"{report_dir}/gst_results"
    print(f"Saving results (loadable) -> {results_dir} ...")
    results.write(results_dir)
    print(
        "  reload later with: "
        f"pygsti.protocols.ModelEstimateResults.from_dir('{results_dir}', name='{results.name}')"
    )

    print(f"Report written to: {report_dir}")
    print(f"Results written to: {results_dir}")
    print(f"Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
