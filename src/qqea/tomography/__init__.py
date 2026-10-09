"""State and process tomography, MLE reconstruction, GST dataset helpers.

Modules: ``state`` (density matrices), ``process`` (QPT), ``iswap`` (iSWAP model and angles),
``gst`` (pyGSTi datasets) and ``plotting``. Everything public is importable from here.
Importing the package does not import matplotlib (until a plotting function is used) or pyGSTi.
``qqea.tomography.tomography`` is the old flat ``tomography_tools`` namespace, kept for
``from tomography_tools import *`` style code.
"""
from ._paulis import (
    I, sigma_x, sigma_y, sigma_z, PAULI_1Q, PAULI_2Q, PAULI_LABELS, Y_PARITY,
    II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ,
)
from .state import (
    STOKES_LABELS, STOKES_PAULIS, STOKES_PLOT_ORDER,
    joint_probabilities, calculate_density_matrix, rho_from_stokes, compute_stokes,
    rearrange_stokes, extract_population_phase,
    t2T, T2t, T2rho, rho2T, mle_residuals, mle_rho, project_rho,
)
from .process import (
    PMatrix, CHI_DIMS, op_label,
    func_c2, map_from_bloch_state_to_pauli_basis2, chi_from_p00_linear, as_chi,
    get_spam_corrected_gate, project_to_cp_chi,
    readout_povm, forward_matrix, mle_chi_from_p00, mle_chi_spam_corrected, reconstruct,
    closest_unitary, chi_to_unitary, unitary_pauli_coeffs, proc_fid_to_unitary,
    QPT_fidelity, avg_gate_fidelity, ptm, fit_chi2,
)
from .iswap import iSWAP_RPE, extract_angles, fit_iswap_rpe_to_chi, deg
from .gst import (
    ds_from_dataset_txt, gst_ds_datasets_homogeneity, concat_gst_datasets,
    gst_ds_to_pygsti_dataset_txt,
)

_PLOTTING = ("qpt_plot_cmap", "plot_process_tomography2")


def __getattr__(name):
    # plotting imports matplotlib, so load it on first use only
    if name in _PLOTTING:
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "I", "sigma_x", "sigma_y", "sigma_z", "PAULI_1Q", "PAULI_2Q", "PAULI_LABELS", "Y_PARITY",
    "II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ",
    "STOKES_LABELS", "STOKES_PAULIS", "STOKES_PLOT_ORDER",
    "joint_probabilities", "calculate_density_matrix", "rho_from_stokes", "compute_stokes",
    "rearrange_stokes", "extract_population_phase",
    "t2T", "T2t", "T2rho", "rho2T", "mle_residuals", "mle_rho", "project_rho",
    "PMatrix", "CHI_DIMS", "op_label",
    "func_c2", "map_from_bloch_state_to_pauli_basis2", "chi_from_p00_linear", "as_chi",
    "get_spam_corrected_gate", "project_to_cp_chi",
    "readout_povm", "forward_matrix", "mle_chi_from_p00", "mle_chi_spam_corrected", "reconstruct",
    "closest_unitary", "chi_to_unitary", "unitary_pauli_coeffs", "proc_fid_to_unitary",
    "QPT_fidelity", "avg_gate_fidelity", "ptm", "fit_chi2",
    "iSWAP_RPE", "extract_angles", "fit_iswap_rpe_to_chi", "deg",
    "ds_from_dataset_txt", "gst_ds_datasets_homogeneity", "concat_gst_datasets",
    "gst_ds_to_pygsti_dataset_txt",
    *_PLOTTING,
]
