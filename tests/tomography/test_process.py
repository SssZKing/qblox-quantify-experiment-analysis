import numpy as np
import pytest
import qutip as qt

from qqea.tomography import (
    PMatrix, QPT_fidelity, as_chi, chi_from_p00_linear, chi_to_unitary, closest_unitary,
    fit_chi2, forward_matrix, func_c2, get_spam_corrected_gate, map_from_bloch_state_to_pauli_basis2,
    mle_chi_from_p00, mle_chi_spam_corrected, proc_fid_to_unitary, project_to_cp_chi, ptm,
    readout_povm, reconstruct, unitary_pauli_coeffs,
)
from qqea.tomography.process import _inversion_data, _linear_init

from ._helpers import choi_to_chi


def _choi(chi):
    return qt.to_choi(chi).full()


# ---------------------------------------------------------------- linear inversion

def test_func_c2_table_matches_legacy(legacy):
    for i in range(-1, 18):
        for j in range(-1, 7):
            for k in range(-1, 7):
                assert func_c2(i, j, k) == legacy.func_c2(i, j, k), (i, j, k)


def test_pmatrix_npy_equals_legacy_pickle(legacy):
    assert PMatrix.dtype == legacy.PMatrix.dtype
    np.testing.assert_array_equal(PMatrix, legacy.PMatrix)


def test_inversion_data_matches_legacy_loop(legacy):
    arr = np.random.default_rng(7).uniform(size=(6, 6, 6, 6))
    for q, n in [(0, 0), (1, 5), (5, 1), (10, 15), (15, 15)]:
        np.testing.assert_allclose(map_from_bloch_state_to_pauli_basis2(q, n, arr),
                                   legacy.map_from_bloch_state_to_pauli_basis2(q, n, arr), atol=1e-12)
    old = [legacy.map_from_bloch_state_to_pauli_basis2(v // 16, v % 16, arr) for v in range(256)]
    np.testing.assert_allclose(_inversion_data(arr), old, atol=1e-12)


def test_linear_inversion_matches_legacy(legacy, noisy_p00):
    p00 = noisy_p00[0]
    np.testing.assert_allclose(chi_from_p00_linear(p00).full(),
                               legacy.chi_from_p00_linear(p00).full(), atol=1e-10)


# ---------------------------------------------------------------- forward model and MLE

def test_forward_matrix_matches_legacy(legacy):
    np.testing.assert_array_equal(forward_matrix(), legacy._mle_forward_matrix())


def test_readout_forward_matrix():
    np.testing.assert_array_equal(forward_matrix([[1, 1], [1, 1]]), forward_matrix())
    readout = [[0.97, 0.90], [0.95, 0.88]]
    np.testing.assert_allclose(readout_povm(readout),
                               [0.97 * 0.95, 0.97 * 0.12, 0.10 * 0.95, 0.10 * 0.12])
    np.testing.assert_allclose(readout_povm([readout, readout]), readout_povm(readout))

    # identity process: no rotations reads |00> as 00 with F_g1 F_g2; X180 on both prepares
    # |11> and reads it as 00 with (1 - F_e1)(1 - F_e2). Row = prep * 36 + meas.
    choi_id = qt.to_choi(qt.to_super(qt.tensor(qt.qeye(2), qt.qeye(2)))).full()
    p = np.real(forward_matrix(readout) @ choi_id.ravel())
    np.testing.assert_allclose(p[0], 0.97 * 0.95)
    np.testing.assert_allclose(p[(1 * 6 + 1) * 36], 0.10 * 0.12)


def test_mle_recovers_simulated_channel(gate_unitary, noisy_choi, noisy_p00):
    _, p_true, n_shots = noisy_p00
    chi = mle_chi_from_p00(p_true, n_shots)
    np.testing.assert_allclose(_choi(chi), noisy_choi, atol=1e-2)
    assert proc_fid_to_unitary(chi, gate_unitary) == pytest.approx(0.95 + 0.05 / 16, abs=2e-3)
    assert fit_chi2(chi, p_true, n_shots) < 0.05


def test_linear_init_matches_legacy(legacy, noisy_p00):
    p00 = noisy_p00[0]
    np.testing.assert_allclose(_linear_init(p00).full(), legacy._linear_init(p00).full(), atol=1e-10)


# The MLE comparisons start both versions from the same point, so they run the same
# optimization; the starting points themselves are compared above.

def test_mle_matches_legacy(legacy, noisy_p00):
    p00, _, n_shots = noisy_p00
    chi0 = legacy._linear_init(p00)
    np.testing.assert_allclose(mle_chi_from_p00(p00, n_shots, chi_init=chi0).full(),
                               legacy.mle_chi_from_p00(p00, n_shots, chi_init=chi0).full(), atol=1e-6)


def test_spam_corrected_mle_matches_legacy(legacy, noisy_p00):
    p00, _, n_shots = noisy_p00
    choi_id = qt.to_choi(qt.to_super(qt.tensor(qt.qeye(2), qt.qeye(2)))).full()
    ref_chi = choi_to_chi(0.97 * choi_id + 0.03 * np.eye(16) / 4)
    chi0 = legacy._linear_init(p00)
    np.testing.assert_allclose(
        mle_chi_spam_corrected(p00, ref_chi, n_shots, chi_init=chi0).full(),
        legacy.mle_chi_spam_corrected(p00, ref_chi, n_shots, chi_init=chi0).full(), atol=1e-6)


def test_reconstruct_with_ideal_reference(gate_unitary, noisy_choi, noisy_p00):
    _, p_true, n_shots = noisy_p00
    choi_id = qt.to_choi(qt.to_super(qt.tensor(qt.qeye(2), qt.qeye(2)))).full()
    ref_p00 = np.real(forward_matrix() @ choi_id.ravel())
    rec = reconstruct(p_true, ref_p00, n_shots)
    assert set(rec) == {"ref", "corr", "raw", "U_fit"}
    assert proc_fid_to_unitary(rec["ref"], qt.tensor(qt.qeye(2), qt.qeye(2))) > 0.995
    np.testing.assert_allclose(_choi(rec["corr"]), noisy_choi, atol=1e-2)
    assert abs(proc_fid_to_unitary(rec["corr"], gate_unitary)
               - (0.95 + 0.05 / 16)) < 5e-3


# ---------------------------------------------------------------- unitaries and fidelities

def test_unitary_helpers_match_legacy(legacy, gate_unitary, noisy_choi):
    chi = choi_to_chi(noisy_choi)
    np.testing.assert_allclose(unitary_pauli_coeffs(gate_unitary),
                               legacy.unitary_pauli_coeffs(gate_unitary), atol=1e-14)
    assert proc_fid_to_unitary(chi, gate_unitary) == pytest.approx(
        legacy.proc_fid_to_unitary(chi, gate_unitary), abs=1e-12)
    U, w = closest_unitary(chi)
    U_old, w_old = legacy.closest_unitary(chi)
    np.testing.assert_allclose(U.full(), U_old.full(), atol=1e-12)
    np.testing.assert_allclose(w, w_old, atol=1e-12)
    np.testing.assert_allclose(chi_to_unitary(chi).full(), legacy.chi_to_unitary(chi).full(), atol=1e-12)


def test_proc_fid_to_unitary_is_qutip_process_fidelity(gate_unitary, noisy_choi):
    chi = choi_to_chi(noisy_choi)
    assert proc_fid_to_unitary(chi, gate_unitary) == pytest.approx(
        qt.process_fidelity(chi, gate_unitary), abs=1e-10)
    assert proc_fid_to_unitary(qt.to_chi(qt.to_super(gate_unitary)), gate_unitary) == pytest.approx(1)


def test_closest_unitary_recovers_gate(gate_unitary, noisy_choi):
    U, _ = closest_unitary(choi_to_chi(noisy_choi))
    overlap = abs(np.trace(U.full().conj().T @ gate_unitary.full())) / 4
    assert overlap == pytest.approx(1, abs=1e-10)


def test_projection_and_spam_inversion_match_legacy(legacy, noisy_p00):
    chi_lin = chi_from_p00_linear(noisy_p00[0])
    np.testing.assert_allclose(project_to_cp_chi(chi_lin).full(),
                               legacy.project_to_cp_chi(chi_lin).full(), atol=1e-10)
    choi_cp = _choi(project_to_cp_chi(chi_lin))
    assert np.linalg.eigvalsh((choi_cp + choi_cp.conj().T) / 2).min() > -1e-10

    rng = np.random.default_rng(8)
    E = np.eye(16) + 0.02 * rng.normal(size=(16, 16))
    R = np.eye(16) + 0.05 * rng.normal(size=(16, 16))
    for p in (0, 0.5):
        np.testing.assert_allclose(get_spam_corrected_gate(R, E, p),
                                   legacy.get_spam_corrected_gate(R, E, p), atol=1e-12)


def test_qpt_fidelity_matches_legacy(legacy, gate_unitary, noisy_choi):
    chi = choi_to_chi(noisy_choi).full() / 16
    chi_ideal = qt.to_chi(qt.to_super(gate_unitary)).full() / 16
    np.testing.assert_allclose(QPT_fidelity(chi, chi_ideal), legacy.QPT_fidelity(chi, chi_ideal))


def test_ptm():
    np.testing.assert_allclose(ptm(qt.to_chi(qt.to_super(qt.tensor(qt.qeye(2), qt.qeye(2))))),
                               np.eye(16), atol=1e-12)
    # X on Q1 flips the sign of every Pauli with Y or Z on Q1 (labels "Y?" and "Z?")
    R = ptm(qt.to_chi(qt.to_super(qt.tensor(qt.sigmax(), qt.qeye(2)))))
    expected = np.diag([1] * 8 + [-1] * 8)
    np.testing.assert_allclose(R, expected, atol=1e-12)
    np.testing.assert_allclose(ptm(as_chi(qt.to_chi(qt.to_super(qt.tensor(qt.sigmax(), qt.qeye(2)))).full())),
                               expected, atol=1e-12)
