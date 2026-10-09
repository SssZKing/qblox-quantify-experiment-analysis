import numpy as np
import pytest
import xarray as xr

from qqea.tomography import (
    T2rho, T2t, calculate_density_matrix, compute_stokes, joint_probabilities, mle_rho,
    project_rho, rearrange_stokes, rho2T, rho_from_stokes, t2T,
)

# Setting order of the state tomography schedule (Q1 basis, Q2 basis).
SETTINGS = ["ZX", "ZY", "ZZ", "XZ", "XX", "XY", "YZ", "YX", "YY"]


def _dataset(shots_per_setting):
    """Two-qubit dataset from a list of (q1, q2) shot lists, one list per setting."""
    y0 = np.array([[s[0] for s in shots] for shots in shots_per_setting]).T
    y1 = np.array([[s[1] for s in shots] for shots in shots_per_setting]).T
    dims = ("repetition", "acq_index_0")
    return xr.Dataset({"y0": (dims, y0.astype(float)), "y1": (dims, y1.astype(float))})


def _random_dataset(rng, n_shots=500, n_settings=9):
    dims = ("repetition", "acq_index_0")
    return xr.Dataset({
        "y0": (dims, rng.integers(0, 2, (n_shots, n_settings)).astype(float)),
        "y1": (dims, rng.integers(0, 2, (n_shots, n_settings)).astype(float)),
    })


def test_bell_state_is_reconstructed_exactly():
    # |Phi+> = (|00> + |11>)/sqrt(2): <XX> = 1, <YY> = -1, <ZZ> = 1, every other Pauli 0.
    uniform = [(0, 0), (0, 1), (1, 0), (1, 1)]
    even = [(0, 0), (1, 1)] * 2
    odd = [(0, 1), (1, 0)] * 2
    shots = [even if s in ("ZZ", "XX") else odd if s == "YY" else uniform for s in SETTINGS]
    rho, stokes = calculate_density_matrix(_dataset(shots))

    phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    np.testing.assert_allclose(rho, np.outer(phi, phi), atol=1e-12)
    expected = np.zeros(15)
    expected[[4, 9, 14]] = [1, -1, 1]       # XX, YY, ZZ
    np.testing.assert_allclose(stokes, expected, atol=1e-12)


@pytest.mark.parametrize("with_cm", [False, True])
def test_density_matrix_matches_legacy(legacy, with_cm):
    rng = np.random.default_rng(2)
    ds = _random_dataset(rng)
    cm = None
    if with_cm:
        cm = np.eye(4) * 0.9 + rng.uniform(0, 0.05, (4, 4))
        cm /= cm.sum(axis=0, keepdims=True)
    rho, stokes = calculate_density_matrix(ds, cm)
    rho_old, stokes_old = legacy.calculate_density_matrix(ds, cm)
    np.testing.assert_allclose(rho, rho_old, atol=1e-12)
    np.testing.assert_allclose(stokes, stokes_old, atol=1e-12)


def test_wrong_number_of_settings_is_rejected():
    with pytest.raises(ValueError, match="9 tomography settings"):
        calculate_density_matrix(_random_dataset(np.random.default_rng(0), n_settings=8))


def test_joint_probabilities_sum_to_one():
    P = joint_probabilities(_random_dataset(np.random.default_rng(3)))
    assert P.shape == (4, 9)
    np.testing.assert_allclose(P.sum(axis=0), 1)


def test_stokes_round_trip_and_plot_order(legacy):
    s = np.random.default_rng(4).uniform(-0.5, 0.5, 15)
    np.testing.assert_allclose(compute_stokes(rho_from_stokes(s)), s, atol=1e-12)
    np.testing.assert_array_equal(rearrange_stokes(s), legacy.rearrange_stokes(s))


def test_cholesky_round_trip(legacy):
    rng = np.random.default_rng(5)
    t = rng.normal(size=16)
    T = t2T(t)
    np.testing.assert_allclose(T, np.tril(T))
    np.testing.assert_allclose(T2t(T), t)
    np.testing.assert_allclose(T2rho(T), legacy.T2rho(legacy.t2T(t)))
    rho = T2rho(T)
    np.testing.assert_allclose(T2rho(rho2T(rho)), rho, atol=1e-10)


def test_project_rho_agrees_with_least_squares_mle():
    # a Bell state with a negative eigenvalue along |01> and a small traceless Hermitian error,
    # like a noisy linear inversion
    rng = np.random.default_rng(6)
    phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    psi = np.array([0, 1, 0, 0])
    H = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = (H + H.conj().T) / 2
    H -= np.trace(H) / 4 * np.eye(4)
    rho_lin = 1.04 * np.outer(phi, phi) - 0.04 * np.outer(psi, psi) + 0.01 * H / np.linalg.norm(H)
    assert np.linalg.eigvalsh(rho_lin).min() < 0

    rho_proj = project_rho(rho_lin)
    assert np.linalg.eigvalsh(rho_proj).min() > -1e-12
    np.testing.assert_allclose(np.trace(rho_proj), 1, atol=1e-12)
    np.testing.assert_allclose(rho_proj, mle_rho(rho_lin), atol=1e-3)
