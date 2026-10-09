import numpy as np
import pytest
import qutip as qt

from qqea.tomography import deg, extract_angles, fit_iswap_rpe_to_chi, iSWAP_RPE

PARAMS = dict(theta_p=0.4, phi_p=0.3, theta_1=0.2, theta_2=-0.5, phi_zz=0.7)


def _U():
    return iSWAP_RPE([PARAMS[k] for k in ("theta_p", "phi_p", "theta_1", "theta_2", "phi_zz")])


@pytest.mark.parametrize("phi_p", [None, PARAMS["phi_p"]])
def test_extract_angles_recovers_parameters(phi_p):
    out = extract_angles(_U(), phi_p=phi_p)
    for k, v in PARAMS.items():
        assert out[k] == pytest.approx(v, abs=1e-12), k
    assert out["theta_sum"] == pytest.approx(PARAMS["theta_1"] + PARAMS["theta_2"], abs=1e-12)


def test_extract_angles_matches_legacy(legacy):
    U = _U()
    for phi_p in (None, 0, 0.3):
        new, old = extract_angles(U, phi_p=phi_p), legacy.extract_angles(U, phi_p=phi_p)
        assert new.keys() == old.keys()
        for k in new:
            np.testing.assert_allclose(new[k], old[k], atol=1e-14)


def test_fit_iswap_rpe_to_chi_matches_legacy(legacy):
    chi = qt.to_chi(qt.to_super(iSWAP_RPE([np.pi / 2, 0.0, 0.3, -0.2, -0.5])))
    angles, fid, _ = fit_iswap_rpe_to_chi(chi)
    angles_old, fid_old, _ = legacy.fit_iswap_rpe_to_chi(chi)
    assert fid == pytest.approx(1, abs=1e-6)
    assert fid == pytest.approx(fid_old, abs=1e-12)
    for k in angles:
        np.testing.assert_allclose(angles[k], angles_old[k], atol=1e-10)


def test_deg():
    out = deg({"a": np.pi, "b": None, "c": float("nan")})
    assert out["a"] == pytest.approx(180)
    assert np.isnan(out["b"]) and np.isnan(out["c"])
