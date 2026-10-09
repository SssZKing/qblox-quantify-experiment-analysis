"""Shared fixtures for the qqea.tomography tests.

``legacy`` is a frozen copy of the original tomography_tools.py (tests/tomography/legacy/), so the
rebuilt modules can be checked against it number for number. It is imported from this repo only;
nothing outside the repo is read.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

LEGACY_PATH = Path(__file__).parent / "legacy" / "tomography_tools.py"


@pytest.fixture(scope="session")
def legacy():
    spec = importlib.util.spec_from_file_location("legacy_tomography_tools", LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def gate_unitary():
    """An asymmetric partial iSWAP (U != U^T), so Kraus-transpose mistakes show up."""
    from qqea.tomography import iSWAP_RPE
    return iSWAP_RPE([1.2, 0.0, 0.3, -0.2, -0.5])


@pytest.fixture(scope="session")
def noisy_choi(gate_unitary):
    """Choi matrix (qutip convention, Tr = 4) of the gate followed by 5% depolarization."""
    choi_u = qt.to_choi(qt.to_super(gate_unitary)).full()
    return 0.95 * choi_u + 0.05 * np.eye(16) / 4


@pytest.fixture(scope="session")
def noisy_p00(noisy_choi):
    """(p00 sampled with 2000 shots per setting, the exact p00, n_shots)."""
    from qqea.tomography import forward_matrix
    n_shots = 2000
    p_true = np.clip(np.real(forward_matrix() @ noisy_choi.ravel()), 0, 1)
    rng = np.random.default_rng(1)
    p00 = rng.binomial(n_shots, p_true) / n_shots
    return p00, p_true, n_shots

