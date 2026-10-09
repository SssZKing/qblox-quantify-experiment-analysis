"""Pauli bases shared by the tomography modules.

Two-qubit operators are Kronecker products with Q1 (dataset ``y0``) as the first factor, in the
order II, IX, IY, IZ, XI, ..., ZZ -- the same order qutip uses for its chi matrices.
"""
import numpy as np

I = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

PAULI_1Q = [I, sigma_x, sigma_y, sigma_z]
PAULI_LABELS = [a + b for a in "IXYZ" for b in "IXYZ"]
PAULI_2Q = [np.kron(a, b) for a in PAULI_1Q for b in PAULI_1Q]

(II, IX, IY, IZ,
 XI, XX, XY, XZ,
 YI, YX, YY, YZ,
 ZI, ZX, ZY, ZZ) = PAULI_2Q

# (-1)^(number of Y factors) of each two-qubit Pauli, in PAULI_2Q order. c(U^T) = Y_PARITY * c(U)
# for the Pauli coefficients c of a unitary, so D chi D (D = diag(Y_PARITY)) swaps between the
# chi of a channel and the chi of its Kraus-transposed channel.
Y_PARITY = np.array([(-1) ** label.count("Y") for label in PAULI_LABELS])
