# The MIT License (MIT)

# Copyright (c) 2016 DiCarlo lab-QuTech-Delft University of Technology

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import contextlib
from functools import cache
from os import mkdir
from os.path import abspath, dirname, exists, join
from typing import TYPE_CHECKING
from zlib import crc32

import numpy as np

from .clifford_decompositions import epstein_efficient_decomposition

if TYPE_CHECKING:
    from collections.abc import Iterable

hash_dir = join(abspath(dirname(__file__)), "clifford_hash_tables")

"""
This file contains Clifford decompositions for the two qubit Clifford group.

The Clifford decomposition closely follows two papers:
Corcoles et al. Process verification ... Phys. Rev. A. 2013
    http://journals.aps.org/pra/pdf/10.1103/PhysRevA.87.030301
for the different classes of two-qubit Cliffords.

and
Barends et al. Superconducting quantum circuits at the ... Nature 2014
    https://www.nature.com/articles/nature13171?lang=en
for writing the cliffords in terms of CZ gates.

and
    10.1103/PhysRevX.11.021058
for writing the Cliffords in terms of iSWAP gates.

###########################################################################
2-qubit clifford decompositions

The two qubit clifford group (C2) consists of 11520 two-qubit cliffords
These gates can be subdivided into four classes.
    1. The Single-qubit like class  | 576 elements  (24^2)
    2. The CNOT-like class          | 5184 elements (24^2 * 3^2)
    3. The iSWAP-like class         | 5184 elements (24^2 * 3^2)
    4. The SWAP-like class          | 576  elements (24^2)
    --------------------------------|------------- +
    Two-qubit Clifford group C2     | 11520 elements

========= CZ Decompositions =========
1. The Single-qubit like class
    -- C1 --
    -- C1 --

2. The CNOT-like class
    --C1--•--S1--      --C1--•--S1------
          |        ->        |
    --C1--⊕--S1--      --C1--•--S1^Y90--

3. The iSWAP-like class
    --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
          |       ->        |        |
    --C1--*--S1--     --C1--•--mY90--•--S1^X90--

4. The SWAP-like class
    --C1--x--     --C1--•-mY90--•--Y90--•-------
          |   ->        |       |       |
    --C1--x--     --C1--•--Y90--•-mY90--•--Y90--

========= iSWAP Decompositions =========
1. The Single-qubit like class
    -- C1 --
    -- C1 --

2. The CNOT-like class
    --C1--•--S1--      --C1--*--X90--*--S1------
          |        ->        |       |
    --C1--⊕--S1--      --C1--*-------*--S1^Z90--

3. The iSWAP-like class
    --C1--*--S1--     
          |       
    --C1--*--S1--   

4. The SWAP-like class
    --C1--x--     --C1--*-------*-mX90--*-------
          |   ->        |       |       |
    --C1--x--     --C1--*-mX90--*-------*-mX90--


C1: element of the single qubit Clifford group
    N.B. we use the decomposition defined in Epstein et al. here

S1: element of the S1 group, a subgroup of the single qubit Clifford group

S1[0] = I
S1[1] = rY90, rX90
S1[2] = rXm90, rYm90

Important clifford indices:

        I    : Cl 0
        X90  : Cl 16
        Y90  : Cl 21
        X180 : Cl 3
        Y180 : Cl 6
        Z180 : Cl 9
        CZ   : 4368 (with CZ decomposition)
        iSWAP: 5760 (with iSWAP decomposition)

"""


# Pauli transfer matrix representation of the single qubit Clifford group C1
#
# For more information, see:
# - J. M. Chow et al., Phys. Rev. Lett. 109, 060501 (2012)
# - Epstein et al. Phys. Rev. A 89, 062321 (2014)

I = np.eye(4)  # noqa: E741

# Pauli
X = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=int)
Y = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=int)
Z = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=int)

# Exchange
S = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=int)
S2 = np.dot(S, S)

# Hadamard
H = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=int)

CZ = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=int,
)

CNOT = np.array(
    [
        [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
        [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ],
    dtype=int,
)

iSWAP = np.array(
    [
        [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
        [ 0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]
    ],
    dtype=int,
)


"""
Please see Table I of Epstein et al. Phys. Rev. A 89, 062321 (2014)
"""

C1 = [np.empty([4, 4])] * (24)
# Below, operators are ordered in time, i.e. the order in which operators should be applied to qubits so that to realize a given Clifford net gate.
# That means that the Clifford gates are equal to the matrix product of the listed operators in REVERSED order.
C1[0] = np.linalg.multi_dot([I, I, I][::-1])
C1[1] = np.linalg.multi_dot([I, I, S][::-1])
C1[2] = np.linalg.multi_dot([I, I, S2][::-1])
C1[3] = np.linalg.multi_dot([X, I, I][::-1])
C1[4] = np.linalg.multi_dot([X, I, S][::-1])
C1[5] = np.linalg.multi_dot([X, I, S2][::-1])
C1[6] = np.linalg.multi_dot([Y, I, I][::-1])
C1[7] = np.linalg.multi_dot([Y, I, S][::-1])
C1[8] = np.linalg.multi_dot([Y, I, S2][::-1])
C1[9] = np.linalg.multi_dot([Z, I, I][::-1])
C1[10] = np.linalg.multi_dot([Z, I, S][::-1])
C1[11] = np.linalg.multi_dot([Z, I, S2][::-1])

C1[12] = np.linalg.multi_dot([I, H, I][::-1])
C1[13] = np.linalg.multi_dot([I, H, S][::-1])
C1[14] = np.linalg.multi_dot([I, H, S2][::-1])
C1[15] = np.linalg.multi_dot([X, H, I][::-1])
C1[16] = np.linalg.multi_dot([X, H, S][::-1])
C1[17] = np.linalg.multi_dot([X, H, S2][::-1])
C1[18] = np.linalg.multi_dot([Y, H, I][::-1])
C1[19] = np.linalg.multi_dot([Y, H, S][::-1])
C1[20] = np.linalg.multi_dot([Y, H, S2][::-1])
C1[21] = np.linalg.multi_dot([Z, H, I][::-1])
C1[22] = np.linalg.multi_dot([Z, H, S][::-1])
C1[23] = np.linalg.multi_dot([Z, H, S2][::-1])

# S1 is a subgroup of C1 (single qubit Clifford group) used when generating C2
S1 = [
    C1[0],
    C1[1],
    C1[2],
]

# set as a module wide variable instead of argument to function for speed
# reasons
single_qubit_gate_decomposition = epstein_efficient_decomposition

# used to transform the S1 subgroup
I = C1[0]
X90 = C1[16]
Y90 = C1[21]
Z90 = C1[14]
mX90 = C1[13]
mY90 = C1[15]
mZ90 = C1[23]

# A dict containing clifford IDs with common names.
common_cliffords = {
    "I": 0,
    "X": 3,
    "Y": 6,
    "Z": 9,
    "II": 0,
    "IX": 3,
    "IY": 6,
    "IZ": 9,
    "XI": 24 * 3 + 0,
    "XX": 24 * 3 + 3,
    "XY": 24 * 3 + 6,
    "XZ": 24 * 3 + 9,
    "YI": 24 * 6 + 0,
    "YX": 24 * 6 + 3,
    "YY": 24 * 6 + 6,
    "YZ": 24 * 6 + 9,
    "ZI": 24 * 9 + 0,
    "ZX": 24 * 9 + 3,
    "ZY": 24 * 9 + 6,
    "ZZ": 24 * 9 + 9,
    "X90": 16,
    "Y90": 21,
    "X180": 3,
    "Y180": 6,
    "Z180": 9,
    "CZ": 104368,
}

CZ_DECOMP = [(("q0", "q1"), ["CZ"])]
iSWAP_DECOMP = [(("q0", "q1"), ["iSWAP"])]

class Clifford:
    """Abstract Clifford gate."""

    # class variables
    _hash_table = None

    def __init__(self, idx: int) -> None:
        """Create a Clifford group member."""
        self.idx = idx

    def __mul__(self, other: Clifford) -> Clifford:
        """
        Product of two clifford gates.

        returns a new Clifford object that performs the net operation
        that is the product of both operations.
        """
        net_op = np.dot(self.pauli_transfer_matrix, other.pauli_transfer_matrix)
        idx = self._get_clifford_id(net_op)
        return self.__class__(idx)

    def __repr__(self) -> str:
        """Represent the Clifford gate as string."""
        return f"{self.__class__.__name__}(idx={self.idx})"

    def __str__(self) -> str:
        """Represent the Clifford gate as string, including decomposition."""
        return f"{self.__class__.__name__} idx {self.idx}\n Gates: {self.gate_decomposition().__str__()}\n"

    def get_inverse(self) -> Clifford:
        """Return the inverse of the Clifford gate."""
        inverse_ptm = np.linalg.inv(self.pauli_transfer_matrix).astype(int)
        idx = self._get_clifford_id(inverse_ptm)
        return self.__class__(idx)

    ##########################################################################
    # Abstract class methods
    ##########################################################################
    def gate_decomposition(self) -> list:
        """Return the gate decomposition of the Clifford gate."""
        ...

    @classmethod
    def _get_clifford_id(cls, pauli_transfer_matrix: np.ndarray) -> int: ...

    @property
    def pauli_transfer_matrix(self) -> np.ndarray:
        """Return the Pauli transfer matrix."""
        ...


class SingleQubitClifford(Clifford):
    """Single qubit clifford group."""

    # class constants
    group_size = 24

    def __init__(self, idx: int) -> None:
        """Single qubit clifford group."""
        assert idx < self.group_size
        self.idx = idx

    def gate_decomposition(self, qubit: str = "q0") -> list[tuple[tuple, list[str]]]:
        """Return the gate decomposition of the single qubit Clifford group according to the decomposition by Epstein et al."""
        return [((qubit,), single_qubit_gate_decomposition[self.idx])]

    ##########################################################################
    # Class methods
    ##########################################################################

    @classmethod
    def _get_clifford_id(cls, pauli_transfer_matrix: np.ndarray) -> int:
        """Return the unique Id of a Clifford."""
        unique_hash = crc32(pauli_transfer_matrix.astype(int))

        if cls._hash_table is None:
            cls._hash_table = get_single_qubit_clifford_hash_table()

        idx = cls._hash_table.index(unique_hash)
        return idx

    @classmethod
    def decomposition_from_ptm(
        cls, pauli_transfer_matrix: np.ndarray, qubit: str = "q0"
    ) -> list[tuple[tuple, list[str]]]:
        """Return the unique Id of a Clifford."""
        idx = cls._get_clifford_id(pauli_transfer_matrix)
        return cls(idx).gate_decomposition(qubit)

    @property
    def pauli_transfer_matrix(self) -> np.ndarray:
        """Return the Pauli transfer matrix."""
        return C1[self.idx]


# class TwoQubitClifford(Clifford):
#     """Two qubit clifford group, using a CZ decomposition."""

#     # class constants
#     group_size_single_qubit = SingleQubitClifford.group_size**2
#     group_size_s1 = 3**2  # the S1 subgroup of SingleQubitClifford has size 3
#     group_size_cnot = group_size_single_qubit * group_size_s1
#     group_size_iswap = group_size_cnot
#     group_size_swap = group_size_single_qubit
#     group_size = group_size_single_qubit + group_size_cnot + group_size_iswap + group_size_swap

#     # The two qubit Clifford group is composed of multiple sub groups.
#     # These are the sizes of the subgroups, and often used as magic numbers in this class.
#     assert group_size_single_qubit == 576
#     assert group_size_cnot == 5184
#     assert group_size_iswap == 5184
#     assert group_size_swap == 576
#     assert group_size == 11520

#     # These two methods are needed for caching across instances
#     @property
#     def pauli_transfer_matrix(self) -> np.ndarray:
#         """Return the Pauli transfer matrix."""
#         return self._pauli_transfer_matrix(self.idx)

#     def gate_decomposition(self) -> list[tuple[tuple, list[str]]]:
#         """Return the gate decomposition of the two qubit Clifford group."""
#         return self._gate_decomposition(self.idx)

#     ##########################################################################
#     # Class methods
#     ##########################################################################

#     @classmethod
#     @cache
#     def _pauli_transfer_matrix(cls, idx: int) -> np.ndarray:
#         """Return the Pauli transfer matrix."""
#         if idx < 576:  # noqa:PLR2004
#             pauli_transfer_matrix = cls.single_qubit_like_PTM(idx)
#         elif idx < 576 + 5184:
#             pauli_transfer_matrix = cls.CNOT_like_PTM(idx - 576)
#         elif idx < 576 + 2 * 5184:
#             pauli_transfer_matrix = cls.iSWAP_like_PTM(idx - (576 + 5184))
#         else:  # NB: GRP_SIZE checked upon construction
#             pauli_transfer_matrix = cls.SWAP_like_PTM(idx - (576 + 2 * 5184))
#         return pauli_transfer_matrix

#     @classmethod
#     @cache
#     def _gate_decomposition(cls, idx: int) -> list[tuple[tuple, list[str]]]:
#         """
#         Return the gate decomposition of the two qubit Clifford group.

#         Single qubit Cliffords are decomposed according to Epstein et al.
#         """
#         # compute
#         if idx < 576:  # noqa: PLR2004
#             _gate_decomposition = cls.single_qubit_like_gates(idx)
#         elif idx < 576 + 5184:
#             _gate_decomposition = cls.CNOT_like_gates(idx - 576)
#         elif idx < 576 + 2 * 5184:
#             _gate_decomposition = cls.iSWAP_like_gates(idx - (576 + 5184))
#         else:  # NB: GRP_SIZE checked upon construction
#             _gate_decomposition = cls.SWAP_like_gates(idx - (576 + 2 * 5184))
#         return _gate_decomposition

#     @classmethod
#     def _get_clifford_id(cls, pauli_transfer_matrix: np.ndarray) -> int:
#         """Return the unique Id of a Clifford."""
#         unique_hash = crc32(pauli_transfer_matrix.astype(int))

#         if cls._hash_table is None:
#             cls._hash_table = get_two_qubit_clifford_hash_table()

#         idx = cls._hash_table.index(unique_hash)
#         return idx

#     @classmethod
#     def single_qubit_like_PTM(cls, idx: int) -> np.ndarray:
#         """
#         Return the pauli transfer matrix for gates of the single qubit like class.

#             (q0)  -- C1 --
#             (q1)  -- C1 --.
#         """
#         assert idx < cls.group_size_single_qubit
#         idx_q0 = idx % 24
#         idx_q1 = (idx // 24) % 24
#         pauli_transfer_matrix = np.kron(C1[idx_q1], C1[idx_q0])
#         return pauli_transfer_matrix

#     @classmethod
#     def single_qubit_like_gates(cls, idx: int) -> list[tuple[tuple, list[str]]]:
#         """
#         Return the gates for Cliffords of the single qubit like class.

#             (q0)  -- C1 --
#             (q1)  -- C1 --.
#         """
#         gates = []
#         gates += SingleQubitClifford(idx % 24).gate_decomposition("q0")
#         gates += SingleQubitClifford((idx // 24) % 24).gate_decomposition("q1")
#         return gates

#     @classmethod
#     def CNOT_like_PTM(cls, idx: int) -> np.ndarray:
#         """
#         Return the pauli transfer matrix for gates of the cnot like class.

#             (q0)  --C1--•--S1--      --C1--•--S1------
#                         |        ->        |
#             (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--
#         """
#         assert idx < cls.group_size_cnot
#         idx_0 = idx % 24
#         idx_1 = (idx // 24) % 24
#         idx_2 = (idx // 576) % 3
#         idx_3 = idx // 1728

#         C1_q0 = np.kron(np.eye(4), C1[idx_0])
#         C1_q1 = np.kron(C1[idx_1], np.eye(4))
#         # CZ
#         S1_q0 = np.kron(np.eye(4), S1[idx_2])
#         S1y_q1 = np.kron(np.dot(C1[idx_3], Y90), np.eye(4))
#         return np.linalg.multi_dot(list(reversed([C1_q0, C1_q1, CZ, S1_q0, S1y_q1])))

#     @classmethod
#     def CNOT_like_gates(cls, idx: int) -> list[tuple[tuple, list[str]]]:
#         """
#         Return the gates for Cliffords of the cnot like class.

#             (q0)  --C1--•--S1--      --C1--•--S1------
#                         |        ->        |
#             (q1)  --C1--⊕--S1--      --C1--•--S1^Y90--.
#         """
#         assert idx < cls.group_size_cnot
#         idx_2 = (idx // 576) % 3
#         idx_3 = idx // 1728
#         gates = cls.single_qubit_like_gates(idx)
#         gates += CZ_DECOMP
#         gates += SingleQubitClifford.decomposition_from_ptm(S1[idx_2], "q0")
#         gates += SingleQubitClifford.decomposition_from_ptm(np.dot(C1[idx_3], Y90), "q1")
#         return gates

#     @classmethod
#     def iSWAP_like_PTM(cls, idx: int) -> np.ndarray:
#         """
#         Return the pauli transfer matrix for gates of the iSWAP like class.

#             (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
#                         |       ->        |        |
#             (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--
#         """
#         assert idx < cls.group_size_iswap
#         idx_0 = idx % 24
#         idx_1 = (idx // 24) % 24
#         idx_2 = (idx // 576) % 3
#         idx_3 = idx // 1728

#         C1_q0 = np.kron(np.eye(4), C1[idx_0])
#         C1_q1 = np.kron(C1[idx_1], np.eye(4))
#         # CZ
#         sq_swap_gates = np.kron(mY90, Y90)
#         # CZ
#         S1_q0 = np.kron(np.eye(4), np.dot(S1[idx_2], Y90))
#         S1y_q1 = np.kron(np.dot(C1[idx_3], X90), np.eye(4))

#         return np.linalg.multi_dot(
#             list(reversed([C1_q0, C1_q1, CZ, sq_swap_gates, CZ, S1_q0, S1y_q1]))
#         )

#     @classmethod
#     def iSWAP_like_gates(cls, idx: int) -> list[tuple[tuple, list[str]]]:
#         """
#         Return the gates for Cliffords of the iSWAP like class.

#             (q0)  --C1--*--S1--     --C1--•---Y90--•--S1^Y90--
#                         |       ->        |        |
#             (q1)  --C1--*--S1--     --C1--•--mY90--•--S1^X90--.
#         """
#         assert idx < cls.group_size_iswap
#         idx_2 = (idx // 576) % 3
#         idx_3 = idx // 1728

#         gates = cls.single_qubit_like_gates(idx)
#         gates += CZ_DECOMP
#         gates += SingleQubitClifford.decomposition_from_ptm(Y90, "q0")
#         gates += SingleQubitClifford.decomposition_from_ptm(mY90, "q1")
#         gates += CZ_DECOMP
#         gates += SingleQubitClifford.decomposition_from_ptm(np.dot(S1[idx_2], Y90), "q0")
#         gates += SingleQubitClifford.decomposition_from_ptm(np.dot(C1[idx_3], X90), "q1")

#         # S1_q0 = np.kron(np.eye(4), np.dot(S1[idx_2], Y90))
#         # S1y_q1 = np.kron(np.dot(C1[idx_3], X90), np.eye(4))

#         return gates

#     @classmethod
#     def SWAP_like_PTM(cls, idx: int) -> np.ndarray:
#         """
#         Return the pauli transfer matrix for gates of the SWAP like class.

#         (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
#                     |   ->        |       |       |
#         (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
#         """
#         assert idx < cls.group_size_swap
#         idx_q0 = idx % 24
#         idx_q1 = (idx // 24) % 24
#         sq_like_cliff = np.kron(C1[idx_q1], C1[idx_q0])
#         sq_swap_gates_0 = np.kron(Y90, mY90)
#         sq_swap_gates_1 = np.kron(mY90, Y90)
#         sq_swap_gates_2 = np.kron(Y90, np.eye(4))

#         return np.linalg.multi_dot(
#             list(
#                 reversed(
#                     [sq_like_cliff, CZ, sq_swap_gates_0, CZ, sq_swap_gates_1, CZ, sq_swap_gates_2]
#                 )
#             )
#         )

#     @classmethod
#     def SWAP_like_gates(cls, idx: int) -> list[tuple[tuple, list[str]]]:
#         """
#         Return the gates for Cliffords of the SWAP like class.

#         (q0)  --C1--x--     --C1--•-mY90--•--Y90--•-------
#                     |   ->        |       |       |
#         (q1)  --C1--x--     --C1--•--Y90--•-mY90--•--Y90--
#         """
#         assert idx < cls.group_size_swap

#         gates = cls.single_qubit_like_gates(idx)
#         gates += CZ_DECOMP
#         gates += SingleQubitClifford.decomposition_from_ptm(mY90, "q0")
#         gates += SingleQubitClifford.decomposition_from_ptm(Y90, "q1")
#         gates += CZ_DECOMP
#         gates += SingleQubitClifford.decomposition_from_ptm(Y90, "q0")
#         gates += SingleQubitClifford.decomposition_from_ptm(mY90, "q1")
#         gates += CZ_DECOMP
#         gates += SingleQubitClifford.decomposition_from_ptm(I, "q0")
#         gates += SingleQubitClifford.decomposition_from_ptm(Y90, "q1")
#         return gates


class TwoQubitClifford(Clifford):
    """Two qubit clifford group, using a iSWAP decomposition."""

    # class constants
    group_size_single_qubit = SingleQubitClifford.group_size**2
    group_size_s1 = 3**2  # the S1 subgroup of SingleQubitClifford has size 3
    group_size_cnot = group_size_single_qubit * group_size_s1
    group_size_iswap = group_size_cnot
    group_size_swap = group_size_single_qubit
    group_size = group_size_single_qubit + group_size_cnot + group_size_iswap + group_size_swap

    # The two qubit Clifford group is composed of multiple sub groups.
    # These are the sizes of the subgroups, and often used as magic numbers in this class.
    assert group_size_single_qubit == 576
    assert group_size_cnot == 5184
    assert group_size_iswap == 5184
    assert group_size_swap == 576
    assert group_size == 11520

    # These two methods are needed for caching across instances
    @property
    def pauli_transfer_matrix(self) -> np.ndarray:
        """Return the Pauli transfer matrix."""
        return self._pauli_transfer_matrix(self.idx)

    def gate_decomposition(self) -> list[tuple[tuple, list[str]]]:
        """Return the gate decomposition of the two qubit Clifford group."""
        return self._gate_decomposition(self.idx)

    ##########################################################################
    # Class methods
    ##########################################################################

    @classmethod
    @cache
    def _pauli_transfer_matrix(cls, idx: int) -> np.ndarray:
        """Return the Pauli transfer matrix."""
        if idx < 576:  # noqa:PLR2004
            pauli_transfer_matrix = cls.single_qubit_like_PTM(idx)
        elif idx < 576 + 5184:
            pauli_transfer_matrix = cls.CNOT_like_PTM(idx - 576)
        elif idx < 576 + 2 * 5184:
            pauli_transfer_matrix = cls.iSWAP_like_PTM(idx - (576 + 5184))
        else:  # NB: GRP_SIZE checked upon construction
            pauli_transfer_matrix = cls.SWAP_like_PTM(idx - (576 + 2 * 5184))
        return pauli_transfer_matrix

    @classmethod
    @cache
    def _gate_decomposition(cls, idx: int) -> list[tuple[tuple, list[str]]]:
        """
        Return the gate decomposition of the two qubit Clifford group.

        Single qubit Cliffords are decomposed according to Epstein et al.
        """
        # compute
        if idx < 576:  # noqa: PLR2004
            _gate_decomposition = cls.single_qubit_like_gates(idx)
        elif idx < 576 + 5184:
            _gate_decomposition = cls.CNOT_like_gates(idx - 576)
        elif idx < 576 + 2 * 5184:
            _gate_decomposition = cls.iSWAP_like_gates(idx - (576 + 5184))
        else:  # NB: GRP_SIZE checked upon construction
            _gate_decomposition = cls.SWAP_like_gates(idx - (576 + 2 * 5184))
        return _gate_decomposition

    @classmethod
    def _get_clifford_id(cls, pauli_transfer_matrix: np.ndarray) -> int:
        """Return the unique Id of a Clifford."""
        unique_hash = crc32(pauli_transfer_matrix.astype(int))

        if cls._hash_table is None:
            cls._hash_table = get_two_qubit_clifford_hash_table()

        idx = cls._hash_table.index(unique_hash)
        return idx

    @classmethod
    def single_qubit_like_PTM(cls, idx: int) -> np.ndarray:
        """
        Return the pauli transfer matrix for gates of the single qubit like class.

            (q0)  -- C1 --
            (q1)  -- C1 --.
        """
        assert idx < cls.group_size_single_qubit
        idx_q0 = idx % 24
        idx_q1 = (idx // 24) % 24
        pauli_transfer_matrix = np.kron(C1[idx_q1], C1[idx_q0])
        return pauli_transfer_matrix

    @classmethod
    def single_qubit_like_gates(cls, idx: int) -> list[tuple[tuple, list[str]]]:
        """
        Return the gates for Cliffords of the single qubit like class.

            (q0)  -- C1 --
            (q1)  -- C1 --.
        """
        gates = []
        gates += SingleQubitClifford(idx % 24).gate_decomposition("q0")
        gates += SingleQubitClifford((idx // 24) % 24).gate_decomposition("q1")
        return gates

    @classmethod
    def CNOT_like_PTM(cls, idx: int) -> np.ndarray:
        """
        Return the pauli transfer matrix for gates of the cnot like class.

            (q0)  --C1--•--S1--      --C1--*--X90--*--S1------
                        |        ->        |       |
            (q1)  --C1--⊕--S1--      --C1--*-------*--S1^Z90--
        """
        assert idx < cls.group_size_cnot
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728

        C1_q0 = np.kron(np.eye(4), C1[idx_0])
        C1_q1 = np.kron(C1[idx_1], np.eye(4))
        # iSWAP
        sq_swap_gates = np.kron(np.eye(4), X90)
        # iSWAP
        S1_q0 = np.kron(np.eye(4), S1[idx_2])
        S1z_q1 = np.kron(np.dot(S1[idx_3], Z90), np.eye(4))
        return np.linalg.multi_dot(list(reversed([C1_q0, C1_q1, iSWAP, sq_swap_gates, iSWAP, S1_q0, S1z_q1])))

    @classmethod
    def CNOT_like_gates(cls, idx: int) -> list[tuple[tuple, list[str]]]:
        """
        Return the gates for Cliffords of the cnot like class.

            (q0)  --C1--•--S1--      --C1--*--X90--*--S1------
                        |        ->        |       |
            (q1)  --C1--⊕--S1--      --C1--*-------*--S1^Z90--
        """
        assert idx < cls.group_size_cnot
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728
        gates = cls.single_qubit_like_gates(idx)
        gates += iSWAP_DECOMP
        gates += SingleQubitClifford.decomposition_from_ptm(X90, "q0")
        gates += SingleQubitClifford.decomposition_from_ptm(I, "q1")
        gates += iSWAP_DECOMP
        gates += SingleQubitClifford.decomposition_from_ptm(S1[idx_2], "q0")
        gates += SingleQubitClifford.decomposition_from_ptm(np.dot(S1[idx_3], Z90), "q1")
        return gates

    @classmethod
    def iSWAP_like_PTM(cls, idx: int) -> np.ndarray:
        """
        Return the pauli transfer matrix for gates of the iSWAP like class.

            (q0)  --C1--*--S1--
                        |      
            (q1)  --C1--*--S1--
        """
        assert idx < cls.group_size_iswap
        idx_0 = idx % 24
        idx_1 = (idx // 24) % 24
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728

        C1_q0 = np.kron(np.eye(4), C1[idx_0])
        C1_q1 = np.kron(C1[idx_1], np.eye(4))
        # iSWAP
        S1_q0 = np.kron(np.eye(4), S1[idx_2])
        S1_q1 = np.kron(S1[idx_3], np.eye(4))

        return np.linalg.multi_dot(
            list(reversed([C1_q0, C1_q1, iSWAP, S1_q0, S1_q1]))
        )

    @classmethod
    def iSWAP_like_gates(cls, idx: int) -> list[tuple[tuple, list[str]]]:
        """
        Return the gates for Cliffords of the iSWAP like class.

            (q0)  --C1--*--S1--
                        |      
            (q1)  --C1--*--S1--
        """
        assert idx < cls.group_size_iswap
        idx_2 = (idx // 576) % 3
        idx_3 = idx // 1728

        gates = cls.single_qubit_like_gates(idx)
        gates += iSWAP_DECOMP
        gates += SingleQubitClifford.decomposition_from_ptm(S1[idx_2], "q0")
        gates += SingleQubitClifford.decomposition_from_ptm(S1[idx_3], "q1")

        return gates

    @classmethod
    def SWAP_like_PTM(cls, idx: int) -> np.ndarray:
        """
        Return the pauli transfer matrix for gates of the SWAP like class.

        (q0)  --C1--x--     --C1--*-------*-mX90--*-------
                    |   ->        |       |       |
        (q1)  --C1--x--     --C1--*-mX90--*-------*-mX90--
        """
        assert idx < cls.group_size_swap
        idx_q0 = idx % 24
        idx_q1 = (idx // 24) % 24
        sq_like_cliff = np.kron(C1[idx_q1], C1[idx_q0])
        # iSWAP
        sq_swap_gates_0 = np.kron(mX90, np.eye(4))
        # iSWAP
        sq_swap_gates_1 = np.kron(np.eye(4), mX90)
        # iSWAP
        sq_swap_gates_2 = np.kron(mX90, np.eye(4))

        return np.linalg.multi_dot(
            list(
                reversed(
                    [sq_like_cliff, iSWAP, sq_swap_gates_0, iSWAP, sq_swap_gates_1, iSWAP, sq_swap_gates_2]
                )
            )
        )

    @classmethod
    def SWAP_like_gates(cls, idx: int) -> list[tuple[tuple, list[str]]]:
        """
        Return the gates for Cliffords of the SWAP like class.

        (q0)  --C1--x--     --C1--*-------*-mX90--*-------
                    |   ->        |       |       |
        (q1)  --C1--x--     --C1--*-mX90--*-------*-mX90--
        """
        assert idx < cls.group_size_swap

        gates = cls.single_qubit_like_gates(idx)
        gates += iSWAP_DECOMP
        gates += SingleQubitClifford.decomposition_from_ptm(I, "q0")
        gates += SingleQubitClifford.decomposition_from_ptm(mX90, "q1")
        gates += iSWAP_DECOMP
        gates += SingleQubitClifford.decomposition_from_ptm(mX90, "q0")
        gates += SingleQubitClifford.decomposition_from_ptm(I, "q1")
        gates += iSWAP_DECOMP
        gates += SingleQubitClifford.decomposition_from_ptm(I, "q0")
        gates += SingleQubitClifford.decomposition_from_ptm(mX90, "q1")
        return gates


##############################################################################
# It is important that this check is after the Clifford objects as otherwise
# it is impossible to generate the hash tables
##############################################################################


with contextlib.suppress(FileExistsError):
    mkdir(hash_dir)


def construct_clifford_lookuptable(generator: type[Clifford], indices: Iterable[int]):
    lookuptable = []
    for idx in indices:
        clifford = generator(idx=idx)
        # important to use crc32 hashing as this is a non-random hash
        hash_val = crc32(clifford.pauli_transfer_matrix.round().astype(int))
        lookuptable.append(hash_val)
    return lookuptable


def generate_hash_tables():
    # FIXME: text files is probably not the best way to store hash tables
    print("Generating Clifford hash tables.")
    single_qubit_hash_lut = construct_clifford_lookuptable(
        SingleQubitClifford, np.arange(SingleQubitClifford.group_size)
    )
    with open(join(hash_dir, "single_qubit_hash_lut.txt"), "w") as f:
        for h in single_qubit_hash_lut:
            f.write(str(h) + "\n")

    two_qubit_hash_lut = construct_clifford_lookuptable(
        TwoQubitClifford, np.arange(TwoQubitClifford.group_size)
    )
    with open(join(hash_dir, "two_qubit_hash_lut.txt"), "w") as f:
        for h in two_qubit_hash_lut:
            f.write(str(h) + "\n")
    print("Successfully generated Clifford hash tables.")


def regenerate_hash_tables():
    if not exists(join(hash_dir, "single_qubit_hash_lut.txt")) or not exists(
        join(hash_dir, "two_qubit_hash_lut.txt")
    ):
        generate_hash_tables()


regenerate_hash_tables()


def get_single_qubit_clifford_hash_table():
    """
    Get the single qubit clifford hash table.

     Requires this to be generated
    first. To generate, execute this file.
    """
    with open(join(hash_dir, "single_qubit_hash_lut.txt")) as f:
        hash_table = [int(line.rstrip("\n")) for line in f]
    return hash_table


def get_two_qubit_clifford_hash_table():
    """
    Get the two qubit clifford hash table.

    Requires this to be generated
    first. To generate, execute this file.
    """
    with open(join(hash_dir, "two_qubit_hash_lut.txt")) as f:
        hash_table = [int(line.rstrip("\n")) for line in f]
    return hash_table
