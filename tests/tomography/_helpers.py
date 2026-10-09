import qutip as qt

CHI_DIMS = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]


def choi_to_chi(choi):
    """chi Qobj (Tr = 16) of a 16x16 Choi matrix in qutip's convention."""
    return qt.to_chi(qt.Qobj(choi, dims=CHI_DIMS, superrep="choi"))
