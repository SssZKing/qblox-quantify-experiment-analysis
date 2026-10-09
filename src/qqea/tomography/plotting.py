"""3D bar plots of process matrices."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize

from ._paulis import PAULI_LABELS

titlefont = {"color": "black", "weight": "normal", "size": 10}
axisfont = {"color": "black", "weight": "normal", "size": 8}
ticksfont = {"color": "black", "weight": "normal", "size": 4}


def qpt_plot_cmap(chi, lbls_list, title='', cmap=plt.cm.bwr, figsize=(8, 6), threshold=None):
    """Like qt.qpt_plot_combined but with a user-chosen (cyclic) colormap.
    lbls_list is one list of labels per qubit, e.g. op_label. Returns (fig, ax)."""
    from qutip import matrix_histogram

    xlabels = ["".join(p) for p in itertools.product(*lbls_list)]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    matrix_histogram(chi, xlabels, xlabels, bar_style='abs', color_style='phase',
                     cmap=cmap, options={'threshold': threshold}, ax=ax)
    ax.set_title(title)
    return fig, ax


def plot_process_tomography2(chi_vector, save_file: str = None):
    """3D bars of |chi| coloured by arg(chi) for a 16x16 chi matrix (array or flattened),
    in PAULI_LABELS order. Saves to ``save_file`` if given. Returns (fig, ax); the figure is
    left open for the caller to show or close."""
    cmap = colormaps.get_cmap("viridis")

    fig = plt.figure(figsize=(6, 4), dpi=200, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    r = np.arange(16)
    _x, _y = np.meshgrid(r, r)
    x, y = _x.ravel(), _y.ravel()

    chi_vector = np.ravel(chi_vector)
    top = np.abs(chi_vector)
    bottom = np.zeros_like(top)
    width = depth = 0.7

    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    colors = cmap(norm(np.angle(chi_vector)))

    xy_ticks_labels = [f"${label}$" for label in PAULI_LABELS]

    ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors)
    ax.set_xticks(r + 0.5, labels=xy_ticks_labels, fontdict=ticksfont)
    ax.set_yticks(r + 0.5, labels=xy_ticks_labels, fontdict=ticksfont)
    ax.set_zticks([0, (max(top) / 2).round(2), max(top).round(2)])
    ax.set_xlabel("Prepared", fontdict=axisfont)
    ax.set_ylabel("Measured", fontdict=axisfont)
    ax.set_title(r"$\chi$ matrix", fontdict=titlefont)
    ax.view_init(20, -60, 0)

    sc = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_ticks(
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        labels=[r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"],
    )

    if save_file:
        fig.savefig(save_file, bbox_inches="tight")
    return fig, ax
