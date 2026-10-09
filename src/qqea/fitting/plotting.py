"""Plotting helpers: IQ traces, cavity maps / punchouts and a Bloch sphere."""

import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

from qqea.fitting.units import Vp2dBm


def create_ntwk(name, freq, I_data, Q_data, real_imag=True):
    s = np.zeros((len(freq), 2, 2), dtype=complex)
    if real_imag:
        I = I_data
        Q = Q_data
    else:
        I = I_data*np.cos(np.deg2rad(Q_data))
        Q = I_data*np.sin(np.deg2rad(Q_data))
        
    S21 = I + 1j*Q
    s[:,1,0] = S21

    f = rf.Frequency.from_f(freq, unit='Hz')

    ntwk = rf.Network(name = name, frequency=f, s=s)
    
    return ntwk

def plot_IQ(dataset, polar=False):
    frequency = dataset.x0.values
    
    I = dataset.y0 * np.cos(np.deg2rad(dataset.y1))
    Q = dataset.y0 * np.sin(np.deg2rad(dataset.y1))

    if polar == False:
        fig, ax = plt.subplots(2, figsize=(6,6))
        ax[0].plot(frequency/1e9, I*1e3)
        ax[1].plot(frequency/1e9, Q*1e3)
        
        ax[0].set_ylabel('In-phase [mV]')
        ax[1].set_ylabel('Quadrature [mV]')
        ax[1].set_xlabel('Frequency [GHz]')

    else:
        plt.plot(I*1e3, Q*1e3, ls='', marker='.', alpha=0.75)
        plt.plot(0,0, marker='o', color='black')
        plt.xlabel('In-phase [mV]')
        plt.ylabel('Quadrature [mV]')

def cavity_map(dataset, normalized=True, unit="V"):
    """
    Plot the readout cavity map or punchout measurement

    Parameters
    ----------
    dataset: 
        The dataset containing the cavity map with IQ components
    normalized:
        Normalize based on the drive amplitude
    unit:
        Unit between 'V' or 'dB'
    """
    I_data = getattr(dataset, "y0").values.reshape(dataset.ylen,dataset.xlen)
    Q_data = getattr(dataset, "y1").values.reshape(dataset.ylen,dataset.xlen)
    amplitude = np.sqrt(I_data**2 + Q_data**2)
    
    amp_setpoints = dataset.x1.values.reshape(dataset.ylen,dataset.xlen)[:,0]
    frequency_setpoints = dataset.x0.values.reshape(dataset.ylen,dataset.xlen)[0]
    
    if normalized:
        for r in range(amplitude.shape[0]):
            amplitude[r] /= amplitude[r].sum()

    plt.figure(figsize=(6,5))
    plt.xlabel('Frequency [GHz]')
    
    if unit=='dB' or unit=='db':
        S21 = np.copy(amplitude)
        for r in range(S21.shape[0]):
            if normalized:
                S21[r] = amplitude[r]/amplitude[r].max()
            else:
                S21[r] = amplitude[r]/amp_setpoints[r]
        S21 = rf.mathFunctions.mag_2_db(S21)
        
        plt.pcolor(frequency_setpoints/1e9, Vp2dBm(amp_setpoints), S21)
        plt.ylabel('QBLOX Power [dBm]')
        plt.colorbar(label = r'|S21|$^2$ [dB]')

    else: 
        plt.pcolor(frequency_setpoints/1e9, amp_setpoints, amplitude)
        plt.yscale('log')
        plt.ylabel('QBLOX Voltage [V]')
        plt.colorbar(label = 'Amplitude [V]')
    
    if normalized:
        plt.title(f'Normalized Cavity map')
    else:
        plt.title(f'Cavity map')


def bra_tex(s):
    return rf"$\left| {s} \right\rangle$"


class BlochSpherePlot:
    North = np.array((0, 0, 1))
    South = np.array((0, 0, -1))
    East = np.array((1, 0, 0))
    West = np.array((0, 1, 0))

    def __init__(self, elev=20, azim=15, sphere_style=None, circles_style=None, axes_style=None, *args, **kwargs):
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"}, *args, **kwargs)
        self.ax.view_init(elev=elev, azim=azim)

        self.sphere_style = {
            "color": "#ccf3ff",
            "alpha": 0.1,
        }
        if sphere_style:
            self.sphere_style.update(sphere_style)

        self.circles_style = {
            "color": "#333",
            "alpha": 0.2,
            "lw": 1.0,
        }
        if circles_style:
            self.circles_style.update(circles_style)

        self.axes_style = {
            "color": "#333",
            "alpha": 0.2,
            "lw": 1.0,
        }
        if axes_style:
            self.axes_style.update(axes_style)

        self.add_sphere()
        self.add_circles()
        self.add_axes()

        self._prepare_axes()

    def plot_vector(
        self,
        v,
        label=None,
        **kwargs,
    ):
        self.ax.quiver(
            0,
            0,
            0,
            v[0],
            v[1],
            v[2],
            normalize=True,
            arrow_length_ratio=0.08,
            **kwargs,
        )
        if label:
            vn = 1.1 * np.array(v) / np.linalg.norm(v)
            self.ax.text(*vn, label, fontsize="large")

    def label(self, position, label, **kwargs):
        self.ax.text(*position, label, fontsize="large", **kwargs)

    def label_bra(self, position, label, **kwargs):
        self.label(position, bra_tex(label), **kwargs)

    def add_circles(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        c = np.cos(theta)
        s = np.sin(theta)
        z = np.zeros_like(theta)

        self.ax.plot(c, s, z, **self.circles_style)
        self.ax.plot(z, c, s, **self.circles_style)
        self.ax.plot(s, z, c, **self.circles_style)

    def add_sphere(self):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_surface(x, y, z, **self.sphere_style)

    def add_axes(self):
        u = np.linspace(-1, 1, 2)
        z = np.zeros_like(u)
        self.ax.plot(u, z, z, **self.axes_style)
        self.ax.plot(z, z, u, **self.axes_style)
        self.ax.plot(z, u, z, **self.axes_style)

    def _prepare_axes(self):
        self.ax.set(
            aspect="equal",
        )

        self.ax.tick_params(
            labelbottom=False,
            labelleft=False,
        )
        self.ax.set_axis_off()
        self.ax.grid(False)
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
