import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.signal import savgol_filter
import skrf as rf
from qblox_instruments import Cluster, ClusterType
import contextlib

from quantify_core.data.handling import (
    default_datadir,
    set_datadir,
    get_datadir,
    load_dataset,
    load_snapshot,
)

from quantify_core.analysis.cosine_analysis import CosineAnalysis
from quantify_core.analysis.single_qubit_timedomain import (
    RabiAnalysis, 
    RamseyAnalysis, 
    T1Analysis,
    EchoAnalysis,
    AllXYAnalysis,
)
from quantify_core.analysis.spectroscopy_analysis import (
    ResonatorSpectroscopyAnalysis,
    QubitSpectroscopyAnalysis,
)
from quantify_core.analysis.readout_calibration_analysis import (
    ReadoutCalibrationAnalysis,
)
from quantify_core.analysis.fitting_models import ResonatorModel, LorentzianModel, ExpDecayModel, DecayOscillationModel
resonator_model = ResonatorModel()
lorentzian_model = LorentzianModel()
exp_decay_model = ExpDecayModel()
decay_osci_model = DecayOscillationModel()

from quantify_core.analysis.base_analysis import Basic2DAnalysis
from datetime import datetime

def experiment_duration(tuid_arr):
    # Define the format
    time_format = '%Y%m%d-%H%M%S'
    
    # Parse the strings into datetime objects
    dt1 = datetime.strptime(tuid_arr[0], time_format)
    dt2 = datetime.strptime(tuid_arr[1], time_format)
    dt3 = datetime.strptime(tuid_arr[-1], time_format)

    dt_mins = round((dt2 - dt1).total_seconds()/60, 2)
    T_hours = round((dt3 - dt1).total_seconds()/3600, 2)

    print("Dataset mins:", dt_mins)
    print("Total hours:", T_hours)

    return dt_mins, T_hours

def Vp2dBm(Vp, Z=50):
    return 10*np.log10(Vp**2 / (2*Z*1e-3))
def dBm2Vp(dBm, Z=50):
    return np.sqrt(10**(dBm/10)*2*Z*1e-3)
def W2dBm(W):
    return 10 * np.log10(W) + 30
def dBm2W(dBm):
    return 10 ** ((dBm - 30) / 10)

def Z2P(Z):
    return (1 - Z) / 2
def P2Z(P):
    return 1 - 2*P

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

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def _false_detections(threshold, Ig, Ie):
    if np.mean(Ig) < np.mean(Ie):
        false_detections_var = np.sum(Ig > threshold) + np.sum(Ie < threshold)
    else:
        false_detections_var = np.sum(Ig < threshold) + np.sum(Ie > threshold)
    return false_detections_var


def two_state_discriminator(Ig, Qg, Ie, Qe, b_print=True, b_plot=True):
    """
    Given two blobs in the IQ plane representing two states, finds the optimal threshold to discriminate between them
    and calculates the fidelity. Also returns the angle in which the data needs to be rotated in order to have all the
    information in the `I` (`X`) axis.

    .. note::
        This function assumes that there are only two blobs in the IQ plane representing two states (ground and excited)
        Unexpected output will be returned in other cases.


    :param float Ig: A vector containing the `I` quadrature of data points in the ground state
    :param float Qg: A vector containing the `Q` quadrature of data points in the ground state
    :param float Ie: A vector containing the `I` quadrature of data points in the excited state
    :param float Qe: A vector containing the `Q` quadrature of data points in the excited state
    :param bool b_print: When true (default), prints the results to the console.
    :param bool b_plot: When true (default), plots the results in a new figure.
    :returns: A tuple of (angle, threshold, fidelity, gg, ge, eg, ee).
        angle - The angle (in radians) in which the IQ plane has to be rotated in order to have all the information in
            the `I` axis.
        threshold - The threshold in the rotated `I` axis. The excited state will be when the `I` is larger (>) than
            the threshold.
        fidelity - The fidelity for discriminating the states.
        gg - The matrix element indicating a state prepared in the ground state and measured in the ground state.
        ge - The matrix element indicating a state prepared in the ground state and measured in the excited state.
        eg - The matrix element indicating a state prepared in the excited state and measured in the ground state.
        ee - The matrix element indicating a state prepared in the excited state and measured in the excited state.
    """
    # Condition to have the Q equal for both states:
    angle = np.arctan2(np.mean(Qe) - np.mean(Qg), np.mean(Ig) - np.mean(Ie))
    C = np.cos(angle)
    S = np.sin(angle)
    # Condition for having e > Ig
    if np.mean((Ig - Ie) * C - (Qg - Qe) * S) > 0:
        angle += np.pi
        C = np.cos(angle)
        S = np.sin(angle)

    Ig_rotated = Ig * C - Qg * S
    Qg_rotated = Ig * S + Qg * C

    Ie_rotated = Ie * C - Qe * S
    Qe_rotated = Ie * S + Qe * C

    fit = minimize(
        _false_detections,
        0.5 * (np.mean(Ig_rotated) + np.mean(Ie_rotated)),
        (Ig_rotated, Ie_rotated),
        method="Nelder-Mead",
    )
    threshold = fit.x[0]

    gg = np.sum(Ig_rotated < threshold) / len(Ig_rotated)
    ge = np.sum(Ig_rotated > threshold) / len(Ig_rotated)
    eg = np.sum(Ie_rotated < threshold) / len(Ie_rotated)
    ee = np.sum(Ie_rotated > threshold) / len(Ie_rotated)

    fidelity = 100 * (gg + ee) / 2

    if b_print:
        print(
            f"""
        Fidelity Matrix:
        -----------------
        | {gg:.3f} | {ge:.3f} |
        ----------------
        | {eg:.3f} | {ee:.3f} |
        -----------------
        IQ plane rotated by: {180 / np.pi * angle:.1f}{chr(176)}
        Threshold: {threshold:.3e}
        Fidelity: {fidelity:.1f}%
        """
        )

    if b_plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(Ig, Qg, ".", alpha=0.1, label="Ground", markersize=2)
        ax1.plot(Ie, Qe, ".", alpha=0.1, label="Excited", markersize=2)
        ax1.axis("equal")
        ax1.legend(["Ground", "Excited"])
        ax1.set_xlabel("I")
        ax1.set_ylabel("Q")
        ax1.set_title("Original Data")

        ax2.plot(Ig_rotated, Qg_rotated, ".", alpha=0.1, label="Ground", markersize=2)
        ax2.plot(Ie_rotated, Qe_rotated, ".", alpha=0.1, label="Excited", markersize=2)
        ax2.axis("equal")
        ax2.set_xlabel("I")
        ax2.set_ylabel("Q")
        ax2.set_title("Rotated Data")

        ax3.hist(Ig_rotated, bins=50, alpha=0.75, label="Ground")
        ax3.hist(Ie_rotated, bins=50, alpha=0.75, label="Excited")
        ax3.axvline(x=threshold, color="k", ls="--", alpha=0.5)
        text_props = dict(
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax3.transAxes,
        )
        ax3.text(0.7, 0.9, f"{threshold:.3e}", text_props)
        ax3.set_xlabel("I")
        ax3.set_title("1D Histogram")

        ax4.imshow(np.array([[gg, ge], [eg, ee]]))
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(labels=["|g>", "|e>"])
        ax4.set_yticklabels(labels=["|g>", "|e>"])
        ax4.set_ylabel("Prepared")
        ax4.set_xlabel("Measured")
        ax4.text(0, 0, f"{100 * gg:.1f}%", ha="center", va="center", color="k")
        ax4.text(1, 0, f"{100 * ge:.1f}%", ha="center", va="center", color="w")
        ax4.text(0, 1, f"{100 * eg:.1f}%", ha="center", va="center", color="w")
        ax4.text(1, 1, f"{100 * ee:.1f}%", ha="center", va="center", color="k")
        ax4.set_title("Fidelities")
        fig.tight_layout()

    return angle, threshold, fidelity, gg, ge, eg, ee

def IQ_blob_SNR(g_blob, e_blob):
    var0_I = g_blob.real.var()
    var0_Q = g_blob.imag.var()
    var1_I = e_blob.real.var()
    var1_Q = e_blob.imag.var()
    
    avg0_I = g_blob.real.mean()
    avg0_Q = g_blob.imag.mean()
    avg1_I = e_blob.real.mean()
    avg1_Q = e_blob.imag.mean()
    
    Z = (avg1_I - avg0_I) + 1j * (avg1_Q - avg0_Q)
    var = (var0_I + var0_Q + var1_I + var1_Q) / 4
    SNR = ((np.abs(Z)) ** 2) / (2 * var)

    return SNR

def RO_freq_fit(dataset, freq_list, plot=False):
    """Find the readout frequency that maximizes ground/excited IQ-blob SNR.

    ``dataset`` is raw complex acquisition data (as returned by
    ``instrument_coordinator.retrieve_acquisition()``) with ground and excited
    shots interleaved column-wise per swept frequency. For each frequency,
    computes the SNR between the ground and excited IQ blobs, then smooths the
    SNR-vs-frequency trace with a Savitzky-Golay filter and returns the
    frequency at the smoothed peak.

    A filter is used instead of a parabola fit because the real SNR trace
    isn't reliably parabolic across the whole sweep (asymmetric tails, a
    plateau near the top, etc.) — smoothing follows the actual shape of the
    data instead of forcing a symmetric model onto it.

    Returns
    -------
    float or None
        The optimal readout frequency in Hz, or ``None`` if the peak doesn't
        clearly rise above the trace's own noise floor (so callers should
        skip updating the qubit).
    """
    g_append = dataset[list(dataset.data_vars)[0]].values[:, 0::2]
    e_append = dataset[list(dataset.data_vars)[0]].values[:, 1::2]

    SNR_list = np.array([
        IQ_blob_SNR(g_append[:, i], e_append[:, i])
        for i in range(g_append.shape[1])
    ])
    fid_list = np.array([
        two_state_discriminator(
            g_append[:, i].real, g_append[:, i].imag,
            e_append[:, i].real, e_append[:, i].imag,
            b_print=False, b_plot=False
        )[2]
        for i in range(g_append.shape[1])
    ])
    freq_list = np.asarray(freq_list)

    # Savitzky-Golay window must be odd and no larger than the number of points
    window_length = min(7, freq_list.size if freq_list.size % 2 else freq_list.size - 1)
    polyorder = min(3, window_length - 1)

    if window_length < 3:
        print('RO frequency fit failed: not enough points to smooth. Not updated.')
        return None

    smoothed = savgol_filter(SNR_list, window_length=window_length, polyorder=polyorder)

    peak_idx = np.argmax(smoothed)
    optimal_freq = freq_list[peak_idx]

    # goodness check: the smoothed peak should clearly rise above the noise
    # the filter smoothed away (the residual between raw data and the trend)
    residual_std = np.std(SNR_list - smoothed)
    peak_prominence = smoothed[peak_idx] - np.median(smoothed)
    fit_ok = residual_std > 0 and peak_prominence > 1 * residual_std

    if plot:
        # plot relative to the sweep center in kHz -- at ~GHz absolute
        # frequencies, absolute kHz ticks are unreadable (matplotlib falls
        # back to a "1e6" scientific offset), so show a small, meaningful number
        freq_center = np.mean(freq_list)
        rel_khz = (freq_list - freq_center) / 1e3

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        l1, = ax1.plot(rel_khz, SNR_list, 'o', c='C0', label='SNR (data)')
        l2, = ax1.plot(rel_khz, smoothed, '-', c='C0', label='SNR (smoothed)')
        l3, = ax2.plot(rel_khz, fid_list, 's', c='C2', label='Fidelity')
        vline = ax1.axvline((optimal_freq - freq_center) / 1e3, color='red', ls=':',
                             label=f'Optimal freq: {optimal_freq / 1e9:.6f} GHz '
                                   f'({(optimal_freq - freq_center) / 1e3:+.1f} kHz)')

        ax1.set_xlabel(f'Readout Frequency Offset from {freq_center / 1e9:.6f} GHz [kHz]')
        ax1.set_ylabel('SNR', color='C0')
        ax2.set_ylabel('Fidelity', color='C2')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax2.tick_params(axis='y', labelcolor='C2')

        ax1.legend(handles=[l1, l2, l3, vline], loc='best')
        ax1.set_title('Readout Frequency Optimization')
        plt.show()

    if not fit_ok:
        print(f'RO frequency fit failed (peak prominence={peak_prominence:.2f}, '
              f'noise={residual_std:.2f}). Not updated.')
        return None

    print(f'Optimal RO frequency = {optimal_freq / 1e9:.6f} GHz  '
          f'(peak prominence={peak_prominence:.2f}, noise={residual_std:.2f})')
    return optimal_freq

def RO_amp_fit(dataset, amp_list, plot=False):
    """Find the readout pulse amplitude that maximizes g/e discrimination fidelity.

    ``dataset`` is raw complex acquisition data (as returned by
    ``instrument_coordinator.retrieve_acquisition()``) with ground and excited
    shots interleaved column-wise per swept amplitude. For each amplitude,
    runs ``two_state_discriminator`` on the ground/excited IQ blobs to get the
    average fidelity and the ground-state fidelity, then smooths the
    average-fidelity-vs-amplitude trace with a Savitzky-Golay filter and
    returns the amplitude at the smoothed peak. Ground fidelity is reported
    alongside as a diagnostic only -- it does not drive the optimal amplitude.

    Returns
    -------
    float or None
        The optimal readout pulse amplitude, or ``None`` if the peak doesn't
        clearly rise above the trace's own noise floor (so callers should
        skip updating the qubit).
    """
    g_append = dataset[list(dataset.data_vars)[0]].values[:, 0::2]
    e_append = dataset[list(dataset.data_vars)[0]].values[:, 1::2]

    discrim = [
        two_state_discriminator(
            g_append[:, i].real, g_append[:, i].imag,
            e_append[:, i].real, e_append[:, i].imag,
            b_print=False, b_plot=False
        )
        for i in range(g_append.shape[1])
    ]
    avg_fid_list = np.array([d[2] for d in discrim])
    ground_fid_list = np.array([d[3] * 100 for d in discrim])
    amp_list = np.asarray(amp_list)

    # Savitzky-Golay window must be odd and no larger than the number of points
    window_length = min(7, amp_list.size if amp_list.size % 2 else amp_list.size - 1)
    polyorder = min(3, window_length - 1)

    if window_length < 3:
        print('RO amplitude fit failed: not enough points to smooth. Not updated.')
        return None

    smoothed = savgol_filter(avg_fid_list, window_length=window_length, polyorder=polyorder)

    peak_idx = np.argmax(smoothed)
    optimal_amp = amp_list[peak_idx]

    # goodness check: the smoothed peak should clearly rise above the noise
    # the filter smoothed away (the residual between raw data and the trend)
    residual_std = np.std(avg_fid_list - smoothed)
    peak_prominence = smoothed[peak_idx] - np.median(smoothed)
    fit_ok = residual_std > 0 and peak_prominence > 1 * residual_std

    if plot:
        fig, ax = plt.subplots()
        ax.plot(amp_list, avg_fid_list, 'o', c='C0', label='Avg fidelity (data)')
        ax.plot(amp_list, smoothed, '-', c='C0', label='Avg fidelity (smoothed)')
        ax.plot(amp_list, ground_fid_list, 's', c='C1', label='Ground fidelity')
        ax.axvline(optimal_amp, color='red', ls=':',
                   label=f'Optimal amp: {optimal_amp:.4f}')

        ax.set_xlabel('Readout Pulse Amplitude')
        ax.set_ylabel('Fidelity [%]')
        ax.legend(loc='best')
        ax.set_title('Readout Amplitude Optimization')
        plt.show()

    if not fit_ok:
        print(f'RO amplitude fit failed (peak prominence={peak_prominence:.2f}, '
              f'noise={residual_std:.2f}). Not updated.')
        return None

    print(f'Optimal RO amplitude = {optimal_amp:.4f}  '
          f'(avg fid={smoothed[peak_idx]:.2f}%, ground fid={ground_fid_list[peak_idx]:.2f}%, '
          f'peak prominence={peak_prominence:.2f}, noise={residual_std:.2f})')
    return optimal_amp

def RO_len_fit(dataset, duration_list, plot=False):
    """Find the readout duration that maximizes g/e discrimination fidelity.

    ``dataset`` is raw complex acquisition data (as returned by
    ``instrument_coordinator.retrieve_acquisition()``) with ground and excited
    shots interleaved column-wise per swept readout duration -- the readout
    pulse duration and integration time are swept together (see
    ``readout_len_optimization_TWPA``). For each duration, runs
    ``two_state_discriminator`` on the ground/excited IQ blobs to get the
    average fidelity and the ground-state fidelity, then smooths the
    average-fidelity-vs-duration trace with a Savitzky-Golay filter and
    returns the duration at the smoothed peak. Ground fidelity is reported
    alongside as a diagnostic only -- it does not drive the optimal duration.

    Returns
    -------
    float or None
        The optimal readout duration in seconds, or ``None`` if the peak
        doesn't clearly rise above the trace's own noise floor (so callers
        should skip updating the qubit).
    """
    g_append = dataset[list(dataset.data_vars)[0]].values[:, 0::2]
    e_append = dataset[list(dataset.data_vars)[0]].values[:, 1::2]

    discrim = [
        two_state_discriminator(
            g_append[:, i].real, g_append[:, i].imag,
            e_append[:, i].real, e_append[:, i].imag,
            b_print=False, b_plot=False
        )
        for i in range(g_append.shape[1])
    ]
    avg_fid_list = np.array([d[2] for d in discrim])
    ground_fid_list = np.array([d[3] * 100 for d in discrim])
    duration_list = np.asarray(duration_list)

    # Savitzky-Golay window must be odd and no larger than the number of points
    window_length = min(7, duration_list.size if duration_list.size % 2 else duration_list.size - 1)
    polyorder = min(3, window_length - 1)

    if window_length < 3:
        print('RO duration fit failed: not enough points to smooth. Not updated.')
        return None

    smoothed = savgol_filter(avg_fid_list, window_length=window_length, polyorder=polyorder)

    peak_idx = np.argmax(smoothed)
    optimal_duration = duration_list[peak_idx]

    # goodness check: the smoothed peak should clearly rise above the noise
    # the filter smoothed away (the residual between raw data and the trend)
    residual_std = np.std(avg_fid_list - smoothed)
    peak_prominence = smoothed[peak_idx] - np.median(smoothed)
    fit_ok = residual_std > 0 and peak_prominence > 1 * residual_std

    if plot:
        fig, ax = plt.subplots()
        ax.plot(duration_list * 1e6, avg_fid_list, 'o', c='C0', label='Avg fidelity (data)')
        ax.plot(duration_list * 1e6, smoothed, '-', c='C0', label='Avg fidelity (smoothed)')
        ax.plot(duration_list * 1e6, ground_fid_list, 's', c='C1', label='Ground fidelity')
        ax.axvline(optimal_duration * 1e6, color='red', ls=':',
                   label=f'Optimal duration: {optimal_duration * 1e6:.3f} us')

        ax.set_xlabel('Readout Duration [us]')
        ax.set_ylabel('Fidelity [%]')
        ax.legend(loc='best')
        ax.set_title('Readout Duration Optimization')
        plt.show()

    if not fit_ok:
        print(f'RO duration fit failed (peak prominence={peak_prominence:.2f}, '
              f'noise={residual_std:.2f}). Not updated.')
        return None

    print(f'Optimal RO duration = {optimal_duration * 1e6:.3f} us  '
          f'(avg fid={smoothed[peak_idx]:.2f}%, ground fid={ground_fid_list[peak_idx]:.2f}%, '
          f'peak prominence={peak_prominence:.2f}, noise={residual_std:.2f})')
    return optimal_duration

def readout_weight_extractor(g_trace, e_trace, qubit, lo_freq, division_length=10, plot=False):
    """Extract matched-filter readout integration weights from averaged g/e traces.

    ``g_trace``/``e_trace`` are the raw acquisition objects returned by
    ``readout_weight_optimization_TWPA`` for the qubit prepared in |g> and
    |e> (each a single hardware-averaged Trace acquisition, still modulated
    at the intermediate frequency). Demodulates both to baseband using the
    qubit's intermediate frequency (readout clock frequency minus
    ``lo_freq``), downsamples by averaging into ``division_length``-wide bins
    (smooths high-frequency noise out of the weight function before it's used
    as an integration weight), takes the normalized e-g difference as the
    matched filter, then upsamples back to the full 1 ns sample grid and
    L1-normalizes -- the format ``qubit.measure.acq_weights_a``/
    ``acq_weights_b`` expect.

    Returns
    -------
    (np.ndarray, np.ndarray) or None
        ``(weights_a, weights_b)``, the real/imaginary integration weight
        arrays, or ``None`` if the g/e traces show no measurable separation
        (so callers should skip updating the qubit).
    """
    g_trace = np.asarray(g_trace[0]).reshape(-1)
    e_trace = np.asarray(e_trace[0]).reshape(-1)

    pulse_duration = qubit.measure.pulse_duration()
    time_trace = np.arange(0, pulse_duration - 0.5e-9, 1e-9)
    nco_freq = qubit.clock_freqs.readout() - lo_freq

    n_samples = min(g_trace.size, e_trace.size, time_trace.size)
    time_trace = time_trace[:n_samples]
    demod_g = g_trace[:n_samples] * np.exp(-2j * np.pi * nco_freq * time_trace)
    demod_e = e_trace[:n_samples] * np.exp(-2j * np.pi * nco_freq * time_trace)

    n_bins = n_samples // division_length
    demod_g_div = np.mean(demod_g[:n_bins * division_length].reshape(-1, division_length), axis=1)
    demod_e_div = np.mean(demod_e[:n_bins * division_length].reshape(-1, division_length), axis=1)

    subtracted = demod_e_div - demod_g_div
    norm = np.sqrt(np.sum(np.abs(subtracted) ** 2))

    if not np.isfinite(norm) or norm == 0:
        print('Readout weight extraction failed: no measurable separation between g/e traces. Not updated.')
        return None

    normalized = subtracted / norm
    normalized = normalized / np.max(np.abs(normalized))

    weights_a = np.repeat(normalized.real, division_length)
    weights_a = weights_a / np.abs(weights_a).sum()

    weights_b = np.repeat(normalized.imag, division_length)
    weights_b = weights_b / np.abs(weights_b).sum()

    if plot:
        time_bins = time_trace[:n_bins * division_length].reshape(-1, division_length)[:, 0]
        plt.plot(time_bins * 1e6, normalized.real, label='real')
        plt.plot(time_bins * 1e6, normalized.imag, label='imag')
        plt.title('Readout Integration Weights')
        plt.xlabel('Readout time [us]')
        plt.ylabel('Normalized e-g separation [a.u.]')
        plt.legend()
        plt.show()

    print(f'Readout weights extracted (separation norm={norm:.3e}).')
    return weights_a, weights_b

def rotate_IQ_blob(IQ_ds):
    if isinstance(IQ_ds, str):
        ro_analysis = ReadoutCalibrationAnalysis(tuid=IQ_ds)
    else:
        ro_analysis = ReadoutCalibrationAnalysis(IQ_ds)
    ro_analysis.run()
    
    threshold = ro_analysis.quantities_of_interest['acq_threshold'].nominal_value
    rotation = 2*np.pi-ro_analysis.quantities_of_interest['acq_rotation_rad'].nominal_value
    
    ro_rotated = IQ_ds.copy()
    I_temp = np.cos(rotation) * ro_rotated.y0.values + np.sin(rotation) * ro_rotated.y1.values
    Q_temp = -np.sin(rotation) * ro_rotated.y0.values + np.cos(rotation) * ro_rotated.y1.values

    # if I_temp.mean() < 0:
        # threshold = -threshold
    
    ro_rotated.y0.values = I_temp
    ro_rotated.y1.values = Q_temp
    
    state0 = ro_rotated.where(ro_rotated.x0 == 0, drop=True)
    state1 = ro_rotated.where(ro_rotated.x0 == 1, drop=True)
    
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].plot(state0.y0.values, state0.y1.values, ".", alpha=0.1, label="Ground", markersize=5, c='r')
    ax[0].plot(state1.y0.values, state1.y1.values, ".", alpha=0.1, label="Excited", markersize=5, c='b')
    ax[0].axvline(x=threshold, color="k", ls="--", alpha=0.5)
    ax[0].axis("equal")
    ax[0].set_xlabel("I [V]")
    ax[0].set_ylabel("Q [V]")
    ax[0].set_title("Rotated Data")
    ax[0].legend()
    
    ax[1].hist(state0.y0.values, bins=50, alpha=0.5, label="Ground", color='r')
    ax[1].hist(state1.y0.values, bins=50, alpha=0.5, label="Excited", color='b')
    ax[1].axvline(x=threshold, color="k", ls="--", alpha=0.5)
    text_props = dict(
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[1].transAxes,
    )
    ax[1].text(0.7, 0.9, f"{threshold*1e6:.02f} $\mu V$", text_props)
    ax[1].set_xlabel("I [V]")
    ax[1].set_title("1D Histogram")

    ax[0].set_aspect('equal', 'box')
    fig.tight_layout()

from scipy.optimize import curve_fit

def v_line(x, a, b):
    return np.abs(a*x + b)

def dataset_current(tuid, source = 'GS210_1'):
    if not isinstance(tuid, str):
        tuid = tuid.tuid
    snapshot = load_snapshot(tuid)
    return round(snapshot['instruments'][source]['parameters']['output_level']['value']*1e3, 3)

def dataset_power(tuid, source = 'SGS100A_1'):
    if not isinstance(tuid, str):
        tuid = tuid.tuid
    snapshot = load_snapshot(tuid)
    return round(snapshot['instruments'][source]['parameters']['power']['value'], 3)

def rotate_real_imag(real, imag):
    """Rotate the IQ plane so that (almost) all of the signal lies on one axis.

    Finds the rotation angle that maximises the variance of the real
    component (the principal axis of the IQ cloud) and applies it to the
    complex signal ``real + 1j*imag``.

    Parameters
    ----------
    real, imag : array_like
        In-phase (I) and quadrature (Q) components.

    Returns
    -------
    rot_real, rot_imag : np.ndarray
        Rotated components. ``rot_real`` carries the information, ``rot_imag``
        is left close to noise.
    angle : float
        Applied rotation angle [rad].
    """
    real = np.asarray(real, dtype=float)
    imag = np.asarray(imag, dtype=float)

    z = real + 1j * imag
    z_c = z - z.mean()  # centre before estimating the principal axis

    # angle maximising Var(Re(z * exp(-1j*angle)))
    angle = 0.5 * np.arctan2(
        2 * np.mean(z_c.real * z_c.imag),
        np.mean(z_c.real**2) - np.mean(z_c.imag**2),
    )

    z_rot = z * np.exp(-1j * angle)
    return z_rot.real, z_rot.imag, angle
    
def ramsey_chevron_fit(tuid, qubit, plot_data = False, plot_fit = False, real_imag = True):
    if not isinstance(tuid, str):
        tuid = tuid.tuid
    if not isinstance(qubit, str):
        qubit = qubit.name
        
    ramsey_f01 = load_snapshot(tuid)['instruments'][qubit]['submodules']['clock_freqs']['parameters']['f01']['value']
    ram_che_ds = load_dataset(tuid)
    
    xlen = ram_che_ds.xlen
    ylen = ram_che_ds.ylen
    
    tau = np.unique(ram_che_ds.x0)
    detuning = np.unique(ram_che_ds.x1)
    if real_imag:
        signal = rotate_real_imag(ram_che_ds.y0.values, ram_che_ds.y1.values)[0]
        y0data = signal.reshape(ylen, xlen)
    else:
        signal = ram_che_ds.y0.values
        y0data = signal.reshape(ylen, xlen)

    if plot_data:
        plt.pcolor(tau*1e6, detuning/1e3, y0data, rasterized=True)
        plt.ylabel('Detuning [kHz]')
        plt.xlabel(r'$\tau$ [μs]')
        plt.show()

    T2_ram_arr = []
    fit_f_arr = []
    
    for i in range(ylen):
        sel = ram_che_ds.x1.values == detuning[i]
        t_i = ram_che_ds.x0.values[sel]
        guess = decay_osci_model.guess(signal[sel], t=t_i)
        fit_result = decay_osci_model.fit(signal[sel], params=guess, t=t_i)
    
        T2_ram_arr.append(fit_result.params['tau'].value)
        fit_f = fit_result.params['frequency'].value
        fit_f_arr.append(fit_f)
    
    T2_ram_arr = np.array(T2_ram_arr)
    fit_f_arr = np.array(fit_f_arr)

    # try both slope signs (the |a*x+b| kink makes curve_fit prone to getting stuck
    # in a local minimum depending on the initial guess) and keep the better fit
    x0_guess = detuning[np.argmin(fit_f_arr)]
    best = None
    for a0 in (1, -1):
        try:
            trial_params, trial_pcov = curve_fit(v_line, detuning, fit_f_arr, p0=[a0, x0_guess])
        except RuntimeError:
            continue
        trial_ss_res = np.sum((fit_f_arr - v_line(detuning, *trial_params)) ** 2)
        if best is None or trial_ss_res < best[0]:
            best = (trial_ss_res, trial_params, trial_pcov)

    ss_res, params, pcov = best

    f01_fit = ramsey_f01 + (-params[1] / params[0])

    # --- goodness-of-fit check for the V-line ---
    ss_tot = np.sum((fit_f_arr - np.mean(fit_f_arr)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    cov_finite = np.all(np.isfinite(pcov))                # inf covariance => unconstrained fit

    fit_ok = (r_squared > 0.9) and cov_finite

    q1, q3 = np.percentile(T2_ram_arr, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    cleaned_data = T2_ram_arr[(T2_ram_arr >= lower_bound) & (T2_ram_arr <= upper_bound)]
    mean_val = np.mean(cleaned_data)

    if fit_ok:
        print(f'f01 from Ramsey Chevron: {round(f01_fit)} [Hz]  (R^2={r_squared:.3f})')
    else:
        print(f'V-line fit failed (R^2={r_squared:.3f}, cov_finite={cov_finite}). '
              f'f01 not updated.')
    print(f'Ramsey T2: {round(mean_val * 1e6, 2)} [μs]')

    if plot_fit:
        x_fit = np.linspace(min(detuning), max(detuning), 200)
        y_fit = v_line(x_fit, *params)
        
        plt.scatter(detuning/1e3, fit_f_arr/1e3, label="Data")
        plt.plot(x_fit/1e3, y_fit/1e3, ls='--', label="Fitted V-line")
        
        plt.xlabel('Detuning [kHz]')
        plt.ylabel('Ramsey Frequency [kHz]')
        plt.grid()
        plt.legend()
        plt.show()

    if not fit_ok:
        return None

    return f01_fit

def conditional_ramsey_fit(con_ram_ds, plot = False, real_imag = True):
    if isinstance(con_ram_ds, str):
        con_ram_ds = load_dataset(con_ram_ds)
        
    condition = np.unique(con_ram_ds.x1)

    fit_result_list = []
    
    for i, c in enumerate(condition):
        temp_dataset = con_ram_ds.where(con_ram_ds.x1==c, drop=1)
        tau = np.unique(temp_dataset.x0)

        if real_imag:
            y0data = rotate_real_imag(temp_dataset.y0.values, temp_dataset.y1.values)[0]
        else:
            y0data = temp_dataset.y0.values

        guess = decay_osci_model.guess(y0data, t=tau)
        fit_result = decay_osci_model.fit(y0data, params=guess, t=tau)
        fit_result_list.append(fit_result)

        if c == 0:
            freq_0 = fit_result.params['frequency'].value
        elif c == 1:
            freq_1 = fit_result.params['frequency'].value
            delta_freq = freq_0 - freq_1

        if plot == True:
            if i == 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                
            t2     = fit_result.params['tau'].value
            freq   = fit_result.params['frequency'].value
            t_fine = np.linspace(tau.min(), tau.max(), 500)

            ax.plot(tau * 1e6, y0data, 'o', ms=5, color=f'C{i}',
                    label=f'ctrl |{c:g}⟩ data')
            # ax.plot(t_fine * 1e6, fit_result.eval(t=t_fine), '--', lw=1.5, color=f'C{i}', alpha=0.75,
            #         label=f'ctrl |{c:g}⟩ fit (T2*={np.round(t2*1e6):.0f} μs, f={np.round(freq/1e3):.0f} kHz)')
            ax.plot(t_fine * 1e6, fit_result.eval(t=t_fine), '--', lw=1.5, color=f'C{i}', alpha=0.75,
                    label=f'ctrl |{c:g}⟩ fit (f={np.round(freq/1e3):.0f} kHz)')

    if plot == True:
        ax.set_xlabel(r'Ramsey delay $\tau$ (μs)')
        ax.set_ylabel('Signal (a.u.)')
        ax.set_title(f'Conditional Ramsey\n{con_ram_ds.tuid}')
        ax.legend(fontsize=10, loc='upper right')
        fig.tight_layout()

    return delta_freq, fit_result_list

def rabi_amplification_fit(dataset, plot=False, real_imag=True):
    """Find the pi-pulse amplitude from an error-amplification Rabi dataset.

    The signal is summed along the number-of-pulses axis; the correct amplitude
    sits at the minimum of that sum. A local parabola is fit around the minimum,
    which gives an analytic vertex, an uncertainty, and a goodness metric.

    Returns
    -------
    float or None
        The calibrated Rabi amplitude, or ``None`` if the parabola fit is
        untrustworthy (so callers should skip updating amp180).
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    xlen = dataset.xlen
    ylen = dataset.ylen

    rabi_amp = np.unique(dataset.x0)
    nb_rabi = np.unique(dataset.x1)
    if real_imag:
        y0data = np.abs(rotate_real_imag(dataset.y0.values, dataset.y1.values)[0].reshape(ylen, xlen))
    else:
        y0data = dataset.y0.values.reshape(ylen, xlen)

    # The sign of the rotated signal is arbitrary, so the pi amplitude can be a dark stripe or a bright
    # ridge. Use the deviation from the zero-pulse row instead: at the correct amplitude every even number
    # of pi pulses returns the qubit to the state of the N = 0 row, so the summed |deviation| is smallest
    # there whatever the polarity. Fall back to the plain sum if the first row is not N = 0.
    if nb_rabi[0] == 0:
        sum_pulses = np.abs(y0data - y0data[0].mean()).sum(axis=0)
    else:
        sum_pulses = y0data.sum(axis=0)

    # window around the minimum: keep points between the flanks that rise above
    # the 60th percentile on either side of the dip
    eliminator = np.where(sum_pulses > np.percentile(sum_pulses, 60))[0]
    left_side = eliminator[eliminator < np.argmin(sum_pulses)]
    right_side = eliminator[eliminator > np.argmin(sum_pulses)]
    if left_side.size == 0 or right_side.size == 0:
        print('Rabi amplification fit failed: the dip is at the edge of the amplitude window '
              '(min at %.4f V, window %.4f - %.4f V). Not updated.' % (rabi_amp[np.argmin(sum_pulses)], rabi_amp[0], rabi_amp[-1]))
        return None
    left_indice = left_side.max()
    right_indice = right_side.min()

    rabi_amp_fit = rabi_amp[left_indice:right_indice + 1]
    sum_pulses_fit = sum_pulses[left_indice:right_indice + 1]

    # local parabola fit near the minimum
    def parabola(x, a, b, c):
        return a * x ** 2 + b * x + c

    p0 = [1.0, 0.0, sum_pulses_fit.min()]
    popt, pcov = curve_fit(parabola, rabi_amp_fit, sum_pulses_fit, p0=p0)
    a, b, c = popt

    RABI_AMP = -b / (2 * a)

    # uncertainty on the vertex, propagated from the a, b covariance
    dv_da = b / (2 * a ** 2)
    dv_db = -1 / (2 * a)
    var_v = dv_da ** 2 * pcov[0, 0] + dv_db ** 2 * pcov[1, 1] + 2 * dv_da * dv_db * pcov[0, 1]
    RABI_AMP_err = np.sqrt(var_v) if var_v > 0 else np.inf

    # goodness of fit
    resid = sum_pulses_fit - parabola(rabi_amp_fit, *popt)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((sum_pulses_fit - sum_pulses_fit.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    amp_span = rabi_amp_fit.max() - rabi_amp_fit.min()
    fit_ok = (
        a > 0                                             # concave up => a real minimum
        and r_squared > 0.85                              # parabola matches the dip
        and rabi_amp_fit.min() < RABI_AMP < rabi_amp_fit.max()  # vertex inside the window
        and RABI_AMP_err < 0.1 * amp_span                 # vertex well-pinned
    )

    if plot:
        plt.pcolor(rabi_amp, nb_rabi, y0data, rasterized=True)
        plt.xlabel('Rabi Pulse Amplitude [V]')
        plt.ylabel(r'Number of $\pi$ Pulses')
        plt.show()

        plt.plot(rabi_amp, sum_pulses, label='Data')
        plt.plot(rabi_amp_fit, parabola(rabi_amp_fit, *popt), ls='--', label='Parabola fit')
        plt.axvline(RABI_AMP, color='k', ls=':', label=f'RABI_AMP = {RABI_AMP:.4f} V')
        plt.xlabel('Rabi Pulse Amplitude [V]')
        plt.ylabel('Sum Along Pulses')
        plt.legend()
        plt.show()

    if not fit_ok:
        print(f'Rabi amplification fit failed (R^2={r_squared:.3f}, a={a:.3g}, '
              f'err={RABI_AMP_err:.4f}). RABI_AMP not updated.')
        return None

    print(f'RABI_AMP = {RABI_AMP:.4f} ± {RABI_AMP_err:.4f} V  (R^2={r_squared:.3f}, a={a:.3g})')

    return RABI_AMP


def _fast_exp_decay(tau, y):
    """Fit y = amplitude * exp(-t/decay) + offset and return (decay, r_squared).

    Identical result to lmfit's ExpDecayModel to the printed digits, about 20x faster:
    the cost in lmfit is the Parameters machinery and its guess(), not the optimisation.
    The seed comes from the data directly, so no guess step is needed.
    """
    try:
        popt, _ = curve_fit(
            lambda t, a, d, c: a * np.exp(-t / d) + c,
            tau, y, p0=[y[0] - y[-1], tau[-1] / 3, y[-1]], maxfev=2000)
    except Exception:
        return np.nan, 0.0
    residual = y - (popt[0] * np.exp(-tau / popt[1]) + popt[2])
    total = y - y.mean()
    return popt[1], 1 - residual.dot(residual) / total.dot(total)

def t1_and_t2_fit(dataset, r_squared=0.8):
    """ 
    Fit T1, T2 (echo) and T2* (Ramsey, case 3) in μs from a dataset with ThresholdAcquisition.
    Might contain multi qubit data.
    returns two lists of lists:
    [[q0_t1, q0_t2], [q1_t1, q1_t2]] when the dataset contains no case 3 data, otherwise
    [[q0_t1, q0_t2, q0_t2_ramsey, q0_ramsey_freq], [q1_t1, q1_t2, q1_t2_ramsey, q1_ramsey_freq]].
    Ramsey frequencies are in MHz (tau is fitted in μs).
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    
    tau = np.unique(dataset.x0) * 1e6  # Convert to microseconds

    t1_dataset = dataset.where(dataset.x1 == 1, drop=1)

    y0_data = t1_dataset.y0.values.reshape(-1, len(tau))
    y1_data = t1_dataset.y1.values.reshape(-1, len(tau))
    multi = not np.isnan(y1_data).any()

    q0_t1_arr = []
    q1_t1_arr = []
    q0_t2_arr = []
    q1_t2_arr = []
    q0_t2_ramsey_arr = []
    q1_t2_ramsey_arr = []
    q0_ramsey_freq_arr = []
    q1_ramsey_freq_arr = []

    # guess_q0 = exp_decay_model.guess(y0_data[0], delay=tau)
    # if multi:
    #     guess_q1 = exp_decay_model.guess(y1_data[0], delay=tau)

    for i in range(y0_data.shape[0]):
        decay_q0, rsq_q0 = _fast_exp_decay(tau, y0_data[i])
        q0_t1_arr.append(decay_q0 if rsq_q0 > r_squared else np.nan)

        if multi:
            decay_q1, rsq_q1 = _fast_exp_decay(tau, y1_data[i])
            q1_t1_arr.append(decay_q1 if rsq_q1 > r_squared else np.nan)

    t2_dataset = dataset.where(dataset.x1 == 2, drop=1)

    y0_data = t2_dataset.y0.values.reshape(-1, len(tau))
    y1_data = t2_dataset.y1.values.reshape(-1, len(tau))
    multi = not np.isnan(y1_data).any()

    # guess_q0 = exp_decay_model.guess(y0_data[0], delay=tau)
    # if multi:
    #     guess_q1 = exp_decay_model.guess(y1_data[0], delay=tau)

    for i in range(y0_data.shape[0]):
        decay_q0, rsq_q0 = _fast_exp_decay(tau, y0_data[i])
        q0_t2_arr.append(decay_q0 if rsq_q0 > r_squared else np.nan)

        if multi:
            decay_q1, rsq_q1 = _fast_exp_decay(tau, y1_data[i])
            q1_t2_arr.append(decay_q1 if rsq_q1 > r_squared else np.nan)

    if not np.any(dataset.x1.values == 3):
        return [q0_t1_arr, q0_t2_arr], [q1_t1_arr, q1_t2_arr]

    ramsey_dataset = dataset.where(dataset.x1 == 3, drop=1)

    y0_data = ramsey_dataset.y0.values.reshape(-1, len(tau))
    y1_data = ramsey_dataset.y1.values.reshape(-1, len(tau))
    multi = not np.isnan(y1_data).any()

    for i in range(y0_data.shape[0]):
        guess_q0 = decay_osci_model.guess(y0_data[i], t=tau)
        fit_result_q0 = decay_osci_model.fit(y0_data[i], params=guess_q0, t=tau)
        good_q0 = fit_result_q0.rsquared > r_squared
        q0_t2_ramsey_arr.append(fit_result_q0.params['tau'].value if good_q0 else np.nan)
        q0_ramsey_freq_arr.append(fit_result_q0.params['frequency'].value if good_q0 else np.nan)

        if multi:
            guess_q1 = decay_osci_model.guess(y1_data[i], t=tau)
            fit_result_q1 = decay_osci_model.fit(y1_data[i], params=guess_q1, t=tau)
            good_q1 = fit_result_q1.rsquared > r_squared
            q1_t2_ramsey_arr.append(fit_result_q1.params['tau'].value if good_q1 else np.nan)
            q1_ramsey_freq_arr.append(fit_result_q1.params['frequency'].value if good_q1 else np.nan)

    return (
        [q0_t1_arr, q0_t2_arr, q0_t2_ramsey_arr, q0_ramsey_freq_arr],
        [q1_t1_arr, q1_t2_arr, q1_t2_ramsey_arr, q1_ramsey_freq_arr],
    )

def iswap_chevron_fit(dataset, plot_data=False, plot_fit=False):
    """Find the iSWAP pump frequency from an iSWAP-chevron dataset.

    The two-qubit populations are swept versus pump duration (x0) and pump
    frequency (x1). The per-frequency contrast (sum of each qubit's population
    std across duration) peaks at the iSWAP resonance, so a Lorentzian is fit to
    contrast versus frequency and its centre gives the pump frequency.

    Returns
    -------
    float or None
        The fitted iSWAP pump frequency in Hz, or ``None`` if the Lorentzian fit
        is untrustworthy (so callers should skip updating the pump frequency).
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    x0_data = np.unique(dataset.x0)
    x1_data = np.unique(dataset.x1)
    y0_data = dataset.y0.values.reshape(x1_data.size, x0_data.size)
    y1_data = dataset.y1.values.reshape(x1_data.size, x0_data.size)
    contrast = np.std(y0_data, axis=1) + np.std(y1_data, axis=1)

    if plot_data:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'wspace': 0.05})

        pcm0 = axes[0].pcolormesh(x0_data * 1e6, x1_data / 1e6, y0_data, shading='auto', vmin=0, vmax=1, rasterized=True)
        axes[0].set_title('Q1 Population')
        axes[0].set_xlabel('Pump Duration [μs]')
        axes[0].set_ylabel('Pump Frequency [MHz]')

        pcm1 = axes[1].pcolormesh(x0_data * 1e6, x1_data / 1e6, y1_data, shading='auto', vmin=0, vmax=1, rasterized=True)
        fig.colorbar(pcm1, ax=axes.tolist(), pad=0.02)
        axes[1].set_title('Q2 Population')
        axes[1].set_xlabel('Pump Duration [μs]')
        axes[1].set_yticks([])
        plt.show()

    guess = lorentzian_model.guess(contrast, x=x1_data)
    guess['a'].min = 0
    guess['x0'].value = x1_data[np.argmax(contrast)]
    guess['c'].value = x1_data[np.argmin(contrast)]
    fit_result = lorentzian_model.fit(contrast, params=guess, x=x1_data)

    freq = round(fit_result.params['x0'].value)
    r_squared = fit_result.rsquared

    fit_ok = (
        fit_result.success
        and np.isfinite(freq)
        and x1_data.min() <= freq <= x1_data.max()   # peak inside the swept range
        and r_squared > 0.9                          # Lorentzian matches the contrast
    )

    if plot_fit:
        fit_result.plot_fit()
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1e6:g}'))
        plt.xlabel('Pump Frequency [MHz]')
        plt.ylabel('Population Std. Dev.')
        plt.title('iSWAP Chevron Frequency Fit')
        plt.show()

    if not fit_ok:
        print(f'iSWAP Chevron fit failed (R^2={r_squared:.3f}, '
              f'freq={freq / 1e6:.3f} MHz). Frequency not updated.')
        return None

    print(f'iSWAP pump frequency = {freq} Hz  (R^2={r_squared:.3f})')

    return freq


def iswap_theta_p_fit(dataset, plot=False):
    """Find the iSWAP pulse duration that gives a swap angle theta_p = pi/2.

    Repeated iSWAPs (x0 = number of gates N) are swept versus pump pulse
    duration (x1); y0/y1 are the two qubits' excited-state populations. The
    excitation hops qubit on every gate (RPE sequence of arXiv 2604.27080,
    Fig. 5(a,b)), and after N gates

        P_start = total * (1 + c) / 2 + b_start
        P_other = total * (1 - c) / 2 + b_other
        c       = cos(2 N theta_p) * exp(-N / tau2)
        total   = a * exp(-N / tau1)

    Both qubits are fit jointly per duration with a shared theta_p. The two
    decays are kept separate on purpose: tau1 is excitation lost to T1 (both
    qubits -> ground), tau2 is lost swap contrast. Folding them into one
    envelope that decays to the mixed state forces theta_p to fake the
    population loss and leaves a spurious floor of ~0.004 pi at every duration.

    cos(2 N theta_p) is identical for pi/2 + delta and pi/2 - delta, so one
    duration only yields |delta|, and delta = 0 is a stationary point of the
    model -- hence the multi-start. theta_p grows linearly with duration, so
    |delta| versus duration is a V whose vertex is the pi/2 point. That also
    fixes the sign: durations below the vertex under-rotate, above it
    over-rotate.

    Returns
    -------
    dict or None
        ``pulse_duration`` (s): the fitted pi/2 crossing.
        ``pulse_duration_err`` (s): its 1-sigma uncertainty.
        ``pulse_duration_rounded`` (s): ``pulse_duration`` rounded to 1 ns.
        ``slope`` (rad/s): d theta_p / d duration.
        ``durations`` (s) and ``theta_p`` (rad): the signed swap angle at every
        swept duration.
        ``None`` if the fit is untrustworthy (so callers should skip updating
        the pulse duration).
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    n_gates = np.unique(dataset.x0).astype(float)
    durations = np.unique(dataset.x1)
    y0_data = dataset.y0.values.reshape(durations.size, n_gates.size)
    y1_data = dataset.y1.values.reshape(durations.size, n_gates.size)

    # Whichever qubit holds the excitation at even N follows (1 + c) / 2.
    even = (n_gates % 2 == 0)
    diff = y1_data - y0_data
    start_on_y1 = np.mean(diff[:, even]) - np.mean(diff[:, ~even]) > 0
    y_start, y_other = (y1_data, y0_data) if start_on_y1 else (y0_data, y1_data)

    def model(N2, delta, tau1, tau2, a, b_start, b_other):
        N = N2[:N2.size // 2]           # N2 is N stacked twice: start channel, then other
        total = a * np.exp(-N / tau1)
        c = np.cos(2 * N * (np.pi / 2 + delta)) * np.exp(-N / tau2)
        return np.concatenate([total * (1 + c) / 2 + b_start,
                               total * (1 - c) / 2 + b_other])

    N2 = np.concatenate([n_gates, n_gates])
    bounds = ([0, 1, 1, 0, 0, 0], [0.1 * np.pi, np.inf, np.inf, 1, 1, 1])

    delta = np.full(durations.size, np.nan)
    delta_err = np.full(durations.size, np.nan)
    for i in range(durations.size):
        ydat = np.concatenate([y_start[i], y_other[i]])
        a0 = np.clip(np.max(y_start[i]), 0.1, 1)
        best = None
        for d0 in np.linspace(0.0005, 0.03, 12) * np.pi:   # multi-start: delta = 0 has zero gradient
            try:
                popt, pcov = curve_fit(model, N2, ydat, p0=[d0, 60, 200, a0, 0.01, 0.01], bounds=bounds)
            except (RuntimeError, ValueError):
                continue
            cost = np.sum((model(N2, *popt) - ydat) ** 2)
            if best is None or cost < best[0]:
                best = (cost, popt, pcov)
        if best is not None:
            delta[i] = best[1][0]
            delta_err[i] = np.sqrt(np.diag(best[2]))[0]

    valid = np.isfinite(delta)
    d_ns = durations * 1e9
    if valid.sum() < 4:
        print(f'iSWAP theta_p fit failed ({valid.sum()} of {durations.size} durations fit). '
              f'Pulse duration not updated.')
        return None

    # A fit landing at delta = 0 sits on the bound, where the model depends on
    # delta only through delta^2 -- its covariance blows up and would give the
    # most informative point zero weight. Cap each error at 3x the median.
    sigma = np.clip(delta_err, 1e-5, 3 * np.nanmedian(delta_err[valid]))

    def v_shape(d, k, d0, floor):
        # rounded V; the floor absorbs the delta >= 0 noise bias near the vertex
        return np.sqrt((k * (d - d0)) ** 2 + floor ** 2)

    i_min = np.nanargmin(delta)
    try:
        vopt, vcov = curve_fit(v_shape, d_ns[valid], delta[valid],
                               p0=[0.001 * np.pi, d_ns[i_min], 1e-4],
                               sigma=sigma[valid], absolute_sigma=True)
    except (RuntimeError, ValueError):
        print('iSWAP theta_p V fit did not converge. Pulse duration not updated.')
        return None

    k_ns, d0_ns = abs(vopt[0]), vopt[1]
    d0_err_ns = np.sqrt(vcov[1, 1]) if np.isfinite(vcov[1, 1]) else np.nan

    residuals = delta[valid] - v_shape(d_ns[valid], *vopt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((delta[valid] - np.mean(delta[valid])) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # theta_p grows ~linearly with pulse duration from zero, so the slope should be
    # roughly (pi/2) / d0. A wildly different slope means the V is not theta_p.
    slope_ratio = k_ns / ((np.pi / 2) / d0_ns) if d0_ns > 0 else np.nan

    fit_ok = (
        np.isfinite(d0_ns)
        and np.isfinite(d0_err_ns)
        and np.all(np.isfinite(vcov))
        and 0.5 < slope_ratio < 2
        and r_squared > 0.8
    )

    theta_p = np.pi / 2 + np.sign(d_ns - d0_ns) * delta   # unfolded using the V

    if plot:
        dd = np.linspace(d_ns.min(), d_ns.max(), 300)
        plt.errorbar(d_ns[valid], delta[valid] / np.pi, yerr=sigma[valid] / np.pi, fmt='o',
                     label=r'fit $|\theta_p-\pi/2|$')
        plt.plot(dd, v_shape(dd, *vopt) / np.pi, '-', label='V fit')
        plt.axvline(d0_ns, color='r', ls='--', label=f'{d0_ns:.1f} ns')
        plt.xlabel('iSWAP Pulse Duration [ns]')
        plt.ylabel(r'$|\theta_p - \pi/2|\ /\ \pi$')
        plt.title(r'$\theta_p$ vs Duration')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    if not fit_ok:
        print(f'iSWAP theta_p fit failed (R^2={r_squared:.3f}, slope ratio={slope_ratio:.2f}, '
              f'pi/2 at {d0_ns:.2f} ns). Pulse duration not updated.')
        return None

    print(f'iSWAP theta_p = pi/2 at {d0_ns:.2f} +- {d0_err_ns:.2f} ns  '
          f'(slope {k_ns / np.pi * 1e3:.3f} mpi/ns, R^2={r_squared:.3f})')

    return {
        'pulse_duration': d0_ns * 1e-9,
        'pulse_duration_err': d0_err_ns * 1e-9,
        'pulse_duration_rounded': round(d0_ns) * 1e-9,
        'slope': k_ns * 1e9,
        'durations': durations,
        'theta_p': theta_p,
    }


def rpe_rate_fit(dataset, channel='y0', plot=False):
    """Fit the per-step phase of an iSWAP RPE sequence (``iswap_RPE_d``,
    ``iswap_RPE_f``, ``iswap_RPE_theta_sum``).

    Model: ``z(m) = a exp(-m/tau) cos(w m + phi0) + c`` on ``z = P2Z(P)``,
    with the rate ``w`` in degrees per step, restricted to [0, 180]. A single
    projection cannot tell ``+w`` from ``-w``, nor ``w`` from ``360 - w``, so
    this returns a magnitude; the caller resolves the sign (e.g. from a +/-
    offset bracket). The fit is multi-started over ``w`` because a single
    start lands in local minima (115.93 deg instead of 90.7 deg on RPE_d,
    cell 644).

    Near ``w = 0`` or ``180`` one projection cannot separate slow rotation
    from decay, and the rate error blows up -- the gate below rejects those.

    Returns
    -------
    dict or None
        ``{'rate', 'rate_err', 'phase0', 'tau', 'r2'}`` (angles in degrees),
        or ``None`` if the fit is untrustworthy.
    """
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    x = np.asarray(dataset.x0.values, dtype=float)
    z = P2Z(np.asarray(dataset[channel].values, dtype=float))
    ok = np.isfinite(z)
    x, z = x[ok], z[ok]

    def model(m, a, tau, w, ph, c):
        return a * np.exp(-m / tau) * np.cos(np.deg2rad(w * m + ph)) + c

    best = None
    for w0 in np.linspace(1, 179, 60):
        for ph0 in (-90, 0, 90):
            try:
                popt, pcov = curve_fit(model, x, z, p0=[0.9, 15, w0, ph0, 0.1],
                                       bounds=([0, 1, 0, -360, -1], [2, 300, 180, 360, 1]),
                                       maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            cost = np.sum((model(x, *popt) - z) ** 2)
            if best is None or cost < best[0]:
                best = (cost, popt, pcov)

    if best is None:
        print(f'RPE rate fit failed ({channel}): no converged start.')
        return None

    cost, popt, pcov = best
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r_squared = 1 - cost / ss_tot if ss_tot > 0 else 0.0
    cov_finite = np.all(np.isfinite(pcov))
    rate_err = float(np.sqrt(pcov[2, 2])) if cov_finite else np.inf

    fit_ok = cov_finite and r_squared > 0.85 and rate_err < 5.0

    if plot:
        xx = np.linspace(x.min(), x.max(), 600)
        plt.plot(x, z, 'o', ms=4, label=channel)
        plt.plot(xx, model(xx, *popt), '-', label=f'{popt[2]:.2f} deg/step')
        plt.xlabel('Repetitions')
        plt.ylabel(r'$\langle Z \rangle$')
        plt.title(dataset.attrs.get('name', 'RPE'))
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    if not fit_ok:
        print(f'RPE rate fit rejected ({channel}): rate={popt[2]:.2f} +- {rate_err:.2f} deg, '
              f'R^2={r_squared:.3f}.')
        return None

    print(f'RPE rate ({channel}) = {popt[2]:.2f} +- {rate_err:.2f} deg/step  (R^2={r_squared:.3f})')
    return {
        'rate': float(popt[2]),
        'rate_err': rate_err,
        'phase0': float(popt[3]),
        'tau': float(popt[1]),
        'r2': float(r_squared),
    }


def rpe_quadrature_fit(dataset_x, dataset_y, psi_range=(-180.0, 180.0), channel='y0', plot=False):
    """Joint fit of the cos (closing X90) and sin (closing Y90) quadratures of
    an iSWAP RPE sequence -- ``iswap_RPE_theta_sum`` / ``iswap_RPE_f`` run with
    ``final_axis='x'`` and ``'y'``.

    Model, ``z = P2Z(P)``::

        z_x(m) = c_x + a exp(-m/tau) cos(psi m + phi0)
        z_y(m) = c_y + a exp(-m/tau) sin(psi m + phi0)

    Unlike ``rpe_rate_fit`` this returns the per-step phase ``psi`` with its
    sign, and the one-time phase ``phi0``, and it stays well conditioned at
    ``psi`` near 0 or 180 deg, where a single projection cannot separate
    rotation from decay. Points with ``m = 0`` are excluded from the fit.

    Sign convention (measured 2026-10-02 with +/-20 deg offsets on the edge):
    the correction to add to the edge equals the reading -- ``psi - 180`` on
    the sum for RPE_theta_sum, ``psi`` on the difference for RPE_f.

    Parameters
    ----------
    psi_range : (float, float)
        Search window for ``psi`` in degrees: ``(90, 270)`` for
        RPE_theta_sum (rate near 180/pair), ``(-120, 120)`` for RPE_f.

    Returns
    -------
    dict or None
        ``{'psi', 'psi_err', 'phi0', 'phi0_err', 'tau', 'amp', 'r2'}`` in
        degrees, or ``None`` if the fit is untrustworthy.
    """
    if isinstance(dataset_x, str):
        dataset_x = load_dataset(dataset_x)
    if isinstance(dataset_y, str):
        dataset_y = load_dataset(dataset_y)

    m = np.asarray(dataset_x.x0.values, dtype=float)
    m_y = np.asarray(dataset_y.x0.values, dtype=float)
    if m.shape != m_y.shape or np.any(m != m_y):
        print('RPE quadrature fit: x and y runs have different lengths. Not fitted.')
        return None
    zx = P2Z(np.asarray(dataset_x[channel].values, dtype=float))
    zy = P2Z(np.asarray(dataset_y[channel].values, dtype=float))
    keep = (m >= 1) & np.isfinite(zx) & np.isfinite(zy)
    m, zx, zy = m[keep], zx[keep], zy[keep]

    mm2 = np.concatenate([m, m])
    zz = np.concatenate([zx, zy])

    def model(MM, cx, cy, a, tau, psi, ph):
        n = MM.size // 2
        mm = MM[:n]
        env = a * np.exp(-mm / tau)
        arg = np.deg2rad(psi * mm + ph)
        return np.concatenate([cx + env * np.cos(arg), cy + env * np.sin(arg)])

    lo, hi = psi_range
    best = None
    for psi0 in np.arange(lo + 2, hi, 4.0):
        for ph0 in (-120, 0, 120):
            try:
                popt, pcov = curve_fit(model, mm2, zz, p0=[0, 0, 0.9, 15, psi0, ph0],
                                       bounds=([-1, -1, 0, 1, lo, -540], [1, 1, 2, 500, hi, 540]),
                                       maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            cost = np.sum((model(mm2, *popt) - zz) ** 2)
            if best is None or cost < best[0]:
                best = (cost, popt, pcov)

    if best is None:
        print('RPE quadrature fit failed: no converged start.')
        return None

    cost, popt, pcov = best
    ss_tot = np.sum((zz - np.mean(zz)) ** 2)
    r_squared = 1 - cost / ss_tot if ss_tot > 0 else 0.0
    cov_finite = np.all(np.isfinite(pcov))
    err = np.sqrt(np.diag(pcov)) if cov_finite else np.full(6, np.inf)
    psi, phi0 = float(popt[4]), float((popt[5] + 180) % 360 - 180)

    fit_ok = cov_finite and r_squared > 0.85 and err[4] < 3.0 and popt[2] > 0.3

    if plot:
        xx = np.linspace(m.min(), m.max(), 600)
        fitted = model(np.concatenate([xx, xx]), *popt)
        plt.plot(m, zx, 'o', ms=4, label='cos (X90)')
        plt.plot(m, zy, 's', ms=4, label='sin (Y90)')
        plt.plot(xx, fitted[:xx.size], '-', color='C0')
        plt.plot(xx, fitted[xx.size:], '-', color='C1')
        plt.xlabel('Repetitions')
        plt.ylabel(r'$\langle Z \rangle$')
        plt.title(f'{dataset_x.attrs.get("name", "RPE")}: psi = {psi:.2f} deg')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    if not fit_ok:
        print(f'RPE quadrature fit rejected: psi={psi:.2f} +- {err[4]:.2f} deg, '
              f'amp={popt[2]:.2f}, R^2={r_squared:.3f}.')
        return None

    print(f'RPE quadrature: psi = {psi:.2f} +- {err[4]:.2f} deg/step, phi0 = {phi0:+.1f} '
          f'+- {err[5]:.1f} deg  (R^2={r_squared:.3f})')
    return {'psi': psi, 'psi_err': float(err[4]), 'phi0': phi0, 'phi0_err': float(err[5]),
            'tau': float(popt[3]), 'amp': float(popt[2]), 'r2': float(r_squared)}


def rb_simple_fit(dataset, plot=True):
    num_gates = np.unique(dataset.x0)
    num_runs = np.arange(np.unique(dataset.x1).size)
    
    if len(dataset.data_vars) == 2:
        print("Single Qubit RB")
        y_vars = [('y0', 'y1', 'Single Qubit')]
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            axs = [ax]
    elif len(dataset.data_vars) == 4:
        print("Simultaneous RB")
        y_vars = [('y0', 'y1', 'Qubit 1'), ('y2', 'y3', 'Qubit 2')]
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    else:
        print("Unknown dataset format")
        return
    
    gate_fid_arr = []
    for idx, (y_amp, y_phase, label) in enumerate(y_vars):
        def rb_decay(
            m: int,
            alpha: float,
        ) -> float:
            """Exponential decay consistent with eq.(1) of arxiv:1712.06550."""
            return 0.5 * alpha**m + 0.5

        def power_law(power, p, a, b):
            return a * (p**power) + b

        def rb_fidelity(p, n_avg=1.875):
            F_cliff = (1 + p) / 2
            F_gate = 1 - (1 - F_cliff) / n_avg
            return F_cliff, F_gate

        S21_RB = getattr(dataset, y_amp) * np.cos(np.deg2rad(getattr(dataset, y_phase))) + 1j * getattr(dataset, y_amp) * np.sin(np.deg2rad(getattr(dataset, y_phase)))
        S21_RB = S21_RB.values.reshape(num_runs.size, num_gates.size)
        I_RB = S21_RB.real
        Q_RB = S21_RB.imag

        e_popu_full = []
        for run in range(num_runs.size):
            test_I = I_RB[run][:-2]
            test_Q = Q_RB[run][:-2]
            test_I_g = I_RB[run][-2]
            test_Q_g = Q_RB[run][-2]
            test_I_e = I_RB[run][-1]
            test_Q_e = Q_RB[run][-1]

            ge_distance = np.linalg.norm([test_I_e - test_I_g, test_Q_e - test_Q_g])**2

            e_popu_arr = []
            for i in range(test_I.size):
                e_popu_arr.append(np.dot([test_I[i] - test_I_g, test_Q[i] - test_Q_g], [test_I_e - test_I_g, test_Q_e - test_Q_g]) / ge_distance)
            e_popu_full.append(e_popu_arr)

        e_popu_full = np.array(e_popu_full)
        g_popu_full = 1 - e_popu_full

        popt, pcov = curve_fit(
            rb_decay,
            num_gates[:-2],
            g_popu_full.mean(axis=0),
            p0=[0.99],
        )
        
        gate_fid = rb_fidelity(popt[0])[1]
        print(f"{label} single gate fidelity: {gate_fid:.4f}")
        gate_fid_arr.append(gate_fid)

        if plot:
            ax = axs[idx]
            ax.plot(num_gates[:-2], g_popu_full.T, '.', c='grey', alpha=0.5)
            ax.plot(num_gates[:-2], g_popu_full.mean(axis=0), 'o')
            ax.plot(num_gates[:-2], rb_decay(num_gates[:-2], *popt), c='black')
            ax.grid()

            ax.set_ylabel('Ground state population')
            ax.set_xlabel('Number of Clifford gates')
            ax.set_title(f'Randomized Benchmarking on {label} \n {dataset.tuid}')

            ax.text(num_gates[-3], 1, f'α={round(popt[0],4)}', ha='right', va='top')
            ax.text(num_gates[-3], 0.95, f'single gate F={round(gate_fid,4)}', ha='right', va='top')
    if plot:
        plt.tight_layout()
        plt.show()
    return gate_fid_arr

######################################
# Set-up a Bloch sphere for plotting #  (can be removed if not used)
######################################
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

# Section for saving retrieve_acquisition dataset
# -----------------------------------------------------------------------------
from pathlib import Path
import json
import re
from quantify_core.data import handling as dh

# Helpers: sanitize dataset for netCDF write (for retrieve_acquisition dataset)
# -----------------------------------------------------------------------------
_NETCDF_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _netcdf_safe_name(name: str) -> bool:
    return bool(_NETCDF_SAFE.match(name))

def _netcdf_safe_attr_value(v):
    """
    Coerce attrs to netCDF-safe scalar/list-ish types.
    """
    if v is None:
        return "None"
    if isinstance(v, (str, int, float, bool, bytes)):
        return v
    if isinstance(v, np.number):
        return v.item()
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (list, tuple)):
        return type(v)(_netcdf_safe_attr_value(x) for x in v)
    if isinstance(v, dict):
        # netCDF attrs cannot be dicts; store as JSON string
        return json.dumps(
            {str(k): _netcdf_safe_attr_value(val) for k, val in v.items()},
            default=str,
        )
    return str(v)

def _sanitize_dataset_for_netcdf(ds):
    """
    Make an xarray.Dataset safe for dh.write_dataset():
      - ensure variable/coord names are strings and netCDF-safe
        (e.g. rename integer keys 0, 1, ... to 'acq_ch0', 'acq_ch1', ...)
      - ensure attrs are JSON / netCDF serializable
    """
    ds2 = ds.copy()

    # Rename data variables with unsafe names (e.g. integer 0 -> "y0")
    rename_map = {}
    for k in list(ds2.data_vars.keys()):
        k_str = str(k)
        if not isinstance(k, str) or not _netcdf_safe_name(k_str):
            rename_map[k] = f"y{k_str}"

    # Rename coords if needed as well
    for k in list(ds2.coords.keys()):
        k_str = str(k)
        if not isinstance(k, str) or not _netcdf_safe_name(k_str):
            base = f"x{k_str}"
            new_name = base
            i = 0
            while new_name in ds2.coords or new_name in ds2.data_vars:
                i += 1
                new_name = f"{base}_{i}"
            rename_map[k] = new_name

    if rename_map:
        ds2 = ds2.rename(rename_map)

    # Sanitize dataset attrs
    ds2.attrs = {str(k): _netcdf_safe_attr_value(v) for k, v in ds2.attrs.items()}

    # Sanitize coord and data_var attrs
    for cname in list(ds2.coords):
        ds2.coords[cname].attrs = {
            str(k): _netcdf_safe_attr_value(v)
            for k, v in ds2.coords[cname].attrs.items()
        }
    for vname in list(ds2.data_vars):
        ds2.data_vars[vname].attrs = {
            str(k): _netcdf_safe_attr_value(v)
            for k, v in ds2.data_vars[vname].attrs.items()
        }

    return ds2

def save_retrieve_acquisition_dataset(ds, name: str) -> tuple[str, Path]:
    """
    Save an xarray.Dataset returned by InstrumentCoordinator.retrieve_acquisition()
    into a Quantify-style experiment folder:

      <datadir>/<YYYY-mm-dd>/<tuid>-<name>/
        dataset.hdf5 (or dataset.nc, depending on quantify-core)
        snapshot.json

    Returns
    -------
    tuid : str
        The unique identifier of the experiment.
    exp_dir : Path
        The path to the created experiment folder.
    """
    tuid = dh.gen_tuid()
    exp_dir = Path(dh.create_exp_folder(tuid=tuid, name=name))

    ds_to_write = ds.copy()
    ds_to_write.attrs["tuid"] = tuid
    ds_to_write.attrs["name"] = name

    # Sanitize for netCDF (handles integer keys etc.)
    ds_to_write = _sanitize_dataset_for_netcdf(ds_to_write)

    dataset_name = getattr(dh, "DATASET_NAME", "dataset.hdf5")
    dh.write_dataset(exp_dir / dataset_name, ds_to_write)

    # Store a snapshot of the current instrument state,
    # similar to MeasurementControl.run()
    with open(exp_dir / "snapshot.json", "w", encoding="utf-8") as f:
        json.dump(dh.snapshot(), f, indent=2, default=str)

    return tuid

# Define linear function
def linear(x, a, b):
    return a * x + b

def DRAG_fit(dataset, motzoi_list, plot=True):
    """Find the optimal DRAG motzoi coefficient from an AllXY-style calibration.

    Two AllXY sequences ([(180,0),(90,90)] and [(180,90),(90,0)]) are swept over
    motzoi; each is equally sensitive to phase errors but with opposite sign, so
    their readout-signal difference is linear in motzoi and crosses zero at the
    motzoi that cancels the phase error. Real and imaginary parts are fit
    independently and their intersection is taken as the optimal motzoi.

    Returns
    -------
    float or None
        The optimal motzoi coefficient, or ``None`` if either linear fit is
        untrustworthy (so callers should skip updating rxy.motzoi).
    """
    g0 = dataset[list(dataset.data_vars)[0]].values[0::2]
    g1 = dataset[list(dataset.data_vars)[0]].values[1::2]
    diff = g0 - g1

    popt1, _ = curve_fit(linear, motzoi_list, diff.real)
    popt2, _ = curve_fit(linear, motzoi_list, diff.imag)

    def r_squared(y, y_fit):
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_real = r_squared(diff.real, linear(motzoi_list, *popt1))
    r2_imag = r_squared(diff.imag, linear(motzoi_list, *popt2))

    # Get intersection point
    a1, b1 = popt1
    a2, b2 = popt2
    optimal_motzoi = (b2 - b1) / (a1 - a2)

    if plot:
        plt.plot(motzoi_list, np.real(diff), label='real', c='C0')
        plt.plot(motzoi_list, np.imag(diff), label='imag', c='C1')
        plt.plot(motzoi_list, linear(motzoi_list, *popt1), label='Fit Real', linestyle='--', c='C0')
        plt.plot(motzoi_list, linear(motzoi_list, *popt2), label='Fit Imag', linestyle='--', c='C1')
        plt.axvline(optimal_motzoi, color='red', linestyle=':', label=f'Optimal Motzoi: {optimal_motzoi:.3f}')
        plt.legend()
        plt.title('DRAG Calibration Fit')
        plt.xlabel('Motzoi Coefficient')
        plt.ylabel('Difference in Readout Signal')
        plt.show()

    fit_ok = np.isfinite(optimal_motzoi) and r2_real > 0.8 and r2_imag > 0.8

    if not fit_ok:
        print(f'DRAG fit failed (R^2_real={r2_real:.3f}, R^2_imag={r2_imag:.3f}, '
              f'motzoi={optimal_motzoi:.4f}). Motzoi not updated.')
        return None

    print(f'Optimal motzoi = {optimal_motzoi:.4f}  '
          f'(R^2_real={r2_real:.3f}, R^2_imag={r2_imag:.3f})')

    return optimal_motzoi

# import scqubits as scq
# import scipy as sp

# def find_EJ_EC_g(
#     E01: float, anharmonicity: float, freq_bare: float, freq_g: float, ng=0, ncut=30
# ) -> [float, float]:
#     """
#     Finds the EJ, EC and g values given a qubit splitting `E01` and `anharmonicity`.

#     Parameters
#     ----------
#         E01:
#             qubit transition energy
#         anharmonicity:
#             absolute qubit anharmonicity, (E2-E1) - (E1-E0)
#         freq_bare:
#             bare resonator frequency
#         freq_g:
#             resonator frequency for dressed g state
#         ng:
#             offset charge (default: 0)
#         ncut:
#             charge number cutoff (default: 30)

#     Returns
#     -------
#         A list of the EJ, EC and g values representing the best fit.
#     """
#     init_EJ, init_EC = scq.Transmon.find_EJ_EC(E01, anharmonicity)
#     qubit = scq.Transmon(init_EJ, init_EC, ng=ng, ncut=ncut, truncated_dim=10)
#     resonator = scq.Oscillator(E_osc=freq_bare, truncated_dim=10)

#     init_g = freq_bare * 0.1 * (init_EJ/(2*init_EC))**(1/4) * np.sqrt(np.pi*50/(25.8e3))
    
#     hilbertspace = scq.HilbertSpace([qubit, resonator])

#     operator1 = qubit.n_operator()
#     operator2 = resonator.creation_operator() + resonator.annihilation_operator()
    
#     hilbertspace.add_interaction(
#         g=init_g,
#         op1=(operator1, qubit),
#         op2=(operator2, resonator)
#     )
#     hilbertspace.generate_lookup()
    
#     start_EJ_EC_g = [init_EJ, init_EC, init_g]

#     def cost_func(EJ_EC_g: [float, float, float]) -> float:
#         EJ, EC, g = EJ_EC_g
#         # update fitting parameters to Hamiltonian(hilbertspace)
#         hilbertspace.subsystem_list[0].EJ = EJ
#         hilbertspace.subsystem_list[0].EC = EC
#         hilbertspace.interaction_list[0].g_strength = g

#         # calculate Energies and desired quantities
#         evals, _ = hilbertspace.eigensys(evals_count=10)
#         computed_E01 = evals[hilbertspace.dressed_index((1,0))] - evals[0]
#         computed_anharmonicity = evals[hilbertspace.dressed_index((2,0))] - evals[0] - 2*computed_E01
#         computed_freq_g = evals[hilbertspace.dressed_index((0,1))] - evals[0]

#         # construct cost
#         cost = 1*(E01 - computed_E01) ** 2
#         cost += 5*(freq_g - computed_freq_g) ** 2
#         cost += 5*(anharmonicity - computed_anharmonicity) ** 2
#         return cost

#     return sp.optimize.minimize(cost_func, start_EJ_EC_g).x

# def quantities_interest(fit_EJ, fit_EC, fit_g, freq_bare, verbose=1, vec=0):
    
#     DIM_QUBIT = 10
#     DIM_RES = 10

#     qubit = scq.Transmon(fit_EJ, fit_EC, ng=0, ncut=30, truncated_dim=DIM_QUBIT)
    
#     E01 = scq.Transmon(fit_EJ,fit_EC,ng=0,ncut=30,truncated_dim=DIM_QUBIT).E01()
#     anharmonicity = scq.Transmon(fit_EJ,fit_EC,ng=0,ncut=30,truncated_dim=DIM_QUBIT).anharmonicity()
    
#     resonator = scq.Oscillator(E_osc=freq_bare,truncated_dim=DIM_RES)

#     dispersion = qubit._compute_dispersion('ng', 'EJ', [fit_EJ])[1][0][0]
#     print(f'dispersion = {dispersion*1e6:.04f} kHz')

#     hilbertspace = scq.HilbertSpace([qubit, resonator])

#     operator1 = qubit.n_operator()
#     operator2 = resonator.creation_operator() + resonator.annihilation_operator()
    
#     hilbertspace.add_interaction(
#         g=fit_g,
#         op1=(operator1, qubit),
#         op2=(operator2, resonator),
#         add_hc=False
#     )
    
#     hilbertspace.generate_lookup()
#     evals, evecs = hilbertspace.eigensys(evals_count=10)
#     evals -= evals[0]

#     state_qubit = 1
#     state_res = 0
#     state_index = hilbertspace.dressed_index((state_qubit,state_res))
#     fit_f01 = evals[state_index]
    
#     state_qubit = 2
#     state_res = 0
#     state_index = hilbertspace.dressed_index((state_qubit,state_res))
#     fit_anharmonicity = evals[state_index]-2*fit_f01
    
#     state_qubit = 0
#     state_res = 1
#     state_index = hilbertspace.dressed_index((state_qubit,state_res))
#     fit_freq_g = evals[state_index]
    
#     state_qubit = 1
#     state_res = 1
#     state_index = hilbertspace.dressed_index((state_qubit,state_res))
#     fit_freq_e = evals[state_index]-fit_f01
    
#     fit_2chi = (fit_freq_e-fit_freq_g)

#     if verbose:
#         print(f'fit_f01 = {fit_f01:.06f} [GHz]')
#         print(f'fit_anharmonicity = {fit_anharmonicity:.06f} [GHz]')
#         print(f'fit_freq_g = {fit_freq_g:.06f} [GHz]')
#         print(f'fit_freq_e = {fit_freq_e:.06f} [GHz]')
#         print(f'fit_2chi = {fit_2chi*1e3:.05f} [MHz]')

#     if vec == 1:
#         return evals, evecs
        
#     return [fit_f01, fit_anharmonicity, fit_freq_g, fit_freq_e, fit_2chi]