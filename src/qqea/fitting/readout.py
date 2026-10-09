"""Readout analysis: g/e discrimination, readout frequency/amplitude/duration
optimization, integration weights and IQ rotation."""

import matplotlib.pyplot as plt
import numpy as np
from quantify_core.analysis.readout_calibration_analysis import ReadoutCalibrationAnalysis
from quantify_core.data.handling import load_dataset
from scipy.optimize import minimize
from scipy.signal import savgol_filter


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

def _split_ge_shots(dataset):
    """Ground/excited shots of a raw ``retrieve_acquisition()`` sweep.

    The first data variable holds complex shots, shape (shots, 2 * n_setpoints),
    with ground and excited interleaved column-wise per setpoint.
    """
    data = dataset[list(dataset.data_vars)[0]].values
    return data[:, 0::2], data[:, 1::2]


def _discriminate_columns(g_append, e_append):
    """``two_state_discriminator`` on every setpoint column, without printing or plotting."""
    return [
        two_state_discriminator(
            g_append[:, i].real, g_append[:, i].imag,
            e_append[:, i].real, e_append[:, i].imag,
            b_print=False, b_plot=False
        )
        for i in range(g_append.shape[1])
    ]


def _smoothed_peak(trace, setpoints):
    """Savitzky-Golay smoothed peak of ``trace`` and whether it clears the noise.

    Returns ``None`` when there are too few points to smooth, otherwise
    ``(peak_idx, smoothed, residual_std, peak_prominence, fit_ok)``. The peak
    counts as real when it rises above the median of the smoothed trace by more
    than the residual noise the filter removed.
    """
    # Savitzky-Golay window must be odd and no larger than the number of points
    window_length = min(7, setpoints.size if setpoints.size % 2 else setpoints.size - 1)
    polyorder = min(3, window_length - 1)

    if window_length < 3:
        return None

    smoothed = savgol_filter(trace, window_length=window_length, polyorder=polyorder)

    peak_idx = np.argmax(smoothed)

    # goodness check: the smoothed peak should clearly rise above the noise
    # the filter smoothed away (the residual between raw data and the trend)
    residual_std = np.std(trace - smoothed)
    peak_prominence = smoothed[peak_idx] - np.median(smoothed)
    fit_ok = residual_std > 0 and peak_prominence > 1 * residual_std
    return peak_idx, smoothed, residual_std, peak_prominence, fit_ok


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
    g_append, e_append = _split_ge_shots(dataset)

    SNR_list = np.array([
        IQ_blob_SNR(g_append[:, i], e_append[:, i])
        for i in range(g_append.shape[1])
    ])
    fid_list = np.array([d[2] for d in _discriminate_columns(g_append, e_append)])
    freq_list = np.asarray(freq_list)

    peak = _smoothed_peak(SNR_list, freq_list)
    if peak is None:
        print('RO frequency fit failed: not enough points to smooth. Not updated.')
        return None
    peak_idx, smoothed, residual_std, peak_prominence, fit_ok = peak
    optimal_freq = freq_list[peak_idx]

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


def _ro_fidelity_sweep(dataset, setpoints, plot, what, short, scale, fmt, unit, xlabel, title):
    """Shared body of ``RO_amp_fit`` and ``RO_len_fit``.

    Runs ``two_state_discriminator`` per setpoint, smooths average fidelity
    versus setpoint and returns the setpoint at the smoothed peak, or ``None``.
    ``what``/``short`` name the setpoint in messages; ``scale``, ``fmt`` and
    ``unit`` only change how it is plotted and printed.
    """
    g_append, e_append = _split_ge_shots(dataset)
    discrim = _discriminate_columns(g_append, e_append)
    avg_fid_list = np.array([d[2] for d in discrim])
    ground_fid_list = np.array([d[3] * 100 for d in discrim])
    setpoints = np.asarray(setpoints)

    peak = _smoothed_peak(avg_fid_list, setpoints)
    if peak is None:
        print(f'RO {what} fit failed: not enough points to smooth. Not updated.')
        return None
    peak_idx, smoothed, residual_std, peak_prominence, fit_ok = peak
    optimal = setpoints[peak_idx]

    if plot:
        x = setpoints * scale
        fig, ax = plt.subplots()
        ax.plot(x, avg_fid_list, 'o', c='C0', label='Avg fidelity (data)')
        ax.plot(x, smoothed, '-', c='C0', label='Avg fidelity (smoothed)')
        ax.plot(x, ground_fid_list, 's', c='C1', label='Ground fidelity')
        ax.axvline(optimal * scale, color='red', ls=':',
                   label=f'Optimal {short}: {optimal * scale:{fmt}}{unit}')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Fidelity [%]')
        ax.legend(loc='best')
        ax.set_title(title)
        plt.show()

    if not fit_ok:
        print(f'RO {what} fit failed (peak prominence={peak_prominence:.2f}, '
              f'noise={residual_std:.2f}). Not updated.')
        return None

    print(f'Optimal RO {what} = {optimal * scale:{fmt}}{unit}  '
          f'(avg fid={smoothed[peak_idx]:.2f}%, ground fid={ground_fid_list[peak_idx]:.2f}%, '
          f'peak prominence={peak_prominence:.2f}, noise={residual_std:.2f})')
    return optimal


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
    return _ro_fidelity_sweep(dataset, amp_list, plot, what='amplitude', short='amp',
                              scale=1, fmt='.4f', unit='',
                              xlabel='Readout Pulse Amplitude',
                              title='Readout Amplitude Optimization')


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
    return _ro_fidelity_sweep(dataset, duration_list, plot, what='duration', short='duration',
                              scale=1e6, fmt='.3f', unit=' us',
                              xlabel='Readout Duration [us]',
                              title='Readout Duration Optimization')

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
    """Plot single-shot g/e readout data rotated onto the I axis, with the threshold.

    ``IQ_ds`` is a readout-calibration dataset (x0 = prepared state 0/1, y0/y1 = I/Q)
    or its tuid. The rotation and threshold come from quantify's
    ``ReadoutCalibrationAnalysis``.
    """
    if isinstance(IQ_ds, str):
        IQ_ds = load_dataset(IQ_ds)
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
    ax[1].text(0.7, 0.9, rf"{threshold*1e6:.02f} $\mu V$", text_props)
    ax[1].set_xlabel("I [V]")
    ax[1].set_title("1D Histogram")

    ax[0].set_aspect('equal', 'box')
    fig.tight_layout()


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


def readout_fidelities(ds):
    """(F_g, F_e) per channel of a Multiplexed Readout Calibration dataset, from the raw IQ.

    Each channel's shots are projected onto the line through the |0> and |1> means and
    the threshold that maximises F_g + F_e is used (the quantify analysis folder of this
    run holds only the last channel's result, since the second analysis overwrites it).
    """
    states = ds.x0.values
    out = []
    for i in range(len(ds.data_vars) // 2):
        z = ds["y%d" % (2 * i)].values + 1j * ds["y%d" % (2 * i + 1)].values
        m0, m1 = z[states == 0].mean(), z[states == 1].mean()
        proj = np.real((z - m0) * np.conj(m1 - m0)) / abs(m1 - m0)
        p0, p1 = np.sort(proj[states == 0]), np.sort(proj[states == 1])
        cut = np.sort(proj)
        f_g = np.searchsorted(p0, cut, side="left") / p0.size
        f_e = 1 - np.searchsorted(p1, cut, side="left") / p1.size
        best = np.argmax(f_g + f_e)
        out.append((f_g[best], f_e[best]))
    return out
