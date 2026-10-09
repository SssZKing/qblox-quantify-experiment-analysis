"""Single-qubit calibration fits: Ramsey chevron, conditional Ramsey, Rabi
error amplification, T1/T2/T2* and DRAG."""

import matplotlib.pyplot as plt
import numpy as np
from quantify_core.data.handling import load_dataset, load_snapshot
from scipy.optimize import curve_fit

from qqea.fitting.models import decay_osci_model
from qqea.fitting.readout import rotate_real_imag


def v_line(x, a, b):
    return np.abs(a*x + b)


def ramsey_chevron_fit(tuid, qubit, plot_data = False, plot_fit = False, real_imag = True):
    """Find f01 from a Ramsey chevron (Ramsey delay x0 versus drive detuning x1).

    Each detuning row is fit with a decaying cosine; the fitted Ramsey frequency
    versus detuning is a V, ``|a*detuning + b|``, whose vertex is the detuning
    at which the drive sits on f01. The f01 stored in the dataset snapshot for
    ``qubit`` is the reference. Also prints the mean Ramsey T2* (outliers
    removed by the 1.5 IQR rule).

    Returns
    -------
    float or None
        The fitted f01 in Hz, or ``None`` if the V-line fit is untrustworthy
        (so callers should skip updating f01).
    """
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

    if best is None:
        print('V-line fit did not converge for either slope sign. f01 not updated.')
        return None

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
    """Ramsey frequency shift of a target qubit conditioned on the control state.

    ``con_ram_ds`` (or its tuid) holds Ramsey traces versus delay (x0) for each
    control preparation (x1, normally 0 and 1). Each trace is fit with a
    decaying cosine.

    Returns
    -------
    delta_freq : float
        Ramsey frequency for the first control value minus the second, in Hz
        (``f(|0>) - f(|1>)`` for the usual 0/1 sweep); NaN if the dataset has
        fewer than two control values.
    fit_result_list : list of lmfit.model.ModelResult
        One fit per control value, in ascending order of x1.
    """
    if isinstance(con_ram_ds, str):
        con_ram_ds = load_dataset(con_ram_ds)
        
    condition = np.unique(con_ram_ds.x1)

    fit_result_list = []
    freqs = []
    
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
        freqs.append(fit_result.params['frequency'].value)

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

    if len(freqs) < 2:
        print(f'Conditional Ramsey: only {len(freqs)} control value(s) in the dataset, '
              f'no frequency difference.')
        delta_freq = np.nan
    else:
        delta_freq = freqs[0] - freqs[1]

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
    motzoi_list = np.asarray(motzoi_list, dtype=float)
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
