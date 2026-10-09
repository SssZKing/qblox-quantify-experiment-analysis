"""iSWAP fits: chevron pump frequency, swap angle theta_p and RPE phases."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from quantify_core.data.handling import load_dataset
from scipy.optimize import curve_fit

from qqea.fitting.models import lorentzian_model
from qqea.fitting.units import P2Z


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
