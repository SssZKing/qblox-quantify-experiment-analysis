"""Single-qubit nodes: spectroscopy, frequency, pi pulse, DRAG, AllXY,
coherence and randomized benchmarking."""

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from qcodes import ManualParameter
from quantify_core.analysis.single_qubit_timedomain import AllXYAnalysis, RabiAnalysis
from quantify_core.analysis.spectroscopy_analysis import ResonatorSpectroscopyAnalysis
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

from qqea.experiments.nodes._base import NodeBase, batched_parameter, node
from qqea.experiments.runners import DRAG_calibration_sched
from qqea.fitting.fits import (
    DRAG_fit,
    exp_decay_model,
    rabi_amplification_fit,
    ramsey_chevron_fit,
    t1_and_t2_fit,
)
from qqea.schedules.single_qubit import (
    allxy_sched,
    echo_sched,
    heterodyne_spec_sched_nco,
    multi_qubit_t1_and_t2 as multi_qubit_t1_and_t2_sched,
    rabi_amplification,
    rabi_sched,
    ramsey_sched,
    t1_sched,
)
from qqea.schedules.two_qubit import RBAnalysis, randomized_benchmarking_schedule

# Average number of physical gates per single-qubit Clifford for this
# compiler's decomposition; converts RB error per Clifford to error per gate.
PHYSICAL_GATES_PER_CLIFFORD = 1.875


class SingleQubitNodes(NodeBase):

    @node
    def resonator_calibration(self, qubit: BasicTransmonElement, frequency_setpoints=None,
                              amp=0.2, duration=5e-6, repetitions=100, plot=False, update=True):
        """Calibrate the resonator frequency of a qubit via spectroscopy.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        frequency_setpoints : np.ndarray, optional
            Frequencies to sweep, in Hz. Defaults to 800 points spanning the
            qubit's current readout frequency +/-2 MHz.
        amp : float
            Amplitude of the spectroscopy pulse.
        duration : float
            Duration of the spectroscopy pulse.
        repetitions : int, optional
            Number of repetitions for averaging (default 100).
        plot : bool, optional
            Whether to display the analysis figures (default False).
        update : bool, optional
            Whether to write the fitted frequency back to the qubit on
            success (default True).

        Returns
        -------
        (bool, float or None)
            ``(True, resonator_freq)`` on a successful fit, else
            ``(False, None)``. On success, if ``update`` is True, the qubit's
            readout clock frequency is updated in place.
        """
        if frequency_setpoints is None:
            freq_center = qubit.clock_freqs.readout()
            frequency_setpoints = np.linspace(freq_center - 2e6, freq_center + 2e6, 800)

        freq = batched_parameter("freq", unit="Hz", label="Frequency",
                                 batch_size=frequency_setpoints.size)
        spec_sched_kwargs = dict(
            pulse_amp=amp,
            pulse_duration=duration,
            frequencies=freq,
            acquisition_delay=self.acq_delay,
            integration_time=duration,
            init_duration=10e-6,
            port=qubit.ports.readout(),
            clock=qubit.name + ".ro",
        )
        rs_ds = self._run_sweep(
            heterodyne_spec_sched_nco, spec_sched_kwargs, freq, frequency_setpoints,
            "resonator spectroscopy", repetitions, real_imag=False,
        )

        rs_analysis = ResonatorSpectroscopyAnalysis(dataset=rs_ds, plot_figures=plot)
        rs_analysis.run()
        if plot:
            rs_analysis.display_figs_mpl()

        qoi = rs_analysis.quantities_of_interest
        if not qoi['fit_success']:
            print("Resonator fit failed. Please check the data.")
            return False, None

        fr = qoi["fr"].nominal_value
        print('Resonator Calibrated:')
        print(
            f'fr = {fr / 1e9:.6f} GHz, '
            f'Ql = {qoi["Ql"].nominal_value / 1e3:.0f} k, '
            f'Qc = {qoi["Qc"].nominal_value / 1e3:.0f} k, '
            f'Qi = {qoi["Qi"].nominal_value / 1e3:.0f} k'
        )

        if update:
            qubit.clock_freqs.readout(fr)

        return True, fr

    @node
    def ramsey_chevron_calibration(self, qubit: BasicTransmonElement,
                                   plot_fit=False, plot_data=False,
                                   tau_setpoints=np.arange(0, 20e-6, 0.2e-6),
                                   detuning_setpoints=np.linspace(-300e3, 300e3, 11),
                                   repetitions=100, update=True):
        """Calibrate the qubit f01 frequency via a Ramsey chevron measurement.

        The chevron runs with DRAG off (``rxy.motzoi`` = 0): with DRAG on, Qblox
        rejects the sequence as too long. The previous motzoi is restored once
        the measurement is done, so a chevron after a DRAG calibration does not
        undo it.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        plot_fit : bool, optional
            Whether to plot the V-line fit of Ramsey frequency vs detuning.
        plot_data : bool, optional
            Whether to plot the raw chevron data.
        tau_setpoints : np.ndarray, optional
            Ramsey free-evolution times to sweep (the batched axis).
        detuning_setpoints : np.ndarray, optional
            Artificial detunings to sweep.
        repetitions : int, optional
            Number of repetitions for averaging (default 100).
        update : bool, optional
            Whether to write the fitted f01 back to the qubit on success
            (default True).

        Returns
        -------
        (bool, float or None)
            ``(True, f01)`` on a successful fit, else ``(False, None)``. On
            success, if ``update`` is True, the qubit's f01 clock frequency
            is updated in place.
        """
        tau = batched_parameter("tau", unit="s", label="Time")
        detuning = ManualParameter(name="detuning", unit="Hz", label="Detuning")
        detuning.batched = False

        ramsey_sched_kwargs = {
            "qubit": qubit,
            "times": tau,
            "artificial_detuning": detuning,
        }

        prev_motzoi = qubit.rxy.motzoi()
        qubit.rxy.motzoi(0)
        try:
            ram_che_ds = self._run_sweep(
                ramsey_sched, ramsey_sched_kwargs, [tau, detuning],
                [tau_setpoints, detuning_setpoints], f"{qubit.name} Ramsey Chevron",
                repetitions, grid=True, soft_avg=1,
                real_imag=True, max_batch_size=tau_setpoints.size,
            )
        finally:
            qubit.rxy.motzoi(prev_motzoi)

        f01 = ramsey_chevron_fit(
            ram_che_ds.tuid, qubit,
            plot_data=plot_data, plot_fit=plot_fit, real_imag=True,
        )

        if f01 is None or not np.isfinite(f01):
            print("Ramsey chevron fit failed. Please check the data.")
            return False, None

        if update:
            qubit.clock_freqs.f01(f01)

        return True, f01

    @node
    def rabi_calibration(self, qubit: BasicTransmonElement,
                         rabi_amplitude_setpoints=np.linspace(-1, 1, 101),
                         repetitions=100, plot=False, update=True):
        """Calibrate the pi-pulse amplitude (amp180) of a qubit via a Rabi sweep.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        rabi_amplitude_setpoints : np.ndarray, optional
            Rabi pulse amplitudes to sweep (default ``np.linspace(-1, 1, 101)``).
        repetitions : int, optional
            Number of repetitions for averaging (default 100).
        plot : bool, optional
            Whether to display the analysis figures (default False).
        update : bool, optional
            Whether to write the fitted amp180 back to the qubit on success
            (default True).

        Returns
        -------
        (bool, float or None)
            ``(True, amp180)`` on a successful fit, else ``(False, None)``. On
            success, if ``update`` is True, the qubit's amp180 is updated in
            place.
        """
        rabi_pulse_amp = batched_parameter("rabi_pulse_amplitude", unit="V",
                                           label="Rabi Pulse Amplitude", batch_size=200)
        rabi_sched_kwargs = {
            "pulse_amp": rabi_pulse_amp,
            "pulse_duration": qubit.rxy.duration(),
            "frequency": qubit.clock_freqs.f01(),
            "qubit": qubit,
        }
        rabi_ds = self._run_sweep(
            rabi_sched, rabi_sched_kwargs, rabi_pulse_amp, rabi_amplitude_setpoints,
            f"{qubit.name} Rabi", repetitions, soft_avg=1, real_imag=True,
        )

        rabi_analysis = RabiAnalysis(dataset=rabi_ds, plot_figures=plot)
        rabi_analysis.run()
        if plot:
            rabi_analysis.display_figs_mpl()

        qoi = rabi_analysis.quantities_of_interest
        if not qoi['fit_success']:
            print("Rabi fit failed. Please check the data.")
            return False, None

        amp180 = qoi["Pi-pulse amplitude"].nominal_value
        print('Rabi Calibrated:')
        print(f'amp180 = {amp180:.4f} V')

        if update:
            qubit.rxy.amp180(amp180)

        return True, amp180

    @node
    def rabi_amplification_calibration(self, qubit: BasicTransmonElement,
                                       rabi_amplitude_setpoints=None,
                                       pi_number_setpoints=np.arange(0, 80, 2),
                                       repetitions=50, plot=False, update=True):
        """Fine-calibrate amp180 via a Rabi error-amplification measurement.

        Sweeps the pulse amplitude against an increasing number of pi pulses;
        amplitude errors accumulate with the pulse count, so the correct amp180
        is pinned far more tightly than by a single Rabi. The fit
        (:func:`rabi_amplification_fit`) sums along the pulse axis and fits a
        parabola to the resulting dip.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        rabi_amplitude_setpoints : np.ndarray, optional
            Pulse amplitudes to sweep. Defaults to a +/-0.025 V window of 21
            points centred on the qubit's current ``amp180``.
        pi_number_setpoints : np.ndarray, optional
            Numbers of pi pulses to sweep (default ``np.arange(0, 80, 2)``).
        repetitions : int, optional
            Number of repetitions for averaging (default 50).
        plot : bool, optional
            Whether to display the analysis figures (default False).
        update : bool, optional
            Whether to write the fitted amp180 back to the qubit on success
            (default True).

        Returns
        -------
        (bool, float or None)
            ``(True, amp180)`` on a successful fit, else ``(False, None)``. On
            success, if ``update`` is True, the qubit's amp180 is updated in
            place.
        """
        if rabi_amplitude_setpoints is None:
            amp180 = qubit.rxy.amp180()
            rabi_amplitude_setpoints = np.linspace(amp180 - 0.025, amp180 + 0.025, 21)

        rabi_pulse_amp = batched_parameter("rabi_pulse_amplitude", unit="V",
                                           label="Rabi Pulse Amplitude", batch_size=100)
        pi_number = ManualParameter(name="number_of_pi", label="Number of Pi")

        rabi_sched_kwargs = {
            "pulse_amp": rabi_pulse_amp,
            "pulse_duration": qubit.rxy.duration(),
            "frequency": qubit.clock_freqs.f01(),
            "qubit": qubit,
            "pi_number": pi_number,
        }
        rabi_amp_ds = self._run_sweep(
            rabi_amplification, rabi_sched_kwargs, [rabi_pulse_amp, pi_number],
            [rabi_amplitude_setpoints, pi_number_setpoints],
            f"{qubit.name} Rabi Amplification", repetitions, grid=True, soft_avg=1,
            real_imag=True,
        )

        amp180 = rabi_amplification_fit(rabi_amp_ds, plot=plot, real_imag=True)

        if amp180 is None or not np.isfinite(amp180):
            print("Rabi amplification fit failed. Please check the data.")
            return False, None

        if update:
            qubit.rxy.amp180(amp180)

        return True, amp180

    def _decay_calibration(self, qubit, schedule_function, run_label, quantity,
                           tau_setpoints, repetitions, plot):
        """Run an exponential-decay experiment (T1 or echo) and fit the time constant.

        Shared implementation for :meth:`T1_calibration` and
        :meth:`T2_echo_calibration`; they differ only in the schedule function,
        the run label, and the default delay sweep.

        Returns
        -------
        (bool, float or None)
            ``(True, tau_us)`` if the exponential fit is trustworthy, else
            ``(False, None)``. ``tau_us`` is the decay time constant in
            microseconds. The qubit is not modified.
        """
        tau = batched_parameter("tau_delay", unit="s", label="Delay",
                                batch_size=tau_setpoints.size)
        sched_kwargs = {
            "qubit": qubit,
            "times": tau,
            "acq_protocol": "ThresholdedAcquisition",
        }
        ds = self._run_sweep(
            schedule_function, sched_kwargs, tau, tau_setpoints, run_label, repetitions,
            soft_avg=1, real_imag=False,
        )

        # x0 is the delay in seconds; fit in microseconds so tau comes out in us
        delay_us = ds.x0.values * 1e6
        guess = exp_decay_model.guess(ds.y0.values, delay=delay_us)
        result = exp_decay_model.fit(ds.y0.values, t=delay_us, params=guess)

        if plot:
            result.plot_fit()
            plt.ylim(0, 1)
            plt.grid()
            plt.xlabel(r'$\tau$ [μs]')
            plt.ylabel('Excited State Population')
            plt.title(f'{quantity} Decay Fit')
            plt.show()

        value = result.params['tau'].value
        value_err = result.params['tau'].stderr
        r_squared = result.rsquared

        tau_max_us = tau_setpoints.max() * 1e6
        fit_ok = (
            result.success
            and np.isfinite(value) and value > 0
            and r_squared > 0.9
            and value < 5 * tau_max_us          # tau not absurdly beyond the sweep
        )

        if not fit_ok:
            print(f'{quantity} fit failed (R^2={r_squared:.3f}, {quantity}={value:.1f} us). '
                  f'Not trusted.')
            return False, None

        err_str = f' +/- {value_err:.1f}' if value_err is not None else ''
        print(f'{quantity} = {value:.1f}{err_str} [us]  (R^2={r_squared:.3f})')
        return True, value

    @node
    def T1_calibration(self, qubit: BasicTransmonElement,
                       tau_setpoints=np.hstack((np.arange(0, 40, 0.4),
                                                np.arange(40, 200, 2))) * 1e-6,
                       repetitions=200, plot=False):
        """Measure the qubit T1 relaxation time via an inversion-recovery decay.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to characterise.
        tau_setpoints : np.ndarray, optional
            Delay times to sweep, in seconds.
        repetitions : int, optional
            Number of repetitions for averaging (default 200).
        plot : bool, optional
            Whether to display the decay-fit figure (default False).

        Returns
        -------
        (bool, float or None)
            ``(True, T1_us)`` if the fit is trustworthy, else ``(False, None)``.
        """
        return self._decay_calibration(
            qubit, t1_sched, f"{qubit.name} T1 experiment", "T1",
            tau_setpoints, repetitions, plot,
        )

    @node
    def T2_echo_calibration(self, qubit: BasicTransmonElement,
                            tau_setpoints=np.hstack((np.arange(0, 80, 0.8),
                                                     np.arange(80, 400, 4))) * 1e-6,
                            repetitions=200, plot=False):
        """Measure the qubit T2 echo time via a Hahn-echo decay.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to characterise.
        tau_setpoints : np.ndarray, optional
            Total free-evolution times to sweep, in seconds.
        repetitions : int, optional
            Number of repetitions for averaging (default 200).
        plot : bool, optional
            Whether to display the decay-fit figure (default False).

        Returns
        -------
        (bool, float or None)
            ``(True, T2_echo_us)`` if the fit is trustworthy, else
            ``(False, None)``.
        """
        return self._decay_calibration(
            qubit, echo_sched, f"{qubit.name} T2 echo", "T2 echo",
            tau_setpoints, repetitions, plot,
        )

    @node
    def multi_qubit_t1_and_t2(self, qubits: Iterable[BasicTransmonElement],
                              tau_setpoints=np.hstack((np.arange(0, 40, 2),
                                                       np.arange(40, 120, 4))) * 1e-6,
                              cases=(1, 2, 3), num_repeats=1,
                              artificial_detuning=100e3, repetitions=200,
                              r_squared=0.8, plot=False):
        """Measure T1, T2 echo and T2* (Ramsey) of one or two qubits in one run.

        Runs the ``multi_qubit_t1_and_t2`` schedule with the delay ``tau`` batched and
        the experiment ``case`` stepped as the outer sweep (1 = T1, 2 = Hahn
        echo, 3 = Ramsey with an ``artificial_detuning`` phase ramp). Each qubit
        is mapped to its own acquisition channel (qubits[i] -> channel i), and
        the mapping is left in place. The data is fitted with
        :func:`t1_and_t2_fit`.

        This is a characterisation node: no qubit parameter is modified.
        Deviates from the single-qubit contract because the measurement is
        inherently multi-qubit — see Returns.

        Parameters
        ----------
        qubits : Iterable[BasicTransmonElement]
            One or two qubits, mapped in order to channels 0, 1
            (``t1_and_t2_fit`` reads only y0/y1).
        tau_setpoints : np.ndarray, optional
            Delay times to sweep, in seconds.
        cases : sequence of int, optional
            Which experiments to run, any subset of (1, 2, 3) (default all).
        num_repeats : int, optional
            How many times the ``cases`` block is repeated in the same run
            (``case`` setpoints are ``np.tile(cases, num_repeats)``); each
            fitted list has one entry per repeat (default 1).
        artificial_detuning : float, optional
            Ramsey (case 3) detuning in Hz (default 100 kHz).
        repetitions : int, optional
            Number of repetitions for averaging (default 200).
        r_squared : float, optional
            Minimum R^2 for a fit to be accepted (default 0.8).
        plot : bool, optional
            Whether to plot the repeat-averaged populations per case
            (default False).

        Returns
        -------
        (bool, tuple)
            ``(all_ok, fit_results)``, where ``fit_results`` is the unmodified
            return of :func:`t1_and_t2_fit`:
            ``([q0_t1, q0_t2, q0_t2_ramsey, q0_ramsey_freq],
            [q1_t1, q1_t2, q1_t2_ramsey, q1_ramsey_freq])`` (the Ramsey lists
            are absent when case 3 isn't run). Each entry is a per-repeat list,
            times in us and frequency in MHz, ``NaN`` where the fit fell below
            ``r_squared``. With one qubit the second list's entries are empty.
            ``all_ok`` is True only if every measured quantity of every qubit
            has at least one good fit.
        """
        qubits = list(qubits)
        if not 1 <= len(qubits) <= 2:
            raise ValueError(f"multi_qubit_t1_and_t2 supports 1 or 2 qubits, got {len(qubits)}.")
        self._assign_acq_channels(qubits)

        tau = batched_parameter("tau_delay", unit="s", label="Delay",
                                batch_size=tau_setpoints.size)
        case = ManualParameter(name="case", label="Case")

        sched_kwargs = {
            "qubit_specifier": qubits,
            "artificial_detuning": artificial_detuning,
            "times": tau,
            "case": case,
            "acq_protocol": "ThresholdedAcquisition",
        }
        case_setpoints = np.tile(np.asarray(cases), num_repeats)
        run_label = " ".join(q.name for q in qubits) + " T1 and T2"
        ds = self._run_sweep(
            multi_qubit_t1_and_t2_sched, sched_kwargs, [tau, case],
            [tau_setpoints, case_setpoints], run_label, repetitions,
            grid=True, soft_avg=1, real_imag=True,
        )

        fit_results = t1_and_t2_fit(ds, r_squared=r_squared)
        names = ["T1", "T2_echo", "T2_ramsey", "ramsey_freq"]
        units = {"T1": "us", "T2_echo": "us", "T2_ramsey": "us", "ramsey_freq": "MHz"}
        measured = {1: ["T1"], 2: ["T2_echo"], 3: ["T2_ramsey", "ramsey_freq"]}
        keys = [k for c in sorted(set(cases)) for k in measured[c]]

        # all_ok and the printed summary only; the per-repeat fits are returned as-is
        all_ok = True
        for qubit, per_qubit in zip(qubits, fit_results):
            parts = []
            for name, arr in zip(names, per_qubit):
                if name not in keys:
                    continue
                arr = np.asarray(arr, dtype=float)
                good = arr[np.isfinite(arr)]
                if good.size == 0:
                    all_ok = False
                    parts.append(f"{name} = fit failed")
                else:
                    parts.append(f"{name} = {good.mean():.2f} {units[name]} "
                                 f"({good.size}/{arr.size} fits)")
            print(f"{qubit.name}: " + ", ".join(parts))

        if plot:
            tau_us = np.unique(ds.x0) * 1e6
            titles = {1: "T1", 2: "T2 echo", 3: "Ramsey"}
            fig, axes = plt.subplots(len(qubits), 1, figsize=(7, 3 * len(qubits)),
                                     squeeze=False, sharex=True)
            for i, qubit in enumerate(qubits):
                ax = axes[i, 0]
                for c in sorted(set(cases)):
                    y = ds.where(ds.x1 == c, drop=True)[f"y{i}"].values
                    ax.plot(tau_us, y.reshape(-1, tau_us.size).mean(axis=0),
                            'o-', ms=3, label=titles[c])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Excited State Population')
                ax.set_title(qubit.name)
                ax.grid()
                ax.legend()
            axes[-1, 0].set_xlabel(r'$\tau$ [μs]')
            plt.tight_layout()
            plt.show()

        return all_ok, fit_results

    @node
    def drag_calibration(self, qubit: BasicTransmonElement,
                         motzoi_list=np.linspace(-0.1, 0.1, 41),
                         repetitions=1000, plot=False, update=True):
        """Calibrate the DRAG motzoi coefficient of a qubit.

        Unlike the sweep nodes, this measurement bypasses meas_ctrl/ScheduleGettable
        and drives the instrument coordinator directly
        (:func:`qqea.experiments.runners.DRAG_calibration_sched`), since it needs
        the raw complex acquisition. The raw data is saved to the data
        directory with ``motzoi_list`` stored as an attribute. Requires
        ``instrument_coordinator`` to have been passed to the constructor.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        motzoi_list : np.ndarray, optional
            Motzoi coefficients to sweep (default ``np.linspace(-0.1, 0.1, 41)``).
        repetitions : int, optional
            Number of repetitions for averaging (default 1000).
        plot : bool, optional
            Whether to display the DRAG fit figure (default False).
        update : bool, optional
            Whether to write the fitted motzoi back to the qubit on success
            (default True).

        Returns
        -------
        (bool, float or None)
            ``(True, motzoi)`` if the fit is trustworthy, else ``(False, None)``.
            On success, if ``update`` is True, the qubit's rxy.motzoi is
            updated in place.
        """
        self._require_instrument_coordinator("drag_calibration")

        drag_ds = DRAG_calibration_sched(
            qubit=qubit,
            motzoi_list=motzoi_list,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )
        self._save_raw(drag_ds, f"{qubit.name} DRAG calibration",
                       motzoi_list=np.asarray(motzoi_list), repetitions=repetitions)

        motzoi = DRAG_fit(drag_ds, motzoi_list, plot=plot)

        if motzoi is None or not np.isfinite(motzoi):
            print("DRAG fit failed. Please check the data.")
            return False, None

        if update:
            qubit.rxy.motzoi(motzoi)

        return True, motzoi

    @node
    def allxy_calibration(self, qubit: BasicTransmonElement,
                          repetitions=1000, deviation_threshold=0.05, plot=False):
        """Run an AllXY diagnostic sequence to check single-qubit gate calibration.

        Unlike the other nodes, this is a diagnostic check, not a fit — there
        is no physical parameter to write back to the qubit, and no R² gate.
        ``AllXYAnalysis`` compares the 21-element AllXY sequence against its
        ideal curve and reports the mean absolute deviation; a low deviation
        indicates amp180/motzoi/frequency are well calibrated, while a
        systematic deviation points to a specific gate error (see the AllXY
        figure for which elements deviate).

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to check.
        repetitions : int, optional
            Number of repetitions for averaging (default 1000).
        deviation_threshold : float, optional
            Maximum acceptable mean absolute deviation from the ideal AllXY
            curve (default 0.05).
        plot : bool, optional
            Whether to display the AllXY figure (default False).

        Returns
        -------
        (bool, float)
            ``(True, deviation)`` if the deviation is below threshold, else
            ``(False, deviation)``. ``deviation`` is a directly measured
            quantity (not a fit result), so it's returned either way rather
            than ``None`` on failure — nothing is ever written back to the
            qubit, so there's no risk of acting on an untrustworthy value.
        """
        elements = batched_parameter("elements", batch_size=21)
        allxy_sched_kwargs = {"qubit": qubit, "element_select_idx": elements}

        with self._quiet():
            ds = self._run_sweep(
                allxy_sched, allxy_sched_kwargs, elements, np.arange(21),
                f"{qubit.name} ALLXY", repetitions, soft_avg=1, real_imag=True,
            )

        allxy_analysis = AllXYAnalysis(dataset=ds, plot_figures=plot)
        allxy_analysis.run()
        if plot:
            allxy_analysis.display_figs_mpl()

        deviation = allxy_analysis.quantities_of_interest["deviation"]

        if deviation > deviation_threshold:
            print(f'AllXY deviation {deviation:.4f} exceeds threshold {deviation_threshold}.')
            return False, deviation

        print(f'AllXY Calibrated: deviation = {deviation:.4f}')
        return True, deviation

    @node
    def randomized_benchmarking(self, qubit: BasicTransmonElement,
                                length_setpoints=np.arange(0, 341, 20),
                                seed_setpoints=None, num_seeds=20,
                                repetitions=300, variance_threshold=0.5,
                                plot=False):
        """Characterize single-qubit gate fidelity via randomized benchmarking.

        Unlike the other nodes, this is a diagnostic characterization, not a
        calibration — there is no physical parameter to write back to the
        qubit. ``RBAnalysis`` fits an exponential decay of sequence fidelity
        vs Clifford length to extract the depolarizing parameter ``alpha`` and
        the error per Clifford ``r``, and separately reports whether the
        seed-to-seed variance is consistent with finite-sample noise
        (``sample_size_variance``: a low value indicates coherent errors
        beyond simple depolarization). ``RBAnalysis`` doesn't expose an
        R²/fit-success metric for the underlying decay fit, so the gate here
        is physical instead: ``alpha`` must be a valid depolarizing parameter
        in (0, 1), and ``sample_size_variance`` must exceed
        ``variance_threshold``.

        Because randomized benchmarking's statistical power comes from
        averaging over independently random Clifford sequences,
        ``seed_setpoints`` defaults to ``None`` and is regenerated fresh
        (``num_seeds`` random seeds) on every call, rather than reusing a
        fixed default array — pass an explicit array only if you specifically
        want reproducible sequences.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to characterize.
        length_setpoints : np.ndarray, optional
            Numbers of Cliffords per sequence to sweep (default
            ``np.arange(0, 341, 20)``).
        seed_setpoints : np.ndarray, optional
            Explicit random seeds for the Clifford sequences. Defaults to
            ``None``, which draws ``num_seeds`` fresh random seeds each call.
        num_seeds : int, optional
            Number of random seeds to draw when ``seed_setpoints`` is
            ``None`` (default 20).
        repetitions : int, optional
            Number of repetitions for averaging (default 300).
        variance_threshold : float, optional
            Minimum acceptable fraction of seed-to-seed variance explained by
            finite sample size (default 0.5); below this, coherent errors are
            likely present.
        plot : bool, optional
            Whether to display the RB decay figure (default False).

        Returns
        -------
        (bool, float or None)
            ``(True, r_gate)`` (error per physical gate) if ``alpha`` is
            physical and the variance check passes, else ``(False, None)``.
            Nothing is written back to the qubit either way.

            ``r_gate`` divides the directly-measured error per Clifford by
            ``PHYSICAL_GATES_PER_CLIFFORD`` (1.875), the assumed average
            number of physical gates per Clifford for this compiler's
            decomposition — if that decomposition ever changes, this
            conversion factor needs to change with it.
        """
        if seed_setpoints is None:
            seed_setpoints = np.random.randint(0, 2**31 - 1, size=num_seeds, dtype=np.int32)

        length = batched_parameter("length", unit="#", label="Number of Clifford gates")
        seed = batched_parameter("seed", unit="", label="Seeds")

        rb_sched_kwargs = {
            "qubit_specifier": qubit,
            "lengths": length,
            "seeds": seed,
        }
        with self._quiet():
            ds = self._run_sweep(
                randomized_benchmarking_schedule, rb_sched_kwargs, [length, seed],
                [length_setpoints, seed_setpoints], f"{qubit.name} Randomized Benchmarking",
                repetitions, grid=True,
                real_imag=False, num_channels=1, max_batch_size=length_setpoints.size,
            )

        rb_result = RBAnalysis(
            dataset=ds,
            label="Randomized benchmarking",
            settings_overwrite={"mpl_transparent_background": False},
            repetitions=repetitions,
            plot_figures=plot,
        ).run()
        if plot:
            rb_result.display_figs_mpl()

        alpha = rb_result.alpha
        r = rb_result.r
        alpha_ok = np.isfinite(alpha) and 0 < alpha < 1
        variance_ok = rb_result.sample_size_variance > variance_threshold

        if not (alpha_ok and variance_ok):
            print(f'RB failed (alpha={alpha}, sample_size_variance='
                  f'{rb_result.sample_size_variance:.2f}). Not trusted.')
            return False, None

        r_gate = r / PHYSICAL_GATES_PER_CLIFFORD
        gate_fidelity = 1 - r_gate
        print(f'RB Characterized: alpha={alpha:.4f}, r_g={r_gate:.3e}, '
              f'F_g={gate_fidelity * 100:.3f}%, '
              f'sample_size_variance={rb_result.sample_size_variance:.2f}')

        return True, r_gate
