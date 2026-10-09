"""Calibration nodes extracted from the SNL315 notebook.

Each measurement/calibration routine ("node") is a method of
:class:`CalibrationNodes`. The shared runtime instruments
(``quantum_device``, ``meas_ctrl``, ``instrument_coordinator``) and constants
(``ACQ_DELAY``) are held on the instance, so the notebook only needs to
construct the object once::

    from calibration_nodes import CalibrationNodes
    nodes = CalibrationNodes(quantum_device, meas_ctrl, ACQ_DELAY, instrument_coordinator=ic)
    ok, fr = nodes.resonator_calibration(qubit1, amp=..., duration=...,
                                         freq_center=..., freq_span=...)

``instrument_coordinator`` is only needed by :meth:`CalibrationNodes.drag_calibration`,
which drives the instrument coordinator directly instead of going through
``meas_ctrl`` — it can be omitted if you never call that node.

The ``qubit`` object stays a per-call argument since it changes between nodes.
This class is also the natural container for a roadmap/runner that loops over
several nodes.
"""

from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from qcodes import ManualParameter
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_core.analysis.spectroscopy_analysis import ResonatorSpectroscopyAnalysis
from quantify_core.analysis.single_qubit_timedomain import RabiAnalysis, AllXYAnalysis
from quantify_core.analysis.readout_calibration_analysis import ReadoutCalibrationAnalysis
from pycqed_randomized_benchmarking.utils import RBAnalysis, randomized_benchmarking_schedule
from pycqed_randomized_benchmarking.utils import iswap_RPE_f, iswap_RPE_theta_sum

from TWPA_schedule import *
from helpers import *

class CalibrationNodes:
    """Container for calibration nodes sharing the same runtime instruments."""

    def __init__(self, quantum_device, meas_ctrl, acq_delay, instrument_coordinator=None):
        self.quantum_device = quantum_device
        self.meas_ctrl = meas_ctrl
        self.acq_delay = acq_delay
        self.instrument_coordinator = instrument_coordinator

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

        freq = ManualParameter(name="freq", unit="Hz", label="Frequency")
        freq.batched = True
        freq.batch_size = frequency_setpoints.size

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
        gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=heterodyne_spec_sched_nco_TWPA,
            schedule_kwargs=spec_sched_kwargs,
            real_imag=False,
            batched=True,
        )

        self.meas_ctrl.gettables(gettable)

        self.quantum_device.cfg_sched_repetitions(repetitions)

        self.meas_ctrl.settables(freq)
        self.meas_ctrl.setpoints(frequency_setpoints)

        rs_ds = self.meas_ctrl.run("resonator spectroscopy")
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

    def ramsey_chevron_calibration(self, qubit: BasicTransmonElement,
                                   plot_fit=False, plot_data=False,
                                   tau_setpoints=np.arange(0, 20e-6, 0.2e-6),
                                   detuning_setpoints=np.linspace(-300e3, 300e3, 11),
                                   repetitions=100, update=True):
        """Calibrate the qubit f01 frequency via a Ramsey chevron measurement.

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
        qubit.rxy.motzoi(0)

        tau = ManualParameter(name="tau", unit="s", label="Time")
        tau.batched = True

        detuning = ManualParameter(name="detuning", unit="Hz", label="Detuning")
        detuning.batched = False

        ramsey_sched_kwargs = {
            "qubit": qubit,
            "times": tau,
            "artificial_detuning": detuning,
        }

        gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=ramsey_sched_TWPA,
            schedule_kwargs=ramsey_sched_kwargs,
            real_imag=True,
            batched=True,
            max_batch_size=tau_setpoints.size,
        )
        self.meas_ctrl.gettables(gettable)

        self.meas_ctrl.settables([tau, detuning])
        self.meas_ctrl.setpoints_grid([tau_setpoints, detuning_setpoints])

        self.quantum_device.cfg_sched_repetitions(repetitions)
        ram_che_ds = self.meas_ctrl.run(f"{qubit.name} Ramsey Chevron", soft_avg=1)

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
        rabi_pulse_amp = ManualParameter(name="rabi_pulse_amplitude", unit="V",
                                         label="Rabi Pulse Amplitude")
        rabi_pulse_amp.batched = True
        rabi_pulse_amp.batch_size = 200

        rabi_sched_kwargs = {
            "pulse_amp": rabi_pulse_amp,
            "pulse_duration": qubit.rxy.duration(),
            "frequency": qubit.clock_freqs.f01(),
            "qubit": qubit,
        }

        rabi_gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=rabi_sched_TWPA,
            schedule_kwargs=rabi_sched_kwargs,
            real_imag=True,
            batched=True,
        )

        self.meas_ctrl.gettables(rabi_gettable)
        self.quantum_device.cfg_sched_repetitions(repetitions)

        self.meas_ctrl.settables(rabi_pulse_amp)
        self.meas_ctrl.setpoints(rabi_amplitude_setpoints)

        rabi_ds = self.meas_ctrl.run(f"{qubit.name} Rabi", soft_avg=1)
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

        rabi_pulse_amp = ManualParameter(name="rabi_pulse_amplitude", unit="V",
                                         label="Rabi Pulse Amplitude")
        rabi_pulse_amp.batched = True
        rabi_pulse_amp.batch_size = 100

        pi_number = ManualParameter(name="number_of_pi", label="Number of Pi")

        rabi_sched_kwargs = {
            "pulse_amp": rabi_pulse_amp,
            "pulse_duration": qubit.rxy.duration(),
            "frequency": qubit.clock_freqs.f01(),
            "qubit": qubit,
            "pi_number": pi_number,
        }

        rabi_gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=rabi_amplification_TWPA,
            schedule_kwargs=rabi_sched_kwargs,
            real_imag=True,
            batched=True,
        )

        self.meas_ctrl.gettables(rabi_gettable)
        self.quantum_device.cfg_sched_repetitions(repetitions)

        self.meas_ctrl.settables([rabi_pulse_amp, pi_number])
        self.meas_ctrl.setpoints_grid([rabi_amplitude_setpoints, pi_number_setpoints])

        rabi_amp_ds = self.meas_ctrl.run(f"{qubit.name} Rabi Amplification", soft_avg=1)

        RABI_AMP = rabi_amplification_fit(rabi_amp_ds, plot=plot, real_imag=True)

        if RABI_AMP is None or not np.isfinite(RABI_AMP):
            print("Rabi amplification fit failed. Please check the data.")
            return False, None

        if update:
            qubit.rxy.amp180(RABI_AMP)

        return True, RABI_AMP

    def readout_calibration(self, qubit: BasicTransmonElement, num_points=6000,
                            fidelity_threshold=0.9, plot=False, update=True):
        """Calibrate the readout threshold and rotation via an IQ-blob measurement.

        Prepares the qubit in |0> and |1> ``num_points`` times each, fits the two
        IQ blobs, and stores the optimal acquisition threshold and rotation on the
        qubit. Considered successful when the average of the two single-shot
        assignment fidelities exceeds ``fidelity_threshold``.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        num_points : int, optional
            Number of single shots per prepared state (default 6000).
        fidelity_threshold : float, optional
            Minimum average assignment fidelity to accept (default 0.9).
        plot : bool, optional
            Whether to display the analysis figures (default False).
        update : bool, optional
            Whether to write the fitted threshold/rotation back to the qubit
            on success (default True).

        Returns
        -------
        (bool, float or None)
            ``(True, avg_fidelity)`` if the fidelity gate passes, else
            ``(False, None)``. On success, if ``update`` is True, the qubit's
            acq_threshold and acq_rotation are updated in place.
        """
        states = ManualParameter(name="states")
        states.batched = True
        states.batch_size = 600

        RO_cali_sched_kwargs = {
            "qubit": qubit,
            "prepared_states": states,
            "acq_protocol": "SSBIntegrationComplex",
        }  # SSBIntegrationComplex or NumericalSeparatedWeightedIntegration

        gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=readout_calibration_sched_TWPA,
            schedule_kwargs=RO_cali_sched_kwargs,
            real_imag=True,
            batched=True,
        )
        self.meas_ctrl.gettables(gettable)

        states_setpoints = np.hstack((np.zeros(num_points), np.ones(num_points)))
        self.meas_ctrl.settables(states)
        self.meas_ctrl.setpoints(states_setpoints)
        self.quantum_device.cfg_sched_repetitions(1)

        ro_ds = self.meas_ctrl.run(f"{qubit.name} Readout Calibration")

        ro_analysis = ReadoutCalibrationAnalysis(
            dataset=ro_ds.where(ro_ds.x0 != 2, drop=True), plot_figures=plot
        )
        ro_analysis.run()
        if plot:
            ro_analysis.display_figs_mpl()

        qoi = ro_analysis.quantities_of_interest
        avg_fid = (qoi['fid_est_0'] + qoi['fid_est_1']) / 2

        if avg_fid <= fidelity_threshold:
            print(f"Readout calibration failed: avg fidelity {avg_fid:.4f} "
                  f"below {fidelity_threshold}. Threshold/rotation not updated.")
            return False, None

        threshold = qoi['acq_threshold'].nominal_value
        rotation = np.rad2deg(qoi['acq_rotation_rad'].nominal_value) % 360
        if update:
            qubit.measure.acq_threshold(threshold)
            qubit.measure.acq_rotation(rotation)

        print('Readout Calibrated:')
        print(f'avg fidelity = {avg_fid:.4f}, threshold = {threshold * 1e3:.2f} mV, '
              f'rotation = {rotation:.1f} deg')

        return True, avg_fid

    def multiplexed_readout_calibration(self, qubits: Iterable[BasicTransmonElement], num_points=6000,
                                        fidelity_threshold=0.9, plot=False, update=True):
        """Calibrate readout threshold/rotation for several qubits read out at once.

        Runs a single multiplexed IQ-blob measurement, mapping each qubit to its
        own acquisition channel (qubits[i] -> channel i), then analyses each
        qubit's blobs separately. A qubit's acq_threshold and acq_rotation are
        updated only if its own average assignment fidelity exceeds
        ``fidelity_threshold``.

        Parameters
        ----------
        qubits : Iterable[BasicTransmonElement]
            The qubits to calibrate, mapped in order to channels 0, 1, 2, ...
        num_points : int, optional
            Number of single shots per prepared state (default 6000).
        fidelity_threshold : float, optional
            Minimum average assignment fidelity to accept per qubit (default 0.9).
        plot : bool, optional
            Whether to display the analysis figures (default False).
        update : bool, optional
            Whether to write the fitted threshold/rotation back to each
            passing qubit (default True).

        Returns
        -------
        (bool, dict)
            ``(all_ok, {name: fidelity, ...})``. ``all_ok`` is True only if every
            qubit passes the fidelity gate; the dict always reports each qubit's
            fidelity. Only qubits that pass have their readout params updated,
            and only if ``update`` is True.
        """
        qubits = list(qubits)
        for i, qubit in enumerate(qubits):
            qubit.measure.acq_channel(i)

        states = ManualParameter(name="states")
        states.batched = True
        states.batch_size = 600

        RO_cali_sched_kwargs = {
            "qubits": qubits,
            "prepared_states": states,
            "acq_protocol": "SSBIntegrationComplex",
        }  # SSBIntegrationComplex or NumericalSeparatedWeightedIntegration

        gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=multiplexed_readout_calibration_sched_TWPA,
            schedule_kwargs=RO_cali_sched_kwargs,
            real_imag=True,
            batched=True,
            num_channels=len(qubits),
        )
        self.meas_ctrl.gettables(gettable)

        states_setpoints = np.hstack((np.zeros(num_points), np.ones(num_points)))
        self.meas_ctrl.settables(states)
        self.meas_ctrl.setpoints(states_setpoints)

        self.quantum_device.cfg_sched_repetitions(1)
        multi_ro_ds = self.meas_ctrl.run("Multiplexed Readout Calibration")

        fidelities = {}
        all_ok = True
        for i, qubit in enumerate(qubits):
            # channel i lives in y{2i}/y{2i+1}; remap it onto y0/y1 (swapping the
            # current occupants aside) so the single-qubit analysis can read it.
            if i == 0:
                ds = multi_ro_ds
            else:
                ds = multi_ro_ds.rename({
                    'y0': f'y{2 * i}_swap', 'y1': f'y{2 * i + 1}_swap',
                    f'y{2 * i}': 'y0', f'y{2 * i + 1}': 'y1',
                })

            ro_analysis = ReadoutCalibrationAnalysis(dataset=ds, plot_figures=plot)
            ro_analysis.run()
            if plot:
                ro_analysis.display_figs_mpl()

            qoi = ro_analysis.quantities_of_interest
            avg_fid = (qoi['fid_est_0'] + qoi['fid_est_1']) / 2
            fidelities[qubit.name] = avg_fid

            if avg_fid > fidelity_threshold:
                threshold = qoi['acq_threshold'].nominal_value
                rotation = np.rad2deg(qoi['acq_rotation_rad'].nominal_value) % 360
                if update:
                    qubit.measure.acq_threshold(threshold)
                    qubit.measure.acq_rotation(rotation)
                print(f'{qubit.name} readout calibrated: fidelity = {avg_fid:.4f}, '
                      f'threshold = {threshold * 1e3:.2f} mV, rotation = {rotation:.1f} deg')
            else:
                all_ok = False
                print(f'{qubit.name} readout calibration failed: fidelity {avg_fid:.4f} '
                      f'below {fidelity_threshold}. Threshold/rotation not updated.')

        return all_ok, fidelities

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
        tau = ManualParameter(name="tau_delay", unit="s", label="Delay")
        tau.batched = True
        tau.batch_size = tau_setpoints.size

        sched_kwargs = {
            "qubit": qubit,
            "times": tau,
            "acq_protocol": "ThresholdedAcquisition",
        }

        gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=schedule_function,
            schedule_kwargs=sched_kwargs,
            real_imag=False,
            batched=True,
        )
        self.meas_ctrl.gettables(gettable)
        self.meas_ctrl.settables(tau)
        self.meas_ctrl.setpoints(tau_setpoints)

        self.quantum_device.cfg_sched_repetitions(repetitions)
        ds = self.meas_ctrl.run(run_label, soft_avg=1)

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
            qubit, t1_sched_TWPA, f"{qubit.name} T1 experiment", "T1",
            tau_setpoints, repetitions, plot,
        )

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
            qubit, echo_sched_TWPA, f"{qubit.name} T2 echo", "T2 echo",
            tau_setpoints, repetitions, plot,
        )

    def multi_qubit_t1_and_t2(self, qubits: Iterable[BasicTransmonElement],
                              tau_setpoints=np.hstack((np.arange(0, 40, 2),
                                                       np.arange(40, 120, 4))) * 1e-6,
                              cases=(1, 2, 3), num_repeats=1,
                              artificial_detuning=100e3, repetitions=200,
                              r_squared=0.8, plot=False):
        """Measure T1, T2 echo and T2* (Ramsey) of one or two qubits in one run.

        Runs ``multi_qubit_t1_and_t2_TWPA`` with the delay ``tau`` batched and
        the experiment ``case`` stepped as the outer sweep (1 = T1, 2 = Hahn
        echo, 3 = Ramsey with an ``artificial_detuning`` phase ramp). Each qubit
        is mapped to its own acquisition channel (qubits[i] -> channel i). The
        data is fitted with :func:`helpers.t1_and_t2_fit`.

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
            return of :func:`helpers.t1_and_t2_fit`:
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
        for i, qubit in enumerate(qubits):
            qubit.measure.acq_channel(i)

        tau = ManualParameter(name="tau_delay", unit="s", label="Delay")
        tau.batched = True
        tau.batch_size = tau_setpoints.size

        case = ManualParameter(name="case", label="Case")

        sched_kwargs = {
            "qubit_specifier": qubits,
            "artificial_detuning": artificial_detuning,
            "times": tau,
            "case": case,
            "acq_protocol": "ThresholdedAcquisition",
        }

        gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=multi_qubit_t1_and_t2_TWPA,
            schedule_kwargs=sched_kwargs,
            real_imag=True,
            batched=True,
        )
        self.meas_ctrl.gettables(gettable)

        case_setpoints = np.tile(np.asarray(cases), num_repeats)
        self.meas_ctrl.settables([tau, case])
        self.meas_ctrl.setpoints_grid([tau_setpoints, case_setpoints])

        self.quantum_device.cfg_sched_repetitions(repetitions)
        run_label = " ".join(q.name for q in qubits) + " T1 and T2"
        ds = self.meas_ctrl.run(run_label, soft_avg=1)

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

    def drag_calibration(self, qubit: BasicTransmonElement,
                         motzoi_list=np.linspace(-0.1, 0.1, 41),
                         repetitions=1000, plot=False, update=True):
        """Calibrate the DRAG motzoi coefficient of a qubit.

        Unlike the other nodes, this measurement bypasses meas_ctrl/ScheduleGettable
        and drives the instrument coordinator directly (as DRAG_calibration_sched
        does), since it needs the raw complex acquisition rather than a processed
        quantify dataset. Requires ``instrument_coordinator`` to have been passed
        to the constructor.

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
        if self.instrument_coordinator is None:
            raise RuntimeError(
                "drag_calibration requires instrument_coordinator to be set on "
                "CalibrationNodes — pass instrument_coordinator=ic to the constructor."
            )

        drag_ds = DRAG_calibration_sched(
            qubit=qubit,
            motzoi_list=motzoi_list,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )

        motzoi = DRAG_fit(drag_ds, motzoi_list, plot=plot)

        if motzoi is None or not np.isfinite(motzoi):
            print("DRAG fit failed. Please check the data.")
            return False, None

        if update:
            qubit.rxy.motzoi(motzoi)

        return True, motzoi

    def ro_freq_optimization(self, qubit: BasicTransmonElement,
                             frequency_setpoints=None, freq_span=0.2e6, num_points=41,
                             repetitions=1500, plot=False, update=True):
        """Optimize the readout frequency for maximum ground/excited state SNR.

        Like ``drag_calibration``, this bypasses ``meas_ctrl``/``ScheduleGettable``
        and drives ``self.instrument_coordinator`` directly (via
        ``readout_freq_optimization_TWPA``), since it needs raw complex
        acquisition data rather than a processed quantify dataset. For each
        swept frequency the schedule measures the qubit prepared in |g> and
        |e>; ``RO_freq_fit`` computes the IQ-blob SNR per frequency and fits a
        parabola to find the SNR-maximizing frequency.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        frequency_setpoints : np.ndarray, optional
            Readout frequencies to sweep, in Hz. Defaults to ``num_points``
            points spanning the qubit's current readout frequency
            +/- ``freq_span``.
        freq_span : float, optional
            Half-width of the default sweep (default 0.2 MHz; used only when
            ``frequency_setpoints`` is None).
        num_points : int, optional
            Number of points in the default sweep (default 41; used only when
            ``frequency_setpoints`` is None).
        repetitions : int, optional
            Number of g/e shot pairs averaged per frequency (default 1500).
        plot : bool, optional
            Whether to display the SNR-vs-frequency fit figure (default False).
        update : bool, optional
            Whether to write the optimal frequency back to the qubit on
            success (default True).

        Returns
        -------
        (bool, float or None)
            ``(True, optimal_freq)`` if the fit is trustworthy, else
            ``(False, None)``. On success, if ``update`` is True, the qubit's
            readout clock frequency is updated in place.
        """
        if self.instrument_coordinator is None:
            raise RuntimeError(
                "ro_freq_optimization requires instrument_coordinator to be set on "
                "CalibrationNodes — pass instrument_coordinator=ic to the constructor."
            )

        if frequency_setpoints is None:
            freq_center = qubit.clock_freqs.readout()
            frequency_setpoints = np.linspace(freq_center - freq_span, freq_center + freq_span, num_points)

        ro_freq_ds = readout_freq_optimization_TWPA(
            qubit=qubit,
            frequencies=frequency_setpoints,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )

        optimal_freq = RO_freq_fit(ro_freq_ds, frequency_setpoints, plot=plot)

        if optimal_freq is None or not np.isfinite(optimal_freq):
            print("RO frequency optimization failed. Please check the data.")
            return False, None

        if update:
            qubit.clock_freqs.readout(optimal_freq)

        return True, optimal_freq

    def ro_amp_optimization(self, qubit: BasicTransmonElement,
                            amp_setpoints=None, num_points=41,
                            repetitions=1500, plot=False, update=True):
        """Optimize the readout pulse amplitude for maximum g/e discrimination fidelity.

        Like ``ro_freq_optimization``, this bypasses ``meas_ctrl``/``ScheduleGettable``
        and drives ``self.instrument_coordinator`` directly (via
        ``readout_amp_optimization_TWPA``), since it needs raw complex
        acquisition data rather than a processed quantify dataset. For each
        swept amplitude the schedule measures the qubit prepared in |g> and
        |e>; ``RO_amp_fit`` runs ``two_state_discriminator`` per amplitude to
        get the average fidelity and the ground-state fidelity (SNR is not
        used here), and returns the amplitude that maximizes the average
        fidelity.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        amp_setpoints : np.ndarray, optional
            Readout pulse amplitudes to sweep. Defaults to ``num_points``
            points spanning 0.8x-1.2x the qubit's current readout pulse
            amplitude.
        num_points : int, optional
            Number of points in the default sweep (default 41; used only when
            ``amp_setpoints`` is None).
        repetitions : int, optional
            Number of g/e shot pairs averaged per amplitude (default 1500).
        plot : bool, optional
            Whether to display the fidelity-vs-amplitude fit figure (default False).
        update : bool, optional
            Whether to write the optimal amplitude back to the qubit on
            success (default True).

        Returns
        -------
        (bool, float or None)
            ``(True, optimal_amp)`` if the fit is trustworthy, else
            ``(False, None)``. On success, if ``update`` is True, the qubit's
            readout pulse amplitude is updated in place.
        """
        if self.instrument_coordinator is None:
            raise RuntimeError(
                "ro_amp_optimization requires instrument_coordinator to be set on "
                "CalibrationNodes — pass instrument_coordinator=ic to the constructor."
            )

        if amp_setpoints is None:
            base_amp = qubit.measure.pulse_amp()
            amp_setpoints = np.linspace(base_amp * 0.8, base_amp * 1.2, num_points)

        ro_amp_ds = readout_amp_optimization_TWPA(
            qubit=qubit,
            amps=amp_setpoints,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )

        optimal_amp = RO_amp_fit(ro_amp_ds, amp_setpoints, plot=plot)

        if optimal_amp is None or not np.isfinite(optimal_amp):
            print("RO amplitude optimization failed. Please check the data.")
            return False, None

        if update:
            qubit.measure.pulse_amp(optimal_amp)

        return True, optimal_amp

    def ro_len_optimization(self, qubit: BasicTransmonElement,
                            len_setpoints=None, num_points=41,
                            repetitions=1500, plot=False, update=True):
        """Optimize the readout duration for maximum g/e discrimination fidelity.

        Like ``ro_amp_optimization``, this bypasses ``meas_ctrl``/``ScheduleGettable``
        and drives ``self.instrument_coordinator`` directly (via
        ``readout_len_optimization_TWPA``), sweeping the readout pulse
        duration and integration time together (both set to the same swept
        value per point) via ``SSBIntegrationComplex`` acquisitions. ``RO_len_fit``
        runs ``two_state_discriminator`` per duration to get the average
        fidelity and the ground-state fidelity, and returns the duration that
        maximizes the average fidelity.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        len_setpoints : np.ndarray, optional
            Readout durations to sweep, in seconds. Defaults to
            ``num_points`` points spaced 0.1 us apart, centred on the qubit's
            current readout pulse duration.
        num_points : int, optional
            Number of points in the default sweep (default 41; used only when
            ``len_setpoints`` is None).
        repetitions : int, optional
            Number of g/e shot pairs averaged per duration (default 1500).
        plot : bool, optional
            Whether to display the fidelity-vs-duration fit figure (default False).
        update : bool, optional
            Whether to write the optimal duration back to the qubit on
            success (default True).

        Returns
        -------
        (bool, float or None)
            ``(True, optimal_duration)`` if the fit is trustworthy, else
            ``(False, None)``. On success, if ``update`` is True, the qubit's
            readout pulse duration and integration time are both updated in
            place.
        """
        if self.instrument_coordinator is None:
            raise RuntimeError(
                "ro_len_optimization requires instrument_coordinator to be set on "
                "CalibrationNodes — pass instrument_coordinator=ic to the constructor."
            )

        if len_setpoints is None:
            base_len = qubit.measure.pulse_duration()
            len_setpoints = base_len + 0.1e-6 * (np.arange(num_points) - num_points // 2)

        ro_len_ds = readout_len_optimization_TWPA(
            qubit=qubit,
            lens=len_setpoints,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )

        optimal_duration = RO_len_fit(ro_len_ds, len_setpoints, plot=plot)

        if optimal_duration is None or not np.isfinite(optimal_duration):
            print("RO duration optimization failed. Please check the data.")
            return False, None

        if update:
            qubit.measure.pulse_duration(optimal_duration)
            qubit.measure.integration_time(optimal_duration)

        return True, optimal_duration

    def ro_weight_calibration(self, qubit: BasicTransmonElement, lo_freq=None,
                              division_length=10, repetitions=5000,
                              plot=False, update=True):
        """Extract matched-filter readout integration weights from averaged g/e traces.

        Like ``drag_calibration``, this bypasses ``meas_ctrl``/``ScheduleGettable``
        and drives ``self.instrument_coordinator`` directly (via
        ``readout_weight_optimization_TWPA``), since it needs the raw Trace
        acquisition rather than a processed quantify dataset. Acquires one
        hardware-averaged Trace for the qubit prepared in |g> and one for
        |e> (averaged over ``repetitions``), then ``readout_weight_extractor``
        demodulates both and derives the matched-filter weight functions from
        their normalized difference.

        Parameters
        ----------
        qubit : BasicTransmonElement
            The qubit object to calibrate.
        lo_freq : float, optional
            Readout LO frequency, needed to demodulate the raw trace to
            baseband. Defaults to the LO frequency configured for this
            qubit's readout portclock in
            ``quantum_device.generate_hardware_config()``.
        division_length : int, optional
            Number of consecutive 1 ns samples averaged into each weight bin
            before upsampling back (default 10; smooths noise out of the
            weight function).
        repetitions : int, optional
            Hardware averages for the g/e trace pair (default 5000).
        plot : bool, optional
            Whether to display the extracted weight functions (default False).
        update : bool, optional
            Whether to write the extracted weights back to the qubit on
            success (default True).

        Returns
        -------
        (bool, (np.ndarray, np.ndarray) or None)
            ``(True, (weights_a, weights_b))`` if extraction succeeds, else
            ``(False, None)``. On success, if ``update`` is True, the qubit's
            acq_weights_a/acq_weights_b are updated in place.
        """
        if self.instrument_coordinator is None:
            raise RuntimeError(
                "ro_weight_calibration requires instrument_coordinator to be set on "
                "CalibrationNodes — pass instrument_coordinator=ic to the constructor."
            )

        if lo_freq is None:
            portclock = f"{qubit.name}:res-{qubit.name}.ro"
            lo_freq = (
                self.quantum_device.generate_hardware_config()
                ['hardware_options']['modulation_frequencies'][portclock]['lo_freq']
            )

        g_trace, e_trace = readout_weight_optimization_TWPA(
            qubit=qubit,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )

        weights = readout_weight_extractor(
            g_trace, e_trace, qubit, lo_freq,
            division_length=division_length, plot=plot,
        )

        if weights is None:
            print("Readout weight calibration failed. Please check the data.")
            return False, None

        weights_a, weights_b = weights
        if update:
            qubit.measure.acq_weights_a(weights_a)
            qubit.measure.acq_weights_b(weights_b)

        return True, weights

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
        elements = ManualParameter(name="elements")
        elements.batched = True
        elements.batch_size = 21

        allxy_sched_kwargs = {"qubit": qubit, "element_select_idx": elements}

        gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=allxy_sched_TWPA,
            schedule_kwargs=allxy_sched_kwargs,
            real_imag=True,
            batched=True,
        )
        self.meas_ctrl.gettables(gettable)
        self.meas_ctrl.verbose(False)
        self.quantum_device.cfg_sched_repetitions(repetitions)

        self.meas_ctrl.settables(elements)
        self.meas_ctrl.setpoints(np.arange(21))

        ds = self.meas_ctrl.run(f"{qubit.name} ALLXY", soft_avg=1)

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
            1.875, the assumed average number of physical gates per Clifford
            for this compiler's decomposition — if that decomposition ever
            changes, this conversion factor needs to change with it.
        """
        if seed_setpoints is None:
            seed_setpoints = np.random.randint(0, 2**31 - 1, size=num_seeds, dtype=np.int32)

        length = ManualParameter(name="length", unit="#", label="Number of Clifford gates")
        length.batched = True

        seed = ManualParameter(name="seed", unit="", label="Seeds")
        seed.batched = True

        rb_sched_kwargs = {
            "qubit_specifier": qubit,
            "lengths": length,
            "seeds": seed,
        }

        gettable = ScheduleGettable(
            self.quantum_device,
            schedule_function=randomized_benchmarking_schedule,
            schedule_kwargs=rb_sched_kwargs,
            real_imag=False,
            batched=True,
            num_channels=1,
            max_batch_size=length_setpoints.size,
        )
        self.meas_ctrl.gettables(gettable)
        self.meas_ctrl.verbose(False)
        self.quantum_device.cfg_sched_repetitions(repetitions)

        self.meas_ctrl.settables([length, seed])
        self.meas_ctrl.setpoints_grid([length_setpoints, seed_setpoints])

        ds = self.meas_ctrl.run(f"{qubit.name} Randomized Benchmarking")

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

        # error per Clifford -> error per physical gate, assuming an average
        # of 1.875 physical gates per Clifford for this compiler
        r_gate = r / 1.875
        gate_fidelity = 1 - r_gate
        print(f'RB Characterized: alpha={alpha:.4f}, r_g={r_gate:.3e}, '
              f'F_g={gate_fidelity * 100:.3f}%, '
              f'sample_size_variance={rb_result.sample_size_variance:.2f}')

        return True, r_gate

    # ------------------------------------------------------------------
    # iSWAP AC-Stark phases (two-qubit edge)
    #
    # The phases live on the edge, qubit1_qubit2.iswap.ac_stark_phase_q1/q2
    # (q1 = qubit1 = parent element, q2 = qubit2 = child), which every schedule builder reads
    # through utils._ac_stark_phases. So they are saved in each dataset
    # snapshot, next to the pump frequency they belong to.
    # ------------------------------------------------------------------
    def _stark_edge_params(self, qubits):
        """The edge's ``(ac_stark_phase for qubits[0], for qubits[1])``
        qcodes parameters, in ``qubits`` order."""
        for edge_name, swapped in ((f"{qubits[0].name}_{qubits[1].name}", False),
                                   (f"{qubits[1].name}_{qubits[0].name}", True)):
            try:
                edge = self.quantum_device.get_edge(edge_name)
            except KeyError:
                continue
            params = edge.iswap.parameters
            if "ac_stark_phase_q1" not in params or "ac_stark_phase_q2" not in params:
                raise RuntimeError(
                    f"Edge {edge_name} has no iswap.ac_stark_phase_q1/q2 parameters -- "
                    "update the CompositeiSWAPEdge definition."
                )
            pair = (params["ac_stark_phase_q1"], params["ac_stark_phase_q2"])
            return pair[::-1] if swapped else pair
        raise RuntimeError(f"No iSWAP edge between {qubits[0].name} and {qubits[1].name}.")

    @staticmethod
    def _bracket_null(m_plus, m_minus, delta, err=0.0):
        """Locate the null ``n`` of a V-shaped magnitude ``|x - n|`` measured
        at ``x = +delta`` and ``x = -delta``.

        Three cases, each with its own consistency condition:
        ``|n| < delta``: ``m+ + m- = 2 delta``, ``n = (m- - m+)/2``;
        ``n > delta``: ``m- - m+ = 2 delta``, ``n = (m+ + m-)/2``;
        ``n < -delta``: ``m+ - m- = 2 delta``, ``n = -(m+ + m-)/2``.
        Returns ``None`` if none holds within tolerance (drift between the two
        runs, or a fold past 0/180 deg). The tolerance includes 3x the
        combined fit error ``err``: RPE_f rates carry 2-4 deg, so a fixed 15%
        rejected a 6.8 deg mismatch that was within 2 sigma (2026-10-02).
        """
        tol = max(3.0, 0.15 * 2 * delta, 3 * err)
        candidates = []
        for mismatch, null in ((abs(m_plus + m_minus - 2 * delta), (m_minus - m_plus) / 2),
                               (abs(m_minus - m_plus - 2 * delta), (m_plus + m_minus) / 2),
                               (abs(m_plus - m_minus - 2 * delta), -(m_plus + m_minus) / 2)):
            if mismatch < tol:
                candidates.append((mismatch, null))
        return min(candidates)[1] if candidates else None

    def _stark_phase_bracket(self, qubits, schedule_function, extra_kwargs, length_setpoints,
                             offsets, magnitude, run_label, repetitions, plot):
        """Run ``schedule_function`` with the AC-Stark constants shifted by
        ``+offsets`` and then ``-offsets``, fit both rates with
        ``rpe_rate_fit`` (channel y0, i.e. qubits[0]) and convert them with
        ``magnitude``. The edge's phases are always restored afterwards.

        Returns ``(base, (m_plus, m_minus), err)`` with ``err`` the two rate
        errors added in quadrature, or ``(base, None, None)`` if either fit
        was rejected.
        """
        p0, p1 = self._stark_edge_params(qubits)
        base = (p0(), p1())
        for idx, q in enumerate(qubits):
            q.measure.acq_channel(idx)

        length = ManualParameter(name="length", unit="#", label="Repetitions")
        length.batched = True

        mags, errs = {}, {}
        try:
            for sign in (+1, -1):
                p0(base[0] + sign * offsets[0])
                p1(base[1] + sign * offsets[1])

                sched_kwargs = dict(
                    qubit_specifier=qubits,
                    lengths=length,
                    acq_protocol="ThresholdedAcquisition",
                    quantum_device=self.quantum_device,
                    **extra_kwargs,
                )
                gettable = ScheduleGettable(
                    self.quantum_device,
                    schedule_function=schedule_function,
                    schedule_kwargs=sched_kwargs,
                    batched=True,
                    num_channels=len(qubits),
                    max_batch_size=length_setpoints.size,
                )
                self.meas_ctrl.gettables(gettable)
                self.meas_ctrl.settables(length)
                self.meas_ctrl.setpoints(length_setpoints)
                self.quantum_device.cfg_sched_repetitions(repetitions)

                ds = self.meas_ctrl.run(f"{run_label} {'plus' if sign > 0 else 'minus'}", soft_avg=1)
                fit = rpe_rate_fit(ds, channel='y0', plot=plot)
                if fit is None:
                    return base, None, None
                mags[sign] = magnitude(fit['rate'])
                errs[sign] = fit['rate_err']
        finally:
            p0(base[0])
            p1(base[1])

        return base, (mags[+1], mags[-1]), float(np.hypot(errs[+1], errs[-1]))

    def stark_phase_difference_calibration(self, qubits: Iterable[BasicTransmonElement],
                                           offset=15.0, length_setpoints=np.arange(1, 16),
                                           repetitions=200, plot=False, update=True):
        """Calibrate the difference of the edge's ``iswap.ac_stark_phase_q1 - q2``
        with an ``iswap_RPE_f`` bracket.

        RPE_f oscillates at ``|theta_1 - theta_2 - 2 phi_p|`` per repetition.
        It is run with the difference shifted by ``+2*offset`` (qubits[0] +
        offset, qubits[1] - offset) and by ``-2*offset``; the two rates locate the null
        (``_bracket_null``) with its sign, which a single run cannot.

        The difference contains the pump phase ``phi_p``, which jumps whenever
        the pump frequency is reprogrammed -- recalibrate after every pump
        retune.

        Parameters
        ----------
        qubits : Iterable[BasicTransmonElement]
            ``[q0, q1]`` of the iSWAP edge, in the order used by the
            schedules. ``q.measure.acq_channel`` is set to 0 / 1.
        offset : float, optional
            Per-qubit offset in degrees (default 15, i.e. +/-30 on the
            difference). With 10, one side's rate drops to 10-18 deg per
            repetition -- under one cycle in 15 repetitions -- and the fit is
            rejected; 15-16 fitted to 0.8-2.3 deg every time (2026-10-02).
        length_setpoints : np.ndarray, optional
            RPE_f repetitions (default 1..15).
        repetitions : int, optional
            Shots per point (default 200).
        plot : bool, optional
            Show the two rate fits (default False).
        update : bool, optional
            Write the corrected phases to the iSWAP edge on success
            (default True).

        Returns
        -------
        (bool, dict or None)
            ``(True, {'residual', 'residual_err', <qubits[0].name>,
            <qubits[1].name>, 'magnitudes'})`` on success, the new phase keyed
            by qubit name --
            a dict rather than a float since the edge has two phases -- else
            ``(False, None)``. The edge is only changed on success.
        """
        qubits = list(qubits)
        delta = 2 * offset
        base, mags, err = self._stark_phase_bracket(
            qubits, iswap_RPE_f, {}, length_setpoints, (offset, -offset),
            magnitude=lambda rate: rate,
            run_label=f"{qubits[0].name} {qubits[1].name} Stark difference",
            repetitions=repetitions, plot=plot,
        )
        if mags is None:
            print('Stark phase difference: rate fit rejected. Constants not updated.')
            return False, None

        null = self._bracket_null(mags[0], mags[1], delta, err)
        if null is None:
            print(f'Stark phase difference: inconsistent bracket (|+|={mags[0]:.2f}, '
                  f'|-|={mags[1]:.2f}, 2*delta={2 * delta:.1f}). Constants not updated.')
            return False, None

        new = (base[0] + null / 2, base[1] - null / 2)
        print(f'Stark phase difference residual = {null:+.2f} +- {err / 2:.2f} deg  '
              f'(|+|={mags[0]:.2f}, |-|={mags[1]:.2f})  ->  '
              f'{qubits[0].name} {new[0]:.2f}, {qubits[1].name} {new[1]:.2f}')
        if update:
            p0, p1 = self._stark_edge_params(qubits)
            p0(new[0])
            p1(new[1])
        return True, {'residual': null, 'residual_err': err / 2, qubits[0].name: new[0],
                      qubits[1].name: new[1], 'magnitudes': mags}

    def stark_phase_sum_calibration(self, qubits: Iterable[BasicTransmonElement],
                                    offset=10.0, length_setpoints=np.arange(1, 26),
                                    repetitions=200, warmup_iswaps=0, iswap_spacing=116e-9,
                                    plot=False, update=True):
        """Calibrate the sum of the edge's ``iswap.ac_stark_phase_q1 + q2`` with an
        ``iswap_RPE_theta_sum`` bracket.

        RPE_theta_sum oscillates at ``180 - |theta_1 + theta_2|`` deg per iSWAP
        pair, free of ``phi_zz`` and ``phi_p``. It is run with both constants
        shifted by ``+offset`` and by ``-offset`` (sum +/- ``2*offset``); the
        two magnitudes locate the null with its sign (``_bracket_null``).

        ``iswap_spacing`` (default 116 ns, one single-qubit slot) separates the
        iSWAPs as in a real circuit. Back to back (spacing 0, 26 ns RF gap) a
        pump pulse lowers the sum phase of the next iSWAP by ~14 deg, so a
        back-to-back calibration is ~30 deg off for any circuit with a 1Q gate
        between iSWAPs (measured 2026-10-02; with spacing >= ~100 ns the
        RPE and single-gate tomography sums agree within ~3 deg).

        Parameters
        ----------
        qubits : Iterable[BasicTransmonElement]
            ``[q0, q1]`` of the iSWAP edge. ``q.measure.acq_channel`` is set to
            0 / 1.
        offset : float, optional
            Per-qubit offset in degrees (default 10, i.e. +/-20 on the sum).
        length_setpoints : np.ndarray, optional
            Number of iSWAP pairs (default 1..25).
        repetitions : int, optional
            Shots per point (default 200).
        warmup_iswaps : int, optional
            Passed to ``iswap_RPE_theta_sum`` (default 0; the rate did not
            depend on it).
        iswap_spacing : float, optional
            Idle between consecutive iSWAPs in seconds (default 116e-9). Set
            0 only to calibrate for back-to-back iSWAPs.
        plot : bool, optional
            Show the two rate fits (default False).
        update : bool, optional
            Write the corrected phases to the iSWAP edge on success
            (default True).

        Returns
        -------
        (bool, dict or None)
            As ``stark_phase_difference_calibration``.
        """
        qubits = list(qubits)
        delta = 2 * offset
        base, mags, err = self._stark_phase_bracket(
            qubits, iswap_RPE_theta_sum,
            {"warmup_iswaps": warmup_iswaps, "iswap_spacing": iswap_spacing}, length_setpoints,
            (offset, offset),
            magnitude=lambda rate: 180.0 - rate,
            run_label=f"{qubits[0].name} {qubits[1].name} Stark sum",
            repetitions=repetitions, plot=plot,
        )
        if mags is None:
            print('Stark phase sum: rate fit rejected. Constants not updated.')
            return False, None

        null = self._bracket_null(mags[0], mags[1], delta, err)
        if null is None:
            print(f'Stark phase sum: inconsistent bracket (|+|={mags[0]:.2f}, '
                  f'|-|={mags[1]:.2f}, 2*delta={2 * delta:.1f}). Constants not updated.')
            return False, None

        new = (base[0] + null / 2, base[1] + null / 2)
        print(f'Stark phase sum residual = {null:+.2f} +- {err / 2:.2f} deg  '
              f'(|+|={mags[0]:.2f}, |-|={mags[1]:.2f})  ->  '
              f'{qubits[0].name} {new[0]:.2f}, {qubits[1].name} {new[1]:.2f}')
        if update:
            p0, p1 = self._stark_edge_params(qubits)
            p0(new[0])
            p1(new[1])
        return True, {'residual': null, 'residual_err': err / 2, qubits[0].name: new[0],
                      qubits[1].name: new[1], 'magnitudes': mags}

    def _rpe_quadrature_pair(self, qubits, schedule_function, length_setpoints, run_label,
                             repetitions, **sched_kwargs):
        """Run ``schedule_function`` twice, closing with X90 and with Y90 on
        qubits[0], and return the two datasets."""
        for idx, q in enumerate(qubits):
            q.measure.acq_channel(idx)
        out = []
        for axis in ("x", "y"):
            length = ManualParameter(name="length", unit="#", label="Repetitions")
            length.batched = True
            gettable = ScheduleGettable(
                self.quantum_device,
                schedule_function=schedule_function,
                schedule_kwargs=dict(qubit_specifier=qubits, lengths=length,
                                     acq_protocol="ThresholdedAcquisition",
                                     quantum_device=self.quantum_device,
                                     final_axis=axis, **sched_kwargs),
                batched=True,
                num_channels=len(qubits),
                max_batch_size=length_setpoints.size,
            )
            self.meas_ctrl.gettables(gettable)
            self.meas_ctrl.settables(length)
            self.meas_ctrl.setpoints(length_setpoints)
            self.quantum_device.cfg_sched_repetitions(repetitions)
            out.append(self.meas_ctrl.run(f"{run_label} {axis}", soft_avg=1))
        return out

    def stark_phase_quadrature_calibration(self, qubits: Iterable[BasicTransmonElement],
                                           sum_spacing=116e-9,
                                           sum_length_setpoints=np.arange(0, 21),
                                           difference_length_setpoints=np.arange(1, 16),
                                           repetitions=400, plot=False, update=True):
        """Calibrate both AC-Stark phases of the iSWAP edge from cos/sin
        quadrature pairs -- four short runs, no offsets, signs included.

        * Sum: ``iswap_RPE_theta_sum`` with ``iswap_spacing=sum_spacing``
          (default one 1Q slot, see ``stark_phase_sum_calibration``), closed
          with X90 and with Y90. The correction to the sum is ``psi - 180``.
        * Difference: ``iswap_RPE_f`` closed with X90 and with Y90. The
          correction to the difference is ``psi``.

        Both sign conventions were measured on 2026-10-02 with +/-20 deg
        offsets (slopes -1.04 and -1.01). Compared with the bracket nodes this
        needs no offsets, so a null near one side cannot make a fit fail, and
        the one-time phase ``phi0`` comes out for free (it was ~25 deg with
        back-to-back iSWAPs and ~0-8 deg with spaced ones).

        Parameters
        ----------
        qubits : Iterable[BasicTransmonElement]
            ``[q1, q2]`` of the edge, in schedule order.
        sum_spacing : float, optional
            Idle between consecutive iSWAPs of the sum sequence, seconds.
        sum_length_setpoints, difference_length_setpoints : np.ndarray, optional
            Pairs for RPE_theta_sum (default 0..20; m = 0 is a plain Ramsey and
            is excluded from the fit) and repetitions for RPE_f (1..15).
        repetitions : int, optional
            Shots per point (default 400).
        plot : bool, optional
            Show the quadrature fits (default False).
        update : bool, optional
            Write the corrected phases to the edge on success (default True).

        Returns
        -------
        (bool, dict or None)
            ``(True, {'sum_residual', 'sum_err', 'difference_residual',
            'difference_err', 'phi0_sum', <qubits[0].name>, <qubits[1].name>})``
            on success, else ``(False, None)``. The edge is only changed if
            both fits pass.
        """
        qubits = list(qubits)
        name = f"{qubits[0].name} {qubits[1].name}"

        ds_sx, ds_sy = self._rpe_quadrature_pair(
            qubits, iswap_RPE_theta_sum, sum_length_setpoints, f"{name} Stark sum quadrature",
            repetitions, iswap_spacing=sum_spacing)
        fit_s = rpe_quadrature_fit(ds_sx, ds_sy, psi_range=(90.0, 270.0), plot=plot)

        ds_fx, ds_fy = self._rpe_quadrature_pair(
            qubits, iswap_RPE_f, difference_length_setpoints, f"{name} Stark difference quadrature",
            repetitions)
        fit_f = rpe_quadrature_fit(ds_fx, ds_fy, psi_range=(-180.0, 180.0), plot=plot)

        if fit_s is None or fit_f is None:
            print('Stark phase quadrature: a fit was rejected. Edge not updated.')
            return False, None

        r_sum = fit_s['psi'] - 180.0
        r_diff = fit_f['psi']
        p0, p1 = self._stark_edge_params(qubits)
        new = (p0() + (r_sum + r_diff) / 2, p1() + (r_sum - r_diff) / 2)
        print(f'Stark sum residual {r_sum:+.2f} +- {fit_s["psi_err"]:.2f} deg (one-time phase '
              f'{fit_s["phi0"]:+.1f} deg), difference residual {r_diff:+.2f} +- {fit_f["psi_err"]:.2f} deg  '
              f'->  {qubits[0].name} {new[0]:.2f}, {qubits[1].name} {new[1]:.2f}')
        if update:
            p0(new[0])
            p1(new[1])
        return True, {'sum_residual': r_sum, 'sum_err': fit_s['psi_err'],
                      'difference_residual': r_diff, 'difference_err': fit_f['psi_err'],
                      'phi0_sum': fit_s['phi0'], qubits[0].name: new[0], qubits[1].name: new[1]}

    def run_stark_phase_calibration(self, qubits: Iterable[BasicTransmonElement], rounds=3,
                                    tol_difference=5.0, tol_sum=2.0,
                                    offset_difference=15.0, offset_sum=10.0,
                                    repetitions=200, plot=False):
        """Orchestrate the AC-Stark phase calibration of an iSWAP edge.

        Each round runs ``stark_phase_difference_calibration`` then
        ``stark_phase_sum_calibration`` (each updates the phases on success),
        and stops once the difference residual is below ``tol_difference``
        and the sum residual below ``tol_sum`` degrees. A rejected bracket --
        usually because the null sits near one side of it, where that rate
        is close to 0 or 180 deg -- is retried once with 1.6x its offset.
        This is the orchestration layer; the nodes themselves never retry.

        The tolerances differ because the two halves repeat differently
        (2026-10-02, 15 deg difference offset): successive sum brackets agree
        to <1 deg, while successive difference brackets scatter by ~+/-4 deg
        with alternating sign (-3.35, +5.47, -3.24; fit errors ~1 deg) and a
        net change of only -1.1 deg. A difference tolerance below ~5 deg
        just chases that scatter. The sum drifts by ~5-10 deg over ten
        minutes, so recalibrate it shortly before the runs that need it.

        Returns
        -------
        (bool, list of dict)
            ``(converged, log)``; one log record per node call.
        """
        qubits = list(qubits)
        prev_verbose = self.meas_ctrl.verbose()
        self.meas_ctrl.verbose(False)
        log = []
        converged = False
        try:
            for rnd in range(1, rounds + 1):
                print(f'\n=== Stark phase round {rnd} ===')
                residuals = {}
                for label, node, offset, tol in (
                        ('difference', self.stark_phase_difference_calibration, offset_difference, tol_difference),
                        ('sum', self.stark_phase_sum_calibration, offset_sum, tol_sum)):
                    ok, value = node(qubits, offset=offset, repetitions=repetitions, plot=plot)
                    if not ok:
                        print(f'{label}: retrying with offset {1.6 * offset:.1f} deg')
                        ok, value = node(qubits, offset=1.6 * offset, repetitions=repetitions, plot=plot)
                    log.append({'round': rnd, 'step': label, 'ok': ok, 'value': value})
                    if not ok:
                        print(f'{label} failed twice -- stopping.')
                        return False, log
                    residuals[label] = (value['residual'], tol)

                if all(abs(r) < t for r, t in residuals.values()):
                    converged = True
                    break
        finally:
            self.meas_ctrl.verbose(prev_verbose)

        print(f"\nStark phases {'converged' if converged else 'NOT converged'} "
              f"after {rnd} round(s): ")
        p0, p1 = self._stark_edge_params(qubits)
        print(f'{qubits[0].name} {p0():.2f} deg, {qubits[1].name} {p1():.2f} deg (stored on the edge).')
        return converged, log

    def run_roadmap(self, roadmap, qubits, stop_on_fail=True, retries=2):
        """Run an ordered sequence of calibration nodes across one or more qubits.

        Each roadmap step is called as ``getattr(self, method_name)(qubit,
        **kwargs)`` and must follow the standard node contract
        (``(bool, value)``, no exception on a bad fit) — this only orchestrates
        and reports, it doesn't change how any individual node decides
        success/failure. Steps that need more than one qubit at once (e.g.
        ``multiplexed_readout_calibration``) don't fit this per-qubit loop and
        should be called separately, outside the roadmap.

        ``meas_ctrl.verbose`` is muted for the duration of the roadmap (so a
        multi-step, multi-qubit run doesn't spam progress bars) and restored
        to whatever it was before, even if a step raises.

        Parameters
        ----------
        roadmap : list of (str, str, dict)
            Ordered ``(label, method_name, kwargs)`` triples, run in order for
            each qubit. ``method_name`` must name a single-qubit node method on
            this class.
        qubits : BasicTransmonElement or Iterable[BasicTransmonElement]
            Qubit(s) to run the roadmap on, in order.
        stop_on_fail : bool, optional
            If True (default), stop running further steps *for that qubit* as
            soon as one step fails all its attempts (other qubits still run
            their full roadmap). If False, run every step regardless of
            earlier failures.
        retries : int, optional
            Extra attempts for a step that fails, before it's counted as a
            real failure (default 2, so up to 3 attempts total).

        Returns
        -------
        list of dict
            One record per (qubit, step): ``{'qubit', 'step', 'ok', 'value',
            'attempts'}``, in the order the steps ran.
        """
        if hasattr(qubits, "name"):
            qubits = [qubits]
        else:
            qubits = list(qubits)

        prev_verbose = self.meas_ctrl.verbose()
        self.meas_ctrl.verbose(False)
        try:
            log = []
            for qubit in qubits:
                print(f'\n=== {qubit.name} ===')
                for label, method_name, kwargs in roadmap:
                    method = getattr(self, method_name)
                    ok, value = False, None
                    for attempt in range(retries + 1):
                        try:
                            ok, value = method(qubit, **kwargs)
                        except Exception as exc:
                            ok, value = False, None
                            print(f'{qubit.name}: {label} raised {type(exc).__name__}: {exc}')

                        if ok:
                            break
                        if attempt < retries:
                            print(f'{qubit.name}: {label} failed '
                                  f'(attempt {attempt + 1}/{retries + 1}), retrying...')

                    log.append({'qubit': qubit.name, 'step': label, 'ok': ok,
                                'value': value, 'attempts': attempt + 1})

                    if not ok and stop_on_fail:
                        print(f'{qubit.name}: stopping roadmap after "{label}" '
                              f'failed {retries + 1} times.')
                        break
        finally:
            self.meas_ctrl.verbose(prev_verbose)

        n_fail = sum(1 for r in log if not r['ok'])
        print(f'\nRoadmap complete: {len(log) - n_fail}/{len(log)} steps passed.')
        if n_fail:
            print('Failed steps:')
            for r in log:
                if not r['ok']:
                    print(f"  - {r['qubit']}: {r['step']}")

        return log