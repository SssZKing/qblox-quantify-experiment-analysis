"""Readout nodes: IQ-blob threshold/rotation and readout frequency, amplitude,
duration and integration-weight optimization."""

from typing import Iterable

import numpy as np
from quantify_core.analysis.readout_calibration_analysis import ReadoutCalibrationAnalysis
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

from qqea.experiments.nodes._base import NodeBase, batched_parameter, node
from qqea.experiments.runners import (
    readout_amp_optimization,
    readout_freq_optimization,
    readout_len_optimization,
    readout_weight_optimization,
)
from qqea.fitting.fits import RO_amp_fit, RO_freq_fit, RO_len_fit, readout_weight_extractor
from qqea.schedules.single_qubit import (
    multiplexed_readout_calibration_sched,
    readout_calibration_sched,
)


def _average_fidelity(qoi):
    return (qoi['fid_est_0'] + qoi['fid_est_1']) / 2


def _threshold_and_rotation(qoi):
    """``(threshold, rotation in degrees mod 360)`` from a ReadoutCalibrationAnalysis."""
    threshold = qoi['acq_threshold'].nominal_value
    rotation = np.rad2deg(qoi['acq_rotation_rad'].nominal_value) % 360
    return threshold, rotation


class ReadoutNodes(NodeBase):

    @node
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
        states = batched_parameter("states", batch_size=600)
        ro_cali_sched_kwargs = {
            "qubit": qubit,
            "prepared_states": states,
            "acq_protocol": "SSBIntegrationComplex",
        }  # SSBIntegrationComplex or NumericalSeparatedWeightedIntegration

        states_setpoints = np.hstack((np.zeros(num_points), np.ones(num_points)))
        ro_ds = self._run_sweep(
            readout_calibration_sched, ro_cali_sched_kwargs, states, states_setpoints,
            f"{qubit.name} Readout Calibration", 1, real_imag=True,
        )

        ro_analysis = ReadoutCalibrationAnalysis(
            dataset=ro_ds.where(ro_ds.x0 != 2, drop=True), plot_figures=plot
        )
        ro_analysis.run()
        if plot:
            ro_analysis.display_figs_mpl()

        qoi = ro_analysis.quantities_of_interest
        avg_fid = _average_fidelity(qoi)

        if avg_fid <= fidelity_threshold:
            print(f"Readout calibration failed: avg fidelity {avg_fid:.4f} "
                  f"below {fidelity_threshold}. Threshold/rotation not updated.")
            return False, None

        threshold, rotation = _threshold_and_rotation(qoi)
        if update:
            qubit.measure.acq_threshold(threshold)
            qubit.measure.acq_rotation(rotation)

        print('Readout Calibrated:')
        print(f'avg fidelity = {avg_fid:.4f}, threshold = {threshold * 1e3:.2f} mV, '
              f'rotation = {rotation:.1f} deg')

        return True, avg_fid

    @node
    def multiplexed_readout_calibration(self, qubits: Iterable[BasicTransmonElement], num_points=6000,
                                        fidelity_threshold=0.9, plot=False, update=True):
        """Calibrate readout threshold/rotation for several qubits read out at once.

        Runs a single multiplexed IQ-blob measurement, mapping each qubit to its
        own acquisition channel (qubits[i] -> channel i, left in place), then
        analyses each qubit's blobs separately. A qubit's acq_threshold and
        acq_rotation are updated only if its own average assignment fidelity
        exceeds ``fidelity_threshold``.

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
        self._assign_acq_channels(qubits)

        states = batched_parameter("states", batch_size=600)
        ro_cali_sched_kwargs = {
            "qubits": qubits,
            "prepared_states": states,
            "acq_protocol": "SSBIntegrationComplex",
        }  # SSBIntegrationComplex or NumericalSeparatedWeightedIntegration

        states_setpoints = np.hstack((np.zeros(num_points), np.ones(num_points)))
        multi_ro_ds = self._run_sweep(
            multiplexed_readout_calibration_sched, ro_cali_sched_kwargs, states,
            states_setpoints, "Multiplexed Readout Calibration", 1,
            real_imag=True, num_channels=len(qubits),
        )

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
            avg_fid = _average_fidelity(qoi)
            fidelities[qubit.name] = avg_fid

            if avg_fid > fidelity_threshold:
                threshold, rotation = _threshold_and_rotation(qoi)
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

    @node
    def ro_freq_optimization(self, qubit: BasicTransmonElement,
                             frequency_setpoints=None, freq_span=0.2e6, num_points=41,
                             repetitions=1500, plot=False, update=True):
        """Optimize the readout frequency for maximum ground/excited state SNR.

        Like ``drag_calibration``, this bypasses ``meas_ctrl``/``ScheduleGettable``
        and drives ``self.instrument_coordinator`` directly (via
        :func:`qqea.experiments.runners.readout_freq_optimization`); the raw
        data is saved to the data directory. For each swept frequency the
        schedule measures the qubit prepared in |g> and |e>; ``RO_freq_fit``
        computes the IQ-blob SNR per frequency and fits a parabola to find the
        SNR-maximizing frequency.

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
        self._require_instrument_coordinator("ro_freq_optimization")

        if frequency_setpoints is None:
            freq_center = qubit.clock_freqs.readout()
            frequency_setpoints = np.linspace(freq_center - freq_span, freq_center + freq_span, num_points)

        ro_freq_ds = readout_freq_optimization(
            qubit=qubit,
            frequencies=frequency_setpoints,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )
        self._save_raw(ro_freq_ds, f"{qubit.name} RO frequency optimization",
                       frequency_setpoints=np.asarray(frequency_setpoints),
                       repetitions=repetitions)

        optimal_freq = RO_freq_fit(ro_freq_ds, frequency_setpoints, plot=plot)

        if optimal_freq is None or not np.isfinite(optimal_freq):
            print("RO frequency optimization failed. Please check the data.")
            return False, None

        if update:
            qubit.clock_freqs.readout(optimal_freq)

        return True, optimal_freq

    @node
    def ro_amp_optimization(self, qubit: BasicTransmonElement,
                            amp_setpoints=None, num_points=41,
                            repetitions=1500, plot=False, update=True):
        """Optimize the readout pulse amplitude for maximum g/e discrimination fidelity.

        Like ``ro_freq_optimization``, this drives the instrument coordinator
        directly (via :func:`qqea.experiments.runners.readout_amp_optimization`)
        and saves the raw data. For each swept amplitude the schedule measures
        the qubit prepared in |g> and |e>; ``RO_amp_fit`` runs
        ``two_state_discriminator`` per amplitude to get the average fidelity
        and the ground-state fidelity (SNR is not used here), and returns the
        amplitude that maximizes the average fidelity.

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
        self._require_instrument_coordinator("ro_amp_optimization")

        if amp_setpoints is None:
            base_amp = qubit.measure.pulse_amp()
            amp_setpoints = np.linspace(base_amp * 0.8, base_amp * 1.2, num_points)

        ro_amp_ds = readout_amp_optimization(
            qubit=qubit,
            amps=amp_setpoints,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )
        self._save_raw(ro_amp_ds, f"{qubit.name} RO amplitude optimization",
                       amp_setpoints=np.asarray(amp_setpoints), repetitions=repetitions)

        optimal_amp = RO_amp_fit(ro_amp_ds, amp_setpoints, plot=plot)

        if optimal_amp is None or not np.isfinite(optimal_amp):
            print("RO amplitude optimization failed. Please check the data.")
            return False, None

        if update:
            qubit.measure.pulse_amp(optimal_amp)

        return True, optimal_amp

    @node
    def ro_len_optimization(self, qubit: BasicTransmonElement,
                            len_setpoints=None, num_points=41,
                            repetitions=1500, plot=False, update=True):
        """Optimize the readout duration for maximum g/e discrimination fidelity.

        Like ``ro_amp_optimization``, this drives the instrument coordinator
        directly (via :func:`qqea.experiments.runners.readout_len_optimization`)
        and saves the raw data, sweeping the readout pulse duration and
        integration time together (both set to the same swept value per point)
        via ``SSBIntegrationComplex`` acquisitions. ``RO_len_fit`` runs
        ``two_state_discriminator`` per duration to get the average fidelity
        and the ground-state fidelity, and returns the duration that maximizes
        the average fidelity.

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
        self._require_instrument_coordinator("ro_len_optimization")

        if len_setpoints is None:
            base_len = qubit.measure.pulse_duration()
            len_setpoints = base_len + 0.1e-6 * (np.arange(num_points) - num_points // 2)

        ro_len_ds = readout_len_optimization(
            qubit=qubit,
            lens=len_setpoints,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )
        self._save_raw(ro_len_ds, f"{qubit.name} RO duration optimization",
                       len_setpoints=np.asarray(len_setpoints), repetitions=repetitions)

        optimal_duration = RO_len_fit(ro_len_ds, len_setpoints, plot=plot)

        if optimal_duration is None or not np.isfinite(optimal_duration):
            print("RO duration optimization failed. Please check the data.")
            return False, None

        if update:
            qubit.measure.pulse_duration(optimal_duration)
            qubit.measure.integration_time(optimal_duration)

        return True, optimal_duration

    @node
    def ro_weight_calibration(self, qubit: BasicTransmonElement, lo_freq=None,
                              division_length=10, repetitions=5000,
                              plot=False, update=True):
        """Extract matched-filter readout integration weights from averaged g/e traces.

        Like ``drag_calibration``, this drives the instrument coordinator
        directly (via :func:`qqea.experiments.runners.readout_weight_optimization`),
        since it needs the raw Trace acquisition; both traces are saved to
        the data directory. Acquires one hardware-averaged Trace for the qubit
        prepared in |g> and one for |e> (averaged over ``repetitions``), then
        ``readout_weight_extractor`` demodulates both and derives the
        matched-filter weight functions from their normalized difference.

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
        self._require_instrument_coordinator("ro_weight_calibration")

        if lo_freq is None:
            portclock = f"{qubit.name}:res-{qubit.name}.ro"
            lo_freq = (
                self.quantum_device.generate_hardware_config()
                ['hardware_options']['modulation_frequencies'][portclock]['lo_freq']
            )

        g_trace, e_trace = readout_weight_optimization(
            qubit=qubit,
            instrument_coordinator=self.instrument_coordinator,
            quantum_device=self.quantum_device,
            repetitions=repetitions,
        )
        for state, trace in (("g", g_trace), ("e", e_trace)):
            self._save_raw(trace, f"{qubit.name} RO weight trace {state}",
                           prepared_state=state, repetitions=repetitions)

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
