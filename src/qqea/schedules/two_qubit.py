"""Two-qubit schedules: RB, iSWAP characterisation, state/process tomography and GST.

The builders live in their own modules (``gates``, ``phase``, ``rb``, ``iswap``,
``tomography``, ``gst``). This module re-exports them, so imports from
``qqea.schedules.two_qubit`` keep working. ``RBAnalysis`` stays here until it moves
to ``qqea.fitting``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from quantify_core.analysis.single_qubit_timedomain import SingleQubitTimedomainAnalysis
from quantify_core.visualization.mpl_plotting import (
    set_suptitle_from_dataset,
    set_xlabel,
    set_ylabel,
)
from scipy.optimize import curve_fit

from qqea.schedules.gates import (
    ISWAP_CONSECUTIVE_GAP,
    ISWAP_SETTLE,
    index_to_operation,
    iSWAP,
    iSWAP_DELAY,
)
from qqea.schedules.gst import (
    GST_pre_build,
    GST_schedule,
    pygsti_circuit_to_gate_decomposition,
    pygsti_circuit_to_operation,
)
from qqea.schedules.iswap import (
    iswap_delay_schedule,
    iswap_pulsed_pump_schedule,
    iswap_RPE_d,
    iswap_RPE_f,
    iswap_RPE_theta_sum,
    iswap_tail_1q_schedule,
    iswap_virtual_z_schedule,
    pump_probe_ramsey_schedule,
    pump_probe_swap_schedule,
)
from qqea.schedules.phase import (
    AC_STARK_PHASE_Q0,
    AC_STARK_PHASE_Q1,
    PUMP_OFF_DBM,
    compute_iswap_t1,
)
from qqea.schedules.rb import (
    randomized_benchmarking_schedule,
    simultaneous_randomized_benchmarking_schedule,
)
from qqea.schedules.tomography import (
    process_tomography_pre_build,
    process_tomography_schedule,
    tomography_pre_build,
    tomography_schedule,
)

if TYPE_CHECKING:
    from xarray import Dataset

__all__ = [
    "iSWAP_DELAY",
    "ISWAP_SETTLE",
    "ISWAP_CONSECUTIVE_GAP",
    "iSWAP",
    "index_to_operation",
    "PUMP_OFF_DBM",
    "AC_STARK_PHASE_Q0",
    "AC_STARK_PHASE_Q1",
    "compute_iswap_t1",
    "randomized_benchmarking_schedule",
    "simultaneous_randomized_benchmarking_schedule",
    "iswap_pulsed_pump_schedule",
    "iswap_tail_1q_schedule",
    "pump_probe_ramsey_schedule",
    "pump_probe_swap_schedule",
    "iswap_virtual_z_schedule",
    "iswap_RPE_f",
    "iswap_RPE_theta_sum",
    "iswap_RPE_d",
    "iswap_delay_schedule",
    "tomography_pre_build",
    "tomography_schedule",
    "process_tomography_pre_build",
    "process_tomography_schedule",
    "pygsti_circuit_to_gate_decomposition",
    "pygsti_circuit_to_operation",
    "GST_pre_build",
    "GST_schedule",
    "RBAnalysis",
]


class RBAnalysis(SingleQubitTimedomainAnalysis):
    """
    Analysis class for the randomized benchmarking (RB) experiment.

    This class extends the SingleQubitTimedomainAnalysis class, which in turn extends the
    BaseAnalysis class:
    - BaseAnalysis.run() runs all steps in the AnalysisSteps class:
        1. process_data                  # Empty
        2. run_fitting                   # Empty
        3. analyze_fit_results           # Empty
        4. create_figures                # Empty
        5. adjust_figures                # Defined
        6. save_figures                  # Defined
        7. save_quantities_of_interest   # Defined
        8. save_processed_dataset        # Defined
        9. save_fit_results              # Defined
    - SingleQubitTimedomainAnalysis extends BaseAnalysis:
        - run() defines self.calibration_points
        - process_data() populates dataset_processed.S21 and dataset_processed.pop_exc
    - RBAnalysis extends SingleQubitTimedomainAnalysis:
        - process_data() is extended by calculating:
            - pop_exc
        - create_figures() is defined
    """

    def __init__(  # noqa: D107
        self,
        dataset: Dataset = None,
        tuid: str = None,
        label: str = "",
        settings_overwrite: dict = None,
        plot_figures: bool = True,
        repetitions: int = 1,
    ) -> None:
        super().__init__(dataset, tuid, label, settings_overwrite, plot_figures)
        self.repetitions = repetitions

    def run(self):  # noqa: F811
        """
        Run the SingleQubitTimedomainAnalysis with calibration_points.

        This removes the calibration points (last two) and converts
        the rest of the IQ values to a population (pop_exc).
        """
        return super().run(calibration_points=True)

    def process_data(self):  # noqa: D102
        def _error_per_clifford(
            alpha: float,
            n_qubits: int = 1,
        ) -> float:
            """Error per Clifford as defined in eq.(1) of arxiv:1712.06550."""
            return (2**n_qubits - 1) / 2**n_qubits * (1 - alpha)

        def _scaled_binomial_var(
            pop_exc: float,
            scale_factor: float,
        ) -> float:
            r"""Return the variance of measuring state <1|\psi>=pop_exc."""
            return scale_factor * pop_exc * (1 - pop_exc) / self.repetitions

        def _rb_decay(
            m: int,
            alpha: float,
        ) -> float:
            """Exponential decay consistent with eq.(1) of arxiv:1712.06550."""
            # TODO: This inherits SingleQubitTimeDomainAnalysis, but for n qubits
            # this would be (1-1/2^n) * alpha**m + 1/2^n. If we want to take
            # into account T1 decay as well, we should use generic
            # A * alpha**m + (1-A), although this would massively increase the
            # error in our estimate of alpha (2 fitting params instead of 1).
            # One option is to use an optional argument
            # asymptotic_fidelity: float | None = 0.5, and use A, (1-A)
            # if asymptotic_fidelity is None
            return 0.5 * alpha**m + 0.5

        # The processed data set gives us the excited state population vs time.
        # From this, we can calculate the error rate and fidelity.
        super().process_data()

        # TODO: If we don't go back to the initial state, then 1-<1|\psi> is
        # no longer the right metric.
        overlap = 1 - self.dataset_processed["pop_exc"].values
        overlap[-2:] = [np.nan, np.nan]  # Remove calibration points
        # Add the overlap to the dataset that's returned to the user
        self.dataset_processed["overlap"] = (["x0"], overlap)

        # m_values are the setpoints for the number of Cliffords per measurement
        # for this specific batch of measurements
        m_values = self.dataset_processed["overlap"].coords["x0"].values
        # TODO: once the problem with calibration points is fixed (QTFY-11),
        # replace by NaN mask that doesn't mask values 0,1
        nan_mask = np.array(
            [  # Mask the NaNs, as well as 0s and 1s
                not math.isnan(el) and el not in (0, 1) for el in overlap
            ]
        )
        # Fit exponential decay
        popt, pcov = curve_fit(
            _rb_decay,
            m_values[nan_mask],  # Mask the [-2:] calibration NaNs
            overlap[nan_mask],  # Mask the [-2:] calibration NaNs
            p0=(0.9),  # Use alpha=0.9 as starting point
        )
        (self.alpha,) = popt

        # Convert alpha to r as defined in eq.(1) of arxiv:1712.06550
        self.r = _error_per_clifford(alpha=self.alpha, n_qubits=1)
        # Since the measurement is 2D, seed setpoints [A,B] and m setpoints
        # [1,2,3] will give m_values for this batch of [1,2,3,1,2,3], but we
        # only want [1,2,3].
        unique_m = np.unique(m_values)
        # Add the unique m as a coordinate axis to the dataset
        self.dataset_processed = self.dataset_processed.assign_coords(
            unique_m=("unique_m", unique_m)
        )
        # Add the fit to the dataset
        self.dataset_processed["fitted_overlap"] = (["unique_m"], _rb_decay(unique_m, *popt))

        # Calculate standard deviations for each m, i.e. the spread between the
        # seeds
        variances = []
        for m_idx, datasubset in self.dataset_processed.groupby("x0"):
            nan_mask = np.array(
                [  # Mask the NaNs, as well as 0s and 1s
                    not math.isnan(el) and el not in (0, 1) for el in datasubset["overlap"].values
                ]
            )
            variances.append(np.var(datasubset["overlap"].values[nan_mask]))
        # Add the variance to the dataset
        self.dataset_processed["var"] = (["unique_m"], variances)

        # We now fit the fitted overlap vs the variance. A larger spread between
        # seeds than 1/n indicates coherent errors.
        popt, pcov = curve_fit(
            _scaled_binomial_var,
            self.dataset_processed["fitted_overlap"].values,
            variances,
            p0=(1),
        )
        # The sole fitting parameter is a constant scaling factor between the
        # expected curve and the data
        (self.variance_scale_factor,) = popt
        # To give an error estimate of 2 sigma, we take 2 times the
        # top-left component of the covariance matrix of the fit we just did
        self.variance_scale_factor_err = 2 * np.sqrt(np.diag(pcov)[0])
        # If we get X times larger variance than the sample size would suggest,
        # then the sample size only accounts for 1/X
        self.sample_size_variance = 1 / self.variance_scale_factor
        # Calculate a lower bound on the contribution of sample size to the variance
        self.szv_lo_err = self.sample_size_variance - 1 / (
            self.variance_scale_factor + self.variance_scale_factor_err
        )
        # Calculate an upper bound on the contribution of sample size to the variance
        self.szv_hi_err = (
            1 / (self.variance_scale_factor - self.variance_scale_factor_err)
            - self.sample_size_variance
        )

    def create_figures(self):  # noqa: D102
        def _binomial_var(pop_exc: float) -> float:
            r"""Return the variance of measuring state <1|\psi>=pop_exc."""
            return pop_exc * (1 - pop_exc)

        def _forward_scale(y: float) -> float:
            """Rescale y axis such that the plot is linear in m."""
            scale = np.full(y.shape, np.nan)  # Initialize with NaN
            log_scale_cutoff = 0.5
            mask = y > log_scale_cutoff  # Boolean mask for valid elements
            scale[mask] = np.log(2 * y[mask] - 1)  # Transform only valid elements
            return scale

        def _inverse_scale(scaled_y: float) -> float:
            """Inverse of _forward_scale to get data values from y coordinates."""
            return (np.exp(scaled_y) + 1) / 2

        nice_markers = [
            "o",
            "v",
            "^",
            "<",
            ">",
            "s",
            "p",
            "P",
            "*",
            "h",
            "H",
            "+",
            "x",
            "X",
            "D",
            "d",
        ]
        fig_id = "Randomized benchmarking"
        fig, ax = plt.subplots(constrained_layout=True)
        self.figs_mpl[fig_id] = fig
        self.axs_mpl[fig_id] = ax
        data_per_seed = self.dataset_processed.groupby("x1")
        for seed_idx, (_, ds) in enumerate(data_per_seed):
            ax.scatter(ds["x0"], ds["overlap"], marker=nice_markers[seed_idx % len(nice_markers)])
        ax.set_yscale("function", functions=(_forward_scale, _inverse_scale))
        ax.set_xlim(0, None)
        ax.set_ylim(0.54, 1)
        ax.autoscale(enable=True, axis="x")
        ax.margins(x=0.05)

        m_values = self.dataset_processed["unique_m"].values
        fitted_overlap = self.dataset_processed["fitted_overlap"].values

        # Plot the fit of gate string length vs fidelity
        ax.plot(
            np.append(m_values, 0),
            np.append(fitted_overlap, 1),
            label=rf"Fit: $\alpha=${self.alpha:.3f}, $r=${self.r:.3f}",
        )
        ax.legend()
        set_xlabel(r"Gate string length $m$")
        set_ylabel(r"Excited state population")
        set_suptitle_from_dataset(fig, self.dataset)

        # Calculate the variance vs fidelity
        p = np.linspace(min(fitted_overlap), 1, 201)  # probabilities on the x-axis
        # You'd expect the variance between repetitions to scale like
        # p(1-p) / n,
        # because we're drawing from a biased set of [0,1] with replacement.
        # Note that we ignore for the moment the variance that comes from
        # different physical gate lengths for different seeds. 1 single-qubit
        # Clifford may transpile to anywhere between 0 (I) and 3 physical gates.
        # The estimate is further skewed for small batch sizes because we remove
        # the last 2 data points (QTFY-11)
        exp_variance = _binomial_var(p) / self.repetitions
        fig_id = "Fidelity variance"
        fig, ax = plt.subplots(constrained_layout=True)
        ax.scatter(
            fitted_overlap, self.dataset_processed["var"], label="Measured variance between seeds"
        )
        ax.plot(p, exp_variance, label=f"Expected variance for {self.repetitions} repetitions")
        (fit_curve,) = ax.plot(
            p,
            self.variance_scale_factor * exp_variance,
            label=f"Fitted variance with scaling {self.variance_scale_factor:.2f}",
        )
        ax.fill_between(
            p,
            (self.variance_scale_factor - self.variance_scale_factor_err) * exp_variance,
            (self.variance_scale_factor + self.variance_scale_factor_err) * exp_variance,
            color=fit_curve.get_color(),
            alpha=0.3,
            zorder=1,
        )

        ax.set_xscale("function", functions=(_forward_scale, _inverse_scale))
        ax.set_ylim(0, None)
        ax.autoscale(enable=True, axis="x")
        ax.margins(x=0.05)
        ax.legend()
        set_xlabel(r"Fidelity $m$")
        set_ylabel(r"Fidelity variance between seeds")
        set_suptitle_from_dataset(fig, self.dataset)
