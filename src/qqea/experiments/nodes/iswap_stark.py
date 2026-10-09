"""iSWAP AC-Stark phase nodes for a two-qubit edge."""

from typing import Iterable

import numpy as np
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

from qqea.experiments.nodes._base import NodeBase, batched_parameter, node
from qqea.fitting.fits import rpe_quadrature_fit, rpe_rate_fit
from qqea.schedules.two_qubit import iswap_RPE_f, iswap_RPE_theta_sum


class IswapStarkNodes(NodeBase):

    # ------------------------------------------------------------------
    # iSWAP AC-Stark phases (two-qubit edge)
    #
    # The phases live on the edge, qubit1_qubit2.iswap.ac_stark_phase_q1/q2
    # (q1 = qubit1 = parent element, q2 = qubit2 = child), which every schedule builder reads
    # through the schedules' _ac_stark_phases. So they are saved in each dataset
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
        self._assign_acq_channels(qubits)

        length = batched_parameter("length", unit="#", label="Repetitions")

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
                ds = self._run_sweep(
                    schedule_function, sched_kwargs, length, length_setpoints,
                    f"{run_label} {'plus' if sign > 0 else 'minus'}", repetitions, soft_avg=1,
                    num_channels=len(qubits), max_batch_size=length_setpoints.size,
                )
                fit = self._try_fit(rpe_rate_fit, ds, channel='y0', plot=plot)
                if fit is None:
                    return base, None, None
                mags[sign] = magnitude(fit['rate'])
                errs[sign] = fit['rate_err']
        finally:
            p0(base[0])
            p1(base[1])

        return base, (mags[+1], mags[-1]), float(np.hypot(errs[+1], errs[-1]))

    @node
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

    @node
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
        self._assign_acq_channels(qubits)
        out = []
        for axis in ("x", "y"):
            length = batched_parameter("length", unit="#", label="Repetitions")
            out.append(self._run_sweep(
                schedule_function,
                dict(qubit_specifier=qubits, lengths=length,
                     acq_protocol="ThresholdedAcquisition",
                     quantum_device=self.quantum_device,
                     final_axis=axis, **sched_kwargs),
                length, length_setpoints, f"{run_label} {axis}", repetitions, soft_avg=1,
                num_channels=len(qubits), max_batch_size=length_setpoints.size,
            ))
        return out

    @node
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
        fit_s = self._try_fit(rpe_quadrature_fit, ds_sx, ds_sy, psi_range=(90.0, 270.0), plot=plot)

        ds_fx, ds_fy = self._rpe_quadrature_pair(
            qubits, iswap_RPE_f, difference_length_setpoints, f"{name} Stark difference quadrature",
            repetitions)
        fit_f = self._try_fit(rpe_quadrature_fit, ds_fx, ds_fy, psi_range=(-180.0, 180.0), plot=plot)

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

    @node
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
        log = []
        converged = False
        with self._quiet():
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

        print(f"\nStark phases {'converged' if converged else 'NOT converged'} "
              f"after {rnd} round(s): ")
        p0, p1 = self._stark_edge_params(qubits)
        print(f'{qubits[0].name} {p0():.2f} deg, {qubits[1].name} {p1():.2f} deg (stored on the edge).')
        return converged, log
