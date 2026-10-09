"""Run an ordered list of single-qubit nodes over one or more qubits."""

from qqea.experiments.nodes._base import NodeBase


class RoadmapNodes(NodeBase):

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

        log = []
        with self._quiet():
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

        n_fail = sum(1 for r in log if not r['ok'])
        print(f'\nRoadmap complete: {len(log) - n_fail}/{len(log)} steps passed.')
        if n_fail:
            print('Failed steps:')
            for r in log:
                if not r['ok']:
                    print(f"  - {r['qubit']}: {r['step']}")

        return log