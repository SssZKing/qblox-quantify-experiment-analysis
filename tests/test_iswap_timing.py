"""iSWAP context-phase compensation and DG645 run grouping (pure functions, no device).

Run in the qblox_dev env (never qblox_env):  python -m pytest tests
"""

import pytest

phase = pytest.importorskip("qqea.schedules.phase")
gst = pytest.importorskip("qqea.schedules.gst")

PULSE = 500e-9
LEAD = 40e-9


def test_context_phases_off_by_default():
    assert phase.ISWAP_CONTEXT_PHASE is None
    assert phase.iswap_context_phases([1e-6, 2e-6], PULSE) is None


def test_context_phases_follow_back_to_back_runs():
    context = {"max_gap": 50e-9, "train": [(10.0, 12.0), (18.0, 20.0)]}
    t0 = 1e-6
    back_to_back = [t0 + k * (PULSE + LEAD) for k in range(4)]
    isolated = back_to_back[-1] + PULSE + 1e-6
    phases = phase.iswap_context_phases(back_to_back + [isolated], PULSE, context)
    # first of a run (0, 0), then train entries with the last one repeating, then a fresh run
    assert phases == [(0.0, 0.0), (10.0, 12.0), (18.0, 20.0), (18.0, 20.0), (0.0, 0.0)]


def test_rz_corrections_subtract_the_context_inside_the_register():
    t = [1e-6, 1.54e-6]
    plain = phase._iswap_rz_corrections(t, 0.0, (5.0, -3.0))
    ctx = phase._iswap_rz_corrections(t, 0.0, (5.0, -3.0), [(0.0, 0.0), (10.0, 12.0)])
    assert ctx[0] == plain[0]
    assert ctx[1] == pytest.approx((plain[1][0] - 10.0, plain[1][1] - 12.0))
    with pytest.raises(ValueError):
        phase._iswap_rz_corrections(t, 0.0, (5.0, -3.0), [(0.0, 0.0)])


def test_iswap_run_starts():
    iswap = (("q0", "q1"), ["iSWAP"])
    one_q = (("q0",), ["X90"])
    decomp = [one_q, iswap, iswap, iswap, one_q, iswap, one_q]
    assert gst._iswap_run_starts(decomp) == [0, 3, 0, 0, 0, 1, 0]
    assert gst._iswap_run_starts([one_q]) == [0]
