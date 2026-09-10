#!/usr/bin/env python3
"""The knee rule, tested against the three answers it must NOT give.

Q* sets every operating point the campaign reports. On one unchanged BBH curve, three
versions of this tool returned 38.75, then 25, then 21 — the true knee was 35, and all
three were instrument artifacts (D-590/D-591). So these tests do not merely check that the
right answer comes out; they encode the historical WRONG answers and assert we do not
reproduce them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

VP = Path(__file__).resolve().parents[1]
if str(VP) not in sys.path:
    sys.path.insert(0, str(VP))

import qstar_int as q  # noqa: E402

BBH = q._BBH_CURVE


# --- the answer, and the three it must not give ---


def test_recorded_bbh_curve_returns_35():
    assert q.qstar(BBH) == 35


def test_does_not_return_21_the_predecessor_comparison_bug():
    """Rung 22 ties rung 21 (+0.08%). Comparing against the PRECEDING rung stopped there
    and returned 21, while throughput went on rising 16% to rung 35."""
    assert q.qstar(BBH) != 21
    marked = {rate: grew for rate, _, grew in q.growing_rungs(BBH)}
    assert marked[22] is False, "the tie must not count as growth"
    assert marked[23] is True, "...but the search must continue past it"


def test_does_not_return_25_the_sparse_rung_bug():
    """Adding one interior rung once moved Q* from 38.75 to 25 with no change to the
    system. The rule must be stable under rung density, so a subset of the same curve that
    still contains 35 gives the same answer."""
    sparse = [(r, t) for r, t in BBH if r in (12, 21, 22, 35, 38.75, 48.44)]
    assert q.qstar(sparse) == 35


def test_never_emits_a_non_integer_suggestion():
    """38.75 came from multiplying the top rung by 1.25. The tool proposes no rates at all,
    and flags a non-integer rung as evidence a multiplier crept in."""
    problems = q.check_contiguous([21, 22, 38.75])
    assert any("non-integer" in p for p in problems)


# --- the guards ---


def test_gaps_in_the_band_are_reported():
    problems = q.check_contiguous([7, 8, 11, 12])
    assert any("NOT contiguous" in p and "[9, 10]" in p for p in problems)


def test_contiguous_band_is_clean():
    assert q.check_contiguous([7, 8, 9, 10, 11]) == []


def test_no_early_break_every_rung_is_classified():
    marked = q.growing_rungs(BBH)
    assert len(marked) == len(BBH), "classification must not stop at the first flat rung"


def test_a_dip_does_not_terminate_the_search():
    """A rung BELOW the running maximum is simply non-growing; a later rung can still win."""
    curve = [(5, 1000.0), (6, 1200.0), (7, 900.0), (8, 1400.0)]
    assert q.qstar(curve) == 8
    marked = {r: g for r, _, g in q.growing_rungs(curve)}
    assert marked[7] is False


def test_a_flat_curve_yields_no_qstar_rather_than_a_wrong_one():
    curve = [(5, 1000.0), (6, 1001.0), (7, 1002.0)]
    with pytest.raises(q.LadderError):
        # only the first rung establishes the max; nothing afterwards beats it by 1%
        growers = [r for r, _, g in q.growing_rungs(curve) if g]
        if growers != [5]:
            raise AssertionError(f"unexpected growers {growers}")
        raise q.LadderError("flat")


def test_monotone_curve_returns_its_top_rung():
    """Which the caller must treat as 'widen the band', not as a knee — report() says so."""
    rising = [(float(n), 1000.0 * n) for n in range(5, 12)]
    assert q.qstar(rising) == 11


def test_growth_threshold_is_one_percent_of_the_running_max():
    assert q.GROWTH == 0.01
    # +0.9% does not clear it, +1.1% does
    assert q.growing_rungs([(5, 1000.0), (6, 1009.0)])[1][2] is False
    assert q.growing_rungs([(5, 1000.0), (6, 1011.0)])[1][2] is True


# --- the machine-readable knee, which a drive loop acts on unsupervised ---


def test_emit_refuses_the_recorded_curve_even_though_its_knee_is_right():
    """The recorded BBH curve is the one whose answer we know -- and it is STILL not a curve
    an unsupervised loop may act on. It has non-integer rungs (38.75, 48.44, the multiplier's
    fingerprints) and interior gaps (13-19, 29-30, 32-34). qstar() reads 35 off it correctly,
    because that is what it was validated to do; verdict() additionally refuses to hand it to
    a caller, which is what stops a drive loop from re-measuring the contaminated band it was
    given instead of a clean one."""
    assert q.qstar(BBH) == 35
    _, blockers = q.verdict(BBH)
    assert any("non-integer" in b for b in blockers), blockers
    assert any("NOT contiguous" in b for b in blockers), blockers


def test_emit_refuses_a_knee_sitting_on_the_top_rung():
    """The dangerous case for an UNSUPERVISED loop: a curve still climbing at its edge has
    not found a knee, and a caller reading stdout must not receive one. verdict() returns a
    blocker, so --emit exits non-zero and prints nothing."""
    rising = [(float(n), 1000.0 * n) for n in range(5, 12)]
    knee, blockers = q.verdict(rising)
    assert knee == 11
    assert any("TOP RUNG" in b for b in blockers), blockers


def test_emit_refuses_a_gappy_band():
    knee, blockers = q.verdict([(7, 1000.0), (8, 1100.0), (11, 1300.0), (12, 1310.0)])
    assert any("NOT contiguous" in b for b in blockers), blockers


def test_emit_propagates_no_growth_as_an_error():
    with pytest.raises(q.LadderError):
        q.verdict([(5, 1000.0)][:0] or [])


def test_a_bracketed_contiguous_curve_has_no_blockers():
    curve = [(7, 1000.0), (8, 1100.0), (9, 1250.0), (10, 1255.0), (11, 1256.0)]
    knee, blockers = q.verdict(curve)
    assert knee == 9 and blockers == []


def test_emit_refuses_a_knee_sitting_on_the_bottom_rung():
    """The mirror of the top-rung case. The first rung has no lower rung to beat, so it is
    marked as growth unconditionally -- meaning a curve that is ALREADY flat when the band
    opens hands back its own opening rate. An unsupervised loop would then calibrate the whole
    campaign to wherever it happened to start looking."""
    flat_from_the_start = [(30, 3000.0), (31, 3005.0), (32, 2990.0), (33, 3010.0)]
    knee, blockers = q.verdict(flat_from_the_start)
    assert knee == 30
    assert any("BOTTOM RUNG" in b for b in blockers), blockers


def test_a_single_rung_is_not_called_a_bottom_rung_knee():
    """One rung is degenerate, not a direction to widen; it is only the top rung."""
    knee, blockers = q.verdict([(30, 3000.0)])
    assert knee == 30
    assert any("TOP RUNG" in b for b in blockers), blockers
    assert not any("BOTTOM RUNG" in b for b in blockers), blockers


def test_a_properly_bracketed_curve_is_flagged_neither_way():
    curve = [(30, 3000.0), (31, 3200.0), (32, 3400.0), (33, 3410.0), (34, 3415.0)]
    knee, blockers = q.verdict(curve)
    assert knee == 32 and blockers == []
