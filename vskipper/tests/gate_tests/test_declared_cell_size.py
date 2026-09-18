#!/usr/bin/env python3
"""The cell size is DECLARED. A suite that is not the agreed size cannot produce a cell.

Owner, 2026-09-10: *"4000 is a requirement"* ... *"4000 for bbh only"*.

The failure this encodes: a BBH suite was rebuilt with `--num-requests 3600`, copied from
gsm8k, and 26 ladder cells were measured against it before anyone noticed. Every other gate
passed on those cells -- accounting balanced, zero empties, duration correctly derived -- and
all of it was true OF THE WRONG SUITE. Row counts had only ever been read with `wc -l`, which
reports what the file IS, never what was agreed.

So these tests check the REFUSAL, not the pass: a gate that has never been observed failing
is not evidence (D-256, D-614, D-627 all passed their gates).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

VP = Path(__file__).resolve().parents[3] / "vskipper/src/vskipper/experiments"
if str(VP) not in sys.path:
    sys.path.insert(0, str(VP))

import run_qps_evaluation as runner  # noqa: E402


# --- the declaration itself, which is the part the owner ruled on ---


def test_the_three_campaign_sizes_are_what_the_owner_ruled():
    """gsm8k 3600 and coqa 4000 both date from 09-04 and match v1.3 -- neither was changed.
    BBH is 4000 by the 09-10 ruling, correcting the silent 3600."""
    assert runner.DECLARED_SUITE_ROWS == {"gsm8k": 3600, "coqa": 4000, "bbh_cot": 4000}


# --- the refusal ---


def test_the_3600_row_bbh_suite_is_REFUSED():
    """The exact artifact that cost 26 ladder cells."""
    with pytest.raises(ValueError) as exc:
        runner._refuse_undeclared_cell_size("bbh_cot.3shot.raw", 3600)
    assert "3600 rows" in str(exc.value) and "declared at 4000" in str(exc.value)


def test_a_correctly_sized_suite_passes():
    runner._refuse_undeclared_cell_size("bbh_cot.3shot.raw", 4000)
    runner._refuse_undeclared_cell_size("gsm8k.d179", 3600)
    runner._refuse_undeclared_cell_size("coqa.d179", 4000)


def test_gsm8k_at_4000_is_ALSO_refused():
    """The ruling is 4000 for BBH only. gsm8k stays 3600, so 'fixing' it upward is equally a
    contract deviation -- which is what makes this a declaration rather than a floor."""
    with pytest.raises(ValueError):
        runner._refuse_undeclared_cell_size("gsm8k.d179", 4000)


# --- which suites the declaration reaches ---


@pytest.mark.parametrize(
    "workload,expected",
    [
        ("gsm8k.d179", ("gsm8k", 3600)),
        ("coqa.d179", ("coqa", 4000)),
        ("bbh_cot.3shot.raw", ("bbh_cot", 4000)),
        # the equal-work banks inherit their dataset's size
        ("gsm8k_eqw_r10p45", ("gsm8k", 3600)),
        ("bbh_cot_eqw_r33p25", ("bbh_cot", 4000)),
        ("coqa_nat", ("coqa", 4000)),
    ],
)
def test_campaign_suites_resolve_to_their_dataset(workload, expected):
    assert runner.declared_rows_for(workload) == expected


def test_bbh_cot_is_never_read_as_a_shorter_dataset():
    """Matched longest-first. If `bbh` were ever declared, `bbh_cot` must not collapse into
    it -- the two have different protocols and different agreed sizes."""
    assert runner.declared_rows_for("bbh_cot.3shot.raw")[0] == "bbh_cot"


@pytest.mark.parametrize("workload", ["decode_mix", "mixed", "prefill_core", "humaneval",
                                      "gsm8k_cot_zeroshot.d179"])
def test_a_non_campaign_suite_is_NOT_asserted(workload):
    """A workload outside the campaign has no agreed size. A gate that invented one would be
    firing on its own guesses, which is the opposite of a declaration."""
    assert runner.declared_rows_for(workload) is None
    runner._refuse_undeclared_cell_size(workload, 137)


def test_a_DECLARED_diagnostic_is_exempt_but_a_measurement_is_not():
    """`gsm8k.first100` is a real 100-row smoke suite that resolves to gsm8k. It must run as a
    diagnostic and be refused as a measurement -- the flag already means 'never performance
    evidence', and the 26 BBH cells passed no flag at all, so this exemption does not reopen
    the hole."""
    assert runner.declared_rows_for("gsm8k.first100") == ("gsm8k", 3600)
    runner._refuse_undeclared_cell_size("gsm8k.first100", 100, diagnostic=True)
    with pytest.raises(ValueError):
        runner._refuse_undeclared_cell_size("gsm8k.first100", 100, diagnostic=False)
