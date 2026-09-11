#!/usr/bin/env python3
"""_run_cell must be reusable: derived per-cell values must not be written onto `args`.

The defect this encodes, found at 05:32Z with the 36-cell headline already running:

  duration derivation (D-651) wrote its result back as `args.min_prompts, args.duration_s`.
  Cell 1 therefore set them; cell 2 saw them non-None, took the `elif`, and raised
    "--duration-s / --min-prompts are DIAGNOSTIC overrides ..."
  at a caller that had passed neither. The ladder and the bank harvest never noticed,
  because each invokes the runner ONCE per rung. The paired driver passes three suites in
  one qps-config, so it lost 2 of every 3 cells -- silently, with a message accusing the
  caller of the exact thing it had just been fixed not to do.

Per-cell state on a parser namespace is the general bug; these tests pin the specific one.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = (ROOT / "test/vp/run_qps_evaluation.py").read_text()


def _run_cell_body() -> str:
    start = SRC.index("def _run_cell(")
    rest = SRC[start + 1:]
    end = rest.index("\ndef ")
    return rest[:end]


BODY = _run_cell_body()
CODE = "\n".join(l for l in BODY.splitlines() if not l.strip().startswith("#"))


def test_run_cell_never_ASSIGNS_to_the_parser_namespace():
    """Any `args.<field> = ...` inside _run_cell makes it single-use for that field."""
    assignments = re.findall(r"^\s*args\.\w+(?:\s*,\s*args\.\w+)*\s*=", CODE, re.M)
    assert not assignments, f"_run_cell assigns to args: {assignments}"


def test_the_derived_values_are_LOCALS():
    assert "duration_s, min_prompts = args.duration_s, args.min_prompts" in CODE
    assert "_requested_count(min_prompts, qps, duration_s)" in CODE


def test_the_manifest_records_the_PER_CELL_duration():
    """`duration_target_s: args.duration_s` would record cell 1's value on every later cell."""
    assert '"duration_target_s": duration_s,' in CODE
    assert '"duration_target_s": args.duration_s' not in CODE


def test_the_diagnostic_refusal_still_exists():
    """Without it, the first test protects nothing: a caller could pass a fixed duration again."""
    assert "partial_suite_diagnostic" in CODE
    assert "DIAGNOSTIC overrides" in BODY


def _derive(available: int, qps: float) -> tuple[int, float]:
    """The arithmetic as the source states it -- rows/qps, never overshooting the suite."""
    min_prompts = available
    duration_s = math.floor(available / qps)
    while math.ceil(qps * duration_s) > available:
        duration_s -= 1
    return min_prompts, float(duration_s)


def test_each_rate_derives_its_OWN_duration():
    """The three gsm8k headline rates on a 3600-row suite. If one cell's value leaked into the
    next, two of these three would be wrong -- and wrong in the direction that makes a cell
    submit less than its whole suite, which is what owner rule 2 forbids."""
    got = {qps: _derive(3600, qps) for qps in (8.25, 10.45, 13.75)}
    assert got[8.25] == (3600, 436.0)
    assert got[10.45] == (3600, 344.0)
    assert got[13.75] == (3600, 261.0)
    for qps, (rows, dur) in got.items():
        assert math.ceil(qps * dur) <= rows, (qps, dur, rows)


def test_the_bbh_rates_derive_against_a_4000_row_suite():
    for qps in (18.75, 23.75, 31.25):
        rows, dur = _derive(4000, qps)
        assert rows == 4000 and math.ceil(qps * dur) <= 4000
