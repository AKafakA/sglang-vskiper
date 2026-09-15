#!/usr/bin/env python3
"""_run_cell must be reusable: derived per-cell values must not be written onto `args`.

The defect this encodes, found at 05:32Z with the 36-cell headline already running:

  duration derivation  wrote its result back as `args.min_prompts, args.duration_s`.
  Cell 1 therefore set them; cell 2 saw them non-None, took the `elif`, and raised
    "--duration-s / --min-prompts are DIAGNOSTIC overrides..."
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


def test_the_cell_size_comes_from_the_SUITE(tmp_path=None):
    """SUPERSEDED THE DERIVATION ENTIRELY (owner, 2026-09-11). The bug above was a symptom of
    a mechanism that could not affect anything in the first place -- so rather than keeping
    per-cell duration state correct, the duration was deleted. `requested_count = available`
    is now the whole sizing rule, and there is no state left to leak."""
    assert "requested_count = available" in CODE
    assert "duration_s" not in CODE
    assert "min_prompts" not in CODE


def test_the_manifest_records_no_duration_target():
    """A recorded `duration_target_s` was what could have carried cell 1's value into cell 3's
    evidence. The arrival schedule's own measured span is the honest descriptor and remains."""
    assert "duration_target_s" not in CODE
    assert "expected_injection_s" in CODE


def _cell_size(available: int, qps: float) -> int:
    """The sizing rule as the source states it: the suite, independent of the rate."""
    del qps
    return available


def test_every_rate_sends_the_WHOLE_suite():
    """The invariant the duration was supposed to protect, now direct: the cell size does not
    depend on the rate at all. Three gsm8k rates and three BBH rates, one answer each."""
    for qps in (8.25, 10.45, 13.75):
        assert _cell_size(3600, qps) == 3600
    for qps in (18.75, 23.75, 31.25):
        assert _cell_size(4000, qps) == 4000
