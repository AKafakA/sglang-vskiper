#!/usr/bin/env python3
"""The paired driver's runner invocation, pinned against two defects that killed 100 % of cells.

Both were found by RUNNING the smoke, minutes apart on 2026-09-11, after the driver had been
read several times without either being noticed:

  1. `--experiment paired-{dataset}-{arm}-rep{rep}` while `--system-id` was the bare arm.
     run_qps_evaluation asserts they are equal, so every cell raised
       ValueError: deployment system 'upstream' does not match experiment
                   'paired-gsm8k-upstream-rep1'
     one minute in, before serving a single request.

  2. `--min-prompts`/`--duration-s`, which the runner REFUSES unless the cell is a declared
     diagnostic (owner rule 2 / D-651). The same defect that killed the ladder driver (D-661),
     in a second caller -- and the arithmetic was independently wrong: one duration derived
     from the SLOWEST rate, applied to every rate in the cell, makes the faster rates submit
     less than their whole suite.

These are source-level assertions on purpose. The invocation is built inline inside the
per-cell function, so importing and calling it would require a live server; what actually
needs protecting is the SHAPE of the command, and that is visible in the source.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DRIVER = (ROOT / "test/vp/run_paired_campaign.py").read_text()
RUNNER = (ROOT / "test/vp/run_qps_evaluation.py").read_text()


def _code_lines(text: str) -> list[str]:
    """Source with comment-only lines dropped, so a comment mentioning a flag is not a use."""
    return [l for l in text.splitlines() if not l.strip().startswith("#")]


DRIVER_CODE = _code_lines(DRIVER)


def test_the_runner_still_asserts_experiment_equals_system_id():
    """If this assertion is ever removed, the test below stops protecting anything."""
    assert 'deployment["system_id"] != args.experiment' in RUNNER


def test_experiment_is_the_BARE_ARM_not_a_composite_label():
    uses = [l for l in DRIVER_CODE if '"--experiment"' in l]
    assert uses, "the driver no longer passes --experiment at all"
    for line in uses:
        assert re.search(r'"--experiment",\s*arm\b', line), (
            f"--experiment must be the bare arm to match --system-id; got: {line.strip()}"
        )


def test_system_id_is_the_same_arm_variable():
    uses = [l for l in DRIVER_CODE if '"--system-id"' in l]
    assert uses and all(re.search(r'"--system-id",\s*arm\b', l) for l in uses), uses


def test_the_driver_passes_NO_duration_or_min_prompts():
    """The runner derives duration = rows/qps per cell. A caller-chosen duration makes each
    rate do different work, which is fatal to a paired table whose unit is a within-rep delta."""
    for flag in ("--duration-s", "--min-prompts"):
        offenders = [l for l in DRIVER_CODE if f'"{flag}"' in l]
        assert not offenders, f"{flag} is passed again: {offenders}"


def test_the_dead_slowest_rate_arithmetic_is_gone():
    """`rows // slowest` was the banned fixed-duration pattern wearing a derivation's clothes."""
    assert "// slowest" not in "\n".join(DRIVER_CODE)
    assert "slowest = min(rates.values())" not in "\n".join(DRIVER_CODE)


def test_upstream_baseline_is_passed_only_for_the_upstream_arm():
    """Its presence inverts the attestation check: vp_runtime must be ABSENT. Passing it on the
    fork, or omitting it on upstream, would make the gate assert the opposite of the truth."""
    line = [l for l in DRIVER_CODE if "--upstream-baseline" in l]
    assert line, "the upstream arm no longer gets --upstream-baseline"
    assert "arm_is_upstream(spec, arm)" in line[0], line
