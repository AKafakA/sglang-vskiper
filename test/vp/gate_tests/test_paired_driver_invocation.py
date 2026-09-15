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


def test_a_refused_cell_does_not_abort_its_SIBLING_RATES():
    """An arm-run covers every rate of its dataset in ONE invocation, so without this the
    first refused cell costs the rest. Measured: gsm8k r8p25 refused on skipping_executed
    (peak occupancy 96 vs enter_rows=176) and took r10p45 and r13p75 with it -- both of which
    engage cleanly at peak 389/386 -- leaving rep 1 gsm8k with zero usable pairs.

    The ladder has always passed this flag for the same reason."""
    uses = [l for l in DRIVER_CODE if "--continue-after-accounting-rejection" in l]
    assert uses, "a refused rate will abort the whole arm-run"


def test_that_flag_does_NOT_weaken_the_per_cell_gates():
    """It continues past a refusal; it does not suppress one. The refused cell's artifacts are
    still renamed INVALID.* and GR-1a still fails that rate -- which is what keeps 'the arm
    finished' from being read as 'every rate passed'."""
    assert 'target = path.with_name(f"INVALID.{path.name}")' in RUNNER
    assert "an unchecked pair is a failed pair" in DRIVER


# --- sweep support: one anchor, many treatments (plan N1) -------------------------------

def _load_driver():
    import importlib.util, sys
    path = ROOT / "test/vp/run_paired_campaign.py"
    spec = importlib.util.spec_from_file_location("run_paired_campaign", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_paired_campaign", module)
    spec.loader.exec_module(module)
    return module


def test_treatments_of_accepts_exactly_one_of_treatment_or_treatments():
    m = _load_driver()
    assert m.treatments_of({"arms": {"baseline": "upstream", "treatment": "integrated_it4"}}) == ["integrated_it4"]
    sweep = {"arms": {"baseline": "upstream", "treatments": ["a", "b", "c"]}}
    assert m.treatments_of(sweep) == ["a", "b", "c"]
    assert [r for r, _ in m.campaign_arms(sweep)] == ["baseline", "treatment", "treatment", "treatment"]
    for bad in ({"arms": {"baseline": "upstream"}},
                {"arms": {"baseline": "upstream", "treatment": "x", "treatments": ["y"]}}):
        try:
            m.treatments_of(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"accepted {bad}")


def test_every_treatment_is_gated_against_the_anchor():
    """Both per-rep gates take the treatment explicitly and main loops over the list, so a
    sweep point is never silently compared against the wrong arm."""
    code = _code_lines(DRIVER)
    assert any("def cross_arm_config_gate(" in l for l in code)
    assert any("out_dir: Path, treatment: str) -> bool" in l for l in code)
    assert any("rep: int, out_dir: Path, treatment: str) -> list[str]" in l for l in code)
    assert any("for treatment in treatments:" in l for l in code)
    assert not any('spec["arms"]["treatment"]' in l for l in code), "a hard-coded single treatment survives"


def test_vskipper_is_the_served_arm_and_integrated_it4_is_its_alias():
    from sglang.srt.vpipe.design import ARMS
    assert ARMS["vskipper"] == {"skipper": "flexidepth", "phases": "both", "regime_switch": True}
    assert ARMS["integrated_it4"] is ARMS["vskipper"]
    import pathlib
    active = pathlib.Path(__file__).resolve().parents[3] / "deploy/active_arm"
    assert active.read_text().strip() == "vskipper"
