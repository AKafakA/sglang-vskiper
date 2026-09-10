#!/usr/bin/env python3
"""The runner REFUSES a bad cell -- proven by refusal, not by a pass.

Until this landed, all six gates under `test/vp/gates/` had ZERO callers anywhere in the
repository, and the hand-written drivers that did invoke them piped the gate into
`tail|sed`, discarding its exit code. A refusing gate printed FATAL and the chain printed
DONE nine seconds later.

That is the third instance of one defect: D-614 (`design_attestation()` had no callers),
D-624 finding 5 ("the statistics exist but are not mandatory"), D-627 (a night of quality
numbers describing the no-skip body). Every previous fix added a gate FILE. This tests the
CALL SITE, and it tests the direction that matters -- that a defective cell FAILS.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

VP = Path(__file__).resolve().parents[1]
if str(VP) not in sys.path:
    sys.path.insert(0, str(VP))

import run_qps_evaluation as runner  # noqa: E402

N = 6
OFFSETS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def _server_info(active_phases, *, arm="integrated_it4", skipper="flexidepth"):
    return {
        "internal_states": [
            {
                "vp_runtime": {
                    "served_design": {
                        "arm": arm,
                        "skipper": skipper,
                        "active_phases": list(active_phases),
                    }
                }
            }
        ]
    }


def _write_arrival(path: Path, offsets_s=OFFSETS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"timestamp": t * 1000.0}) + "\n" for t in offsets_s)
    )
    return path


def _write_artifact(path: Path, *, texts=None, offsets_s=OFFSETS, ignore_eos=False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "request_ids": [f"gsm8k:{i}" for i in range(N)],
        "request_start_offsets_s": list(offsets_s),
        "generated_texts": texts if texts is not None else [f"answer {i}" for i in range(N)],
    }
    if ignore_eos:
        record["ignore_eos"] = True
    path.write_text(json.dumps(record) + "\n")
    return path


def _gates(tmp_path: Path, *, intent="natural_serving", routes_decode=False, **artifact):
    return runner._cell_gates(
        output_file=_write_artifact(tmp_path / "cell.jsonl", **artifact),
        arrival_file=_write_arrival(tmp_path / "cell.arrival.requests.jsonl"),
        server_info_before=tmp_path / "before.json",
        server_info_after=tmp_path / "after.json",
        evaluation_intent=intent,
        routes_decode=routes_decode,
        env=dict(os.environ),
    )


# --- the served arm is read from what the server attests, never from a flag we passed ---


def test_routed_arm_is_detected():
    assert runner._arm_routes_decode(_server_info(["decode", "prefill"])) is True


def test_stock_arm_is_not_expected_to_skip():
    # The baseline arm legitimately never skips; gating it on skip>0 would fail every
    # control cell in the campaign.
    assert runner._arm_routes_decode(_server_info([], arm="stock", skipper=None)) is False


def test_prefill_only_arm_does_not_claim_decode_routing():
    assert runner._arm_routes_decode(_server_info(["prefill"])) is False


def test_missing_design_attestation_is_a_refusal_not_a_false():
    """A pre-D-609 tree cannot say what it served, and silence must not read as 'stock'."""
    with pytest.raises(RuntimeError, match="served_design"):
        runner._arm_routes_decode({"internal_states": [{"vp_runtime": {}}]})


# --- the gates themselves ---


def test_clean_natural_cell_passes(tmp_path):
    assert _gates(tmp_path) == []


def test_one_empty_generation_fails_the_cell(tmp_path):
    """The defect that survived a month: an empty body is lost output, not a low score."""
    texts = [f"answer {i}" for i in range(N)]
    texts[3] = ""
    assert "zero_empty" in _gates(tmp_path, texts=texts)


def test_whitespace_only_generation_also_fails(tmp_path):
    texts = [f"answer {i}" for i in range(N)]
    texts[0] = "   \n"
    assert "zero_empty" in _gates(tmp_path, texts=texts)


def test_equal_work_lane_does_not_run_the_zero_empty_gate(tmp_path):
    """`ignore_eos` fills an immediate EOS to budget, so zero-empty would be vacuous --
    and the gate REFUSES such an artifact, which would fail every campaign cell."""
    texts = [f"answer {i}" for i in range(N)]
    texts[3] = ""
    assert _gates(
        tmp_path, intent="production_max_equal_work", texts=texts, ignore_eos=True
    ) == []


def test_unknown_intent_is_still_checked(tmp_path):
    """An intent we do not recognise must be gated, never silently waved through."""
    texts = [f"answer {i}" for i in range(N)]
    texts[2] = ""
    assert "zero_empty" in _gates(tmp_path, intent="mixed_or_unknown", texts=texts)


def test_arrivals_that_departed_from_the_trace_fail(tmp_path):
    """Two arms handed the same trace must see the same traffic, or the paired deltas are
    between different workloads -- invisible to every output gate (D-624 #1)."""
    drifted = [0.0, 1.0, 2.0, 30.0, 60.0, 90.0]
    assert "arrival_fidelity" in _gates(tmp_path, offsets_s=drifted)


# --- a refused cell must not be mistaken for a passing one ---


def test_invalidated_artifacts_are_renamed_and_the_path_is_freed(tmp_path):
    cell = _write_artifact(tmp_path / "cell.jsonl")
    score = tmp_path / "cell.score.json"
    score.write_text("{}\n")

    renamed = runner._invalidate_cell_artifacts([cell, score, tmp_path / "absent.json"])

    assert not cell.exists() and not score.exists()
    assert (tmp_path / "INVALID.cell.jsonl").is_file()
    assert (tmp_path / "INVALID.cell.score.json").is_file()
    assert len(renamed) == 2  # the absent path is skipped, not invented


def test_invalidation_is_repeatable_across_reruns(tmp_path):
    """The runner refuses to overwrite an existing artifact, so a second failed attempt
    must still be able to clear the path."""
    for _ in range(2):
        cell = _write_artifact(tmp_path / "cell.jsonl")
        runner._invalidate_cell_artifacts([cell])
        assert not cell.exists()
    assert (tmp_path / "INVALID.cell.jsonl").is_file()
