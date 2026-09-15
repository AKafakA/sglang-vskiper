"""Unit tests for the pure (no-GPU) parts of flexidepth_oracle_runner.

test/vp is the fork's own suite (not CI-registered). The box venvs have no
pytest, so this file also runs standalone: `python test_flexidepth_oracle_runner.py`.
"""

import json

import msgspec

from flexidepth_oracle_runner import (
    FirstTokenEvidence,
    GenerationEvidence,
    ItemStatus,
    ProbeRecord,
    RunManifest,
    StepMask,
    actions_from_masks,
    checked_layer_masks,
    checked_mask_row,
    normalize_eos_ids,
    parse_spec_list,
)


def test_parse_spec_list_comma() -> None:
    assert parse_spec_list("a, b,,c ") == ["a", "b", "c"]


def test_parse_spec_list_file(tmp_path) -> None:
    listing = tmp_path / "ids.txt"
    listing.write_text("gsm8k:test:1\n\n gsm8k:test:2 \n", encoding="utf-8")
    assert parse_spec_list(f"@{listing}") == ["gsm8k:test:1", "gsm8k:test:2"]


def test_normalize_eos_ids() -> None:
    assert normalize_eos_ids(None) == []
    assert normalize_eos_ids(128009) == [128009]
    assert normalize_eos_ids([128001, 128009]) == [128001, 128009]


def test_checked_mask_row_accepts_binary_floats() -> None:
    assert checked_mask_row([1.0, 0.0, 1], 3, "ctx") == [1, 0, 1]


def test_checked_mask_row_rejects_wrong_length() -> None:
    try:
        checked_mask_row([1.0, 0.0], 3, "ctx")
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("wrong-length mask row accepted")


def test_checked_mask_row_rejects_non_binary() -> None:
    try:
        checked_mask_row([1.0, 0.5, 0.0], 3, "ctx")
    except ValueError as exc:
        assert "non-binary" in str(exc)
    else:
        raise AssertionError("non-binary mask value accepted")


def test_checked_layer_masks_rejects_count_mismatch() -> None:
    try:
        checked_layer_masks(("m",), [16, 17], "ctx")
    except ValueError as exc:
        assert "masks" in str(exc)
    else:
        raise AssertionError("mask/layer count mismatch accepted")
    checked_layer_masks(("a", "b"), [16, 17], "ctx")


def test_actions_from_masks_last_position_and_rate() -> None:
    masks = {"16": [1, 1, 0], "17": [0, 0, 1]}
    last_actions, skip_rate = actions_from_masks([16, 17], masks)
    assert last_actions == {"16": "skip", "17": "run"}
    assert abs(skip_rate - 3 / 6) < 1e-9


def test_probe_record_roundtrip() -> None:
    record = ProbeRecord(
        target_id="gsm8k:test:7",
        prompt_protocol="frozen-input-ids",
        model="xuan-luo/FlexiDepth-Llama-3-8B-Instruct",
        model_commit_hash="2ce73595ad0467fedd539c113e5b9deed046df32",
        dtype="bfloat16",
        transformers_version="5.12.1",
        torch_version="2.11.0",
        prompt_tokens=3,
        prompt_token_sha256="ab" * 32,
        max_new_tokens=8,
        routing_layers=[16, 17],
        router_threshold=0.5,
        prompt_router_masks={"16": [1, 0, 1], "17": [0, 1, 1]},
        last_prompt_position_actions={"16": "run", "17": "run"},
        prompt_skip_rate=1 / 3,
        first_token=FirstTokenEvidence(
            argmax_id=42,
            topk_ids=[42, 7],
            topk_logits=[1.5, 0.5],
            generate_first_id=42,
            consistent=True,
        ),
        generation=GenerationEvidence(
            text="", completion_tokens=1, last_token_id=128009, finish_empty=True
        ),
        step_router_masks=[
            StepMask(step=0, token_id=42, actions={"16": 1, "17": 0})
        ],
        step_generation_matches_generate=True,
    )
    encoded = msgspec.json.encode(record)
    decoded = msgspec.json.decode(encoded, type=ProbeRecord)
    assert decoded == record
    # JSONL consumers read plain dicts — spot-check the H1 field survives.
    as_dict = json.loads(encoded)
    assert as_dict["last_prompt_position_actions"] == {"16": "run", "17": "run"}
    assert as_dict["generation"]["finish_empty"] is True
    assert as_dict["model_commit_hash"].startswith("2ce73595")


def test_run_manifest_roundtrip() -> None:
    manifest = RunManifest(
        model="xuan-luo/FlexiDepth-Llama-3-8B-Instruct",
        revision_requested="2ce73595ad0467fedd539c113e5b9deed046df32",
        resolved_commit_hash="2ce73595ad0467fedd539c113e5b9deed046df32",
        dtype="bfloat16",
        prompt_protocol="frozen-input-ids",
        transformers_version="4.57.1",
        torch_version="2.11.0+cu130",
        runner_git_commit="deadbeef",
        runner_sha256="cd" * 32,
        probe_items_sha256="ef" * 32,
        requests_jsonl_sha256=None,
        dataset_source=None,
        generation_config={"do_sample": False, "num_beams": 1, "eos_token_id": [128001, 128009]},
        started_utc="2026-08-18T00:00:00+00:00",
        finished_utc="2026-08-18T00:01:00+00:00",
        args={"topk": "20"},
        items=[
            ItemStatus(
                target_id="gsm8k:test:7", ok=True, consistent=True, finish_empty=False
            )
        ],
        n_items=1,
        n_consistency_failures=0,
        n_errors=0,
        n_empty=0,
    )
    formatted = msgspec.json.format(msgspec.json.encode(manifest), indent=2)
    as_dict = json.loads(formatted)
    assert as_dict["generation_config"]["num_beams"] == 1
    assert msgspec.json.decode(formatted, type=RunManifest) == manifest


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
