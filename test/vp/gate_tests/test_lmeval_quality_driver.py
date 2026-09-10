#!/usr/bin/env python3
"""The quality lane's protocol comes from the perf suite, and cannot drift from it.

The chat-vs-raw question has already had to be re-answered by hand more than once, and
getting it wrong is not a small error: forcing chat flags onto coqa CREATES a mismatch
between the lanes rather than fixing one, and running gsm8k raw reproduces the
FlexiDepth checkpoint's 51.6 %-empty collapse (D-628/D-632). So the driver reads
`prompt_kind` out of the frozen suite instead of carrying its own table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

VP = Path(__file__).resolve().parents[1]
if str(VP) not in sys.path:
    sys.path.insert(0, str(VP))

import run_lmeval_quality as driver  # noqa: E402


def _suite(tmp_path: Path, workload: str, kinds: list[str]) -> Path:
    (tmp_path / f"{workload}.metadata.jsonl").write_text(
        "".join(json.dumps({"prompt_kind": k}) + "\n" for k in kinds)
    )
    return tmp_path


def _args(**kw) -> argparse.Namespace:
    base = dict(
        base_url="", tokenizer="", num_concurrent=16, hf_model="", hf_dtype="",
        trust_remote_code=False, lmeval_python="python3", output_dir=Path("/tmp/out"),
        batch_size="8",
    )
    base.update(kw)
    return argparse.Namespace(**base)


# --- protocol is read, not chosen ---


def test_chat_suite_yields_chat_protocol(tmp_path):
    assert driver._prompt_kind(_suite(tmp_path, "gsm8k", ["chat_messages"] * 3), "gsm8k") == "chat_messages"


def test_raw_suite_yields_raw_protocol(tmp_path):
    assert driver._prompt_kind(_suite(tmp_path, "coqa", ["raw_completion"] * 3), "coqa") == "raw_completion"


def test_mixed_prompt_kinds_is_refused(tmp_path):
    suite = _suite(tmp_path, "gsm8k", ["chat_messages", "raw_completion"])
    with pytest.raises(SystemExit, match="mixes prompt kinds"):
        driver._prompt_kind(suite, "gsm8k")


def test_missing_suite_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no frozen suite"):
        driver._prompt_kind(tmp_path, "gsm8k")


# --- command construction ---


def test_chat_arm_uses_chat_endpoint_and_flags():
    cmd = driver._lmeval_command(
        _args(base_url="http://h:1/", tokenizer="/m"), "gsm8k", "chat_messages"
    )
    assert "--apply_chat_template" in cmd and "--fewshot_as_multiturn" in cmd
    assert "local-chat-completions" in cmd
    # lm-eval's chat template cannot be served over /v1/completions (D-233).
    assert any("/v1/chat/completions" in part for part in cmd)


def test_raw_arm_gets_no_chat_flags():
    cmd = driver._lmeval_command(
        _args(base_url="http://h:1", tokenizer="/m"), "coqa", "raw_completion"
    )
    assert "--apply_chat_template" not in cmd and "--fewshot_as_multiturn" not in cmd
    assert "local-completions" in cmd
    assert not any("/v1/chat/completions" in part for part in cmd)


def test_native_arm_is_plain_hf_with_its_dtype():
    cmd = driver._lmeval_command(
        _args(hf_model="/ckpt", hf_dtype="float16", trust_remote_code=True),
        "gsm8k",
        "chat_messages",
    )
    assert cmd[cmd.index("--model") + 1] == "hf"
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "pretrained=/ckpt" in model_args
    assert "dtype=float16" in model_args
    assert "trust_remote_code=True" in model_args


def test_log_samples_is_always_requested():
    """Without it there is no samples_*.jsonl, so nothing can be gated."""
    for kind in ("chat_messages", "raw_completion"):
        cmd = driver._lmeval_command(_args(hf_model="/m"), "coqa", kind)
        assert "--log_samples" in cmd


def test_no_limit_is_ever_emitted():
    """Quality numbers require the full split; a --limit run is a diagnostic."""
    for kind in ("chat_messages", "raw_completion"):
        assert "--limit" not in driver._lmeval_command(_args(hf_model="/m"), "gsm8k", kind)


def test_bbh_uses_the_cot_task_not_the_broken_one():
    """`bbh_fewshot` scores 0.0 through a leading-space artifact (D-616)."""
    assert driver.TASKS["bbh_cot"] == "bbh_cot_fewshot"
    assert "bbh_fewshot" not in driver.TASKS.values()


def test_suite_name_defaults_to_workload_but_is_overridable(tmp_path):
    """The built suites are `coqa.d179`, not `coqa` -- the workload name is NOT the file
    prefix. Reading only the workload made the driver refuse every real suite on the box."""
    _suite(tmp_path, "coqa.d179", ["raw_completion"])
    assert driver._prompt_kind(tmp_path, "coqa.d179") == "raw_completion"
    with pytest.raises(SystemExit) as exc:
        driver._prompt_kind(tmp_path, "coqa")
    # The refusal must say what IS there, or the next person repeats the same guess.
    assert "coqa.d179" in str(exc.value)


def test_native_arm_gets_a_batch_size_and_served_does_not():
    """lm-eval's default batch_size is 1, which ran a native arm ~15x slower than a served
    one AND silently differed from the batch_size=8 of D-631/D-632's reference runs. Served
    arms must NOT get it -- the server does the batching."""
    native = driver._lmeval_command(_args(hf_model="/m", batch_size="8"), "gsm8k", "chat_messages")
    assert native[native.index("--batch_size") + 1] == "8"
    served = driver._lmeval_command(
        _args(base_url="http://h:1", tokenizer="/m", batch_size="8"), "gsm8k", "chat_messages"
    )
    assert "--batch_size" not in served
