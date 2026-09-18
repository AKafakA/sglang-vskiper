#!/usr/bin/env python3
"""Direct CPU gate for fair-output and paired-arrival evaluation contracts."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from analyze_termination import analyze_record, classify_repetition
from build_fair_output_workload import build_equal_work_rows, build_natural_rows
from launch_qps_server import _validate_flexidepth_launch_mode
from run_qps_evaluation import _materialize_arrival_schedule
from vskipper.experiments.request_identity import generation_policy_sha256


def _row(request_id: str, *, prefill: bool = False) -> tuple[dict, dict]:
    prompt = [1, 2, 3]
    prompt_hash = hashlib.sha256(
        json.dumps(prompt, separators=(",", ":")).encode()
    ).hexdigest()
    request = {
        "request_id": request_id,
        "prompt": prompt,
        "prompt_len": 3,
        "output_len": 1 if prefill else 8,
        "extra_request_body": {
            "temperature": 0.0,
            "ignore_eos": False,
            **({} if prefill else {"stop": ["done"]}),
        },
    }
    metadata = {
        "index": 0,
        "request_id": request_id,
        "source_request_id": request_id,
        "replica_index": 0,
        "dataset": "mmlu" if prefill else "gsm8k",
        "phase": "prefill" if prefill else "decode",
        "metric": "selection_prefill_probe" if prefill else "gsm8k_strict_match",
        "gold": "220",
        "reference_output_len": 1,
        "requested_output_len": request["output_len"],
        "prompt_len": 3,
        "context_length": 16,
        "output_policy": (
            "prefill_probe_one_token" if prefill else "production_watchdog"
        ),
        "fixed_output_tokens": None,
        "task_reference_max_output_len": 1 if prefill else 8,
        "protocol_schema_version": 4,
        "protocol_suite_id": "flexidepth-lm-eval-0.4.9.1-serving-v1",
        "protocol_id": (
            "lm-eval-0.4.9.1:mmlu-v1:5shot-multiturn-prefill-probe"
            if prefill
            else "lm-eval-0.4.9.1:gsm8k-v3:5shot-multiturn"
        ),
        "quality_semantics": (
            "paper_prompt_prefill_probe" if prefill else "paper_exact"
        ),
        "prompt_kind": "chat_messages",
        "quality_eligible": not prefill,
        "sampling_policy": {},
        "prompt_sha256": prompt_hash,
        "evaluator_data": {},
        "model": "model",
        "model_revision": "revision",
    }
    return request, metadata


def _write_production_artifact(
    path: Path,
    requests: list[dict],
    output_lengths: list[int],
    finish_reasons: list[str],
) -> None:
    record = {
        "request_ids": [row["request_id"] for row in requests],
        "successes": [True] * len(requests),
        "server_reported_output_lens": output_lengths,
        "raw_output_lens": output_lengths,
        "raw_output_ids": [
            list(range(output_len)) for output_len in output_lengths
        ],
        "raw_output_id_sources": ["server_output_ids"] * len(requests),
        "finish_reasons": finish_reasons,
        "generation_policy_sha256s": [
            generation_policy_sha256(
                backend="sglang",
                requested_output_len=int(request["output_len"]),
                request_body=request["extra_request_body"],
            )
            for request in requests
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="vp-fair-eval-") as directory:
        root = Path(directory)
        decode_request, decode_metadata = _row("gsm8k:1")
        prefill_request, prefill_metadata = _row("mmlu:1", prefill=True)
        prefill_metadata["index"] = 1
        natural_requests, natural_metadata, natural_summary = build_natural_rows(
            [decode_request, prefill_request],
            [decode_metadata, prefill_metadata],
            0.3,
        )
        assert natural_requests[0]["output_len"] == 13
        assert (
            natural_requests[0]["extra_request_body"]["frequency_penalty"] == 0.3
        )
        assert natural_requests[0]["extra_request_body"]["ignore_eos"] is False
        assert natural_metadata[0]["output_policy"] == "remaining_model_context"
        assert natural_requests[1]["output_len"] == 1
        assert natural_summary["decode_rows"] == 1

        production_artifacts = []
        for rep, length in enumerate((5, 9, 7), start=1):
            path = root / f"production-rep{rep}.jsonl"
            _write_production_artifact(
                path,
                natural_requests,
                [length, 1],
                ["stop", "length"],
            )
            production_artifacts.append(path)
        equal_requests, equal_metadata, equal_summary = build_equal_work_rows(
            natural_requests,
            natural_metadata,
            production_artifacts,
        )
        assert equal_requests[0]["output_len"] == 9
        assert equal_requests[0]["extra_request_body"]["ignore_eos"] is True
        assert "stop" not in equal_requests[0]["extra_request_body"]
        assert equal_metadata[0]["equal_work_production_output_lens"] == [5, 9, 7]
        assert equal_metadata[0]["quality_eligible"] is False
        assert equal_summary["minimum_output_length"] == 9
        assert equal_summary["maximum_output_length"] == 9

        source = root / "requests.jsonl"
        source.write_text(
            "".join(
                json.dumps(
                    {
                        "request_id": f"gsm8k:{index}",
                        "prompt": [index],
                        "output_len": 8,
                    }
                )
                + "\n"
                for index in range(8)
            ),
            encoding="utf-8",
        )
        left = _materialize_arrival_schedule(
            source, root / "left.jsonl", 8, 3.5, 42
        )
        right = _materialize_arrival_schedule(
            source, root / "right.jsonl", 8, 3.5, 42
        )
        other_rep = _materialize_arrival_schedule(
            source, root / "other-rep.jsonl", 8, 3.5, 43
        )
        assert left["sha256"] == right["sha256"]
        assert left["request_ids_sha256"] == right["request_ids_sha256"]
        assert left["sha256"] != other_rep["sha256"]

        repeated = classify_repetition(
            "\n".join(["reasoning", "#### 220", "#### 220", "#### 220"]),
            list(range(40)),
        )
        assert repeated["loop_any"] is True
        assert repeated["answer_marker_loop"]["detected"] is True
        normal = classify_repetition(
            "We calculate the answer carefully.\n#### 220",
            list(range(80)),
        )
        assert normal["loop_any"] is False
        analysis = analyze_record(
            {
                "request_ids": ["gsm8k:1"],
                "generated_texts": ["work\n#### 220"],
                "raw_output_ids": [list(range(12))],
                "finish_reasons": ["length"],
                "successes": [True],
            },
            [natural_metadata[0]],
        )
        assert analysis["rows"][0]["context_limit_hit"] is True
        assert analysis["rows"][0]["quality_score"] == 1.0

        _validate_flexidepth_launch_mode(
            "graphless_overlap",
            [],
            {
                "SGLANG_FD_EXECUTION_MODE": "full_graph",
                "SGLANG_VP_V4_DEVICE_REBATCHING_EAGER_SEMANTIC": "1",
                "SGLANG_FD_PARITY_TRACE_RID": "request-0",
                "SGLANG_FD_PARITY_TRACE_DIR": str(root / "trace"),
            },
            {
                "runtime": {
                    "v4": {
                        "execution": {"enable_device_rebatching": True}
                    }
                }
            },
        )
        _validate_flexidepth_launch_mode(
            "graphless_overlap",
            [],
            {
                "SGLANG_FD_EXECUTION_MODE": "full_graph",
                "SGLANG_VP_V4_DEVICE_REBATCHING_EAGER_SEMANTIC": "1",
            },
            {
                "runtime": {
                    "v4": {
                        "execution": {"enable_device_rebatching": True}
                    }
                }
            },
        )

    print(
        "PASS fair-output, arrival-replay, termination, and eager trace "
        "contracts"
    )


if __name__ == "__main__":
    main()
