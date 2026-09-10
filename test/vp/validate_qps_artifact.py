#!/usr/bin/env python3
"""Reconcile VP serving artifacts without discarding completed requests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_METRIC_ACCOUNTING_VERSION = 5
PER_REQUEST_FIELDS = (
    "request_ids",
    "logical_request_ids",
    "transport_request_ids",
    "prompt_sha256s",
    "generation_policy_sha256s",
    "declared_input_lens",
    "server_reported_input_lens",
    "input_lens",
    "input_len_sources",
    "requested_output_lens",
    "server_reported_output_lens",
    "raw_output_lens",
    "raw_output_ids",
    "raw_output_id_sources",
    "retokenized_output_lens",
    "retokenized_output_ids",
    "output_ids",
    "output_id_sources",
    "output_lens",
    "output_len_sources",
    "finish_reasons",
    "e2e_latencies",
    "successes",
    "ttfts",
    "itls",
    "generated_texts",
    "errors",
)


def read_last_record(path: Path) -> dict[str, Any]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"benchmark artifact is empty: {path}")
    return records[-1]


def validate_record(
    record: dict[str, Any],
    expected_requests: int,
    *,
    expected_request_ids: list[str] | None = None,
    expected_prompt_sha256s: list[str] | None = None,
    expected_input_lens: list[int] | None = None,
    expected_output_lens: list[int] | None = None,
    expected_output_policies: list[str] | None = None,
    require_server_usage: bool = True,
    require_request_timing: bool = False,
    reject_cap_hits: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    version = int(record.get("metric_accounting_version") or 0)
    if version < REQUIRED_METRIC_ACCOUNTING_VERSION:
        errors.append(
            f"metric_accounting_version={version}, expected at least "
            f"{REQUIRED_METRIC_ACCOUNTING_VERSION}"
        )

    arrays: dict[str, list[Any]] = {}
    for field in PER_REQUEST_FIELDS:
        value = record.get(field)
        if not isinstance(value, list):
            errors.append(f"{field} is missing or is not a list")
            arrays[field] = []
        else:
            arrays[field] = value
            if len(value) != expected_requests:
                errors.append(
                    f"{field} has {len(value)} rows, expected {expected_requests}"
                )

    submitted = int(record.get("submitted") or 0)
    started = int(record.get("started") or 0)
    completed = int(record.get("completed") or 0)
    reported = int(record.get("reported") or 0)
    if submitted != expected_requests:
        errors.append(f"submitted={submitted}, expected {expected_requests}")
    if started != expected_requests:
        errors.append(f"started={started}, expected {expected_requests}")
    if completed != expected_requests:
        errors.append(f"completed={completed}, expected {expected_requests}")
    if reported != expected_requests:
        errors.append(f"reported={reported}, expected {expected_requests}")
    if arrays["successes"] and not all(arrays["successes"]):
        errors.append("one or more requests have success=false")
    if any(str(value or "") for value in arrays["errors"]):
        errors.append("one or more requests contain an error")

    timing_rows = 0
    if require_request_timing:
        start_unix_s = record.get("benchmark_start_unix_s")
        starts = record.get("request_start_offsets_s")
        ends = record.get("request_end_offsets_s")
        if (
            not isinstance(start_unix_s, (int, float))
            or not math.isfinite(start_unix_s)
            or start_unix_s <= 0
        ):
            errors.append("benchmark_start_unix_s is missing or invalid")
        if not isinstance(starts, list) or len(starts) != expected_requests:
            errors.append("request_start_offsets_s does not match expected requests")
            starts = []
        if not isinstance(ends, list) or len(ends) != expected_requests:
            errors.append("request_end_offsets_s does not match expected requests")
            ends = []
        for index, (start, end) in enumerate(zip(starts, ends)):
            if (
                not isinstance(start, (int, float))
                or not math.isfinite(start)
                or start < 0
            ):
                errors.append(f"request {index}: invalid start offset {start!r}")
                continue
            if (
                not isinstance(end, (int, float))
                or not math.isfinite(end)
                or end < start
            ):
                errors.append(f"request {index}: invalid end offset {end!r}")
                continue
            timing_rows += 1

    request_ids = arrays["request_ids"]
    logical_request_ids = arrays["logical_request_ids"]
    transport_request_ids = arrays["transport_request_ids"]
    prompt_sha256s = arrays["prompt_sha256s"]
    generation_policy_sha256s = arrays["generation_policy_sha256s"]
    declared_input = arrays["declared_input_lens"]
    server_input = arrays["server_reported_input_lens"]
    effective_input = arrays["input_lens"]
    input_sources = arrays["input_len_sources"]
    requested = arrays["requested_output_lens"]
    server = arrays["server_reported_output_lens"]
    raw = arrays["raw_output_lens"]
    raw_output_ids = arrays["raw_output_ids"]
    raw_output_id_sources = arrays["raw_output_id_sources"]
    retokenized = arrays["retokenized_output_lens"]
    retokenized_output_ids = arrays["retokenized_output_ids"]
    output_ids = arrays["output_ids"]
    output_id_sources = arrays["output_id_sources"]
    effective = arrays["output_lens"]
    sources = arrays["output_len_sources"]
    finish_reasons = arrays["finish_reasons"]

    for index, (e2e, ttft, itls) in enumerate(
        zip(arrays["e2e_latencies"], arrays["ttfts"], arrays["itls"])
    ):
        if not isinstance(e2e, (int, float)) or not math.isfinite(e2e) or e2e <= 0:
            errors.append(f"request {index}: invalid E2E latency {e2e!r}")
        if (
            not isinstance(ttft, (int, float))
            or not math.isfinite(ttft)
            or ttft < 0
            or (isinstance(e2e, (int, float)) and ttft > e2e)
        ):
            errors.append(f"request {index}: invalid TTFT {ttft!r}")
        if not isinstance(itls, list) or any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in itls
        ):
            errors.append(f"request {index}: invalid ITL sequence")

    if any(not isinstance(value, str) or not value for value in request_ids):
        errors.append("one or more request_ids are missing or invalid")
    if len(set(request_ids)) != len(request_ids):
        errors.append("request_ids are not unique")
    if logical_request_ids != request_ids:
        errors.append("logical_request_ids do not match request_ids")
    if any(
        not isinstance(value, str) or not value for value in transport_request_ids
    ):
        errors.append("one or more transport_request_ids are missing or invalid")
    if len(set(transport_request_ids)) != len(transport_request_ids):
        errors.append("transport_request_ids are not unique")
    if transport_request_ids != logical_request_ids:
        errors.append("transport_request_ids do not match logical_request_ids")
    for field, values in (
        ("prompt_sha256s", prompt_sha256s),
        ("generation_policy_sha256s", generation_policy_sha256s),
    ):
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in values
        ):
            errors.append(f"{field} contains an invalid SHA-256 digest")
    if expected_request_ids is not None and request_ids != expected_request_ids:
        errors.append("request_ids do not exactly match frozen metadata order")
    if (
        expected_prompt_sha256s is not None
        and prompt_sha256s != expected_prompt_sha256s
    ):
        errors.append("prompt_sha256s do not exactly match frozen metadata order")
    for index, (token_ids, token_count) in enumerate(
        zip(retokenized_output_ids, retokenized)
    ):
        if not isinstance(token_ids, list) or any(
            not isinstance(token_id, int) for token_id in token_ids
        ):
            errors.append(f"request {index}: invalid retokenized output IDs")
        elif len(token_ids) != token_count:
            errors.append(
                f"request {index}: retokenized output ID count does not match length"
            )
    for index, (token_ids, token_count, source) in enumerate(
        zip(raw_output_ids, raw, raw_output_id_sources)
    ):
        if not isinstance(token_ids, list) or any(
            not isinstance(token_id, int) for token_id in token_ids
        ):
            errors.append(f"request {index}: invalid raw output IDs")
        elif not isinstance(token_count, int) or token_count < 0:
            errors.append(f"request {index}: invalid raw output length")
        elif len(token_ids) != token_count:
            errors.append(
                f"request {index}: raw output ID count does not match length"
            )
        if source != "server_output_ids":
            errors.append(
                f"request {index}: raw output ID source {source!r} is not "
                "server_output_ids"
            )
    if output_ids != raw_output_ids:
        errors.append("output_ids do not match raw_output_ids")
    if any(source != "server_output_ids" for source in output_id_sources):
        errors.append("one or more output_id_sources are not server_output_ids")

    for index in range(
        min(map(len, (declared_input, server_input, effective_input, input_sources)))
    ):
        declared_len = declared_input[index]
        server_len = server_input[index]
        effective_len = effective_input[index]
        source = input_sources[index]
        if not isinstance(declared_len, int) or declared_len <= 0:
            errors.append(f"request {index}: invalid declared input {declared_len!r}")
            continue
        if expected_input_lens is not None and declared_len != expected_input_lens[index]:
            errors.append(
                f"request {index}: declared input {declared_len} does not match "
                f"frozen metadata {expected_input_lens[index]}"
            )
        if require_server_usage and not isinstance(server_len, int):
            errors.append(f"request {index}: missing server prompt-token usage")
        if isinstance(server_len, int):
            if server_len <= 0:
                errors.append(f"request {index}: invalid server input {server_len}")
            if server_len != declared_len:
                errors.append(
                    f"request {index}: server input {server_len} does not match "
                    f"declared input {declared_len}"
                )
            if effective_len != server_len:
                errors.append(
                    f"request {index}: effective input {effective_len} does not "
                    f"match server usage {server_len}"
                )
        if source != "server_usage" and require_server_usage:
            errors.append(
                f"request {index}: effective input source {source!r} is not server_usage"
            )

    for index in range(
        min(map(len, (requested, server, raw, retokenized, effective)))
    ):
        requested_len = requested[index]
        server_len = server[index]
        raw_len = raw[index]
        retokenized_len = retokenized[index]
        effective_len = effective[index]
        if not isinstance(requested_len, int) or requested_len <= 0:
            errors.append(f"request {index}: invalid requested output {requested_len!r}")
            continue
        if expected_output_lens is not None and requested_len != expected_output_lens[index]:
            errors.append(
                f"request {index}: requested output {requested_len} does not match "
                f"frozen metadata {expected_output_lens[index]}"
            )
        if require_server_usage and not isinstance(server_len, int):
            errors.append(f"request {index}: missing server completion-token usage")
        if not isinstance(raw_len, int) or raw_len < 0:
            errors.append(f"request {index}: invalid raw output length {raw_len!r}")
        if not isinstance(retokenized_len, int) or retokenized_len < 0:
            errors.append(f"request {index}: invalid retokenized length {retokenized_len!r}")
        if not isinstance(effective_len, int) or effective_len < 0:
            errors.append(f"request {index}: invalid effective length {effective_len!r}")
            continue
        if effective_len > requested_len:
            errors.append(
                f"request {index}: effective length {effective_len} exceeds "
                f"requested maximum {requested_len}"
            )
        if isinstance(server_len, int):
            if server_len > requested_len:
                errors.append(
                    f"request {index}: server length {server_len} exceeds "
                    f"requested maximum {requested_len}"
                )
            if effective_len != server_len:
                errors.append(
                    f"request {index}: effective length {effective_len} does not "
                    f"match server usage {server_len}"
                )
            if raw_len != server_len:
                errors.append(
                    f"request {index}: raw output length {raw_len} does not "
                    f"match server usage {server_len}"
                )
        if isinstance(raw_len, int) and effective_len != raw_len:
            errors.append(
                f"request {index}: effective length {effective_len} does not "
                f"match raw output length {raw_len}"
            )

    if require_server_usage and any(source != "server_usage" for source in sources):
        errors.append("one or more effective output lengths do not use server usage")
    missing_finish = [
        index for index, reason in enumerate(finish_reasons) if reason is None
    ]
    if missing_finish:
        errors.append(f"missing finish reason for {len(missing_finish)} requests")
    # [G3, Codex review 2 P1-4] Only ABSENT finish reasons were rejected, so a request that
    # finished with `error`, `abort`, `content_filter` or "" passed the zero-errors gate --
    # the gate accepted the very failures it exists to catch. Exact-budget policies caught
    # these incidentally through their required `length` finish; natural-length and
    # calibration cells were exposed.
    #
    # Only two outcomes are a request that RAN: `stop` (EOS honored, natural lane) and
    # `length` (cap or ignore_eos budget reached). Anything else is a failed request.
    _FINISHED_OK = {"stop", "length"}
    bad_finish = sorted(
        {
            str(reason)
            for reason in finish_reasons
            if reason is not None and str(reason) not in _FINISHED_OK
        }
    )
    if bad_finish:
        counts = {
            reason: sum(1 for r in finish_reasons if str(r) == reason)
            for reason in bad_finish
        }
        errors.append(
            "failed/filtered finish reasons present (a request that did not finish is a "
            f"failed request, not a datum): {counts}"
        )
    # [D-628] ZERO-EMPTY HARD GATE (owner order, 2026-09-10). A request that returns no text
    # is LOST OUTPUT, not a low score: it silently deflates quality and inflates
    # throughput-per-token, and it survived from D-066 (2026-08-09) to now because nothing
    # asserted it. finish_reason cannot catch it -- `stop`/`matched: 128009` (<|eot_id|>) is
    # also the NORMAL healthy ending, 82% of a clean run -- so the body must be checked.
    #
    # Equal-work rows are EXEMPT and must be: `ignore_eos` fills the budget, so a first-token
    # EOS is invisible there. That exemption is exactly why every headline performance cell
    # missed this, so exempting them here is not a loophole -- it records where the gate has
    # no power, and the natural lane is where it bites.
    generated_texts = arrays["generated_texts"]
    empty_rows = [
        index
        for index, text in enumerate(generated_texts)
        if not str(text or "").strip()
        and (
            expected_output_policies is None
            or index >= len(expected_output_policies)
            or expected_output_policies[index] != "production_max_equal_work"
        )
    ]
    if empty_rows:
        errors.append(
            f"{len(empty_rows)} natural-lane requests returned EMPTY text "
            f"(first rows {empty_rows[:10]}). Lost output, not a datum. Check the prompt "
            "protocol first: raw few-shot rendering reproduces this at 42-52% while the chat "
            "protocol measures 0.0% on the same tree (D-628)."
        )

    length_finish_indices = [
        index for index, reason in enumerate(finish_reasons) if reason == "length"
    ]
    prefill_length_indices = []
    equal_work_length_indices = []
    context_limit_indices = []
    if expected_output_policies is not None:
        prefill_length_indices = [
            index
            for index in length_finish_indices
            if expected_output_policies[index] == "prefill_probe_one_token"
            and index < len(requested)
            and index < len(effective)
            and requested[index] == 1
            and effective[index] == 1
        ]
        equal_work_length_indices = [
            index
            for index in length_finish_indices
            if expected_output_policies[index] == "production_max_equal_work"
            and index < len(requested)
            and index < len(effective)
            and requested[index] == effective[index]
        ]
        context_limit_indices = [
            index
            for index in length_finish_indices
            if expected_output_policies[index] == "remaining_model_context"
            and index < len(requested)
            and index < len(effective)
            and requested[index] == effective[index]
        ]
        for index, policy in enumerate(expected_output_policies):
            if policy != "production_max_equal_work":
                continue
            if index >= len(requested) or index >= len(effective):
                continue
            if requested[index] != effective[index]:
                errors.append(
                    f"request {index}: equal-work output did not reach its exact budget"
                )
            if index >= len(finish_reasons) or finish_reasons[index] != "length":
                errors.append(
                    f"request {index}: equal-work output did not finish by length"
                )
    intentional_length_indices = sorted(
        set(prefill_length_indices)
        | set(equal_work_length_indices)
        | set(context_limit_indices)
    )
    intentional_length_set = set(intentional_length_indices)
    cap_hit_indices = [
        index
        for index in length_finish_indices
        if index not in intentional_length_set
    ]
    if reject_cap_hits and cap_hit_indices:
        errors.append(
            f"non-intentional length finish by {len(cap_hit_indices)} requests"
        )

    recorded_total_input = record.get("total_input_tokens")
    if effective_input and (
        not isinstance(recorded_total_input, int)
        or recorded_total_input != sum(effective_input)
    ):
        errors.append("total_input_tokens does not equal sum(input_lens)")
    recorded_total_output = record.get("total_output_tokens")
    if effective and (
        not isinstance(recorded_total_output, int)
        or recorded_total_output != sum(effective)
    ):
        errors.append("total_output_tokens does not equal sum(output_lens)")
    recorded_retokenized_total = record.get("total_output_tokens_retokenized")
    if retokenized and (
        not isinstance(recorded_retokenized_total, int)
        or recorded_retokenized_total != sum(retokenized)
    ):
        errors.append(
            "total_output_tokens_retokenized does not equal "
            "sum(retokenized_output_lens)"
        )
    duration = float(record.get("duration") or 0.0)
    if duration <= 0:
        errors.append(f"invalid benchmark duration {duration}")
    else:
        expected_request_throughput = expected_requests / duration
        reported_request_throughput = float(record.get("request_throughput") or 0.0)
        if not math.isclose(
            reported_request_throughput,
            expected_request_throughput,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append("request_throughput does not reconcile with completed requests")
    if duration > 0 and effective_input:
        expected_input_throughput = sum(effective_input) / duration
        reported_input_throughput = float(record.get("input_throughput") or 0.0)
        if not math.isclose(
            reported_input_throughput,
            expected_input_throughput,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append("input_throughput does not reconcile with actual input tokens")
    if duration > 0 and effective:
        expected_throughput = sum(effective) / duration
        reported_throughput = float(record.get("output_throughput") or 0.0)
        if not math.isclose(
            reported_throughput, expected_throughput, rel_tol=1e-9, abs_tol=1e-9
        ):
            errors.append(
                "output_throughput does not reconcile with actual output tokens"
            )

    reported_input_rows = record.get("server_input_usage_reported")
    actual_input_rows = sum(isinstance(value, int) for value in server_input)
    if reported_input_rows != actual_input_rows:
        errors.append("server_input_usage_reported does not match per-request rows")
    reported_output_rows = record.get("server_usage_reported")
    actual_output_rows = sum(isinstance(value, int) for value in server)
    if reported_output_rows != actual_output_rows:
        errors.append("server_usage_reported does not match per-request rows")
    recorded_raw_rows = record.get("raw_output_ids_reported")
    actual_raw_rows = sum(
        source == "server_output_ids" for source in raw_output_id_sources
    )
    if recorded_raw_rows != actual_raw_rows:
        errors.append("raw_output_ids_reported does not match per-request rows")
    recorded_raw_total = record.get("total_output_tokens_raw")
    actual_raw_total = sum(
        value for value in raw if isinstance(value, int) and value >= 0
    )
    if recorded_raw_total != actual_raw_total:
        errors.append("total_output_tokens_raw does not match per-request rows")

    usage_mismatch = []
    retokenization_mismatch_indices = []
    for index in range(min(len(server), len(retokenized))):
        if isinstance(server[index], int) and isinstance(retokenized[index], int):
            delta = server[index] - retokenized[index]
            usage_mismatch.append(delta)
            if delta != 0:
                retokenization_mismatch_indices.append(index)

    return {
        "status": "passed" if not errors else "failed",
        "metric_accounting_version": version,
        "expected_requests": expected_requests,
        "submitted": submitted,
        "started": started,
        "completed": completed,
        "reported": reported,
        "request_identity_rows": sum(
            isinstance(value, str) and bool(value) for value in request_ids
        ),
        "transport_identity_rows": sum(
            isinstance(value, str) and bool(value)
            for value in transport_request_ids
        ),
        "prompt_hash_rows": sum(
            isinstance(value, str) and len(value) == 64 for value in prompt_sha256s
        ),
        "generation_policy_hash_rows": sum(
            isinstance(value, str) and len(value) == 64
            for value in generation_policy_sha256s
        ),
        "request_timing_required": require_request_timing,
        "request_timing_rows": timing_rows,
        "server_input_usage_rows": actual_input_rows,
        "server_usage_rows": actual_output_rows,
        "raw_output_id_rows": actual_raw_rows,
        "retokenized_fallback_rows": sum(
            source == "retokenized_fallback" for source in sources
        ),
        "missing_finish_reason_rows": len(missing_finish),
        "length_finish_count": len(length_finish_indices),
        "length_finish_indices": length_finish_indices,
        "intentional_length_completion_count": len(intentional_length_indices),
        "intentional_length_completion_indices": intentional_length_indices,
        "prefill_length_completion_count": len(prefill_length_indices),
        "prefill_length_completion_indices": prefill_length_indices,
        "equal_work_length_completion_count": len(equal_work_length_indices),
        "equal_work_length_completion_indices": equal_work_length_indices,
        "context_limit_completion_count": len(context_limit_indices),
        "context_limit_completion_indices": context_limit_indices,
        "cap_hit_count": len(cap_hit_indices),
        "cap_hit_indices": cap_hit_indices,
        "length_finish_policy": "reject" if reject_cap_hits else "tag_only",
        "server_minus_retokenized_min": min(usage_mismatch) if usage_mismatch else None,
        "server_minus_retokenized_max": max(usage_mismatch) if usage_mismatch else None,
        "retokenization_mismatch_count": len(retokenization_mismatch_indices),
        "retokenization_mismatch_indices": retokenization_mismatch_indices,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-jsonl", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-retokenized-fallback", action="store_true")
    parser.add_argument("--require-request-timing", action="store_true")
    parser.add_argument(
        "--reject-length-finishes",
        action="store_true",
        help="Opt-in diagnostic: make non-intentional length finishes fail validation.",
    )
    parser.add_argument("--allow-cap-hits", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    record = read_last_record(args.bench_jsonl)
    metadata = []
    with args.metadata.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    if len(metadata) < args.expected_requests:
        raise ValueError(
            f"metadata has {len(metadata)} rows, expected at least "
            f"{args.expected_requests}"
        )
    expected = metadata[: args.expected_requests]
    audit = validate_record(
        record,
        args.expected_requests,
        expected_request_ids=[str(row["request_id"]) for row in expected],
        expected_prompt_sha256s=[str(row["prompt_sha256"]) for row in expected],
        expected_input_lens=[int(row["prompt_len"]) for row in expected],
        expected_output_lens=[int(row["requested_output_len"]) for row in expected],
        expected_output_policies=[str(row["output_policy"]) for row in expected],
        require_server_usage=not args.allow_retokenized_fallback,
        require_request_timing=args.require_request_timing,
        reject_cap_hits=args.reject_length_finishes and not args.allow_cap_hits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
