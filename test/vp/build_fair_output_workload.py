#!/usr/bin/env python3
"""Materialize natural-context and production-max equal-work workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from build_labeled_workload import (
    DEFAULT_FREQUENCY_PENALTY,
    declared_frequency_penalty,
    validate_frozen_output_policy,
)
from labeled_workload import read_jsonl, write_jsonl
from sglang.benchmark.request_identity import (
    SGLANG_NATIVE_BACKENDS,
    generation_policy_sha256,
    split_sglang_native_request_body,
)


def _client_effective_body(body: dict[str, Any], backend: str) -> dict[str, Any]:
    """Reproduce the benchmark client's effective generation body.

    The client hashes the body it actually SENDS (native sampling split +
    defaults + stream), not the raw workload body — mirror of
    sglang/benchmark/serving.py's construction under the runner's fixed
    invocation (--disable-ignore-eos on, streaming on).
    """
    if backend in SGLANG_NATIVE_BACKENDS:
        native_body, native_sampling = split_sglang_native_request_body(dict(body))
        native_sampling.setdefault("temperature", 0.0)
        native_sampling.setdefault("ignore_eos", False)
        native_body["sampling_params"] = native_sampling
        native_body.setdefault("stream", True)
        return native_body
    effective = dict(body)
    effective.setdefault("temperature", 0.0)
    effective.setdefault("ignore_eos", False)
    effective.setdefault("stream", True)
    return effective
from validate_qps_artifact import read_last_record


NATURAL_POLICY = "remaining_model_context"
EQUAL_WORK_POLICY = "production_max_equal_work"
PREFILL_POLICY = "prefill_probe_one_token"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_penalty(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("frequency penalty must be finite and non-negative")
    return value


def _natural_body(body: dict[str, Any], penalty: float) -> dict[str, Any]:
    result = dict(body)
    result["temperature"] = 0.0
    result["top_p"] = 1.0
    result["ignore_eos"] = False
    result["frequency_penalty"] = penalty
    return result


def build_natural_rows(
    requests: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    frequency_penalty: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Restore decode rows to their full remaining context with EOS enabled."""

    penalty = _validate_penalty(frequency_penalty)
    if len(requests) != len(metadata):
        raise ValueError("request and metadata counts differ")
    output_requests: list[dict[str, Any]] = []
    output_metadata: list[dict[str, Any]] = []
    decode_rows = 0
    for request, row in zip(requests, metadata):
        frozen_request = dict(request)
        frozen_row = dict(row)
        policy = str(row.get("output_policy") or "")
        if policy == PREFILL_POLICY:
            body = dict(frozen_request.get("extra_request_body") or {})
            body.pop("frequency_penalty", None)
            body["temperature"] = 0.0
            body["top_p"] = 1.0
            body["ignore_eos"] = False
            frozen_request["output_len"] = 1
            frozen_request["extra_request_body"] = body
            frozen_row["requested_output_len"] = 1
            frozen_row["frequency_penalty"] = None
        else:
            prompt_len = int(row["prompt_len"])
            context_length = int(row["context_length"])
            remaining = context_length - prompt_len
            if remaining <= 0:
                raise ValueError(f"{row['request_id']} has no output capacity")
            frozen_request["output_len"] = remaining
            frozen_request["extra_request_body"] = _natural_body(
                dict(frozen_request.get("extra_request_body") or {}),
                penalty,
            )
            frozen_row.update(
                {
                    "requested_output_len": remaining,
                    "output_policy": NATURAL_POLICY,
                    "fixed_output_tokens": None,
                    "quality_eligible": True,
                    "frequency_penalty": penalty,
                }
            )
            decode_rows += 1
        output_requests.append(frozen_request)
        output_metadata.append(frozen_row)

    context_lengths = {int(row["context_length"]) for row in output_metadata}
    models = {str(row.get("model") or "") for row in output_metadata}
    revisions = {str(row.get("model_revision") or "") for row in output_metadata}
    if len(context_lengths) != 1 or len(models) != 1 or len(revisions) != 1:
        raise ValueError("workload does not pin one model, revision, and context")
    validate_frozen_output_policy(
        output_requests,
        output_metadata,
        next(iter(context_lengths)),
        next(iter(models)),
        next(iter(revisions)),
    )
    return output_requests, output_metadata, {
        "schema_version": 1,
        "policy": NATURAL_POLICY,
        "frequency_penalty": penalty,
        "decode_rows": decode_rows,
        "prefill_rows": len(output_requests) - decode_rows,
        "rows": len(output_requests),
    }


def _production_lengths(
    requests: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    artifacts: list[Path],
    backend: str,
) -> tuple[dict[str, list[int]], dict[str, list[str]], set[str]]:
    if not artifacts:
        raise ValueError("equal-work construction requires production artifacts")
    request_by_id = {str(row["request_id"]): request for request, row in zip(requests, metadata)}
    expected_ids = [str(row["request_id"]) for row in metadata]
    decode_ids = {
        str(row["request_id"])
        for row in metadata
        if row.get("output_policy") != PREFILL_POLICY
    }
    lengths = {request_id: [] for request_id in decode_ids}
    finish_reasons = {request_id: [] for request_id in decode_ids}
    context_hits: set[str] = set()

    required_arrays = (
        "request_ids",
        "successes",
        "server_reported_output_lens",
        "raw_output_lens",
        "raw_output_ids",
        "raw_output_id_sources",
        "finish_reasons",
        "generation_policy_sha256s",
    )
    for artifact in artifacts:
        record = read_last_record(artifact)
        arrays = {name: record.get(name) for name in required_arrays}
        if not all(isinstance(value, list) for value in arrays.values()):
            raise ValueError(f"{artifact} lacks Accounting-v5 raw-ID arrays")
        if any(len(value) != len(expected_ids) for value in arrays.values()):
            raise ValueError(f"{artifact} has incomplete Accounting-v5 arrays")
        if [str(value) for value in arrays["request_ids"]] != expected_ids:
            raise ValueError(f"{artifact} request IDs do not match frozen order")

        for index, request_id in enumerate(expected_ids):
            if arrays["successes"][index] is not True:
                raise ValueError(f"{artifact} failed request {request_id}")
            request = request_by_id[request_id]
            expected_policy_hash = generation_policy_sha256(
                backend=backend,
                requested_output_len=int(request["output_len"]),
                request_body=_client_effective_body(
                    dict(request.get("extra_request_body") or {}), backend
                ),
            )
            if arrays["generation_policy_sha256s"][index] != expected_policy_hash:
                raise ValueError(
                    f"{artifact} generation policy mismatch for {request_id}"
                )
            if request_id not in decode_ids:
                continue
            raw_ids = arrays["raw_output_ids"][index]
            raw_len = arrays["raw_output_lens"][index]
            server_len = arrays["server_reported_output_lens"][index]
            source = arrays["raw_output_id_sources"][index]
            finish_reason = arrays["finish_reasons"][index]
            if (
                not isinstance(raw_ids, list)
                or any(not isinstance(token_id, int) for token_id in raw_ids)
                or not isinstance(raw_len, int)
                or raw_len <= 0
                or raw_len != len(raw_ids)
                or server_len != raw_len
                or source != "server_output_ids"
            ):
                raise ValueError(
                    f"{artifact} has invalid raw output accounting for {request_id}"
                )
            if not isinstance(finish_reason, str) or not finish_reason:
                raise ValueError(f"{artifact} has no finish reason for {request_id}")
            # [Audit] Any non-empty string passed, so a generation that ABORTED or
            # ERRORED was harvested as if it were a natural length -- and every other arm is
            # then pinned to that truncated budget, making the whole cell's equal-work
            # reference wrong while G2 still passes. This is the sibling of the finish-reason
            # hole in validate_qps_artifact.py: the same defect, one stage upstream.
            if finish_reason not in ("stop", "length"):
                raise ValueError(
                    f"{artifact}: {request_id} finished with {finish_reason!r}; only 'stop' "
                    "(EOS) and 'length' (budget reached) are real generations. A failed "
                    "request must never calibrate an equal-work bank"
                )
            lengths[request_id].append(raw_len)
            finish_reasons[request_id].append(finish_reason)
            if finish_reason == "length":
                context_hits.add(request_id)
    return lengths, finish_reasons, context_hits


def build_equal_work_rows(
    requests: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    production_artifacts: list[Path],
    *,
    backend: str = "sglang",
    declared_penalty: float = DEFAULT_FREQUENCY_PENALTY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Freeze each decode request to max raw production work across three reps."""

    if len(requests) != len(metadata):
        raise ValueError("request and metadata counts differ")
    # The per-row `frequency_penalty` is written by THIS builder onto its own output (see
    # `_pinned_body`), and by older versions of `build_labeled_workload.py` onto natural rows.
    # The current natural builder does NOT emit it per row -- it declares the penalty once, in
    # the suite summary. Reading the row field unconditionally therefore made the equal-work
    # path fail on every freshly built natural suite (KeyError, 2026-09-04); the historical
    # banks only worked because their input suites came from the older builder.
    #
    # `declared_penalty` is the suite-level declaration, which is the authoritative record --
    # `workload_config_sha256` binds it into the build identity. Falling back to it is faithful,
    # not permissive: the per-row value, when present, must still agree with it, and a row that
    # disagrees is a real defect and still raises below.
    penalties = {
        float(row.get("frequency_penalty", declared_penalty))
        for row in metadata
        if row.get("output_policy") != PREFILL_POLICY
    }
    if len(penalties) != 1:
        raise ValueError("natural workload must pin one decode frequency penalty")
    penalty = _validate_penalty(next(iter(penalties)))
    if declared_penalty is not None and penalty != _validate_penalty(declared_penalty):
        raise ValueError(
            f"rows carry frequency_penalty={penalty} but the suite summary declares "
            f"{declared_penalty}; a generation penalty may only come from the recorded "
            "builder input"
        )
    lengths, finish_reasons, context_hits = _production_lengths(
        requests, metadata, production_artifacts, backend
    )
    source_artifacts = [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in production_artifacts
    ]
    manifest_rows: list[dict[str, Any]] = []
    for row in metadata:
        if row.get("output_policy") == PREFILL_POLICY:
            continue
        request_id = str(row["request_id"])
        observations = lengths[request_id]
        manifest_rows.append(
            {
                "request_id": request_id,
                "raw_output_lengths": observations,
                "selected_output_length": max(observations),
                "finish_reasons": finish_reasons[request_id],
                "production_context_hit": request_id in context_hits,
            }
        )
    manifest_payload = {
        "schema_version": 1,
        "policy": EQUAL_WORK_POLICY,
        "backend": backend,
        "frequency_penalty": penalty,
        "production_repetitions": len(production_artifacts),
        "production_artifacts": source_artifacts,
        "rows": manifest_rows,
    }
    manifest_sha256 = _canonical_sha256(manifest_payload)

    output_requests: list[dict[str, Any]] = []
    output_metadata: list[dict[str, Any]] = []
    selected_lengths: list[int] = []
    manifest_by_id = {row["request_id"]: row for row in manifest_rows}
    for request, row in zip(requests, metadata):
        frozen_request = dict(request)
        frozen_row = dict(row)
        if row.get("output_policy") == PREFILL_POLICY:
            output_requests.append(frozen_request)
            output_metadata.append(frozen_row)
            continue
        request_id = str(row["request_id"])
        manifest_row = manifest_by_id[request_id]
        selected = int(manifest_row["selected_output_length"])
        remaining = int(row["context_length"]) - int(row["prompt_len"])
        if selected > remaining:
            raise ValueError(
                f"{request_id} production maximum {selected} exceeds remaining "
                f"context {remaining}"
            )
        body = dict(frozen_request.get("extra_request_body") or {})
        body.pop("stop", None)
        body["temperature"] = 0.0
        body["top_p"] = 1.0
        body["frequency_penalty"] = penalty
        body["ignore_eos"] = True
        frozen_request["output_len"] = selected
        frozen_request["extra_request_body"] = body
        frozen_row.update(
            {
                "requested_output_len": selected,
                "output_policy": EQUAL_WORK_POLICY,
                "fixed_output_tokens": None,
                "quality_eligible": False,
                "frequency_penalty": penalty,
                "equal_work_production_output_lens": list(
                    manifest_row["raw_output_lengths"]
                ),
                "equal_work_production_finish_reasons": list(
                    manifest_row["finish_reasons"]
                ),
                "equal_work_production_max_output_len": selected,
                "equal_work_production_repetitions": len(
                    manifest_row["raw_output_lengths"]
                ),
                "equal_work_production_context_hit": bool(
                    manifest_row["production_context_hit"]
                ),
                "equal_work_manifest_sha256": manifest_sha256,
            }
        )
        selected_lengths.append(selected)
        output_requests.append(frozen_request)
        output_metadata.append(frozen_row)

    context_lengths = {int(row["context_length"]) for row in output_metadata}
    models = {str(row.get("model") or "") for row in output_metadata}
    revisions = {str(row.get("model_revision") or "") for row in output_metadata}
    if len(context_lengths) != 1 or len(models) != 1 or len(revisions) != 1:
        raise ValueError("workload does not pin one model, revision, and context")
    validate_frozen_output_policy(
        output_requests,
        output_metadata,
        next(iter(context_lengths)),
        next(iter(models)),
        next(iter(revisions)),
    )
    summary = dict(manifest_payload)
    summary.update(
        {
            "manifest_sha256": manifest_sha256,
            "decode_rows": len(selected_lengths),
            "prefill_rows": len(output_requests) - len(selected_lengths),
            "minimum_output_length": min(selected_lengths),
            "maximum_output_length": max(selected_lengths),
        }
    )
    return output_requests, output_metadata, summary


def _write_outputs(
    source_summary: Path,
    output_requests: Path,
    output_metadata: Path,
    output_summary: Path,
    requests: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    policy_summary: dict[str, Any],
) -> None:
    outputs = (output_requests, output_metadata, output_summary)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite fair-output artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    applied_penalty = policy_summary.get("frequency_penalty")
    # Declare the penalty this build actually wrote into the request bodies at the
    # top level, where freeze_workload_suite.py reads it. Recording it only under
    # output_policy leaves the suite declaring the source summary's penalty while
    # the bodies carry this one — the undeclared-mutation case the freeze gate
    # rejects. Modes that strip the penalty from the bodies declare 0.0.
    summary["frequency_penalty"] = 0.0 if applied_penalty is None else float(applied_penalty)
    summary["output_policy"] = {
        "decode": policy_summary["policy"],
        "prefill": PREFILL_POLICY,
        "frequency_penalty": applied_penalty,
    }
    summary["fair_output_policy"] = policy_summary
    summary["rows"] = len(requests)
    write_jsonl(output_requests, requests)
    write_jsonl(output_metadata, metadata)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("natural", "equal-work"):
        sub = subparsers.add_parser(mode)
        sub.add_argument("--requests", type=Path, required=True)
        sub.add_argument("--metadata", type=Path, required=True)
        sub.add_argument("--summary", type=Path, required=True)
        sub.add_argument("--output-requests", type=Path, required=True)
        sub.add_argument("--output-metadata", type=Path, required=True)
        sub.add_argument("--output-summary", type=Path, required=True)
    natural = subparsers.choices["natural"]
    natural.add_argument("--frequency-penalty", type=float, required=True)
    equal = subparsers.choices["equal-work"]
    equal.add_argument(
        "--production-artifact", type=Path, action="append", required=True
    )
    equal.add_argument("--backend", default="sglang")
    args = parser.parse_args()

    requests = read_jsonl(args.requests)
    metadata = read_jsonl(args.metadata)
    if args.mode == "natural":
        output_requests, output_metadata, policy_summary = build_natural_rows(
            requests, metadata, args.frequency_penalty
        )
    else:
        output_requests, output_metadata, policy_summary = build_equal_work_rows(
            requests,
            metadata,
            args.production_artifact,
            backend=args.backend,
            declared_penalty=declared_frequency_penalty(
                json.loads(args.summary.read_text(encoding="utf-8"))
            ),
        )
    _write_outputs(
        args.summary,
        args.output_requests,
        args.output_metadata,
        args.output_summary,
        output_requests,
        output_metadata,
        policy_summary,
    )
    print(json.dumps(policy_summary, sort_keys=True))


if __name__ == "__main__":
    main()
