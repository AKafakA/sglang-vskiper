#!/usr/bin/env python3
"""Freeze per-request natural-generation watchdogs from production outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from build_labeled_workload import validate_frozen_output_policy
from labeled_workload import read_jsonl, write_jsonl
from validate_qps_artifact import read_last_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _production_lengths(
    artifacts: list[Path], expected_ids: set[str]
) -> tuple[dict[str, list[int]], set[str]]:
    lengths = {request_id: [] for request_id in expected_ids}
    context_hits: set[str] = set()
    for artifact in artifacts:
        record = read_last_record(artifact)
        request_ids = record.get("request_ids")
        output_lens = record.get("server_reported_output_lens")
        finish_reasons = record.get("finish_reasons")
        successes = record.get("successes")
        arrays = (request_ids, output_lens, finish_reasons, successes)
        if not all(isinstance(value, list) for value in arrays):
            raise ValueError(f"{artifact} lacks accounting-v5 request arrays")
        if len({len(value) for value in arrays}) != 1:
            raise ValueError(f"{artifact} has inconsistent request arrays")
        observed: set[str] = set()
        for request_id, output_len, finish_reason, success in zip(*arrays):
            request_id = str(request_id)
            if request_id not in expected_ids:
                continue
            if request_id in observed:
                raise ValueError(f"{artifact} repeats request ID {request_id}")
            observed.add(request_id)
            if success is not True:
                raise ValueError(f"{artifact} failed request {request_id}")
            if finish_reason == "length":
                context_hits.add(request_id)
            if not isinstance(output_len, int) or output_len <= 0:
                raise ValueError(
                    f"{artifact} has invalid production length for {request_id}"
                )
            lengths[request_id].append(output_len)
        missing = expected_ids - observed
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            raise ValueError(
                f"{artifact} lacks {len(missing)} natural request IDs: {sample}"
            )
    return lengths, context_hits


def build_watchdog_rows(
    requests: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    artifacts: list[Path],
    multiplier: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("watchdog multiplier must be finite and at least 1.0")
    if len(requests) != len(metadata):
        raise ValueError("request and metadata counts differ")
    natural_ids = {
        str(row["request_id"])
        for row in metadata
        if row.get("output_policy") != "prefill_probe_one_token"
    }
    if not natural_ids:
        raise ValueError("workload contains no natural-generation requests")
    lengths, context_hits = _production_lengths(artifacts, natural_ids)
    repetitions = len(artifacts)
    frozen_requests: list[dict[str, Any]] = []
    frozen_metadata: list[dict[str, Any]] = []
    cap_values: list[int] = []
    for request, row in zip(requests, metadata):
        frozen_request = dict(request)
        frozen_row = dict(row)
        if row.get("output_policy") != "prefill_probe_one_token":
            request_id = str(row["request_id"])
            production_max = max(lengths[request_id])
            task_floor = int(row.get("task_reference_max_output_len") or 0)
            if task_floor <= 0:
                raise ValueError(f"{request_id} has no task-reference floor")
            remaining = int(row["context_length"]) - int(row["prompt_len"])
            cap = min(
                remaining,
                max(task_floor, math.ceil(multiplier * production_max)),
            )
            if cap <= 0:
                raise ValueError(f"{request_id} has no output capacity")
            body = dict(frozen_request.get("extra_request_body") or {})
            if body.get("ignore_eos") is not False:
                raise ValueError(f"{request_id} does not preserve EOS")
            frozen_request["output_len"] = cap
            frozen_request["extra_request_body"] = body
            frozen_row.update(
                {
                    "requested_output_len": cap,
                    "output_policy": "production_watchdog",
                    "watchdog_multiplier": multiplier,
                    "watchdog_production_max_output_len": production_max,
                    "watchdog_production_repetitions": repetitions,
                    "watchdog_production_context_hit": (
                        request_id in context_hits
                    ),
                }
            )
            cap_values.append(cap)
        frozen_requests.append(frozen_request)
        frozen_metadata.append(frozen_row)
    context_lengths = {int(row["context_length"]) for row in frozen_metadata}
    models = {str(row.get("model") or "") for row in frozen_metadata}
    revisions = {str(row.get("model_revision") or "") for row in frozen_metadata}
    if len(context_lengths) != 1 or len(models) != 1 or len(revisions) != 1:
        raise ValueError("workload does not pin one model, revision, and context")
    validate_frozen_output_policy(
        frozen_requests,
        frozen_metadata,
        next(iter(context_lengths)),
        next(iter(models)),
        next(iter(revisions)),
    )
    return frozen_requests, frozen_metadata, {
        "policy": "production_watchdog",
        "multiplier": multiplier,
        "production_repetitions": repetitions,
        "natural_request_count": len(natural_ids),
        "production_context_hit_count": len(context_hits),
        "production_context_hit_request_ids": sorted(context_hits),
        "minimum_cap": min(cap_values),
        "maximum_cap": max(cap_values),
        "production_artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--production-artifact", type=Path, action="append", required=True
    )
    parser.add_argument("--multiplier", type=float, default=1.05)
    parser.add_argument("--output-requests", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_requests, args.output_metadata, args.output_summary)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite watchdog artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    requests = read_jsonl(args.requests)
    metadata = read_jsonl(args.metadata)
    source_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    frozen_requests, frozen_metadata, summary = build_watchdog_rows(
        requests,
        metadata,
        args.production_artifact,
        args.multiplier,
    )
    output_summary = dict(source_summary)
    output_policy = dict(output_summary.get("output_policy") or {})
    output_policy["decode"] = "production_watchdog"
    output_policy["fixed_output_tokens"] = None
    output_summary["output_policy"] = output_policy
    output_summary["production_watchdog"] = summary
    output_summary.update(
        {
            "source_requests": str(args.requests.resolve()),
            "source_requests_sha256": _sha256(args.requests),
            "source_metadata": str(args.metadata.resolve()),
            "source_metadata_sha256": _sha256(args.metadata),
            "source_summary": str(args.summary.resolve()),
            "source_summary_sha256": _sha256(args.summary),
            "rows": len(frozen_requests),
        }
    )
    write_jsonl(args.output_requests, frozen_requests)
    write_jsonl(args.output_metadata, frozen_metadata)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(output_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_summary, sort_keys=True))


if __name__ == "__main__":
    main()
