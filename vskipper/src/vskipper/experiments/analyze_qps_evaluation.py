#!/usr/bin/env python3
"""Analyze sustainable QPS, latency, throughput, and quality by experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_METRIC_ACCOUNTING_VERSION = 5


def _nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _latency_distributions(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Request-weighted percentiles + token-weighted TPOT from the raw arrays.

    The sealed artifact already carries per-request `ttfts`, `itls` and
    `e2e_latencies` (serving.py:2132-2135); only mean and p99 were being
    surfaced, so p50/p90/p95 and the token-weighted TPOT were computable all
    along and simply never printed.

    Token-weighted TPOT pools every inter-token latency, so it is NOT the mean
    of the per-request TPOTs: the two legitimately disagree when output lengths
    are skewed, which is exactly why both are reported.
    """
    successes = list(benchmark.get("successes") or [])
    ttfts = list(benchmark.get("ttfts") or [])
    itls = list(benchmark.get("itls") or [])
    e2es = list(benchmark.get("e2e_latencies") or [])
    indices = [index for index, success in enumerate(successes) if success]
    if not indices:
        return {"status": "no_successful_requests"}

    def _ms(values: list[Any]) -> list[float]:
        return [float(value) * 1000.0 for value in values if value is not None]

    ttft_ms = _ms([ttfts[i] for i in indices if i < len(ttfts)])
    e2e_ms = _ms([e2es[i] for i in indices if i < len(e2es)])
    per_request_itls = [
        [float(v) for v in itls[i] if v is not None]
        for i in indices
        if i < len(itls) and itls[i]
    ]
    per_request_itls = [row for row in per_request_itls if row]
    tpot_ms = _ms([sum(row) / len(row) for row in per_request_itls])

    pooled = [value for row in per_request_itls for value in row]
    summary: dict[str, Any] = {
        "token_weighted_tpot_ms": (
            1000.0 * sum(pooled) / len(pooled) if pooled else None
        ),
        "decode_token_count": len(pooled),
        "requests": len(indices),
    }
    for name, series in (("ttft", ttft_ms), ("tpot", tpot_ms), ("e2e", e2e_ms)):
        if not series:
            continue
        ordered = sorted(series)
        summary[f"mean_{name}_ms"] = sum(ordered) / len(ordered)
        for percentile in (0.50, 0.90, 0.95, 0.99):
            index = max(0, math.ceil(percentile * len(ordered)) - 1)
            summary[f"p{int(percentile * 100)}_{name}_ms"] = ordered[index]
    return summary


def _output_accounting_summary(
    benchmark: dict[str, Any], command: dict[str, Any]
) -> dict[str, Any]:
    successes = list(benchmark.get("successes") or [])
    effective = list(benchmark.get("output_lens") or [])
    retokenized = list(benchmark.get("retokenized_output_lens") or [])
    requested = list(benchmark.get("requested_output_lens") or [])
    server_reported = list(benchmark.get("server_reported_output_lens") or [])
    finish_reasons = list(benchmark.get("finish_reasons") or [])
    if not effective or len(successes) != len(effective):
        return {"status": "missing_detailed_output_accounting"}
    indices = [index for index, success in enumerate(successes) if success]
    output_lens = [int(effective[index]) for index in indices]
    retokenized_lens = [
        int(retokenized[index])
        for index in indices
        if index < len(retokenized) and retokenized[index] is not None
    ]
    reasons = [
        finish_reasons[index] if index < len(finish_reasons) else None
        for index in indices
    ]
    fixed_values = [int(value) for value in command.get("fixed_output_tokens", [])]
    fixed_compliant = None
    fixed_intent = command.get("evaluation_intent") in {
        "fixed_token_capacity",
        "mixed_fixed_token_capacity",
    }
    if fixed_intent:
        mixed_fixed = command.get("evaluation_intent") == "mixed_fixed_token_capacity"
        fixed_compliant = bool(
            indices
            and len(fixed_values) == 1
            and all(
                index < len(requested)
                and requested[index] is not None
                and (
                    int(requested[index]) == fixed_values[0]
                    or (mixed_fixed and int(requested[index]) == 1)
                )
                and index < len(server_reported)
                and server_reported[index] is not None
                and int(server_reported[index]) == int(requested[index])
                and index < len(retokenized)
                and retokenized[index] is not None
                and int(effective[index]) == int(requested[index])
                and (
                    int(requested[index]) == 1
                    or reasons[position] == "length"
                )
                for position, index in enumerate(indices)
            )
        )
    return {
        "status": "present",
        "successful_requests": len(indices),
        "total_effective_output_tokens": sum(output_lens),
        "total_retokenized_output_tokens": sum(retokenized_lens),
        "mean_effective_output_tokens": (
            statistics.fmean(output_lens) if output_lens else None
        ),
        "median_effective_output_tokens": (
            statistics.median(output_lens) if output_lens else None
        ),
        "p95_effective_output_tokens": _nearest_rank(output_lens, 0.95),
        "p99_effective_output_tokens": _nearest_rank(output_lens, 0.99),
        "max_effective_output_tokens": max(output_lens) if output_lens else None,
        "finish_reason_counts": dict(
            sorted(Counter(str(reason) for reason in reasons).items())
        ),
        "fixed_length_compliant": fixed_compliant,
    }


def _injection_output_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    starts = list(benchmark.get("request_start_offsets_s") or [])
    ttfts = list(benchmark.get("ttfts") or [])
    itls = list(benchmark.get("itls") or [])
    successes = list(benchmark.get("successes") or [])
    output_lens = list(benchmark.get("output_lens") or [])
    count = len(starts)
    if not count or not all(
        len(values) == count for values in (ttfts, itls, successes, output_lens)
    ):
        return {"status": "missing_request_token_timing"}
    window_start = min(float(value) for value in starts)
    window_end = max(float(value) for value in starts)
    span = window_end - window_start
    if span <= 0:
        return {"status": "zero_injection_span"}
    expected_events = 0
    timestamped_events = 0
    events_in_window = 0
    for start, ttft, request_itls, success, output_len in zip(
        starts, ttfts, itls, successes, output_lens
    ):
        if not success:
            continue
        expected = int(output_len)
        expected_events += expected
        if expected <= 0:
            continue
        token_times = [float(start) + float(ttft)]
        for interval in request_itls:
            token_times.append(token_times[-1] + float(interval))
        timestamped_events += len(token_times)
        events_in_window += sum(
            window_start <= timestamp <= window_end for timestamp in token_times
        )
    exact_coverage = expected_events > 0 and timestamped_events == expected_events
    return {
        "status": "present" if exact_coverage else "incomplete_token_timing",
        "window_start_offset_s": window_start,
        "window_end_offset_s": window_end,
        "window_span_s": span,
        "expected_output_tokens": expected_events,
        "timestamped_output_events": timestamped_events,
        "timing_coverage_ratio": (
            timestamped_events / expected_events if expected_events else None
        ),
        "output_events_in_window": events_in_window,
        "output_tps": events_in_window / span if exact_coverage else None,
    }


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_path(command_path: Path, recorded_path: str) -> Path:
    path = Path(recorded_path)
    if path.is_file():
        return path
    return command_path.parent / path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts_match(path: Path, command: dict[str, Any]) -> bool:
    records = command.get("artifacts")
    if not isinstance(records, dict) or not records:
        return False
    for record in records.values():
        if not isinstance(record, dict) or not record.get("path"):
            return False
        artifact = _resolve_artifact_path(path, record["path"])
        if not artifact.is_file():
            return False
        if record.get("sha256") != _sha256(artifact):
            return False
        if int(record.get("size_bytes") or -1) != artifact.stat().st_size:
            return False
    return True


def _injection_load_summary(
    benchmark: dict[str, Any], load_path: Path
) -> dict[str, Any]:
    start_unix_s = benchmark.get("benchmark_start_unix_s")
    offsets = benchmark.get("request_start_offsets_s")
    if start_unix_s is None or not isinstance(offsets, list) or not offsets:
        return {"status": "missing_request_timing"}
    numeric_offsets = [float(value) for value in offsets]
    injection_start = float(start_unix_s) + min(numeric_offsets)
    injection_end = float(start_unix_s) + max(numeric_offsets)
    samples = []
    if load_path.is_file():
        for line in load_path.read_text(encoding="utf-8").splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = sample.get("time_unix_s")
            if (
                timestamp is not None
                and injection_start <= float(timestamp) <= injection_end
                and "waiting_requests" in sample
                and "running_requests" in sample
            ):
                samples.append(sample)
    if not samples:
        return {
            "status": "missing_load_samples",
            "injection_start_unix_s": injection_start,
            "injection_end_unix_s": injection_end,
        }
    duration = max(0.0, injection_end - injection_start)
    quarter = duration / 4.0
    first = [
        int(sample["waiting_requests"])
        for sample in samples
        if float(sample["time_unix_s"]) <= injection_start + quarter
    ]
    last = [
        int(sample["waiting_requests"])
        for sample in samples
        if float(sample["time_unix_s"]) >= injection_end - quarter
    ]
    waiting = [int(sample["waiting_requests"]) for sample in samples]
    running = [int(sample["running_requests"]) for sample in samples]
    realized_arrival_qps = (
        (len(numeric_offsets) - 1) / duration
        if len(numeric_offsets) > 1 and duration > 0
        else None
    )
    return {
        "status": "present",
        "request_count": len(numeric_offsets),
        "injection_start_unix_s": injection_start,
        "injection_end_unix_s": injection_end,
        "injection_span_s": duration,
        "realized_arrival_qps": realized_arrival_qps,
        "sample_count": len(samples),
        "max_running_requests": max(running),
        "max_waiting_requests": max(waiting),
        "waiting_at_last_sample": waiting[-1],
        "first_quarter_mean_waiting": (
            sum(first) / len(first) if first else None
        ),
        "last_quarter_mean_waiting": sum(last) / len(last) if last else None,
        "quarter_waiting_growth": (
            sum(last) / len(last) - sum(first) / len(first)
            if first and last
            else None
        ),
    }


def _injection_load_sustainable(summary: dict[str, Any]) -> bool | None:
    if summary.get("status") != "present":
        return None
    request_count = int(summary.get("request_count") or 0)
    tolerance = max(1, math.ceil(0.01 * request_count))
    last_waiting = summary.get("waiting_at_last_sample")
    growth = summary.get("quarter_waiting_growth")
    if last_waiting is None or growth is None:
        return None
    return float(last_waiting) <= tolerance and float(growth) <= tolerance


def _cell_from_command(path: Path, tracking_threshold: float) -> dict[str, Any]:
    command = _read(path)
    if command.get("status") != "completed":
        return {**command, "sustainable": False, "reason": "cell_not_completed"}
    score_payload = _read(_resolve_artifact_path(path, command["score_file"]))
    scores = score_payload.get("scores") or []
    if not scores:
        return {**command, "sustainable": False, "reason": "missing_score_record"}
    score = scores[-1]
    performance = score.get("performance") or {}
    offered = float(command["qps"])
    achieved = float(performance.get("request_throughput") or 0.0)
    completed = int(performance.get("completed") or 0)
    expected = int(command["num_prompts"])
    tracking_ratio = achieved / offered if offered > 0 else 0.0
    status_counts = score.get("status_counts") or {}
    request_errors = int(status_counts.get("request_error", 0))
    tail_tracking_sustainable = (
        completed == expected
        and request_errors == 0
        and tracking_ratio >= tracking_threshold
    )
    evidence_class = command.get("evidence_class", "legacy_quarantined")
    accounting_version = int(
        performance.get("metric_accounting_version")
        or command.get("metric_accounting_version_required")
        or 0
    )
    audit = None
    if command.get("audit_file"):
        audit_path = _resolve_artifact_path(path, command["audit_file"])
        audit = _read(audit_path) if audit_path.is_file() else None
    accounting_passed = bool(audit and audit.get("status") == "passed")
    artifact_integrity_passed = _artifacts_match(path, command)
    identity_alignment_passed = score.get("alignment_source") == "request_id"
    performance_evidence_eligible = (
        evidence_class == "performance"
        and accounting_version >= REQUIRED_METRIC_ACCOUNTING_VERSION
        and accounting_passed
        and artifact_integrity_passed
        and identity_alignment_passed
    )
    load_path = _resolve_artifact_path(path, command["load_file"])
    load_summary_path = load_path.with_suffix(".summary.json")
    load_summary = _read(load_summary_path) if load_summary_path.is_file() else None
    benchmark_path = _resolve_artifact_path(path, command["output_file"])
    benchmark_records = [
        json.loads(line)
        for line in benchmark_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    injection_load = (
        _injection_load_summary(benchmark_records[-1], load_path)
        if benchmark_records
        else {"status": "missing_benchmark_record"}
    )
    output_accounting = (
        _output_accounting_summary(benchmark_records[-1], command)
        if benchmark_records
        else {"status": "missing_benchmark_record"}
    )
    injection_output = (
        _injection_output_summary(benchmark_records[-1])
        if benchmark_records
        else {"status": "missing_benchmark_record"}
    )
    latency_distributions = (
        _latency_distributions(benchmark_records[-1])
        if benchmark_records
        else {"status": "missing_benchmark_record"}
    )
    injection_sustainable = _injection_load_sustainable(injection_load)
    load_sustainable = (
        completed == expected
        and request_errors == 0
        and (
            injection_sustainable
            if injection_sustainable is not None
            else tail_tracking_sustainable
        )
    )
    sustainable = load_sustainable and performance_evidence_eligible
    return {
        "label": command["label"],
        "experiment": command.get("experiment", "legacy_vanilla"),
        "evidence_class": evidence_class,
        "performance_evidence_eligible": performance_evidence_eligible,
        "metric_accounting_version": accounting_version,
        "accounting_audit": audit,
        "artifact_integrity_passed": artifact_integrity_passed,
        "identity_alignment_passed": identity_alignment_passed,
        "deployment_manifest_sha256": command.get(
            "deployment_manifest_sha256"
        ),
        "workload": command["workload"],
        "evaluation_intent": command.get("evaluation_intent", "unknown"),
        "output_policies": command.get("output_policies", []),
        "fixed_output_tokens": command.get("fixed_output_tokens", []),
        "qps": offered,
        "rep": command["rep"],
        "num_prompts": expected,
        "completed": completed,
        "achieved_qps": achieved,
        "tracking_ratio": tracking_ratio,
        "tail_tracking_sustainable": tail_tracking_sustainable,
        "injection_sustainable": injection_sustainable,
        "load_sustainable": load_sustainable,
        "sustainable": sustainable,
        "request_errors": request_errors,
        "duration_s": performance.get("duration"),
        "output_throughput": performance.get("output_throughput"),
        "injection_window_output_tps": injection_output.get("output_tps"),
        "mean_ttft_ms": performance.get("mean_ttft_ms"),
        "p99_ttft_ms": performance.get("p99_ttft_ms"),
        "mean_tpot_ms": performance.get("mean_tpot_ms"),
        "p99_tpot_ms": performance.get("p99_tpot_ms"),
        "mean_e2e_latency_ms": performance.get("mean_e2e_latency_ms"),
        "p99_e2e_latency_ms": performance.get("p99_e2e_latency_ms"),
        "latency_distributions": latency_distributions,
        "quality": score.get("quality") or {},
        "status_counts": status_counts,
        "load": load_summary,
        "injection_load": injection_load,
        "output_accounting": output_accounting,
        "injection_output": injection_output,
        "command_file": str(path),
    }


def _bracket(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_qps: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if (
            "tracking_ratio" not in cell
            or "achieved_qps" not in cell
            or not cell.get("performance_evidence_eligible", False)
        ):
            continue
        by_qps[float(cell["qps"])].append(cell)
    levels = []
    for qps in sorted(by_qps):
        repetitions = by_qps[qps]
        sustainable = all(bool(cell["sustainable"]) for cell in repetitions)
        levels.append(
            {
                "qps": qps,
                "sustainable": sustainable,
                "repetitions": len(repetitions),
                "tracking_ratios": [cell["tracking_ratio"] for cell in repetitions],
                "achieved_qps": [cell["achieved_qps"] for cell in repetitions],
            }
        )
    passing = [level for level in levels if level["sustainable"]]
    failing = [level for level in levels if not level["sustainable"]]
    if not passing:
        return {
            "status": "not_bracketed_below",
            "levels": levels,
            "next_qps": levels[0]["qps"] / 2 if levels else None,
        }
    last_pass = passing[-1]["qps"]
    higher_failures = [level["qps"] for level in failing if level["qps"] > last_pass]
    if not higher_failures:
        return {
            "status": "not_bracketed_above",
            "levels": levels,
            "sustainable_qps_lower_bound": last_pass,
            "next_qps": levels[-1]["qps"] * 2,
        }
    first_fail = min(higher_failures)
    return {
        "status": "bracketed_needs_confirmation",
        "levels": levels,
        "sustainable_qps_candidate": last_pass,
        "first_overload_qps": first_fail,
        "confirmation_qps": sorted(
            {
                max(levels[0]["qps"], last_pass / 2),
                last_pass,
                first_fail,
            }
        ),
        "derived_load_qps": {
            "low": 0.25 * last_pass,
            "medium": 0.50 * last_pass,
            "high": 0.80 * last_pass,
            "extra_high": 1.10 * last_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--tracking-threshold", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    command_paths = sorted(
        path
        for artifact_dir in args.artifact_dir
        for path in artifact_dir.glob("*_qps*_rep*.command.json")
    )
    cells = [
        _cell_from_command(path, args.tracking_threshold)
        for path in command_paths
    ]
    cell_keys = [
        (str(cell.get("experiment")), str(cell.get("label"))) for cell in cells
    ]
    if len(cell_keys) != len(set(cell_keys)):
        raise ValueError("artifact directories contain duplicate experiment/cell labels")
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_experiment[cell["experiment"]].append(cell)
    experiments = {}
    for experiment, experiment_cells in sorted(by_experiment.items()):
        by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cell in experiment_cells:
            by_workload[cell["workload"]].append(cell)
        experiments[experiment] = {
            "deployment_manifest_sha256s": sorted(
                {
                    cell["deployment_manifest_sha256"]
                    for cell in experiment_cells
                    if cell["deployment_manifest_sha256"]
                }
            ),
            "workloads": {
                workload: {
                    "bracket": _bracket(workload_cells),
                    "cells": workload_cells,
                }
                for workload, workload_cells in sorted(by_workload.items())
            },
        }
    payload = {
        "artifact_dirs": [str(path) for path in args.artifact_dir],
        "tracking_threshold": args.tracking_threshold,
        "experiments": experiments,
    }
    output = args.output or args.artifact_dir[0] / "qps_analysis.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for experiment, experiment_value in payload["experiments"].items():
        for workload, value in experiment_value["workloads"].items():
            print(experiment, workload, json.dumps(value["bracket"], sort_keys=True))


if __name__ == "__main__":
    main()
