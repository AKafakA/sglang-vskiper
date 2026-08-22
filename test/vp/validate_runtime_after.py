#!/usr/bin/env python3
"""Validate post-run runtime identity and required graph activity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from launch_qps_server import _validate_server_profile
from qps_deployment import validate_expected_runtime


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _stage_graph_runtime_key(expected: dict) -> str | None:
    """Which runtime generation's graph counters to audit after the run.

    This used to look only at runtime.v2 / runtime.v4 stage_graphs. Both
    generations are deleted and the live attestation emits ``v3_full_graph``
    instead, so the activity branch below was UNREACHABLE: the audit reported
    passed even with zero graph dispatches or non-zero eager dispatches. A
    post-run gate that cannot fail is worse than no gate, because it retires
    the question.
    """
    runtime = expected.get("runtime", {})
    if bool(runtime.get("v3_full_graph", {}).get("enabled")):
        return "v3_full_graph"
    legacy = [
        key
        for key in ("v2", "v4")
        if bool(runtime.get(key, {}).get("stage_graphs", {}).get("enabled"))
    ]
    if legacy:
        raise ValueError(
            f"runtime expectation names removed generation(s) {legacy}; "
            "V2/V4 are not part of this build -- use v3_full_graph"
        )
    return None


def validate_v3_graph_activity(attestations: list) -> list:
    """Post-run: the routed decode path must have ACTUALLY executed, captured.

    Asserts on counters the live attestation really emits, so the gate can fail.
    """
    activity = []
    if not attestations:
        raise ValueError("v3_full_graph expectation has no runtime attestation")
    for rank, runtime in enumerate(attestations):
        block = runtime.get("v3_full_graph") or {}
        counters = (runtime.get("fd_c3") or {}).get("counters", {})
        dispatches = int(block.get("decode_dispatches_total", 0) or 0)
        graphed = int(block.get("decode_graph_dispatches_total", 0) or 0)
        eager = int(counters.get("eager_skip_decode_layer_calls", 0) or 0)
        if dispatches <= 0:
            raise ValueError(
                f"rank {rank}: zero routed decode dispatches after the run"
            )
        if graphed <= 0:
            raise ValueError(
                f"rank {rank}: zero captured decode dispatches after the run "
                f"({dispatches} ran, none graphed)"
            )
        if eager:
            raise ValueError(
                f"rank {rank}: {eager} eager skip-decode layer call(s); the "
                "routed body must be structurally unreachable in serving"
            )
        activity.append(
            {
                "rank": rank,
                "decode_dispatches_total": dispatches,
                "decode_graph_dispatches_total": graphed,
                "eager_skip_decode_layer_calls": eager,
            }
        )
    return activity


def validate_runtime_after(
    server_info: dict, expected: dict, server_profile: str
) -> dict:
    profile = _validate_server_profile(server_info, server_profile)
    attestations = validate_expected_runtime(server_info, expected)
    graph_activity = []
    stage_graph_runtime_key = _stage_graph_runtime_key(expected)
    if stage_graph_runtime_key == "v3_full_graph":
        graph_activity = validate_v3_graph_activity(attestations)
    elif stage_graph_runtime_key is not None:
        if not attestations:
            raise ValueError("stage graph expectation has no runtime attestation")
        for rank, runtime in enumerate(attestations):
            graph = runtime.get(stage_graph_runtime_key, {}).get("stage_graphs", {})
            counters = graph.get("counters", {})
            rejection_reasons = graph.get("rejection_reasons", {})
            required = {
                name: int(counters.get(name, 0))
                for name in ("captures", "replays", "executions")
            }
            if any(value <= 0 for value in required.values()):
                raise ValueError(
                    f"scheduler rank {rank} has no complete stage graph activity: "
                    f"{required}"
                )
            rejected = int(counters.get("rejected_dispatches", 0))
            if rejected or rejection_reasons:
                raise ValueError(
                    f"scheduler rank {rank} rejected stage graph dispatches: "
                    f"count={rejected} reasons={rejection_reasons}"
                )
            graph_activity.append(
                {
                    "rank": rank,
                    "runtime": stage_graph_runtime_key,
                    "backend": graph.get("backend"),
                    "buckets": graph.get("buckets"),
                    "captured_keys": graph.get("captured_keys"),
                    "counters": counters,
                }
            )
    return {
        "status": "passed",
        "server_profile": profile,
        "runtime_ranks": len(attestations),
        "stage_graph_activity": graph_activity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-info", type=Path, required=True)
    parser.add_argument("--expected-runtime", type=Path, required=True)
    parser.add_argument("--server-profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    server_info = _load(args.server_info)
    expected = _load(args.expected_runtime)
    output = validate_runtime_after(server_info, expected, args.server_profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
