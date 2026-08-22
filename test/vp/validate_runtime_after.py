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
    runtime = expected.get("runtime", {})
    keys = [
        key
        for key in ("v2", "v4")
        if bool(runtime.get(key, {}).get("stage_graphs", {}).get("enabled"))
    ]
    if len(keys) > 1:
        raise ValueError("runtime expectation enables both V2 and V4 stage graphs")
    return keys[0] if keys else None


def validate_runtime_after(
    server_info: dict, expected: dict, server_profile: str
) -> dict:
    profile = _validate_server_profile(server_info, server_profile)
    attestations = validate_expected_runtime(server_info, expected)
    graph_activity = []
    stage_graph_runtime_key = _stage_graph_runtime_key(expected)
    if stage_graph_runtime_key is not None:
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
