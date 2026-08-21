#!/usr/bin/env python3
"""Per-prompt per-layer engagement calibration (fallback (B) prerequisite).

Sends suite prompts one at a time (conc-1, 1 new token) against a V-pre
full-graph server running with SGLANG_FD_FULL_GRAPH_LAYER_COUNTERS=1 and
diffs the ACCUMULATING per-layer route counters around each request —
yielding each prompt's PROJECT fraction per routed layer. The instrument
for the layer-16-vs-pass-aggregate correlation that gates option (B).
Single-chunk prompts only (prompt must fit one prefill pass): asserted
via a chunk budget argument, fail-closed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def http_json(
    url: str, payload: dict | None = None, timeout: float = 600.0
) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def find_per_layer(node):
    """Locate the per-layer route counter list in a server_info tree."""

    if isinstance(node, dict):
        for key, value in node.items():
            if key == "per_layer" and isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, dict) and "layer_id" in first:
                    return value
            found = find_per_layer(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = find_per_layer(item)
            if found is not None:
                return found
    return None


def snapshot(url: str) -> dict[int, tuple[int, int]]:
    info = http_json(url + "/server_info")
    per_layer = find_per_layer(info)
    if per_layer is None:
        raise SystemExit(
            "server_info has no per_layer route counters — is "
            "SGLANG_FD_FULL_GRAPH_LAYER_COUNTERS=1 set on the server?"
        )
    out = {}
    for record in per_layer:
        if "project_rows" not in record:
            raise SystemExit(
                "per-layer counters are not RUN/PROJECT records "
                f"(got keys {sorted(record)})"
            )
        out[int(record["layer_id"])] = (
            int(record["layer_rows"]),
            int(record["project_rows"]),
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--first-n", type=int, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument(
        "--chunk-budget",
        type=int,
        default=8192,
        help="fail on prompts longer than one prefill chunk",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    if args.first_n <= 0:
        raise SystemExit("first-n must be positive")

    rows = []
    with args.requests_jsonl.open(encoding="utf-8") as source:
        for line in source:
            rows.append(json.loads(line))
            if len(rows) == args.first_n:
                break
    if len(rows) < args.first_n:
        raise SystemExit(
            f"suite holds {len(rows)} requests, {args.first_n} requested"
        )

    url = args.url.rstrip("/")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    previous = snapshot(url)
    with args.output_jsonl.open("w", encoding="utf-8") as sink:
        for position, row in enumerate(rows):
            prompt = row["prompt"]
            if len(prompt) > args.chunk_budget:
                raise SystemExit(
                    f"{row['request_id']}: prompt {len(prompt)} tokens "
                    f"exceeds the single-chunk budget {args.chunk_budget}"
                )
            http_json(
                url + "/generate",
                {
                    "input_ids": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 1,
                    },
                },
            )
            current = snapshot(url)
            per_layer = {}
            for layer_id, (rows_now, project_now) in current.items():
                rows_before, project_before = previous.get(layer_id, (0, 0))
                delta_rows = rows_now - rows_before
                delta_project = project_now - project_before
                if delta_rows <= 0:
                    raise SystemExit(
                        f"{row['request_id']}: layer {layer_id} counter did "
                        f"not advance (delta_rows={delta_rows}) — counters "
                        "not accumulating as expected"
                    )
                per_layer[layer_id] = {
                    "rows": delta_rows,
                    "project": delta_project,
                    "engagement": delta_project / delta_rows,
                }
            previous = current
            first_layer = min(per_layer)
            record = {
                "workload": args.workload,
                "request_id": row["request_id"],
                "prompt_tokens": len(prompt),
                "per_layer": per_layer,
                "first_layer_engagement": per_layer[first_layer][
                    "engagement"
                ],
                "aggregate_engagement": (
                    sum(v["project"] for v in per_layer.values())
                    / sum(v["rows"] for v in per_layer.values())
                ),
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(
                f"{position + 1}/{len(rows)} {row['request_id']} "
                f"L{first_layer}={record['first_layer_engagement']:.3f} "
                f"agg={record['aggregate_engagement']:.3f}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
