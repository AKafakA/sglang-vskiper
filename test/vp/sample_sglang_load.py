#!/usr/bin/env python3
"""Sample SGLang request occupancy during an official serving benchmark."""

from __future__ import annotations

import argparse
import json
import signal
import time
import urllib.request
from pathlib import Path


_stop = False


def _request_stop(_signum, _frame) -> None:
    global _stop
    _stop = True


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def fetch_sample(endpoint: str, started: float) -> dict[str, int | float]:
    with urllib.request.urlopen(endpoint, timeout=2) as response:
        payload = json.load(response)
    loads = payload["loads"]
    running = sum(int(item["num_running_reqs"]) for item in loads)
    waiting = sum(int(item["num_waiting_reqs"]) for item in loads)
    return {
        "time_unix_s": started,
        "running_requests": running,
        "waiting_requests": waiting,
        "offered_requests": running + waiting,
        "resident_tokens": sum(int(item["num_total_tokens"]) for item in loads),
        "pending_tokens": sum(
            int(item["num_total_tokens"]) - int(item["num_used_tokens"])
            for item in loads
        ),
    }


def summarize(samples: list[dict[str, int | float]]) -> dict[str, int | float]:
    running_values = [int(sample["running_requests"]) for sample in samples]
    waiting_values = [int(sample["waiting_requests"]) for sample in samples]
    return {
        "samples": len(samples),
        "max_running_requests": max(running_values, default=0),
        "p50_running_requests": _percentile(running_values, 0.50),
        "p90_running_requests": _percentile(running_values, 0.90),
        "max_waiting_requests": max(waiting_values, default=0),
        "p90_waiting_requests": _percentile(waiting_values, 0.90),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=1000)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, int | float]] = []
    endpoint = f"{args.base_url.rstrip('/')}/v1/loads?include=core"
    interval = max(args.interval_ms, 10) / 1000.0

    with args.output.open("w", encoding="utf-8") as output:
        while not _stop:
            started = time.time()
            try:
                sample = fetch_sample(endpoint, started)
                samples.append(sample)
                output.write(json.dumps(sample, sort_keys=True) + "\n")
                output.flush()
            except Exception as exc:
                output.write(
                    json.dumps(
                        {"time_unix_s": started, "sample_error": str(exc)},
                        sort_keys=True,
                    )
                    + "\n"
                )
                output.flush()
            time.sleep(max(0.0, interval - (time.time() - started)))

    summary = summarize(samples)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
