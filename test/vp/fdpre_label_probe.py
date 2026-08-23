#!/usr/bin/env python3
"""W1.2 FDpre labeling probe client.

Replays the FIRST N rows of a frozen suite (exact token IDs) against a
running FDpre server via /generate, greedy, and records per item: text,
empty label, finish reason, and the RAW first-token logprob entries (top-k)
for the fork-vs-oracle first-token diff. Client only — no server changes.
Design: codex/asplos-plan/2026-08-18-w12-fdpre-labeling-run-design.md.
Style matches the sibling probe clients (plain dicts + json; T1 labeling
instrument, never performance).
"""

from __future__ import annotations

import argparse
import json

import vp_stream
import math
import sys
import time
from pathlib import Path
from urllib import request as urlrequest


def http_json(url: str, payload: dict | None = None, timeout: float = 900.0):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urlrequest.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urlrequest.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def http_stream_generate(url: str, payload: dict, timeout: float = 900.0):
    """Stream /generate; return (final_chunk_dict, ttft_s, e2e_s).

    Timing is CLIENT-side wall clock: t0 at request send, TTFT at the first
    SSE data chunk, e2e at stream end. Directional instrument only — the
    sealed open-loop runner remains the sole source of citable metrics.
    """
    payload = dict(payload)
    payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    ttft = None
    final = None
    saw_done = False
    # Parsing is shared with the rest of the harness via vp_stream; the
    # COMPLETION POLICY below stays local, because this probe requires the
    # [DONE] terminator and vp_stream.collect deliberately does not.
    with urlrequest.urlopen(req, timeout=timeout) as response:
        for kind, obj in vp_stream.decode_lines(response):
            if kind == "done":
                saw_done = True
                break
            if ttft is None:
                ttft = time.monotonic() - t0
            final = obj
    e2e = time.monotonic() - t0
    if final is None:
        raise RuntimeError("stream ended without any data chunk")
    if not saw_done:
        # EOF before the terminator: the last chunk is a cumulative PARTIAL
        # label — recording it would silently mislabel, so fail loudly.
        raise RuntimeError("stream ended without the [DONE] terminator")
    return final, ttft, e2e


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--first-n", type=int, default=100)
    parser.add_argument("--url", default="http://127.0.0.1:30111")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="in-flight requests; >1 exercises PACKED multi-request prefill "
        "batches (the H4 axis — the D-090 quality lane ran concurrently)",
    )
    parser.add_argument(
        "--stream-timing",
        action="store_true",
        help="stream /generate and record per-request client-side ttft_s/"
        "e2e_s + aggregate tpot/tps (DIRECTIONAL ONLY — closed-loop client; "
        "citable metrics come from the sealed open-loop runner)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.first_n < 1 or args.topk < 1 or args.max_new_tokens < 1:
        parser.error("--first-n/--topk/--max-new-tokens must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")

    rows = []
    with args.requests_jsonl.open(encoding="utf-8") as source:
        for line in source:
            if len(rows) >= args.first_n:
                break
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < args.first_n:
        raise SystemExit(f"suite has only {len(rows)} rows, need {args.first_n}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "labels.jsonl"
    if labels_path.exists():
        raise SystemExit(f"refusing to overwrite {labels_path}")

    server_info = http_json(args.url.rstrip("/") + "/server_info")
    (args.output_dir / "server_info.json").write_text(
        json.dumps(server_info, indent=1) + "\n", encoding="utf-8"
    )

    def generate_one(position_row):
        position, row = position_row
        payload = {
            "input_ids": row["prompt"],
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "max_new_tokens": args.max_new_tokens,
            },
            "return_logprob": True,
            "top_logprobs_num": args.topk,
        }
        if args.stream_timing:
            response, ttft_s, e2e_s = http_stream_generate(
                args.url.rstrip("/") + "/generate", payload
            )
            return position, row, response, ttft_s, e2e_s
        response = http_json(args.url.rstrip("/") + "/generate", payload)
        return position, row, response, None, None

    # Wall clock starts BEFORE submission: executor.map schedules work
    # eagerly, so a later t0 would overstate achieved_rps/output_tps.
    wall_t0 = time.monotonic()
    if args.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.concurrency)
        results_iter = executor.map(generate_one, enumerate(rows))
    else:
        results_iter = map(generate_one, enumerate(rows))

    n_empty = 0
    timings = []
    with labels_path.open("w", encoding="utf-8") as sink:
        for position, row, response, ttft_s, e2e_s in results_iter:
            request_id = row["request_id"]
            text = response["text"]
            meta = response["meta_info"]
            token_logprobs = meta.get("output_token_logprobs") or []
            top_logprobs = meta.get("output_top_logprobs") or []
            record = {
                "request_id": request_id,
                "suite_pos": position,
                "prompt_tokens": len(row["prompt"]),
                "text": text,
                "finish_empty": not text.strip(),
                "completion_tokens": meta.get("completion_tokens"),
                "finish_reason": meta.get("finish_reason"),
                # RAW first-position entries — format parsed at diff time.
                "first_token_logprob_raw": token_logprobs[0] if token_logprobs else None,
                "first_top_logprobs_raw": top_logprobs[0] if top_logprobs else None,
                "ttft_s": ttft_s,
                "e2e_s": e2e_s,
            }
            if ttft_s is not None:
                timings.append(
                    (ttft_s, e2e_s, record["completion_tokens"] or 0)
                )
            n_empty += int(record["finish_empty"])
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(
                f"{position + 1}/{len(rows)} {request_id} "
                f"empty={record['finish_empty']} toks={record['completion_tokens']}",
                file=sys.stderr,
                flush=True,
            )

    wall_s = time.monotonic() - wall_t0
    summary = {
        "n_items": len(rows),
        "n_empty": n_empty,
        "empty_rate": n_empty / len(rows),
        "wall_s": round(wall_s, 2),
        "requests_jsonl": str(args.requests_jsonl),
        "first_n": args.first_n,
        "max_new_tokens": args.max_new_tokens,
        "topk": args.topk,
        "concurrency": args.concurrency,
    }
    if timings:
        ttfts = sorted(t[0] for t in timings)
        e2es = sorted(t[1] for t in timings)
        tpots = sorted(
            (t[1] - t[0]) / max(1, t[2] - 1) for t in timings
        )
        total_tokens = sum(t[2] for t in timings)
        def pct(vals, q):
            # Nearest-rank: ceil(q*n)-1 (int(n*q) picks the max at p99/n=100).
            rank = max(0, math.ceil(q * len(vals)) - 1)
            return round(vals[min(len(vals) - 1, rank)], 4)
        summary["directional_timing_closed_loop"] = {
            "NOTE": "closed-loop client-side timing — DIRECTIONAL ONLY, never headline",
            "achieved_rps": round(len(timings) / wall_s, 3),
            "output_tps": round(total_tokens / wall_s, 1),
            "ttft_s": {"mean": round(sum(ttfts) / len(ttfts), 4), "p50": pct(ttfts, 0.5), "p99": pct(ttfts, 0.99)},
            "tpot_s": {"mean": round(sum(tpots) / len(tpots), 5), "p50": pct(tpots, 0.5), "p99": pct(tpots, 0.99)},
            "e2e_s": {"mean": round(sum(e2es) / len(e2es), 3), "p50": pct(e2es, 0.5), "p99": pct(e2es, 0.99)},
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    print(f"empties {n_empty}/{len(rows)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
