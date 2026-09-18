#!/usr/bin/env python3
"""EOS-margin telemetry for the routed-length-inflation debug (L1).

Greedy /generate with per-step top-k logprobs; for every decode step
records margin = best-EOS logprob - chosen-token logprob (<= 0 until
the request stops; None when no EOS id lands in the top-k window).
Banks one jsonl row per request with the full margin trajectory and
chosen token ids so paired arms can be diffed at their divergence
points offline. Debug instrument only — never a performance lane.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def parse_step_entry(entry) -> tuple[float, int]:
    """(logprob, token_id) from a raw output_token_logprobs element."""

    logprob, token_id = float(entry[0]), int(entry[1])
    return logprob, token_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--first-n", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--eos-token-ids",
        default="128009,128001",
        help="comma-separated ids treated as EOS (Llama-3: eot_id,eos)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.first_n <= 0 or args.max_new_tokens <= 0 or args.topk <= 0:
        raise SystemExit("first-n, max-new-tokens, and topk must be positive")
    if args.concurrency < 1:
        raise SystemExit("concurrency must be >= 1")
    try:
        eos_ids = {int(part) for part in args.eos_token_ids.split(",") if part}
    except ValueError as error:
        raise SystemExit(f"malformed --eos-token-ids: {error}") from error
    if not eos_ids or any(token_id < 0 for token_id in eos_ids):
        raise SystemExit("EOS token ids must be non-negative and non-empty")

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
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        response = http_json(args.url.rstrip("/") + "/generate", payload)
        return position, row, response

    wall_t0 = time.monotonic()
    if args.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.concurrency)
        results_iter = executor.map(generate_one, enumerate(rows))
    else:
        results_iter = map(generate_one, enumerate(rows))

    margins_path = args.output_dir / "eos_margins.jsonl"
    n_empty = 0
    completion_lengths = []
    sink = margins_path.open("w", encoding="utf-8")
    try:
        for position, row, response in results_iter:
            meta = response["meta_info"]
            token_logprobs = meta.get("output_token_logprobs")
            top_logprobs = meta.get("output_top_logprobs")
            completion_tokens = meta.get("completion_tokens")
            # Missing/short telemetry is a failure, never empty evidence.
            if not token_logprobs or not top_logprobs:
                raise SystemExit(
                    f"{row['request_id']}: server returned no logprob "
                    "telemetry (return_logprob not honored?)"
                )
            if len(top_logprobs) != len(token_logprobs):
                raise SystemExit(
                    f"{row['request_id']}: top/chosen step-count mismatch "
                    f"{len(top_logprobs)}/{len(token_logprobs)}"
                )
            if completion_tokens != len(token_logprobs):
                raise SystemExit(
                    f"{row['request_id']}: trajectory length "
                    f"{len(token_logprobs)} != completion_tokens "
                    f"{completion_tokens}"
                )
            margins = []
            chosen_ids = []
            for step, (chosen_raw, top_raw) in enumerate(
                zip(token_logprobs, top_logprobs)
            ):
                chosen_logprob, chosen_id = parse_step_entry(chosen_raw)
                chosen_ids.append(chosen_id)
                if not top_raw:
                    raise SystemExit(
                        f"{row['request_id']} step {step}: empty top-k list"
                    )
                eos_best = None
                chosen_seen_in_top = False
                for candidate in top_raw:
                    logprob, token_id = parse_step_entry(candidate)
                    if token_id == chosen_id and (
                        abs(logprob - chosen_logprob) < 1e-5
                    ):
                        chosen_seen_in_top = True
                    if token_id in eos_ids and (
                        eos_best is None or logprob > eos_best
                    ):
                        eos_best = logprob
                # Greedy alignment invariants: the chosen token must appear
                # in its own step's top list with a matching logprob, and no
                # EOS candidate may beat the chosen token unless chosen IS
                # EOS — violations mean shifted/reordered telemetry.
                if not chosen_seen_in_top:
                    raise SystemExit(
                        f"{row['request_id']} step {step}: chosen token "
                        f"{chosen_id} absent from its top-{args.topk} list "
                        "— trajectories misaligned"
                    )
                if (
                    eos_best is not None
                    and chosen_id not in eos_ids
                    and eos_best > chosen_logprob + 1e-5
                ):
                    raise SystemExit(
                        f"{row['request_id']} step {step}: EOS logprob "
                        f"{eos_best} exceeds chosen {chosen_logprob} under "
                        "greedy — telemetry inconsistent"
                    )
                if chosen_id in eos_ids:
                    eos_best = (
                        chosen_logprob
                        if eos_best is None
                        else max(eos_best, chosen_logprob)
                    )
                margins.append(
                    None if eos_best is None else eos_best - chosen_logprob
                )
            observed = [value for value in margins if value is not None]
            record = {
                "request_id": row["request_id"],
                "suite_pos": position,
                "prompt_tokens": len(row["prompt"]),
                "completion_tokens": completion_tokens,
                "finish_reason": meta.get("finish_reason"),
                "text_empty": not response["text"].strip(),
                "chosen_token_ids": chosen_ids,
                # Rounded for serialization only — the stats above use raw.
                "eos_margins": [
                    None if value is None else round(value, 4)
                    for value in margins
                ],
                "eos_in_topk_steps": len(observed),
                "max_eos_margin": max(observed) if observed else None,
                "near_eos_steps_gt_minus2": sum(
                    1 for value in observed if value > -2.0
                ),
            }
            n_empty += int(record["text_empty"])
            completion_lengths.append(len(chosen_ids))
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(
                f"{position + 1}/{len(rows)} {row['request_id']} "
                f"toks={len(chosen_ids)} eos_seen={len(observed)}",
                file=sys.stderr,
                flush=True,
            )
        sink.close()
    except BaseException:
        # Never leave a plausible-looking partial dataset behind.
        sink.close()
        margins_path.rename(margins_path.with_suffix(".jsonl.FAILED"))
        if args.concurrency > 1:
            executor.shutdown(wait=False, cancel_futures=True)
        raise

    summary = {
        "n": len(rows),
        "n_empty": n_empty,
        "topk": args.topk,
        "concurrency": args.concurrency,
        "eos_token_ids": sorted(eos_ids),
        "mean_completion_tokens": (
            sum(completion_lengths) / len(completion_lengths)
            if completion_lengths
            else 0.0
        ),
        "wall_s": round(time.monotonic() - wall_t0, 3),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
