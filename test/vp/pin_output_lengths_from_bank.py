#!/usr/bin/env python3
"""Pin per-request output lengths from a natural-length bank ( dual banks).

Emits a length-controlled copy of a frozen suite: every decode request's
``output_len`` is set from the bank (``{"lengths": {request_id: tokens}}``)
with ``ignore_eos`` + greedy — the exact body mutation
``build_fair_output_workload.build_equal_work_rows`` performs, applied from a
declared BANK instead of 3-rep production artifacts. Purpose: cross-arm WORK
IDENTITY (GR-1a) for directional A/Bs whose length source is the vPipe or
P-def natural distribution (owner instruction 2026-08-19: served-length
differences are a known confound; performance A/Bs pin lengths).

Fail-closed: every request must have a bank entry; every pinned length must
fit the remaining context. The output summary records the bank path + sha256
so the length source travels with the suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--output-requests", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    outputs = (args.output_requests, args.output_metadata, args.output_summary)
    inputs = (args.requests, args.metadata, args.bank)
    resolved_out = [p.resolve() for p in outputs]
    if len(set(resolved_out)) != len(resolved_out):
        raise SystemExit("output paths must be distinct")
    for out in outputs:
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        if out.resolve() in {p.resolve() for p in inputs}:
            raise SystemExit(f"output {out} aliases an input")

    bank_bytes = args.bank.read_bytes()
    bank = json.loads(bank_bytes)
    lengths = bank["lengths"]

    rows = read_jsonl(args.requests)
    metadata = read_jsonl(args.metadata)
    if len(metadata) != len(rows):
        raise SystemExit("request and metadata counts differ")
    missing = [
        str(row["request_id"]) for row in rows
        if str(row["request_id"]) not in lengths
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} request_ids absent from the bank "
            f"(first: {missing[:3]}) — wrong bank or wrong suite"
        )

    bank_sha = hashlib.sha256(bank_bytes).hexdigest()
    pinned = []
    pinned_metadata = []
    pinned_lengths = []
    for row, meta_row in zip(rows, metadata):
        if str(meta_row.get("request_id")) != str(row.get("request_id")):
            raise SystemExit("metadata order does not match requests")
        request_id = str(row["request_id"])
        raw_selected = lengths[request_id]
        if type(raw_selected) is not int or raw_selected < 1:
            raise SystemExit(
                f"{request_id}: bank length must be a positive JSON integer, "
                f"got {raw_selected!r}"
            )
        selected = raw_selected
        prompt_len = int(meta_row.get("prompt_len") or 0)
        if prompt_len <= 0 or prompt_len != len(row["prompt"]):
            raise SystemExit(f"{request_id}: prompt length metadata mismatch")
        #: the window is the MODEL's (the suite's per-request `context_length`, 40960 for Qwen3-4B), not a
        # default; the flag remains the fallback for suites frozen before the field existed.
        remaining = int(meta_row.get("context_length") or args.context_length) - prompt_len
        if selected > remaining:
            raise SystemExit(
                f"{request_id}: bank length {selected} exceeds remaining "
                f"context {remaining}"
            )
        out = dict(row)
        body = dict(out.get("extra_request_body") or {})
        body.pop("stop", None)
        body["temperature"] = 0.0
        body["top_p"] = 1.0
        body["frequency_penalty"] = 0.0
        body["ignore_eos"] = True
        out["output_len"] = selected
        out["extra_request_body"] = body
        pinned.append(out)
        meta_out = dict(meta_row)
        meta_out.update(
            {
                "requested_output_len": selected,
                "output_policy": "bank_pinned_natural_length",
                "bank_pinned_output_len": selected,
                "bank_sha256": bank_sha,
                "fixed_output_tokens": None,
                "quality_eligible": False,
                "frequency_penalty": 0.0,
            }
        )
        pinned_metadata.append(meta_out)
        pinned_lengths.append(selected)

    with args.output_requests.open("w", encoding="utf-8") as sink:
        for row in pinned:
            sink.write(json.dumps(row) + "\n")
    with args.output_metadata.open("w", encoding="utf-8") as sink:
        for row in pinned_metadata:
            sink.write(json.dumps(row) + "\n")
    summary = {
        "policy": "bank_pinned_output_lengths_ignore_eos_greedy",
        "source_requests": str(args.requests),
        "bank": str(args.bank),
        "bank_sha256": bank_sha,
        "output_policy": "bank_pinned_natural_length",
        "bank_meta": bank.get("meta"),
        "n": len(pinned),
        "length_min": min(pinned_lengths),
        "length_mean": round(sum(pinned_lengths) / len(pinned_lengths), 1),
        "length_max": max(pinned_lengths),
        "note": (
            "work-identity instrument for directional A/Bs; the length "
            "source (vPipe vs P-def natural bank) is a DECLARED choice"
        ),
    }
    args.output_summary.write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("n", "length_mean", "bank_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
