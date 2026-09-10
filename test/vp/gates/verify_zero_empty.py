#!/usr/bin/env python3
"""HARD GATE: a natural-lane cell must contain ZERO empty generations.

[D-628, owner-ordered 2026-09-10] The empty-generation defect survived from D-066 (2026-08-09)
to now -- through two campaigns and every existing gate -- because nothing ever asserted the
one thing that would have caught it. Three separate blind spots kept it invisible:

1. **The equal-work lane cannot show it.** Equal-work sets ``ignore_eos``, so a first-token
   EOS is ignored and the budget is filled. Every headline performance cell is equal-work, so
   the defect was structurally unobservable there. This gate therefore applies to the
   NATURAL lane, and refuses to be run against an equal-work artifact.

2. **lm-eval's extraction filter hides it.** gsm8k's filter rewrites an empty generation as
   the literal string ``"[invalid]"``, which is not empty. Counting ``filtered_resps`` reports
   0.0% on a run whose raw responses are 51.6% empty (measured 2026-09-10; D-070 recorded the
   same trap in August). This gate reads RAW text only.

3. **finish_reason does not discriminate.** ``matched: 128009`` (``<|eot_id|>``) is also the
   NORMAL healthy ending -- 82% of a clean run ends that way. The discriminator is an empty
   body, equivalently ``completion_tokens == 1``.

Usage:
    verify_zero_empty.py --artifact cell.jsonl            # performance lane (serving cell)
    verify_zero_empty.py --lmeval-dir out/                 # quality lane (lm-eval samples)
    [--max-empty N]   allow N empties (default 0 -- the gate is ZERO by order)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def _flatten(value):
    while isinstance(value, list) and value:
        value = value[0]
    return str(value or "")


def _from_lmeval(directory: Path) -> tuple[int, list[str]]:
    files = sorted(glob.glob(f"{directory}/**/samples_*.jsonl", recursive=True))
    if not files:
        sys.exit(f"FATAL: no lm-eval samples under {directory}")
    rows = [json.loads(line) for line in open(files[-1]) if line.strip()]
    # RAW resps, never filtered_resps -- see (2) above.
    empties = [r.get("doc_id") for r in rows if not _flatten(r.get("resps")).strip()]
    return len(rows), [str(d) for d in empties]


def _from_artifact(path: Path) -> tuple[int, list[str]]:
    record = json.loads(path.read_text().strip().splitlines()[-1])
    if record.get("ignore_eos") or (record.get("generation_policy") or {}).get("ignore_eos"):
        sys.exit(
            "FATAL: this is an EQUAL-WORK artifact (ignore_eos). An immediate EOS is filled to "
            "budget there, so a zero-empty result would be vacuous. Run this gate on the "
            "NATURAL lane."
        )
    texts = record.get("generated_texts")
    if texts is None:
        sys.exit(
            "FATAL: artifact has no generated_texts. The gate cannot certify an artifact that "
            "did not retain its output -- do not pass it by defaulting to zero."
        )
    ids = record.get("request_ids") or list(range(len(texts)))
    empties = [str(ids[i]) for i, t in enumerate(texts) if not str(t).strip()]
    return len(texts), empties


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--lmeval-dir", type=Path)
    ap.add_argument("--max-empty", type=int, default=0)
    args = ap.parse_args()

    if bool(args.artifact) == bool(args.lmeval_dir):
        sys.exit("FATAL: pass exactly one of --artifact or --lmeval-dir")

    total, empties = (_from_artifact(args.artifact) if args.artifact
                      else _from_lmeval(args.lmeval_dir))
    if total == 0:
        sys.exit("FATAL: zero responses examined -- an empty cell is not a passing cell")

    share = 100.0 * len(empties) / total
    print(f"responses examined : {total}")
    print(f"empty generations  : {len(empties)} ({share:.2f}%)  [limit {args.max_empty}]")

    if len(empties) > args.max_empty:
        print(f"\nREFUSING: {len(empties)} empty generations. A request that returns no text is "
              "lost output, not a low score -- it silently deflates quality and inflates "
              "throughput-per-token. First offending ids: " + ", ".join(empties[:10]))
        print("Check the prompt protocol first: raw few-shot rendering reproduces this at "
              "42-52% while the chat protocol measures 0.0% on the same tree (D-628).")
        return 1

    print("\nOK: no empty generations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
