#!/usr/bin/env python3
"""Timed client for the paired refactor A/B: per-request TTFT / TPOT / E2E.

Deliberately minimal and identical across arms. Streams so TTFT is the real
first-token latency rather than end-to-end. Records the served token counts so
WORK IDENTITY between arms can be checked before any timing is compared -- an
unmatched token mass invalidates a cross-arm comparison (GR-1a).

OPEN-LOOP: arrivals are Poisson at a fixed offered rate, so the LOAD CONTROL is
the rate -- never a client concurrency cap. A closed-loop cap makes the client
throttle itself to the server's speed, which hides exactly the latency
differences a regression check is looking for (the evaluation law's open-loop
requirement). Every submitted request is awaited and counted; nothing is
dropped or tail-filtered.

Greedy decode with a fixed max_new_tokens and ignore_eos so both arms are forced
to produce the same amount of work per request.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def one(url, prompt, max_new):
    # The frozen suites carry `prompt` as TOKEN IDS, so it goes as input_ids --
    # sending it as `text` is a 400. Matches tools/fdpre_label_probe.py.
    body = json.dumps({
        "input_ids": prompt,
        "sampling_params": {
            "temperature": 0.0, "top_p": 1.0, "frequency_penalty": 0.0,
            "max_new_tokens": max_new, "ignore_eos": True,
        },
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url + "/generate", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    last = None
    malformed = 0
    last_err = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload in ("", "[DONE]"):
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            try:
                obj = json.loads(payload)
                m = obj.get("meta_info") or {}
                if "completion_tokens" in m:
                    ntok = m["completion_tokens"]
                last = obj
            except Exception as exc:
                # Do NOT swallow this. A malformed SSE payload used to be
                # ignored, so a request that decoded no valid token event still
                # returned normally and was counted in `ok`.
                malformed += 1
                last_err = f"malformed SSE payload: {str(exc)[:60]}"
    e2e = time.perf_counter() - t0
    if ntok == 0 and last is not None:
        ntok = (last.get("meta_info") or {}).get("completion_tokens", 0)

    # sampling_params sets ignore_eos=True with a fixed max_new_tokens, so a
    # SUCCESSFUL request produces exactly that many tokens. Anything else is a
    # failed request, not a fast one. Without this, two arms both returning
    # empty responses give positive request counts, zero errors, identical zero
    # token totals and adequate achieved rate -- and the comparator reports
    # latency deltas for no inference work at all.
    if malformed:
        raise RuntimeError(f"{malformed} malformed SSE payload(s); {last_err}")
    if ntok != max_new:
        raise RuntimeError(
            f"incomplete generation: {ntok} tokens, expected exactly {max_new} "
            f"(ignore_eos=True). An empty or truncated response is a failure."
        )
    tpot = (e2e - (ttft or 0.0)) / max(1, ntok - 1)
    return ttft or e2e, tpot, e2e, ntok


ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--requests-jsonl", required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--rate", type=float, default=4.0,
                help="offered requests/sec (Poisson). THE load control.")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--max-new-tokens", type=int, default=128)
ap.add_argument("--out", required=True)
a = ap.parse_args()

prompts = []
with open(a.requests_jsonl) as fh:
    for line in fh:
        if not line.strip():
            continue
        rec = json.loads(line)
        p = rec.get("prompt")          # token ids
        if p:
            prompts.append(p)
        if len(prompts) >= a.n:
            break

# EXACTLY the requested count, or refuse. Reading "up to n" means a workload
# with fewer valid prompts silently shrinks the run -- and because BOTH arms
# shrink the same way, zero errors, identical tokens, identical request counts
# and sub-saturation all still hold. The comparator would report a clean A/B
# over work neither arm was asked to do. Equal work is not the same as the
# intended work.
if len(prompts) != a.n:
    raise SystemExit(
        f"FATAL: requested --n {a.n} but {a.requests_jsonl} yielded "
        f"{len(prompts)} usable prompts. Refusing: a symmetrically truncated "
        f"A/B passes every gate while measuring less work than intended."
    )

# Poisson arrival schedule, identical across arms for the same seed.
rng = random.Random(a.seed)
gaps = [rng.expovariate(a.rate) for _ in prompts]
offsets, acc = [], 0.0
for g in gaps:
    acc += g
    offsets.append(acc)

res = []
lock = threading.Lock()


def fire(i, p, due):
    delay = due - (time.perf_counter() - wall0)
    if delay > 0:
        time.sleep(delay)
    try:
        ttft, tpot, e2e, ntok = one(a.url, p, a.max_new_tokens)
        with lock:
            res.append({"i": i, "ttft": ttft, "tpot": tpot, "e2e": e2e, "toks": ntok})
    except Exception as exc:
        with lock:
            res.append({"i": i, "error": str(exc)[:120]})


wall0 = time.perf_counter()
ts = [threading.Thread(target=fire, args=(i, p, off), daemon=True)
      for (i, p), off in zip(enumerate(prompts), offsets)]
[t.start() for t in ts]
[t.join() for t in ts]          # every submitted request finishes and is counted
wall = time.perf_counter() - wall0

ok = [r for r in res if "error" not in r]
errs = [r for r in res if "error" in r]
ttfts = [r["ttft"] * 1000 for r in ok]
tpots = [r["tpot"] * 1000 for r in ok]
out = {
    # requested_n travels with the artifact so the comparator can gate on the
    # count that was ASKED for, not merely on the two arms agreeing.
    "requested_n": a.n,
    "max_new_tokens": a.max_new_tokens,
    "expected_total_output_tokens": a.n * a.max_new_tokens,
    "requests": len(res), "ok": len(ok), "errors": len(errs),
    "offered_rate": a.rate, "achieved_rate": round(len(ok) / wall, 3) if wall else 0,
    "total_output_tokens": sum(r["toks"] for r in ok),
    "wall_s": round(wall, 3),
    "ttft_ms": {"mean": round(statistics.fmean(ttfts), 2) if ttfts else 0,
                "p50": round(pct(ttfts, 50), 2), "p90": round(pct(ttfts, 90), 2)},
    "tpot_ms": {"mean": round(statistics.fmean(tpots), 2) if tpots else 0,
                "p50": round(pct(tpots, 50), 2), "p90": round(pct(tpots, 90), 2)},
    "error_samples": [r["error"] for r in errs[:3]],
}
with open(a.out, "w") as fh:
    json.dump({"summary": out, "per_request": res}, fh, indent=1)
print(json.dumps(out, indent=1))

# EXIT NON-ZERO when anything failed. The process used to return 0 even if every
# request errored, so a caller checking only the exit status -- including this
# repository's own per-repetition abort -- could never detect a failed run.
if errs:
    raise SystemExit(f"{len(errs)} of {len(res)} requests failed")
if out["total_output_tokens"] != out["expected_total_output_tokens"]:
    raise SystemExit(
        f"token mass {out['total_output_tokens']} != expected "
        f"{out['expected_total_output_tokens']}"
    )
