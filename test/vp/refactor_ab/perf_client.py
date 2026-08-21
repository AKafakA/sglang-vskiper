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
            except Exception:
                pass
    e2e = time.perf_counter() - t0
    if ntok == 0 and last is not None:
        ntok = (last.get("meta_info") or {}).get("completion_tokens", 0)
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
