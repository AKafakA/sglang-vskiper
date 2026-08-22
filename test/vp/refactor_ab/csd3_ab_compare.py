"""Compare the two CSD3 arms. WORK IDENTITY IS A GATE: if the arms did not do
the same amount of work, the timings are not comparable and no delta is
printed (GR-1a). Saturation is also a gate -- a saturated run measures the
queue, not the code."""
import json, pathlib, statistics, sys

RES = pathlib.Path(sys.argv[1])

EXPECT_REPS = 3

def reps(arm):
    """Collect the arm's reps. MISSING REPS ARE A FAILURE, NOT A SMALLER SAMPLE.

    Collecting whatever happens to be on disk let one rep be compared against
    three: the runner `continue`s past an arm whose server failed, and a client
    that died on rep 2 leaves rep 1 behind. Both produce a partial arm that the
    old `if not A or not B` accepted, so a mean over one sample was reported
    against a mean over three as though they were the same instrument.
    """
    out, missing = [], []
    for r in range(1, EXPECT_REPS + 1):
        p = RES / arm / f"rep{r}.json"
        if not p.exists():
            missing.append(f"rep{r}")
            continue
        out.append(json.load(open(p))["summary"])
    return out, missing

A, miss_a = reps("refactored")
B, miss_b = reps("frozen")
print("\n" + "=" * 72)
print("CSD3 A/B  refactored (sglang.srt.vpipe)  vs  frozen (sglang.srt.vp @ c3a1302668)")
print("=" * 72)
print(f"\nGATE completeness: expect {EXPECT_REPS} reps/arm; "
      f"refactored={len(A)} frozen={len(B)}")
if miss_a or miss_b:
    print(f"  -> *** INCOMPLETE: refactored missing {miss_a or 'none'}, "
          f"frozen missing {miss_b or 'none'} ***")
    print("\nONE OR MORE GATES FAILED -- no deltas reported.")
    raise SystemExit(2)
print("  -> PASS")

def agg(rs, path):
    vals = []
    for r in rs:
        cur = r
        for k in path:
            cur = cur[k]
        vals.append(float(cur))
    return vals

# ---- gate 1: errors ----
errs = sum(r["errors"] for r in A) + sum(r["errors"] for r in B)
print(f"\nGATE errors: refactored={sum(r['errors'] for r in A)} frozen={sum(r['errors'] for r in B)}"
      f"  -> {'PASS' if errs == 0 else 'FAIL'}")

# ---- gate 2: work identity ----
ta = [r["total_output_tokens"] for r in A]
tb = [r["total_output_tokens"] for r in B]
print(f"GATE work identity: refactored tokens={ta} frozen tokens={tb}")
work_ok = sorted(ta) == sorted(tb)
print(f"  -> {'EXACT' if work_ok else '*** MISMATCH -- timings NOT comparable ***'}")

# Request-count identity. Token identity alone can be satisfied by arms that
# completed different numbers of requests, and a per-request mean over a
# different denominator is a different statistic.
oka = [r["ok"] for r in A]
okb = [r["ok"] for r in B]
count_ok = len(set(oka + okb)) == 1
print(f"GATE request count: refactored ok={oka} frozen ok={okb}")
print(f"  -> {'IDENTICAL' if count_ok else '*** DIFFERENT -- per-request means have different denominators ***'}")
work_ok = work_ok and count_ok

# ---- gate 3: saturation ----
acc_a = agg(A, ["achieved_rate"]); acc_b = agg(B, ["achieved_rate"])
off = A[0]["offered_rate"]
sat_ok = min(acc_a + acc_b) >= 0.85 * off
print(f"GATE saturation: offered={off} achieved refactored={acc_a} frozen={acc_b}")
print(f"  -> {'PASS (sub-saturation)' if sat_ok else '*** SATURATED -- deltas measure the queue, not the code ***'}")

if not (errs == 0 and work_ok and sat_ok):
    print("\nONE OR MORE GATES FAILED -- no deltas reported.")
    raise SystemExit(1)

# ---- full metric table ----
print("\nFull metric table (3 reps/arm, mean +- sample stdev):\n")
print(f"  {'metric':22s} {'refactored':>22s} {'frozen':>22s} {'delta':>10s}")
print("  " + "-" * 78)
METRICS = [
    ("ttft_ms mean", ["ttft_ms", "mean"]), ("ttft_ms p50", ["ttft_ms", "p50"]),
    ("ttft_ms p90", ["ttft_ms", "p90"]),
    ("tpot_ms mean", ["tpot_ms", "mean"]), ("tpot_ms p50", ["tpot_ms", "p50"]),
    ("tpot_ms p90", ["tpot_ms", "p90"]),
    ("achieved_rate", ["achieved_rate"]), ("wall_s", ["wall_s"]),
    ("total_output_tokens", ["total_output_tokens"]),
]
rows = []
for name, path in METRICS:
    va, vb = agg(A, path), agg(B, path)
    ma, mb = statistics.fmean(va), statistics.fmean(vb)
    sa = statistics.stdev(va) if len(va) > 1 else 0.0
    sb = statistics.stdev(vb) if len(vb) > 1 else 0.0
    d = (ma - mb) / mb * 100 if mb else 0.0
    rows.append((name, ma, sa, mb, sb, d))
    print(f"  {name:22s} {ma:14.3f}+-{sa:6.3f} {mb:14.3f}+-{sb:6.3f} {d:+9.2f}%")

print("\n  Deltas are refactored relative to frozen. Negative TTFT/TPOT = faster.")
worst = max(abs(r[5]) for r in rows if r[0].startswith(("ttft", "tpot")))
noisy = [r for r in rows if r[0].startswith(("ttft", "tpot")) and (r[2] + r[4]) > abs(r[1] - r[3])]
print(f"  Largest |delta| across latency metrics: {worst:.2f}%")
print(f"  Metrics whose combined stdev exceeds the arm difference (i.e. NOISE): "
      f"{[r[0] for r in noisy] or 'none'}")
print("\n  REGRESSION CHECK ONLY -- not a headline number. There is no frozen QPS")
print("  suite on CSD3; building one ad hoc would be a contract deviation.")
