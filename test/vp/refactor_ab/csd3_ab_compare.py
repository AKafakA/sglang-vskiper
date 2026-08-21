"""Compare the two CSD3 arms. WORK IDENTITY IS A GATE: if the arms did not do
the same amount of work, the timings are not comparable and no delta is
printed (GR-1a). Saturation is also a gate -- a saturated run measures the
queue, not the code."""
import json, pathlib, statistics, sys

RES = pathlib.Path(sys.argv[1])

def reps(arm):
    out = []
    for r in (1, 2, 3):
        p = RES / arm / f"rep{r}.json"
        if p.exists():
            out.append(json.load(open(p))["summary"])
    return out

A, B = reps("refactored"), reps("frozen")
print("\n" + "=" * 72)
print("CSD3 A/B  refactored (sglang.srt.vpipe)  vs  frozen (sglang.srt.vp @ c3a1302668)")
print("=" * 72)
if not A or not B:
    print(f"MISSING REPS: refactored={len(A)} frozen={len(B)} -- cannot compare")
    raise SystemExit(2)

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
