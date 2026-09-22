#!/usr/bin/env python3
"""Reused (prefix-cached) share of prefill tokens per workload and arm, from the servers' own batch
logs ("Prefill batch ... #new-token: N, #cached-token: C"), summed over every repetition present.
Section 5 quotes it. [v1.8] The served design keys the prefix cache by body (a prefix computed in one mode is reused only by
requests pinned to that mode), so the arms may differ slightly: the tool emits both arms' shares and the largest between-arm
difference (\vpCacheArmDiffMax) for the text to cite, instead of asserting the v1.7 sentence "within 0.1 pp".

  cache_share.py <headline root with rep*/<ds>/<arm>/server.log> --macros out.tex
"""
import argparse, glob, os, re, sys
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--macros", required=True); ap.add_argument("--treatment", default="integrated_it4"); ap.add_argument("--baseline", default="upstream")   # [v1.6] D-824
a = ap.parse_args(); MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}; out = []; diffs = []
for ds, w in MW.items():
    share = {}
    for arm in (a.baseline, a.treatment):
        new = cached = 0; logs = glob.glob(os.path.join(a.root, "rep*", ds, arm, "server.log"))
        if not logs: sys.exit(f"FATAL: no server.log for {ds}/{arm} under {a.root}")
        for f in logs:
            for line in open(f, errors="ignore"):
                m = re.search(r"#new-token: (\d+), #cached-token: (\d+)", line)
                if m: new += int(m.group(1)); cached += int(m.group(2))
        if new + cached == 0: sys.exit(f"FATAL: no prefill batches logged for {ds}/{arm}")
        share[arm] = 100.0 * cached / (new + cached)
    diffs.append(abs(share[a.baseline] - share[a.treatment]))
    out.append(f"\\newcommand{{\\vpCache{w}}}{{{share[a.baseline]:.1f}}}")
    out.append(f"\\newcommand{{\\vpCache{w}Vs}}{{{share[a.treatment]:.1f}}}")
    print(f"  {ds:8s} cached share {share[a.baseline]:.1f}% (upstream) {share[a.treatment]:.1f}% (treatment), {len(logs)} reps")
out.append(f"\\newcommand{{\\vpCacheArmDiffMax}}{{{max(diffs):.1f}}}")
print(f"  largest between-arm difference {max(diffs):.2f} pp")
open(a.macros, "w").write("\n".join(out) + "\n"); print("wrote", a.macros)
