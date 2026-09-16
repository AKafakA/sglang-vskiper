#!/usr/bin/env python3
"""Reused (prefix-cached) share of prefill tokens per workload and arm, from the servers' own batch
logs ("Prefill batch ... #new-token: N, #cached-token: C"), summed over every repetition present.
Section 5 quotes it; both arms must agree within 0.1 pp or the run refuses.

  cache_share.py <headline root with rep*/<ds>/<arm>/server.log> --macros out.tex
"""
import argparse, glob, os, re, sys
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--macros", required=True); ap.add_argument("--treatment", default="integrated_it4")
a = ap.parse_args(); MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}; out = []
for ds, w in MW.items():
    share = {}
    for arm in ("upstream", a.treatment):
        new = cached = 0; logs = glob.glob(os.path.join(a.root, "rep*", ds, arm, "server.log"))
        if not logs: sys.exit(f"FATAL: no server.log for {ds}/{arm} under {a.root}")
        for f in logs:
            for line in open(f, errors="ignore"):
                m = re.search(r"#new-token: (\d+), #cached-token: (\d+)", line)
                if m: new += int(m.group(1)); cached += int(m.group(2))
        if new + cached == 0: sys.exit(f"FATAL: no prefill batches logged for {ds}/{arm}")
        share[arm] = 100.0 * cached / (new + cached)
    if abs(share["upstream"] - share[a.treatment]) > 0.1:
        sys.exit(f"FATAL {ds}: cached share differs between arms by more than 0.1 pp: {share}")
    out.append(f"\\newcommand{{\\vpCache{w}}}{{{share['upstream']:.1f}}}")
    print(f"  {ds:8s} cached share {share['upstream']:.1f}% (upstream) {share[a.treatment]:.1f}% (treatment), {len(logs)} reps")
open(a.macros, "w").write("\n".join(out) + "\n"); print("wrote", a.macros)
