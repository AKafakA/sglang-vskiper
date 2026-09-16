#!/usr/bin/env python3
"""Stage 3: the natural-lane summary `natural_lane_table.py` reads, built from the harvest cells.

Each natural-lane cell is one harvest root -- a stack serving a frozen suite to its own end-of-sequence, one
repetition -- and every number the appendix table prints comes from that cell's own `score.json`, the artifact the
accounting gate signed. Nothing here recomputes a metric from the per-request arrays, so the summary cannot drift
from the gated cell.

Root naming decides the arm, which is how the harvest drivers named them:
    harvest-<ds>-<label>                         -> vskipper     (the served arm's own lane)
    harvest-upstream-<ds>-<label>                -> upstream     (Base + upstream)
    harvest-<arm>-<ds>-<label>                   -> alwaysroute  (step 13, arm integrated_alwaysskip)

usage: natural_lane_summarize.py <harvest-root> [...] --out summary.json
       natural_lane_summarize.py <campaign-dir>/harvest-* --out summary.json
"""
import argparse, json, pathlib, re, sys

DATASETS = ("gsm8k", "bbh_cot", "coqa")
ALWAYS = "integrated_alwaysskip"


def classify(name: str) -> tuple[str, str, str]:
    """(arm, dataset, rate_label) from a harvest root's directory name."""
    m = re.fullmatch(r"harvest-(?:(upstream|" + ALWAYS + r")-)?(" + "|".join(DATASETS) + r")-(r[0-9p]+)", name)
    if not m:
        raise SystemExit(f"cannot classify harvest root {name!r}")
    prefix, ds, label = m.group(1), m.group(2), m.group(3)
    arm = {None: "vskipper", "upstream": "upstream", ALWAYS: "alwaysroute"}[prefix]
    return arm, ds, label


def summarize(root: pathlib.Path) -> dict:
    """The cell's own accounting, read from its score.json."""
    scores = sorted(root.glob("cell/*.score.json")) or sorted(root.glob("*.score.json"))
    if len(scores) != 1:
        raise SystemExit(f"{root.name}: expected exactly one score.json, found {len(scores)}")
    rec = json.loads(scores[0].read_text())["scores"][0]
    p = rec["performance"]
    completed = p["completed"]
    if not completed:
        raise SystemExit(f"{root.name}: cell completed 0 requests -- not a measurement")
    return {"completed": completed,
            "scored": rec["scored_requests"],
            "cap_hits": p["cap_hit_count"],
            "duration_s": p["duration"],
            "mean_out_tokens": p["total_output_tokens"] / completed,
            "mean_e2e_s": p["mean_e2e_latency_ms"] / 1000.0,
            "mean_ttft_ms": p["mean_ttft_ms"],
            "mean_tpot_ms": p["mean_tpot_ms"],
            "completions_per_s": p["request_throughput"],
            "rate": root.name.rsplit("-", 1)[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out: dict = {}
    for spec in args.roots:
        root = pathlib.Path(spec)
        arm, ds, label = classify(root.name)
        cell = out.setdefault(ds, {}).setdefault(label, {})
        if arm in cell:
            raise SystemExit(f"{ds}/{label}: two roots claim arm {arm}")
        cell[arm] = summarize(root)
    n = sum(len(a) for d in out.values() for a in d.values())
    open(args.out, "w").write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"  {n} natural-lane cells across {len(out)} datasets -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
