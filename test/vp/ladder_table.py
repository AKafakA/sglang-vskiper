#!/usr/bin/env python3
"""Appendix table: the upstream Q* ladders that calibrate every load point in the paper.

One row per workload: the contiguous integer offered rates that were served, the knee Q*, the achieved output
token throughput at the knee and at the top rung, and the growth left above the knee. Reading the top rung rather
than the best rung is deliberate -- the question the row answers is "how much throughput is still on the table
above Q*", and the top rung is the furthest the ladder went.

Reads each rung's achieved throughput from the runner's own stdout summary ("Output token throughput (tok/s)"),
which is the number `qstar_int.py` applies its knee rule to, so the table and the knee cannot disagree. Works from
a clone with no host: the per-rung directories carry the summary, so the multi-gigabyte per-request arrays are not
needed for this table.

usage: ladder_table.py gsm8k=<root> bbh_cot=<root> coqa=<root> --knee gsm8k=11 ... --rows out.tex [--macros out.tex]
       (h100_gsm8k=<root> for the H100 replicate's own GSM8K ladder; a rung directory named r<N>-<suffix> is a
       retry that never served and is skipped, exactly as an unserved rung is)
"""
import argparse, pathlib, re, sys

NAMES = {"gsm8k": "GSM8K", "bbh_cot": "BBH", "coqa": "CoQA", "h100_gsm8k": "GSM8K (H100)"}
MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa", "h100_gsm8k": "HundredGsm"}
TPS = re.compile(r"Output token throughput \(tok/s\):\s*([0-9.]+)")


def curve(root: pathlib.Path) -> dict[int, float]:
    """{offered integer rate: achieved output tok/s} from the ladder's per-rung directories."""
    out: dict[int, float] = {}
    for d in sorted(root.glob("run-*/r*")):
        m = re.fullmatch(r"r(\d+)", d.name)
        if not m:
            continue
        logs = sorted(d.glob("*.stdout.log"))
        if not logs:
            continue                      # a rung staged but never served
        hit = TPS.search(logs[0].read_text(errors="replace"))
        if not hit:
            continue
        rate = int(m.group(1))
        if rate in out:
            raise SystemExit(f"{root.name}: rung r{rate} appears twice -- disambiguate the run directory")
        out[rate] = float(hit.group(1))
    if not out:
        raise SystemExit(f"{root.name}: no rung carried an output-throughput summary")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", metavar="dataset=path")
    ap.add_argument("--knee", action="append", required=True, metavar="dataset=Q*")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--macros")
    args = ap.parse_args()

    knees = {k: int(v) for k, v in (s.split("=", 1) for s in args.knee)}
    rows, macros = [], []
    for spec in args.roots:
        ds, path = spec.split("=", 1)
        c = curve(pathlib.Path(path))
        rungs = sorted(c)
        knee = knees[ds]
        if knee not in c:
            raise SystemExit(f"{ds}: knee {knee} is not a served rung ({rungs[0]}--{rungs[-1]})")
        at_knee, at_top = c[knee], c[rungs[-1]]
        growth = 100.0 * (at_top - at_knee) / at_knee
        rows.append(f"{NAMES.get(ds, ds)} & {rungs[0]}--{rungs[-1]} & {knee} & "
                    f"{at_knee:.0f} & {at_top:.0f} & {growth:+.1f} \\\\")
        p = MW.get(ds, ds.title())
        macros += [f"\\newcommand{{\\vpLadder{p}Knee}}{{{knee}}}",
                   f"\\newcommand{{\\vpLadder{p}Rungs}}{{{rungs[0]}--{rungs[-1]}}}",
                   f"\\newcommand{{\\vpLadder{p}AtKnee}}{{{at_knee:.0f}}}",
                   f"\\newcommand{{\\vpLadder{p}AtTop}}{{{at_top:.0f}}}",
                   f"\\newcommand{{\\vpLadder{p}Growth}}{{{growth:+.1f}}}"]
    open(args.rows, "w").write("\n".join(rows) + "\n")
    if args.macros:
        open(args.macros, "w").write("\n".join(macros) + "\n")
    print(f"  {len(rows)} ladder rows -> {args.rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
