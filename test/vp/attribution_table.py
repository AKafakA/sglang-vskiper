#!/usr/bin/env python3
"""Native loop-attribution table ( rule; paper banks appendix).

Reads the loopers set (`set.json`: request ids per group, drawn from the served cells) and the native
PyTorch runs produced by `native_loop_check_v2.py` (`native_<tag>.jsonl`, one record per prompt:
`loop`, `stopped`, `output_len`). Emits one LaTeX row per (group, model): prompts, loops, loop rate,
generations that hit the context window, mean and p90 output length. A run is used only when it is
COMPLETE (every prompt of every group present) so the table never carries a partial column; the
macros `\vpAttr<Model><Group>Loops` / `...Pct` / `...N` feed the two sentences of prose.

usage: attribution_table.py <loopers dir> --rows out.tex --macros out.tex [--model raw=Llama-3-8B-Instruct...]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

GROUPS = [("vskipper_only", "looped under \\sys{} only", "Vsk"),
          ("production_only", "looped under upstream only", "Prod"),
          ("control", "control (never looped)", "Ctl")]
DEFAULT_MODELS = [("raw", "Llama-3-8B-Instruct (base)", "Raw"),
                  ("fd_b1", "FlexiDepth, batch 1", "FdBOne"),
                  ("fd", "FlexiDepth, batch 16", "FdBSixteen")]


def load_records(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a truncated last line of a run still writing
                continue
            out[rec["request_id"]] = rec
    return out


def rid(entry) -> str:
    return entry if isinstance(entry, str) else entry["request_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("loopers", type=Path)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--macros", type=Path, required=True)
    ap.add_argument("--model", action="append", default=[], help="tag=label (default: raw, fd_b1, fd)")
    ap.add_argument("--population", type=Path, default=None, help="dir of the seeded random-sample run (native_fd_pop500.jsonl + population.json): population rows")
    a = ap.parse_args()

    groups = json.loads((a.loopers / "set.json").read_text())
    ids = {g: [rid(e) for e in groups[g]] for g, _, _ in GROUPS if g in groups}
    models = DEFAULT_MODELS if not a.model else [(m.split("=", 1)[0], m.split("=", 1)[1], m.split("=", 1)[0].title().replace("_", "")) for m in a.model]

    rows, macros, used = [], [], []
    for tag, label, mkey in models:
        f = a.loopers / f"native_{tag}.jsonl"
        if not f.exists():
            print(f"  {tag}: no file, skipped", file=sys.stderr)
            continue
        recs = load_records(f)
        missing = {g: sum(1 for i in ids[g] if i not in recs) for g in ids}
        if any(missing.values()):
            print(f"  {tag}: INCOMPLETE ({sum(len(recs.get(i, {})) > 0 for g in ids for i in ids[g])} of {sum(len(v) for v in ids.values())} prompts) -> not emitted", file=sys.stderr)
            continue
        used.append(tag)
        first = True
        for g, gdesc, gkey in GROUPS:
            if g not in ids:
                continue
            rs = [recs[i] for i in ids[g]]
            n = len(rs)
            loops = sum(1 for r in rs if r["loop"] or not r["stopped"])
            cap = sum(1 for r in rs if not r["stopped"])
            lens = sorted(r["output_len"] for r in rs)
            mean = statistics.mean(lens)
            p90 = lens[int(round(0.9 * (n - 1)))]
            pct = 100.0 * loops / n
            head = label if first else ""
            first = False
            rows.append(f"    {head} & {gdesc} & {n} & {loops} & {pct:.1f} & {cap} & {mean:.0f} & {p90} \\\\")
            for name, val in (("N", str(n)), ("Loops", str(loops)), ("Pct", f"{pct:.1f}"), ("Cap", str(cap)), ("Mean", f"{mean:.0f}"), ("Pninety", str(p90))):
                macros.append(f"\\newcommand{{\\vpAttr{mkey}{gkey}{name}}}{{{val}}}")
        rows.append("    \\midrule")
    # the seeded random sample of the whole population (2026-09-14, native_loop_check_v2_2: early stop on detected repetition)
    if a.population:
        import math
        pg = json.loads((a.population / "population.json").read_text()); vsk = set(pg.get("served_loopers", []))
        for tag, label, pkey in (("raw", "Llama-3-8B-Instruct (base), batch 16, sampled", "Raw"), ("fd", "FlexiDepth, batch 16, sampled", "Fd")):
            f = a.population / f"native_{tag}_pop500.jsonl"
            if not f.exists():
                continue
            pop = load_records(f)
            if len(pop) < 500:
                print(f"  population {tag}: INCOMPLETE ({len(pop)}/500) -> not emitted", file=sys.stderr); continue
            first = True
            for gname, gdesc, gkey in (("population", "random sample of the cell", "Pop"), ("served_loopers", "of which looped under \\sys{}", "PopVsk"), ("never_looped", "of which never looped", "PopCtl")):
                rs = [r for i, r in pop.items() if gname == "population" or (i in vsk) == (gname == "served_loopers")]
                n = len(rs); loops = sum(1 for r in rs if r["loop"]); cap = sum(1 for r in rs if r.get("stop_reason") == "window")
                lens = sorted(r["output_len"] for r in rs); mean = statistics.mean(lens); p90 = lens[int(round(0.9 * (n - 1)))]; pct = 100.0 * loops / n
                rows.append(f"    {label if first else ''} & {gdesc} & {n} & {loops} & {pct:.1f} & {cap} & {mean:.0f} & {p90} \\\\"); first = False
                for name, val in (("N", str(n)), ("Loops", str(loops)), ("Pct", f"{pct:.1f}"), ("Cap", str(cap)), ("Mean", f"{mean:.0f}"), ("Pninety", str(p90))):
                    macros.append(f"\\newcommand{{\\vpAttr{pkey}{gkey}{name}}}{{{val}}}")
            pp = sum(1 for r in pop.values() if r["loop"]) / len(pop); macros.append(f"\\newcommand{{\\vpAttr{pkey}PopCi}}{{{100 * 1.96 * math.sqrt(pp * (1 - pp) / len(pop)):.1f}}}")
            rows.append("    \\midrule")
    if rows and rows[-1].strip() == "\\midrule":
        rows.pop()
    a.rows.write_text("\n".join(rows) + "\n")
    macros.append(f"\\newcommand{{\\vpAttrModels}}{{{len(used)}}}")
    a.macros.write_text("\n".join(macros) + "\n")
    print(f"  attribution: {len(used)} complete model run(s) {used} -> {a.rows} ({len([r for r in rows if '&' in r])} rows), {a.macros} ({len(macros)} macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
