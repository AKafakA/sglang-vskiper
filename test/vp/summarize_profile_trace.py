"""Summarize SGLang torch-profiler Chrome traces.

Usage:
  python test/vp/summarize_profile_trace.py TRACE_OR_SUMMARY_JSON [...]

If an input is a ``summary.json`` from ``profile_decode_modes.py``, the script
summarizes its ``decode_trace_files``. Output is JSON so it can be logged or
post-processed without opening Chrome trace manually.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


KEYWORDS = (
    "cudaLaunchKernel",
    "cuLaunchKernel",
    "index_select",
    "index_copy",
    "copy_",
    "nonzero",
    "where",
    "cat",
    "slice",
    "gather",
    "scatter",
    "flashinfer",
    "triton",
    "gemm",
    "matmul",
    "linear",
    "qwen",
    "forward",
)


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text())


def expand_inputs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.name == "summary.json":
            data = load_json(path)
            out.extend(Path(p) for p in data.get("decode_trace_files", []))
        else:
            out.append(path)
    return out


def summarize(path: Path, topn: int) -> dict[str, Any]:
    data = load_json(path)
    events = data.get("traceEvents", [])
    by_name: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "dur_us": 0.0}
    )
    by_cat: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "dur_us": 0.0})
    keyword_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "dur_us": 0.0}
    )
    x_events = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        cat = str(event.get("cat", ""))
        dur = float(event.get("dur", 0.0) or 0.0)
        x_events += 1
        by_name[(cat, name)]["count"] += 1
        by_name[(cat, name)]["dur_us"] += dur
        by_cat[cat]["count"] += 1
        by_cat[cat]["dur_us"] += dur
        lname = name.lower()
        for key in KEYWORDS:
            if key.lower() in lname:
                keyword_totals[key]["count"] += 1
                keyword_totals[key]["dur_us"] += dur
    top = sorted(
        (
            {
                "cat": cat,
                "name": name,
                "count": int(vals["count"]),
                "dur_us": vals["dur_us"],
                "avg_us": vals["dur_us"] / vals["count"] if vals["count"] else 0.0,
            }
            for (cat, name), vals in by_name.items()
        ),
        key=lambda item: item["dur_us"],
        reverse=True,
    )[:topn]
    return {
        "path": str(path),
        "x_events": x_events,
        "categories": {
            cat: {"count": int(vals["count"]), "dur_us": vals["dur_us"]}
            for cat, vals in sorted(by_cat.items())
        },
        "keyword_totals": {
            key: {"count": int(vals["count"]), "dur_us": vals["dur_us"]}
            for key, vals in sorted(keyword_totals.items())
        },
        "top_events": top,
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: summarize_profile_trace.py TRACE_OR_SUMMARY_JSON [...]")
    topn = int(__import__("os").environ.get("TOPN", "30"))
    paths = expand_inputs(sys.argv[1:])
    print(json.dumps([summarize(path, topn) for path in paths], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
