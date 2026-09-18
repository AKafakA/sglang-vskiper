#!/usr/bin/env python3
"""Length-calibration check for bank-pinned debug cells (the ONLY post-step
of the CSD3 debug harness — no quality scoring).

Verifies, per request of each cell result: the raw served output length
equals the requested (bank-pinned) length, and the requested length equals
the pinned suite's `bank_pinned_output_len`. Exit 2 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-jsonl", type=Path, required=True, nargs="+")
    parser.add_argument("--suite-metadata", type=Path, required=True,
                        help="the pinned suite's <workload>.metadata.jsonl")
    args = parser.parse_args()

    pinned: dict[str, int] = {}
    with args.suite_metadata.open() as f:
        for line in f:
            row = json.loads(line)
            if "bank_pinned_output_len" in row:
                pinned[row["request_id"]] = int(row["bank_pinned_output_len"])
            elif row.get("phase") == "prefill":
                # one_token_probe prefill suites: every request serves exactly
                # one output token; that IS the calibration contract.
                pinned[row["request_id"]] = 1
            else:
                raise SystemExit(
                    f"metadata row {row.get('request_id')!r} has neither "
                    "bank_pinned_output_len nor prefill phase — unknown "
                    "calibration contract"
                )

    failures = 0
    for cell in args.cell_jsonl:
        r = json.loads(cell.open().readline())
        ids = r["logical_request_ids"]
        requested = r["requested_output_lens"]
        raw = r["raw_output_lens"]
        if not (len(ids) == len(requested) == len(raw)):
            print(f"FAIL {cell.name}: array length mismatch "
                  f"ids={len(ids)} requested={len(requested)} raw={len(raw)}")
            failures += 1
            continue
        mismatch_bank = [
            (i, requested[k], pinned.get(i))
            for k, i in enumerate(ids)
            if pinned.get(i) != requested[k]
        ]
        mismatch_served = [
            (i, raw[k], requested[k])
            for k, i in enumerate(ids)
            if raw[k] != requested[k]
        ]
        if mismatch_bank or mismatch_served:
            failures += 1
            print(f"FAIL {cell.name}: bank-mismatch={len(mismatch_bank)} "
                  f"served-mismatch={len(mismatch_served)}")
            for tag, rows in (("bank", mismatch_bank), ("served", mismatch_served)):
                for i, got, want in rows[:5]:
                    print(f"  {tag}: {i} got={got} want={want}")
        else:
            print(f"PASS {cell.name}: n={len(ids)} all served lengths == "
                  f"pinned bank lengths (exact)")
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
