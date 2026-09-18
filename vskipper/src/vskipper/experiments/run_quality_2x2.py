#!/usr/bin/env python3
"""Boot each SERVED arm of the 2x2 once and run the lm-eval quality lane against it.

Why this exists, and why it is a committed file rather than box shell.

The 2x2 faithfulness table is the release gate: A = raw Llama in PyTorch, B = the released
FlexiDepth checkpoint under its own code, C = a stock serving stack, D = vSkipper serving
FlexiDepth. It passes when there is no collapse of D against C *and* `(D-C)` is comparable
to `(B-A)`. Arms A and B are native and are driven directly by `run_lmeval_quality.py`.

Arms C and D need a booted server, and until now that boot was hand-typed on a rented box.
Two consequences, both measured:

  * **Arm C was `ARMS["stock"]` -- this fork with the skipper off.** The gate condition it
    is supposed to test literally reads "no quality collapse vs UPSTREAM sglang" (D-587),
    so the comparison the table names has never actually been run. `Server` already knows
    how to serve the separate upstream tree and refuses if the fork answers instead; this
    driver reaches that code from the quality lane.
  * **Nothing gated the boot.** `verify_served_design` (G1b) and `campaign_preflight` (G1a)
    both existed with no caller on this path, so a quality cell could measure any arm at
    all and report it under the label that was typed. D-627 is the receipt: a whole night of
    quality numbers described the production all-RUN body.

So the boot, the attestation and both pre-gates are reused verbatim from
`run_paired_campaign` rather than re-expressed here. A second implementation of "boot the
upstream arm" is a second thing that can silently boot the fork.

**A refused workload does not abort its arm.** One refused rate once killed an entire arm's
worth of GPU time (D-692) and the refusal was itself the finding. Every requested workload
runs; refusals are recorded per row and the process exits non-zero at the end.

Usage:
  run_quality_2x2.py --spec quality_spec.json --out-dir OUT --port 32095 \\
      --arm upstream --arm integrated_alwaysskip \\
      --workload gsm8k --workload coqa --workload bbh_cot \\
      --lmeval-python /opt/vpipe/lmeval-fd/bin/python

The spec is the paired-campaign spec plus `quality_suites`, and without `datasets`:

  {"tree": ..., "upstream_tree": ..., "python": ..., "model_path": ...,
   "suites_dir": ..., "staging_root": ..., "host_config": ..., "source_revision": ...,
   "quality_suites": {"gsm8k": "gsm8k.d179", "coqa": "coqa.d179",
                      "bbh_cot": "bbh_cot.3shot.raw"}}

Suite names live in the SPEC, never in this file: the protocol (chat vs raw) is read from
whichever suite is named, and BBH's protocol is the difference between a +10.83 pp and a
-6.54 pp row (D-643).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paired_campaign import (  # noqa: E402
    Server,
    arm_is_upstream,
    campaign_preflight_gate,
    log,
    serving_pythonpath,
)

ROOT = Path(__file__).resolve().parents[4]
REQUIRED_SPEC_KEYS = ("tree", "upstream_tree", "python", "model_path",
                      "suites_dir", "staging_root", "host_config", "quality_suites")


def gpu_is_free(max_mib: int) -> tuple[bool, str]:
    """Is the card actually free -- read from the DEVICE, not from a kill's exit code.

    Killing a chain's parent does not reap the server: three times this session a server
    survived at `ppid=1` holding ~72 GB at 0 % utilisation, and the only check that caught
    it was nvidia-smi plus a process count. A boot onto an occupied card OOMs after paying
    for the model load.
    """
    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"nvidia-smi failed: {exc}"
    if used.returncode != 0:
        return False, f"nvidia-smi rc={used.returncode}: {used.stderr.strip()[:120]}"
    mib = [int(line) for line in used.stdout.split() if line.strip().isdigit()]
    if not mib:
        return False, f"nvidia-smi returned no memory rows: {used.stdout.strip()[:120]!r}"
    if max(mib) > max_mib:
        return False, f"{max(mib)} MiB still resident (limit {max_mib})"
    return True, f"{max(mib)} MiB resident"


def run_workload(spec: dict[str, Any], arm: str, workload: str, port: int,
                 out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """One (arm, workload) quality cell against the already-booted server."""
    suite = spec["quality_suites"][workload]
    cell = out_dir / arm / workload
    command = [
        sys.executable, str(Path(spec["tree"]) / "vskipper/src/vskipper/experiments/run_lmeval_quality.py"),
        "--workload", workload,
        "--arm", arm,
        "--suite-dir", spec["suites_dir"],
        "--suite-name", suite,
        "--output-dir", str(cell),
        "--base-url", f"http://127.0.0.1:{port}",
        "--tokenizer", spec["model_path"],
        "--num-concurrent", str(args.num_concurrent),
        "--lmeval-python", args.lmeval_python,
    ]
    if arm_is_upstream(spec, arm):
        # Inverts the attestation: vp_runtime must be ABSENT, and the skipping gate does
        # not apply to an arm that has no skipper.
        command.append("--upstream-baseline")
    log(f"    {arm} / {workload} (suite {suite}) starting")
    started = time.time()
    done = subprocess.run(command)
    minutes = (time.time() - started) / 60.0
    log(f"    {arm} / {workload}: {minutes:.0f} min, rc={done.returncode}")
    return {"arm": arm, "workload": workload, "suite": suite,
            "returncode": done.returncode, "minutes": round(minutes, 1),
            "output_dir": str(cell)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arm", action="append", default=[], required=True,
                    help="served arm; repeat. Order is the order they boot.")
    ap.add_argument("--workload", action="append", default=[], required=True,
                    help="workload; repeat. Every one runs against every arm.")
    ap.add_argument("--port", type=int, default=32095)
    ap.add_argument("--lmeval-python", default=sys.executable)
    ap.add_argument("--num-concurrent", type=int, default=16)
    ap.add_argument("--max-resident-mib", type=int, default=2048,
                    help="refuse to boot if more than this is already on the card")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight and G1a only; boot nothing")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    missing = [key for key in REQUIRED_SPEC_KEYS if key not in spec]
    if missing:
        print(f"PREFLIGHT FAILED — spec is missing {missing}", file=sys.stderr)
        return 2

    problems: list[str] = []
    for workload in args.workload:
        suite = spec["quality_suites"].get(workload)
        if not suite:
            problems.append(f"workload {workload!r} has no entry in spec.quality_suites")
            continue
        metadata = Path(spec["suites_dir"]) / f"{suite}.metadata.jsonl"
        if not metadata.is_file():
            problems.append(f"workload {workload!r}: no frozen suite at {metadata}")
    driver = Path(spec["tree"]) / "vskipper/src/vskipper/experiments/run_lmeval_quality.py"
    if not driver.is_file():
        problems.append(f"no quality driver at {driver}")
    if problems:
        print("PREFLIGHT FAILED — nothing has booted:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    log(f"preflight OK: arms {args.arm}, workloads {args.workload}")

    # G1a, per arm, before anything boots.
    for arm in args.arm:
        done = campaign_preflight_gate(spec, arm)
        if done.returncode != 0:
            print(f"G1a campaign preflight REFUSED for arm {arm!r} — nothing has booted:",
                  file=sys.stderr)
            sys.stderr.write(done.stdout[-2500:] + done.stderr[-1000:])
            return 2
        log(f"  G1a campaign preflight PASS ({arm}, serving {serving_pythonpath(spec, arm)})")
    if args.dry_run:
        for arm in args.arm:
            for workload in args.workload:
                log(f"  would run {arm} / {workload} "
                    f"(suite {spec['quality_suites'][workload]})")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    failures = 0

    for arm in args.arm:
        free, why = gpu_is_free(args.max_resident_mib)
        if not free:
            log(f"REFUSING to boot {arm}: the card is not free — {why}")
            index.append({"arm": arm, "workload": None, "returncode": 86,
                          "error": f"card not free: {why}"})
            failures += 1
            break
        log(f"=== arm {arm} (card free: {why}) ===")
        arm_log = args.out_dir / f"server.{arm}.log"
        try:
            with Server(spec, arm, args.port, arm_log) as server:  # noqa: F841
                for workload in args.workload:
                    row = run_workload(spec, arm, workload, args.port, args.out_dir, args)
                    index.append(row)
                    failures += row["returncode"] != 0
        except Exception as exc:  # boot / attestation failure is fatal for THIS arm only
            log(f"    arm {arm} FAILED to serve: {exc}")
            index.append({"arm": arm, "workload": None, "returncode": 87,
                          "error": str(exc)[:400]})
            failures += 1

    written = args.out_dir / "quality_index.json"
    written.write_text(json.dumps(
        {"spec": str(args.spec), "source_revision": spec.get("source_revision"),
         "arms": args.arm, "workloads": args.workload, "rows": index},
        indent=2, sort_keys=True) + "\n")
    log(f"--- {len(index)} rows, {failures} refused/failed — {written} ---")
    for row in index:
        log(f"    {row['arm']:>24} / {str(row['workload']):<9} rc={row['returncode']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
