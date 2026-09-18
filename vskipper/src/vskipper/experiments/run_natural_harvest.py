#!/usr/bin/env python3
"""The natural-lane cell of one served arm: the serving benchmark as the quality table's second client.

The quality table has two clients. lm-eval drives the unloaded 2x2 (run_quality_2x2.py). Under load
the same held-out documents are served through the serving benchmark, generating to their own stop on
the `.qual` suites, and are scored afterwards with lm-eval's own filters
(score_natural_lane_lmeval.py). This driver produces that second kind of cell -- the `harvest-<arm>-
<dataset>-<label>/` roots the analysis reads (loaded_quality_table.py, client_equivalence.py,
bank_from_harvest.py) -- and is the in-tree form of the box shell that produced the paper's cells.

Everything that touches the card is reused from run_paired_campaign: the boot (Server: G1 served
design, Gate E default conformance, the live arm name), the deployment manifest, and the runner argv
(runner_command). A second implementation of "boot the upstream arm" is a second thing that can
silently boot the fork; the upstream arm is served from spec["upstream_tree"] by the same code path
the perf lane uses.

Usage:
  run_natural_harvest.py --spec paired_spec.json --out-dir OUT \\
      --arm integrated_it4 --dataset gsm8k --suite gsm8k.qual --rate 10.45 --label r10p45

The rate is the load point in requests per second and the label is its file-name form (10.45 ->
r10p45); the runner derives the cell's duration as rows / rate, so every request of the suite is
submitted. The suite must be a `.qual` suite (eval rows plus the padding the frozen-suite builder
adds); the scorer keeps the eval rows by `source_split`.

Exit status: 0 = the cell completed and passed the empty-generation gate; 1 = the runner refused
the cell or the gate failed (the root is kept, renamed VOID-<reason>); 2 = preflight refused, nothing
booted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paired_campaign import (  # noqa: E402
    Server,
    campaign_preflight_gate,
    log,
    model_revision,
    runner_command,
    served_model_name,
    server_info,
    serving_pythonpath,
)

REQUIRED_SPEC_KEYS = ("tree", "upstream_tree", "python", "model_path", "suites_dir",
                      "staging_root", "expect_dir", "host_config", "source_revision")


def cell_artifact(cells_dir: Path, suite: str) -> Path | None:
    """The one results .jsonl of the cell, or None when there is not exactly one."""
    matches = [p for p in cells_dir.rglob(f"{suite}_qps*_rep*.jsonl")
               if "arrival" not in p.name and "load" not in p.name
               and not p.name.startswith("INVALID")]
    return matches[0] if len(matches) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="the campaign root; the cell lands in <out-dir>/harvest-<arm>-<dataset>-<label>")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--suite", required=True, help="a frozen .qual suite name in spec.suites_dir")
    ap.add_argument("--rate", type=float, required=True, help="offered load, requests per second")
    ap.add_argument("--label", required=True, help="the rate's file-name form, e.g. r10p45")
    ap.add_argument("--port", type=int, default=32041)
    ap.add_argument("--dry-run", action="store_true", help="preflight and G1a only; boot nothing")
    args = ap.parse_args()

    spec: dict[str, Any] = json.loads(args.spec.read_text())
    missing = [key for key in REQUIRED_SPEC_KEYS if key not in spec]
    if missing:
        print(f"PREFLIGHT FAILED -- spec is missing {missing}", file=sys.stderr)
        return 2
    problems: list[str] = []
    suites = Path(spec["suites_dir"])
    for suffix in ("requests", "metadata"):
        if not (suites / f"{args.suite}.{suffix}.jsonl").is_file():
            problems.append(f"no frozen suite file {suites / f'{args.suite}.{suffix}.jsonl'}")
    if args.suite.startswith("INVALID"):
        problems.append(f"suite {args.suite!r} is marked INVALID")
    expected_label = "r" + f"{args.rate:g}".replace(".", "p")
    if args.label != expected_label:
        problems.append(f"label {args.label!r} does not spell rate {args.rate:g} ({expected_label!r})")
    if not (Path(spec["expect_dir"]) / f"expect_{args.arm}.json").is_file():
        problems.append(f"no expected-runtime file for arm {args.arm!r} in {spec['expect_dir']}")
    if problems:
        print("PREFLIGHT FAILED -- nothing has booted:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    done = campaign_preflight_gate(spec, args.arm)
    if done.returncode != 0:
        print(f"G1a campaign preflight REFUSED arm {args.arm!r} -- nothing has booted:", file=sys.stderr)
        sys.stderr.write(done.stdout[-2500:] + done.stderr[-1000:])
        return 2
    log(f"G1a campaign preflight PASS ({args.arm}, serving {serving_pythonpath(spec, args.arm)})")

    root = args.out_dir / f"harvest-{args.arm}-{args.dataset}-{args.label}"
    qps_config = root / "qps.json"
    manifest = root / "deployment_manifest.json"
    cells = root / "cell"
    if args.dry_run:
        log(f"would serve {args.arm} and run {args.suite} at {args.rate:g} req/s into {root}")
        log("runner argv: " + " ".join(runner_command(spec, args.arm, manifest, qps_config, cells, args.port)))
        return 0

    if root.exists():
        voided = root.with_name(root.name + f".VOID-attempt.{time.strftime('%H%M%S', time.gmtime())}")
        shutil.move(str(root), str(voided))
        log(f"previous root moved aside: {voided}")
    root.mkdir(parents=True)
    qps_config.write_text(json.dumps({"workloads": {args.suite: [args.rate]}}) + "\n")

    tree = Path(spec["tree"])
    with Server(spec, args.arm, args.port, root / "server.log") as server:
        (root / "server_info.before.json").write_text(
            json.dumps(server_info(args.port), indent=2, sort_keys=True) + "\n")
        launch = root / "launch_command.json"
        launch.write_text(json.dumps(server.command()) + "\n")
        subprocess.run(
            [spec["python"], str(tree / "vskipper/src/vskipper/experiments/make_deployment_manifest.py"),
             "--deployment-id", root.name,
             "--system-id", args.arm, "--model", served_model_name(spec),
             "--model-revision", model_revision(spec),
             "--client-tokenizer-path", spec["model_path"],
             "--source-revision", spec["source_revision"],
             "--launch-command-file", str(launch),
             "--expected-runtime-json", str(Path(spec["expect_dir"]) / f"expect_{args.arm}.json"),
             "--host", "127.0.0.1", "--port", str(args.port), "--out", str(manifest)],
            check=True, capture_output=True,
        )
        log(f"natural run: {args.suite} at {args.rate:g} req/s, every request of the suite")
        with (root / "cell.log").open("wb") as runner_log:
            runner = subprocess.run(
                runner_command(spec, args.arm, manifest, qps_config, cells, args.port),
                check=False, stdout=runner_log, stderr=subprocess.STDOUT)
        (root / "server_info.after.json").write_text(
            json.dumps(server_info(args.port), indent=2, sort_keys=True) + "\n")

    artifact = cell_artifact(cells, args.suite)
    if runner.returncode != 0 or artifact is None:
        log(f"cell REFUSED: runner rc={runner.returncode}, artifact={artifact}")
        shutil.move(str(root), str(root.with_name(root.name + ".VOID-refused")))
        return 1
    gate = subprocess.run(
        [spec["python"], str(tree / "vskipper/src/vskipper/experiments/gates/verify_zero_empty.py"), "--artifact", str(artifact)],
        capture_output=True, text=True)
    if gate.returncode != 0:
        log(f"empty generations in {artifact.name} -- the cell is re-measured, never accepted")
        sys.stderr.write(gate.stdout[-1500:] + gate.stderr[-500:])
        shutil.move(str(root), str(root.with_name(root.name + ".VOID-empties")))
        return 1
    log(f"HARVEST DONE {args.arm} {args.dataset} {args.label}: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
