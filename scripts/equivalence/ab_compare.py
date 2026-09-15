#!/usr/bin/env python3
"""T3 of the equivalence ladder: did two trees serve the same 32 probe requests the same way?

    ab_compare.py <dev out-dir> <ref out-dir> [--skip SEGMENT ...]

Each argument is an output directory of test/vp/gates/box_vpcov_arm.sh, which holds
    <out>/probe/labels.jsonl   the probe's per-request table (text, tokens, finish reason, logprobs)
    <out>/server_info.json     the attestation snapshot taken AFTER the probe finished.
The probe also writes <out>/probe/server_info.json, but that one is taken before its first request
and says nothing about how the traffic was routed; only the post-probe snapshot is compared here.

Two comparisons, both fail-closed:
  1. the labels, request by request, after refusing any pair that is not the same requests in the
     same order (zip() would otherwise compare a prefix);
  2. the attestation, as flattened key paths. Boot-varying fields are skipped by EXACT path segment
     (a port is a port, `host` is the bind address), never by substring: the previous version skipped
     every key containing "path", "host", "dir" or "_id" and so silently dropped
     forced_all_run_fastpath, host_route_readback, policy_row_identity and every per-layer row.
     A key present on one side only is a difference.

Exit status: 0 = identical on both comparisons, 1 = a difference, 2 = refused (not comparable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

BOOT_VARYING = {
    # bind and transport
    "host", "port", "nccl_port", "grpc_port", "grpc_http_sidecar_port",
    "disaggregation_bootstrap_port", "encoder_bootstrap_port", "engine_info_bootstrap_port",
    # staging locations
    "model_path", "tokenizer_path", "download_dir", "file_storage_path", "kt_weight_path",
    "lora_paths", "spec_trace_dir", "export_metrics_to_file_dir",
    # process facts
    "pid", "uptime", "start_time", "timestamp", "version", "cuda_runtime_version",
    "weight_version", "per_boot_ladder_hash",
}
MECHANISM_TAGS = ("digest", "route", "skip", "cohort", "project", "run_rows", "engagement",
                  "coverage", "compact", "band", "regime", "fd_", "vp_runtime")


def flat(obj, prefix="", acc=None):
    acc = {} if acc is None else acc
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat(value, f"{prefix}.{key}" if prefix else key, acc)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            flat(value, f"{prefix}[{index}]", acc)
    else:
        acc[prefix] = obj
    return acc


def segments(path: str) -> list[str]:
    return [s for s in re.split(r"\.|\[\d+\]", path) if s]


def sha16(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def load_rows(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_labels(dev: str, ref: str) -> int:
    dev_rows = load_rows(os.path.join(dev, "probe", "labels.jsonl"))
    ref_rows = load_rows(os.path.join(ref, "probe", "labels.jsonl"))
    print(f"=== probe labels: dev {len(dev_rows)} rows, ref {len(ref_rows)} rows ===")
    if len(dev_rows) != len(ref_rows):
        print("REFUSED: not the same number of requests")
        return 2
    key = lambda row: (row.get("request_id"), row.get("suite_pos"))  # noqa: E731
    if [key(r) for r in dev_rows] != [key(r) for r in ref_rows]:
        print("REFUSED: request ids / positions differ -- not the same requests in the same order")
        return 2
    differing = [i for i, (a, b) in enumerate(zip(dev_rows, ref_rows)) if a != b]
    if not differing:
        print(f"  rows differing: 0 -> bitwise equal across all {len(dev_rows)} requests")
        return 0
    print(f"  rows differing: {len(differing)} -> {differing[:8]}")
    first = differing[0]
    for field in sorted(dev_rows[first]):
        if dev_rows[first].get(field) != ref_rows[first].get(field):
            print(f"    row {first} {field}: dev={str(dev_rows[first][field])[:70]!r} "
                  f"ref={str(ref_rows[first].get(field))[:70]!r}")
    return 1


def compare_attestation(dev: str, ref: str, skip: set[str]) -> int:
    dev_info = flat(json.load(open(os.path.join(dev, "server_info.json"))))
    ref_info = flat(json.load(open(os.path.join(ref, "server_info.json"))))
    keep = lambda path: not any(s in skip for s in segments(path))  # noqa: E731
    only = sorted(k for k in set(dev_info) ^ set(ref_info) if keep(k))
    shared = [k for k in dev_info if k in ref_info and keep(k)]
    mechanism = [k for k in shared if any(t in k.lower() for t in MECHANISM_TAGS)]
    mech_diff = [k for k in mechanism if dev_info[k] != ref_info[k]]
    other_diff = [k for k in shared if k not in mechanism and dev_info[k] != ref_info[k]]
    print("=== post-probe attestation ===")
    print(f"  comparable {len(shared)} | mechanism {len(mechanism)} | mechanism differing "
          f"{len(mech_diff)} | other differing {len(other_diff)} | one side only {len(only)}")
    for k in only[:12]:
        print(f"    one side only: {k}")
    for k in mech_diff[:12]:
        print(f"    {k}: dev={str(dev_info[k])[:60]!r} ref={str(ref_info[k])[:60]!r}")
    for k in other_diff[:12]:
        print(f"    (other) {k}: dev={str(dev_info[k])[:60]!r} ref={str(ref_info[k])[:60]!r}")
    return 1 if (only or mech_diff or other_diff) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dev")
    ap.add_argument("ref")
    ap.add_argument("--skip", action="append", default=[],
                    help="an extra attestation path SEGMENT to treat as boot-varying; repeatable")
    args = ap.parse_args()
    for side in (args.dev, args.ref):
        for rel in ("probe/labels.jsonl", "server_info.json"):
            if not os.path.isfile(os.path.join(side, rel)):
                print(f"REFUSED: {os.path.join(side, rel)} is missing")
                return 2
    print("=== byte level ===")
    for rel in ("probe/labels.jsonl", "probe/summary.json", "server_info.json"):
        a, b = sha16(os.path.join(args.dev, rel)), sha16(os.path.join(args.ref, rel))
        print(f"  {rel:<20} dev={a} ref={b}  {'IDENTICAL' if a == b else 'DIFFER'}")
    labels = compare_labels(args.dev, args.ref)
    if labels == 2:
        return 2
    attestation = compare_attestation(args.dev, args.ref, BOOT_VARYING | set(args.skip))
    verdict = max(labels, attestation)
    print("=== T3 " + ("IDENTICAL" if verdict == 0 else "DIFFERS") + " ===")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
