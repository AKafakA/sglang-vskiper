#!/usr/bin/env python3
"""Construct a run_qps_evaluation deployment manifest from a LIVE server.

The sealed launcher's SERVER_PROFILES are v1-era arg bundles; current-tree
arms (e.g. V-pre prefill-only, fork-dense) are launched by their own scripts.
This builder produces the 14-field manifest the runner requires by querying
the RUNNING server and computing identity with the exact functions the
runner later re-verifies with (``qps_deployment.stable_server_identity`` +
canonical sha), so identity matches by construction and any config drift
between manifest time and run time still fails closed inside the runner.

``--expected-runtime-json`` is a subset-assert block (observed ⊇ expected)
pinning the arm's config markers — pass the markers that distinguish the
arm (phases, deferral, skipper) so a wrong-config server is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib import request as urlrequest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qps_deployment import (  # noqa: E402
    canonical_sha256,
    stable_server_identity,
    validate_expected_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--client-tokenizer-path", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--launch-command-file", type=Path, required=True,
                        help="the arm's launch script (recorded verbatim)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--expected-runtime-json", type=Path, required=True)
    parser.add_argument(
        "--config-epoch",
        default=None,
        help="Optional label naming the arm-config epoch (lane-coupling "
             "protocol: reruns after a lane landing bump the epoch so "
             "cross-epoch cells are never silently pooled)",
    )
    parser.add_argument(
        "--artifact", action="append", default=[],
        help="name=path records hashed into the manifest (e.g. router weights)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    expected_runtime = json.loads(
        args.expected_runtime_json.read_text(encoding="utf-8")
    )
    with urlrequest.urlopen(
        f"http://{args.host}:{args.port}/server_info", timeout=60
    ) as response:
        server_info = json.loads(response.read())

    # Fail here, not at run time, if the live config misses the markers.
    validate_expected_runtime(server_info, expected_runtime)
    identity = stable_server_identity(server_info)

    def file_record(p: Path) -> dict:
        data = p.read_bytes()
        return {
            "path": str(p.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    tokenizer_files = {
        p.name: file_record(p)
        for p in sorted(args.client_tokenizer_path.glob("*"))
        if p.name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
    }
    if not tokenizer_files:
        raise SystemExit(
            f"no tokenizer files under {args.client_tokenizer_path}"
        )
    artifacts = {}
    for spec in args.artifact:
        name, sep, raw = spec.partition("=")
        if not sep or not name or not raw:
            raise SystemExit(f"--artifact must be name=path, got {spec!r}")
        artifacts[name] = file_record(Path(raw))

    manifest = {
        "deployment_id": args.deployment_id,
        "config_epoch": args.config_epoch,
        "system_id": args.system_id,
        "model": args.model,
        "model_revision": args.model_revision,
        "client_tokenizer_path": str(args.client_tokenizer_path.resolve()),
        "client_tokenizer_files": tokenizer_files,
        "artifacts": artifacts,
        "source_revision": args.source_revision,
        "launch_command": args.launch_command_file.read_text(encoding="utf-8"),
        "endpoint": {"host": args.host, "port": args.port},
        "expected_runtime": expected_runtime,
        "observed_runtime": server_info,
        "server_identity": identity,
        "server_identity_sha256": canonical_sha256(identity),
        "manifest_builder": (
            "make_deployment_manifest.py (live-server construction; sealed "
            "launcher profiles are v1-era — DECLARED)"
        ),
    }
    args.out.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "deployment_id": args.deployment_id,
        "config_epoch": args.config_epoch,
        "server_identity_sha256": manifest["server_identity_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
