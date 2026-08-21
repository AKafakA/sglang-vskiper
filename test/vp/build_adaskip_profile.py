#!/usr/bin/env python3
"""Build a strict VP AdaSkip fixed-sublayer profile from calibration JSON."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.srt.vpipe.adaskip_profile import (
    load_adaskip_calibration,
    profile_from_calibration,
    selected_sublayer_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--skip-sublayers", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--enable-online-decode-extra-mlp",
        action="store_true",
        help=(
            "Enable the official per-request 20-token online MLP extension. "
            "Production serving then requires explicit request-slot and graph-row "
            "state capacities."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not args.output.parent.is_dir():
        raise SystemExit(f"output directory does not exist: {args.output.parent}")
    calibration = load_adaskip_calibration(args.calibration)
    profile, payload = profile_from_calibration(
        calibration,
        skip_sublayer_count=args.skip_sublayers,
        online_decode_extra_mlp=args.enable_online_decode_extra_mlp,
    )
    args.output.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    print(f"profile={args.output}")
    print(f"sha256={digest}")
    print(f"routed_layers={','.join(map(str, profile.routed_layer_ids))}")
    print(f"selected={','.join(selected_sublayer_ids(profile))}")
    print(f"online_decode_extra_mlp={profile.online_decode_extra_mlp}")


if __name__ == "__main__":
    main()
