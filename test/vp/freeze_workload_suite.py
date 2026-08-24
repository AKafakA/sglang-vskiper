#!/usr/bin/env python3
"""Validate and hash one immutable workload suite for strict accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_labeled_workload import (
    declared_frequency_penalty,
    validate_frozen_output_policy,
)
from labeled_workload import (
    FREQUENCY_PENALTY_FIELD,
    PROTOCOL_SCHEMA_VERSION,
    PROTOCOL_SUITE_ID,
    read_jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument(
        "--workloads", default="gsm8k,coqa,decode_mix,mixed"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite suite manifest: {args.output}")
    names = [value.strip() for value in args.workloads.split(",") if value.strip()]
    if not names or len(set(names)) != len(names):
        raise ValueError("--workloads must contain unique names")

    artifacts = {}
    identities = set()
    penalties = set()
    for name in names:
        requests_path = args.workload_dir / f"{name}.requests.jsonl"
        metadata_path = args.workload_dir / f"{name}.metadata.jsonl"
        summary_path = args.workload_dir / f"{name}.summary.json"
        requests = read_jsonl(requests_path)
        metadata = read_jsonl(metadata_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("schema_version") or 0) != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"{name} summary is not schema version {PROTOCOL_SCHEMA_VERSION}"
            )
        if summary.get("protocol_suite_id") != PROTOCOL_SUITE_ID:
            raise ValueError(f"{name} summary protocol suite mismatch")
        if len(requests) != int(summary.get("num_requests") or -1):
            raise ValueError(f"{name} request count does not match summary")
        model = str(summary.get("model") or "")
        revision = str(summary.get("model_revision") or "")
        context_length = int(summary.get("model_context_length") or 0)
        if not model or not revision or context_length <= 1:
            raise ValueError(f"{name} has incomplete model identity")
        frequency_penalty = declared_frequency_penalty(summary)
        validate_frozen_output_policy(
            requests,
            metadata,
            context_length,
            model,
            revision,
            frequency_penalty,
        )
        penalties.add(frequency_penalty)
        identities.add((model, revision, context_length))
        artifacts[name] = {
            "requests_sha256": _sha256(requests_path),
            "metadata_sha256": _sha256(metadata_path),
            "summary_sha256": _sha256(summary_path),
            "rows": len(requests),
        }

    if len(identities) != 1:
        raise ValueError("workloads do not share one model identity and context")
    if len(penalties) != 1:
        raise ValueError(
            "workloads do not share one generation penalty: "
            + ", ".join(str(value) for value in sorted(penalties))
        )
    model, revision, context_length = identities.pop()
    payload = {
        "schema_version": 3,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_suite_id": PROTOCOL_SUITE_ID,
        "model": model,
        "model_revision": revision,
        "model_context_length": context_length,
        FREQUENCY_PENALTY_FIELD: penalties.pop(),
        "workloads": artifacts,
    }
    payload["suite_sha256"] = _canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
