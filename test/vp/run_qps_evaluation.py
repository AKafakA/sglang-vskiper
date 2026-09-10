#!/usr/bin/env python3
"""Run labeled open-loop QPS evaluation against an existing endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from build_labeled_workload import validate_frozen_output_policy
from labeled_workload import read_jsonl
from qps_deployment import (
    canonical_sha256,
    stable_server_identity,
    validate_expected_runtime,
)


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_METRIC_ACCOUNTING_VERSION = 5
MAX_SMOKE_REQUESTS_PER_CELL = 32
ARRIVAL_SCHEDULE_VERSION = 1


class StrictAccountingError(RuntimeError):
    """A completed cell whose saved response artifact failed strict accounting."""


def _git_state() -> tuple[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return head, status
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "", ""


def _wait_ready(host: str, port: int, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become ready in {timeout_s}s")


def _fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _stop_process_group(process: subprocess.Popen[Any], timeout_s: int = 30) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def _runner_environment(
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    result = dict(os.environ if environment is None else environment)
    required = [str(ROOT / "python"), str(ROOT / "test/vp")]
    inherited = [
        path
        for path in result.get("PYTHONPATH", "").split(os.pathsep)
        if path and path not in required
    ]
    result["PYTHONPATH"] = os.pathsep.join([*required, *inherited])
    return result


def _count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def _qps_label(qps: float) -> str:
    return f"{qps:g}".replace(".", "p")


def _requested_count(min_prompts: int, qps: float, duration_s: float) -> int:
    return max(min_prompts, math.ceil(qps * duration_s))


def _smoke_request_count_violations(
    workloads: dict[str, list[Any]], min_prompts: int, duration_s: float
) -> list[tuple[str, float, int]]:
    violations = []
    for workload, qps_values in workloads.items():
        for value in qps_values:
            qps = float(value)
            requested_count = _requested_count(min_prompts, qps, duration_s)
            if requested_count > MAX_SMOKE_REQUESTS_PER_CELL:
                violations.append((workload, qps, requested_count))
    return violations


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workload_record(
    requests_path: Path,
    metadata_path: Path,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    record = {
        "requests_path": str(requests_path.resolve()),
        "requests_sha256": _sha256(requests_path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "rows": _count_rows(requests_path),
    }
    if summary_path is not None:
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing workload summary: {summary_path}")
        record.update(
            {
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": _sha256(summary_path),
            }
        )
    return record


def _materialize_arrival_schedule(
    requests_path: Path,
    output_path: Path,
    requested_count: int,
    qps: float,
    seed: int,
) -> dict[str, Any]:
    if requested_count <= 0:
        raise ValueError("arrival schedule requires at least one request")
    if not math.isfinite(qps) or qps <= 0:
        raise ValueError("arrival schedule QPS must be positive and finite")

    rows = []
    with requests_path.open(encoding="utf-8") as source:
        for line in source:
            if len(rows) >= requested_count:
                break
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != requested_count:
        raise ValueError(
            f"arrival schedule needs {requested_count} requests but read {len(rows)}"
        )

    generator = random.Random(seed)
    timestamp_ms = 0.0
    records = []
    request_ids = []
    for index, row in enumerate(rows):
        if index:
            timestamp_ms += generator.expovariate(qps) * 1000.0
        record = dict(row)
        record["timestamp"] = timestamp_ms
        records.append(record)
        request_ids.append(
            str(record.get("request_id"))
            if record.get("request_id") is not None
            else None
        )

    serialized = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    )
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        if existing != serialized:
            raise FileExistsError(
                f"refusing to replace a different arrival schedule: {output_path}"
            )
    else:
        output_path.write_text(serialized, encoding="utf-8")

    return {
        "version": ARRIVAL_SCHEDULE_VERSION,
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "size_bytes": output_path.stat().st_size,
        "source_requests_sha256": _sha256(requests_path),
        "request_count": requested_count,
        "request_ids_sha256": canonical_sha256(request_ids),
        "target_qps": qps,
        "seed": seed,
        "first_arrival_ms": records[0]["timestamp"],
        "last_arrival_ms": records[-1]["timestamp"],
        "injection_span_s": records[-1]["timestamp"] / 1000.0,
        "realized_qps": (
            (requested_count - 1) / (records[-1]["timestamp"] / 1000.0)
            if requested_count > 1 and records[-1]["timestamp"] > 0
            else None
        ),
    }


def _evaluation_intent(metadata: list[dict[str, Any]]) -> str:
    policies = {str(row.get("output_policy") or "") for row in metadata}
    if policies == {"fixed_token_capacity"}:
        return "fixed_token_capacity"
    if policies == {"production_max_equal_work"}:
        return "production_max_equal_work"
    if policies == {
        "production_max_equal_work",
        "prefill_probe_one_token",
    }:
        return "mixed_production_max_equal_work"
    if policies == {"task_reference_limit"} or policies == {
        "task_reference_limit",
        "prefill_probe_one_token",
    }:
        return "paper_quality"
    if policies == {"production_watchdog"} or policies == {
        "production_watchdog",
        "prefill_probe_one_token",
    }:
        return "natural_serving_watchdog"
    if policies == {"prefill_probe_one_token"}:
        return "prefill_probe"
    if "remaining_model_context" in policies:
        return "natural_serving"
    if "fixed_token_capacity" in policies:
        return "mixed_fixed_token_capacity"
    return "mixed_or_unknown"


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_contract_runner_artifact(
    contract: dict[str, Any], revision: str, archive: Path | None
) -> dict[str, Any]:
    expected_revision = contract.get("evaluation_runner_revision")
    if revision != expected_revision:
        raise ValueError(
            "evaluation runner revision does not match evaluation manifest: "
            f"expected {expected_revision!r}, observed {revision!r}"
        )
    if archive is None or not archive.is_file():
        raise ValueError(
            "performance evidence requires the committed runner source archive"
        )
    record = _artifact_record(archive)
    expected_sha256 = contract.get("evaluation_runner_archive_sha256")
    if record["sha256"] != expected_sha256:
        raise ValueError(
            "evaluation runner archive does not match evaluation manifest: "
            f"expected {expected_sha256!r}, observed {record['sha256']!r}"
        )
    return record


def _missing_required_fields(
    payload: dict[str, Any], required_fields: tuple[str, ...]
) -> list[str]:
    return [
        key
        for key in required_fields
        if key not in payload or payload[key] is None or payload[key] == ""
    ]


def _validate_live_deployment(
    host: str, port: int, deployment: dict[str, Any]
) -> dict[str, Any]:
    server_info = _fetch_json(f"http://{host}:{port}/server_info")
    validate_expected_runtime(server_info, deployment["expected_runtime"])
    identity = stable_server_identity(server_info)
    identity_sha256 = canonical_sha256(identity)
    if identity_sha256 != deployment["server_identity_sha256"]:
        raise RuntimeError(
            "live /server_info identity does not match deployment manifest: "
            f"expected {deployment['server_identity_sha256']}, "
            f"observed {identity_sha256}"
        )
    if identity != deployment["server_identity"]:
        raise RuntimeError("live /server_info identity differs from deployment manifest")
    return server_info


def _validate_deployment_artifacts(deployment: dict[str, Any]) -> None:
    groups = (
        deployment.get("artifacts") or {},
        deployment.get("client_tokenizer_files") or {},
    )
    for records in groups:
        if not isinstance(records, dict):
            raise ValueError("deployment artifact records must be objects")
        for name, record in records.items():
            if not isinstance(record, dict) or not record.get("path"):
                raise ValueError(f"invalid deployment artifact record {name!r}")
            path = Path(record["path"])
            if not path.is_file():
                raise FileNotFoundError(f"deployment artifact {name} is missing: {path}")
            if record.get("sha256") != _sha256(path):
                raise RuntimeError(f"deployment artifact {name} hash mismatch: {path}")
            if int(record.get("size_bytes") or -1) != path.stat().st_size:
                raise RuntimeError(f"deployment artifact {name} size mismatch: {path}")


def _validate_contract_system_artifact_shape(
    contract: dict[str, Any], experiment: str
) -> dict[str, Any]:
    systems = contract.get("system_artifacts")
    if not isinstance(systems, dict):
        raise ValueError("evaluation manifest has no system_artifacts object")
    expected = systems.get(experiment)
    if not isinstance(expected, dict):
        raise ValueError(
            f"evaluation manifest has no system artifact for {experiment!r}"
        )
    required_fields = (
        "source_revision",
        "source_archive_sha256",
        "expected_runtime_sha256",
        "router_sha256",
        "server_profile",
    )
    missing = [field for field in required_fields if field not in expected]
    if missing:
        raise ValueError(
            f"system artifact for {experiment!r} is missing fields: {missing}"
        )
    deployed_fields = (
        "deployed_server_revision",
        "deployed_server_archive_sha256",
    )
    present_deployed_fields = [
        field for field in deployed_fields if field in expected
    ]
    if present_deployed_fields and len(present_deployed_fields) != len(
        deployed_fields
    ):
        missing_deployed_fields = [
            field for field in deployed_fields if field not in expected
        ]
        raise ValueError(
            f"system artifact for {experiment!r} has incomplete deployed "
            f"server provenance; missing fields: {missing_deployed_fields}"
        )
    expected_artifacts = expected.get("deployment_artifacts")
    if expected_artifacts is not None and (
        not isinstance(expected_artifacts, dict) or not expected_artifacts
    ):
        raise ValueError(
            f"system artifact for {experiment!r} has an invalid "
            "deployment_artifacts object"
        )
    return expected


def _validate_contract_system_artifact(
    contract: dict[str, Any], experiment: str, deployment: dict[str, Any]
) -> None:
    expected = _validate_contract_system_artifact_shape(contract, experiment)

    artifacts = deployment.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        raise ValueError("deployment artifacts must be an object")

    def artifact_sha256(name: str) -> str | None:
        record = artifacts.get(name)
        if record is None:
            return None
        if not isinstance(record, dict):
            raise ValueError(f"deployment artifact {name!r} must be an object")
        value = record.get("sha256")
        return str(value) if value is not None else None

    deployed_revision_field = (
        "deployed_server_revision"
        if "deployed_server_revision" in expected
        else "source_revision"
    )
    deployed_archive_field = (
        "deployed_server_archive_sha256"
        if "deployed_server_archive_sha256" in expected
        else "source_archive_sha256"
    )
    observed = {
        deployed_revision_field: deployment.get("source_revision"),
        deployed_archive_field: artifact_sha256("source_archive"),
        "expected_runtime_sha256": artifact_sha256("expected_runtime"),
        "router_sha256": artifact_sha256("flexidepth_router"),
        "server_profile": deployment.get("server_profile"),
    }
    required_fields = tuple(observed)
    mismatches = {
        field: {"expected": expected.get(field), "observed": observed.get(field)}
        for field in required_fields
        if expected.get(field) != observed.get(field)
    }
    expected_artifacts = expected.get("deployment_artifacts")
    if expected_artifacts is not None:
        observed_artifacts = {
            str(name): artifact_sha256(str(name))
            for name in expected_artifacts
        }
        artifact_mismatches = {
            name: {
                "expected": expected_artifacts[name],
                "observed": observed_artifacts[name],
            }
            for name in expected_artifacts
            if expected_artifacts[name] != observed_artifacts[name]
        }
        if artifact_mismatches:
            mismatches["deployment_artifacts"] = artifact_mismatches
    if mismatches:
        raise ValueError(
            f"deployment does not match the system artifact for "
            f"{experiment!r}: {json.dumps(mismatches, sort_keys=True)}"
        )
    if deployment.get("source_status", ""):
        raise ValueError("performance deployment server source is dirty")


def _validate_contract_phase_qps(
    contract: dict[str, Any],
    phase: str,
    workloads: dict[str, list[Any]],
) -> None:
    phase_qps = contract.get("phase_qps")
    if not isinstance(phase_qps, dict):
        raise ValueError("evaluation manifest has no phase_qps object")
    expected = phase_qps.get(phase)
    if not isinstance(expected, dict):
        raise ValueError(
            f"evaluation manifest has no absolute QPS grid for phase {phase!r}"
        )
    normalized_observed = {
        workload: [float(value) for value in values]
        for workload, values in workloads.items()
    }
    normalized_expected = {
        workload: [float(value) for value in values]
        for workload, values in expected.items()
    }
    if normalized_observed != normalized_expected:
        raise ValueError(
            f"QPS grid does not match the {phase} evaluation manifest: "
            f"expected {json.dumps(normalized_expected, sort_keys=True)}, "
            f"observed {json.dumps(normalized_observed, sort_keys=True)}"
        )


def _required_contract_repetitions(contract: dict[str, Any], lane: str) -> int:
    repetitions = contract.get("repetitions")
    if not isinstance(repetitions, dict):
        raise ValueError(
            "evaluation manifest repetitions must be a lane-keyed object"
        )
    value = repetitions.get(lane)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"evaluation manifest has no positive repetition count for lane {lane!r}"
        )
    return value


def _validate_workload_inputs(
    workloads: dict[str, list[Any]],
    workload_dir: Path,
    expected_model: str,
    expected_revision: str,
    expected_context_length: int | None,
    evaluation_contract: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    workload_records = {}
    for workload in workloads:
        requests_path = workload_dir / f"{workload}.requests.jsonl"
        metadata_path = workload_dir / f"{workload}.metadata.jsonl"
        summary_path = workload_dir / f"{workload}.summary.json"
        if (
            not requests_path.is_file()
            or not metadata_path.is_file()
            or not summary_path.is_file()
        ):
            raise FileNotFoundError(
                f"missing workload files for {workload} under {workload_dir}"
            )
        requests = read_jsonl(requests_path)
        metadata = read_jsonl(metadata_path)
        context_lengths = {
            int(row.get("context_length") or 0) for row in metadata
        }
        if len(context_lengths) != 1 or 0 in context_lengths:
            raise ValueError(
                f"{workload} metadata does not declare one context length"
            )
        workload_context_length = next(iter(context_lengths))
        workload_models = {str(row.get("model") or "") for row in metadata}
        workload_revisions = {
            str(row.get("model_revision") or "") for row in metadata
        }
        if workload_models != {expected_model}:
            raise ValueError(
                f"{workload} model metadata {workload_models} does not match "
                f"expected model {expected_model}"
            )
        if workload_revisions != {expected_revision}:
            raise ValueError(
                f"{workload} model revisions {workload_revisions} do not match "
                f"expected revision {expected_revision}"
            )
        if (
            expected_context_length is not None
            and workload_context_length != expected_context_length
        ):
            raise ValueError(
                f"{workload} context length {workload_context_length} does not "
                f"match expected context length {expected_context_length}"
            )
        validate_frozen_output_policy(
            requests, metadata, workload_context_length
        )
        record = _workload_record(requests_path, metadata_path, summary_path)
        record["context_length"] = workload_context_length
        record["evaluation_intent"] = _evaluation_intent(metadata)
        record["output_policies"] = sorted(
            {str(row.get("output_policy") or "") for row in metadata}
        )
        record["fixed_output_tokens"] = sorted(
            {
                int(row["fixed_output_tokens"])
                for row in metadata
                if row.get("fixed_output_tokens") is not None
            }
        )
        if evaluation_contract is not None:
            expected_artifacts = (
                evaluation_contract.get("workload_artifacts") or {}
            ).get(workload)
            if not isinstance(expected_artifacts, dict):
                raise ValueError(
                    f"evaluation manifest has no frozen artifacts for {workload}"
                )
            for field in (
                "requests_sha256",
                "metadata_sha256",
                "summary_sha256",
                "rows",
            ):
                if expected_artifacts.get(field) != record.get(field):
                    raise ValueError(
                        f"{workload} {field} does not match the evaluation manifest"
                    )
        workload_records[workload] = record
    return workload_records


def _resume_artifacts_match(
    completed: dict[str, Any], required_artifacts: dict[str, Path]
) -> bool:
    records = completed.get("artifacts")
    if not isinstance(records, dict) or set(records) != set(required_artifacts):
        return False
    for name, path in required_artifacts.items():
        if not path.is_file():
            return False
        record = records.get(name)
        if not isinstance(record, dict):
            return False
        if record.get("sha256") != _sha256(path):
            return False
        if int(record.get("size_bytes") or -1) != path.stat().st_size:
            return False
    try:
        audit = json.loads(required_artifacts["audit"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return audit.get("status") == "passed"


GATES = ROOT / "test/vp/gates"

# Lanes whose artifacts CAN carry an empty generation. Equal-work and probe lanes are
# excluded because `ignore_eos` fills an immediate EOS to budget there, which makes a
# zero-empty result vacuous -- verify_zero_empty refuses such an artifact by design.
# `mixed_or_unknown` is deliberately INCLUDED: an intent we do not recognise must be
# checked, never silently waved through.
_NATURAL_INTENTS = frozenset(
    {
        "paper_quality",
        "natural_serving",
        "natural_serving_watchdog",
        "mixed_or_unknown",
    }
)


def _served_design(server_info: Any) -> Any | None:
    try:
        return server_info["internal_states"][0]["vp_runtime"]["served_design"]
    except (KeyError, IndexError, TypeError):
        return None


def _arm_routes_decode(server_info: Any, *, upstream_baseline: bool = False) -> bool:
    """Does the LIVE server report a skipper that routes the decode phase?

    Read from `vp_runtime.served_design`, the D-609 attestation block -- what the server
    says it resolved, never a flag we passed. Verified against captured server_info:
    a baseline arm reports {arm: stock, skipper: None, active_phases: []} and a routed
    arm {arm: integrated_*, skipper: flexidepth, active_phases: [decode, prefill]}.

    On OUR tree a missing block is a REFUSAL, not a False. It means the served tree
    predates the D-609 attestation, so we cannot tell what ran -- and "measured the wrong
    system while every gate passed" is the exact failure this harness exists to prevent.

    `upstream_baseline` INVERTS that, and this is the point rather than a loophole. The
    paper's baseline is genuinely upstream SGLang (owner order D-587: "directly-freshly
    cloned sglang without our changes"), a tree with no vpipe/ package at all -- so it can
    never carry this block, and the refusal above made the real baseline UNMEASURABLE by
    this harness. That is part of why every campaign to date silently used ARMS["stock"],
    our own fork with the skipper switched off, while calling it upstream.

    So with the flag we assert the block is ABSENT and refuse if it is PRESENT: a
    vp_runtime block on a cell declared upstream means the fork was launched by mistake,
    which is the confusion this whole flag exists to make impossible. Absence is not
    tolerated here, it is the positive attestation -- upstream's /server_info exposes no
    source identity of its own (version "0.0.0.dev0"), so this is the only identity signal
    available at the endpoint, and the tree's content hash carries the rest.
    """
    served = _served_design(server_info)
    if upstream_baseline:
        if served is not None:
            raise RuntimeError(
                "cell declared --upstream-baseline but the live server DOES carry "
                f"vp_runtime.served_design ({served.get('arm')!r}): this is our fork, not "
                "a freshly-cloned upstream tree. Refusing to label it upstream."
            )
        return False
    if served is None:
        raise RuntimeError(
            "live /server_info carries no vp_runtime.served_design: the served tree "
            "predates the D-609 design attestation, so the resolved arm cannot be "
            "verified. Deploy the current tree, or pass --upstream-baseline if this "
            "cell is deliberately serving genuine upstream SGLang."
        )
    return "decode" in (served.get("active_phases") or [])


def _cell_gates(
    *,
    output_file: Path,
    arrival_file: Path,
    server_info_before: Path,
    server_info_after: Path,
    evaluation_intent: str,
    routes_decode: bool,
    env: dict[str, str],
) -> list[str]:
    """Run the per-cell gates. Returns the names of the ones that REFUSED.

    Every gate here already exits 0/1 via `raise SystemExit(main())`, so the contract is
    just: call it, keep the exit code, and let the caller fail the cell. Never pipe a
    gate into `tail`/`sed` -- that discards the status, which is how a refusing gate came
    to print FATAL and be followed by a DONE marker.
    """
    checks: list[tuple[str, list[str]]] = [
        (
            "arrival_fidelity",
            [
                sys.executable,
                str(GATES / "verify_arrival_fidelity.py"),
                "--arrival",
                str(arrival_file),
                "--artifact",
                str(output_file),
            ],
        )
    ]
    if evaluation_intent in _NATURAL_INTENTS:
        checks.append(
            (
                "zero_empty",
                [
                    sys.executable,
                    str(GATES / "verify_zero_empty.py"),
                    "--artifact",
                    str(output_file),
                ],
            )
        )
    if routes_decode:
        checks.append(
            (
                "skipping_executed",
                [
                    sys.executable,
                    str(GATES / "verify_skipping_executed.py"),
                    "--before",
                    str(server_info_before),
                    "--after",
                    str(server_info_after),
                ],
            )
        )
    failed: list[str] = []
    for name, command in checks:
        if subprocess.run(command, cwd=ROOT, env=env, check=False).returncode != 0:
            failed.append(name)
    return failed


def _invalidate_cell_artifacts(paths: list[Path]) -> list[str]:
    """Rename a refused cell's artifacts out of the way.

    The benchmark child writes the .jsonl itself, so a gate cannot prevent the write --
    only make sure nothing downstream mistakes it for a passing cell. Same treatment as
    cell_gates.py. The runner refuses to overwrite an existing artifact, so a rerun
    needs these gone.
    """
    renamed: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        target = path.with_name(f"INVALID.{path.name}")
        if target.exists():
            target.unlink()
        path.rename(target)
        renamed.append(str(target))
    return renamed


def _run_cell(
    args: argparse.Namespace,
    workload: str,
    qps: float,
    rep: int,
    requests_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    experiment = args.experiment
    workload_record = getattr(args, "workload_records", {}).get(workload)
    if workload_record is None:
        workload_record = _workload_record(requests_path, metadata_path)
    requested_count = _requested_count(args.min_prompts, qps, args.duration_s)
    available = _count_rows(requests_path)
    if requested_count > available:
        raise ValueError(
            f"{workload} QPS {qps:g} needs {requested_count} requests for "
            f"{args.duration_s:g}s but only {available} are available"
        )
    label = f"{workload}_qps{_qps_label(qps)}_rep{rep}"
    output_file = args.output_dir / f"{label}.jsonl"
    score_file = args.output_dir / f"{label}.score.json"
    audit_file = args.output_dir / f"{label}.audit.json"
    load_file = args.output_dir / f"{label}.load.jsonl"
    stdout_file = args.output_dir / f"{label}.stdout.log"
    command_file = args.output_dir / f"{label}.command.json"
    arrival_file = args.output_dir / f"{label}.arrival.requests.jsonl"
    # PER-CELL server_info. The run-level pair spans every workload x qps x rep, so it can
    # certify a suite but not the cell that just ran -- and the skipping gate is a per-cell
    # question.
    server_info_before_file = args.output_dir / f"{label}.server_info.before.json"
    server_info_after_file = args.output_dir / f"{label}.server_info.after.json"
    seed = args.seed + rep - 1
    arrival_schedule = _materialize_arrival_schedule(
        requests_path,
        arrival_file,
        requested_count,
        qps,
        seed,
    )
    if getattr(args, "resume_completed", False) and command_file.is_file():
        completed = json.loads(command_file.read_text(encoding="utf-8"))
        required_artifacts = {
            "arrival_schedule": arrival_file,
            "benchmark": output_file,
            "score": score_file,
            "audit": audit_file,
            "load": load_file,
            "load_summary": load_file.with_suffix(".summary.json"),
            "stdout": stdout_file,
        }
        if (
            completed.get("status") == "completed"
            and completed.get("experiment") == experiment
            and completed.get("deployment_manifest_sha256")
            == args.deployment_sha256
            and completed.get("requests_sha256")
            == workload_record["requests_sha256"]
            and completed.get("metadata_sha256")
            == workload_record["metadata_sha256"]
            and completed.get("workload") == workload
            and completed.get("evidence_class") == args.evidence_class
            and completed.get("metric_accounting_version_required")
            == REQUIRED_METRIC_ACCOUNTING_VERSION
            and completed.get(
                "evaluation_intent",
                workload_record.get("evaluation_intent", "unknown"),
            )
            == workload_record.get("evaluation_intent", "unknown")
            and float(completed.get("qps", -1)) == qps
            and int(completed.get("num_prompts", -1)) == requested_count
            and completed.get("arrival_schedule") == arrival_schedule
            and _resume_artifacts_match(completed, required_artifacts)
        ):
            print(
                f"RESUME-SKIP workload={workload} qps={qps:g} rep={rep} "
                f"output={output_file}",
                flush=True,
            )
            return completed
    existing_artifacts = [
        path
        for path in (
            command_file,
            output_file,
            score_file,
            audit_file,
            load_file,
            load_file.with_suffix(".summary.json"),
            stdout_file,
            arrival_file,
        )
        if path.exists()
    ]
    if existing_artifacts == [arrival_file]:
        existing_artifacts = []
    if existing_artifacts:
        raise FileExistsError(
            "refusing to append to or overwrite existing cell artifacts: "
            + ", ".join(str(path) for path in existing_artifacts)
        )
    benchmark_command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        args.backend,
        "--forward-request-id-as-rid",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--tokenizer",
        str(getattr(args, "client_tokenizer_path", args.model)),
        "--dataset-name",
        "autobench",
        "--dataset-path",
        str(arrival_file),
        "--num-prompts",
        str(requested_count),
        "--request-rate",
        str(qps),
        "--use-trace-timestamps",
        "--seed",
        str(seed),
        "--warmup-requests",
        str(args.warmup_requests),
        "--ready-check-timeout-sec",
        "300",
        "--disable-tqdm",
        "--disable-ignore-eos",
        "--flush-cache",
        "--output-details",
        "--output-file",
        str(output_file),
        "--tag",
        label,
    ]
    if getattr(args, "torch_profile", False):
        profile_root = Path(args.torch_profile_output_dir) / label
        benchmark_command.extend(
            [
                "--profile",
                "--profile-activities",
                "GPU",
                "--profile-start-step",
                str(args.torch_profile_start_step),
                "--profile-steps",
                str(args.torch_profile_steps),
                "--profile-output-dir",
                str(profile_root),
                "--profile-prefix",
                label,
            ]
        )
    assert "--max-concurrency" not in benchmark_command
    sampler_command = [
        sys.executable,
        str(ROOT / "test/vp/sample_sglang_load.py"),
        "--base-url",
        f"http://{args.host}:{args.port}",
        "--output",
        str(load_file),
        "--interval-ms",
        str(args.load_sample_interval_ms),
    ]
    child_environment = _runner_environment()
    command_record = {
        "label": label,
        "experiment": experiment,
        "evidence_class": args.evidence_class,
        "metric_accounting_version_required": REQUIRED_METRIC_ACCOUNTING_VERSION,
        "deployment_manifest_sha256": args.deployment_sha256,
        "requests_sha256": workload_record["requests_sha256"],
        "metadata_sha256": workload_record["metadata_sha256"],
        "evaluation_intent": workload_record.get("evaluation_intent", "unknown"),
        "output_policies": workload_record.get("output_policies", []),
        "fixed_output_tokens": workload_record.get("fixed_output_tokens", []),
        "workload": workload,
        "qps": qps,
        "duration_target_s": args.duration_s,
        "expected_injection_s": arrival_schedule["injection_span_s"],
        "arrival_schedule": arrival_schedule,
        "num_prompts": requested_count,
        "rep": rep,
        "seed": seed,
        "benchmark_command": benchmark_command,
        "sampler_command": sampler_command,
        "runner_pythonpath": child_environment["PYTHONPATH"],
        "max_concurrency": None,
        "torch_profile": (
            {
                "activities": ["GPU"],
                "start_step": args.torch_profile_start_step,
                "steps": args.torch_profile_steps,
                "output_dir": str(profile_root),
                "prefix": label,
            }
            if getattr(args, "torch_profile", False)
            else None
        ),
    }
    _write_json(command_file, command_record)
    if args.dry_run:
        print(json.dumps(command_record, sort_keys=True), flush=True)
        return command_record

    server_info_before = _validate_live_deployment(
        args.host, args.port, args.deployment
    )
    _write_json(server_info_before_file, server_info_before)
    routes_decode = _arm_routes_decode(
        server_info_before, upstream_baseline=args.upstream_baseline
    )

    sampler = subprocess.Popen(
        sampler_command,
        cwd=ROOT,
        env=child_environment,
        start_new_session=True,
    )
    started = time.time()
    try:
        with stdout_file.open("w", encoding="utf-8") as output:
            subprocess.run(
                benchmark_command,
                cwd=ROOT,
                env=child_environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        _stop_process_group(sampler, timeout_s=10)
    ended = time.time()
    _write_json(
        server_info_after_file,
        _fetch_json(f"http://{args.host}:{args.port}/server_info"),
    )
    audit_command = [
        sys.executable,
        str(ROOT / "test/vp/validate_qps_artifact.py"),
        "--bench-jsonl",
        str(output_file),
        "--expected-requests",
        str(requested_count),
        "--metadata",
        str(metadata_path),
        "--output",
        str(audit_file),
        "--require-request-timing",
    ]
    audit_result = subprocess.run(
        audit_command, cwd=ROOT, env=child_environment, check=False
    )
    score_command = [
        sys.executable,
        str(ROOT / "test/vp/score_labeled_workload.py"),
        "--bench-jsonl",
        str(output_file),
        "--metadata",
        str(metadata_path),
        "--output",
        str(score_file),
    ]
    if getattr(args, "allow_code_execution", False):
        score_command.append("--allow-code-execution")
    subprocess.run(score_command, cwd=ROOT, env=child_environment, check=True)
    accounting_passed = audit_result.returncode == 0
    failed_gates = _cell_gates(
        output_file=output_file,
        arrival_file=arrival_file,
        server_info_before=server_info_before_file,
        server_info_after=server_info_after_file,
        evaluation_intent=workload_record.get("evaluation_intent", "mixed_or_unknown"),
        routes_decode=routes_decode,
        env=child_environment,
    )
    cell_passed = accounting_passed and not failed_gates
    command_record.update(
        {
            "started_unix_s": started,
            "ended_unix_s": ended,
            "elapsed_wall_s": ended - started,
            "output_file": str(output_file),
            "audit_file": str(audit_file),
            "score_file": str(score_file),
            "load_file": str(load_file),
            "stdout_file": str(stdout_file),
            "score_command": score_command,
            "routes_decode": routes_decode,
            "failed_gates": failed_gates,
            "status": (
                "completed"
                if cell_passed
                else ("accounting_failed" if not accounting_passed else "gate_failed")
            ),
            "artifacts": {
                "arrival_schedule": _artifact_record(arrival_file),
                "benchmark": _artifact_record(output_file),
                "score": _artifact_record(score_file),
                "audit": _artifact_record(audit_file),
                "load": _artifact_record(load_file),
                "load_summary": _artifact_record(
                    load_file.with_suffix(".summary.json")
                ),
                "stdout": _artifact_record(stdout_file),
            },
        }
    )
    if not cell_passed:
        reason = "strict_accounting" if not accounting_passed else "gates"
        command_record["invalidated_artifacts"] = _invalidate_cell_artifacts(
            [output_file, score_file, audit_file]
        )
        _write_json(command_file, command_record)
        print(
            f"REJECTED workload={workload} qps={qps:g} rep={rep} "
            f"reason={reason} gates={','.join(failed_gates) or '-'} "
            f"output={output_file}",
            flush=True,
        )
        raise StrictAccountingError(
            f"cell {label} refused: reason={reason} "
            f"failed_gates={failed_gates or '[]'}"
        )
    _write_json(command_file, command_record)
    print(
        f"COMPLETED workload={workload} qps={qps:g} rep={rep} "
        f"elapsed={ended - started:.1f}s output={output_file}",
        flush=True,
    )
    return command_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--qps-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-class",
        choices=("smoke", "development", "performance"),
        required=True,
    )
    parser.add_argument(
        "--evaluation-contract",
        type=Path,
        default=None,
        help="Optional hash-pinned evaluation manifest; never an authorization gate.",
    )
    parser.add_argument(
        "--performance-phase",
        choices=("calibration", "comparison", "headline"),
        default=None,
    )
    parser.add_argument(
        "--contract-lane",
        default=None,
        help=(
            "Optional contract QPS/repetition lane, separate from the "
            "calibration/comparison/headline evidence phase."
        ),
    )
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--backend",
        choices=("sglang-native",),
        default="sglang-native",
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--campaign-total-reps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--min-prompts", type=int, default=256)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--load-sample-interval-ms", type=int, default=1000)
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Capture a GPU-only fixed-step Torch trace for development evidence.",
    )
    parser.add_argument("--torch-profile-start-step", type=int, default=0)
    parser.add_argument("--torch-profile-steps", type=int, default=0)
    parser.add_argument("--torch-profile-output-dir", type=Path, default=None)
    parser.add_argument("--ready-timeout-s", type=int, default=600)
    parser.add_argument("--max-walltime-s", type=int, default=21600)
    parser.add_argument(
        "--runner-source-revision",
        "--source-revision",
        dest="runner_source_revision",
        default="",
        help="Runner commit hash; required when the source mirror has no Git metadata.",
    )
    parser.add_argument(
        "--runner-source-archive",
        type=Path,
        default=None,
        help="Hash-pinned committed runner archive; required for performance evidence.",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--continue-after-accounting-rejection", action="store_true")
    parser.add_argument(
        "--upstream-baseline",
        action="store_true",
        help="This cell serves GENUINE upstream SGLang (a separate tree with no vpipe/ "
             "package, per owner order D-587), not ARMS['stock'] which is our fork with "
             "the skipper off. Inverts the attestation: vp_runtime.served_design must be "
             "ABSENT, and its PRESENCE refuses the cell as a mislaunched fork. Without "
             "this flag an upstream server cannot be measured at all, which is why the "
             "real baseline never was.",
    )
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Execute HumanEval completions during offline scoring on this host.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--contract-preflight-only",
        action="store_true",
        help="Validate the static evaluation contract and exit before endpoint use.",
    )
    args = parser.parse_args()

    if (
        args.continue_after_accounting_rejection
        and args.evidence_class != "development"
    ):
        parser.error(
            "--continue-after-accounting-rejection is development-only"
        )
    if args.torch_profile:
        if args.evidence_class != "development":
            parser.error("Torch profiling is development-only")
        if args.torch_profile_start_step < 0:
            parser.error("--torch-profile-start-step must be non-negative")
        if args.torch_profile_steps <= 0:
            parser.error("--torch-profile-steps must be positive")
        if args.torch_profile_output_dir is None:
            parser.error("--torch-profile-output-dir is required")
    elif (
        args.torch_profile_start_step != 0
        or args.torch_profile_steps != 0
        or args.torch_profile_output_dir is not None
    ):
        parser.error("Torch profile options require --torch-profile")
    if args.evidence_class == "performance":
        if args.performance_phase is None:
            parser.error("performance evidence requires --performance-phase")
        if args.runner_source_archive is None:
            parser.error("performance evidence requires --runner-source-archive")

    evaluation_contract = None
    if args.evaluation_contract is not None:
        evaluation_contract = json.loads(
            args.evaluation_contract.read_text(encoding="utf-8")
        )
        if (
            int(evaluation_contract.get("metric_accounting_version") or 0)
            != REQUIRED_METRIC_ACCOUNTING_VERSION
        ):
            parser.error(
                f"evaluation contract does not require metric accounting version "
                f"{REQUIRED_METRIC_ACCOUNTING_VERSION}"
            )

    head, status = _git_state()
    if not head:
        if not args.runner_source_revision:
            raise RuntimeError(
                "no Git metadata; pass the exact committed --runner-source-revision"
            )
        head = args.runner_source_revision
    elif args.runner_source_revision and args.runner_source_revision != head:
        raise RuntimeError(
            f"--runner-source-revision {args.runner_source_revision} does not "
            f"match Git HEAD {head}"
        )
    if status and not args.allow_dirty:
        raise RuntimeError("refusing calibration from a dirty source tree")
    runner_source_artifact = None
    if evaluation_contract is not None:
        runner_source_artifact = _validate_contract_runner_artifact(
            evaluation_contract, head, args.runner_source_archive
        )
    elif args.runner_source_archive is not None:
        runner_source_artifact = _artifact_record(args.runner_source_archive)
    config = json.loads(args.qps_config.read_text(encoding="utf-8"))
    duration_from_config = config.get("duration_s")
    if duration_from_config is not None:
        args.duration_s = float(duration_from_config)
    if args.evidence_class == "smoke" and (
        args.reps != 1
        or args.rep_start != 1
        or args.campaign_total_reps not in (0, 1)
        or args.duration_s > 60
        or args.min_prompts > 32
    ):
        parser.error(
            "smoke evidence requires reps=1, duration<=60s, and "
            "min-prompts<=32; it cannot support performance decisions"
        )
    workloads = config.get("workloads") or {}
    if not workloads:
        raise ValueError("QPS config has no workloads")
    for workload, qps_values in workloads.items():
        if not isinstance(qps_values, list) or not qps_values:
            raise ValueError(f"{workload} has no QPS values")
        normalized = [float(value) for value in qps_values]
        if any(not math.isfinite(value) or value <= 0 for value in normalized):
            raise ValueError(f"{workload} contains a non-positive or non-finite QPS")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{workload} contains duplicate QPS values")
    if args.evidence_class == "smoke":
        violations = _smoke_request_count_violations(
            workloads, args.min_prompts, args.duration_s
        )
        if violations:
            details = ", ".join(
                f"{workload}@{qps:g}={count}"
                for workload, qps, count in violations
            )
            parser.error(
                f"smoke evidence allows at most {MAX_SMOKE_REQUESTS_PER_CELL} "
                f"actual requests per cell; computed {details}"
            )
    if evaluation_contract is not None:
        systems = set(evaluation_contract.get("systems") or ())
        if args.experiment not in systems:
            parser.error(
                f"experiment {args.experiment!r} is not in the evaluation manifest"
            )
        allowed_workloads = set(evaluation_contract.get("workloads") or ())
        unknown_workloads = sorted(set(workloads) - allowed_workloads)
        if unknown_workloads:
            parser.error(
                f"QPS config contains workloads outside the contract: {unknown_workloads}"
            )
        contract_lane = args.contract_lane or args.performance_phase
        _validate_contract_phase_qps(evaluation_contract, contract_lane, workloads)
        traffic = evaluation_contract.get("traffic") or {}
        if args.duration_s != float(traffic.get("duration_seconds") or 0):
            parser.error("duration does not match the evaluation manifest")
        if args.min_prompts < int(traffic.get("minimum_requests") or 0):
            parser.error("minimum prompts is below the evaluation manifest")
        if args.warmup_requests != int(traffic.get("warmup_requests") or 0):
            parser.error("warmup requests does not match the evaluation manifest")
        expected_system = _validate_contract_system_artifact_shape(
            evaluation_contract, args.experiment
        )
        if expected_system["source_revision"] != head:
            raise ValueError(
                f"system artifact source revision does not match runner: "
                f"expected {expected_system['source_revision']}, observed {head}"
            )
        if (
            runner_source_artifact is not None
            and expected_system["source_archive_sha256"]
            != runner_source_artifact["sha256"]
        ):
            raise ValueError(
                "system artifact source archive does not match runner archive"
            )
        required_reps = _required_contract_repetitions(
            evaluation_contract, contract_lane
        )
        complete_in_one_run = args.reps == required_reps and args.rep_start == 1
        campaign_slice = (
            args.reps == 1
            and args.campaign_total_reps == required_reps
            and 1 <= args.rep_start <= required_reps
        )
        if not (complete_in_one_run or campaign_slice):
            parser.error(
                f"{contract_lane} requires {required_reps} total "
                "repetitions, either in one run or as an identified campaign slice"
            )

    if args.contract_preflight_only:
        if evaluation_contract is None:
            parser.error("contract preflight requires --evaluation-contract")
        _validate_workload_inputs(
            workloads,
            args.workload_dir,
            str(evaluation_contract.get("model") or ""),
            str(evaluation_contract.get("model_revision") or ""),
            int(evaluation_contract.get("model_context_length") or 0),
            evaluation_contract,
        )
        print(
            json.dumps(
                {
                    "contract_id": evaluation_contract.get("contract_id"),
                    "contract_lane": args.contract_lane or args.performance_phase,
                    "experiment": args.experiment,
                    "status": "passed",
                },
                sort_keys=True,
            )
        )
        return

    deployment_path = args.deployment_manifest.resolve()
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    required_deployment_fields = (
        "deployment_id",
        "system_id",
        "model",
        "model_revision",
        "client_tokenizer_path",
        "client_tokenizer_files",
        "source_revision",
        "launch_command",
        "endpoint",
        "expected_runtime",
        "observed_runtime",
        "server_identity",
        "server_identity_sha256",
    )
    missing = _missing_required_fields(deployment, required_deployment_fields)
    if missing:
        raise ValueError(f"deployment manifest is missing required fields: {missing}")
    endpoint = deployment["endpoint"]
    if not isinstance(endpoint, dict) or not endpoint.get("host") or not endpoint.get(
        "port"
    ):
        raise ValueError("deployment endpoint must contain host and port")
    args.host = args.host or str(endpoint["host"])
    args.port = args.port or int(endpoint["port"])
    args.model = args.model or str(deployment["model"])
    if args.model != deployment["model"]:
        raise ValueError(
            f"runner model {args.model!r} does not match deployment model "
            f"{deployment['model']!r}"
        )
    if deployment["system_id"] != args.experiment:
        raise ValueError(
            f"deployment system {deployment['system_id']!r} does not match "
            f"experiment {args.experiment!r}"
        )
    if evaluation_contract is not None:
        if deployment["model"] != evaluation_contract.get("model"):
            raise ValueError("deployment model does not match evaluation contract")
        if deployment["model_revision"] != evaluation_contract.get(
            "model_revision"
        ):
            raise ValueError(
                "deployment model revision does not match evaluation contract"
            )
        _validate_contract_system_artifact(
            evaluation_contract, args.experiment, deployment
        )
    deployment_sha256 = _sha256(deployment_path)
    args.deployment_sha256 = deployment_sha256
    args.deployment = deployment
    args.client_tokenizer_path = str(deployment["client_tokenizer_path"])
    _validate_deployment_artifacts(deployment)
    workload_records = _validate_workload_inputs(
        workloads,
        args.workload_dir,
        str(deployment["model"]),
        str(deployment["model_revision"]),
        (
            int(evaluation_contract.get("model_context_length") or 0)
            if evaluation_contract is not None
            else None
        ),
        evaluation_contract,
    )
    args.workload_records = workload_records

    run_manifest_path = args.output_dir / "run_manifest.json"
    if args.output_dir.exists():
        if not args.resume_completed:
            raise FileExistsError(
                f"refusing to reuse evaluation output directory: {args.output_dir}"
            )
        if not run_manifest_path.is_file():
            raise FileExistsError(
                f"resume directory has no run manifest: {args.output_dir}"
            )
    else:
        args.output_dir.mkdir(parents=True)
    run_manifest = {
        "created_unix_s": time.time(),
        "runner_git_head": head,
        "runner_git_status": status,
        "runner_source_artifact": runner_source_artifact,
        "experiment": args.experiment,
        "evidence_class": args.evidence_class,
        "metric_accounting_version_required": REQUIRED_METRIC_ACCOUNTING_VERSION,
        "evaluation_contract_path": (
            str(args.evaluation_contract.resolve())
            if args.evaluation_contract is not None
            else None
        ),
        "evaluation_contract_sha256": (
            _sha256(args.evaluation_contract)
            if args.evaluation_contract is not None
            else None
        ),
        "evaluation_contract": evaluation_contract,
        "performance_phase": args.performance_phase,
        "contract_lane": args.contract_lane or args.performance_phase,
        "model": args.model,
        "endpoint": {"host": args.host, "port": args.port},
        "deployment_manifest_path": str(deployment_path),
        "deployment_manifest_sha256": deployment_sha256,
        "deployment": deployment,
        "qps_config_path": str(args.qps_config.resolve()),
        "qps_config_sha256": _sha256(args.qps_config),
        "max_concurrency": None,
        "qps_config": config,
        "backend": args.backend,
        "arrival_schedule_version": ARRIVAL_SCHEDULE_VERSION,
        "workloads": workload_records,
        "duration_s": args.duration_s,
        "min_prompts": args.min_prompts,
        "reps": args.reps,
        "rep_start": args.rep_start,
        "campaign_total_reps": args.campaign_total_reps or args.reps,
        "seed": args.seed,
        "max_walltime_s": args.max_walltime_s,
        "load_sample_interval_ms": args.load_sample_interval_ms,
        "resume_completed": args.resume_completed,
        "continue_after_accounting_rejection": (
            args.continue_after_accounting_rejection
        ),
        "allow_code_execution": args.allow_code_execution,
    }
    if run_manifest_path.is_file():
        existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        immutable_fields = (
            "runner_git_head",
            "runner_source_artifact",
            "experiment",
            "evidence_class",
            "metric_accounting_version_required",
            "evaluation_contract_sha256",
            "model",
            "deployment_manifest_sha256",
            "qps_config_sha256",
            "backend",
            "arrival_schedule_version",
            "duration_s",
            "min_prompts",
            "reps",
            "rep_start",
            "campaign_total_reps",
            "seed",
            "continue_after_accounting_rejection",
            "allow_code_execution",
        )
        mismatches = [
            field
            for field in immutable_fields
            if existing_manifest.get(field) != run_manifest.get(field)
        ]
        if mismatches:
            raise RuntimeError(
                "resume manifest differs in immutable fields: " + ", ".join(mismatches)
            )
        run_manifest = existing_manifest
    else:
        _write_json(run_manifest_path, run_manifest)
    if args.dry_run:
        print(json.dumps(run_manifest, sort_keys=True), flush=True)
        for workload, qps_values in workloads.items():
            for rep in range(args.rep_start, args.rep_start + args.reps):
                for qps in qps_values:
                    _run_cell(
                        args,
                        workload,
                        float(qps),
                        rep,
                        args.workload_dir / f"{workload}.requests.jsonl",
                        args.workload_dir / f"{workload}.metadata.jsonl",
                    )
        return

    suite_started = time.monotonic()
    _wait_ready(args.host, args.port, args.ready_timeout_s)
    server_info = _validate_live_deployment(args.host, args.port, deployment)
    _write_json(args.output_dir / "server_info.before.json", server_info)
    rejected_cells = []
    try:
        for workload, qps_values in workloads.items():
            requests_path = args.workload_dir / f"{workload}.requests.jsonl"
            metadata_path = args.workload_dir / f"{workload}.metadata.jsonl"
            for rep in range(args.rep_start, args.rep_start + args.reps):
                for qps_value in qps_values:
                    elapsed = time.monotonic() - suite_started
                    if elapsed >= args.max_walltime_s:
                        raise TimeoutError(
                            f"suite reached hard wall-time limit {args.max_walltime_s}s"
                        )
                    try:
                        _run_cell(
                            args,
                            workload,
                            float(qps_value),
                            rep,
                            requests_path,
                            metadata_path,
                        )
                    except StrictAccountingError as error:
                        if not args.continue_after_accounting_rejection:
                            raise
                        qps_label = _qps_label(float(qps_value))
                        label = f"{workload}_qps{qps_label}_rep{rep}"
                        rejected_cells.append(
                            {
                                "label": label,
                                "error": str(error),
                            }
                        )
                        print(
                            f"DEVELOPMENT_CONTINUE rejected_cell={label}",
                            flush=True,
                        )
    except Exception as error:
        run_manifest["failed_unix_s"] = time.time()
        run_manifest["status"] = "failed"
        run_manifest["error"] = f"{type(error).__name__}: {error}"
        try:
            server_info = _validate_live_deployment(args.host, args.port, deployment)
            _write_json(args.output_dir / "server_info.after.json", server_info)
        except Exception as capture_error:
            run_manifest["server_info_after_error"] = (
                f"{type(capture_error).__name__}: {capture_error}"
            )
        _write_json(run_manifest_path, run_manifest)
        raise
    server_info = _validate_live_deployment(args.host, args.port, deployment)
    _write_json(args.output_dir / "server_info.after.json", server_info)
    if rejected_cells:
        run_manifest["failed_unix_s"] = time.time()
        run_manifest["status"] = "failed"
        run_manifest["rejected_cells"] = rejected_cells
        run_manifest["error"] = (
            f"{len(rejected_cells)} cells failed strict accounting"
        )
        _write_json(run_manifest_path, run_manifest)
        raise RuntimeError(run_manifest["error"])
    run_manifest["completed_unix_s"] = time.time()
    run_manifest["status"] = "completed"
    _write_json(run_manifest_path, run_manifest)


if __name__ == "__main__":
    main()
