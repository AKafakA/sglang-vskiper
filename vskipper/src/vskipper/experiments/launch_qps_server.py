#!/usr/bin/env python3
"""Launch one auditable SGLang deployment for endpoint-based QPS evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from qps_deployment import (
    canonical_sha256,
    stable_server_identity,
    validate_expected_runtime,
)


ROOT = Path(__file__).resolve().parents[4]
SERVER_PROFILES = (
    "production",
    "full_decode",
    "breakable_dynamic",
    "breakable_decode",
    "vp_stage_graph",
    "graphless_overlap",
    "matched_eager",
)
RESERVED_SERVER_ARGS = {
    "--model",
    "--model-path",
    "--revision",
    "--served-model-name",
    "--host",
    "--port",
    "--mem-fraction-static",
}
_SHARDED_WEIGHT_PATTERN = re.compile(
    r"^(?:model|pytorch_model)-(\d{5})-of-(\d{5})\.(?:safetensors|bin)$"
)


def _git_state(root: Path) -> tuple[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return head, status
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "", ""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_assignment(value: str, option: str) -> tuple[str, str]:
    key, separator, assigned = value.partition("=")
    if not separator or not key or not assigned:
        raise ValueError(f"{option} expects NAME=VALUE, got {value!r}")
    return key, assigned


def _server_profile_args(profile: str) -> list[str]:
    if profile == "production":
        return []
    if profile == "full_decode":
        return ["--cuda-graph-backend-prefill", "disabled"]
    if profile == "breakable_dynamic":
        return [
            "--cuda-graph-backend-decode",
            "breakable",
            "--cuda-graph-backend-prefill",
            "breakable",
        ]
    if profile == "breakable_decode":
        return [
            "--cuda-graph-backend-decode",
            "breakable",
            "--cuda-graph-backend-prefill",
            "disabled",
        ]
    if profile in {"vp_stage_graph", "graphless_overlap"}:
        return ["--disable-cuda-graph"]
    if profile == "matched_eager":
        return [
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--disable-overlap-schedule",
        ]
    raise ValueError(f"unknown server profile {profile!r}")


def _validate_server_profile(
    server_info: dict[str, Any], profile: str
) -> dict[str, Any]:
    config = server_info.get("cuda_graph_config")
    if not isinstance(config, dict):
        raise ValueError("/server_info is missing cuda_graph_config")
    decode = config.get("decode")
    prefill = config.get("prefill")
    if not isinstance(decode, dict) or not isinstance(prefill, dict):
        raise ValueError("/server_info has incomplete CUDA graph phase config")
    observed = {
        "disable_cuda_graph": bool(server_info.get("disable_cuda_graph")),
        "decode_backend": decode.get("backend"),
        "prefill_backend": prefill.get("backend"),
        "decode_batch_sizes": decode.get("bs"),
        "decode_max_batch_size": decode.get("max_bs"),
        "prefill_batch_sizes": prefill.get("bs"),
        "prefill_max_batch_size": prefill.get("max_bs"),
    }
    if profile == "production":
        if observed["disable_cuda_graph"]:
            raise ValueError("production profile unexpectedly disabled CUDA graphs")
        if observed["decode_backend"] == "disabled":
            raise ValueError("production profile has disabled decode CUDA graphs")
        if observed["prefill_backend"] == "disabled":
            raise ValueError("production profile has disabled prefill CUDA graphs")
    elif profile == "full_decode":
        if observed["disable_cuda_graph"] or (
            observed["decode_backend"], observed["prefill_backend"]
        ) != ("full", "disabled"):
            raise ValueError(
                "full_decode profile did not resolve to full/disabled"
            )
    elif profile == "breakable_dynamic":
        if observed["disable_cuda_graph"] or (
            observed["decode_backend"], observed["prefill_backend"]
        ) != ("breakable", "breakable"):
            raise ValueError(
                "breakable_dynamic profile did not resolve both phases to breakable"
            )
    elif profile == "breakable_decode":
        if observed["disable_cuda_graph"] or (
            observed["decode_backend"], observed["prefill_backend"]
        ) != ("breakable", "disabled"):
            raise ValueError(
                "breakable_decode profile did not resolve to breakable/disabled"
            )
    elif profile in {"vp_stage_graph", "graphless_overlap", "matched_eager"}:
        if not observed["disable_cuda_graph"]:
            raise ValueError(f"{profile} profile unexpectedly enabled CUDA graphs")
        if (
            observed["decode_backend"] != "disabled"
            or observed["prefill_backend"] != "disabled"
        ):
            raise ValueError(f"{profile} profile retained a CUDA graph backend")
    else:
        raise ValueError(f"unknown server profile {profile!r}")
    return observed


def _validate_server_args(server_args: list[str], cap_purpose: str) -> None:
    for value in server_args:
        option = value.split("=", 1)[0]
        if option in RESERVED_SERVER_ARGS:
            raise ValueError(f"--server-arg may not override reserved option {option}")
    has_running_cap = any(
        value.split("=", 1)[0] == "--max-running-requests"
        for value in server_args
    )
    if has_running_cap and cap_purpose != "pressure":
        raise ValueError(
            "--max-running-requests requires --server-cap-purpose pressure"
        )


def _effective_decode_graph_backend(
    profile: str, server_args: list[str]
) -> str:
    backend = {
        "production": "full",
        "full_decode": "full",
        "breakable_dynamic": "breakable",
        "breakable_decode": "breakable",
        "vp_stage_graph": "disabled",
        "graphless_overlap": "disabled",
        "matched_eager": "disabled",
    }[profile]
    index = 0
    while index < len(server_args):
        value = server_args[index]
        option, separator, assigned = value.partition("=")
        if option == "--disable-cuda-graph":
            backend = "disabled"
        elif option == "--cuda-graph-backend-decode":
            if separator:
                backend = assigned.strip().lower()
            elif index + 1 < len(server_args):
                index += 1
                backend = server_args[index].strip().lower()
            else:
                raise ValueError("--cuda-graph-backend-decode is missing a value")
        index += 1
    return backend


def _validate_regime_switch_launch(
    configured_environment: dict[str, str],
    expected_runtime: dict[str, Any],
) -> None:
    """Cross-check ``SGLANG_VP_REGIME_SWITCH`` against the attested regime state.

    The switch is active iff the env value is a non-empty, non-``"off"``
    string.  When active the expected runtime must attest
    ``regime_switch.enabled=true``; when inactive it must not.
    """

    raw = configured_environment.get("SGLANG_VP_REGIME_SWITCH", "").strip()
    env_enabled = bool(raw) and raw.lower() != "off"
    attested = (
        expected_runtime.get("runtime", {})
        .get("regime_switch", {})
        .get("enabled")
    )
    if env_enabled and attested is not True:
        raise ValueError(
            "SGLANG_VP_REGIME_SWITCH is set but the expected runtime does not "
            "attest regime_switch.enabled=true"
        )
    if not env_enabled and attested is True:
        raise ValueError(
            "the expected runtime attests regime_switch.enabled=true but "
            "SGLANG_VP_REGIME_SWITCH is not set"
        )


def _validate_coverage_dense_launch(
    configured_environment: dict[str, str],
    expected_runtime: dict[str, Any],
) -> None:
    """Bidirectionally cross-check the (c3) coverage-dense arming.

    The mechanism arms iff FD weights are staged, execution mode is
    ``full_graph``, AND the ``SGLANG_VP_COVERAGE_DENSE`` kill switch is not
    explicitly OFF (default ON).  Armed boots must attest ``fd_c3.enabled=true``
    in the expected runtime; unarmed boots (production, direct-eager, or a
    gate-OFF byte-parity diagnostic) must not carry the block.
    """

    raw = configured_environment.get("SGLANG_VP_COVERAGE_DENSE", "").strip().lower()
    if raw not in {"", "1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError("SGLANG_VP_COVERAGE_DENSE must be boolean")
    gate_on = raw not in {"0", "false", "no", "off"}
    fd_deployed = bool(
        configured_environment.get("SGLANG_FD_WEIGHTS", "").strip()
        and configured_environment.get("SGLANG_FD_EXECUTION_MODE", "")
        .strip()
        .lower()
        == "full_graph"
    )
    armed = fd_deployed and gate_on
    attested = (
        expected_runtime.get("runtime", {}).get("fd_c3", {}).get("enabled")
    )
    if armed and attested is not True:
        raise ValueError(
            "the (c3) coverage-dense mechanism is armed (FD full_graph, "
            "SGLANG_VP_COVERAGE_DENSE on) but the expected runtime does not "
            "attest fd_c3.enabled=true"
        )
    if not armed and attested is True:
        raise ValueError(
            "the expected runtime attests fd_c3.enabled=true but the launch "
            "does not arm the (c3) coverage-dense mechanism"
        )


def _validate_flexidepth_launch_mode(
    profile: str,
    server_args: list[str],
    configured_environment: dict[str, str],
    expected_runtime: dict[str, Any],
) -> None:
    _validate_regime_switch_launch(configured_environment, expected_runtime)
    _validate_coverage_dense_launch(configured_environment, expected_runtime)
    execution_mode = configured_environment.get(
        "SGLANG_FD_EXECUTION_MODE", "direct_eager"
    ).strip().lower()
    raw_debug = configured_environment.get(
        "SGLANG_FD_FULL_GRAPH_EAGER_SEMANTIC_DEBUG", "0"
    ).strip().lower()
    valid_bools = {"0", "1", "false", "true", "no", "yes", "off", "on", ""}
    if raw_debug not in valid_bools:
        raise ValueError(
            "SGLANG_FD_FULL_GRAPH_EAGER_SEMANTIC_DEBUG must be boolean"
        )
    eager_semantic_debug = raw_debug in {"1", "true", "yes", "on"}
    raw_rebatching_debug = configured_environment.get(
        "SGLANG_VP_V4_DEVICE_REBATCHING_EAGER_SEMANTIC", "0"
    ).strip().lower()
    if raw_rebatching_debug not in valid_bools:
        raise ValueError(
            "SGLANG_VP_V4_DEVICE_REBATCHING_EAGER_SEMANTIC must be boolean"
        )
    rebatching_eager_semantic = raw_rebatching_debug in {
        "1",
        "true",
        "yes",
        "on",
    }
    decode_backend = _effective_decode_graph_backend(profile, server_args)
    if eager_semantic_debug and rebatching_eager_semantic:
        raise ValueError("only one eager semantic debug mode may be enabled")
    if eager_semantic_debug and execution_mode != "full_graph":
        raise ValueError(
            "SGLANG_FD_FULL_GRAPH_EAGER_SEMANTIC_DEBUG=1 requires "
            "SGLANG_FD_EXECUTION_MODE=full_graph"
        )
    if rebatching_eager_semantic and execution_mode != "full_graph":
        raise ValueError(
            "SGLANG_VP_V4_DEVICE_REBATCHING_EAGER_SEMANTIC=1 requires "
            "SGLANG_FD_EXECUTION_MODE=full_graph"
        )
    if execution_mode != "full_graph":
        return
    if eager_semantic_debug:
        if profile != "matched_eager" or decode_backend != "disabled":
            raise ValueError(
                "full-graph eager semantic debug requires the matched_eager profile"
            )
        for name in (
            "SGLANG_FD_PARITY_TRACE_RID",
            "SGLANG_FD_PARITY_TRACE_DIR",
        ):
            if not configured_environment.get(name, "").strip():
                raise ValueError(
                    "full-graph eager semantic debug requires " + name
                )
        expected_debug = (
            expected_runtime.get("runtime", {})
            .get("v3_full_graph", {})
            .get("eager_semantic_debug")
        )
        if expected_debug is not True:
            raise ValueError(
                "expected runtime must attest "
                "v3_full_graph.eager_semantic_debug=true"
            )
        return
    if rebatching_eager_semantic:
        if profile != "graphless_overlap" or decode_backend != "disabled":
            raise ValueError(
                "rebatching eager semantic debug requires the "
                "graphless_overlap profile"
            )
        trace_rid = configured_environment.get(
            "SGLANG_FD_PARITY_TRACE_RID", ""
        ).strip()
        trace_dir = configured_environment.get(
            "SGLANG_FD_PARITY_TRACE_DIR", ""
        ).strip()
        if bool(trace_rid) != bool(trace_dir):
            raise ValueError(
                "rebatching eager semantic parity tracing requires both "
                "SGLANG_FD_PARITY_TRACE_RID and "
                "SGLANG_FD_PARITY_TRACE_DIR"
            )
        expected_rebatching = (
            expected_runtime.get("runtime", {})
            .get("v4", {})
            .get("execution", {})
            .get("enable_device_rebatching")
        )
        if expected_rebatching is not True:
            raise ValueError(
                "expected runtime must attest "
                "v4.execution.enable_device_rebatching=true"
            )
        return
    if decode_backend != "full":
        raise ValueError(
            "SGLANG_FD_EXECUTION_MODE=full_graph requires an effective full "
            "decode CUDA-graph backend"
        )


def _model_server_args(
    model: str, model_path: str | None, revision: str
) -> list[str]:
    source = model_path or model
    result = ["--model-path", source, "--revision", revision]
    if model_path:
        result.extend(["--served-model-name", model])
    return result


def _build_environment(
    assignments: list[str],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    environment = os.environ.copy()
    removed = [
        key
        for key in sorted(environment)
        if key.startswith(("SGLANG_FD_", "SGLANG_VP_", "VP_"))
    ]
    for key in removed:
        environment.pop(key)
    configured: dict[str, str] = {}
    for assignment in assignments:
        key, value = _parse_assignment(assignment, "--server-env")
        if not key.startswith("SGLANG_") and key not in {
            "HF_HOME",
            "HF_HUB_OFFLINE",
        }:
            raise ValueError(
                "--server-env only accepts explicit SGLANG_* variables or the "
                f"pinned Hugging Face cache controls, got {key!r}"
            )
        if key in configured:
            raise ValueError(f"duplicate --server-env key {key!r}")
        configured[key] = value
    environment.update(configured)
    return environment, configured, removed


def _resolve_model_snapshot(
    model: str,
    revision: str,
    environment: dict[str, str],
) -> Path:
    local = Path(model).expanduser()
    if local.is_dir():
        return local.resolve()
    if "/" not in model:
        raise FileNotFoundError(
            f"model is neither a local directory nor a repository id: {model}"
        )
    cache_root = Path(
        environment.get("HF_HOME", "~/.cache/huggingface")
    ).expanduser()
    repo_dir = f"models--{model.replace('/', '--')}"
    candidates = (
        cache_root / "hub" / repo_dir / "snapshots" / revision,
        cache_root / repo_dir / "snapshots" / revision,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"pinned model snapshot is not staged for {model}@{revision}: "
        + ", ".join(str(path) for path in candidates)
    )


def _validate_model_snapshot(snapshot: Path, revision: str) -> dict[str, Any]:
    if snapshot.parent.name == "snapshots" and snapshot.name != revision:
        raise ValueError(
            f"model snapshot {snapshot} does not match revision {revision}"
        )
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model snapshot is missing {config_path}")

    index_paths = (
        snapshot / "model.safetensors.index.json",
        snapshot / "pytorch_model.bin.index.json",
    )
    index_path = next((path for path in index_paths if path.is_file()), None)
    if index_path is not None:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"model index has no weight_map: {index_path}")
        raw_shard_names = list(weight_map.values())
        if not all(isinstance(name, str) and name for name in raw_shard_names):
            raise ValueError(f"model index has invalid shard names: {index_path}")
        shard_names = sorted(set(raw_shard_names))
        missing = [name for name in shard_names if not (snapshot / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"incomplete sharded model snapshot {snapshot}; missing {missing}"
            )
    else:
        shard_paths = sorted(
            path
            for pattern in ("model-*-of-*.safetensors", "pytorch_model-*-of-*.bin")
            for path in snapshot.glob(pattern)
        )
        if shard_paths:
            parsed = [
                _SHARDED_WEIGHT_PATTERN.fullmatch(path.name)
                for path in shard_paths
            ]
            if any(match is None for match in parsed):
                raise ValueError(f"unrecognized model shard names in {snapshot}")
            totals = {int(match.group(2)) for match in parsed if match is not None}
            indices = {int(match.group(1)) for match in parsed if match is not None}
            if len(totals) != 1 or indices != set(
                range(1, next(iter(totals)) + 1)
            ):
                raise FileNotFoundError(
                    f"incomplete sharded model snapshot {snapshot}; "
                    f"observed shards {[path.name for path in shard_paths]}"
                )
            shard_names = [path.name for path in shard_paths]
        else:
            unsharded = next(
                (
                    path
                    for path in (
                        snapshot / "model.safetensors",
                        snapshot / "pytorch_model.bin",
                    )
                    if path.is_file()
                ),
                None,
            )
            if unsharded is None:
                raise FileNotFoundError(
                    f"model snapshot has no complete weight file or index: {snapshot}"
                )
            shard_names = [unsharded.name]

    shard_paths = [snapshot / name for name in shard_names]
    return {
        "config_sha256": _sha256(config_path),
        "index_path": str(index_path) if index_path is not None else None,
        "index_sha256": _sha256(index_path) if index_path is not None else None,
        "path": str(snapshot),
        "shard_count": len(shard_paths),
        "shard_names": shard_names,
        "total_weight_bytes": sum(path.stat().st_size for path in shard_paths),
    }


def _artifact_records(assignments: list[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        name, raw_path = _parse_assignment(assignment, "--artifact")
        if name in records:
            raise ValueError(f"duplicate --artifact name {name!r}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing deployment artifact {name}: {path}")
        records[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def _wait_ready(
    host: str, port: int, process: subprocess.Popen[Any], timeout_s: int
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before readiness with {process.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    with urllib.request.urlopen(
                        f"http://{host}:{port}/server_info", timeout=30
                    ) as info_response:
                        return json.load(info_response)
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become ready in {timeout_s}s")


def _require_unused_endpoint(host: str, port: int) -> None:
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        connection = socket.create_connection((connect_host, port), timeout=1)
    except OSError:
        return
    connection.close()
    raise RuntimeError(f"refusing to launch on occupied endpoint {host}:{port}")


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


def _raise_keyboard_interrupt(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--client-tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-source-root", type=Path, required=True)
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--server-profile", choices=SERVER_PROFILES, required=True)
    parser.add_argument("--server-env", action="append", default=[])
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument(
        "--server-cap-purpose", choices=("auto", "pressure"), default="auto"
    )
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--expected-runtime", type=Path, required=True)
    parser.add_argument("--ready-timeout-s", type=int, default=600)
    parser.add_argument("--launcher-source-revision", default="")
    parser.add_argument("--server-source-revision", "--source-revision", default="")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.mem_fraction_static <= 1.0:
        parser.error("--mem-fraction-static must be in (0, 1]")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    _validate_server_args(args.server_arg, args.server_cap_purpose)

    launcher_head, launcher_status = _git_state(ROOT)
    if not launcher_head:
        if not args.launcher_source_revision:
            raise RuntimeError(
                "launcher has no Git metadata; pass --launcher-source-revision"
            )
        launcher_head = args.launcher_source_revision
    elif (
        args.launcher_source_revision
        and args.launcher_source_revision != launcher_head
    ):
        raise RuntimeError(
            f"--launcher-source-revision {args.launcher_source_revision} does not "
            f"match Git HEAD {launcher_head}"
        )
    if launcher_status and not args.allow_dirty:
        raise RuntimeError("refusing deployment from a dirty launcher source tree")

    server_root = args.server_source_root.resolve()
    server_python = args.server_python.expanduser().absolute()
    if not (server_root / "python/sglang").is_dir():
        raise FileNotFoundError(
            f"server source root has no python/sglang package: {server_root}"
        )
    if not server_python.is_file() or not os.access(server_python, os.X_OK):
        raise FileNotFoundError(f"server Python is not executable: {server_python}")
    server_head, server_status = _git_state(server_root)
    if not server_head:
        if not args.server_source_revision:
            raise RuntimeError(
                "server source has no Git metadata; pass --server-source-revision"
            )
        server_head = args.server_source_revision
    elif args.server_source_revision and args.server_source_revision != server_head:
        raise RuntimeError(
            f"--server-source-revision {args.server_source_revision} does not "
            f"match server Git HEAD {server_head}"
        )
    if server_status and not args.allow_dirty:
        raise RuntimeError("refusing deployment from a dirty server source tree")

    client_tokenizer_path = args.client_tokenizer_path.resolve()
    if client_tokenizer_path.name != args.model_revision:
        raise ValueError(
            "client tokenizer snapshot directory does not match --model-revision"
        )
    tokenizer_files = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        path = client_tokenizer_path / name
        if not path.is_file():
            raise FileNotFoundError(f"client tokenizer snapshot is missing {path}")
        tokenizer_files[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }

    environment, configured_environment, removed_environment = _build_environment(
        args.server_env
    )
    environment["PYTHONPATH"] = os.pathsep.join((str(server_root / "vskipper/src"), str(server_root / "python"))) if (server_root / "vskipper/src").is_dir() else str(server_root / "python")
    model_source = args.model_path or args.model
    model_snapshot = _validate_model_snapshot(
        _resolve_model_snapshot(model_source, args.model_revision, environment),
        args.model_revision,
    )
    artifacts = _artifact_records(args.artifact)
    expected_runtime_path = args.expected_runtime.resolve()
    expected_runtime = json.loads(expected_runtime_path.read_text(encoding="utf-8"))
    if not isinstance(expected_runtime, dict) or not expected_runtime:
        raise ValueError("--expected-runtime must contain a non-empty JSON object")
    _validate_flexidepth_launch_mode(
        args.server_profile,
        args.server_arg,
        configured_environment,
        expected_runtime,
    )
    artifacts["expected_runtime"] = {
        "path": str(expected_runtime_path),
        "sha256": _sha256(expected_runtime_path),
        "size_bytes": expected_runtime_path.stat().st_size,
    }
    server_command = [
        str(server_python),
        "-m",
        "sglang.launch_server",
        *_model_server_args(args.model, args.model_path, args.model_revision),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        *_server_profile_args(args.server_profile),
        *args.server_arg,
    ]
    server_working_directory = args.output_dir.absolute() / "runtime"
    manifest = {
        "deployment_id": args.deployment_id,
        "system_id": args.system_id,
        "model": args.model,
        "model_path": model_source,
        "model_revision": args.model_revision,
        "model_snapshot": model_snapshot,
        "client_tokenizer_path": str(client_tokenizer_path),
        "client_tokenizer_files": tokenizer_files,
        "source_revision": server_head,
        "source_status": server_status,
        "server_source_root": str(server_root),
        "server_python": str(server_python),
        "server_python_realpath": os.path.realpath(server_python),
        "server_pythonpath": environment["PYTHONPATH"],
        "server_working_directory": str(server_working_directory),
        "launcher_source_revision": launcher_head,
        "launcher_source_status": launcher_status,
        "launcher_source_root": str(ROOT),
        "endpoint": {"host": args.host, "port": args.port},
        "server_profile": args.server_profile,
        "launch_command": server_command,
        "configured_environment": configured_environment,
        "removed_inherited_mode_variables": removed_environment,
        "artifacts": artifacts,
        "expected_runtime": expected_runtime,
        "max_running_requests": (
            "explicit"
            if any(
                value.split("=", 1)[0] == "--max-running-requests"
                for value in args.server_arg
            )
            else "auto"
        ),
        "server_cap_purpose": args.server_cap_purpose,
        "created_unix_s": time.time(),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse deployment output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    server_working_directory.mkdir()
    _require_unused_endpoint(args.host, args.port)
    starting_path = args.output_dir / "deployment.starting.json"
    manifest_path = args.output_dir / "deployment_manifest.json"
    state_path = args.output_dir / "deployment_state.json"
    server_log_path = args.output_dir / "server.log"
    _write_json(starting_path, manifest)
    with server_log_path.open("w", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            server_command,
            cwd=server_working_directory,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    previous_sigint = signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        server_info = _wait_ready(args.host, args.port, process, args.ready_timeout_s)
        # Persist the OBSERVED runtime BEFORE any validation judges it: a boot
        # rejected by the profile/expectation gates otherwise leaves no record
        # of what it actually was, which blocks the sanctioned derive-expected
        # procedure (07-25) and weakens auditability. Observation is evidence;
        # saving it is unconditional (immediate-mirror ops law).
        _write_json(args.output_dir / "server_info.observed.json", server_info)
        observed_server_profile = _validate_server_profile(
            server_info, args.server_profile
        )
        observed_runtime = validate_expected_runtime(server_info, expected_runtime)
        server_identity = stable_server_identity(server_info)
        manifest["ready_unix_s"] = time.time()
        manifest["server_info"] = server_info
        manifest["observed_server_profile"] = observed_server_profile
        manifest["observed_runtime"] = observed_runtime
        manifest["server_identity"] = server_identity
        manifest["server_identity_sha256"] = canonical_sha256(server_identity)
        _write_json(manifest_path, manifest)
        _write_json(
            state_path,
            {"status": "ready", "server_pid": process.pid, "updated_unix_s": time.time()},
        )
        print(f"READY manifest={manifest_path} pid={process.pid}", flush=True)
        returncode = process.wait()
        _write_json(
            state_path,
            {
                "status": "exited",
                "server_pid": process.pid,
                "returncode": returncode,
                "updated_unix_s": time.time(),
            },
        )
        if returncode:
            raise SystemExit(returncode)
    except BaseException:
        _stop_process_group(process)
        _write_json(
            state_path,
            {
                "status": "stopped",
                "server_pid": process.pid,
                "returncode": process.poll(),
                "updated_unix_s": time.time(),
            },
        )
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
