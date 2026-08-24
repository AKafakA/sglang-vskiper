#!/usr/bin/env python3
"""Build frozen request/label manifests for the VP QPS benchmark suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from labeled_workload import (
    ALL_DATASETS,
    DATASET_PROTOCOLS,
    DEFAULT_FREQUENCY_PENALTY,
    FREQUENCY_PENALTY_FIELD,
    PROTOCOL_SCHEMA_VERSION,
    PROTOCOL_SUITE_ID,
    WORKLOAD_DATASETS,
    build_workload,
    parse_weights,
    read_jsonl,
    validate_frequency_penalty,
    write_jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def workload_config_sha256(config_path: Path, frequency_penalty: float) -> str:
    """Config identity of a build: the reviewed config file AND the penalty.

    The generation penalty is a builder input that never appears in the config
    file, so hashing the file alone lets a rebuild drop the policy while keeping
    the recorded identity — exactly the failure this binding closes. The default
    policy hashes to the bare file digest so that a penalty-free suite keeps the
    identity it has always had; every other policy produces a different digest.
    """

    penalty = validate_frequency_penalty(frequency_penalty)
    file_sha256 = _sha256(config_path)
    if penalty == DEFAULT_FREQUENCY_PENALTY:
        return file_sha256
    return _canonical_sha256(
        {
            FREQUENCY_PENALTY_FIELD: penalty,
            "workload_config_file_sha256": file_sha256,
        }
    )


def declared_frequency_penalty(summary: dict) -> float:
    """Read the penalty a frozen summary declares; absence is an error.

    An absent field means the suite was built before the penalty was a recorded
    builder input, or by a post-pass outside the sealed builder. Either way the
    suite cannot state its own generation policy, so it is not admissible.
    """

    if FREQUENCY_PENALTY_FIELD not in summary:
        raise ValueError(
            f"summary does not declare {FREQUENCY_PENALTY_FIELD}; the generation "
            "penalty must be an explicit recorded builder input, never an "
            "implicit default"
        )
    return validate_frequency_penalty(summary[FREQUENCY_PENALTY_FIELD])


def validate_frozen_output_policy(
    requests: list[dict],
    metadata: list[dict],
    context_length: int,
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_frequency_penalty: float | None = None,
) -> None:
    if len(requests) != len(metadata):
        raise ValueError("request and metadata counts differ")
    if expected_frequency_penalty is not None:
        expected_frequency_penalty = validate_frequency_penalty(
            expected_frequency_penalty
        )
    request_ids: list[str] = []
    for index, (request, row) in enumerate(zip(requests, metadata)):
        if int(row.get("protocol_schema_version") or 0) != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(f"row {index} is not protocol schema v4 evidence")
        if int(row.get("context_length") or 0) != context_length:
            raise ValueError(
                f"row {index} is not frozen context-length evidence"
            )
        if expected_model is not None and row.get("model") != expected_model:
            raise ValueError(f"row {index} model identity mismatch")
        if (
            expected_model_revision is not None
            and row.get("model_revision") != expected_model_revision
        ):
            raise ValueError(f"row {index} model revision mismatch")
        request_id = str(row.get("request_id") or "")
        if not request_id or request.get("request_id") != request_id:
            raise ValueError(f"row {index} request ID mismatch")
        request_ids.append(request_id)
        if row.get("protocol_suite_id") != PROTOCOL_SUITE_ID:
            raise ValueError(f"row {index} protocol suite mismatch")
        if not row.get("protocol_id") or not row.get("quality_semantics"):
            raise ValueError(f"row {index} has incomplete dataset protocol metadata")
        dataset = str(row.get("dataset") or "")
        expected_protocol = DATASET_PROTOCOLS.get(dataset)
        if expected_protocol is None:
            raise ValueError(f"row {index} has unknown dataset {dataset!r}")
        if row["protocol_id"] != expected_protocol["id"]:
            raise ValueError(f"row {index} dataset protocol ID mismatch")
        if row["quality_semantics"] != expected_protocol["quality_semantics"]:
            raise ValueError(f"row {index} quality semantics mismatch")
        if int(row.get("index", -1)) != index:
            raise ValueError(f"row {index} metadata index mismatch")
        prompt = request.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(not isinstance(token, int) for token in prompt)
        ):
            raise ValueError(f"row {index} does not contain frozen prompt token IDs")
        prompt_len = int(row.get("prompt_len") or 0)
        if prompt_len <= 0:
            raise ValueError(f"row {index} has invalid prompt length {prompt_len}")
        requested = int(row.get("requested_output_len") or 0)
        if int(request.get("prompt_len") or 0) != prompt_len:
            raise ValueError(f"row {index} prompt length mismatch")
        if len(prompt) != prompt_len:
            raise ValueError(f"row {index} frozen token count mismatch")
        prompt_payload = json.dumps(prompt, separators=(",", ":")).encode()
        if row.get("prompt_sha256") != hashlib.sha256(prompt_payload).hexdigest():
            raise ValueError(f"row {index} frozen prompt hash mismatch")
        if int(request.get("output_len") or 0) != requested:
            raise ValueError(f"row {index} requested output mismatch")
        policy = row.get("output_policy")
        if policy == "remaining_model_context":
            if requested != context_length - prompt_len:
                raise ValueError(f"row {index} does not use remaining context")
        elif policy == "task_reference_limit":
            task_limit = int(row.get("task_reference_max_output_len") or 0)
            if task_limit <= 0:
                raise ValueError(f"row {index} has no task reference limit")
            if requested != min(task_limit, context_length - prompt_len):
                raise ValueError(f"row {index} does not use its task reference limit")
        elif policy == "production_watchdog":
            task_limit = int(row.get("task_reference_max_output_len") or 0)
            production_max = int(
                row.get("watchdog_production_max_output_len") or 0
            )
            production_context_hit = row.get("watchdog_production_context_hit")
            multiplier = row.get("watchdog_multiplier")
            repetitions = int(row.get("watchdog_production_repetitions") or 0)
            if task_limit <= 0:
                raise ValueError(f"row {index} has no watchdog task floor")
            if production_max <= 0:
                raise ValueError(f"row {index} has no production watchdog length")
            if not isinstance(production_context_hit, bool):
                raise ValueError(
                    f"row {index} has no production context-hit classification"
                )
            if (
                not isinstance(multiplier, (int, float))
                or not math.isfinite(float(multiplier))
                or float(multiplier) < 1.0
            ):
                raise ValueError(f"row {index} has an invalid watchdog multiplier")
            if repetitions < 1:
                raise ValueError(
                    f"row {index} has no production watchdog repetitions"
                )
            watchdog_limit = max(
                task_limit, math.ceil(float(multiplier) * production_max)
            )
            if requested != min(watchdog_limit, context_length - prompt_len):
                raise ValueError(
                    f"row {index} does not use its production watchdog limit"
                )
        elif policy == "production_max_equal_work":
            observations = row.get("equal_work_production_output_lens")
            finish_reasons = row.get("equal_work_production_finish_reasons")
            selected = int(row.get("equal_work_production_max_output_len") or 0)
            repetitions = int(row.get("equal_work_production_repetitions") or 0)
            manifest_sha256 = str(row.get("equal_work_manifest_sha256") or "")
            if (
                not isinstance(observations, list)
                or len(observations) != 3
                or any(not isinstance(value, int) or value <= 0 for value in observations)
            ):
                raise ValueError(
                    f"row {index} has invalid equal-work production lengths"
                )
            if (
                not isinstance(finish_reasons, list)
                or len(finish_reasons) != 3
                or any(not isinstance(value, str) or not value for value in finish_reasons)
            ):
                raise ValueError(
                    f"row {index} has invalid equal-work finish reasons"
                )
            if repetitions != 3 or selected != max(observations):
                raise ValueError(
                    f"row {index} does not use the three-repetition production maximum"
                )
            if requested != selected or prompt_len + requested > context_length:
                raise ValueError(f"row {index} has invalid equal-work capacity")
            if (
                len(manifest_sha256) != 64
                or any(char not in "0123456789abcdef" for char in manifest_sha256)
            ):
                raise ValueError(f"row {index} has invalid equal-work manifest hash")
        elif policy == "bank_pinned_natural_length":
            # Owner-ordered work-identity instrument (2026-08-19): per-request
            # lengths pinned from ONE natural-length bank (vPipe or P-def
            # distribution), ignore_eos bodies. One observation per request
            # BY DESIGN — never dressed up as the 3-rep production policy.
            pinned = int(row.get("bank_pinned_output_len") or 0)
            bank_sha256 = str(row.get("bank_sha256") or "")
            if pinned <= 0 or requested != pinned:
                raise ValueError(
                    f"row {index} does not match its bank-pinned length"
                )
            if prompt_len + requested > context_length:
                raise ValueError(f"row {index} has invalid bank-pinned capacity")
            if (
                len(bank_sha256) != 64
                or any(char not in "0123456789abcdef" for char in bank_sha256)
            ):
                raise ValueError(f"row {index} has invalid bank hash")
        elif policy == "prefill_probe_one_token":
            if requested != 1:
                raise ValueError(f"row {index} prefill probe is not one token")
        elif policy == "fixed_token_capacity":
            fixed_output_tokens = int(row.get("fixed_output_tokens") or 0)
            if fixed_output_tokens <= 0 or requested != fixed_output_tokens:
                raise ValueError(
                    f"row {index} does not use its fixed capacity length"
                )
            if prompt_len + requested > context_length:
                raise ValueError(f"row {index} fixed capacity length exceeds context")
        else:
            raise ValueError(f"row {index} has unknown output policy {policy!r}")
        request_body = request.get("extra_request_body") or {}
        if policy in {
            "fixed_token_capacity",
            "production_max_equal_work",
            "bank_pinned_natural_length",
        }:
            if request_body.get("ignore_eos") is not True:
                raise ValueError(f"row {index} equal work must ignore EOS")
            if "stop" in request_body:
                raise ValueError(f"row {index} equal work must not use stop strings")
        elif request_body.get("ignore_eos") is not False:
            raise ValueError(f"row {index} natural output must honor EOS")
        if expected_frequency_penalty is not None:
            body_penalty = validate_frequency_penalty(
                request_body.get(FREQUENCY_PENALTY_FIELD, DEFAULT_FREQUENCY_PENALTY)
            )
            if body_penalty != expected_frequency_penalty:
                raise ValueError(
                    f"row {index} carries {FREQUENCY_PENALTY_FIELD}={body_penalty} "
                    f"but the suite declares {expected_frequency_penalty}; a "
                    "generation penalty may only come from the recorded builder "
                    "input"
                )
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("frozen workload contains duplicate request IDs")


def extend_existing_rows(
    existing_requests: list[dict],
    existing_metadata: list[dict],
    candidate_requests: list[dict],
    candidate_metadata: list[dict],
    target_count: int,
) -> tuple[list[dict], list[dict]]:
    if len(existing_requests) != len(existing_metadata):
        raise ValueError("existing request and metadata counts differ")
    if len(candidate_requests) != len(candidate_metadata):
        raise ValueError("candidate request and metadata counts differ")
    requests = list(existing_requests[:target_count])
    metadata = [dict(row) for row in existing_metadata[:target_count]]
    seen = {str(row["request_id"]) for row in metadata}
    if len(seen) != len(metadata):
        raise ValueError("existing metadata contains duplicate request IDs")
    for request, row in zip(candidate_requests, candidate_metadata):
        if len(requests) >= target_count:
            break
        request_id = str(row["request_id"])
        if request_id in seen:
            continue
        seen.add(request_id)
        appended = dict(row)
        appended["index"] = len(requests)
        requests.append(request)
        metadata.append(appended)
    if len(requests) < target_count:
        raise ValueError(
            f"could only extend workload to {len(requests)} unique rows; "
            f"target is {target_count}"
        )
    for index, row in enumerate(metadata):
        row["index"] = index
    return requests, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload", choices=sorted(WORKLOAD_DATASETS), required=True
    )
    parser.add_argument(
        "--mode", choices=("decode", "prefill", "mixed"), default=""
    )
    parser.add_argument(
        "--datasets",
        default="",
        help=f"Comma-separated override; available: {','.join(ALL_DATASETS)}",
    )
    parser.add_argument("--dataset-weights", default="")
    parser.add_argument("--workload-config", type=Path, default=None)
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=DEFAULT_FREQUENCY_PENALTY,
        help=(
            "Uniform generation penalty applied to every request body and "
            "recorded in the suite summary and config identity hash."
        ),
    )
    parser.add_argument("--decode-ratio", type=float, default=None)
    parser.add_argument("--num-requests", type=int, default=0)
    parser.add_argument("--request-rate", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--extend-existing", action="store_true")
    parser.add_argument("--repeat-exhausted", action="store_true")
    parser.add_argument(
        "--full-dataset-pool",
        action="store_true",
        help=(
            "Perf/throughput lane only: include the TRAIN split as serving prompts "
            "for gsm8k (train+test ~8.8k) and coqa (train+val ~7.7k). The quality lane "
            "and cross-lane references stay on the eval split (eval-split items are "
            "byte-identical when off). humaneval is unaffected."
        ),
    )
    args = parser.parse_args()
    frequency_penalty = validate_frequency_penalty(args.frequency_penalty)
    summary_path = args.summary or args.requests.with_suffix(".summary.json")
    if not args.extend_existing:
        existing_outputs = [
            path
            for path in (args.requests, args.metadata, summary_path)
            if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "refusing to overwrite frozen workload artifacts: "
                + ", ".join(str(path) for path in existing_outputs)
            )

    num_requests = args.num_requests
    if num_requests <= 0:
        if args.request_rate <= 0:
            parser.error("set --num-requests or a positive --request-rate")
        num_requests = max(1, math.ceil(args.request_rate * args.duration_s))

    workload_config = {}
    if args.workload_config:
        workload_config = json.loads(args.workload_config.read_text(encoding="utf-8"))
    else:
        parser.error(
            "--workload-config is required; output policy must be explicit and "
            "reviewed"
        )
    if int(workload_config.get("schema_version") or 0) != PROTOCOL_SCHEMA_VERSION:
        parser.error(
            f"workload-config schema_version must equal {PROTOCOL_SCHEMA_VERSION}"
        )
    if workload_config.get("protocol_suite_id") != PROTOCOL_SUITE_ID:
        parser.error(f"workload-config must pin protocol_suite_id={PROTOCOL_SUITE_ID}")
    configured_model = str(workload_config.get("model") or "")
    configured_revision = str(workload_config.get("model_revision") or "")
    if args.model and configured_model and args.model != configured_model:
        parser.error("--model does not match workload-config model")
    if (
        args.model_revision
        and configured_revision
        and args.model_revision != configured_revision
    ):
        parser.error("--model-revision does not match workload-config revision")
    args.model = args.model or configured_model
    args.model_revision = args.model_revision or configured_revision
    if not args.model or not args.model_revision:
        parser.error("workload model and immutable model revision are required")
    output_policy = workload_config.get("output_policy") or {}
    decode_output_policy = output_policy.get("decode")
    if decode_output_policy not in {
        "remaining_model_context",
        "task_reference_limit",
        "fixed_token_capacity",
    }:
        parser.error(
            "output_policy.decode must be 'remaining_model_context' or "
            "'task_reference_limit' or 'fixed_token_capacity'"
        )
    fixed_decode_output_len = output_policy.get("fixed_output_tokens")
    if decode_output_policy == "fixed_token_capacity":
        fixed_decode_output_len = int(fixed_decode_output_len or 0)
        if fixed_decode_output_len <= 0:
            parser.error(
                "fixed_token_capacity requires output_policy.fixed_output_tokens"
            )
    elif fixed_decode_output_len is not None:
        parser.error(
            "output_policy.fixed_output_tokens is only valid for "
            "fixed_token_capacity"
        )
    context_length = int(output_policy.get("context_length") or 0)
    if context_length <= 1:
        parser.error("output_policy.context_length must be explicit")
    categories = workload_config.get("categories") or {}
    sampling_overrides = workload_config.get("sampling_overrides") or {}
    if not isinstance(sampling_overrides, dict):
        parser.error("sampling_overrides must be an object keyed by dataset")
    allowed_sampling_keys = {"stop_regex", "no_stop_trim", "frequency_penalty"}
    for dataset, policy in sampling_overrides.items():
        if dataset not in ALL_DATASETS or not isinstance(policy, dict):
            parser.error(f"invalid sampling override for {dataset!r}")
        unknown_sampling_keys = sorted(set(policy) - allowed_sampling_keys)
        if unknown_sampling_keys:
            parser.error(
                f"sampling override for {dataset} has unsupported keys: "
                + ", ".join(unknown_sampling_keys)
            )
    decode_config = categories.get("decode") or {}
    prefill_config = categories.get("prefill") or {}
    selected = [value.strip() for value in args.datasets.split(",") if value.strip()]
    explicit_dataset_filter = bool(selected)
    if not selected and args.workload in ALL_DATASETS:
        selected = [args.workload]
        explicit_dataset_filter = True
    elif (
        not selected
        and args.workload == "decode_mix"
        and decode_config.get("datasets")
    ):
        selected = list(decode_config["datasets"])
    elif not selected and args.workload == "mixed" and categories:
        selected = list(decode_config.get("datasets") or []) + list(
            prefill_config.get("datasets") or []
        )
    configured_weights = {
        **(decode_config.get("weights") or {}),
        **(prefill_config.get("weights") or {}),
    }
    command_weights = parse_weights(args.dataset_weights)
    configured_weights.update(command_weights)
    if explicit_dataset_filter:
        for name in selected:
            if name not in command_weights and configured_weights.get(name, 0.0) <= 0:
                configured_weights[name] = 1.0
    else:
        selected = [
            name for name in selected if configured_weights.get(name, 1.0) > 0
        ]
    decode_ratio = (
        args.decode_ratio
        if args.decode_ratio is not None
        else float(workload_config.get("decode_ratio", 0.5))
    )
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision
    )
    model_config = AutoConfig.from_pretrained(
        args.model, revision=args.model_revision
    )
    model_context_length = int(
        getattr(model_config, "max_position_embeddings", 0) or 0
    )
    if model_context_length != context_length:
        parser.error(
            f"contract context_length={context_length} does not match model "
            f"max_position_embeddings={model_context_length}"
        )

    existing_requests: list[dict] = []
    existing_metadata: list[dict] = []
    if args.extend_existing:
        if args.requests.is_file() != args.metadata.is_file():
            raise ValueError("both existing request and metadata files are required")
        if args.requests.is_file():
            existing_requests = read_jsonl(args.requests)
            existing_metadata = read_jsonl(args.metadata)
            validate_frozen_output_policy(
                existing_requests,
                existing_metadata,
                context_length,
                args.model,
                args.model_revision,
                frequency_penalty,
            )
    candidate_count = num_requests
    candidate_requests, candidate_metadata, summary = build_workload(
        workload=args.workload,
        num_requests=candidate_count,
        seed=args.seed,
        tokenizer=tokenizer,
        context_length=context_length,
        datasets_filter=selected or None,
        phase_mode=args.mode or None,
        decode_ratio=decode_ratio,
        weights=configured_weights,
        repeat_exhausted=args.repeat_exhausted,
        decode_output_policy=decode_output_policy,
        fixed_decode_output_len=fixed_decode_output_len,
        sampling_overrides=sampling_overrides,
        frequency_penalty=frequency_penalty,
        include_train_pool=args.full_dataset_pool,
    )
    for row in candidate_metadata:
        row["model"] = args.model
        row["model_revision"] = args.model_revision
    if existing_requests:
        requests, metadata = extend_existing_rows(
            existing_requests,
            existing_metadata,
            candidate_requests,
            candidate_metadata,
            num_requests,
        )
    else:
        requests = candidate_requests[:num_requests]
        metadata = candidate_metadata[:num_requests]
    validate_frozen_output_policy(
        requests,
        metadata,
        context_length,
        args.model,
        args.model_revision,
        frequency_penalty,
    )
    dataset_counts = Counter(row["dataset"] for row in metadata)
    phase_counts = Counter(row["phase"] for row in metadata)
    summary.update(
        {
            "num_requests": len(requests),
            "datasets": dict(sorted(dataset_counts.items())),
            "phases": dict(sorted(phase_counts.items())),
            "quality_eligible": sum(
                bool(row["quality_eligible"]) for row in metadata
            ),
            "extended_from_requests": len(existing_requests) or None,
            "model": args.model,
            "model_revision": args.model_revision,
            "model_context_length": model_context_length,
            "request_rate": args.request_rate or None,
            "duration_s": args.duration_s if args.request_rate > 0 else None,
            "repeat_exhausted": args.repeat_exhausted,
            "workload_config": args.workload_config.name,
            "workload_config_sha256": workload_config_sha256(
                args.workload_config, frequency_penalty
            ),
            "requests_path": args.requests.name,
            "metadata_path": args.metadata.name,
        }
    )
    write_jsonl(args.requests, requests)
    write_jsonl(args.metadata, metadata)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
