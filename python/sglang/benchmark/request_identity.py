"""Request-identity helpers for reproducible SGLang serving benchmarks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SGLANG_OPENAI_BACKENDS = frozenset({"sglang-oai", "sglang-oai-chat"})
SGLANG_IDENTITY_BACKENDS = frozenset(
    {"sglang", "sglang-native", *SGLANG_OPENAI_BACKENDS}
)
SGLANG_NATIVE_BACKENDS = frozenset({"sglang", "sglang-native"})
SGLANG_NATIVE_SAMPLING_KEYS = frozenset(
    {
        "custom_params",
        "ebnf",
        "frequency_penalty",
        "ignore_eos",
        "json_schema",
        "logit_bias",
        "max_new_tokens",
        "min_new_tokens",
        "min_p",
        "n",
        "no_stop_trim",
        "presence_penalty",
        "regex",
        "repetition_penalty",
        "sampling_seed",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "stop",
        "stop_regex",
        "stop_token_ids",
        "stream_interval",
        "structural_tag",
        "temperature",
        "top_k",
        "top_p",
    }
)


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using one stable serialization."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_sglang_native_request_body(
    request_body: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move flat OpenAI-style sampling controls into native sampling params."""
    body = dict(request_body)
    nested_sampling = body.pop("sampling_params", {})
    if nested_sampling is None:
        nested_sampling = {}
    if not isinstance(nested_sampling, Mapping):
        raise ValueError("native sampling_params must be an object")
    sampling_params = dict(nested_sampling)
    for key in SGLANG_NATIVE_SAMPLING_KEYS:
        if key not in body:
            continue
        value = body.pop(key)
        if key in sampling_params and sampling_params[key] != value:
            raise ValueError(
                f"conflicting native sampling parameter {key!r}: "
                f"flat={value!r}, nested={sampling_params[key]!r}"
            )
        sampling_params[key] = value
    return body, sampling_params


def generation_policy_sha256(
    *,
    backend: str,
    requested_output_len: int,
    request_body: Mapping[str, Any],
) -> str:
    """Hash effective generation controls without transport identity."""
    if requested_output_len < 0:
        raise ValueError("requested_output_len must be non-negative")
    body = dict(request_body)
    body.pop("rid", None)
    return canonical_json_sha256(
        {
            "schema": "sglang-generation-policy-v1",
            "backend": backend,
            "requested_output_len": requested_output_len,
            "request_body": body,
        }
    )


def build_one_batch_logical_ids(
    *,
    dataset_name: str,
    seed: int,
    prompt_sha256s: list[str],
    generation_policy_sha256s: list[str],
) -> list[str]:
    """Build stable row IDs for one immutable one-batch request set."""
    if not dataset_name:
        raise ValueError("dataset_name must be non-empty")
    if not prompt_sha256s or len(prompt_sha256s) != len(
        generation_policy_sha256s
    ):
        raise ValueError("prompt and generation-policy hashes must align")
    for values in (prompt_sha256s, generation_policy_sha256s):
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in values
        ):
            raise ValueError("batch identity requires lowercase SHA-256 digests")
    batch_identity = canonical_json_sha256(
        {
            "schema": "sglang-one-batch-request-set-v1",
            "dataset_name": dataset_name,
            "seed": seed,
            "batch_size": len(prompt_sha256s),
            "prompt_sha256s": prompt_sha256s,
            "generation_policy_sha256s": generation_policy_sha256s,
        }
    )
    return [
        f"one-batch:{batch_identity[:24]}:{index:06d}"
        for index in range(len(prompt_sha256s))
    ]


def reconcile_transport_request_id(
    current_request_id: str | None,
    response_request_id: Any,
) -> str | None:
    """Accept a missing stream ID or require one stable non-empty string."""
    if response_request_id is None:
        return current_request_id
    if not isinstance(response_request_id, str) or not response_request_id:
        raise ValueError(f"invalid server response id: {response_request_id!r}")
    if current_request_id not in (None, response_request_id):
        raise ValueError(
            f"server response changed transport id from {current_request_id!r} "
            f"to {response_request_id!r}"
        )
    return response_request_id


def reconcile_streamed_output_ids(
    current_output_ids: list[int],
    response_output_ids: Any,
    completion_tokens: Any,
) -> tuple[list[int], str]:
    """Reconcile native SGLang cumulative or incremental output-ID chunks."""
    if not isinstance(current_output_ids, list) or any(
        not isinstance(token_id, int) for token_id in current_output_ids
    ):
        raise ValueError("current output IDs must be a list of integers")
    if not isinstance(response_output_ids, list) or any(
        not isinstance(token_id, int) for token_id in response_output_ids
    ):
        raise ValueError("server output IDs must be a list of integers")
    if not isinstance(completion_tokens, int) or completion_tokens < 0:
        raise ValueError(
            f"invalid server completion-token count: {completion_tokens!r}"
        )

    if not response_output_ids and len(current_output_ids) == completion_tokens:
        return list(current_output_ids), "server_empty_delta"
    if len(response_output_ids) == completion_tokens:
        if current_output_ids and response_output_ids[: len(current_output_ids)] != (
            current_output_ids
        ):
            raise ValueError(
                "server cumulative output IDs do not preserve the prior prefix"
            )
        return list(response_output_ids), "server_cumulative"
    if len(current_output_ids) + len(response_output_ids) == completion_tokens:
        return [*current_output_ids, *response_output_ids], "server_incremental"
    raise ValueError(
        "server output-ID chunk cannot be reconciled with completion-token "
        f"count: current={len(current_output_ids)}, "
        f"chunk={len(response_output_ids)}, completion_tokens={completion_tokens}"
    )


def attach_stable_sglang_rid(
    backend: str,
    request_id: str,
    extra_request_body: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a request body whose SGLang RID is the frozen benchmark ID."""
    if backend not in SGLANG_IDENTITY_BACKENDS:
        raise ValueError(
            "forwarding benchmark request IDs requires an SGLang backend"
        )
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("forwarding benchmark request IDs requires a non-empty ID")
    body = dict(extra_request_body)
    configured_rid = body.get("rid")
    if configured_rid is not None and configured_rid != request_id:
        raise ValueError(
            f"request body RID {configured_rid!r} conflicts with frozen "
            f"request ID {request_id!r}"
        )
    body["rid"] = request_id
    return body
