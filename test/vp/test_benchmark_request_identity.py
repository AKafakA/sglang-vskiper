import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "benchmark"
    / "request_identity.py"
)
SPEC = importlib.util.spec_from_file_location("request_identity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
REQUEST_IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REQUEST_IDENTITY)


def test_attach_stable_sglang_rid_preserves_body_without_mutation() -> None:
    source = {"temperature": 0.0}
    result = REQUEST_IDENTITY.attach_stable_sglang_rid(
        "sglang-oai-chat", "gsm8k:test:1", source
    )
    assert result == {"temperature": 0.0, "rid": "gsm8k:test:1"}
    assert source == {"temperature": 0.0}
    assert REQUEST_IDENTITY.attach_stable_sglang_rid(
        "sglang-native", "gsm8k:test:1", source
    ) == {"temperature": 0.0, "rid": "gsm8k:test:1"}


def test_attach_stable_sglang_rid_rejects_ambiguous_identity() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        REQUEST_IDENTITY.attach_stable_sglang_rid(
            "sglang-oai-chat", "frozen", {"rid": "different"}
        )
    with pytest.raises(ValueError, match="non-empty"):
        REQUEST_IDENTITY.attach_stable_sglang_rid("sglang-oai-chat", "", {})
    with pytest.raises(ValueError, match="SGLang backend"):
        REQUEST_IDENTITY.attach_stable_sglang_rid("vllm-chat", "frozen", {})


def test_canonical_json_hash_is_order_independent_for_mappings() -> None:
    left = REQUEST_IDENTITY.canonical_json_sha256({"b": [2, 3], "a": 1})
    right = REQUEST_IDENTITY.canonical_json_sha256({"a": 1, "b": [2, 3]})
    assert left == right
    assert len(left) == 64


def test_generation_policy_hash_excludes_transport_rid_only() -> None:
    left = REQUEST_IDENTITY.generation_policy_sha256(
        backend="sglang-oai",
        requested_output_len=256,
        request_body={"temperature": 0.0, "ignore_eos": True, "rid": "left"},
    )
    right = REQUEST_IDENTITY.generation_policy_sha256(
        backend="sglang-oai",
        requested_output_len=256,
        request_body={"ignore_eos": True, "temperature": 0.0, "rid": "right"},
    )
    changed = REQUEST_IDENTITY.generation_policy_sha256(
        backend="sglang-oai",
        requested_output_len=256,
        request_body={"temperature": 0.0, "ignore_eos": False, "rid": "left"},
    )
    assert left == right
    assert left != changed


def test_native_request_body_moves_flat_sampling_controls() -> None:
    body, sampling = REQUEST_IDENTITY.split_sglang_native_request_body(
        {
            "rid": "request-1",
            "temperature": 0.0,
            "ignore_eos": True,
            "stop": ["done"],
            "return_logprob": True,
        }
    )
    assert body == {"rid": "request-1", "return_logprob": True}
    assert sampling == {
        "temperature": 0.0,
        "ignore_eos": True,
        "stop": ["done"],
    }


def test_native_request_body_rejects_conflicting_sampling_controls() -> None:
    with pytest.raises(ValueError, match="conflicting native sampling"):
        REQUEST_IDENTITY.split_sglang_native_request_body(
            {
                "temperature": 0.0,
                "sampling_params": {"temperature": 1.0},
            }
        )


def test_one_batch_logical_ids_are_deterministic_and_content_bound() -> None:
    kwargs = {
        "dataset_name": "random",
        "seed": 42,
        "prompt_sha256s": ["0" * 64, "1" * 64],
        "generation_policy_sha256s": ["2" * 64, "3" * 64],
    }
    first = REQUEST_IDENTITY.build_one_batch_logical_ids(**kwargs)
    second = REQUEST_IDENTITY.build_one_batch_logical_ids(**kwargs)
    changed = REQUEST_IDENTITY.build_one_batch_logical_ids(
        **{**kwargs, "prompt_sha256s": ["0" * 64, "f" * 64]}
    )
    assert first == second
    assert first != changed
    assert len(first) == len(set(first)) == 2


def test_transport_request_id_must_remain_stable_across_stream_chunks() -> None:
    reconcile = REQUEST_IDENTITY.reconcile_transport_request_id
    assert reconcile(None, None) is None
    assert reconcile(None, "request-1") == "request-1"
    assert reconcile("request-1", None) == "request-1"
    assert reconcile("request-1", "request-1") == "request-1"
    with pytest.raises(ValueError, match="changed transport id"):
        reconcile("request-1", "request-2")
    with pytest.raises(ValueError, match="invalid server response id"):
        reconcile(None, "")


def test_reconcile_streamed_output_ids_accepts_cumulative_and_incremental() -> None:
    reconcile = REQUEST_IDENTITY.reconcile_streamed_output_ids
    output_ids, mode = reconcile([], [10], 1)
    assert output_ids == [10]
    assert mode == "server_cumulative"

    output_ids, mode = reconcile(output_ids, [11], 2)
    assert output_ids == [10, 11]
    assert mode == "server_incremental"

    output_ids, mode = reconcile(output_ids, [10, 11, 12], 3)
    assert output_ids == [10, 11, 12]
    assert mode == "server_cumulative"

    output_ids, mode = reconcile(output_ids, [], 3)
    assert output_ids == [10, 11, 12]
    assert mode == "server_empty_delta"


def test_reconcile_streamed_output_ids_rejects_unaccounted_chunks() -> None:
    reconcile = REQUEST_IDENTITY.reconcile_streamed_output_ids
    with pytest.raises(ValueError, match="do not preserve"):
        reconcile([10], [11, 12], 2)
    with pytest.raises(ValueError, match="cannot be reconciled"):
        reconcile([10], [11, 12], 4)
    with pytest.raises(ValueError, match="list of integers"):
        reconcile([], ["not-an-id"], 1)
