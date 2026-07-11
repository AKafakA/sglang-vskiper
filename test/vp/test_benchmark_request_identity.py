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


def test_attach_stable_sglang_rid_rejects_ambiguous_identity() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        REQUEST_IDENTITY.attach_stable_sglang_rid(
            "sglang-oai-chat", "frozen", {"rid": "different"}
        )
    with pytest.raises(ValueError, match="non-empty"):
        REQUEST_IDENTITY.attach_stable_sglang_rid("sglang-oai-chat", "", {})
    with pytest.raises(ValueError, match="SGLang OpenAI"):
        REQUEST_IDENTITY.attach_stable_sglang_rid("vllm-chat", "frozen", {})
