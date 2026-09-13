"""Fixture test of the default-conformance core: the 2026-09-13 headline server_info
against the defaults upstream computed for it, with the D-187 exemptions declared."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_default_conformance import conformance  # noqa: E402

DEFAULTS = {"dtype": "auto", "mem_fraction_static": 0.83, "attention_backend": "flashinfer",
            "prefill_attention_backend": None, "decode_attention_backend": None,
            "sampling_backend": "flashinfer", "chunked_prefill_size": 8192,
            "max_running_requests": None, "revision": None, "port": 30000}
EXEMPT = {"attention_backend": {"value": "triton", "decision": "D-187"},
          "prefill_attention_backend": {"value": "triton", "decision": "D-187"},
          "decode_attention_backend": {"value": "triton", "decision": "D-187"},
          "sampling_backend": {"value": "pytorch", "decision": "D-187/D-757"}}
HEADLINE_20260913 = {"dtype": "float16", "mem_fraction_static": 0.8,
                     "attention_backend": "triton", "prefill_attention_backend": "triton",
                     "decode_attention_backend": "triton", "sampling_backend": "pytorch",
                     "chunked_prefill_size": 8192, "max_running_requests": None,
                     "revision": "53346005fb0ef11d3b6a83b12c895cca40156b6b", "port": 31001}


def test_the_incident_is_refused():
    declared, undeclared = conformance(HEADLINE_20260913, DEFAULTS, EXEMPT)
    assert len(declared) == 4
    assert [l.split(":")[0].strip("! ") for l in undeclared] == ["dtype", "mem_fraction_static"]


def test_default_config_passes_with_declared_substrate():
    served = dict(HEADLINE_20260913, dtype="auto", mem_fraction_static=0.83)
    declared, undeclared = conformance(served, DEFAULTS, EXEMPT)
    assert undeclared == [] and len(declared) == 4


def test_exempted_field_with_a_different_value_is_refused():
    served = dict(HEADLINE_20260913, dtype="auto", mem_fraction_static=0.83,
                  sampling_backend="ascend")
    _, undeclared = conformance(served, DEFAULTS, EXEMPT)
    assert len(undeclared) == 1 and "exempted value is 'pytorch'" in undeclared[0]


def test_model_identity_is_not_a_knob():
    served = dict(HEADLINE_20260913, dtype="auto", mem_fraction_static=0.83)
    _, undeclared = conformance(served, DEFAULTS, EXEMPT)
    assert not any("revision" in l or "port" in l for l in undeclared)


def test_profile_scoped_exemptions_apply_only_under_their_profile():
    """The paper profile declares fp16/0.8; sglang_default must NOT inherit them."""
    from verify_default_conformance import conformance as conf
    paper = dict(EXEMPT, dtype={"value": "float16", "decision": "D-757 add.", "profile": "paper"},
                 mem_fraction_static={"value": 0.8, "decision": "D-757 add.", "profile": "paper"})
    declared, undeclared = conf(HEADLINE_20260913, DEFAULTS, paper)
    assert undeclared == [] and len(declared) == 6
    default_only = {k: v for k, v in paper.items() if v.get("profile") is None}
    _, undeclared = conf(HEADLINE_20260913, DEFAULTS, default_only)
    assert [l.split(":")[0].strip("! ") for l in undeclared] == ["dtype", "mem_fraction_static"]
