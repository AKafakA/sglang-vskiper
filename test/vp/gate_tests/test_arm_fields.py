"""[v1.5] gate_mode / compact / routed_layers are ARM fields; the Llama arms are unchanged."""
import contextlib


@contextlib.contextmanager
def _serving(name):
    from sglang.srt.vpipe import design

    saved = design._ARM_CACHE.get("name")
    design._ARM_CACHE["name"] = name
    try:
        yield
    finally:
        if saved is None:
            design._ARM_CACHE.pop("name", None)
        else:
            design._ARM_CACHE["name"] = saved


def test_llama_arms_keep_the_served_design():
    from sglang.srt.vpipe import design
    from sglang.srt.vpipe.config import full_graph_layer_policies

    for arm in ("vskipper", "integrated_it4", "integrated_alwaysskip", "vdec_fd", "vpre_binarycohort"):
        with _serving(arm):
            att = design.design_attestation()
            assert att["gate_mode"] == "released", arm
            assert att["compact_enabled"] is True, arm
            assert att["routed_layers"] == list(range(16, 32)), arm
            assert set(full_graph_layer_policies()) == set(range(16, 32)), arm


def test_ablation_and_qwen_arms_declare_their_fields():
    from sglang.srt.vpipe import design
    from sglang.srt.vpipe.common import full_graph_gate_mode
    from sglang.srt.vpipe.config import full_graph_compact_config, full_graph_layer_policies

    with _serving("vskipper_nocompact"):
        att = design.design_attestation()
        assert att["compact_enabled"] is False and att["gate_mode"] == "released"
        assert full_graph_compact_config() is False
        assert att["routed_layers"] == list(range(16, 32))
    for arm in ("vskipper_qwen3_4b", "vskipper_qwen3_4b_alwaysroute"):
        with _serving(arm):
            att = design.design_attestation()
            assert att["gate_mode"] == "hard_mask" and full_graph_gate_mode() == "hard_mask"
            assert att["compact_enabled"] is False
            assert att["routed_layers"] == list(range(18, 36))
            assert set(full_graph_layer_policies()) == set(range(18, 36))
    assert design.ARMS["vskipper_qwen3_4b"]["regime_switch"] is True
    assert design.ARMS["vskipper_qwen3_4b_alwaysroute"]["regime_switch"] is False


def test_hard_mask_payload_semantics_are_attested():
    from sglang.srt.vpipe.skipper import FlexiDepthFullGraphAdapter

    with _serving("vskipper"):
        assert "*weight" in FlexiDepthFullGraphAdapter.__dict__["payload_semantics"].fget(object.__new__(FlexiDepthFullGraphAdapter))
    with _serving("vskipper_qwen3_4b"):
        assert FlexiDepthFullGraphAdapter.__dict__["payload_semantics"].fget(object.__new__(FlexiDepthFullGraphAdapter)) == "run=mlp(hidden);project_only=proj(hidden)"
