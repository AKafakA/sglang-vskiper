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


def test_qwen_arm_band_follows_the_rule_with_its_own_inputs():
    from sglang.srt.vpipe import design
    from sglang.srt.vpipe.roofline import arm_kv_rule_inputs, assert_kv_band_follows_rule, derived_kv_band

    arm = design.ARMS["vskipper_qwen3_4b"]
    inputs = arm_kv_rule_inputs(arm)
    assert inputs["routed_layers"] == 18 and abs(inputs["skip_ratio"] - 0.25) < 1e-9
    assert derived_kv_band("NVIDIA_A100", **inputs) == tuple(arm["decode_kv_band"]["NVIDIA_A100"]) == (320_000, 390_000)
    assert_kv_band_follows_rule(served_exit_kv_tokens=320_000, served_enter_kv_tokens=390_000, device_key="NVIDIA_A100", **inputs)
    # the Llama band must refuse under the Qwen arm's inputs, and vice versa
    for band, kw in (((160_000, 200_000), inputs), ((320_000, 390_000), arm_kv_rule_inputs(design.ARMS["vskipper"]))):
        try:
            assert_kv_band_follows_rule(served_exit_kv_tokens=band[0], served_enter_kv_tokens=band[1], device_key="NVIDIA_A100", **kw)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{band} must not pass with inputs {kw}")
    # the Llama arm's inputs reproduce the served Llama band
    assert derived_kv_band("NVIDIA_A100", **arm_kv_rule_inputs(design.ARMS["vskipper"])) == (160_000, 200_000)


def test_weights_key_selects_the_checkpoint_file(tmp_path):
    import json, os
    from sglang.srt.vpipe import design

    llama = tmp_path / "llama.pt"; llama.write_bytes(b"x")
    qwen = tmp_path / "qwen.pt"; qwen.write_bytes(b"y")
    host = tmp_path / "host.json"
    host.write_text(json.dumps({"host": "t", "flexidepth_weights": str(llama), "flexidepth_weights_qwen3_4b": str(qwen), "served_model_revision": "r", "gate_workdir": str(tmp_path), "serve_python": "python", "model_path": str(tmp_path)}))
    saved_env = os.environ.get(design._HOST_CONFIG_ENV); os.environ[design._HOST_CONFIG_ENV] = str(host)
    design._HOST_CACHE.clear()
    try:
        with _serving("vskipper"):
            assert design.flexidepth_weights_path() == str(llama)
        with _serving("vskipper_qwen3_4b"):
            assert design.flexidepth_weights_path() == str(qwen)
        host.write_text(json.dumps({"host": "t", "flexidepth_weights": str(llama), "served_model_revision": "r", "gate_workdir": str(tmp_path), "serve_python": "python", "model_path": str(tmp_path)}))
        design._HOST_CACHE.clear()
        with _serving("vskipper_qwen3_4b"):
            try:
                design.flexidepth_weights_path()
            except ValueError as e:
                assert "flexidepth_weights_qwen3_4b" in str(e)
            else:
                raise AssertionError("a missing checkpoint key must refuse")
    finally:
        design._HOST_CACHE.clear()
        if saved_env is None: os.environ.pop(design._HOST_CONFIG_ENV, None)
        else: os.environ[design._HOST_CONFIG_ENV] = saved_env


def test_qwen_sharedband_arm_is_a_declared_deviation():
    """[D-778] The shared-band posture serves the GLOBAL (Llama) band under the Qwen arm's inputs: the rule
    would refuse that band, so `decode_kv_band_policy: shared` must be the only reason it boots, and the
    attestation must say so; every other field is the learned arm's."""
    from sglang.srt.vpipe import design
    from sglang.srt.vpipe.roofline import arm_kv_rule_inputs, assert_kv_band_follows_rule

    arm = design.ARMS["vskipper_qwen3_4b_sharedband"]; base = design.ARMS["vskipper_qwen3_4b"]
    assert arm["decode_kv_band_policy"] == "shared" and base.get("decode_kv_band_policy", "rule") == "rule"
    assert tuple(arm["decode_kv_band"]["NVIDIA_A100"]) == tuple(design.SERVED_DECODE_KV_BAND_BY_DEVICE["NVIDIA_A100"]) == (160_000, 200_000)
    assert {k: v for k, v in arm.items() if k not in ("decode_kv_band", "decode_kv_band_policy")} == {k: v for k, v in base.items() if k != "decode_kv_band"}
    try:
        assert_kv_band_follows_rule(served_exit_kv_tokens=160_000, served_enter_kv_tokens=200_000, device_key="NVIDIA_A100", **arm_kv_rule_inputs(arm))
    except RuntimeError:
        pass
    else:
        raise AssertionError("the rule must refuse the Llama band under the Qwen arm's inputs; the policy is the only permit")
    with _serving("vskipper_qwen3_4b_sharedband"):
        att = design.design_attestation()
        assert att["decode_kv_band_policy"] == "shared" and att["decode_kv_band_by_device"]["NVIDIA_A100"] == [160_000, 200_000]
    with _serving("vskipper_qwen3_4b"):
        assert design.design_attestation()["decode_kv_band_policy"] == "rule"
