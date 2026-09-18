"""Startup gates for durable arm selection and invalid execution postures."""
import pytest

from vskipper.runtime import common, design, validation

ROUTED = list(range(16, 32))


@pytest.fixture(autouse=True)
def _host_and_arm(monkeypatch):
    monkeypatch.setitem(design._ARM_CACHE, "name", "vskipper")
    monkeypatch.setattr(design, "_host", lambda: {
        "flexidepth_weights": "/fixture/router.pt",
        "conditional_graph_helper": "/fixture/helper.so",
    })


def _validate(loaded, *, graphs=True, tp=1, pp=1, fd=None, env=None):
    validation.validate_full_graph_model_configuration(
        loaded_layers=loaded,
        loaded_flexidepth_layers=loaded if fd is None else fd,
        tp_size=tp, pp_size=pp, quant_config=None,
        environ={} if env is None else env, cuda_graph_enabled=graphs,
    )


@pytest.mark.parametrize("graphs", [False, True])
def test_stock_arm_starts_without_routed_layers(monkeypatch, graphs):
    monkeypatch.setitem(design._ARM_CACHE, "name", "stock")
    assert common.flexidepth_execution_mode({}) == "direct_eager"
    _validate([], graphs=graphs)


def test_served_arm_uses_full_graph_and_starts():
    assert common.flexidepth_execution_mode({}) == "full_graph"
    assert common.flexidepth_execution_mode({"SGLANG_FD_EXECUTION_MODE": "direct_eager"}) == "full_graph"
    _validate(ROUTED)


def test_routed_eager_plus_graphs_is_rejected_at_startup(monkeypatch):
    # Inject an inconsistent execution reader to test the fail-closed boundary.
    # The serving arm's real reader is fixed to full_graph, checked above.
    monkeypatch.setattr(validation, "flexidepth_execution_mode", lambda values: "direct_eager")
    with pytest.raises(ValueError, match="cannot run with CUDA"):
        _validate(ROUTED)


def test_loaded_router_layers_must_match():
    with pytest.raises(ValueError, match="routed layers must match"):
        _validate(ROUTED, fd=ROUTED[:-1])


@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_unsupported_parallel_topology_is_rejected(tp, pp):
    with pytest.raises(ValueError, match="TP=1 and PP=1"):
        _validate(ROUTED, tp=tp, pp=pp)


FLAG_CASES = [
    ({"SGLANG_VP_SCHED": value}, rejected)
    for value, rejected in (("1", True), ("true", True), ("on", True),
                            ("0", False), ("false", False), ("no", False),
                            ("off", False), ("False", False), ("banana", True))
] + [
    ({"SGLANG_FD_VP_PROJECT": "false"}, False),
    ({"SGLANG_FD_VP_PROJECT": "1"}, True),
    ({"SGLANG_FD_VP_STAGE_ROUTE": "off"}, False),
    ({"SGLANG_VP_V4_CONFIG": "/fixture/config.json"}, True),
    ({"SGLANG_VP_V4_CONFIG": "0"}, True),
    ({"SGLANG_VP_V4_CONFIG": ""}, False),
    ({"SGLANG_VP_V2_CONFIG": "/fixture/config.json"}, True),
]


@pytest.mark.parametrize("env,rejected", FLAG_CASES)
def test_removed_mode_flag_rejection_matches_shape(env, rejected):
    if rejected:
        with pytest.raises(ValueError, match="not part of|must be a boolean|removed V2/V4"):
            validation.assert_no_removed_execution_flags(env)
    else:
        validation.assert_no_removed_execution_flags(env)
