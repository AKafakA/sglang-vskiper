"""Exercise the documented plugin against the real tensor/action contract."""
import math

import pytest
import torch

from example_skipper_plugin import (
    StaticDepthSkipper,
    build_static_depth,
)
from sglang.srt.vpipe.skipper import (
    FLEXIDEPTH_PROJECTOR,
    available_skippers,
    build_skipper,
    register_skipper,
)
from sglang.srt.vpipe.types import LogicalAction


@pytest.mark.parametrize("ratio,selected", [
    (0.0, frozenset()), (0.5, frozenset((13, 17))),
    (1.0, frozenset((2, 7, 13, 17))),
])
def test_routes_and_padding_on_device(ratio, selected):
    assert torch.cuda.is_available(), "Run plugin feasibility on an allocated GPU host"
    hidden = torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda")
    valid = torch.tensor([True, False, True, False], device="cuda")
    layers = (2, 7, 13, 17)
    adapter = StaticDepthSkipper(ratio)
    state = adapter.prepare_batch(
        hidden_states=hidden, valid_rows=valid, route_layer_order=layers)
    assert state[0] == selected
    for layer in layers:
        actions = adapter.route(hidden, router=None, layer_id=layer, batch_state=state)
        expected = ((~valid) if layer in selected else torch.ones_like(valid)).view(-1, 1)
        assert actions.adapter_name == "static_depth"
        assert actions.payload_semantics == "binary_static_depth"
        assert actions.supported_actions == frozenset(
            (LogicalAction.RUN, LogicalAction.PROJECT_ONLY))
        assert actions.threshold is None and actions.forced_action is None
        assert actions.branch_weights.shape == (4, 1)
        assert actions.branch_weights.dtype == hidden.dtype
        assert actions.branch_weights.device == hidden.device
        assert torch.equal(actions.explicit_run_mask, expected)
        assert torch.equal(actions.branch_weights, expected.to(hidden.dtype))
        out = torch.empty_like(expected)
        actions.write_run_storage(out)
        assert torch.equal(out, expected)
        assert out[~valid].all(), "Padding must always RUN"
        again = adapter.route(hidden, router=None, layer_id=layer, batch_state=state)
        assert again.branch_weights.data_ptr() == actions.branch_weights.data_ptr()


@pytest.mark.parametrize("ratio", [-0.1, 1.1, math.nan, math.inf, -math.inf, "invalid"])
def test_invalid_ratios(ratio):
    with pytest.raises(ValueError):
        StaticDepthSkipper(ratio)


def test_empty_routed_set():
    adapter = StaticDepthSkipper(1)
    state = adapter.prepare_batch(
        hidden_states=torch.zeros((2, 4)), valid_rows=torch.tensor([True, False]),
        route_layer_order=())
    assert state[0] == frozenset()
    actions = adapter.route(None, router=None, layer_id=0, batch_state=state)
    assert actions.explicit_run_mask.all()


@pytest.mark.parametrize("kwargs", [
    {}, {"layer_id": 2}, {"batch_state": ()},
    {"layer_id": 2, "batch_state": (), "forced_action": LogicalAction.RUN},
])
def test_missing_context_and_forced_routes(kwargs):
    with pytest.raises(ValueError):
        StaticDepthSkipper(0.5).route(None, router=None, **kwargs)


def test_registration_and_attestation():
    assert "static_depth" in available_skippers()
    register_skipper("static_depth", build_static_depth)  # same factory is idempotent
    adapter = build_skipper("static_depth", {"static_depth_ratio": 0.5})
    assert isinstance(adapter, StaticDepthSkipper)
    assert adapter.attestation() == {
        "name": "static_depth", "decision": "static_tail_fraction",
        "static_depth_ratio": 0.5, "requires_router_weights": False,
        "projector_kind": FLEXIDEPTH_PROJECTOR,
        "supported_actions": ["run", "project_only"],
    }
    with pytest.raises(ValueError, match="already registered"):
        register_skipper("static_depth", lambda arm: StaticDepthSkipper(0))
