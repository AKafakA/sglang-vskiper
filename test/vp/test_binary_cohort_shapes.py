"""The count-adaptive binary-cohort body at the served models' REAL shapes, both gate modes.

Llama-3-8B (hidden 4096, intermediate 14336, projector bottleneck 896) is what every production
launch has exercised; Qwen3-4B (2560 / 9728 / 608) is the v1.5 feasibility model and the first shape
where K is not a multiple of every tuned BK (608 % 64 = 32) -- the first Qwen boot died with an
illegal memory access inside `count_matmul_gridexit` (D-747). GPU only; run with
CUDA_LAUNCH_BLOCKING=1 to attribute a fault to its launch.
"""
import contextlib
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SHAPES = {"llama3_8b": (4096, 14336, 896), "qwen3_4b": (2560, 9728, 608)}


@contextlib.contextmanager
def _serving_arm(name, fields):
    from sglang.srt.vpipe import design

    saved = design._ARM_CACHE.get("name"); design.ARMS[name] = fields; design._ARM_CACHE["name"] = name
    try:
        yield
    finally:
        design.ARMS.pop(name, None)
        if saved is None: design._ARM_CACHE.pop("name", None)
        else: design._ARM_CACHE["name"] = saved


def _linear(w):
    return types.SimpleNamespace(weight=w)


def _fake_modules(hidden, inter, bottleneck, device, dtype, gen):
    run_gate_up = (torch.randn(2 * inter, hidden, generator=gen, device=device) * 0.02).to(dtype)
    run_down = (torch.randn(hidden, inter, generator=gen, device=device) * 0.02).to(dtype)
    proj_gate_down = (torch.randn(2 * bottleneck, hidden, generator=gen, device=device) * 0.02).to(dtype)
    proj_up = (torch.randn(hidden, bottleneck, generator=gen, device=device) * 0.02).to(dtype)
    layer = types.SimpleNamespace(layer_id=20, mlp=types.SimpleNamespace(gate_up_proj=_linear(run_gate_up), down_proj=_linear(run_down)))
    proj = types.SimpleNamespace(up_proj=_linear(proj_up), _fused_gate_down_weight=lambda: proj_gate_down)
    return layer, proj, (run_gate_up, run_down, proj_gate_down, proj_up)


def _reference(hidden_states, run_mask, weights, valid, w, gate_mode):
    """fp32 reference: released = w*MLP / (1-w)*PROJ; hard_mask = hard selection, no scaling."""
    import torch.nn.functional as F
    run_gate_up, run_down, proj_gate_down, proj_up = (t.float() for t in w)
    x = hidden_states.float()
    gu = x @ run_gate_up.t(); g, u = gu.chunk(2, dim=1); run = (F.silu(g) * u) @ run_down.t()
    pgd = x @ proj_gate_down.t(); pg, pd = pgd.chunk(2, dim=1); proj = (F.silu(pg) * pd) @ proj_up.t()
    wf = weights.float().view(-1, 1)
    if gate_mode == "released":
        run, proj = run * wf, proj * (1.0 - wf)
    out = torch.where(run_mask.view(-1, 1), run, proj)
    return torch.where(valid.view(-1, 1), out, torch.zeros_like(out))


def _prime_scratch(device, hidden, inter, dtype, ceiling):
    """Pre-allocate the body's scratch at a known ceiling (a bare process has no global server args;
    the scratch key uses the tensor's own device, i.e. cuda:0, not the bare 'cuda' alias)."""
    import types as _t
    from sglang.srt import server_args as _sa
    from sglang.srt.vpipe import mlp as m
    _sa.get_global_server_args = lambda: _t.SimpleNamespace(chunked_prefill_size=ceiling, max_prefill_tokens=ceiling)
    key = (device, hidden, inter, dtype)
    if key not in m._BINARY_COHORT_SCRATCH:
        z = lambda *s: torch.zeros(s, device=device, dtype=dtype)
        m._BINARY_COHORT_SCRATCH[key] = {"ceiling": ceiling, "compact": z(ceiling, hidden), "gate_up": z(ceiling, 2 * inter),
                                          "activated": z(ceiling, inter), "final": z(ceiling, hidden), "out": z(ceiling, hidden)}


@pytest.mark.parametrize("model", sorted(SHAPES))
@pytest.mark.parametrize("rows", [4, 64, 205, 512])
@pytest.mark.parametrize("gate_mode", ["released", "hard_mask"])
def test_binary_cohort_body_at_model_shapes(model, rows, gate_mode):
    from sglang.srt.vpipe.mlp import _binary_cohort_mlp

    hidden, inter, bottleneck = SHAPES[model]
    device, dtype = torch.device("cuda:0"), torch.float16
    gen = torch.Generator(device=device); gen.manual_seed(rows * 31 + len(model))
    layer, proj, w = _fake_modules(hidden, inter, bottleneck, device, dtype, gen)
    _prime_scratch(device, hidden, inter, dtype, 1024)
    x = (torch.randn(rows, hidden, generator=gen, device=device) * 0.5).to(dtype)
    run_mask = (torch.rand(rows, generator=gen, device=device) >= 0.5).view(-1, 1)
    weights = torch.rand(rows, generator=gen, device=device).to(dtype).view(-1, 1)
    valid = torch.rand(rows, generator=gen, device=device) >= 0.1
    base = {"skipper": "flexidepth", "phases": "both", "regime_switch": True, "gate_mode": gate_mode}
    with _serving_arm("_t_shapes", base):
        got = _binary_cohort_mlp(layer, proj, x, weights, run_mask, valid)
        torch.cuda.synchronize()
    want = _reference(x, run_mask.view(-1), weights.view(-1), valid, w, gate_mode)
    err = (got[:rows].float() - want).abs().max().item()
    assert err < 5e-3, f"{model} rows={rows} {gate_mode}: max abs err {err:.3e}"
