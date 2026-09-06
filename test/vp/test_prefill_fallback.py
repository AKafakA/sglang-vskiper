"""D-508 — per-layer break-even fallback on eager prefill passes
(SGLANG_FD_FULL_GRAPH_PREFILL_FALLBACK_MIN_PROJECT=<share>).

Below the PROJECT-share threshold a routed layer must run the exact dense full-dual body (bit-identical to
`_full_dual_mlp`, padded rows zeroed) and count the decision; at or above it the dispatcher must continue to the
configured body and count only the check. The env reader fails closed on malformed values. CPU-only environments run
the reader tests and skip the GPU dispatch tests.
"""

import os

import pytest
import torch

os.environ.setdefault("SGLANG_FD_FULL_GRAPH_COMPACT", "1")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="dispatch test needs CUDA")


class _SA:
    chunked_prefill_size = 8192
    max_prefill_tokens = 16384
    enable_deterministic_inference = False

    def __getattr__(self, name):
        return None


class _Lin:
    def __init__(self, w):
        self.weight = w


class _StubMLP(torch.nn.Module):
    def __init__(self, H, I, dtype, dev):
        super().__init__()
        from sglang.srt.layers.activation import SiluAndMul

        g = torch.Generator(device=dev).manual_seed(11)
        self.gate_up_proj = _Lin((torch.randn(2 * I, H, device=dev, generator=g) * 0.02).to(dtype))
        self.down_proj = _Lin((torch.randn(H, I, device=dev, generator=g) * 0.02).to(dtype))
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gu = torch.nn.functional.linear(x, self.gate_up_proj.weight)
        return torch.nn.functional.linear(self.act_fn(gu), self.down_proj.weight)


class _Layer:
    def __init__(self, mlp):
        self.mlp = mlp
        self.layer_id = 24


def _fixture(rows, run_frac, seed):
    from sglang.srt.vpipe.routing import FDProj

    dev = torch.device("cuda")
    dtype = torch.float16
    H, I = 4096, 14336
    layer = _Layer(_StubMLP(H, I, dtype, dev))
    proj = FDProj(hidden_size=H, intermediate_size=I).to(device=dev, dtype=dtype)
    g = torch.Generator(device=dev).manual_seed(seed)
    h = (torch.randn(rows, H, device=dev, generator=g) * 0.5).to(dtype)
    w = torch.rand(rows, 1, device=dev, generator=g).to(dtype)
    run_mask = torch.rand(rows, 1, device=dev, generator=g) < run_frac
    return layer, proj, h, w, run_mask


def test_env_reader_fails_closed():
    from sglang.srt.vpipe.common import full_graph_prefill_fallback_min_project as reader

    key = "SGLANG_FD_FULL_GRAPH_PREFILL_FALLBACK_MIN_PROJECT"
    assert reader({}) is None
    assert reader({key: ""}) is None
    assert reader({key: "0.22"}) == pytest.approx(0.22)
    assert reader({key: "1"}) == 1.0
    for bad in ("0", "-0.1", "1.5", "abc"):
        with pytest.raises(ValueError):
            reader({key: bad})


@cuda
@pytest.mark.parametrize("rows,run_frac", [(771, 0.92), (2048, 0.9)])
def test_low_project_share_falls_back_to_full_dual(monkeypatch, rows, run_frac):
    import sglang.srt.server_args as sa
    from sglang.srt.vpipe import mlp as vp_mlp

    monkeypatch.setattr(sa, "get_global_server_args", lambda: _SA())
    layer, proj, h, w, run_mask = _fixture(rows, run_frac, rows)
    # padded tail rows: the fallback must zero them exactly as every other body does
    valid = torch.ones(rows, dtype=torch.bool, device=h.device)
    valid[-7:] = False
    key = str(h.device)
    with torch.no_grad():
        # the fallback body is dense-filtered-project (dense MLP every row + projector on PROJECT rows only);
        # it must be bit-identical to that body and agree with the full-dual reference to fp16 rounding
        ref = vp_mlp._dense_filtered_project_mlp(layer, proj, h, w, run_mask)
        ref = torch.where(valid.view(-1, 1), ref, torch.zeros_like(ref))
        dual = vp_mlp._full_dual_mlp(layer, proj, h, w, run_mask)
        dual = torch.where(valid.view(-1, 1), dual, torch.zeros_like(dual))
        before = vp_mlp._PREFILL_FALLBACK.get(key, (0, 0))
        out = vp_mlp.fd_conditional_mlp_full_graph(
            layer,
            proj,
            h,
            w,
            run_mask,
            valid_rows=valid,
            compact_phase_enabled=True,
            force_dense_all_run=False,
            prefill_fallback_min_project=0.22,
        )
    after = vp_mlp._PREFILL_FALLBACK[key]
    assert after == (before[0] + 1, before[1] + 1)
    assert torch.equal(out, ref), "fallback must be the dense-filtered-project body"
    scale = dual.float().abs().max().item()
    tol = 4e-3 * scale + 1e-3  # a few fp16 ulps: the filtered projector accumulates in a different order
    assert (out.float() - dual.float()).abs().max().item() <= tol, "fallback must agree with the full-dual reference"
    # PROJECT rows checked on their own scale (Codex review: a global RUN-row scale could hide projector errors)
    proj_rows = (~run_mask.view(-1)) & valid
    if proj_rows.any():
        pscale = dual[proj_rows].float().abs().max().item()
        perr = (out[proj_rows].float() - dual[proj_rows].float()).abs().max().item()
        assert perr <= 4e-3 * pscale + 1e-3, f"PROJECT rows differ from the full-dual projector beyond fp16 rounding ({perr} vs scale {pscale})"
    assert not out[-7:].any(), "padded rows must be zero"


@cuda
def test_all_run_pass_falls_back_without_project_rows(monkeypatch):
    """Share 0 (< threshold): the filtered projector runs with an EMPTY expert; output must equal w*MLP on every row."""
    import sglang.srt.server_args as sa
    from sglang.srt.vpipe import mlp as vp_mlp

    monkeypatch.setattr(sa, "get_global_server_args", lambda: _SA())
    layer, proj, h, w, _ = _fixture(771, 1.0, 5)
    run_mask = torch.ones(771, 1, dtype=torch.bool, device=h.device)
    key = str(h.device)
    with torch.no_grad():
        ref = layer.mlp(h) * w
        before = vp_mlp._PREFILL_FALLBACK.get(key, (0, 0))
        out = vp_mlp.fd_conditional_mlp_full_graph(
            layer,
            proj,
            h,
            w,
            run_mask,
            valid_rows=None,
            compact_phase_enabled=True,
            force_dense_all_run=False,
            prefill_fallback_min_project=0.22,
        )
    assert vp_mlp._PREFILL_FALLBACK[key] == (before[0] + 1, before[1] + 1)
    assert torch.isfinite(out.float()).all(), "empty PROJECT expert must not produce NaN/inf"
    assert torch.equal(out, ref), "all-RUN pass must be exactly w*MLP"


@cuda
def test_high_project_share_does_not_fall_back(monkeypatch):
    import sglang.srt.server_args as sa
    from sglang.srt.vpipe import mlp as vp_mlp

    monkeypatch.setattr(sa, "get_global_server_args", lambda: _SA())
    # PROJECT share 0.5 >= 0.22: the dispatcher must continue past the fallback (binary_cohort on layer 24)
    monkeypatch.setenv("SGLANG_FD_FULL_GRAPH_LAYER_POLICIES", "24:binary_cohort")
    layer, proj, h, w, run_mask = _fixture(771, 0.5, 7)
    key = str(h.device)
    with torch.no_grad():
        cohort = vp_mlp._binary_cohort_mlp(layer, proj, h, w, run_mask, None).clone()
        before = vp_mlp._PREFILL_FALLBACK.get(key, (0, 0))
        # body selection is asserted through the binary_cohort evidence counter
        # (executor calls), not only through output equality: the bodies agree
        # on outputs by design, so equality alone cannot prove which one ran.
        calls_before = int(vp_mlp._BINARY_COHORT_STATS[h.device][0].item())
        out = vp_mlp.fd_conditional_mlp_full_graph(
            layer,
            proj,
            h,
            w,
            run_mask,
            valid_rows=None,
            compact_phase_enabled=True,
            force_dense_all_run=False,
            prefill_fallback_min_project=0.22,
        )
    after = vp_mlp._PREFILL_FALLBACK[key]
    assert after == (before[0] + 1, before[1]), "checked but not fallen"
    assert torch.equal(out, cohort), "above the threshold the configured body runs unchanged"
    calls_after = int(vp_mlp._BINARY_COHORT_STATS[h.device][0].item())
    assert calls_after == calls_before + 1, "the binary_cohort body must have executed once"


@cuda
def test_unset_threshold_leaves_counters_untouched(monkeypatch):
    import sglang.srt.server_args as sa
    from sglang.srt.vpipe import mlp as vp_mlp

    monkeypatch.setattr(sa, "get_global_server_args", lambda: _SA())
    monkeypatch.setenv("SGLANG_FD_FULL_GRAPH_LAYER_POLICIES", "24:binary_cohort")
    layer, proj, h, w, run_mask = _fixture(771, 0.92, 3)
    key = str(h.device)
    before = dict(vp_mlp._PREFILL_FALLBACK)
    with torch.no_grad():
        vp_mlp.fd_conditional_mlp_full_graph(
            layer,
            proj,
            h,
            w,
            run_mask,
            valid_rows=None,
            compact_phase_enabled=True,
            force_dense_all_run=False,
            prefill_fallback_min_project=None,
        )
    assert vp_mlp._PREFILL_FALLBACK.get(key) == before.get(key)
