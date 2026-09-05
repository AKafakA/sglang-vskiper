"""P3 — binary_cohort branch GEMMs through cuBLAS on eager prefill passes (SGLANG_FD_FULL_GRAPH_PREFILL_CUBLAS=1).

GPU test: the cuBLAS branch must agree with the count-GEMM (Triton) branch and with the full-dual reference to fp16
rounding, on the same routes, for row counts across the tuned bands (including rows that leave both branches non-empty),
and it must record executed passes in the attestation. CPU-only environments skip.
"""

import os

import pytest
import torch

os.environ.setdefault("SGLANG_FD_FULL_GRAPH_COMPACT", "1")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuBLAS/Triton branch needs CUDA")


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
        self.I = I

    def forward(self, x):
        gu = torch.nn.functional.linear(x, self.gate_up_proj.weight)
        return torch.nn.functional.linear(self.act_fn(gu), self.down_proj.weight)


class _Layer:
    def __init__(self, mlp):
        self.mlp = mlp
        self.layer_id = 24


@cuda
@pytest.mark.parametrize("rows,run_frac", [(64, 0.5), (256, 0.11), (771, 0.55), (2048, 0.5)])
def test_cublas_branch_matches_triton_and_reference(monkeypatch, rows, run_frac):
    import sglang.srt.server_args as sa
    from sglang.srt.vpipe import mlp as vp_mlp
    from sglang.srt.vpipe.routing import FDProj

    monkeypatch.setattr(sa, "get_global_server_args", lambda: _SA())
    dev = torch.device("cuda")
    dtype = torch.float16
    H, I = 4096, 14336
    layer = _Layer(_StubMLP(H, I, dtype, dev))
    proj = FDProj(hidden_size=H, intermediate_size=I).to(device=dev, dtype=dtype)
    g = torch.Generator(device=dev).manual_seed(rows)
    h = (torch.randn(rows, H, device=dev, generator=g) * 0.5).to(dtype)
    w = torch.rand(rows, 1, device=dev, generator=g).to(dtype)
    run_mask = torch.rand(rows, 1, device=dev, generator=g) < run_frac
    with torch.no_grad():
        ref = vp_mlp._full_dual_mlp(layer, proj, h, w, run_mask)
        before = dict(vp_mlp._BINARY_COHORT_CUBLAS_PASSES)
        triton = vp_mlp._binary_cohort_mlp(layer, proj, h, w, run_mask, None).clone()
        assert dict(vp_mlp._BINARY_COHORT_CUBLAS_PASSES) == before, "Triton path must not count cuBLAS passes"
        cublas = vp_mlp._binary_cohort_mlp(layer, proj, h, w, run_mask, None, prefill_cublas=True).clone()
    after = vp_mlp._BINARY_COHORT_CUBLAS_PASSES[str(dev)]
    assert after[0] == before.get(str(dev), (0, 0))[0] + 1 and after[1] == before.get(str(dev), (0, 0))[1] + rows
    scale = ref.float().abs().max().item()
    tol = 4e-3 * scale + 1e-3  # a few fp16 ulps at the output magnitude (accumulation order differs per GEMM)
    assert (cublas.float() - ref.float()).abs().max().item() <= tol
    assert (cublas.float() - triton.float()).abs().max().item() <= tol
    # every row written exactly once: rows with |ref| > 0 are non-zero in cublas too
    assert torch.equal((cublas != 0).any(-1), (ref != 0).any(-1))


def test_env_reader_and_validation_shape():
    from sglang.srt.vpipe.common import full_graph_prefill_cublas_enabled

    assert full_graph_prefill_cublas_enabled({}) is False
    assert full_graph_prefill_cublas_enabled({"SGLANG_FD_FULL_GRAPH_PREFILL_CUBLAS": "1"}) is True
    with pytest.raises(ValueError):
        full_graph_prefill_cublas_enabled({"SGLANG_FD_FULL_GRAPH_PREFILL_CUBLAS": "yes"})
