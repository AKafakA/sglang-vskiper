"""Memory contract of `count_matmul_gridexit`: the weight tile is never read past N (D-747).

An out-of-bounds READ is numerically invisible (the store mask drops the padded columns) and
silent whenever the over-read lands in mapped memory, which is why the Qwen3-4B fault surfaced
only at the 8192-token capture. This test makes the over-read deterministic without a memory
checker: with PyTorch's expandable segments, a fresh allocation is mapped in whole 2 MiB pages
inside a reserved (unmapped) address range, so a weight placed at the END of an allocation whose
size is a page multiple has unmapped address space right after it, and any read past its end is
an illegal address. A CUDA context does not survive that, so every probe runs in a subprocess.

Three probes: (1) the guard itself — a one-element read past the allocation must fault (proves
the layout; a probe that passed here for the wrong reason would be no evidence); (2) the Qwen3-4B
projector launch (N = 1216, BN = 128, EVEN_K) must not fault; (3) a Llama-3-8B launch (N = 1792,
every tuned BN divides it) must not fault. `--tree <path>` runs the launches against another
tree's `python/` (the unfixed `355980f844` faults on probe 2).
"""
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_PREAMBLE = textwrap.dedent(
    """
    import sys, torch
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    dev = torch.device("cuda:0")
    PAGE = 2 * 1024 * 1024

    def tail_weight(n, k):
        # the weight is the LAST n*k bf16 elements of an allocation that is a whole number of
        # 2 MiB pages; nothing is allocated after it before the launch under test
        elems = n * k
        pages = (elems * 2 + PAGE - 1) // PAGE
        flat = torch.empty(pages * PAGE // 2, dtype=torch.bfloat16, device=dev)
        w = flat[flat.numel() - elems :].view(n, k)
        w.copy_((torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16))
        return flat, w
    """
)

_GUARD = _PREAMBLE + textwrap.dedent(
    """
    import triton, triton.language as tl

    @triton.jit
    def _peek(ptr, off, out_ptr):
        tl.store(out_ptr, tl.load(ptr + off).to(tl.float32))

    out = torch.zeros(1, dtype=torch.float32, device=dev)
    flat, w = tail_weight(1216, 2560)
    _peek[(1,)](w, w.numel() - 1, out); torch.cuda.synchronize()   # last element: fine
    print("GUARD inside ok", flush=True)
    _peek[(1,)](w, w.numel(), out); torch.cuda.synchronize()       # one past the end: must fault
    print("GUARD past-the-end READ SUCCEEDED (layout gives no guard)", flush=True)
    """
)

_LAUNCH = _PREAMBLE + textwrap.dedent(
    """
    from sglang.srt.vpipe.kernel import count_matmul_gridexit
    rows, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    bm, bn, bk, warps, stages = (int(v) for v in sys.argv[4:9])
    torch.manual_seed(0)
    a = (torch.randn(rows, K, device=dev) * 0.5).to(torch.bfloat16)
    idx = torch.arange(rows, dtype=torch.int32, device=dev)
    count = torch.tensor([rows], dtype=torch.int32, device=dev)
    out = torch.zeros(rows, N, dtype=torch.bfloat16, device=dev)
    flat, w = tail_weight(N, K)          # allocated last: the weight ends at the mapped edge
    count_matmul_gridexit(a, w, count, out, gather_index=idx, block_m=bm, block_n=bn,
                          block_k=bk, num_warps=warps, num_stages=stages)
    torch.cuda.synchronize()
    err = (out.float() - a.float() @ w.float().t()).abs().max().item()
    print(f"LAUNCH rows={rows} K={K} N={N} BM/BN/BK={bm}/{bn}/{bk} N%BN={N % bn} max_abs_err={err:.3e}",
          flush=True)
    assert err < 0.05, err
    """
)


def _run(code: str, *argv: str, tree: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if tree:
        env["PYTHONPATH"] = os.path.join(tree, "python") + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
    return subprocess.run(
        [sys.executable, "-c", code, *argv], capture_output=True, text=True, env=env, timeout=600
    )


def _tree() -> str | None:
    return os.environ.get("VP_COUNT_GEMM_GUARD_TREE") or None


def test_guard_layout_faults_one_past_the_end():
    proc = _run(_GUARD)
    assert "GUARD inside ok" in proc.stdout, proc.stdout + proc.stderr
    assert proc.returncode != 0 and "SUCCEEDED" not in proc.stdout, (
        "the layout gave no guard page; the launch probes below would prove nothing\n"
        + proc.stdout + proc.stderr
    )
    assert "illegal" in (proc.stdout + proc.stderr).lower(), proc.stdout + proc.stderr


@pytest.mark.parametrize(
    "shape",
    [
        # Qwen3-4B projector gate/down: N = 2 * 608 is not a multiple of BN = 128 (projgd@205 and
        # projgd@8192 tiles on NVIDIA_A100); EVEN_K (2560 % 64 == 0) -> the path that read past N
        ("qwen3_4b projgd@205", 205, 2560, 1216, (32, 128, 64, 4, 3)),
        ("qwen3_4b projgd@8192", 8192, 2560, 1216, (64, 128, 32, 4, 3)),
        # Qwen3-4B projector up: K = 608 is not a multiple of BK = 64 -> the masked path
        ("qwen3_4b projup@205", 205, 608, 2560, (32, 256, 64, 8, 4)),
        # Llama-3-8B projector gate/down: every tuned BN divides N = 1792
        ("llama3_8b projgd@205", 205, 4096, 1792, (32, 128, 64, 4, 3)),
    ],
    ids=lambda s: s[0] if isinstance(s, tuple) else s,
)
def test_launch_never_reads_past_the_weight(shape):
    label, rows, k, n, cfg = shape
    proc = _run(_LAUNCH, str(rows), str(k), str(n), *(str(v) for v in cfg), tree=_tree())
    assert proc.returncode == 0 and "LAUNCH" in proc.stdout, (
        f"{label}: launch faulted or mismatched\n" + proc.stdout + proc.stderr[-2000:]
    )
