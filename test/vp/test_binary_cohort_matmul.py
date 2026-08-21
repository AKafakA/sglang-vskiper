#!/usr/bin/env python3
"""GPU unit test for the count-bounded persistent matmul (design v2).

Exact counts around every tile boundary (review finding 12), fp16 and
bf16, prefix correctness vs torch matmul, and rows >= count proven
UNTOUCHED via a sentinel fill.
"""

import sys

import torch

from sglang.srt.vpipe.kernel import count_matmul_gridexit

SENTINEL = 7777.0


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 1
    torch.manual_seed(20260820)
    device = torch.device("cuda")
    cap, k_dim, n_dim = 256, 512, 192
    failures = 0
    for dtype, rtol, atol in (
        (torch.float16, 2e-2, 3e-2),
        (torch.bfloat16, 2e-2, 3e-2),
    ):
        a = (torch.randn(cap, k_dim, device=device) * 0.05).to(dtype)
        w = (torch.randn(n_dim, k_dim, device=device) * 0.05).to(dtype)
        for count_value in (0, 1, 63, 64, 65, 127, 128, 200, 255, 256):
            c = torch.full((cap, n_dim), SENTINEL, device=device).to(dtype)
            count = torch.tensor(
                count_value, dtype=torch.int32, device=device
            )
            count_matmul_gridexit(a, w, count, c)
            want = (
                a[:count_value].float() @ w.float().T
            ).to(dtype)
            prefix_ok = (
                count_value == 0
                or torch.allclose(
                    c[:count_value].float(),
                    want.float(),
                    rtol=rtol,
                    atol=atol,
                )
            )
            quantized_sentinel = (
                torch.tensor(SENTINEL, device=device).to(dtype).float()
            )
            tail_ok = bool(
                (c[count_value:].float() == quantized_sentinel).all()
            ) if count_value < cap else True
            ok = prefix_ok and tail_ok
            print(
                f"{dtype} count={count_value} prefix={'OK' if prefix_ok else 'BAD'} "
                f"tail={'OK' if tail_ok else 'BAD'}"
            )
            failures += 0 if ok else 1
    print("ALL_OK" if failures == 0 else f"FAILURES={failures}")
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
