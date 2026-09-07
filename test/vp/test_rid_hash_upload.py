"""Lane-2 tax-removal track, runtime tax R1 (2026-09-07): the per-batch request-identity hashes must reach the device
WITHOUT a host-blocking copy. The served profile showed `ForwardBatch.init_new -> _hash_rids_to_tensor` stalling the
forward thread for a whole preceding forward on EVERY batch (pageable `torch.tensor(..., device=cuda)` = synchronous
cudaMemcpy that drains the stream); production never takes that branch. Contract: identical values/dtype/device to the
reference upload, and the call returns while the stream is still busy."""

import time

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import (
    _bootstrap_rooms_to_tensor,
    _hash_rids_to_tensor,
    _stable_hash_str_to_i64,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(rids, device):
    return torch.tensor([_stable_hash_str_to_i64(r) for r in rids], dtype=torch.int64, device=device)


@pytest.mark.parametrize("n", [1, 7, 64, 256, 1024])
def test_values_identical_to_reference(n):
    rids = [f"req-{i}-{i * 7919 % 1000003}" for i in range(n)]
    dev = torch.device("cuda")
    out = _hash_rids_to_tensor(rids=rids, device=dev)
    ref = _reference(rids, dev)
    torch.cuda.synchronize()
    assert out.dtype == torch.int64 and out.device == ref.device and out.shape == ref.shape
    assert torch.equal(out, ref)


def test_bootstrap_rooms_identical():
    dev = torch.device("cuda")
    rooms = [None, 3, 0, None, 99]
    out = _bootstrap_rooms_to_tensor(bootstrap_rooms=rooms, device=dev)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.tensor([-1, 3, 0, -1, 99], dtype=torch.int64, device=dev))


def test_upload_does_not_drain_the_stream():
    # queue ~hundreds of ms of GPU work, then upload: the call must return long before the queue drains, and the
    # values must still be right once it does (stream ordering).
    dev = torch.device("cuda")
    a = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(60):
        a = a @ a * 1e-4
    rids = [f"busy-{i}" for i in range(512)]
    t0 = time.perf_counter()
    out = _hash_rids_to_tensor(rids=rids, device=dev)
    host_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    drain_ms = (time.perf_counter() - t1) * 1000
    assert torch.equal(out, _reference(rids, dev))
    assert drain_ms > 20, f"the GPU queue drained too fast to test ordering ({drain_ms:.1f} ms)"
    assert host_ms < drain_ms / 4, f"upload blocked the host: {host_ms:.1f} ms with {drain_ms:.1f} ms of queued work"


def test_cpu_device_path_unchanged():
    rids = ["a", "b", "c"]
    out = _hash_rids_to_tensor(rids=rids, device=torch.device("cpu"))
    assert torch.equal(out, torch.tensor([_stable_hash_str_to_i64(r) for r in rids], dtype=torch.int64))
