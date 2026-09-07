"""Exactness gate for the fused route decision (Lane-2 Track B / F1).

Compares, on random inputs, the fused kernel against the unfused sequence it
replaces per routed layer:

    torch.gt(w, thr, out=tape_row)            # RUN mask
    weight_tape_row.copy_(w)                  # branch-weight tape
    build_route_maps_from_mask(mask, valid, stats)   # Block 1B-1 maps + counts + stats
    (run | ~valid).all().to(int32)            # conditional-graph predicate

Mask, tape, counts, stats and predicate must be torch.equal; the maps are
compared as sorted sets (slot order under atomics is unspecified in both).
GPU only (box or A100)."""

import pytest
import torch

from sglang.srt.vpipe.routing import (
    accumulate_route_counts,
    all_run_predicate_from_counts,
    build_route_maps_from_mask,
    route_decide_and_maps,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(w, valid, threshold):
    mask = torch.empty(w.numel(), dtype=torch.bool, device=w.device)
    torch.gt(w, threshold, out=mask)
    stats = torch.zeros(3, dtype=torch.int64, device=w.device)
    run, proj, counts = build_route_maps_from_mask(mask, valid, stats)
    pred = ((mask | ~valid) if valid is not None else mask).all().to(torch.int32)
    return mask, run, proj, counts, stats, pred


@pytest.mark.parametrize("rows", [1, 7, 255, 256, 257, 1000, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_valid", [False, True])
@pytest.mark.parametrize("tape", ["none", "same", "fp32"])
def test_fused_matches_unfused(rows, dtype, with_valid, tape):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1000 * rows + 10 * int(with_valid) + len(tape))
    w = torch.rand(rows, generator=gen, device="cuda", dtype=torch.float32)
    planted = torch.tensor([0.5, 0.5 + 1e-7, 0.5 - 1e-7, 0.0, 1.0], device="cuda")
    w[: min(rows, 5)] = planted[: min(rows, 5)]
    w = w.to(dtype)
    valid = (torch.rand(rows, generator=gen, device="cuda") > 0.3) if with_valid else None

    ref_mask, ref_run, ref_proj, ref_counts, ref_stats, ref_pred = _reference(w, valid, 0.5)

    mask = torch.empty(rows, dtype=torch.bool, device="cuda")
    tape_out = None
    if tape != "none":
        tape_out = torch.empty(rows, dtype=(dtype if tape == "same" else torch.float32), device="cuda")
    run, proj, counts = route_decide_and_maps(w, 0.5, valid, mask, tape_out)
    stats = torch.zeros(3, dtype=torch.int64, device="cuda")
    accumulate_route_counts(counts, stats)
    pred = torch.empty((), dtype=torch.int32, device="cuda")
    all_run_predicate_from_counts(counts, pred)

    assert torch.equal(mask, ref_mask)
    if tape_out is not None:
        ref_tape = torch.empty_like(tape_out)
        ref_tape.copy_(w)
        assert torch.equal(tape_out, ref_tape)
    assert torch.equal(counts, ref_counts)
    assert torch.equal(stats, ref_stats)
    n_run, n_proj = (int(v) for v in counts.tolist())
    assert torch.equal(torch.sort(run[:n_run]).values, torch.sort(ref_run[:n_run]).values)
    assert torch.equal(torch.sort(proj[:n_proj]).values, torch.sort(ref_proj[:n_proj]).values)
    assert int(pred.item()) == int(ref_pred.item())


def test_all_run_predicate_true_and_false():
    rows = 300
    w = torch.full((rows,), 0.9, device="cuda", dtype=torch.float16)
    valid = torch.ones(rows, dtype=torch.bool, device="cuda")
    valid[-8:] = False
    mask = torch.empty(rows, dtype=torch.bool, device="cuda")
    # all valid rows RUN -> predicate 1 even though padded rows would PROJECT
    w[-8:] = 0.1
    _, _, counts = route_decide_and_maps(w, 0.5, valid, mask, None)
    pred = torch.empty((), dtype=torch.int32, device="cuda")
    all_run_predicate_from_counts(counts, pred)
    assert int(pred.item()) == 1
    # one valid PROJECT row -> predicate 0
    w[3] = 0.1
    _, _, counts = route_decide_and_maps(w, 0.5, valid, mask, None)
    all_run_predicate_from_counts(counts, pred)
    assert int(pred.item()) == 0


def test_empty_rows_keeps_stats_semantics():
    w = torch.empty(0, device="cuda")
    mask = torch.empty(0, dtype=torch.bool, device="cuda")
    _, _, counts = route_decide_and_maps(w, 0.5, None, mask, None)
    stats = torch.zeros(3, dtype=torch.int64, device="cuda")
    accumulate_route_counts(counts, stats)
    ref_stats = torch.zeros(3, dtype=torch.int64, device="cuda")
    build_route_maps_from_mask(mask, None, ref_stats)
    assert torch.equal(stats, ref_stats)
    assert stats.tolist() == [1, 0, 0]


def test_fails_closed_on_strided_or_misaligned():
    w = torch.rand(64, 2, device="cuda")[:, 0]  # strided view
    mask = torch.empty(64, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError):
        route_decide_and_maps(w, 0.5, None, mask, None)
    w = torch.rand(64, device="cuda")
    with pytest.raises(ValueError):
        route_decide_and_maps(w, 0.5, None, torch.empty(63, dtype=torch.bool, device="cuda"), None)
    with pytest.raises(ValueError):
        route_decide_and_maps(w, 0.5, torch.ones(63, dtype=torch.bool, device="cuda"), mask, None)
