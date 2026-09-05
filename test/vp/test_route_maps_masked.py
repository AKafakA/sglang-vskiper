"""Identity test for Block 1B-1: build_route_maps_from_mask == build_route_maps + mask algebra + stats.

Run on a CUDA host (the kernel is Triton):  python -m pytest test/vp/test_route_maps_masked.py -q
The claim is IDENTITY, not tolerance: same row sets per branch, same counts, same stats deltas,
for every (rows, valid_rows) shape the cohort body sees, including the padded-tail case.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")


def _reference(run_mask, valid_rows, stats):
    from sglang.srt.vpipe.routing import build_route_maps

    run_active = run_mask.clone()
    project_active = ~run_mask
    if valid_rows is not None:
        run_active = run_active & valid_rows
        project_active = project_active & valid_rows
    run_rows, project_rows, counts = build_route_maps(run_active, project_active)
    stats[0].add_(1)
    stats[1:3].add_(counts.to(torch.int64))
    return run_rows, project_rows, counts


def _as_sets(run_rows, project_rows, counts):
    n_run, n_project = int(counts[0].item()), int(counts[1].item())
    return set(run_rows[:n_run].tolist()), set(project_rows[:n_project].tolist())


@pytest.mark.parametrize("rows", [1, 7, 256, 257, 1000, 4096])
@pytest.mark.parametrize("with_valid", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_masked_route_maps_identical(rows, with_valid, seed):
    from sglang.srt.vpipe.routing import build_route_maps_from_mask

    gen = torch.Generator(device="cuda").manual_seed(seed)
    run_mask = torch.rand(rows, device="cuda", generator=gen) < 0.5
    valid_rows = None
    if with_valid:
        valid_rows = torch.ones(rows, dtype=torch.bool, device="cuda")
        # a padded tail like the graph buckets produce, plus a few interior holes
        valid_rows[max(rows - rows // 5, 0) :] = False
        valid_rows[torch.rand(rows, device="cuda", generator=gen) < 0.05] = False

    stats_ref = torch.zeros(3, dtype=torch.int64, device="cuda")
    stats_new = torch.zeros(3, dtype=torch.int64, device="cuda")
    ref = _reference(run_mask, valid_rows, stats_ref)
    new = build_route_maps_from_mask(run_mask, valid_rows, stats_new)

    assert torch.equal(ref[2], new[2]), "counts differ"
    assert _as_sets(*ref) == _as_sets(*new), "row sets differ"
    assert torch.equal(stats_ref, stats_new), "evidence stats differ"
    # every valid row is in exactly one map, invalid rows in none
    run_set, project_set = _as_sets(*new)
    assert not (run_set & project_set)
    valid = valid_rows if valid_rows is not None else torch.ones(rows, dtype=torch.bool, device="cuda")
    assert len(run_set) + len(project_set) == int(valid.sum().item())


def test_calls_counter_increments_once_per_call():
    from sglang.srt.vpipe.routing import build_route_maps_from_mask

    stats = torch.zeros(3, dtype=torch.int64, device="cuda")
    run_mask = torch.zeros(4096, dtype=torch.bool, device="cuda")  # 16 programs of 256 rows
    run_mask[::2] = True
    for _ in range(5):
        build_route_maps_from_mask(run_mask, None, stats)
    assert stats.tolist() == [5, 5 * 2048, 5 * 2048]


@pytest.mark.parametrize("rows", [300, 4096])
@pytest.mark.parametrize("pattern", ["all_run", "all_project", "all_invalid"])
def test_degenerate_multi_block_patterns(rows, pattern):
    from sglang.srt.vpipe.routing import build_route_maps_from_mask

    run_mask = torch.full((rows,), pattern == "all_run", dtype=torch.bool, device="cuda")
    valid_rows = torch.full((rows,), pattern != "all_invalid", dtype=torch.bool, device="cuda")
    stats_ref = torch.zeros(3, dtype=torch.int64, device="cuda")
    stats_new = torch.zeros(3, dtype=torch.int64, device="cuda")
    ref = _reference(run_mask, valid_rows, stats_ref)
    new = build_route_maps_from_mask(run_mask, valid_rows, stats_new)
    assert torch.equal(ref[2], new[2]) and _as_sets(*ref) == _as_sets(*new)
    assert torch.equal(stats_ref, stats_new)


def test_zero_rows_counts_one_call():
    from sglang.srt.vpipe.routing import build_route_maps_from_mask

    stats = torch.zeros(3, dtype=torch.int64, device="cuda")
    run_rows, project_rows, counts = build_route_maps_from_mask(
        torch.zeros(0, dtype=torch.bool, device="cuda"), None, stats
    )
    assert run_rows.numel() == 0 and project_rows.numel() == 0
    assert counts.tolist() == [0, 0] and stats.tolist() == [1, 0, 0]


def test_rejects_strided_inputs():
    from sglang.srt.vpipe.routing import build_route_maps_from_mask

    stats = torch.zeros(3, dtype=torch.int64, device="cuda")
    base = torch.zeros(64, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError):
        build_route_maps_from_mask(base[::2], None, stats)
    with pytest.raises(ValueError):
        build_route_maps_from_mask(base[:32], base[::2], stats)
    with pytest.raises(ValueError):
        build_route_maps_from_mask(base[:32], None, torch.zeros(6, dtype=torch.int64, device="cuda")[::2])


def test_rejects_wrong_dtypes():
    from sglang.srt.vpipe.routing import build_route_maps_from_mask

    stats = torch.zeros(3, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError):
        build_route_maps_from_mask(torch.zeros(8, dtype=torch.int32, device="cuda"), None, stats)
    with pytest.raises(ValueError):
        build_route_maps_from_mask(
            torch.zeros(8, dtype=torch.bool, device="cuda"), None, torch.zeros(3, dtype=torch.int32, device="cuda")
        )
