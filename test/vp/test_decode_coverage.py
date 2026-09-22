"""P5 coverage-as-code: the decode capture list is extended to the admission cap with the stock bucket spacing."""

import pytest

from sglang.srt.vpipe.common import coverage_capture_bs, vp_decode_coverage_enabled


def _stock_generate(max_bs: int) -> list[int]:
    # mirror of ServerArgs._generate_decode_cuda_graph_batch_sizes (no speculative decoding)
    bs = [1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16)) + list(range(512, max_bs + 1, 32))
    bs = [b for b in bs if b <= max_bs]
    if max_bs not in bs:
        bs.append(max_bs)
    return bs


def test_extends_to_target_with_stock_spacing():
    base = _stock_generate(256)
    out, added = coverage_capture_bs(base, 391, _stock_generate)
    assert out[-1] == 391 and added > 0
    assert all(b in out for b in (272, 288, 304, 320, 336, 352, 368, 384))
    assert out[: len(base)] == base  # existing buckets untouched
    assert out == sorted(set(out))


def test_noop_when_already_covered():
    base = _stock_generate(1024)
    out, added = coverage_capture_bs(base, 391, _stock_generate)
    assert out == base and added == 0


def test_target_exact_bucket():
    base = _stock_generate(256)
    out, added = coverage_capture_bs(base, 512, _stock_generate)
    assert out[-1] == 512 and 512 in out


def test_rejects_bad_target():
    with pytest.raises(ValueError):
        coverage_capture_bs([1, 2, 4], 0, _stock_generate)


def test_env_switch():
    assert vp_decode_coverage_enabled({}) is True
    assert vp_decode_coverage_enabled({"SGLANG_VP_DECODE_COVERAGE": "0"}) is False
    with pytest.raises(ValueError):
        vp_decode_coverage_enabled({"SGLANG_VP_DECODE_COVERAGE": "maybe"})


def test_max_bs_env():
    from sglang.srt.vpipe.common import vp_decode_coverage_max_bs

    assert vp_decode_coverage_max_bs({}) is None
    assert vp_decode_coverage_max_bs({"SGLANG_VP_DECODE_COVERAGE_MAX_BS": "320"}) == 320
    with pytest.raises(ValueError):
        vp_decode_coverage_max_bs({"SGLANG_VP_DECODE_COVERAGE_MAX_BS": "0"})


def test_device_ladder_table():
    """[D-849 add. 42] the RTX A6000's ladder comes from the tree table; every other card keeps the 4x rule."""
    from sglang.srt.vpipe.common import (
        DECODE_COVERAGE_LADDER_BY_DEVICE,
        vp_decode_coverage_device_bound,
    )

    assert DECODE_COVERAGE_LADDER_BY_DEVICE == {"NVIDIA RTX A6000": 256}
    assert vp_decode_coverage_device_bound("NVIDIA RTX A6000") == 256
    assert vp_decode_coverage_device_bound(" NVIDIA RTX A6000 ") == 256
    assert vp_decode_coverage_device_bound("NVIDIA A100-SXM4-80GB") is None
    assert vp_decode_coverage_device_bound("NVIDIA H100 80GB HBM3") is None
    # the declared A6000 same-ladder baseline must name the same number
    import json, pathlib
    decl = json.loads(pathlib.Path(__file__).resolve().parents[2].joinpath("deploy/execution_differences.rtxa6000.json").read_text())
    assert decl["declared"]["upstream_g256"]["decode_ladder_top"] == 256
    assert decl["declared"]["vskipper"]["decode_ladder_top"] == 256
    assert decl["fork_default"]["decode_ladder_top"] == 256
