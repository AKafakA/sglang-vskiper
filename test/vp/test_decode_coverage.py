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
