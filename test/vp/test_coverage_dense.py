"""CPU gates for the (c3) coverage-aware body-selection mechanism (SS5.1-5.8).

No GPU: exercises the pure decision mirror, the stamp/reset helpers, the
``flexidepth_phase_enabled`` coverage clause (including the W1-OFF hole the
standalone clause exists to close), the self-sized ladder derivation +
bounded-descent primitives, the two-site parity falsifiability, the site-B
dedup, and the E-C3 production-parity properties.

Design of record:
``vPipe-doc/codex/asplos-plan/2026-08-04-c3-coverage-aware-admission-design.md``.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.vpipe import coverage as coverage_dense
from sglang.srt.vpipe import attestation as vpipe_attestation
from sglang.srt.vpipe.attestation import (
    per_boot_ladder_hash,
)
from sglang.srt.vpipe.coverage import (
    BODY_DENSE_EAGER,
    BODY_EAGER_SKIP,
    BODY_GRAPH_ALLRUN,
    BODY_GRAPH_SKIP,
    COVERAGE_REASON_INELIGIBLE,
    COVERAGE_REASON_NO_RUNNER,
    COVERAGE_REASON_ROWS,
    cdopt_skip_only_bucket_legal,
    coverage_dense_counters,
    coverage_parity_ok,
    record_dense_body_pass,
    record_eager_skip_decode_layer_call,
    record_recapture_event,
    reset_coverage_dense_state,
    reset_coverage_stamps,
    select_decode_body,
    stamp_coverage_dense,
    vp_graph_lifecycle_active,
    vp_graph_lifecycle_mark,
)



@pytest.fixture(autouse=True)
def _fresh_state():
    reset_coverage_dense_state()
    yield
    reset_coverage_dense_state()


def _mode(*, decode: bool) -> SimpleNamespace:
    return SimpleNamespace(is_decode=lambda: decode)


def _batch(*, decode: bool = True, batch_size: int = 8) -> SimpleNamespace:
    # Mirrors the declared ForwardBatch defaults for every (c3) field.
    return SimpleNamespace(
        forward_mode=_mode(decode=decode),
        batch_size=batch_size,
        vp_fd_decode_coverage_dense=False,
        vp_fd_coverage_counted=False,
        fd_full_graph_force_production_attention=False,
    )


def _runner(reason=None) -> SimpleNamespace:
    return SimpleNamespace(_last_reject_reason=reason)


# --- SS5.1: pure decision function, exhaustive property ----------------------


def test_select_decode_body_never_eager_skip_and_fail_closed() -> None:
    rows_values = (0, 1, 175, 176, 256, 257, 375, 900, 4096, 4097)
    max_bs_values = (None, 16, 256, 1024, 4096)
    for rows, max_bs, oracle_ok, fd_enabled, band_state, w1 in itertools.product(
        rows_values,
        max_bs_values,
        (False, True),
        (False, True),
        (None, "prod_allrun", "skip"),
        (False, True),
    ):
        body = select_decode_body(
            rows, max_bs, oracle_ok, fd_enabled, band_state, w1
        )
        # R-A totality: no input maps to the eager skip body.
        assert body != BODY_EAGER_SKIP
        # R-E: oracle unavailable resolves to dense, always.
        if not oracle_ok:
            assert body == BODY_DENSE_EAGER
        if max_bs is None:
            assert body == BODY_DENSE_EAGER
        if oracle_ok and max_bs is not None and rows > max_bs:
            assert body == BODY_DENSE_EAGER




def test_select_decode_body_composition_priority() -> None:
    # Covered + W1 low band -> captured production all-RUN body.
    assert (
        select_decode_body(100, 1024, True, True, "prod_allrun", True)
        == BODY_GRAPH_ALLRUN
    )
    # Covered + W1 high band -> captured skip body.
    assert select_decode_body(300, 1024, True, True, "skip", True) == BODY_GRAPH_SKIP
    # Covered, W1 off -> the single FD body (skip), band state ignored.
    assert (
        select_decode_body(300, 1024, True, True, "prod_allrun", False)
        == BODY_GRAPH_SKIP
    )
    # Coverage beats band: uncovered + held-high band -> dense (never eager skip).
    assert (
        select_decode_body(2000, 1024, True, True, "skip", True) == BODY_DENSE_EAGER
    )
    # Production (no FD) covered -> its own captured all-RUN graph.
    assert select_decode_body(100, 256, True, False, None, False) == BODY_GRAPH_ALLRUN


# --- SS5.2: stamp set/reset + reasons ----------------------------------------


def test_stamp_sets_both_stamps_and_reset_clears() -> None:
    fb = _batch(batch_size=300)
    stamped = stamp_coverage_dense(
        fb, runner=_runner(COVERAGE_REASON_ROWS), can_run_graph=False, w1_active=False
    )
    assert stamped is True
    assert fb.vp_fd_decode_coverage_dense is True
    assert fb.fd_full_graph_force_production_attention is True
    counters = coverage_dense_counters()
    assert counters.dense_overflow_decisions == 1
    assert counters.dense_overflow_rows == 300
    assert counters.overflow_reason[COVERAGE_REASON_ROWS] == 1
    reset_coverage_stamps(fb)
    assert fb.vp_fd_decode_coverage_dense is False
    assert fb.vp_fd_coverage_counted is False
    assert fb.fd_full_graph_force_production_attention is False


def test_covered_pass_never_stamps() -> None:
    fb = _batch(batch_size=64)
    stamped = stamp_coverage_dense(
        fb, runner=_runner(None), can_run_graph=True, w1_active=False
    )
    assert stamped is False
    assert fb.vp_fd_decode_coverage_dense is False
    assert fb.fd_full_graph_force_production_attention is False
    counters = coverage_dense_counters()
    assert counters.dense_overflow_decisions == 0
    assert counters.dispatch_graph_steps == 1


def test_stamp_reason_channel() -> None:
    counters = coverage_dense_counters()
    # Runner absent -> fail-closed dense with no_graph_runner (R-E).
    fb = _batch(batch_size=4)
    assert stamp_coverage_dense(fb, runner=None, can_run_graph=False, w1_active=False)
    assert counters.overflow_reason[COVERAGE_REASON_NO_RUNNER] == 1
    # Eligibility veto recorded by the predicate.
    fb = _batch(batch_size=4)
    assert stamp_coverage_dense(
        fb,
        runner=_runner(COVERAGE_REASON_INELIGIBLE),
        can_run_graph=False,
        w1_active=False,
    )
    assert counters.overflow_reason[COVERAGE_REASON_INELIGIBLE] == 1
    # A stale/None reason from the runner falls back to not_graph_eligible.
    fb = _batch(batch_size=4)
    assert stamp_coverage_dense(
        fb, runner=_runner(None), can_run_graph=False, w1_active=False
    )
    assert counters.overflow_reason[COVERAGE_REASON_INELIGIBLE] == 2


def test_stamp_w1_composition_counter() -> None:
    fb = _batch(batch_size=4)
    stamp_coverage_dense(
        fb, runner=_runner(COVERAGE_REASON_ROWS), can_run_graph=False, w1_active=True
    )
    assert coverage_dense_counters().coverage_stamp_w1_composed == 1


# --- SS5.3: flexidepth_phase_enabled truth table -----------------------------


REGIME_JSON = json.dumps(
    {
        "version": 1,
        "prefill": {
            "enabled": True,
            "min_tokens": 448,
            "row_correction_alpha": 0.0,
            "include_mixed": True,
        },
        "decode": {
            "enabled": True,
            "enter_rows": 176,
            "exit_rows": 144,
            "low_body": "prod_allrun",
            "high_body": "skip",
        },
    }
)


def _phase_fb(*, decode: bool, coverage: bool, w1_decode_dense: bool = False):
    mode = SimpleNamespace(
        is_decode=lambda: decode,
        is_extend=lambda: not decode,
    )
    fb = SimpleNamespace(forward_mode=mode)
    if decode:
        fb.vp_fd_decode_coverage_dense = coverage
        fb.vp_fd_decode_dense = w1_decode_dense
    else:
        fb.vp_fd_prefill_dense = False
    return fb


@pytest.mark.parametrize("w1_on", [False, True])
@pytest.mark.parametrize("stamped", [False, True])
@pytest.mark.parametrize("decode", [False, True])
def test_phase_enabled_truth_table(monkeypatch, w1_on, stamped, decode) -> None:
    from sglang.srt.vpipe.common import (
        flexidepth_phase_enabled,
    )

    monkeypatch.setenv("SGLANG_FD_ACTIVE_PHASES", "both")
    if w1_on:
        monkeypatch.setenv("SGLANG_VP_REGIME_SWITCH", REGIME_JSON)
    else:
        monkeypatch.delenv("SGLANG_VP_REGIME_SWITCH", raising=False)
    fb = _phase_fb(decode=decode, coverage=stamped)
    result = flexidepth_phase_enabled(fb)
    if decode and stamped:
        # The load-bearing case: a stamped decode pass is dense EVEN WITH W1
        # OFF — the standalone clause closes the W1-config-guard inertness
        # hole (flexidepth_full_graph._regime_switch_decode_forced_dense
        # returns False whenever the switch config is None).
        assert result is False
    else:
        # Prefill passes never consult the coverage stamp; unstamped decode
        # passes keep the byte-identical FlexiDepth path.
        assert result is True


# --- SS5.4: ladder derivation + descent --------------------------------------


def _fake_model_runner(*, pool_size: int, tier_bs: list[int], locked: set):
    from sglang.srt.server_args import ServerArgs

    fake_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(bs=list(tier_bs))
        ),
        _cuda_graph_config_locked=set(locked),
        disable_cuda_graph_padding=False,
        speculative_algorithm=None,
        enable_two_batch_overlap=False,
        enable_dp_attention=False,
        disable_attn_tp_gather=True,
        torch_compile_max_bs=None,
    )
    fake_args._generate_decode_cuda_graph_batch_sizes = (
        ServerArgs._generate_decode_cuda_graph_batch_sizes.__get__(fake_args)
    )
    return SimpleNamespace(
        server_args=fake_args,
        req_to_token_pool=SimpleNamespace(size=pool_size),
    )


def _fake_parallel(monkeypatch) -> None:
    # get_batch_sizes_to_capture's stock mul_base alignment reads
    # get_parallel().attn_{tp,cp}_size, whose accessors assert live distributed
    # groups (parallel_state.get_attn_cp_group). Capture always runs after
    # distributed init on a real server, so this is purely a unit-test seam:
    # patch the symbol AS IMPORTED by base_cuda_graph_runner (module-level
    # `from ... import get_parallel`) to a single-rank stand-in. Needed by BOTH
    # armed and unarmed ladder tests.
    from sglang.srt.model_executor.runner import base_cuda_graph_runner as _bcgr

    monkeypatch.setattr(
        _bcgr,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=1, attn_cp_size=1, tp_size=1),
    )


def _arm(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_FD_WEIGHTS", "/dev/shm/fake_router.pt")
    monkeypatch.setenv("SGLANG_FD_EXECUTION_MODE", "full_graph")
    _fake_parallel(monkeypatch)
    reset_coverage_dense_state()


TIER_DEFAULT_256 = [1, 2, 4, 8, 12] + list(range(16, 257, 8))


















def test_per_boot_ladder_hash_stability() -> None:
    ladder = [1, 2, 4, 8]
    assert per_boot_ladder_hash(ladder) == per_boot_ladder_hash(list(ladder))
    assert per_boot_ladder_hash(ladder) != per_boot_ladder_hash([1, 2, 4])


# --- SS5.5: regime composition + C-D-opt corrected legality ------------------


def test_seam_observe_is_level_triggered_idempotent_in_state() -> None:
    from sglang.srt.vpipe.common import (
        regime_switch_config,
    )
    from sglang.srt.vpipe.regime import (
        DecodeRegimeDispatch,
    )

    cfg = regime_switch_config({"SGLANG_VP_REGIME_SWITCH": REGIME_JSON})
    dispatch = DecodeRegimeDispatch(cfg)
    # Enter high, then double-observe the same rows (seam + load_batch): the
    # BAND STATE cannot double-advance (F22 — level-triggered, idempotent in
    # rows). NOTE: the per-body pass COUNTS do accumulate per observe call by
    # construction; covered passes are observed at both the seam and
    # load_batch, so W1's identity-stripped decode counts read ~2x per covered
    # pass on a (c3) tree (recorded in the design's evidence semantics).
    assert dispatch.observe(300) == "skip"
    assert dispatch.observe(300) == "skip"
    # Eager episode (uncovered): band stays live through eager-only observes.
    assert dispatch.observe(150) == "skip"  # inside the band -> hold
    assert dispatch.observe(144) == "prod_allrun"  # exit edge
    assert dispatch.observe(176) == "skip"  # re-enter edge
    # Coverage boundary >> enter_rows: the first covered pass after an eager
    # episode re-enters high immediately (D1 benign-lag argument).
    assert dispatch.observe(1024) == "skip"


def test_decode_regime_kv_criterion_keys_on_resident_tokens() -> None:
    """Lane-2 cut1: with enter/exit_kv_tokens set the band ignores rows and keys
    on the batch's resident KV tokens (rows x context); 0/0 keeps the rows band."""
    import json

    import pytest

    from sglang.srt.vpipe.common import regime_switch_config
    from sglang.srt.vpipe.regime import DecodeRegimeDispatch

    base = json.loads(REGIME_JSON)
    base["decode"].update(
        {"enabled": True, "enter_kv_tokens": 200_000, "exit_kv_tokens": 160_000}
    )
    cfg = regime_switch_config({"SGLANG_VP_REGIME_SWITCH": json.dumps(base)})
    assert cfg.decode.kv_criterion
    dispatch = DecodeRegimeDispatch(cfg)
    # 512 rows x 256 tokens = 131k KV tokens: below the band -> low body even
    # though rows >> enter_rows; 256 rows x 1k = 262k -> high.
    assert dispatch.observe(512, 131_072) == "prod_allrun"
    assert dispatch.observe(256, 262_144) == "skip"
    assert dispatch.observe(64, 180_000) == "skip"  # inside the band -> hold
    assert dispatch.observe(64, 160_000) == "prod_allrun"  # exit edge
    assert dispatch.observe(8, 200_000) == "skip"  # re-enter edge, rows irrelevant
    with pytest.raises(ValueError):
        dispatch.observe(300)  # kv criterion needs seq_lens_sum
    # Rows criterion (0/0) still ignores kv_tokens.
    rows_cfg = regime_switch_config({"SGLANG_VP_REGIME_SWITCH": REGIME_JSON})
    if rows_cfg.decode.enabled:
        rows_dispatch = DecodeRegimeDispatch(rows_cfg)
        assert rows_dispatch.observe(300, 1) == "skip"
    bad = json.loads(REGIME_JSON)
    bad["decode"].update({"enabled": True, "enter_kv_tokens": 100, "exit_kv_tokens": 0})
    with pytest.raises(ValueError):
        regime_switch_config({"SGLANG_VP_REGIME_SWITCH": json.dumps(bad)})


def test_cdopt_corrected_legality_padded_bucket_premise() -> None:
    ladder = [8, 16, 168, 176, 256]
    # Bucket 176 maps raw rows 169..176; raw 169-175 under a held-low band
    # would compose (176, prod_allrun) -> a skip-only 176 graph set is ILLEGAL
    # with enter_rows=176 (the merged draft's raw-rows premise was wrong, F6).
    assert not cdopt_skip_only_bucket_legal(ladder, 176, enter_rows=176)
    # Bucket 256 maps raw rows 177..256, all >= enter_rows -> legal.
    assert cdopt_skip_only_bucket_legal(ladder, 256, enter_rows=176)
    with pytest.raises(ValueError):
        cdopt_skip_only_bucket_legal(ladder, 300, enter_rows=176)


# --- SS5.6: counter falsifiability -------------------------------------------


def test_two_site_parity_falsifiable() -> None:
    fb = _batch(batch_size=300)
    stamp_coverage_dense(
        fb, runner=_runner(COVERAGE_REASON_ROWS), can_run_graph=False, w1_active=False
    )
    # Site B missing -> parity FAILS (site-A-only evidence is rejected).
    assert not coverage_parity_ok()
    # Site B executes (first routed layer of the stamped pass) -> parity holds.
    record_dense_body_pass(fb)
    assert coverage_parity_ok()
    counters = coverage_dense_counters()
    assert counters.dense_body_passes == counters.dense_overflow_decisions == 1
    assert counters.dense_body_rows == counters.dense_overflow_rows == 300
    assert counters.fd_tokens_dense_overflow == 300
    # Site A missing (body ran without a seam decision) -> parity FAILS.
    reset_coverage_dense_state()
    fb = _batch(batch_size=8)
    fb.vp_fd_decode_coverage_dense = True  # stamped out-of-band
    record_dense_body_pass(fb)
    assert not coverage_parity_ok()


def test_sentinel_can_fire_and_lifecycle_excludes() -> None:
    counters = coverage_dense_counters()
    fb = _batch(decode=True)
    # Planted synthetic eager-skip execution: the gate CAN fire.
    record_eager_skip_decode_layer_call(fb)
    assert counters.eager_skip_decode_layer_calls == 1
    # Planted warmup/capture-context execution: the F1 exclusion holds.
    with vp_graph_lifecycle_mark():
        assert vp_graph_lifecycle_active()
        record_eager_skip_decode_layer_call(fb)
    assert not vp_graph_lifecycle_active()
    assert counters.eager_skip_decode_layer_calls == 1
    assert counters.graph_lifecycle_sections == 1
    # Nested lifecycle sections (recapture -> capture -> warmup) stay marked.
    with vp_graph_lifecycle_mark():
        with vp_graph_lifecycle_mark():
            record_eager_skip_decode_layer_call(fb)
        assert vp_graph_lifecycle_active()
    assert counters.eager_skip_decode_layer_calls == 1
    # Prefill entries never count.
    record_eager_skip_decode_layer_call(_batch(decode=False))
    assert counters.eager_skip_decode_layer_calls == 1
    assert counters.eager_skip_decode_layer_calls == 1


def test_recapture_events_are_visible_evidence() -> None:
    record_recapture_event(41)
    record_recapture_event(1077)
    assert coverage_dense_counters().recapture_events == [41, 1077]


# --- SS5.7: site-B dedup ------------------------------------------------------


def test_site_b_dedup_multi_layer_traversal() -> None:
    fb = _batch(batch_size=64)
    stamp_coverage_dense(
        fb, runner=_runner(COVERAGE_REASON_ROWS), can_run_graph=False, w1_active=False
    )
    # 16 routed layers (16..31) all reach the dense fall-through of one
    # stamped pass; the witness increments exactly once.
    for _ in range(16):
        record_dense_body_pass(fb)
    counters = coverage_dense_counters()
    assert counters.dense_body_passes == 1
    assert counters.dense_body_rows == 64
    # The next stamped pass counts again after the try/finally reset.
    reset_coverage_stamps(fb)
    stamp_coverage_dense(
        fb, runner=_runner(COVERAGE_REASON_ROWS), can_run_graph=False, w1_active=False
    )
    for _ in range(16):
        record_dense_body_pass(fb)
    assert coverage_dense_counters().dense_body_passes == 2
    # Unstamped passes (production semantics, W1 band-dense, prefill) never
    # touch the witness.
    reset_coverage_stamps(fb)
    record_dense_body_pass(fb)
    assert coverage_dense_counters().dense_body_passes == 2


# --- SS5.8: E-C3 production parity -------------------------------------------


def test_forward_batch_declares_c3_fields_default_false() -> None:
    import dataclasses

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    fields = {field.name: field for field in dataclasses.fields(ForwardBatch)}
    for name in (
        "vp_fd_decode_coverage_dense",
        "vp_fd_coverage_counted",
        "fd_full_graph_force_production_attention",
    ):
        assert name in fields, name
        assert fields[name].default is False, name


def test_kill_switch_env_strict_parse_and_default_on() -> None:
    assert envs.SGLANG_VP_COVERAGE_DENSE.get() is True  # default ON
    with envs.SGLANG_VP_COVERAGE_DENSE.override(False):
        assert coverage_dense.coverage_dense_enabled() is False
    os.environ["SGLANG_VP_COVERAGE_DENSE"] = "definitely"
    try:
        with pytest.raises(ValueError):
            envs.SGLANG_VP_COVERAGE_DENSE.get()
    finally:
        del os.environ["SGLANG_VP_COVERAGE_DENSE"]


def test_production_consults_zero_new_state(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_FD_WEIGHTS", raising=False)
    monkeypatch.delenv("SGLANG_FD_EXECUTION_MODE", raising=False)
    reset_coverage_dense_state()
    assert coverage_dense.fd_skip_decode_deployed() is False
    assert coverage_dense.coverage_dense_armed() is False
    # No FD -> the attestation block is ABSENT, keeping the production
    # /server_info byte-identical to stock.
    model_runner = SimpleNamespace()
    assert (
        vpipe_attestation.coverage_dense_runtime_attestation(model_runner) is None
    )




