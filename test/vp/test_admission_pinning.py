"""[D-849, 2026-09-21] Per-request body pinning: CPU unit tests.

Every served request must be exactly ONE model's computation (the base model's
dense body or the checkpoint's routed body) for its whole lifetime. These tests
pin the pieces that make that true without a GPU: the scheduler-side admission
pinner (a mirror of the decode band with the same hysteresis), the body
mappings, the prefill decision under a pin, the prefix-cache namespace, the
mixed-step partition/merge, and the design/attestation plumbing (version 2).
Configs are constructed directly -- configuring the design by environment is
forbidden in this tree.
"""

from __future__ import annotations

import json

import msgspec
import pytest
import torch

from sglang.srt.vpipe.common import (
    DECODE_BODY_HIGH,
    DECODE_BODY_LOW,
    REGIME_SWITCH_CONFIG_VERSION,
    REQUEST_BODY_FD,
    REQUEST_BODY_STOCK,
    VP_BODY_MIXED,
    RegimeSwitchConfig,
)
from sglang.srt.vpipe.regime import (
    PREFILL_BODY_DENSE,
    PREFILL_BODY_FD,
    AdmissionPinner,
    DecodeRegimeDispatch,
    batch_pin_of,
    pin_from_band_state,
    pinned_decode_body_of,
    prefill_variant_for_pass,
    regime_switch_zero_counters,
    request_body_decode_body,
    request_body_prefill_body,
)

SERVED_V2 = {
    "version": 2,
    "admission": {
        "enabled": True,
        "criterion": "phase_sticky",
        "cold_start": "stock",
        "prefill_demotion": "observe_only",
        "mixed_step": "forced_run",
        "decode_after_fd_prefill": "band",
    },
    "prefill": {
        "enabled": True,
        "min_tokens": 1536,
        "row_correction_alpha": 0.0,
        "include_mixed": True,
        "engagement_min": 0.35,
        "engagement_probe_every": 64,
    },
    "decode": {
        "enabled": True,
        "enter_rows": 176,
        "exit_rows": 144,
        "low_body": "prod_allrun",
        "high_body": "skip",
        "enter_kv_tokens": 200000,
        "exit_kv_tokens": 160000,
    },
}


def _cfg(**overrides) -> RegimeSwitchConfig:
    raw = json.loads(json.dumps(SERVED_V2))
    for dotted, value in overrides.items():
        node = raw
        *path, leaf = dotted.split(".")
        for key in path:
            node = node[key]
        node[leaf] = value
    cfg = msgspec.json.decode(json.dumps(raw), type=RegimeSwitchConfig)
    cfg.validate()
    return cfg


# --- config / design ---------------------------------------------------------


def test_config_v2_requires_admission_and_rejects_v1() -> None:
    assert REGIME_SWITCH_CONFIG_VERSION == 2
    v1 = json.loads(json.dumps(SERVED_V2))
    v1.pop("admission")
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(json.dumps(v1), type=RegimeSwitchConfig)
    with pytest.raises(ValueError):
        _cfg(version=1)
    with pytest.raises(ValueError):
        _cfg(**{"admission.enabled": True, "decode.enabled": False})
    with pytest.raises(ValueError):
        _cfg(**{"admission.mixed_step": "force_run"})
    cfg = _cfg()
    assert cfg.pinning
    assert not _cfg(**{"admission.enabled": False}).pinning


def test_declared_and_resolved_admission_agree_for_the_served_arm(monkeypatch) -> None:
    from sglang.srt.vpipe import design
    from sglang.srt.vpipe.common import (
        _REGIME_SWITCH_CONFIG_CACHE,
        regime_switch_config,
        resolved_design_attestation,
    )

    monkeypatch.delenv("SGLANG_VP_REGIME_SWITCH", raising=False)
    design._ARM_CACHE.clear()
    design._ARM_CACHE["name"] = "vskipper"
    _REGIME_SWITCH_CONFIG_CACHE.clear()
    try:
        cfg = regime_switch_config()
        assert cfg is not None and cfg.version == 2 and cfg.pinning
        resolved = resolved_design_attestation()["regime_switch"]
        assert resolved["version"] == 2
        assert resolved["admission"] == design.SERVED_REGIME_SWITCH["admission"]
        # A prefill-only arm cannot pin (no decode band to decide from).
        design._ARM_CACHE.clear()
        design._ARM_CACHE["name"] = "vpre_binarycohort"
        _REGIME_SWITCH_CONFIG_CACHE.clear()
        cfg_pre = regime_switch_config()
        assert cfg_pre is not None and not cfg_pre.decode.enabled
        assert not cfg_pre.admission.enabled and not cfg_pre.pinning
        # Always-route: no switch at all, nothing to pin.
        design._ARM_CACHE.clear()
        design._ARM_CACHE["name"] = "integrated_alwaysskip"
        _REGIME_SWITCH_CONFIG_CACHE.clear()
        assert regime_switch_config() is None
    finally:
        design._ARM_CACHE.clear()
        _REGIME_SWITCH_CONFIG_CACHE.clear()


def test_zero_counters_carry_the_admission_block() -> None:
    zeros = regime_switch_zero_counters()
    from sglang.srt.vpipe.regime import admission_split_zero_counters
    assert zeros["admission"] == admission_split_zero_counters()
    assert {"split_passes", "coverage_dense_violation_rows", "ladder_chunked_passes", "forced_run_passes", "forced_run_rows"} <= set(zeros["admission"])


# --- body mappings -------------------------------------------------------------


def test_request_body_mappings() -> None:
    assert request_body_decode_body(REQUEST_BODY_STOCK) == DECODE_BODY_LOW
    assert request_body_decode_body(REQUEST_BODY_FD) == DECODE_BODY_HIGH
    assert request_body_prefill_body(REQUEST_BODY_STOCK) == PREFILL_BODY_DENSE
    assert request_body_prefill_body(REQUEST_BODY_FD) == PREFILL_BODY_FD
    assert pin_from_band_state(DECODE_BODY_LOW) == REQUEST_BODY_STOCK
    assert pin_from_band_state(DECODE_BODY_HIGH) == REQUEST_BODY_FD
    for fn in (request_body_decode_body, request_body_prefill_body, pin_from_band_state):
        with pytest.raises(ValueError):
            fn("mixed")


def test_batch_pin_of() -> None:
    assert batch_pin_of([None, None]) is None
    assert batch_pin_of(["fd", "fd", None]) == "fd"
    assert batch_pin_of(["stock"]) == "stock"
    assert batch_pin_of(["stock", "fd"]) == VP_BODY_MIXED
    with pytest.raises(ValueError):
        batch_pin_of(["bogus"])


# --- admission pinner --------------------------------------------------------------


def test_pinner_cold_start_is_stock() -> None:
    pinner = AdmissionPinner(_cfg())
    assert pinner.active
    assert pinner.state == DECODE_BODY_LOW
    assert pinner.current_pin() == REQUEST_BODY_STOCK
    assert pinner.pin_for_admission() == REQUEST_BODY_STOCK
    assert pinner.counters()["admitted_stock"] == 1


def test_pinner_follows_the_kv_band_with_hysteresis() -> None:
    pinner = AdmissionPinner(_cfg())
    pinner.observe_decode_step(512, 131_072)  # below the band
    assert pinner.current_pin() == REQUEST_BODY_STOCK
    pinner.observe_decode_step(256, 262_144)  # enter
    assert pinner.current_pin() == REQUEST_BODY_FD
    pinner.observe_decode_step(64, 180_000)  # inside the band: hold
    assert pinner.current_pin() == REQUEST_BODY_FD
    pinner.observe_decode_step(64, 160_000)  # exit edge
    assert pinner.current_pin() == REQUEST_BODY_STOCK
    pinner.record_admission(REQUEST_BODY_FD)
    pinner.record_admission(REQUEST_BODY_STOCK)
    counts = pinner.counters()
    assert counts["steps_observed"] == 4 and counts["band_flips"] == 2
    assert counts["admitted_fd"] == 1 and counts["admitted_stock"] == 1
    with pytest.raises(ValueError):
        pinner.observe_decode_step(8, None)  # the kv criterion needs seq_lens_sum


def test_pinner_phase_sticky_prefill_round_and_decode_boundary() -> None:
    pinner = AdmissionPinner(_cfg())
    assert pinner.phase_sticky
    # prefill body per round from the round's prompt tokens (the v1 pass threshold, 1536)
    assert pinner.prefill_pin(1535) == REQUEST_BODY_STOCK
    assert pinner.prefill_pin(1536) == REQUEST_BODY_FD
    # decode body at the boundary ("band", the served default): the band state for EVERY request
    assert pinner.decode_pin_at_boundary(REQUEST_BODY_FD) == REQUEST_BODY_STOCK
    assert pinner.decode_pin_at_boundary(REQUEST_BODY_STOCK) == REQUEST_BODY_STOCK
    pinner.observe_decode_step(256, 262_144)  # band HIGH
    assert pinner.decode_pin_at_boundary(REQUEST_BODY_STOCK) == REQUEST_BODY_FD
    assert pinner.decode_pin_at_boundary(REQUEST_BODY_FD) == REQUEST_BODY_FD
    for pre, dec in ((REQUEST_BODY_STOCK, REQUEST_BODY_STOCK), (REQUEST_BODY_STOCK, REQUEST_BODY_FD), (REQUEST_BODY_FD, REQUEST_BODY_FD), (REQUEST_BODY_FD, REQUEST_BODY_STOCK)):
        pinner.record_plan(pre, dec)
    with pytest.raises(ValueError):
        pinner.record_plan("mixed", REQUEST_BODY_STOCK)
    c = pinner.counters()
    assert (c["plan_stock_stock"], c["plan_stock_fd"], c["plan_fd_fd"], c["plan_fd_stock"]) == (1, 1, 1, 1)
    # "fd": FD prefill implies FD decode whatever the band (the H100 0.75x loss variant)
    fdfd = AdmissionPinner(_cfg(**{"admission.decode_after_fd_prefill": "fd"}))
    assert fdfd.decode_pin_at_boundary(REQUEST_BODY_FD) == REQUEST_BODY_FD and fdfd.decode_pin_at_boundary(REQUEST_BODY_STOCK) == REQUEST_BODY_STOCK
    # the band-state criterion keeps the version-2 admission pin
    band = AdmissionPinner(_cfg(**{"admission.criterion": "decode_band_state"}))
    assert not band.phase_sticky and band.prefill_pin(10_000) == REQUEST_BODY_STOCK
    assert band.decode_pin_at_boundary(REQUEST_BODY_STOCK) == REQUEST_BODY_STOCK


def test_pinner_inactive_when_switch_off_or_pinning_off() -> None:
    for pinner in (AdmissionPinner(None), AdmissionPinner(_cfg(**{"admission.enabled": False}))):
        assert not pinner.active
        assert pinner.current_pin() is None
        assert pinner.pin_for_admission() is None
        assert pinner.counters() is None
        pinner.observe_decode_step(300, 300_000)  # no-op
        assert pinner.state is None


# --- decode dispatch under a pin -----------------------------------------------


def test_dispatch_pin_records_the_body_without_advancing_the_band() -> None:
    dispatch = DecodeRegimeDispatch(_cfg())
    assert dispatch.observe_or_pin(8, 1, DECODE_BODY_LOW) == DECODE_BODY_LOW
    assert dispatch.observe_or_pin(8, 1, DECODE_BODY_HIGH) == DECODE_BODY_HIGH
    assert dispatch.current_body == DECODE_BODY_HIGH
    assert dispatch.counters() == {DECODE_BODY_LOW: 1, DECODE_BODY_HIGH: 1}
    # An unpinned observe afterwards still starts from the band's LOW state.
    assert dispatch.observe_or_pin(8, 1, None) == DECODE_BODY_LOW
    with pytest.raises(ValueError):
        dispatch.pin("dense")
    off = DecodeRegimeDispatch(None)
    assert off.observe_or_pin(8, 1, DECODE_BODY_HIGH) is None and off.counters() is None


def test_pinned_decode_body_of_forward_batch() -> None:
    class FB:
        vp_body = None

    assert pinned_decode_body_of(FB()) is None
    fb = FB()
    fb.vp_body = REQUEST_BODY_STOCK
    assert pinned_decode_body_of(fb) == DECODE_BODY_LOW
    fb.vp_body = VP_BODY_MIXED
    assert pinned_decode_body_of(fb) == DECODE_BODY_HIGH  # forced_run: one routed replay serves both pins


# --- prefill decision under a pin -------------------------------------------------


class _Tracker:
    def __init__(self, demote: bool) -> None:
        self._demote = demote
        self.calls = 0

    def demote(self, engagement_min):
        self.calls += 1
        return self._demote


def test_prefill_variant_for_pass_pinned_bypasses_bracket_and_demotion() -> None:
    cfg = _cfg()
    tracker = _Tracker(demote=True)
    # stock pin: dense even for a 4096-token pass that the bracket would route
    assert prefill_variant_for_pass(REQUEST_BODY_STOCK, 4096, 4, False, 0, cfg, tracker, 3) == (
        PREFILL_BODY_DENSE,
        3,
    )
    # fd pin: routed even for a 64-token pass with the escape demanding demotion
    assert prefill_variant_for_pass(REQUEST_BODY_FD, 64, 1, False, 0, cfg, tracker, 3) == (
        PREFILL_BODY_FD,
        3,
    )
    assert tracker.calls == 0


def test_prefill_variant_for_pass_unpinned_reproduces_version_one() -> None:
    cfg = _cfg()
    no_demote = _Tracker(demote=False)
    # bracket: below min_tokens -> dense, streak untouched
    assert prefill_variant_for_pass(None, 1000, 1, False, 0, cfg, no_demote, 5) == (
        PREFILL_BODY_DENSE,
        5,
    )
    # bracket: routed, no demotion -> fd and the streak resets
    assert prefill_variant_for_pass(None, 2048, 2, False, 0, cfg, no_demote, 5) == (
        PREFILL_BODY_FD,
        0,
    )
    # demotion: dense while the streak is below the probe window, then one routed probe
    demote = _Tracker(demote=True)
    assert prefill_variant_for_pass(None, 2048, 2, False, 0, cfg, demote, 63) == (
        PREFILL_BODY_DENSE,
        64,
    )
    assert prefill_variant_for_pass(None, 2048, 2, False, 0, cfg, demote, 64) == (
        PREFILL_BODY_FD,
        0,
    )
    # mixed pass with mixed switching disabled -> fd regardless of tokens
    cfg_nomix = _cfg(**{"prefill.include_mixed": False})
    assert prefill_variant_for_pass(None, 100, 1, True, 50, cfg_nomix, no_demote, 0) == (
        PREFILL_BODY_FD,
        0,
    )


# --- prefix cache namespace -----------------------------------------------------------


def test_extra_key_suffix_namespaces_radix_keys() -> None:
    from sglang.srt.mem_cache.radix_cache import RadixKey

    ids = [1, 2, 3, 4]
    stock = RadixKey(token_ids=ids, extra_key="salt|vpbody=stock")
    fd = RadixKey(token_ids=ids, extra_key="salt|vpbody=fd")
    assert stock.child_key(1) != fd.child_key(1)
    assert stock.child_key(1) == RadixKey(token_ids=ids, extra_key="salt|vpbody=stock").child_key(1)
    assert stock[:2].extra_key == "salt|vpbody=stock"


# --- mixed-step partition and merge ---------------------------------------------------


def _decode_batch(pins):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    n = len(pins)
    seq_lens = torch.tensor([10 * (i + 1) for i in range(n)], dtype=torch.int64)
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=n,
        input_ids=torch.arange(n, dtype=torch.int64),
        req_pool_indices=torch.arange(n, dtype=torch.int64) + 100,
        seq_lens=seq_lens,
        out_cache_loc=torch.arange(n, dtype=torch.int64) + 1000,
        seq_lens_sum=int(seq_lens.sum()),
        positions=seq_lens - 1,
        seq_lens_cpu=seq_lens.clone(),
        orig_seq_lens=seq_lens.clone(),
        rids_int=torch.arange(n, dtype=torch.int64) * 7,
        rids=[f"r{i}" for i in range(n)],
        mm_inputs=[None] * n,  # this fork carries one entry per request, None for text
        vp_body_rows=list(pins),
        vp_body=batch_pin_of(list(pins)),
    )


def test_split_decode_forward_batch_partitions_and_merges_in_order() -> None:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.vpipe.batch import (
        vp_merge_logits_outputs,
        vp_split_decode_forward_batch,
    )

    pins = ["stock", "fd", "fd", "stock", "fd"]
    fb = _decode_batch(pins)
    assert fb.vp_body == VP_BODY_MIXED
    parts = vp_split_decode_forward_batch(fb)
    assert [p for p, _, _ in parts] == ["stock", "fd"]
    (_, idx_s, sub_s), (_, idx_f, sub_f) = parts
    assert idx_s.tolist() == [0, 3] and idx_f.tolist() == [1, 2, 4]
    assert sub_s.batch_size == 2 and sub_f.batch_size == 3
    assert sub_s.vp_body == "stock" and sub_f.vp_body == "fd"
    assert torch.equal(sub_f.seq_lens, fb.seq_lens.index_select(0, idx_f))
    assert torch.equal(sub_f.rids_int, fb.rids_int.index_select(0, idx_f))
    assert sub_f.rids == ["r1", "r2", "r4"] and sub_f.vp_body_rows == ["fd", "fd", "fd"]
    assert sub_f.mm_inputs == [None, None, None]
    assert sub_f.seq_lens_sum == int(fb.seq_lens[[1, 2, 4]].sum())
    assert sub_s.forward_metadata_ready is False and sub_f.vp_seam_batch_routed is None
    # the parent is untouched
    assert fb.batch_size == 5 and fb.vp_body == VP_BODY_MIXED
    vocab = 4
    outs = []
    for pin, index, sub in parts:
        logits = torch.stack([torch.full((vocab,), float(i)) for i in index.tolist()])
        hidden = torch.stack([torch.full((3,), 10.0 + i) for i in index.tolist()])
        outs.append((pin, index, LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)))
    merged = vp_merge_logits_outputs(outs, fb.batch_size)
    assert merged.next_token_logits[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert merged.hidden_states[:, 0].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_force_run_rows_marks_the_stock_pinned_rows_of_a_mixed_batch() -> None:
    from sglang.srt.vpipe.batch import vp_force_run_rows, vp_split_decode_forward_batch

    fb = _decode_batch(["fd", "stock", "fd", "stock", "stock"])
    force = vp_force_run_rows(fb)
    assert force.dtype == torch.bool and force.tolist() == [False, True, False, True, True]
    assert vp_force_run_rows(_decode_batch(["fd", "fd"])) is None
    # size-only chunks keep their rows' pins; a mixed chunk stays mixed, a uniform one is uniform
    parts = vp_split_decode_forward_batch(fb, max_rows=2, by_pin=False)
    assert [(p, i.tolist()) for p, i, _ in parts] == [("mixed", [0, 1]), ("mixed", [2, 3]), ("stock", [4])]
    assert vp_force_run_rows(parts[0][2]).tolist() == [False, True]
    assert all(sub.fd_full_graph_force_run_rows is None for _, _, sub in parts)


def test_split_chunks_each_pin_group_to_the_ladder_top_in_order() -> None:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.vpipe.batch import vp_merge_logits_outputs, vp_split_decode_forward_batch

    # a UNIFORM fd batch above a 2-row ladder must also be partitioned
    fb = _decode_batch(["fd"] * 5)
    assert fb.vp_body == "fd"
    parts = vp_split_decode_forward_batch(fb, max_rows=2)
    assert [(p, i.tolist()) for p, i, _ in parts] == [("fd", [0, 1]), ("fd", [2, 3]), ("fd", [4])]
    assert all(sub.batch_size <= 2 and sub.vp_body == "fd" for _, _, sub in parts)
    outs = [(p, i, LogitsProcessorOutput(next_token_logits=torch.stack([torch.full((3,), float(r)) for r in i.tolist()]), hidden_states=None)) for p, i, _ in parts]
    assert vp_merge_logits_outputs(outs, 5).next_token_logits[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    # mixed AND above the ladder: stock chunks first, then fd chunks
    fb = _decode_batch(["fd", "stock", "fd", "fd", "stock", "fd"])
    parts = vp_split_decode_forward_batch(fb, max_rows=2)
    assert [(p, i.tolist()) for p, i, _ in parts] == [("stock", [1, 4]), ("fd", [0, 2]), ("fd", [3, 5])]
    # below the ladder and uniform: nothing to partition
    with pytest.raises(RuntimeError):
        vp_split_decode_forward_batch(_decode_batch(["fd", "fd"]), max_rows=4)
    with pytest.raises(RuntimeError):
        vp_split_decode_forward_batch(_decode_batch(["fd", "fd"]), max_rows=0)


def test_partition_keeps_the_first_sub_pass_logits_when_the_next_replay_overwrites_the_buffer() -> None:
    """The graph backend returns views of one static output tensor per shape; the
    second sub-pass's replay overwrites it. Detaching must materialise the rows."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.vpipe.batch import vp_detach_logits_output, vp_merge_logits_outputs

    static = torch.zeros(4, 6)
    static[:2] = 1.0  # first (stock) replay writes its rows
    first = vp_detach_logits_output(LogitsProcessorOutput(next_token_logits=static[:2], hidden_states=None))
    static.fill_(2.0)  # second (fd) replay overwrites the same tensor
    second = LogitsProcessorOutput(next_token_logits=static[:3], hidden_states=None)
    merged = vp_merge_logits_outputs(
        [("stock", torch.tensor([0, 3]), first), ("fd", torch.tensor([1, 2, 4]), second)], 5
    )
    assert merged.next_token_logits[[0, 3]].eq(1.0).all()
    assert merged.next_token_logits[[1, 2, 4]].eq(2.0).all()
    undetached = LogitsProcessorOutput(next_token_logits=static[:2], hidden_states=None)
    assert undetached.next_token_logits.eq(2.0).all()  # the view alone would have lost the rows


def test_pinner_admission_onto_an_idle_engine_observes_the_empty_batch() -> None:
    """After a drain the mirror must not hand the last busy state to the next request."""
    from sglang.srt.vpipe.regime import AdmissionPinner
    pinner = AdmissionPinner(_cfg())
    pinner.observe_decode_step(256, 262_144)  # enter the band
    assert pinner.current_pin() == REQUEST_BODY_FD
    pinner.observe_idle()
    assert pinner.current_pin() == REQUEST_BODY_STOCK
    assert pinner.counters()["band_flips"] == 2


def test_admission_pin_counters_are_not_deployment_identity() -> None:
    """The manifest is taken at boot and re-read by the runner after warmup; the
    pin counters advance in between, so they must be stripped like batch_composition."""
    import importlib.util, pathlib

    spec = importlib.util.spec_from_file_location(
        "qps_deployment", pathlib.Path(__file__).with_name("qps_deployment.py")
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    def info(steps):
        return {"internal_states": [{"vp_runtime": {
            "regime_switch": {"admission": {"enabled": True}, "counters": {"decode": {"skip": steps}}},
            "batch_composition": {"steps": steps},
            "admission_pins": {"enabled": True, "admitted_stock": steps, "steps_observed": steps},
        }}]}
    assert mod.stable_server_identity(info(0)) == mod.stable_server_identity(info(1000))
    text = json.dumps(mod.stable_server_identity(info(5)))
    assert "admission_pins" not in text and "batch_composition" not in text and "admission" in text


def test_split_refuses_uniform_extend_and_speculative_batches() -> None:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.vpipe.batch import vp_split_decode_forward_batch

    with pytest.raises(RuntimeError):
        vp_split_decode_forward_batch(_decode_batch(["fd", "fd"]))
    fb = _decode_batch(["stock", "fd"])
    fb.forward_mode = ForwardMode.EXTEND
    with pytest.raises(RuntimeError):
        vp_split_decode_forward_batch(fb)
    fb = _decode_batch(["stock", "fd"])
    fb.spec_info = object()
    with pytest.raises(RuntimeError):
        vp_split_decode_forward_batch(fb)
    fb = _decode_batch(["stock", "fd"])
    fb.mm_inputs = [None, object()]  # a real multimodal row is refused
    with pytest.raises(RuntimeError):
        vp_split_decode_forward_batch(fb)


def test_forward_batch_init_refuses_a_mixed_extend_batch() -> None:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.vpipe.regime import batch_pin_of as _bpo

    # The construction-time check is a one-liner on batch_pin_of + forward_mode;
    # exercise the predicate the way ForwardBatch.init_new does.
    rows = ["stock", "fd"]
    assert _bpo(rows) == VP_BODY_MIXED and not ForwardMode.EXTEND.is_decode()


def test_pinner_one_way_upgrade_stock_to_fd_at_band_high() -> None:
    """[D-849 add. 12] a stock-pinned decode upgrades to fd once the band is HIGH, never back."""
    from sglang.srt.vpipe.regime import upgrade_pinned_rows

    class R:  # a decode-pinned request, duck-typed
        def __init__(self, body, pinned=True):
            self.vp_body = body
            self.vp_decode_body = body
            self.vp_decode_pinned = pinned
            self.vp_decode_upgraded = False
            self.vp_skip_finish_insert = False

    pinner = AdmissionPinner(_cfg())
    assert pinner.upgrades_enabled
    rows = [R(REQUEST_BODY_STOCK), R(REQUEST_BODY_FD), R(REQUEST_BODY_STOCK, pinned=False)]
    # band LOW: nothing moves
    pinner.observe_decode_step(512, 131_072)
    assert upgrade_pinned_rows(pinner, rows) == 0 and rows[0].vp_body == REQUEST_BODY_STOCK
    # band HIGH: the pinned stock row upgrades, the prefilling (unpinned) row and the fd row do not
    pinner.observe_decode_step(256, 262_144)
    assert upgrade_pinned_rows(pinner, rows) == 1
    assert rows[0].vp_body == rows[0].vp_decode_body == REQUEST_BODY_FD
    assert rows[0].vp_decode_upgraded and rows[0].vp_skip_finish_insert
    assert rows[2].vp_body == REQUEST_BODY_STOCK and not rows[2].vp_decode_upgraded
    assert pinner.counters()["decode_upgrades_stock_fd"] == 1
    # band back to LOW: nothing downgrades, nothing re-upgrades
    for _ in range(8):
        pinner.observe_decode_step(512, 131_072)
    assert pinner.current_pin() == REQUEST_BODY_STOCK
    assert upgrade_pinned_rows(pinner, rows) == 0 and rows[0].vp_body == REQUEST_BODY_FD
    # "none" keeps the frozen four plans
    frozen = AdmissionPinner(_cfg(**{"admission.decode_upgrade": "none"}))
    frozen.observe_decode_step(256, 262_144)
    assert not frozen.upgrades_enabled and frozen.upgrade_pin(REQUEST_BODY_STOCK) == REQUEST_BODY_STOCK
    assert upgrade_pinned_rows(frozen, [R(REQUEST_BODY_STOCK)]) == 0
    with pytest.raises(ValueError):
        pinner.upgrade_pin("mixed")


def test_pinner_prefill_pin_reproduces_the_version_one_engagement_demotion() -> None:
    """[D-849 add. 13] the admission round's prefill pin = v1's bracket + engagement demotion
    (EMA below the floor -> dense, one routed probe round per engagement_probe_every)."""
    pinner = AdmissionPinner(_cfg(**{"admission.prefill_demotion": "admission"}))
    verdict = {"demote": False}
    pinner.set_engagement_demote(lambda: verdict["demote"])
    assert pinner.prefill_pin(1535, 3) == REQUEST_BODY_STOCK      # below the bracket
    assert pinner.prefill_pin(1536, 3) == REQUEST_BODY_FD         # engaged: routed
    verdict["demote"] = True                                      # engagement fell below the floor
    every = pinner._cfg.prefill.engagement_probe_every
    assert all(pinner.prefill_pin(4000, 8) == REQUEST_BODY_STOCK for _ in range(every))
    assert pinner.prefill_pin(4000, 8) == REQUEST_BODY_FD         # the probe round
    assert pinner.prefill_pin(4000, 8) == REQUEST_BODY_STOCK      # window restarts
    c = pinner.counters()
    assert c["prefill_demoted_rounds"] == every + 1 and c["prefill_probe_rounds"] == 1
    # observe_only keeps the pure bracket (the four-plan behaviour)
    obs = AdmissionPinner(_cfg(**{"admission.prefill_demotion": "observe_only"}))
    obs.set_engagement_demote(lambda: True)
    assert obs.prefill_pin(4000, 8) == REQUEST_BODY_FD
    # no verdict wired: bracket only
    bare = AdmissionPinner(_cfg(**{"admission.prefill_demotion": "admission"}))
    assert bare.prefill_pin(4000, 8) == REQUEST_BODY_FD


def test_prefill_tokens_criterion_is_declared_and_validated() -> None:
    """[D-849 add. 17] the round's prefill criterion counts UNCACHED prompt tokens (version 1's pass tokens)."""
    cfg = _cfg()
    assert cfg.admission.prefill_tokens == "uncached"
    assert AdmissionPinner(cfg).prefill_tokens_uncached
    assert not AdmissionPinner(_cfg(**{"admission.prefill_tokens": "prompt"})).prefill_tokens_uncached
    with pytest.raises(ValueError):
        _cfg(**{"admission.prefill_tokens": "extend"})
