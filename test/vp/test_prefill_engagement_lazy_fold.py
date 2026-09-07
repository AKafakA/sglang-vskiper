"""R2 (lane-2 tax-removal track, 2026-09-07): the prefill escape gate's engagement EMA must reach EXACTLY the decisions of
the synchronous reader it replaces, while never draining the stream unless a pending reading could flip the predicate.
(1) CPU exactness against a pure-Python reference with arbitrary event-completion patterns; (2) GPU no-drain / straddle
behaviour; (3) CPU-tensor and multi-device paths."""

import random
import time

import pytest
import torch

from sglang.srt.vpipe.regime import PrefillEngagementTracker


class _Reference:
    """Today's reader: fold every reading immediately (synchronously)."""

    def __init__(self):
        self.ema = None
        self.baseline = None
        self.samples = 0

    def read(self, totals):
        base = self.baseline
        if base is not None:
            d_run = totals[0] - base[0]
            d_project = totals[1] - base[1]
            d_total = d_run + d_project
            if d_total > 0:
                s = d_project / d_total
                self.ema = s if self.ema is None else 0.8 * self.ema + 0.2 * s
                self.samples += 1
        self.baseline = totals

    def demote(self, m):
        return self.ema is not None and self.ema < m


class _FakeEvent:
    """Completion is decided by the scripted pattern, not by time."""

    def __init__(self, ready):
        self.ready = ready
        self.synced = 0

    def query(self):
        return self.ready

    def synchronize(self):
        self.synced += 1
        self.ready = True


def _cpu_tracker_with_scripted_events():
    """A tracker whose readings are CPU tensors but which we drive through the CUDA-style pending path by injecting
    fake events, so the fold order / sync gating can be tested without a GPU."""
    t = PrefillEngagementTracker()
    events = []

    def snapshot(values, ready):
        cpu = torch.tensor(values, dtype=torch.int64)
        ev = _FakeEvent(ready)
        events.append(ev)
        t._pending.append(([(None, cpu)], ev))

    return t, snapshot, events


@pytest.mark.parametrize("seed", range(12))
def test_decisions_identical_to_synchronous_reader(seed):
    rng = random.Random(seed)
    ref = _Reference()
    t, snapshot, events = _cpu_tracker_with_scripted_events()
    floor = rng.choice([0.2, 0.35, 0.5])
    run = project = 0
    for step in range(60):
        # a routed pass happened: counters advance with a random PROJECT share (sometimes zero rows)
        if rng.random() < 0.85:
            rows = rng.choice([0, 4, 64, 512])
            p = int(rows * rng.random())
            run += rows - p
            project += p
        totals = (run, project)
        ref.read(totals)
        snapshot([run + project, run, project], ready=rng.random() < 0.6)
        # some steps decide, some do not (dense passes take no decision)
        if rng.random() < 0.7:
            assert t.demote(floor) == ref.demote(floor), f"seed {seed} step {step}"
        # readings complete in the background at arbitrary moments
        for ev in events:
            if rng.random() < 0.5:
                ev.ready = True
    t.sync_all()
    assert t.ema == ref.ema and t.samples == ref.samples
    # the tracker synchronised at most once per straddling decision, never on decided intervals
    assert t.syncs <= 60


def test_no_sync_when_interval_is_decided():
    t, snapshot, events = _cpu_tracker_with_scripted_events()
    t.ema = 0.9
    t._baseline = (0, 0)
    snapshot([10, 5, 5], ready=False)
    assert t.demote(0.35) is False and t.syncs == 0 and t.pending == 1  # hi_1 = 0.92, lo_1 = 0.72: decided
    t.ema = 0.05
    assert t.demote(0.35) is True and t.syncs == 0  # hi_1 = 0.24 < 0.35: decided
    t.ema = 0.3
    assert t.demote(0.35) in (True, False) and t.syncs == 1 and t.pending == 0  # straddle: synchronised once


def test_baseline_missing_semantics():
    t, snapshot, events = _cpu_tracker_with_scripted_events()
    assert t.baseline_missing
    snapshot([1, 1, 0], ready=False)
    assert not t.baseline_missing  # a pending baseline reading counts
    events[0].ready = True
    t.fold_ready()
    assert not t.baseline_missing and t.ema is None and t.samples == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_snapshot_does_not_drain_and_folds_exactly():
    dev = torch.device("cuda")
    stats = torch.zeros(3, dtype=torch.int64, device=dev)
    t = PrefillEngagementTracker()
    t.snapshot({dev: stats})  # baseline (0, 0)
    torch.cuda.synchronize()
    t.fold_ready()
    assert t._baseline == (0, 0) and t.pending == 0
    a = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(60):
        a = a @ a * 1e-4
    stats += torch.tensor([1, 30, 70], dtype=torch.int64, device=dev)  # behind the queued work
    t0 = time.perf_counter()
    t.snapshot({dev: stats})
    t.ema = 0.9
    decided = t.demote(0.35)
    host_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    drain_ms = (time.perf_counter() - t1) * 1000
    assert decided is False and t.syncs == 0
    assert drain_ms > 20, f"queue drained too fast to test ({drain_ms:.1f} ms)"
    assert host_ms < drain_ms / 4, f"snapshot/decision blocked the host: {host_ms:.1f} ms vs {drain_ms:.1f} ms queued"
    t.fold_ready()
    assert t.pending == 0 and t.samples == 1 and abs(t.ema - (0.8 * 0.9 + 0.2 * 0.7)) < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_straddle_synchronises_and_matches_reference():
    dev = torch.device("cuda")
    stats = torch.zeros(3, dtype=torch.int64, device=dev)
    t = PrefillEngagementTracker()
    t.snapshot({dev: stats}); torch.cuda.synchronize(); t.fold_ready()
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    for _ in range(30):
        a = a @ a * 1e-4
    stats += torch.tensor([1, 90, 10], dtype=torch.int64, device=dev)  # share 0.1
    t.snapshot({dev: stats})
    t.ema = 0.3  # straddles 0.35 with one pending sample -> must synchronise and use the exact value
    assert t.demote(0.35) is True and t.syncs == 1 and t.pending == 0
    assert abs(t.ema - (0.8 * 0.3 + 0.2 * 0.1)) < 1e-12


def test_cpu_tensors_take_the_synchronous_path():
    t = PrefillEngagementTracker()
    stats = torch.tensor([1, 40, 60], dtype=torch.int64)
    t.snapshot({torch.device("cpu"): stats})
    assert t.pending == 0 and t._baseline == (40, 60)
    stats += torch.tensor([1, 10, 30], dtype=torch.int64)
    t.snapshot({torch.device("cpu"): stats})
    assert t.samples == 1 and abs(t.ema - 0.75) < 1e-12
