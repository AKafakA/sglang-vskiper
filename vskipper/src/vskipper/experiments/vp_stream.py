"""One SSE decoder for the whole harness.

Nine files in test/vp decoded SGLang's streamed responses independently, with
near-identical loops. That is not merely duplication: four rounds of
correctness hardening landed in ONE of them (refactor_ab/perf_client.py), so
the other eight -- including the official evaluation chain,
run_qps_evaluation.py and labeled_workload.py -- still counted a metadata-only
stream as successful work and swallowed malformed payloads in a bare except.

A fix that reaches one of nine call sites is not a fix. This module is the
single implementation, and it carries the hardened behaviour:

  * a TOKEN-BEARING event is one that actually carries generated output. A
    stream of pure metadata can self-report a full completion count while
    producing nothing, which passes a token-mass check that trusts meta_info.
  * malformed payloads RAISE rather than being ignored, because a request that
    decoded no valid event otherwise returns normally and is counted as ok.
  * streamed ids may arrive CUMULATIVELY or INCREMENTALLY and no per-event
    heuristic separates them: with ignore_eos a repeated token is legitimate,
    so [17],[17],[18] is indistinguishable from a cumulative prefix repeat.
    Both readings are tracked and the final reported count disambiguates.

`collect` is the strict form used where work must be exactly accounted.
`iter_events` is the permissive form for callers that only want the payloads
(workload builders, label probes) and do their own accounting -- it still
refuses malformed payloads, which is the part that must never be optional.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterator, Optional


class StreamError(RuntimeError):
    """A streamed response that cannot be accounted as completed work."""


def iter_events(response, *, strict: bool = True) -> Iterator[dict]:
    """Yield decoded SSE payloads, ignoring the terminator."""
    for kind, obj in decode_lines(response, strict=strict):
        if kind == "event":
            yield obj


def decode_lines(response, *, strict: bool = True) -> Iterator[tuple[str, Optional[dict]]]:
    """Yield ("event", obj) per payload and ("done", None) at the terminator.

    The lowest-level primitive, for callers whose completion policy differs from
    `collect`'s. fdpre_label_probe REQUIRES the [DONE] terminator and fails
    without it, because its last chunk is a cumulative PARTIAL label and
    recording it would silently mislabel; `collect` has no such rule. Sharing
    the parsing without imposing one caller's policy on another is the point --
    forcing them together would have deleted that guard.
    """
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            yield ("done", None)
            return
        if payload == "":
            continue
        try:
            yield ("event", json.loads(payload))
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise StreamError(f"malformed SSE payload: {str(exc)[:60]}") from exc


def token_bearing(obj: dict) -> bool:
    """True when the event carries generated output rather than metadata only."""
    return bool(obj.get("text") or obj.get("output_ids") or obj.get("token_ids"))


class StreamAccounting:
    """Timings and token accounting for one streamed response."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self.t0 = clock()
        self.ttft: Optional[float] = None
        self.reported_tokens = 0
        self.text_events = 0
        self.incremental_ids = 0
        self.cumulative_ids = 0
        self.last: Optional[dict] = None
        self.e2e: Optional[float] = None

    def observe(self, obj: dict) -> None:
        meta = obj.get("meta_info") or {}
        if "completion_tokens" in meta:
            self.reported_tokens = meta["completion_tokens"]
        ids = obj.get("output_ids") or obj.get("token_ids")
        if ids:
            self.incremental_ids += len(ids)
            self.cumulative_ids = max(self.cumulative_ids, len(ids))
        if token_bearing(obj):
            self.text_events += 1
            if self.ttft is None:
                self.ttft = self._clock() - self.t0
        self.last = obj

    def finish(self) -> None:
        self.e2e = self._clock() - self.t0
        if self.reported_tokens == 0 and self.last is not None:
            self.reported_tokens = (self.last.get("meta_info") or {}).get(
                "completion_tokens", 0
            )

    def validate(self, expected_tokens: Optional[int]) -> None:
        """Raise unless this response is complete, accounted work."""
        if self.text_events == 0:
            raise StreamError(
                "no token-bearing response event: the stream carried metadata "
                "only, so the self-reported completion count describes no "
                "generated output"
            )
        if expected_tokens is None:
            return
        if self.reported_tokens != expected_tokens:
            raise StreamError(
                f"incomplete generation: {self.reported_tokens} tokens, expected "
                f"exactly {expected_tokens} (ignore_eos=True). An empty or "
                "truncated response is a failure."
            )
        if (self.incremental_ids or self.cumulative_ids) and expected_tokens not in (
            self.incremental_ids,
            self.cumulative_ids,
        ):
            raise StreamError(
                f"streamed token ids account for {self.cumulative_ids} "
                f"(cumulative) or {self.incremental_ids} (incremental), neither "
                f"of which is the {expected_tokens} requested; meta_info reports "
                f"{self.reported_tokens}"
            )

    @property
    def tpot(self) -> float:
        assert self.e2e is not None, "finish() first"
        return (self.e2e - (self.ttft or 0.0)) / max(1, self.reported_tokens - 1)


def collect(response, *, expected_tokens: Optional[int] = None,
            clock: Callable[[], float] = time.perf_counter) -> StreamAccounting:
    """Consume a streamed response strictly, validating it as completed work."""
    acc = StreamAccounting(clock=clock)
    for obj in iter_events(response, strict=True):
        acc.observe(obj)
    acc.finish()
    acc.validate(expected_tokens)
    return acc
