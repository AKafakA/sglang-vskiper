"""Derive the decode engagement band from device properties instead of declaring it.

`enter_rows=176` / `exit_rows=144` were, by the record, **defaults in a test fixture** in the
commit that created the regime switch (`fa8d914686`, 2026-07-23) — carried ever since, with no
entry anywhere recording how they were chosen (D-705). That is exactly the shape a reviewer
calls a magic number, and the answer is not prose: it is to make the runtime COMPUTE them.

The rule, and why each term is there:

    ridge = peak_flops / peak_bw        For a weight-stationary decode GEMM at batch M with
                                        bf16 weights, FLOPs = 2*M*K*N and weight bytes = 2*K*N,
                                        so arithmetic intensity is EXACTLY M, independent of
                                        layer shape. Below the ridge the pass is
                                        weight-bandwidth-bound: the weights are read whether or
                                        not a row skips, so routing rows away saves no traffic
                                        and only adds overhead. Above it, removed rows are
                                        removed work. The ridge is therefore a row count.

    r0    = nearest capture rung        The decode CUDA-graph ladder is quantised (rungs every 8
                                        up to 256). A threshold between rungs would switch
                                        bodies at a PADDED batch, so the centre must be a rung.

    half  = rung above jitter p90       Hysteresis exists to stop the band chattering while
                                        occupancy oscillates. It must exceed the step-to-step
                                        swing or every drain pays repeated body transitions —
                                        the failure the 2026-07-23 W1 readout diagnosed as
                                        "occupancy oscillates across the 144/176 dead-band ...
                                        cascading into a queue blowup".

    enter = r0 + half    exit = r0 - half

⚠ **The centre is device-derived; the width is NOT.** Occupancy jitter is arrival burstiness
times service time — a property of the WORKLOAD. Report the two separately; calling both
"device-derived" is the small overstatement that makes a reviewer distrust the rest.

⚠ **This is a principled DEFAULT, never a claimed optimum** (owner, D-705). Results obtained
with it are obtained without per-workload tuning, which makes them a lower bound. Sizing the
band online from the server's own occupancy distribution is available and untried.

The peaks come from a committed artifact of PUBLISHED SPECIFICATIONS, keyed like the tuned tile
artifacts and failing closed on an unknown device — never a heuristic fallback, and never a
number measured on our own hardware and then presented as a device property.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

#: Observed step-to-step occupancy swing used for the band half-width, in rows.
#: Measured on the 2026-09-11 headline rep 1, pooled over five cells in the 100-250 row
#: range: p50 1, p75 6, p90 13, p95 15, p99 27 (n=755). p90 and p95 both round to the same
#: rung, so the band is not sensitive to that choice; p99 would widen it to [128, 192].
DEFAULT_JITTER_P90_ROWS = 13


@lru_cache(maxsize=None)
def _roofline_table() -> dict:
    path = Path(__file__).parent / "device_roofline.json"
    return json.loads(path.read_text())


def device_peaks(device_key: str) -> tuple[float, float]:
    """(dense bf16 TFLOP/s, HBM GB/s) for a CANONICAL device key, or refuse.

    Takes the canonical key, not the raw CUDA name: canonicalisation lives in `kernel.py`
    beside the tuned tile artifacts that use the same key, and importing it here would pull
    torch/triton into a module that is otherwise pure arithmetic — which is also what makes
    this rule testable without a GPU.
    """
    key = device_key
    table = _roofline_table()
    entry = table.get(key)
    if entry is None:
        known = sorted(k for k in table if not k.startswith("_"))
        raise RuntimeError(
            f"no published roofline for canonical device key {key!r}; "
            f"known: {known}. Add its vendor spec to device_roofline.json — this must not "
            "fall back to a heuristic."
        )
    return float(entry["peak_tflops_bf16"]), float(entry["peak_bw_gbs"])


def ridge_rows(device_key: str) -> float:
    """The device's machine balance, in decode rows (intensity = M, so FLOP/byte = rows)."""
    tflops, bw_gbs = device_peaks(device_key)
    return tflops * 1e12 / (bw_gbs * 1e9)


def _nearest(value: float, rungs: tuple[int, ...]) -> int:
    return min(rungs, key=lambda r: (abs(r - value), r))


def _rung_at_or_above(value: float, rungs: tuple[int, ...]) -> int:
    above = [r for r in rungs if r >= value]
    if not above:
        raise ValueError(f"no capture rung at or above {value}; ladder tops out at {max(rungs)}")
    return min(above)


def derived_decode_band(
    device_key: str,
    capture_rungs: tuple[int, ...],
    jitter_p90_rows: int = DEFAULT_JITTER_P90_ROWS,
) -> tuple[int, int]:
    """Return (exit_rows, enter_rows) for this device and capture ladder.

    `capture_rungs` is the LIVE ladder, passed in rather than recomputed here: this module
    must not hold a second copy of a quantity the runtime already owns.
    """
    if not capture_rungs:
        raise ValueError("an empty capture ladder cannot quantise a threshold")
    rungs = tuple(sorted(set(int(r) for r in capture_rungs)))
    centre = _nearest(ridge_rows(device_key), rungs)
    # The half-width is a number of ROWS, rounded up to the ladder's own quantum so that both
    # endpoints land on rungs.
    half = _rung_at_or_above(float(jitter_p90_rows), rungs) if jitter_p90_rows > rungs[0] else rungs[0]
    exit_rows, enter_rows = centre - half, centre + half
    if exit_rows <= 0:
        raise ValueError(
            f"derived exit_rows {exit_rows} is not positive (centre {centre}, half {half})"
        )
    return exit_rows, enter_rows
