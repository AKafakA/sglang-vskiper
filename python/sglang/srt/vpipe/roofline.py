"""Derive the decode engagement band from device properties instead of declaring it.

`enter_rows=176` / `exit_rows=144` were, by the record, **defaults in a test fixture** in the
commit that created the regime switch (`fa8d914686`, 2026-07-23) — carried ever since, with no
entry anywhere recording how they were chosen. That is exactly the shape a reviewer
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
                                        "occupancy oscillates across the 144/176 dead-band...
                                        cascading into a queue blowup".

    enter = r0 + half    exit = r0 - half

⚠ **The centre is device-derived; the width is NOT.** Occupancy jitter is arrival burstiness
times service time — a property of the WORKLOAD. Report the two separately; calling both
"device-derived" is the small overstatement that makes a reviewer distrust the rest.

⚠ **This is a principled DEFAULT, never a claimed optimum** (owner). Results obtained
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


def assert_band_follows_rule(
    *,
    served_exit_rows: int,
    served_enter_rows: int,
    device_key: str,
    capture_rungs: tuple[int, ...],
    jitter_p90_rows: int = DEFAULT_JITTER_P90_ROWS,
) -> None:
    """Refuse to serve a band that is not what the rule produces for this device.

    **Asserted, not computed — and that is deliberate.** puts the served design in the
    TREE as constants, so an arm must still DECLARE its band; a band computed at boot would
    silently differ per host, which is the declared-vs-served split exists to prevent.
    This keeps both properties: the design is declared, and the runtime refuses to boot if the
    declaration has drifted from the rule.

    It also makes porting explicit. On a device whose ridge differs, this refuses until the arm
    declares that device's band — which is what turns the H100 row into a PREDICTION (the rule
    says (232, 264) for an H100 NVL) rather than a re-tune.

    Passing changes nothing: this can only refuse.
    """
    expected_exit, expected_enter = derived_decode_band(
        device_key, capture_rungs, jitter_p90_rows
    )
    if (served_exit_rows, served_enter_rows) != (expected_exit, expected_enter):
        raise RuntimeError(
            "served decode band does not follow the derived rule: declared "
            f"(exit={served_exit_rows}, enter={served_enter_rows}), rule gives "
            f"(exit={expected_exit}, enter={expected_enter}) for device {device_key!r} "
            f"(ridge {ridge_rows(device_key):.1f} rows, jitter p90 {jitter_p90_rows}). "
            "Either declare the derived band for this device, or record why it deviates."
        )


# ---------------------------------------------------------------------------------------
# The K/V-VOLUME band (the criterion the served design actually switches on,/).
# ---------------------------------------------------------------------------------------
# Below the ridge a decode step is weight-bandwidth-bound: the weights are streamed whether
# or not a row skips, so what a skipped row saves at a routed layer is that layer's attention
# read of its own context -- b bytes per resident token. Over L_r routed layers with a
# fraction s of decisions projecting, a batch holding V resident K/V tokens removes
# s*L_r*b*V bytes of traffic, i.e. s*L_r*b*V/BW seconds, against the routed body's fixed
# per-step cost tau (its extra per-layer kernels). The routed body pays when
#
#     V > V* = tau * BW / (s * L_r * b)
#
# On the A100-80GB PCIe (BW 1935 GB/s): 2.67 ms * 1.935 TB/s / (0.5 * 16 * 4 KB) = 158k tokens.
# The served band (exit 160k, enter 200k = 1.25 V*) was set from the two-body crossover
# ladder (lane-2 cut1: 131k parity, 262k win) BEFORE this derivation, which reproduces it
# without fitting. tau is's one profile on the 2026-09-04 body: a cheaper body moves V*
# down, so the served constants can only engage LATER than optimal, never in the wrong
# direction. No optimum is claimed. For another device the same rule with its bandwidth is a
# PREDICTION (H100 HBM3: 270k/340k; H100 NVL: 320k/400k), checked by that device's ladder.

#: Routed decode body's fixed per-step cost, ms (: ~12 extra kernels per routed layer,
#: 320 launches per 20 steps; measured once on the A100 testbed, disclosed as such).
DECODE_BODY_TAX_MS = 2.67
#: Fraction of routed-layer decisions that PROJECT, as attested on served decode passes
#: (0.50-0.57 on gsm8k/CoQA cells); the rule uses 0.5.
DESIGN_DECODE_SKIP_RATIO = 0.5
#: K/V bytes per token per layer for the served model: 2 (K,V) * 8 KV heads * 128 * fp16.
LLAMA3_8B_KV_BYTES_PER_TOKEN_LAYER = 2 * 8 * 128 * 2
#: Thresholds are declared on a 10k-token grid.
KV_BAND_ROUND_TOKENS = 10_000
#: enter = this multiple of V*: the hysteresis width the A100 ladder showed (200k over 160k).
KV_BAND_ENTER_FACTOR = 1.25


def kv_crossover_tokens(
    device_key: str,
    *,
    tau_ms: float = DECODE_BODY_TAX_MS,
    skip_ratio: float = DESIGN_DECODE_SKIP_RATIO,
    routed_layers: int = 16,
    kv_bytes_per_token_layer: int = LLAMA3_8B_KV_BYTES_PER_TOKEN_LAYER,
) -> float:
    """V* in resident K/V tokens for this device (see the module note above)."""
    _, bw_gbs = device_peaks(device_key)
    return (tau_ms * 1e-3) * (bw_gbs * 1e9) / (skip_ratio * routed_layers * kv_bytes_per_token_layer)


def derived_kv_band(device_key: str, **rule) -> tuple[int, int]:
    """Return (exit_kv_tokens, enter_kv_tokens) = (V*, 1.25 V*) on the 10k grid."""
    v = kv_crossover_tokens(device_key, **rule)
    r = KV_BAND_ROUND_TOKENS
    return int(round(v / r) * r), int(round(KV_BAND_ENTER_FACTOR * v / r) * r)


def arm_kv_rule_inputs(arm: dict) -> dict:
    """The rule inputs an ARM implies: routed-layer count, its design skip ratio, and the per-step
    tax scaled by routed-layer count (the tax is per routed layer)."""

    from sglang.srt.vpipe.design import SERVED_ROUTED_LAYERS

    routed = len(tuple(arm.get("routed_layers", SERVED_ROUTED_LAYERS)))
    return {
        "routed_layers": routed,
        "skip_ratio": float(arm.get("design_skip_ratio", DESIGN_DECODE_SKIP_RATIO)),
        "tau_ms": DECODE_BODY_TAX_MS * routed / 16.0,
    }


def assert_kv_band_follows_rule(
    *, served_exit_kv_tokens: int, served_enter_kv_tokens: int, device_key: str, **rule
) -> None:
    """Refuse to serve a K/V band that is not what the rule gives for this device.

    Asserted, not computed (: the design is DECLARED in the tree; the runtime refuses to
    boot if the declaration has drifted from the rule). On a device whose bandwidth differs
    this refuses until the arm declares that device's band -- the H100 row is a prediction,
    not a re-tune. Passing changes nothing: this can only refuse.
    """
    expected = derived_kv_band(device_key, **rule)
    if (served_exit_kv_tokens, served_enter_kv_tokens) != expected:
        raise RuntimeError(
            "served decode K/V band does not follow the derived rule: declared "
            f"(exit={served_exit_kv_tokens}, enter={served_enter_kv_tokens}), rule gives "
            f"(exit={expected[0]}, enter={expected[1]}) for device {device_key!r} "
            f"(V* = {kv_crossover_tokens(device_key, **rule):.0f} tokens). Either declare the "
            "derived band for this device, or record why it deviates."
        )
