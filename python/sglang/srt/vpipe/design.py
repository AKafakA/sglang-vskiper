"""The served design and the canonical arms, as code.

[D-609] WHY THIS FILE EXISTS
---------------------------
On 2026-09-09 roughly eighteen hours of A100 time measured a design the owner had
rejected. The intended configuration was pushed in through the environment
(``EXTRA_SERVER_ENV="SGLANG_FD_FULL_GRAPH_LOW_ROW_POLICY=off"``); the variable never
reached the server; every deployment manifest recorded the old default; and every
output gate -- work identity, GR-2, Accounting-v5, bank audits -- passed while the
wrong system was measured. **A green campaign on the wrong system is
indistinguishable from a green campaign on the right one.**

The owner's ruling: *"remove the knobs and record the decisions and then run it"*, and
*"never allowed to use the env to run any experiments"* -- including arm selection.

So: **deploying this tree IS configuring the experiment.** There is no variable to fail
to apply. The design is constants here; an arm is a named, frozen definition here; host
paths come from a committed per-host file. The environment is read in exactly one
remaining place -- to *refuse* stale configuration (``validation.py``) -- because
reading env to refuse is safe and reading env to configure is not.

Changing anything in this file is a design change and belongs in the decision log.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final, Mapping

# ---------------------------------------------------------------------------
# THE SERVED DESIGN
# ---------------------------------------------------------------------------
# Both admission legs. The prefill leg's lower bound stands at 1536 -- raising it to
# 3072 was tested under the same paired protocol and refuted (D-603: gsm8k TTFT +11.4 %
# at the knee, +12.9 % in overload). There is deliberately NO ``max_tokens``: the band's
# upper gate is deleted (D-596), because a hand-set token threshold pre-empted the
# engagement gate that makes the same decision from measurement.
SERVED_REGIME_SWITCH: Final[dict[str, Any]] = {
    "version": 1,
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

# Routed-MLP posture. ``off`` is the design (D-589) -- the low-row body swap was a third
# occupancy threshold outside the two admission legs, and its removal measured as parity
# on gsm8k. This is the exact setting whose env override silently did nothing.
SERVED_LOW_ROW_POLICY: Final[str] = "off"

# Every routed layer runs the count-adaptive binary-cohort body (D-574). Llama-3-8B
# routes layers 16-31; the per-layer capacities and the hand-chosen split at layer 22
# were constants fitted to one dataset and a paired A/B showed they bought nothing.
SERVED_ROUTED_LAYERS: Final[tuple[int, ...]] = tuple(range(16, 32))
SERVED_LAYER_POLICY: Final[str] = "binary_cohort"

SERVED_EXECUTION_MODE: Final[str] = "full_graph"
SERVED_COMPACT_ENABLED: Final[bool] = True
SERVED_COMPACT_PHASES: Final[frozenset[str]] = frozenset(("decode", "prefill"))
SERVED_DEVICE_ROUTE_TAPE: Final[bool] = True
SERVED_DEVICE_ROUTE_DIGEST: Final[bool] = True
SERVED_ROUTE_ACCOUNTING: Final[bool] = True
SERVED_SCHEDULER_CONVERGENCE: Final[bool] = True
SERVED_MASKED_DECODE_ATTENTION: Final[bool] = True
SERVED_CONDITIONAL_GRAPH: Final[bool] = True
SERVED_FUSED_PROJECT_INPUT: Final[bool] = True
# OFF. The gate arm script arm_env_vdec_fd.sh sets this, but the CAMPAIGN's manifest
# (21 variables) never did, so every measured cell ran without it. The design must be what
# was served, not what a neighbouring script happened to export -- taking the value from the
# gate arm instead of the manifest is the same mistake at one remove.
SERVED_DEFER_PROJECT_KV: Final[bool] = False
# Not set in the served manifest either: attestation layer counters, weighted scatter,
# prefill grouped MLP, eager semantic debug.
SERVED_LAYER_COUNTERS: Final[bool] = False
SERVED_WEIGHTED_SCATTER: Final[bool] = False
SERVED_PREFILL_GROUPED_MLP: Final[bool] = False

# Gate arithmetic is a property of the CHECKPOINT, not a tuning knob. ``released`` is the
# published FlexiDepth gate (w*MLP on RUN rows, (1-w)*PROJECT on the rest). An ste_hard
# checkpoint served under ``released`` gets scaling it was never trained with and fails
# silently -- the defect that voided the 09-03 quality gates. Qwen3 arms must override.
SERVED_GATE_MODE: Final[str] = "released"

# OFF because it is not route-identical: the fused kernel's parallel reduction differs in
# the last bit from PyTorch's, enough to flip a row whose sigmoid sits within epsilon of
# the 0.5 RUN/PROJECT threshold (measured: exactly 1 row of 735,888). Deterministic, but
# it breaks exact route equality against the FlexiDepth oracle.
SERVED_FUSED_ROUTER_NORM: Final[bool] = False

# ---------------------------------------------------------------------------
# THE CANONICAL ARMS
# ---------------------------------------------------------------------------
# An arm is WHAT IS BEING MEASURED; the design above is HOW it is served, and does not
# vary between arms. Selected by name -- never by exported variables. These replace
# test/vp/gates/arms/arm_env_*.sh, which are deleted.
ARMS: Final[dict[str, dict[str, Any]]] = {
    # OUR FORK WITH THE SKIPPER OFF. Read that again: this is NOT upstream SGLang, and this
    # comment used to claim it was ("Stock upstream SGLang ... the paper's baseline arm"),
    # which is how at least three sessions concluded the baseline was already upstream and
    # built campaigns on it. It is this tree -- same 193 commits, same hook sites in
    # models/llama.py, same vpipe package imported -- with the treatment not applied.
    #
    # THE PAPER'S BASELINE IS NOT THIS ARM. It is a separate freshly-cloned upstream tree at
    # commit 602c8615a1 carrying no vpipe/ package at all, staged and content-verified
    # against deploy/upstream_baseline.json and served by its own PYTHONPATH (owner order
    # D-587: "it has to be the based directly-freshly cloned sglang without our changes").
    #
    # This arm's legitimate role is the CONTROL that measures what our fork costs when it is
    # not treating anything -- D-597 found +0.06 %/+0.14 % output TPS against fresh upstream,
    # i.e. nothing. But that was measured at 150 commits past upstream and HEAD is now 193,
    # and an equivalence carried across a design change is what voided the v1.3 table. Serve
    # upstream for the baseline and this becomes a control rather than a load-bearing
    # assumption.
    #
    # TODO (owner 2026-09-10): rename this key. NOT to "upstream" -- that would make the old
    # lie permanent -- but to something that says what it is, e.g. "fork_noskip". Deferred
    # only because the name is written into deploy/active_arm, expect_stock.json and every
    # manifest already on the boxes.
    "stock": {"skipper": None, "phases": None, "regime_switch": False},
    # FlexiDepth, decode phase only.
    "vdec_fd": {"skipper": "flexidepth", "phases": "decode", "regime_switch": True},
    # FlexiDepth, prefill phase only.
    "vpre_binarycohort": {"skipper": "flexidepth", "phases": "prefill", "regime_switch": True},
    # FlexiDepth, both phases -- THE SERVED SYSTEM the paper reports.
    "integrated_it4": {"skipper": "flexidepth", "phases": "both", "regime_switch": True},
    # [D-627] THE QUALITY POSTURE. Identical to integrated_it4 except that admission is OFF,
    # so every routed pass routes regardless of load.
    #
    # This restores what the deleted arm_env_*.sh scripts did: they exported NO
    # SGLANG_VP_REGIME_SWITCH, so the mechanism was off and the arm always routed. The D-611
    # refactor set regime_switch=True on every routed arm, and Codex flagged it (review 1, P1
    # F4) -- I dismissed it as matching the campaign manifest. It does match the CAMPAIGN. It
    # does not match the QUALITY lane, and the two need opposite postures:
    #
    #   performance : admission ON  -- measures the served design under real load
    #   quality     : admission OFF -- measures the SKIPPER; quality must not depend on load
    #
    # Measured consequence of getting this wrong: an lm-eval run at concurrency 16 sits far
    # below enter_rows=176, so the decode leg stays in prod_allrun. A quality run on
    # 2026-09-09 executed 16,702 decode passes with skip=0 -- it measured a system that never
    # skipped, and every quality number from it is void.
    "integrated_alwaysskip": {
        "skipper": "flexidepth",
        "phases": "both",
        "regime_switch": False,
    },
    # Arbitrary per-token routes with no semantics: the substrate-generality arm. Proves
    # the runtime assumes nothing about the policy that produced a route, and carries the
    # skip-rate x depth trade-off study.
    #
    # [D-619] BOTH phases, and named to say so. It was `vdec_randomskip`, decode-only, but
    # the study overlays FlexiDepth's operating point on this surface and the served system
    # (integrated_it4) routes both phases -- a decode-only mock would confound skip
    # magnitude with phase coverage. The mock itself is phase-agnostic: prepare_batch
    # `del phase` and derives routes from (request_ids, token_epochs), so this is
    # configuration, not new code. The old name would have been a label asserting something
    # the system does not do.
    "integrated_randomskip": {
        "skipper": "deterministic_mock",
        "phases": "both",
        "regime_switch": True,
        # These three are NOT free parameters: they are the values the deleted
        # arm_env_vdec_randomskip.sh exported, carried over verbatim. Choosing new ones
        # while "porting" the arm would be a silent design change (global rule R3) that
        # makes the generality arm incomparable to every earlier RandomSkip result.
        "mock_token_skip_rate": 0.5,
        "mock_skipped_depth_ratio": 0.5,
        "mock_seed": 1234,
    },
}

# THE SKIP-RATE x DEPTH SWEEP (plan item 3; owner scope 2026-09-10: gsm8k only, ALL 12 points,
# one rate, 3 reps, with the upstream anchor interleaved in the same session).
#
# Twelve named arms, because that is the only parameterisation the repo supports. There is no
# sweep loop, no CLI, and no value-taking arm field, and D-609 forbids expressing any of this
# as an environment variable -- the design lives in the tree. `resolve_arm` is a dict lookup
# and no consumer hard-codes the arm list, so naming them is sufficient.
#
# Built by comprehension rather than twelve hand-written dicts: the entries differ only in two
# floats, and twelve near-identical literals are exactly where a transposed digit silently
# mislabels a sweep point. It is still a module-level constant evaluated at import, which is
# what D-609 asks for. Each point also publishes its own rate/depth/seed through the
# attestation (`skipper.py:284-286`), so a mislabelled cell is detectable in `server_info`
# rather than taken on trust.
#
# The SEED is identical across all twelve: the curve must vary in rate and depth only.
#
# DEPTH is a ratio WITHIN the fixed routed set `SERVED_ROUTED_LAYERS` (range(16, 32)), taken
# from the tail -- 0.25/0.50/0.75 select layers 28-31 / 24-31 / 20-31. Do NOT express depth by
# changing SERVED_ROUTED_LAYERS: that is a design constant shared by every arm, and
# `verify_served_design` will correctly refuse the diff.
#
# `integrated_randomskip` above is LEFT ALONE. Its values are the deleted
# arm_env_vdec_randomskip.sh's, verbatim, which keeps it comparable to prior RandomSkip
# evidence. Note that `..._r50_d50` is configuration-identical to it, same seed included --
# so the two are a free internal consistency check, not a duplicate.
_SWEEP_SKIP_RATES = (0.10, 0.25, 0.50, 0.75)
_SWEEP_DEPTH_RATIOS = (0.25, 0.50, 0.75)

ARMS.update({
    f"integrated_randomskip_r{int(rate * 100)}_d{int(depth * 100)}": {
        "skipper": "deterministic_mock",
        "phases": "both",
        "regime_switch": True,
        "mock_token_skip_rate": rate,
        "mock_skipped_depth_ratio": depth,
        "mock_seed": 1234,
    }
    for rate in _SWEEP_SKIP_RATES
    for depth in _SWEEP_DEPTH_RATIOS
})


def resolve_arm(name: str) -> dict[str, Any]:
    """Return the frozen definition for ``name``, or fail closed."""

    try:
        return dict(ARMS[name])
    except KeyError:
        raise ValueError(
            f"unknown arm {name!r}; canonical arms are {sorted(ARMS)} "
            "(defined in vpipe/design.py, D-609 -- arms are not configured by environment)"
        ) from None


# ---------------------------------------------------------------------------
# HOST PATHS
# ---------------------------------------------------------------------------
# Filesystem locations differ per machine and are NOT design. They come from a committed
# per-host file so they are versioned and auditable, never from shell exports. Absent or
# unreadable = fail closed at boot, which is the opposite of an env var's silent default.
_HOST_CONFIG_ENV = "SGLANG_VP_HOST_CONFIG"  # a PATH to the file, never a design value
_REQUIRED_HOST_KEYS = ("flexidepth_weights",)


def host_config(path: str | None = None) -> dict[str, str]:
    """Load the per-host deployment paths, failing closed."""

    raw = path or os.environ.get(_HOST_CONFIG_ENV, "")
    if not raw:
        raise ValueError(
            f"{_HOST_CONFIG_ENV} must point at a committed host-config file "
            "(deploy/hosts/<host>.json). Host paths are not supplied as environment "
            "values (D-609)."
        )
    p = Path(raw)
    if not p.is_file():
        raise ValueError(f"{_HOST_CONFIG_ENV} points at {raw!r}, which is not a file")
    cfg = json.loads(p.read_text())
    missing = [k for k in _REQUIRED_HOST_KEYS if not str(cfg.get(k, "")).strip()]
    if missing:
        raise ValueError(f"host config {raw} is missing required keys: {missing}")
    for key in ("flexidepth_weights", "conditional_graph_helper", "moe_config_dir"):
        value = str(cfg.get(key, "") or "").strip()
        if value and not Path(value).exists():
            raise ValueError(f"host config {raw}: {key} = {value!r} does not exist")
    return {k: str(v) for k, v in cfg.items()}


def design_attestation() -> dict[str, Any]:
    """The design, for the manifest, so read-back can verify what was served.

    This is what the launch gate compares against: the boot writes this into
    ``observed_runtime`` and the campaign refuses to start if it does not match.
    """

    return {
        "source": "vpipe/design.py (constants, D-609)",
        "regime_switch": SERVED_REGIME_SWITCH,
        "low_row_policy": SERVED_LOW_ROW_POLICY,
        "routed_layers": list(SERVED_ROUTED_LAYERS),
        "layer_policy": SERVED_LAYER_POLICY,
        "execution_mode": SERVED_EXECUTION_MODE,
        "compact_enabled": SERVED_COMPACT_ENABLED,
        "compact_phases": sorted(SERVED_COMPACT_PHASES),
        "gate_mode": SERVED_GATE_MODE,
        "fused_router_norm": SERVED_FUSED_ROUTER_NORM,
    }


# ---------------------------------------------------------------------------
# RESOLVED HOST PATHS
# ---------------------------------------------------------------------------
# The served path asks for a path; it never asks the environment. The one variable
# that survives anywhere is SGLANG_VP_HOST_CONFIG, and it carries a POINTER to a
# committed file -- not a value. If it is absent, or the file is missing, or a path
# inside it does not exist, boot fails loudly. That is the opposite of an env var,
# whose absence produces a silent default.
_HOST_CACHE: dict[str, dict[str, str]] = {}


def _host() -> dict[str, str]:
    key = os.environ.get(_HOST_CONFIG_ENV, "")
    if key not in _HOST_CACHE:
        _HOST_CACHE[key] = host_config(key or None)
    return _HOST_CACHE[key]


def flexidepth_weights_path() -> str:
    """Absolute path to the skipper's router/projector weights."""

    return _host()["flexidepth_weights"]


def conditional_graph_helper_path() -> str:
    return _host().get("conditional_graph_helper", "")


def moe_config_dir() -> str:
    return _host().get("moe_config_dir", "")


def served_model_revision() -> str:
    return _host().get("served_model_revision", "")


# ---------------------------------------------------------------------------
# THE ACTIVE ARM
# ---------------------------------------------------------------------------
# Selected by a DURABLE FILE, not a shell export. The campaign script writes one arm
# name into `deploy/active_arm`; the served path reads it; the manifest records what
# resolved. A file survives the process, can be inspected after the fact, and diffs --
# an `export` leaves no trace and cannot be checked once the shell is gone.
#
# An unknown name raises. There is no "default that quietly applies", which is the
# property that let a wrong configuration run for eighteen hours.
_ACTIVE_ARM_FILE = "deploy/active_arm"
_ARM_CACHE: dict[str, str] = {}


def _repo_root() -> Path:
    # vpipe/design.py -> vpipe -> srt -> sglang -> python -> <tree root>
    return Path(__file__).resolve().parents[4]


def active_arm_name() -> str:
    """Name of the arm this process is serving, from the durable file."""

    if "name" not in _ARM_CACHE:
        f = _repo_root() / _ACTIVE_ARM_FILE
        if not f.is_file():
            # [Codex F12] No silent default. Substituting a default arm here would quietly
            # serve the integrated skipper for a deployment that forgot to state its arm --
            # the "default that quietly applies" property that let a wrong configuration
            # run for eighteen hours. An unstated arm is an incomplete deployment.
            raise ValueError(
                f"{_ACTIVE_ARM_FILE} is missing from the deployed tree ({f}). The arm is "
                f"not defaulted: write one of {sorted(ARMS)} into that file (D-609/D-611)."
            )
        name = f.read_text().strip()
        resolve_arm(name)  # fail closed on an unknown name
        _ARM_CACHE["name"] = name
    return _ARM_CACHE["name"]


def active_arm() -> dict[str, Any]:
    return resolve_arm(active_arm_name())


def skipper_deployed() -> bool:
    """True when the active arm serves a skipper at all (stock serves none)."""

    return active_arm().get("skipper") is not None


def arm_phases() -> frozenset[str]:
    """The request phases the active arm routes. Empty when it serves no skipper."""

    phases = active_arm().get("phases")
    if not phases:
        return frozenset()
    if phases == "both":
        return frozenset(("decode", "prefill"))
    return frozenset((str(phases),))


def mechanism(value: bool, *, decode_only: bool = False) -> bool:
    """Apply a design constant only where the arm can actually run it.

    [D-609] The constants above describe THE SERVED SKIPPER SYSTEM. They are not
    universal truths about the process: an arm serving no skipper (``stock``) must have
    every one of them off, and a decode-phase mechanism needs an arm with a decode phase.

    The old arm_env_*.sh scripts encoded this implicitly by simply not exporting a
    variable in the arms where it did not apply. Deleting the scripts made that
    implicit knowledge disappear, and the smoke gate found it immediately: ``stock``
    demanded full_graph weights it does not have, and ``vpre_binarycohort`` asserted a
    conditional decode graph with no decode phase. Both were unservable. This function
    is where that knowledge now lives, once, checkably -- instead of in six scripts.
    """

    if not skipper_deployed():
        return False
    if decode_only and "decode" not in arm_phases():
        return False
    return value
