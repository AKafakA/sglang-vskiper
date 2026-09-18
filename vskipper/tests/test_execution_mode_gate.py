"""Startup gate for the execution mode: three cases that must all hold.

The direct_eager fix refuses, at startup, a routed FlexiDepth deployment whose
execution mode cannot serve. The danger is over-reach: this validator runs
UNCONDITIONALLY from LlamaForCausalLM.__init__ with no early return, and
resolve_full_graph_skipper() returns the FlexiDepth adapter by default whose
requires_flexidepth_weights defaults to True. Keying the refusal off the adapter
therefore breaks a VANILLA Llama server with no vPipe configuration -- and NO
canonical arm covers that case, so nothing downstream would catch it.

Case A is the regression guard. Run as a script; exits non-zero on failure.
"""
import sys
from vskipper.runtime.validation import validate_full_graph_model_configuration

def call(loaded, fd, env, label, graphs=False):
    try:
        validate_full_graph_model_configuration(
            loaded_layers=loaded, loaded_flexidepth_layers=fd,
            tp_size=1, pp_size=1, quant_config=None, environ=env,
            cuda_graph_enabled=graphs,
        )
        return ("no-raise", label)
    except Exception as e:
        return (f"{type(e).__name__}: {str(e)[:90]}", label)

ROUTED = list(range(16, 32))
results = []
# A: vanilla Llama -- no vPipe config, no routed layers. MUST NOT RAISE.
results.append(call([], [], {}, "A vanilla (no FD, no env)", graphs=True))
# B: routed FD layers loaded, execution mode left at its direct_eager default.
#    MUST RAISE with the actionable message.
# direct_eager + graphs is impossible (data-dependent gather under capture).
results.append(call(ROUTED, ROUTED, {"SGLANG_FD_WEIGHTS": "/x.pt"},
                    "B routed + eager + graphs", graphs=True))
# direct_eager WITHOUT graphs is the supported quality-reference posture.
results.append(call(ROUTED, ROUTED, {"SGLANG_FD_WEIGHTS": "/x.pt"},
                    "D routed + eager + NO graphs", graphs=False))
# C: routed FD layers with full_graph explicitly set -- the canonical arms.
results.append(call(ROUTED, ROUTED,
                    {"SGLANG_FD_WEIGHTS": "/x.pt", "SGLANG_FD_EXECUTION_MODE": "full_graph"},
                    "C routed + full_graph", graphs=True))

# --- removed-mode flags: rejection must match the variable's SHAPE ---
# The V2/V4 keys name a config PATH (any non-empty value selects the removed
# runtime). The other three are BOOLEANS (only a truthy value ever did).
# Rejecting everything "not in ('', '0')" refused an explicit
# SGLANG_VP_SCHED=false -- a deployment saying OFF -- and that over-broad shape
# is the failure mode case A already exists to catch.
FLAG_CASES = [
    ({"SGLANG_VP_SCHED": "1"}, True, "SCHED=1"),
    ({"SGLANG_VP_SCHED": "true"}, True, "SCHED=true"),
    ({"SGLANG_VP_SCHED": "on"}, True, "SCHED=on"),
    ({"SGLANG_VP_SCHED": "0"}, False, "SCHED=0"),
    ({"SGLANG_VP_SCHED": "false"}, False, "SCHED=false"),
    ({"SGLANG_VP_SCHED": "no"}, False, "SCHED=no"),
    ({"SGLANG_VP_SCHED": "off"}, False, "SCHED=off"),
    ({"SGLANG_VP_SCHED": "False"}, False, "SCHED=False (case-insensitive)"),
    ({"SGLANG_VP_SCHED": "banana"}, True, "SCHED=banana (not boolean)"),
    ({"SGLANG_FD_VP_PROJECT": "false"}, False, "VP_PROJECT=false"),
    ({"SGLANG_FD_VP_PROJECT": "1"}, True, "VP_PROJECT=1"),
    ({"SGLANG_FD_VP_STAGE_ROUTE": "off"}, False, "STAGE_ROUTE=off"),
    ({"SGLANG_VP_V4_CONFIG": "/tmp/x.json"}, True, "V4_CONFIG=path"),
    ({"SGLANG_VP_V4_CONFIG": "0"}, True, "V4_CONFIG=0 is still a path value"),
    ({"SGLANG_VP_V4_CONFIG": ""}, False, "V4_CONFIG empty"),
    ({"SGLANG_VP_V2_CONFIG": "/tmp/y.json"}, True, "V2_CONFIG=path"),
]

def flag_rejected(env):
    """True iff the REMOVED-MODE block rejected it (not some later check)."""
    outcome, _ = call([], [], env, "flag", graphs=False)
    return (
        "not part of" in outcome
        or "must be a boolean" in outcome
        or "removed V2/V4" in outcome
    )

flag_results = [(flag_rejected(env), want, label) for env, want, label in FLAG_CASES]

ok = True
for got, want, label in flag_results:
    print(f"  {label:34s} -> rejected={got} (want {want})")
    if got != want:
        print(f"FAIL: removed-flag rejection wrong for {label}")
        ok = False

for outcome, label in results:
    print(f"  {label:34s} -> {outcome}")
for (outcome, label) in results:
    if label.startswith("A") and outcome != "no-raise":
        print("FAIL: vanilla deployment would be BROKEN"); ok = False
    if label.startswith("B") and "cannot run with CUDA" not in outcome:
        print("FAIL: eager+graphs not refused at startup"); ok = False
    if label.startswith("C") and outcome != "no-raise":
        print("FAIL: the CANONICAL full_graph posture is refused"); ok = False
    if label.startswith("D") and outcome != "no-raise":
        print("FAIL: the quality-reference posture (eager, no graphs) is BLOCKED"); ok = False
print("VALIDATOR CASES:", "PASS" if ok else "FAIL")

if __name__ == "__main__":
    sys.exit(0 if ok else 1)


def test_execution_mode_startup_gate():
    """Every case must hold. Asserts, so pytest collection actually gates it.

    Without this the module only signalled through sys.exit under __main__, and
    no repository gate ran it as a script -- a guard that could not fail.
    """

    outcomes = dict((label, outcome) for outcome, label in results)
    assert outcomes["A vanilla (no FD, no env)"] == "no-raise", \
        "a vanilla SGLang server must not be refused"
    assert "cannot run with CUDA" in outcomes["B routed + eager + graphs"], \
        "eager + CUDA graphs must be refused at startup"
    assert outcomes["C routed + full_graph"] == "no-raise", \
        "the canonical full_graph posture must boot"
    assert outcomes["D routed + eager + NO graphs"] == "no-raise", \
        "the quality-reference posture must boot"


def test_removed_mode_flag_rejection_matches_shape():
    """Truthy removed flags refused; falsy ones allowed; config paths always.

    A blanket "not in ('', '0')" check refused SGLANG_VP_SCHED=false, i.e. a
    deployment explicitly turning the removed path OFF.
    """

    for got, want, label in flag_results:
        assert got == want, f"removed-flag rejection wrong for {label}"
