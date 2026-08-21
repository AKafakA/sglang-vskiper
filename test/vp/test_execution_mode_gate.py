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
from sglang.srt.vpipe.validation import validate_full_graph_model_configuration

def call(loaded, fd, env, label):
    try:
        validate_full_graph_model_configuration(
            loaded_layers=loaded, loaded_flexidepth_layers=fd,
            tp_size=1, pp_size=1, quant_config=None, environ=env,
        )
        return ("no-raise", label)
    except Exception as e:
        return (f"{type(e).__name__}: {str(e)[:90]}", label)

ROUTED = list(range(16, 32))
results = []
# A: vanilla Llama -- no vPipe config, no routed layers. MUST NOT RAISE.
results.append(call([], [], {}, "A vanilla (no FD, no env)"))
# B: routed FD layers loaded, execution mode left at its direct_eager default.
#    MUST RAISE with the actionable message.
results.append(call(ROUTED, ROUTED, {"SGLANG_FD_WEIGHTS": "/x.pt"}, "B routed + default mode"))
# C: routed FD layers with full_graph explicitly set -- the canonical arms.
results.append(call(ROUTED, ROUTED,
                    {"SGLANG_FD_WEIGHTS": "/x.pt", "SGLANG_FD_EXECUTION_MODE": "full_graph"},
                    "C routed + full_graph"))

ok = True
for outcome, label in results:
    print(f"  {label:34s} -> {outcome}")
for (outcome, label) in results:
    if label.startswith("A") and outcome != "no-raise":
        print("FAIL: vanilla deployment would be BROKEN"); ok = False
    if label.startswith("B") and "cannot serve routed" not in outcome:
        print("FAIL: direct_eager default not refused at startup"); ok = False
print("VALIDATOR CASES:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
