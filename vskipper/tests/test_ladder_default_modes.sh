#!/usr/bin/env bash
# Every mode vast_a100_ladder.sh names as a DEFAULT must survive its own
# preflight, and every mode its preflight refuses must actually be refused.
#
# This exists because both default mode lists named refused modes at once:
# SGBENCH_MODES defaulted to "vanilla,vanilla_matched,dynamic" while dynamic is
# a removed V1 mode, and sgbench-flexidepth defaulted to a list containing
# flexidepth_vp. A default that cannot run is a harness that fails on first
# use, and nothing in the repository executed the ladder far enough to notice.
#
# It drives the REAL launch_sglang_server preflight -- extracted, with its
# SGBENCH_* inputs stubbed -- rather than re-implementing the globs, so it
# cannot drift away from the guard it is testing.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LADDER="$ROOT/vskipper/src/vskipper/experiments/vast_a100_ladder.sh"
[ -f "$LADDER" ] || { echo "FATAL: $LADDER not found"; exit 2; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
python3 - "$LADDER" > "$WORK/harness.sh" <<'PY'
import re, sys, pathlib
L = pathlib.Path(sys.argv[1]).read_text().splitlines()
start = next(i for i, l in enumerate(L) if l.startswith("launch_sglang_server()"))
# WHOLE function, to its closing brace at column 0. Cutting at the first "esac"
# stopped right after the removed-mode rejection, so an ALLOWED mode returned
# immediately and never reached the weight check, execution-mode and profile
# resolution, the direct_eager graph check, unknown-mode handling, or command
# construction -- the test passed while those were broken.
end = next(i for i in range(start + 1, len(L)) if L[i] == "}")
body = L[start:end + 1]
print("#!/usr/bin/env bash"); print("set -uo pipefail")
# Auto-stub EVERY uppercase environment-looking name the function reads, not a
# hand-kept list: chasing them one NameError at a time is how a stub list drifts
# out of date and quietly turns a real failure into a missing-variable failure.
provided = {"FD_WEIGHTS", "MODEL_PATH", "PORT", "ROOT", "DRY_RUN", "SERVER_PID",
            "PATH", "HOME", "PYTHONPATH", "IFS", "PIPESTATUS", "BASH_SOURCE"}
names = sorted(set(re.findall(r'\$\{?([A-Z][A-Z0-9_]{2,})', "\n".join(body))) - provided)
for n in names:
    print(f'{n}="${{{n}:-0}}"')
# inputs the full function reads beyond SGBENCH_*, stubbed so a supported mode
# can traverse every pre-launch check instead of dying on the environment
# DIRECT assignment, after the auto-stubs. Using ${VAR:-default} here resolved
# to the stub value (0), not the default, because the stub had already bound the
# name -- which surfaced as "unknown server profile 0" the moment the die stub
# was made to exit like the real one.
print('SGBENCH_FD_EXECUTION_MODE=full_graph')
print('SGBENCH_CANDIDATE_SERVER_PROFILE=breakable_dynamic')
print('SGBENCH_BASELINE_SERVER_PROFILE=production')
print('FD_WEIGHTS="${FD_WEIGHTS_STUB:?}"')   # a file this test creates, below
print('MODEL_PATH=/tmp; PORT=30999; ROOT=.; DRY_RUN=1')
print('HF_HOME=/tmp; HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1')
print('MODEL_ID=dummy; SERVER_PID=0')
# Join with the delimiter BETWEEN elements. The previous stub concatenated
# them, so the whole command line came out as one token and any grep over it
# matched everything -- a stub weaker than the function it replaces.
# COPY the real helpers out of the ladder rather than stubbing them. Every stub
# hand-written here has been weaker than the real thing: die() returned where
# the real one exits, so execution continued past failed checks, and join_by
# concatenated without its delimiter, so the command line became one token and
# any grep over it matched everything. Extracting removes that class of drift.
for helper in ("join_by", "die"):
    i = next(k for k, l in enumerate(L) if l.startswith(helper + "()"))
    j = next(k for k in range(i, len(L)) if L[k] == "}")
    print("\n".join(L[i:j + 1]))
# EXIT, not return -- matching the real die() at vast_a100_ladder.sh:162-165.
# A stub that merely returned let execution continue past a failed check, so an
# injected "missing required file" defect sailed through this gate. A stub
# weaker than the thing it stands in for turns the test into theatre.
print('capture_server_info(){ :; }; start_load_sampler(){ :; }; wait_sglang_ready(){ :; }')
print("\n".join(body))
PY
# A weights file this test OWNS. The ladder gates on [[ -s "$FD_WEIGHTS" ]], and
# an earlier version pointed at /etc/hostname because it existed locally -- it
# is EMPTY on the dev box, so the check failed there and passed here. Never
# depend on an incidental system file having content.
FD_WEIGHTS_STUB="$WORK/fd_weights.stub"
printf 'stub\n' > "$FD_WEIGHTS_STUB"
[ -s "$FD_WEIGHTS_STUB" ] || { echo "FATAL: could not create $FD_WEIGHTS_STUB"; exit 2; }
export FD_WEIGHTS_STUB

# shellcheck disable=SC1090
. "$WORK/harness.sh"

refused() {
  local out rc
  out=$(launch_sglang_server "$1" 30999 /dev/null 2>&1 >/dev/null); rc=$?
  [ $rc -eq 2 ] && [[ "$out" == *FATAL* ]]
}

fail=0
# 1. every DEFAULT mode list in the ladder must be runnable
mapfile -t defaults < <(grep -oE 'SGBENCH_MODES="(\$\{SGBENCH_MODES:-)?[a-z_,]+' "$LADDER" \
                        | sed -E 's/.*[:-]-?//; s/SGBENCH_MODES="//' | tr ',' '\n' | sort -u | grep -v '^$')
# The subject must be non-empty. If the grep stops matching -- because the
# declaration is reformatted, say -- the loop body never runs and this gate goes
# green having tested nothing.
if [ "${#defaults[@]}" -eq 0 ]; then
  echo "FATAL: found no SGBENCH_MODES defaults in $LADDER -- this gate would"
  echo "       otherwise pass by testing nothing at all."
  exit 2
fi
echo "default modes found: ${defaults[*]}"
for m in "${defaults[@]}"; do
  out=$(launch_sglang_server "$m" 30999 /dev/null 2>&1); rc=$?
  if [ $rc -eq 2 ] && [[ "$out" == *FATAL* ]]; then
    echo "  *** DEFAULT MODE REFUSED BY PREFLIGHT: $m"; fail=1
  elif [[ "$out" != *"launch_server"* ]]; then
    # "not refused" is too weak: the mode must actually reach command
    # construction. Otherwise a mode that dies midway on some other check
    # still counts as runnable, which is how the missing-profile failure
    # survived this gate.
    echo "  *** DEFAULT MODE DID NOT REACH LAUNCH: $m (rc=$rc) $(echo "$out" | tail -1)"; fail=1
  else
    echo "  ok   default mode reaches launch construction: $m"
  fi
done

# 1b. the ONE live override suffix must still be admitted AND still emit its
# variable. It was refused for a while by an over-broad whitelist even though
# model_runner reads SGLANG_VP_FOREGROUND_STREAM_PRIORITY and changes the CUDA
# stream priority -- a real treatment dropped along with the dead ones.
for m in flexidepth_fgpriority flexidepth_no_fgpriority \
         vanilla_fgpriority vanilla_no_fgpriority \
         vanilla_matched_fgpriority vanilla_matched_no_fgpriority; do
  out=$(launch_sglang_server "$m" 30999 /dev/null 2>&1); rc=$?
  if [ $rc -eq 2 ] || [[ "$out" == *FATAL* ]]; then
    echo "  *** LIVE SUFFIX REFUSED: $m"; fail=1; continue
  fi
  emitted=$(printf '%s' "$out" | grep -o 'SGLANG_VP_FOREGROUND_STREAM_PRIORITY=[^ ]*' | tail -1)
  case "$m" in
    *_no_fgpriority)
      # the clearing entry only; no value
      if [ "$emitted" = "SGLANG_VP_FOREGROUND_STREAM_PRIORITY=" ]; then
        echo "  ok   $m admitted, priority left at default"
      else echo "  *** $m emitted '$emitted', expected no value"; fail=1; fi
      ;;
    *)
      if [ "$emitted" = "SGLANG_VP_FOREGROUND_STREAM_PRIORITY=-1" ]; then
        echo "  ok   $m admitted and emits priority -1"
      else echo "  *** $m emitted '$emitted', expected -1"; fail=1; fi
      ;;
  esac
done

# 1c. a repeated or contradictory suffix chain must be REFUSED, not silently
# resolved to whichever suffix happens to be leftmost.
for m in flexidepth_fgpriority_no_fgpriority flexidepth_no_fgpriority_fgpriority \
         flexidepth_fgpriority_fgpriority; do
  out=$(launch_sglang_server "$m" 30999 /dev/null 2>&1 >/dev/null); rc=$?
  if [ $rc -eq 2 ] && [[ "$out" == *"repeats or contradicts"* ]]; then
    echo "  ok   contradictory chain refused: $m"
  else echo "  *** AMBIGUOUS CHAIN ACCEPTED rc=$rc: $m"; fail=1; fi
done

# 2. removed modes must still be refused (the guard has not rotted)
for m in dynamic dynamic_decode dynamic_both flexidepth_vp flexidepth_vp_sched \
         flexidepth_mixed_async flexidepth_scopedasync flexidepth_kvonly; do
  if refused "$m"; then echo "  ok   removed mode refused: $m"
  else echo "  *** REMOVED MODE NOT REFUSED: $m"; fail=1; fi
done

[ $fail -eq 0 ] && echo "LADDER DEFAULT MODES: PASS" || echo "LADDER DEFAULT MODES: FAIL"
exit $fail
