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
LADDER="$ROOT/test/vp/vast_a100_ladder.sh"
[ -f "$LADDER" ] || { echo "FATAL: $LADDER not found"; exit 2; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
python3 - "$LADDER" > "$WORK/harness.sh" <<'PY'
import re, sys, pathlib
L = pathlib.Path(sys.argv[1]).read_text().splitlines()
start = next(i for i, l in enumerate(L) if l.startswith("launch_sglang_server()"))
end = next(i for i in range(start, len(L)) if L[i].strip() == "esac" and i > start + 40)
body = L[start:end + 1] + ["}"]
print("#!/usr/bin/env bash"); print("set -uo pipefail")
for n in sorted(set(re.findall(r'\$\{?(SGBENCH_[A-Z0-9_]+)', "\n".join(body)))):
    print(f'{n}=0')
print("\n".join(body))
PY
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
echo "default modes found: ${defaults[*]}"
for m in "${defaults[@]}"; do
  if refused "$m"; then echo "  *** DEFAULT MODE REFUSED BY PREFLIGHT: $m"; fail=1
  else echo "  ok   default mode runnable: $m"; fi
done

# 2. removed modes must still be refused (the guard has not rotted)
for m in dynamic dynamic_decode dynamic_both flexidepth_vp flexidepth_vp_sched \
         flexidepth_mixed_async flexidepth_scopedasync flexidepth_kvonly; do
  if refused "$m"; then echo "  ok   removed mode refused: $m"
  else echo "  *** REMOVED MODE NOT REFUSED: $m"; fail=1; fi
done

[ $fail -eq 0 ] && echo "LADDER DEFAULT MODES: PASS" || echo "LADDER DEFAULT MODES: FAIL"
exit $fail
