#!/usr/bin/env bash
# The box gate CHAIN — the proven 2026-08-24 gates_box.sh, repo-resident
# (D-307 Lane H). Gates the tree this script lives in, so a staged archive of
# any branch gates itself: helper build → pytest import gates → ladder default
# modes → the FOUR-ARM gate → the route-digest output-equality A/B vs the
# frozen tree. Reachability/attestation/equality only — NEVER performance.
#
# Host-staged inputs: model, router weights, suites, serve venv, the frozen
# reference tree (c3a1302668), nvcc. See gates/README.md for the env contract.
set -uo pipefail
GATES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${VP_GATE_WORKDIR:-/local/scratch/tmp/wd312}"
TREE="${VP_GATE_TREE:-$(cd "$GATES_DIR/../../.." && pwd)}"
FROZEN="${VP_GATE_FROZEN_TREE:-$W/tree-frozen-c3a1302668}"
V="${VP_GATE_PYTHON:-$W/sglang-sm75/.venv/bin/python}"
PIP="${VP_GATE_PIP:-$W/sglang-sm75/.venv/bin/pip}"
NVCC="${VP_GATE_NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
LOG(){ echo "[$(date -u +%FT%TZ)] $*"; }
LOG "=== BOX GATES start  tree=$TREE"

LOG "--- 0. frozen-tree requires_grad fix (the declared identical fix from the CSD3 A/B)"
FROZEN_TREE_PATH="$FROZEN" python3 - <<'PY'
import os
p = os.environ["FROZEN_TREE_PATH"] + "/python/sglang/srt/models/llama.py"
s = open(p).read()
fix = "            self.fd_router.requires_grad_(False).eval()\n            self.fd_proj.requires_grad_(False).eval()\n"
if "requires_grad_(False)" in s:
    print("fix already present")
else:
    anchor = '''                "up_proj.weight": _sd[f"model.layers.{layer_id}.router_proj.up_proj.weight"],
            })
'''
    assert s.count(anchor) == 1, "anchor not unique"
    s = s.replace(anchor, anchor + fix)
    open(p, "w").write(s)
    print("fix applied (2 lines)")
PY

LOG "--- 1. conditional-graph helper build (nvcc)"
NVCC=$NVCC bash "$TREE/test/vp/build_cuda_conditional_graph_helper.sh" "$TREE" "${VP_GATE_HELPER_OUT:-$W/helper-sm75}"

LOG "--- 2. pytest static import-safety + real package imports"
$PIP -q install pytest 2>&1 | tail -1
(cd "$TREE" && $V -m pytest test/vp/test_module_import_safety.py test/vp/test_package_imports.py -q 2>&1 | tail -4)
echo "PYTEST-RC=$?"

LOG "--- 3. ladder default modes"
(cd "$TREE" && bash test/vp/test_ladder_default_modes.sh 2>&1 | tail -4)
echo "LADDER-TEST-RC=$?"

LOG "--- 4. FOUR-ARM GATE on $TREE"
for ARM in vdec_fd vpre_binarycohort integrated_it4 vdec_randomskip; do
  bash "$GATES_DIR/box_vpcov_arm.sh" $ARM "$TREE"
  echo "ARM $ARM rc=$?"
done

LOG "--- 5. ROUTE-DIGEST A/B (this tree vs frozen, vdec_fd, greedy 32)"
RES="${VP_GATE_AB_OUT:-$W/route-digest-ab}"; rm -rf "$RES"; mkdir -p "$RES"
for PAIR in "refactored:$TREE" "frozen:$FROZEN"; do
  NAME=${PAIR%%:*}; T=${PAIR##*:}
  bash "$GATES_DIR/box_vpcov_arm.sh" vdec_fd "$T" "$W/rdab-out"
  rc=$?
  cp "$W/rdab-out/vdec_fd/server_info.json"   "$RES/$NAME.server_info.json" 2>/dev/null
  cp "$W/rdab-out/vdec_fd/probe/summary.json" "$RES/$NAME.probe.json"       2>/dev/null
  cp "$W/rdab-out/vdec_fd/probe/labels.jsonl" "$RES/$NAME.labels.jsonl"     2>/dev/null
  echo "RDAB $NAME rc=$rc"
done
VP_GATE_AB_OUT="$RES" python3 "$GATES_DIR/route_digest_compare.py" 2>&1 | tee "$RES/verdict.txt"

LOG "--- 6. summary"
for ARM in vdec_fd vpre_binarycohort integrated_it4 vdec_randomskip; do
  SI=$W/vpcov-out/$ARM/server_info.json
  python3 - "$ARM" "$SI" <<'PY'
import json, sys
arm, p = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p))
except Exception as e:
    print(f"  {arm}: NO server_info ({e})"); raise SystemExit
fd = {}
for s in (d.get("internal_states") or []):
    rt = s.get("vp_runtime") or s.get("runtime") or {}
    m = (rt.get("model") or {}).get("flexidepth") or {}
    if m: fd = m; break
sk = (fd.get("skipper_adapter") or {})
routes = fd.get("full_graph_routes") or {}
print(f"  {arm}: phases={fd.get('active_phases')} skipper={sk.get('name')} layer_rows={routes.get('layer_rows')} skip_ratio={round(routes.get('skip_ratio',0),4) if routes else None}")
PY
done
LOG "=== BOX GATES done"
