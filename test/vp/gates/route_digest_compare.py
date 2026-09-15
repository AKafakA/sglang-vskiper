import json, hashlib, os, pathlib
R = pathlib.Path(os.environ.get("VP_GATE_AB_OUT", "/tmp/vpipe-gates/route-digest-ab"))

def routes(p):
    d = json.load(open(p))
    for s in (d.get("internal_states") or []):
        rt = s.get("vp_runtime") or s.get("runtime") or {}
        m = (rt.get("model") or {}).get("flexidepth") or {}
        fr = m.get("full_graph_routes")
        if fr:
            return fr
    return None

def outputs(p):
    if not p.exists():
        return None
    h = hashlib.sha256(); n = 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get("text") or r.get("output") or r.get("completion") or ""
        h.update(str(t).encode()); n += 1
    return h.hexdigest(), n

a = routes(R / "refactored.server_info.json")
b = routes(R / "frozen.server_info.json")
print("\n=== ROUTE TAPE COMPARISON ===")
if not a or not b:
    print("  MISSING full_graph_routes -- cannot compare")
    raise SystemExit(2)

ok = True
for k in ["layer_rows", "run_rows", "project_rows", "skip_ratio"]:
    same = a.get(k) == b.get(k); ok &= same
    print(f"  {k:16s} refactored={a.get(k)!r:>14}  frozen={b.get(k)!r:>14}  "
          f"{'EQUAL' if same else '*** DIFFER ***'}")

ta = a.get("device_route_tape") or {}
tb = b.get("device_route_tape") or {}
for k in ["dispatches", "action_rows", "run_rows", "metadata_rows",
          "digest_sum_u64", "digest_xor_u64"]:
    if k in ta or k in tb:
        same = ta.get(k) == tb.get(k); ok &= same
        print(f"  tape.{k:20s} refactored={ta.get(k)!r:>22}  frozen={tb.get(k)!r:>22}  "
              f"{'EQUAL' if same else '*** DIFFER ***'}")

oa = outputs(R / "refactored.labels.jsonl")
ob = outputs(R / "frozen.labels.jsonl")
print("\n=== GENERATED-TEXT COMPARISON ===")
if oa and ob:
    same = oa[0] == ob[0]; ok &= same
    print(f"  refactored sha256={oa[0][:16]} n={oa[1]}")
    print(f"  frozen     sha256={ob[0][:16]} n={ob[1]}")
    print(f"  {'IDENTICAL' if same else '*** DIFFER ***'}")
else:
    print("  labels.jsonl missing on one side")

print("\n=== VERDICT:", "ROUTE + OUTPUT EQUAL" if ok else "NOT EQUAL", "===")
