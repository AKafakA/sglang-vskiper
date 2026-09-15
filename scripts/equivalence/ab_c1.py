import json, hashlib
def load(p): return [json.loads(l) for l in open(p)]
if len(sys.argv) != 3: sys.exit(f"usage: {sys.argv[0]} <dev probe path> <ref probe path>   (each: .../probe or .../probe/labels.jsonl)")
D = sys.argv[1]   # the dev tree's probe output
R = sys.argv[2]   # the ref tree's probe output
dev, ref = load(D + "/labels.jsonl"), load(R + "/labels.jsonl")

def _same_requests(a, b, key=("request_id", "suite_pos")):
    """Refuse a comparison whose two sides are not the same requests in the same order: zip() would
    silently truncate to the shorter side and report equality on a prefix."""
    if len(a) != len(b):
        sys.exit(f"REFUSED: {len(a)} vs {len(b)} rows -- not the same request set")
    for i, (x, y) in enumerate(zip(a, b)):
        for k in key:
            if k in x or k in y:
                if x.get(k) != y.get(k):
                    sys.exit(f"REFUSED: row {i} is {x.get(k)!r} on one side and {y.get(k)!r} on the other")
_same_requests(dev, ref)

print("rows dev=%d ref=%d" % (len(dev), len(ref)))

SEM = ("text", "completion_tokens", "finish_reason", "finish_empty", "prompt_tokens")
print("=== semantic fields ===")
for k in SEM:
    n = sum(1 for a, b in zip(dev, ref) if a.get(k) != b.get(k))
    print("   %-20s differing %d/%d   %s" % (k, n, len(dev), "IDENTICAL" if n == 0 else "DIFFER"))

txt_d = "".join(a.get("text") or "" for a in dev)
txt_r = "".join(b.get("text") or "" for b in ref)
print("=== concatenated generated text ===")
print("   chars dev=%d ref=%d" % (len(txt_d), len(txt_r)))
print("   sha256 dev=%s" % hashlib.sha256(txt_d.encode()).hexdigest()[:20])
print("   sha256 ref=%s" % hashlib.sha256(txt_r.encode()).hexdigest()[:20])
print("   %s" % ("BITWISE EQUAL" if txt_d == txt_r else "NOT EQUAL"))

mx = 0.0
for a, b in zip(dev, ref):
    u, v = a.get("first_token_logprob_raw"), b.get("first_token_logprob_raw")
    if isinstance(u, list) and isinstance(v, list) and isinstance(u[0], float):
        mx = max(mx, abs(u[0] - v[0]))
print("   max |delta| first-token logprob: %.3e" % mx)

def flat(o, p="", acc=None):
    acc = {} if acc is None else acc
    if isinstance(o, dict):
        for k, v in o.items(): flat(v, (p + "." + k) if p else k, acc)
    elif isinstance(o, list):
        for i, v in enumerate(o): flat(v, "%s[%d]" % (p, i), acc)
    else: acc[p] = o
    return acc
a = flat(json.load(open(D + "/server_info.json"))); b = flat(json.load(open(R + "/server_info.json")))
SKIP = ("digest_sum_u64","uptime","pid","port","start_time","timestamp","elapsed","_ms","host","path","dir","version_hash","_id")
inter = [k for k in a if k in b and not any(t in k for t in SKIP)]
tags = ("digest","route","skip","cohort","project","run_rows","engagement","coverage","compact")
routeish = [k for k in inter if any(t in k.lower() for t in tags)]
dd = [k for k in routeish if a[k] != b[k]]
alld = [k for k in inter if a[k] != b[k]]
print("=== attestation ===")
print("   comparable %d | route/mechanism %d | route differing %d | all differing %d"
      % (len(inter), len(routeish), len(dd), len(alld)))
for k in alld[:8]: print("     %s: dev=%r ref=%r" % (k, str(a[k])[:40], str(b[k])[:40]))
