import json, hashlib
def load(p): return [json.loads(l) for l in open(p)]
import os, sys
D = os.environ.get("EQ_DEV", "/dev/shm/ab/out-dev-c1/vskipper/probe")   # the dev tree\'s probe output
R = os.environ.get("EQ_REF", "/dev/shm/ab/out-ref-c1/vskipper/probe")   # the ref tree\'s probe output
dev, ref = load(D + "/labels.jsonl"), load(R + "/labels.jsonl")
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
