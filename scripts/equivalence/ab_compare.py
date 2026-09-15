import json, hashlib, os, sys
if len(sys.argv) != 3: sys.exit(f"usage: {sys.argv[0]} <dev probe path> <ref probe path>   (each: .../probe or .../probe/labels.jsonl)")
D = sys.argv[1]   # the dev tree's probe output
R = sys.argv[2]   # the ref tree's probe output

def sha(p):
    if not os.path.exists(p):
        return "MISSING"
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

print("=== byte-level ===")
for f in ("labels.jsonl", "summary.json"):
    a, b = sha(D + "/" + f), sha(R + "/" + f)
    verdict = "IDENTICAL" if a == b else "DIFFER"
    print("  %-16s dev=%s ref=%s  %s" % (f, a, b, verdict))

print("=== generated text, request by request ===")
try:
    dl = [json.loads(l) for l in open(D + "/labels.jsonl")]
    rl = [json.loads(l) for l in open(R + "/labels.jsonl")]
    print("  rows dev=%d ref=%d" % (len(dl), len(rl)))
    print("  fields: %s" % sorted(dl[0].keys()))
    if len(dl) != len(rl):
        sys.exit("REFUSED: the two sides are not the same number of requests -- zip() would compare a prefix")
    ids_d = [(x.get("request_id"), x.get("suite_pos")) for x in dl]; ids_r = [(x.get("request_id"), x.get("suite_pos")) for x in rl]
    if ids_d != ids_r:
        sys.exit("REFUSED: request ids / positions differ between the two sides -- not the same requests in the same order")
    diff = [i for i, (a, b) in enumerate(zip(dl, rl)) if a != b]
    if diff:
        print("  rows differing: %d -> %s" % (len(diff), diff[:5]))
        i = diff[0]
        for k in sorted(dl[i]):
            if dl[i].get(k) != rl[i].get(k):
                print("    row %d key %s: dev=%r ref=%r" % (i, k, str(dl[i][k])[:70], str(rl[i].get(k))[:70]))
    else:
        print("  rows differing: 0  => BITWISE EQUAL across all %d requests" % len(dl))
except Exception as e:
    print("  labels compare failed: %r" % (e,))

print("=== attestation (excluding boot-varying fields) ===")
def flat(o, p="", acc=None):
    acc = {} if acc is None else acc
    if isinstance(o, dict):
        for k, v in o.items():
            flat(v, (p + "." + k) if p else k, acc)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            flat(v, "%s[%d]" % (p, i), acc)
    else:
        acc[p] = o
    return acc

a = flat(json.load(open(D + "/server_info.json")))
b = flat(json.load(open(R + "/server_info.json")))
SKIP = ("digest_sum_u64", "uptime", "pid", "port", "start_time", "timestamp",
        "elapsed", "_ms", "host", "path", "dir", "version_hash", "_id", "seed_time")
inter = [k for k in a if k in b and not any(t in k for t in SKIP)]
only = sorted(k for k in set(a) ^ set(b) if not any(t in k for t in SKIP))   # a key on one side only IS a difference
if only:
    print("  keys present on ONE side only (%d): %s" % (len(only), only[:8]))
tags = ("digest", "route", "skip", "cohort", "project", "run_rows", "engagement", "coverage", "compact")
routeish = [k for k in inter if any(t in k.lower() for t in tags)]
dd = [k for k in routeish if a[k] != b[k]]
print("  comparable keys %d | route/mechanism keys %d | differing %d" % (len(inter), len(routeish), len(dd)))
for k in dd[:12]:
    print("    %s: dev=%r ref=%r" % (k, a[k], b[k]))
alldiff = [k for k in inter if a[k] != b[k]]
print("  ALL comparable keys differing: %d" % len(alldiff))
for k in alldiff[:12]:
    print("    %s: dev=%r ref=%r" % (k, str(a[k])[:60], str(b[k])[:60]))
