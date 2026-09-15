import json
if len(sys.argv) != 3: sys.exit(f"usage: {sys.argv[0]} <dev probe path> <ref probe path>   (each: .../probe or .../probe/labels.jsonl)")
D = sys.argv[1]   # the dev tree's probe output
R = sys.argv[2]   # the ref tree's probe output
dl=[json.loads(l) for l in open(D)]; rl=[json.loads(l) for l in open(R)]

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
_same_requests(dl, rl)

SEMANTIC=("text","completion_tokens","finish_reason","finish_empty","prompt_tokens","request_id","suite_pos")
bad={}
for k in SEMANTIC:
    n=sum(1 for a,b in zip(dl,rl) if a.get(k)!=b.get(k))
    bad[k]=n
print("semantic fields, rows differing out of %d:" % len(dl))
for k,n in bad.items():
    print("   %-20s %d   %s" % (k, n, "IDENTICAL" if n==0 else "DIFFER"))
tot=sum(len(a.get("text") or "") for a in dl)
print("total generated characters compared: %d" % tot)
# float fields: quantify the divergence
import math
mx=0.0
for a,b in zip(dl,rl):
    x,y=a.get("first_token_logprob_raw"),b.get("first_token_logprob_raw")
    if isinstance(x,list) and isinstance(y,list) and isinstance(x[0],float) and isinstance(y[0],float):
        mx=max(mx,abs(x[0]-y[0]))
print("max |delta| on first-token logprob: %.3e" % mx)
