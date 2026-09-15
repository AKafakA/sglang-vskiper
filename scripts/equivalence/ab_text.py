import json
import os, sys
D = os.environ.get("EQ_DEV", "/dev/shm/ab/out-dev/vskipper/probe/labels.jsonl")   # the dev tree\'s probe output
R = os.environ.get("EQ_REF", "/dev/shm/ab/out-ref/vskipper/probe/labels.jsonl")   # the ref tree\'s probe output
dl=[json.loads(l) for l in open(D)]; rl=[json.loads(l) for l in open(R)]
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
