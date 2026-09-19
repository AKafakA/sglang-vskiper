#!/bin/bash
# Push a served export (gates/<arm>/infer-step<N>) plus its probe/gate records to the Hub repo asdwb/vskip-flexidepth-qwen-3 under
# <folder>/ on main. Token from HF_TOKEN in the environment only (never on disk). Usage:
#   hf_push_ckpt.sh q8b-ste-c1e4-from7500 15000 qwen3-8b            # the served checkpoint
#   hf_push_ckpt.sh q8b-ste-c5e5-from7500 18750 qwen3-8b-2e-4-step18750
set -eu; ARM=$1; STEP=$2; DEST=$3; REPO=asdwb/vskip-flexidepth-qwen-3; SRC=/mydata/q8b/gates/$ARM/infer-step$STEP
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN unset"; exit 2; }
[ -d "$SRC" ] || { echo "no export at $SRC"; exit 2; }
/mydata/venvs/training/bin/python - "$REPO" "$SRC" "$DEST" "$ARM" "$STEP" <<PY
import glob, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
repo, src, dest, arm, step = sys.argv[1:]
api = HfApi()
api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=dest, commit_message=f"{dest}: {arm} step {step} served export")
records = sorted(glob.glob(f"/mydata/q8b/gates/{arm}/*step{step}*.json") + glob.glob(f"/mydata/q8b/gates/{arm}/gate_step{step}.log")
                 + glob.glob(f"/mydata/q8b/gates/{arm}/probe_step{step}.log"))
ops = [CommitOperationAdd(path_in_repo=f"{dest}/{os.path.basename(p)}", path_or_fileobj=p) for p in records]
if ops:
    api.create_commit(repo_id=repo, operations=ops, commit_message=f"{dest}: probe and gate records for step {step}")
print("PUSHED", repo, dest, len(records), "records")
PY
