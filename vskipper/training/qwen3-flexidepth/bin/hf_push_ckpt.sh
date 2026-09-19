#!/bin/bash
# Maintainer utility: upload a full served export and its probe/gate records.
# Destination and token come from HF_REPO_ID and HF_TOKEN, never from this file.
# Usage: hf_push_ckpt.sh q8b-ste-c1e4-from7500 15000 qwen3-8b
# Delta-only checkpoint folders require their separate packaging workflow.
set -eu
[ "$#" -eq 3 ] || { echo "Usage: $0 ARM STEP DESTINATION_FOLDER" >&2; exit 2; }
ARM=$1
STEP=$2
DEST=$3
REPO=${HF_REPO_ID:?Set HF_REPO_ID to the intended upload repository}
ROOT=${VSKIPPER_TRAINING_ROOT:-/mydata/q8b}
PYTHON_BIN=${VSKIPPER_TRAINING_PYTHON:-/mydata/venvs/training/bin/python}
SRC=$ROOT/gates/$ARM/infer-step$STEP
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN unset" >&2; exit 2; }
[ -d "$SRC" ] || { echo "no export at $SRC" >&2; exit 2; }
"$PYTHON_BIN" - "$REPO" "$SRC" "$DEST" "$ARM" "$STEP" "$ROOT" <<'PY'
import glob
import os
import sys

from huggingface_hub import HfApi, CommitOperationAdd

repo, src, dest, arm, step, root = sys.argv[1:]
api = HfApi()
api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=dest,
                  commit_message=f"{dest}: {arm} step {step} served export")
gate = os.path.join(root, "gates", arm)
records = sorted(glob.glob(f"{gate}/*step{step}*.json")
                 + glob.glob(f"{gate}/gate_step{step}.log")
                 + glob.glob(f"{gate}/probe_step{step}.log"))
ops = [CommitOperationAdd(path_in_repo=f"{dest}/{os.path.basename(p)}",
                          path_or_fileobj=p) for p in records]
if ops:
    api.create_commit(repo_id=repo, operations=ops,
                      commit_message=f"{dest}: probe and gate records for step {step}")
print("PUSHED", repo, dest, len(records), "records")
PY
