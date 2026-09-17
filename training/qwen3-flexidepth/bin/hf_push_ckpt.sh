#!/bin/bash
# Push a served export (gates/<arm>/infer-step<N>) to HF as folder <arm>/checkpoint-<N> on main of (token cannot create branches) asdwb/qwen-3-8b-flexidepth. Token from HF_TOKEN env only.
set -eu; ARM=$1; STEP=$2; REPO=asdwb/qwen-3-8b-flexidepth; SRC=/mydata/q8b/gates/$ARM/infer-step$STEP; BR=$ARM-step$STEP
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN unset"; exit 2; }
/mydata/venvs/training/bin/python - "$REPO" "$SRC" "$BR" "$ARM" "$STEP" <<PY
import sys, json, os
from huggingface_hub import HfApi
repo, src, br, arm, step = sys.argv[1:]
api = HfApi()
gate = f"/mydata/q8b/gates/{arm}/gate_step{step}.json"
readme = f"""# FlexiDepth-Qwen3-8B ({arm}, checkpoint-{step})
Served export of the alignment-only router/projector training arm `{arm}` at step {step} (Qwen3-8B base, thinking off,
routed layers 18-35). This branch is the checkpoint the vSkipper paper v1.6 serves; gate results: `gate_step{step}.json`.
"""
open(os.path.join(src, "README.md"), "w").write(readme)
if os.path.exists(gate):
    import shutil; shutil.copy(gate, os.path.join(src, f"gate_step{step}.json"))
api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=f"{arm}/checkpoint-{step}", commit_message=f"{arm} checkpoint-{step} served export + gate")
print("PUSHED", repo, br)
PY
