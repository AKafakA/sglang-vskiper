# test/vp/gates — the headline execution gates, repo-resident

Until `dev-h-harness-selfcontained` (D-307 Lane H, 2026-08-26) these lived
only on the dev box / in `vPipe-doc` artifacts — the split guide's "the
branch cannot currently reproduce its own headline gate". Logic is the
**proven 2026-08-24 vintage verbatim** (`box_vpcov_arm_v2.sh` +
`gates_box.sh`, reconciled against the live box copies before import; the
five arm postures and the comparator are byte-identical across box, rescue
artifact, and this directory). Only path resolution is parameterized, with
the box defaults preserved.

## What each gate proves

| file | proves |
|---|---|
| `gates_box.sh` | THE chain, gating the tree it lives in: frozen-tree fix → helper build → pytest import gates → ladder default modes → **four-arm gate** → **route-digest output-equality A/B vs frozen** → summary. This is what "box gates GREEN" means. |
| `box_vpcov_arm.sh <arm> [tree] [out]` | ONE arm serves, routes, and attests (32-request greedy probe; server_info re-fetched post-probe — the attested counters are post-probe). NEVER a performance number. |
| `arms/arm_env_*.sh` | THE sealed arm postures (verbatim; CSD3-absolute paths are remapped by the driver — do not edit). Four canonical: `vdec_fd`, `vpre_binarycohort`, `integrated_it4`, `vdec_randomskip`; plus `vdec_adaskip` (inert until the AdaSkip lane restores its `SGLANG_VP_ADASKIP_*` knobs — validation refuses them on this tree by design). |
| `route_digest_compare.py` | the equality comparator (reads `$VP_GATE_AB_OUT`). Known instrument artifact: `tape.digest_sum_u64` differs across boots on an identical tree; the other fields are the signal. |
| `../refactor_ab/csd3_preflight.sh` / `csd3_ab_arm.sh` / `csd3_ab_chain.sh` | the CSD3 INTR A/B chain (fail-closed preflight → per-arm serve+probe → chain driver); `csd3_ab_run.sh` + `csd3_ab_compare.py` were already in `refactor_ab/`. |

Selective route-digest re-run without the full chain:
`box_vpcov_arm.sh vdec_fd <treeA> $W/rdab-out` for each tree, copy the three
result files per arm name, then `VP_GATE_AB_OUT=<dir> python3
route_digest_compare.py` — the chain's step 5 does exactly this.

## Env contract (defaults = the dev box post-08-24 layout)

`VP_GATE_WORKDIR` (`/local/scratch/tmp/wd312`) · `VP_GATE_TREE` (default:
the repo this script lives in) · `VP_GATE_FROZEN_TREE`
(`$W/tree-frozen-c3a1302668`) · `VP_GATE_PYTHON`/`VP_GATE_PIP`
(`$W/sglang-sm75/.venv/...`) · `VP_GATE_MODEL` (frozen Llama-3-8B snapshot)
· `VP_GATE_SUITE` (frozen gsm8k first-100 requests) · `VP_GATE_FD_WEIGHTS`
(router weights, sha `9a1759…`) · `VP_GATE_SM_HELPER` /
`VP_GATE_HELPER_OUT` (arch-matching conditional-graph helper) ·
`VP_GATE_NVCC` · `VP_GATE_ARM_DIR` (defaults to `gates/arms/`) ·
`VP_GATE_AB_OUT`. Host-staged inputs (model, weights, suites, venv, frozen
tree) are deliberately NOT in the repo.

## Deliberately not imported (live in `vPipe-doc/codex/artifacts/vpipe-core-refactor-tools-20260823/`)

`confirm_arms.sh` (one-shot bound to three specific 08-2x commits) · 7
exploration-era `vpre_*` postures superseded by `vpre_binarycohort` ·
`rebuild_review_branches*.sh` + `single_commit_msg.txt` (PR-stack machinery)
· `extract.py`/`normalize.py` (extraction method-of-record) · the v1 driver
and its `csd3_ab_run.final.sh` instance (the in-repo generalized
`refactor_ab/csd3_ab_run.sh` is NEWER — it added the stale-rep freshness
guard) · the coverage tracer (destroyed in the 08-24 wipe; the gates are
attestation-based since v2).
