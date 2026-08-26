# test/vp/gates — the headline execution gates, repo-resident

Until `dev-h-harness-selfcontained` (D-307 Lane H, 2026-08-26) these lived
only on the dev box / in `vPipe-doc` artifacts — the split guide's "the
branch cannot currently reproduce its own headline gate". Logic is the
sealed 2026-08-24 gate vintage verbatim; only path resolution is
parameterized (env-overridable, box defaults preserved).

## What each gate proves

| file | proves |
|---|---|
| `box_vpcov_arm.sh <arm>` | ONE canonical arm serves, routes, and attests on the box (sm75). 32-request reachability smoke — NEVER a performance number. |
| `arms/arm_env_*.sh` | THE sealed arm postures (verbatim, CSD3-absolute paths included — the runner remaps them; do not edit). Four canonical: `vdec_fd`, `vpre_binarycohort`, `integrated_it4`, `vdec_randomskip`; plus `vdec_adaskip` (inert until the AdaSkip lane restores its `SGLANG_VP_ADASKIP_*` knobs — validation refuses them on this tree by design). |
| `route_digest_ab.sh` | OUTPUT equality between two trees: same 32 greedy requests, equal route-tape rows + identical generated-text sha256. Known instrument artifact: `tape.digest_sum_u64` differs across boots on an identical tree. |
| `route_digest_compare.py` | the comparator the A/B drives (reads `$VP_GATE_AB_OUT`). |
| `../refactor_ab/csd3_preflight.sh` / `csd3_ab_arm.sh` / `csd3_ab_chain.sh` | the CSD3 INTR A/B chain (fail-closed preflight → per-arm serve+probe → chain driver). `csd3_ab_run.sh` + the comparator already lived in `refactor_ab/`. |

## Env contract (defaults = the dev box)

`VP_GATE_WORKDIR` (`/local/scratch/tmp/wd312`) · `VP_GATE_PYTHON` (serve
venv) · `VP_GATE_MODEL` (frozen Llama-3-8B snapshot) · `VP_GATE_SUITE`
(frozen gsm8k first-100 requests) · `VP_GATE_FD_WEIGHTS` (router weights,
sha `9a1759…`) · `VP_GATE_SM_HELPER` (arch-matching conditional-graph
helper `.so`) · `VP_GATE_ARM_DIR` (defaults to `gates/arms/` in-repo) ·
`VPCOV_TRACER_DIR` (optional coverage tracer; absent ⇒ attestation-only) ·
`REFACTORED_TREE`/`FROZEN_TREE` + `VP_GATE_AB_OUT` (route-digest A/B) ·
`VPIPE_CSD3_ROOT` / `RUNDIR` (CSD3 chain). Host-staged inputs (model,
weights, suites, venv, helper) are deliberately NOT in the repo.

## Deliberately not imported (live in `vPipe-doc/codex/artifacts/vpipe-core-refactor-tools-20260823/`)

`confirm_arms.sh` (one-shot confirmation bound to three specific 08-2x
commits) · 7 exploration-era `vpre_*` postures superseded by
`vpre_binarycohort` · `rebuild_review_branches*.sh` + `single_commit_msg.txt`
(PR-stack machinery, not a serving gate) · `extract.py`/`normalize.py`
(extraction method-of-record) · `csd3_ab_run.final.sh` (an executed
instance; the generalized `refactor_ab/csd3_ab_run.sh` in-repo is NEWER —
it added the stale-rep freshness guard).

## Provenance caveat (from the rescue README)

These copies are the Aug 20–22 rescue vintage — the vintage the gate
evidence was produced with. The box was wiped and re-provisioned 08-24;
before the first repo-driven gate run, diff against any surviving box
copies (`$VP_GATE_WORKDIR/{vpcov-arms,tools}/`) and reconcile — then the
green repo-driven run supersedes this caveat.
