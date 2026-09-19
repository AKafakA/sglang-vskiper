# FlexiDepth-Qwen3 alignment training (the paper's Qwen3-4B and Qwen3-8B checkpoints)

The two Qwen3 checkpoints the paper serves are alignment-only FlexiDepth skippers (router + projector trained, base frozen,
`ste_hard` gate) on top of `Qwen/Qwen3-4B` (`1cfa9a72`) and `Qwen/Qwen3-8B` (`d117af2f`), thinking disabled through the chat
template at serving and scoring time (`NOTHINK`: `enable_thinking` defaults to false). This directory preserves the trainer and node-side
tooling that produced them; the trainer is the sealed FlexiDepth source (`train/sft_qwen3.py`, patch `debug-20260902 D1-D4`)
and the model code is the checkpoint's own `modeling_ddqwen3.py` / `configuration_ddqwen3.py` (shipped inside every export).

## Recipe (one variable moves between arms: the router penalty)

| | Qwen3-4B (paper row) | Qwen3-8B, first 7,500 steps (`q8b-ste-c25e6`) | Qwen3-8B penalty arms from step 7,500 (`c5e5` = 2e-4/4, `c1e4` = 4e-4/4; the served row = `c1e4`, step 15,000) |
|---|---|---|---|
| arm | `q4b-p6d-stehard-sealed` | `q8b-ste-c25e6` | `q8b-ste-c5e5-from7500`, `q8b-ste-c1e4-from7500` |
| init | sealed coin-flip init, routing layers 18–35, `router_gate_mode=ste_hard` (`make_variant_init.py`) | same, `FlexiDepth-Qwen3-8B-init-sealed-ste` | warm start from `q8b-ste-c25e6` checkpoint-7500 (optimizer + scheduler restored) |
| data | `flexidepth-qwen3-alignment-b14afda6-d117af2f-seq2048-r2` (`allenai/tulu-3-sft-mixture` @ `b14afda6`, seq 2048, tokenised with the 8B tokenizer) | same | same |
| `--router-penalty` / `--router-penalty-accumulation-steps` | 1e-5 / 4 (coefficient 2.5e-6) | 1e-4 / 4 (coefficient 2.5e-5) | 2e-4 / 4 (5e-5) and 4e-4 / 4 (1e-4) |
| optimiser | lr 1e-4, global batch 32 (micro 4 × grad-accum × world), seq 2048, seed 42, max_steps 27,643 (one epoch), save every 1,250 | same | same, checks every 1,250 |
| selected step | 20,000 (of 25,000 run; chat skip 0.256, all five probe tasks within 10 pp of stock) | none served (the 5,000 / 7,500 checkpoints, chat skip 0.383 / 0.428, were the warm start) | **`q8b-ste-c1e4-from7500` step 15,000 is the served checkpoint** (chat probe skip 0.520; native composite GSM8K 70.43 vs base 80.97; served knee cell 75.06 vs upstream 80.59; served decode skip 0.42). Appendix K of the paper also reports `q8b-ste-c5e5-from7500` at steps 10,000 (probe skip 0.445; native 75.21; served 76.42 vs 80.52) and 18,750 (probe skip 0.485; native 78.09; served 80.21 vs 80.59). Both arms ran to step 20,000 with a probe every 1,250 steps (`GATELINES.txt` in the data pack's `probes/`); the owner picked the served step from the three measured candidates |
| hardware | 4-GPU node, 4× A100-SXM4-40GB (Aug 27 – Sep 3, 2026 allocation) | same 4-GPU node, world 2 (phase A, two arms side by side) → world 4 from checkpoint-5000 | same node, world 2 per arm, Sep 16–18, 2026, to step 20,000 |
| rate | — | ≈ 625–775 steps/h at world 2, ≈ 1,250 steps/h at world 4 | ≈ 700 steps/h per arm |

`run_arm.sh` fixes the accumulation at 4 so the effective coefficient (= penalty / accumulation) is invariant when the world size
changes (`consolidate_w4.sh` seeds a new work directory by hard links for a world change; the trainer byte-compares its sealed
`TRAINING_INPUT_CONTRACT.json`; the sealed contracts of the four arms are beside this file). `segment_chain.sh` runs train → gate → rule as a state machine
(`segment_chain_reportonly.sh`: gates report only; `segment_chain_c30.sh`: stop only on a > 30 pp collapse). `gate_node.sh`
builds the served export (`infer-step<N>`: weights hard-linked, the checkpoint's own model code, tokenizer files from the init),
runs the 100-document probe (`gate_mixed.py`, bf16), the behavioural probes, and writes the router delta (`step<N>_router.pt`,
the trained tensors only, 415 MB) plus one `GATELINE`.

The sealed contracts of all four arms are beside this file (`TRAINING_INPUT_CONTRACT.{q4b-p6d-stehard-sealed,q8b-ste-c25e6,q8b-ste-c5e5-from7500,q8b-ste-c1e4-from7500}.json`); `data/` holds the preprocessing script and cluster wrapper that built the tokenized dataset. The probe is a coarse collapse check (±9 pp at n = 100); selection uses the
full 1,319-row composite score (`vskipper/src/vskipper/experiments/run_lmeval_quality.py`, batch 16, bf16).

## Historical training launchers

The `bin/` chain launchers and `data/*.sbatch` files record the original cluster
workflow. They use deployment-specific paths, source archives and sealed inputs;
run them only after staging those inputs and adapting the host paths. The
reviewer commands below use the source bundled in this directory directly.


```bash
# one arm, world 2 on GPUs 0 1, penalty 1e-4 / accumulation 4, stop-and-gate at the listed steps
bin/segment_chain.sh q8b-ste-c25e6 /path/FlexiDepth-Qwen3-8B-init-sealed-ste 1e-4 4 2 "0 1" 29632 2500,5000,7500,final
# world change from a sealed checkpoint
bin/consolidate_w4.sh q8b-ste-c25e6 q8b-ste-c25e6-w4 7500
```

Environment: `/mydata/venvs/training` (torch 2.8.0+cu128, transformers 4.57.0, trl 0.23.1; the cluster lock `528dabc4…`),
OpenMPI 4.1.2, driver 615.71.09 on the node.

## Reviewer checkpoint reproduction

The [checkpoint collection](https://anonymous-hf.com/a/afdrdkb4ckoj/) contains
four folders. Keep each folder's configuration, model code, tokenizer, probe
records, served-knee score and native score together.

| Folder | Checkpoint | Weight format |
|---|---|---|
| `qwen3-4b/` | 4B, step 20,000 | Full export |
| `qwen3-8b/` | 8B, penalty 4e-4, step 15,000 (served) | Full export |
| `qwen3-8b-2e-4-step10000/` | 8B, penalty 2e-4, step 10,000 | Router delta plus configuration/code/tokenizer |
| `qwen3-8b-2e-4-step18750/` | 8B, penalty 2e-4, step 18,750 | Router delta plus configuration/code/tokenizer |

Use a full export directly for serving. Reconstruct either delta-only checkpoint
with the frozen `Qwen/Qwen3-8B` base at the full revision recorded in its
training contract (prefix `d117af2f`). The following command runs from the
repository root; replace all asset paths with absolute paths and choose a new
output directory:

```bash
TRAINING="$PWD/vskipper/training/qwen3-flexidepth"
BASE=/absolute/path/to/pinned-Qwen3-8B
DELTA=/absolute/path/to/qwen3-8b-2e-4-step18750
EXPORT=/absolute/path/to/new-step18750-export
test ! -e "$EXPORT"
python "$TRAINING/build_export.py" \
  --base "$BASE" --delta "$DELTA/step18750_router.pt" \
  --ref "$DELTA" --out "$EXPORT"
```

For step 10,000, select its folder and `step10000_router.pt`. The builder links
the frozen base shards and adds the trained tensors as a separate shard; keep
the base and delta directories available while using the export. When a full
export of the same checkpoint is available, `--verify /absolute/path/to/export`
compares every tensor. Follow [GPU measurements](../../docs/02-run-experiments.md)
for the matched-work serving protocol and [analysis](../../docs/03-run-analysis.md)
for the frozen evidence. Numerical reproduction requires no retraining.

## Training from the bundled source

`data/prepare_qwen3_training_data.py` and `train/sft_qwen3.py` are the
source entrypoints. The four `TRAINING_INPUT_CONTRACT.*.json` files record the
measured runs' dataset/checkpoint hashes, revisions and hyperparameters.

A fresh training run needs the manifest-verified tokenizer, the sealed
FlexiDepth initialization (including its own model code and trained-tensor
layout), the alignment dataset and the training environment above. Resuming a
penalty arm additionally needs the step-7,500 optimizer/scheduler state. Served
exports and router deltas are inference assets, not optimizer-state backups.
`harness/make_variant_init.py` derives variants from an existing initialization;
it does not construct that initialization from stock Qwen weights.

To rebuild the alignment data, use the bundled entrypoint rather than the
historical private-source-archive launcher. Set the revision/hash variables to
the actual staged inputs and use a fresh output directory:

```bash
python vskipper/training/qwen3-flexidepth/data/prepare_qwen3_training_data.py \
  --stage alignment --tokenizer "$TOKENIZER" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --tokenizer-manifest-sha256 "$TOKENIZER_MANIFEST_SHA256" \
  --source-revision "$SOURCE_REVISION" \
  --source-archive-sha256 "$SOURCE_ARCHIVE_SHA256" \
  --output-dir "$NEW_DATASET" --num-proc 32 \
  --max-length 2048 --test-size 0.01 --seed 42

python vskipper/training/qwen3-flexidepth/train/sft_qwen3.py --help
```

`TOKENIZER` is an absolute tokenizer-checkpoint directory containing
`CHECKPOINT_SHA256SUMS`; `TOKENIZER_MANIFEST_SHA256` is the digest of that
manifest. Use the full tokenizer revision from the sealed contract.
`SOURCE_REVISION` and `SOURCE_ARCHIVE_SHA256` identify the source archive
actually used for preprocessing. For a rebuilt dataset, pass its new manifest
digest to the trainer instead of reusing the original run's digest.

The training CLI exposes the checkpoint/dataset manifests, penalty,
accumulation, batch/world size and resume controls used by the recorded recipe.
Retain the resulting input contract with any new run. The release verification
covers checkpoint-serving feasibility and frozen-result analysis; training
reproduction also requires the initialization and resume-state assets above.

`bin/hf_push_ckpt.sh` is a maintainer's full-export upload utility, not a
reproduction step. It requires an explicit destination in `HF_REPO_ID` and a
token in `HF_TOKEN`; it does not build or publish delta-only folders.
