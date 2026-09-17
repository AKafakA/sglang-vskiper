# FlexiDepth-Qwen3 alignment training (the paper's Qwen3-4B and Qwen3-8B checkpoints)

The two Qwen3 checkpoints the paper serves are alignment-only FlexiDepth skippers (router + projector trained, base frozen,
`ste_hard` gate) on top of `Qwen/Qwen3-4B` (`1cfa9a72`) and `Qwen/Qwen3-8B` (`d117af2f`), thinking disabled through the chat
template at serving and scoring time (`NOTHINK`: `enable_thinking` defaults to false). This directory is the exact node-side
tooling that produced them; the trainer is the sealed FlexiDepth source (`train/sft_qwen3.py`, patch `debug-20260902 D1-D4`)
and the model code is the checkpoint's own `modeling_ddqwen3.py` / `configuration_ddqwen3.py` (shipped inside every export).

## Recipe (one variable moves between arms: the router penalty)

| | Qwen3-4B (paper row) | Qwen3-8B, checkpoints 5,000 / 7,500 (paper rows) | Qwen3-8B penalty arms (v1.6.1 candidates) |
|---|---|---|---|
| arm | `q4b-p6d-stehard-sealed` | `q8b-ste-c25e6` | `q8b-ste-c5e5-from7500`, `q8b-ste-c1e4-from7500` |
| init | sealed coin-flip init, routing layers 18–35, `router_gate_mode=ste_hard` (`make_variant_init.py`) | same, `FlexiDepth-Qwen3-8B-init-sealed-ste` | warm start from `q8b-ste-c25e6` checkpoint-7500 (optimizer + scheduler restored) |
| data | `flexidepth-qwen3-alignment-b14afda6-d117af2f-seq2048-r2` (`allenai/tulu-3-sft-mixture` @ `b14afda6`, seq 2048, tokenised with the 8B tokenizer) | same | same |
| `--router-penalty` / `--router-penalty-accumulation-steps` | 1e-5 / 4 (coefficient 2.5e-6) | 1e-4 / 4 (coefficient 2.5e-5) | 2e-4 / 4 (5e-5) and 4e-4 / 4 (1e-4) |
| optimiser | lr 1e-4, global batch 32 (micro 4 × grad-accum × world), seq 2048, seed 42, max_steps 27,643 (one epoch), save every 1,250 | same | same, checks every 1,250 |
| selected step | 20,000 (of 25,000 run; chat skip 0.256, all five probe tasks within 10 pp of stock) | 5,000 (chat skip 0.383) and 7,500 (0.428) | by the D-843 rule: native composite GSM8K ≥ base − 10 pp, then the served knee cell |
| hardware | CloudLab (Wisconsin) `d8545` node, 4× A100-SXM4-40GB (Aug 27 – Sep 3, 2026 allocation) | `d8545-10s10501`, world 2 (phase A, two arms side by side) → world 4 from checkpoint-5000 | same node, world 2 per arm |
| rate | — | ≈ 625–775 steps/h at world 2, ≈ 1,250 steps/h at world 4 | ≈ 700 steps/h per arm |

`run_arm.sh` fixes the accumulation at 4 so the effective coefficient (= penalty / accumulation) is invariant when the world size
changes (`consolidate_w4.sh` seeds a new work directory by hard links for a world change; the trainer byte-compares its sealed
`TRAINING_INPUT_CONTRACT.json`, an example is beside this file). `segment_chain.sh` runs train → gate → rule as a state machine
(`segment_chain_reportonly.sh`: gates report only; `segment_chain_c30.sh`: stop only on a > 30 pp collapse). `gate_node.sh`
builds the served export (`infer-step<N>`: weights hard-linked, the checkpoint's own model code, tokenizer files from the init),
runs the 100-document probe (`gate_mixed.py`, bf16), the behavioural probes, and writes the router delta (`step<N>_router.pt`,
the trained tensors only, 415 MB) plus one `GATELINE`. The probe is a coarse collapse check (±9 pp at n = 100); selection uses the
full 1,319-row composite score (`test/vp/run_lmeval_quality.py`, batch 16, bf16).

## Running it

```bash
# one arm, world 2 on GPUs 0 1, penalty 1e-4 / accumulation 4, stop-and-gate at the listed steps
bin/segment_chain.sh q8b-ste-c25e6 /path/FlexiDepth-Qwen3-8B-init-sealed-ste 1e-4 4 2 "0 1" 29632 2500,5000,7500,final
# world change from a sealed checkpoint
bin/consolidate_w4.sh q8b-ste-c25e6 q8b-ste-c25e6-w4 7500
```

Environment: `/mydata/venvs/training` (torch 2.8.0+cu128, transformers 4.57.0, trl 0.23.1; the CSD3 lock `528dabc4…`),
OpenMPI 4.1.2, driver 615.71.09 on the node. Exports of the served checkpoints are on the Hub under
`asdwb/qwen-3-8b-flexidepth/<arm>/checkpoint-<N>/` with their gate JSON.
