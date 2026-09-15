# Equivalence checks (the "same design, same numbers" evidence)

Two questions the paper depends on, each answered by a script here rather than by an argument.

## 1. Is the pushed tree equivalent to the tree that produced the numbers? (T0–T4)

| | test | driver | comparison | result (2026-09-15) |
|---|---|---|---|---|
| T0 | design attestation, 37 arms | `test/vp/gates/verify_served_design.py` on both trees | byte diff of the attested design | byte-identical |
| T1 | gate suite under pytest | `pytest test/vp/gates` on both trees | identical pass/fail sets | identical |
| T2 | kernel / cohort numerical tests | `pytest test/vp` on both trees | identical failure sets | 507 passed / 14 failed on both, same 14 (pre-existing) |
| T3 | served generations, 32 greedy requests | `test/vp/gates/box_vpcov_arm.sh` (serve + probe + attest) per tree, then `ab_text.py` / `ab_c1.py` | text, token counts, finish reasons, attestation keys, first-token logprob | bitwise equal; 276 attestation keys, none differing; logprob delta 0.0 |
| T4 | paired performance, one headline cell | the campaign runner on both trees | `paired_analysis.py` delta inside the dev tree's measured range | −29.61 % vs [−33.11, −24.68] |

`ab_text.py` compares the semantic fields of two probe outputs (`EQ_DEV`, `EQ_REF` point at the two
`probe/labels.jsonl`); `ab_c1.py` is the same with the per-request table; `ab_compare.py` adds the byte-level
hashes. T3 needs `VP_GATE_PROBE_CONCURRENCY=1`: at concurrency 8 the *same tree* disagrees with itself on 5 of 32
requests, because the regime switch keys on batch-aggregate `seq_lens_sum` and batch composition varies with
arrival timing — an instrument artifact, not a tree difference. `chain_equiv_v1.sh` is the box chain that ran T3
for the projector/policy split (D-702), kept as run.

## 2. Are the two clients of the quality table equivalent? (Appendix E.3)

`test/vp/client_equivalence.py` compares lm-eval's archived responses for the always-route arm against the serving
benchmark's responses for the same arm on the same held-out documents at each load point; `regenerate.sh` emits
its macros. `partc_client_gate.sh` is the CSD3 driver that serves one arm and drives it with both clients
(`VPIPE_CSD3_ROOT` = the campaign root); its 2026-09-15 attempts died at server boot on the cluster's CUDA
toolchain and produced no numbers — the paper's numbers are the offline comparison above.
