# Equivalence checks (the "same design, same numbers" evidence)

Two questions the paper depends on, each answered by a script in the tree rather than by an argument.

## 1. Is the pushed tree equivalent to the tree that produced the numbers? (T0–T4)

| | test | driver | comparison | result (2026-09-15) |
|---|---|---|---|---|
| T0 | design attestation, 37 arms | `test/vp/gates/verify_served_design.py` on both trees | byte diff of the attested design | byte-identical |
| T1 | gate suite under pytest | `pytest test/vp/gates` on both trees | identical pass/fail sets | identical |
| T2 | kernel / cohort numerical tests | `pytest test/vp` on both trees | identical failure sets | 507 passed / 14 failed on both, same 14 (pre-existing) |
| T3 | served generations, 32 greedy requests | `test/vp/gates/box_vpcov_arm.sh` (serve + probe + attest) per tree, then `ab_compare.py` | text, token counts, finish reasons, first-token logprob; post-probe attestation | text bitwise equal at probe concurrency 1 (see below) |
| T4 | paired performance, one headline cell | `test/vp/run_paired_campaign.py` on both trees | `paired_analysis.py` delta inside the dev tree's measured range | −29.61 % vs [−33.11, −24.68] |

Stable server identity (the attestation with its traffic counters stripped) is compared by
`test/vp/verify_tree_equivalence.py --spec SPEC --tree-a A --tree-b B --arm ARM ...`, which boots each arm on both
trees and exits non-zero on any differing field; the reports of its runs are `tree_equivalence.json` beside the
campaign root.

`ab_compare.py <dev out> <ref out>` is T3's comparator. Each argument is a `box_vpcov_arm.sh` output directory:
`probe/labels.jsonl` (the per-request table) and `server_info.json`, the snapshot taken **after** the probe. Exit
0 = identical, 1 = a difference, 2 = not the same requests. Boot-varying fields (ports, staging paths, pids) are
skipped by exact key segment, never by substring, and a key present on one side only is a difference.

Two things about T3's record. The probe is only deterministic at `VP_GATE_PROBE_CONCURRENCY=1`: at concurrency 8
the *same tree* disagrees with itself on 5 of 32 requests, because the regime switch keys on batch-aggregate
resident tokens and batch composition varies with arrival timing. And the 2026-09-15 attestation comparison
("276 keys, none differing") was made on the probe's *pre-traffic* snapshot, which the earlier comparator read by
mistake; it shows the two trees booted identically, not that they routed the probe identically. The
traffic-dependent evidence of T3 is the label comparison; `ab_compare.py` now reads the post-probe snapshot so a
rerun compares the routing too.

## 2. Are the two clients of the quality table equivalent? (Appendix E.3)

The quality table has two clients. lm-eval drives the unloaded 2×2 (`test/vp/run_quality_2x2.py`); under load
the same held-out documents are served through the serving benchmark on the `.qual` suites
(`test/vp/run_natural_harvest.py`, one cell per arm × dataset × rate) and scored afterwards with lm-eval's own
filters (`test/vp/score_natural_lane_lmeval.py --eval-split-only`). Both drivers boot the arm through
`run_paired_campaign.Server`, so the upstream arm is served from the upstream tree on both paths.

`test/vp/client_equivalence.py` then compares, for the always-route arm on the same documents at each load
point, lm-eval's archived responses with the benchmark's: verdict agreement per document and the paired score
difference. `regenerate.sh` / `reproduce_analysis.sh` emit its macros from the archived cells; the paper's
numbers are that offline comparison. To re-measure rather than re-score:

```
run_quality_2x2.py      --spec SPEC --out-dir OUT/lmeval --arm integrated_alwaysskip --workload gsm8k ...
run_natural_harvest.py  --spec SPEC --out-dir OUT --arm integrated_alwaysskip --dataset gsm8k \
                        --suite gsm8k.qual --rate 10.45 --label r10p45
score_natural_lane_lmeval.py --suites SUITES --eval-split-only --cells OUT/harvest-*/cell/*.qual_*_rep1.jsonl --out OUT/loaded.json
client_equivalence.py   ... (see its docstring)
```
