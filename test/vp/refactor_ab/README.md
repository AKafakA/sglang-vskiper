# Refactor A/B — did `sglang.srt.vp` → `sglang.srt.vpipe` change anything?

Three questions, three instruments. Keep them separate; none substitutes for
another.

| question | instrument |
|---|---|
| does it still run and route? | the four canonical arms (`box_vpcov_arm.sh`) |
| does it produce the same OUTPUT? | route counts + generated-text hash |
| did it regress performance? | `perf_client.py` A/B |

## `perf_client.py`

Open-loop by construction: arrivals are Poisson at a fixed offered rate, and
that rate is the load control. There is no client concurrency cap — a
closed-loop cap makes the client throttle itself to the server's speed, hiding
exactly the latency differences a regression check looks for.

Every submitted request is awaited and counted; nothing is dropped or
tail-filtered. Greedy decode with `ignore_eos` and a fixed `max_new_tokens`, so
both arms are forced to produce the same work — which is asserted as **work
identity** before any timing is compared (GR-1a).

Prompts are sent as `input_ids`: the frozen suites carry token IDs, and sending
them as `text` is a 400.

## CSD3 A/B

`csd3_ab_run.sh` (replace `RUNDIR_PLACEHOLDER` with the staged run directory)
plus `csd3_ab_compare.py`. This pair REPLACES the former `csd3_refactor_ab.sh`,
which was removed because it could not fail: it invoked
`test/vp/cross_arm_work_gate.py`, a file that does not exist in this repository,
and swallowed the resulting non-zero status with

    ... || echo 'WORK GATE FAILED - timings NOT comparable' >> work_gate.txt

so the run logged the missing gate and exited 0. A performance comparison could
therefore be reported with its error, equal-work and saturation checks silently
absent. `csd3_ab_compare.py` implements those checks inline and exits non-zero
when any of them fails, printing NO deltas.

## Reading the results

Three gates run before any delta is printed:

1. **errors** must be zero
2. **work identity** — equal total output tokens across arms, or the timings
   are not comparable and no delta is reported
3. **saturation** — achieved must be ≥85% of offered, or the deltas measure the
   queue rather than the code

Discard a warmup run before measuring. A cold Triton JIT cache once produced a
false +34.7% TTFT regression that vanished on re-run.

## What these numbers are not

Not a headline. `box_perf_ab.sh` runs on sm75, and neither script drives the
sealed Accounting-v5 runner over a frozen QPS suite. What is valid is the
RELATIVE comparison, because the only variable between arms is the tree.
