# vSkipper

vSkipper serves a **conditional-depth** language model in production. The checkpoint decides, per token and per
layer, whether to run a transformer layer or to skip it; vSkipper turns those decisions into wall-clock serving
gains instead of a FLOP count that never arrives.

It is a fork of SGLang. The payload is the released `FlexiDepth-Llama-3-8B-Instruct` checkpoint over a frozen
Llama-3-8B base, which routes the last 16 of 32 layers. The performance baseline is **upstream SGLang, served
from its own separate tree** — never this fork with the skipper switched off.

## The served design in one table

| piece | what it does |
|---|---|
| route tape | the per-token, per-layer RUN / PROJECT-ONLY decisions, resident on the device |
| cohort execution | routed rows are gathered into dense cohorts, executed, and scattered back, under a bound on the cohort *count* |
| K/V completeness (Invariant 1) | a skipped token still gets that layer's own K/V from that layer's own projection weights; K/V is never copied across layers |
| metadata rebuild (Invariant 2) | per-layer attention metadata is rebuilt for each cohort, which is what makes varying depth legal inside one batch |
| one captured graph | the whole body stays CUDA-graph-capturable across varying routes |
| engagement rule | the routed body runs only where a roofline break-even estimate says the removed K/V traffic pays for the routed body's fixed cost |

**`python/sglang/srt/vpipe/design.py` is the single source of truth** for what any named arm does. Every served
configuration is an entry there, and the runtime attests its resolved design at boot; the docs below never
restate a design value that `design.py` owns.

## Three stages

Each stage is a separate guide and ends in a check you can run yourself.

| stage | guide | you get |
|---|---|---|
| 1 | [`01-environment.md`](01-environment.md) | separate analysis and serving environments; GPU assets are staged separately |
| 2 | [`02-run-experiments.md`](02-run-experiments.md) | measured cells in a target directory, from a clean A100 |
| 3 | [`03-run-analysis.md`](03-run-analysis.md) | 120 numerical reference outputs and data-driven plots from the reviewer pack, with no serving host |

If you only want to check the paper's numbers, **go straight to stage 3.** The cells are already measured and
shipped; stage 2 exists for re-running them from scratch.

## Also here

- [`gates.md`](gates.md) — the gates that decide whether a cell counts as a measurement.
- [`removed.md`](removed.md) — historical cleanup records; dev retains the original source layout.
- [`decisions.md`](decisions.md) — the rulings the code comments refer to, restated.
