# Stage 3 — regenerate every table and figure

From the data pack to all **16 tables and 6 figures** in the paper. **No GPU, no serving host, no network.**

This path is verified: every generated LaTeX artifact it produces is byte-identical to the one shipped with the
paper.

## Set up

```bash
unzip vskipper-analysis-bundle.zip -d $HOME/vskipper-data
export PACK=$HOME/vskipper-data          # the analysis bundle
export VP=$PWD/test/vp                   # this repo's analysis scripts
export OUT=$PWD/generated                # where the .tex fragments land
mkdir -p $OUT/figures
sha256sum -c $PACK/MANIFEST.sha256       # must pass before you trust anything below
```

Then run the whole thing:

```bash
bash scripts/reproduce_analysis.sh       # wraps every command in the table below
```

The rest of this page is what that script does, so you can run any single artifact on its own.

## What reads what

Three pre-computed JSON inputs ship with the pack because they are the output of the paired analysis over the raw
per-request arrays: `paired_report.json`, `latency_map.json` and `sweep_vstar.json`. Everything else is derived
here. If you want to re-derive those three from raw, you need the **full data** tier, not the analysis bundle.

| paper artifact | generator | reads |
|---|---|---|
| headline table | `paper_table.py --columns main` then `abs_macros.py` | `paired_report.json` |
| tails table | `paper_table.py --columns tails` | `paired_report.json` |
| absolute baseline table | `absolutes_table.py` | `paired_report.json` |
| engagement table | `engagement_table.py` | `$PACK/headline` runner logs |
| 2x2 faithfulness table | `paired_dod_2x2.py` x3 datasets | `$PACK/raw/q2x2-all.tgz`, `$PACK/quality/a6000-*-bbh-2x2` |
| first-token log-probabilities | **hand-entered**, two rows | the 2x2 sample files |
| served-arm-bank companion table | `paper_table.py --columns backup` | `$PACK/headline-served-banks/paired_final_3rep_p95.json` |
| natural generation lengths | `natural_lengths_table.py` | `$PACK/banks/harvest-upstream` |
| natural-lane stacks | `natural_lane_summarize.py` then `natural_lane_table.py` | the harvest roots, `$PACK/natural-lane` |
| loop attribution | `attribution_table.py --population` | `$PACK/attribution` |
| ablation tables (3 datasets) | `ablation_table.py --macro-prefix` | `$PACK/ablation/{gsm8k,bbh_cot,coqa}` |
| Qwen serving tables | `paper_table.py` x4 then `qwen_serving_macros.py` | `$PACK/qwen/*` |
| Qwen quality table | `paired_dod_2x2.py --macro-prefix vpQwen` | `$PACK/qwen/quality/qwen-quality-{ab,cd}-bf16` |
| H100 transfer table | `paper_table.py --knee gsm8k=17` | `$PACK/h100/paired_report_fp16_n3.json` |
| Q\* ladder table | `ladder_table.py` | `$PACK/ladders/upstream` |
| motivation figure | `make_figures.py` | `$PACK/F1/f1_*.json` (4 files) |
| architecture figure | **hand-drawn TikZ** | — |
| sweep maps (a) and (b) | `sweep_heatmap.py` x2 | `$OUT/sweep`, `$OUT/sweep_v2` |
| rule-prediction figure and per-cell table | `sweep_prediction.py` | the sweep report dirs, `$PACK/sweep-v*/occupancy.json` |
| latency map figure | `latency_map_figure.py` | `latency_map.json` |
| V\* macros | `vstar_macros.py` | `sweep_vstar.json` |

Two artifacts are **hand-entered and labelled so**: the first-token log-probability table (two rows read from the
2x2 sample files) and the architecture figure. Nothing else in the paper is typed by hand.

The motivation figure is reproducible from its four pinned JSON files but is **not re-runnable**: the script that
measured them no longer exists. That is stated as a limitation in the paper.

## Two things that will bite you

**The sweep macros are written twice.** `sweep_heatmap.py` runs over the v2 sweep once restricted to the mean
metric, for the figure, and once over all metrics, for the macros. The all-metrics run must come **second**: its
output is a superset, and reversing the order silently drops the p95 macros.

**The natural-lane table needs its always-route arguments.** Without `--alwaysroute` and `--alwaysroute-lmeval`
the table renders two stacks per cell instead of three, and no error is raised.

## Check

```bash
diff -r $OUT versions/1.5.6/generated
```

Every `.tex` fragment must be byte-identical. Two expected exceptions: `bank_policy.tex`, which is one line
written from a shell variable, and the figure images themselves, whose derived macros are compared instead.
