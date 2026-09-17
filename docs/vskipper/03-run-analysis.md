# Stage 3 — regenerate every table and figure (paper v1.6.1)

From the data pack to every generated table, macro and figure in the paper. **No GPU, no serving host, no network.**
Every `.tex` fragment this produces is byte-identical to the one the PDF was typeset from.

## Set up

```bash
export PACK=$HOME/vskipper-data        # the unpacked data pack (layout below)
export VP=$PWD/test/vp                 # this repo's analysis scripts (vskipper-ref)
export OUT=$PWD/paper-out              # where generated/ and figures/ land
sha256sum -c $PACK/MANIFEST.sha256     # must pass before you trust anything below
bash scripts/reproduce_analysis.sh     # the whole thing; the paper repo's regenerate_v16.sh is a wrapper around it
```

## The pack

| directory | what | origin |
|---|---|---|
| `analysis/` | node-side analysis bundle: `paired_report.json` (headline, 6 reps), `latency_map.json`, graph coverage, cache share, engagement rows, loaded shares, natural lengths, suite macros, ladder rows; `ablation/<ds>/`, `sweep*/`, `sweep-occ/`; `loaded_all_v16.json` (27 block-B cells, per-document rows); `natural-lane/`; `qwen/<root>/paired_report.*`; `qwen_loaded_v16.json` + `qwen_loaded_cells/` (Qwen3-4B block B); `qwen_loaded_sweep.json` + `qwen_loaded_cells_sweep/` (Qwen3-8B block B); `rtxa6000/` | `analysis_bundle_v16.sh` on the campaign node + per-lane scorers |
| `evidence/h100-500w/ladder/` | the H100 ladder cells and its paired report | mirrored small evidence |
| `identity-preserved/` | inputs carried unchanged from v1.5: the Llama 2×2 native halves (`q2x2-all/`, `a6000-20260910-bbh-2x2/`), the Qwen3-4B native halves (`qwen-quality-ab-bf16/`), `quality_macros_v15.tex` (App. E.2 filter diagnostic) | v1.5 pack |
| `qwen8b-selection/` | native lm-eval runs of the Qwen3-8B base and the three candidate checkpoints (`<label>/gsm8k/samples_*.jsonl`), `SKIPS.txt` | D-843 selection on the A100 node |

Raw per-request arrays are not in the pack; they are the **full data** tier (`vast-raw-mirror/` tars with `SHA256SUMS`).

## What reads what

| paper artifact | generator | reads |
|---|---|---|
| headline table, tails, absolutes, macros | `paper_table.py --columns main/tails` then `abs_macros.py`, `absolutes_table.py` | `analysis/paired_report.json` |
| latency map figure | `latency_map_figure.py` | `analysis/latency_map.json` |
| graph coverage, cache share, engagement, loaded shares, natural lengths, suite macros, ladder rows | copied from `analysis/` (node-computed over raw) | — |
| H100 and RTX A6000 transfer tables + their ladder rows | `paper_table.py` (knee 14 / 6), `ladder_table.py` | `evidence/h100-500w/ladder/`, `analysis/rtxa6000/` |
| hardware-band appendix table | `hardware_bands_table.py` | the rule (no data input) |
| mechanism ablation tables (3 datasets) | `ablation_table.py` | `analysis/ablation/<ds>/` |
| RandomSkip sweep maps, rule-prediction figure, occupancy/V\* macros, TTFT macros | `sweep_heatmap.py`, `sweep_prediction.py`, `vstar_macros.py` | `analysis/sweep*/`, `analysis/sweep-occ/` |
| Qwen serving rows (4B, shared band, 8B) and the GSM8K all-models table | `paper_table.py` per paired report | `analysis/qwen/<root>/` |
| Qwen block-B rows | `qwen_quality_v16.py` | `analysis/qwen_loaded_{v16,sweep}.json` + cell dirs |
| Qwen 2×2 rows | `paired_dod_2x2_v16.py` | native halves (`identity-preserved/qwen-quality-ab-bf16/`, `qwen8b-selection/`) + the block-B scores |
| Qwen3-8B selection table | `native_composite.py` | `qwen8b-selection/` |
| Llama block-B quality table and 2×2 | `loaded_quality_table.py --layout v16`, `paired_dod_2x2_v16.py` | `analysis/loaded_all_v16.json`, `identity-preserved/q2x2-all/` |
| natural-lane stacks table | `natural_lane_table.py` | `analysis/natural-lane/` |
| Pareto figure | `pareto_figure.py` | `paired_report.json`, `analysis/ablation/`, `loaded_all_v16.json` |
| cross-model Pareto pairs figure (upstream → vSkipper per model at its own knee) | `pareto_models_figure.py` | the three paired reports + `loaded_all_v16.json`, `qwen_loaded_{v16,sweep}.json` |
| kernel-headroom table | `tile_ratio_table.py` | the committed tile artifacts under `python/sglang/srt/vpipe/binary_cohort_configs/` |

The GSM8K composite reading (`exact_match,marker-composite`: strict-match where the marker parses, flexible extraction otherwise)
is produced by `score_natural_lane_lmeval.py` for every served cell and by `native_composite.py` / `paired_dod_2x2.load_arm` for native
runs. No reading in the paper is computed by hand.

## Check

```bash
diff -r $OUT/generated <paper-repo>/generated
```

Every `.tex` fragment must be byte-identical; figure images are compared through their derived macros. `generated/` is the directory
the PDF is typeset from, and it is committed, so it cannot drift from the paper you are holding; the `versions/<x.y.z>/` directories are
snapshots of past builds.
