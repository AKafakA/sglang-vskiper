#!/usr/bin/env bash
# Stage 3: regenerate every table and figure in the paper from the data pack.
#
# No GPU, no serving host, no network. Reads $PACK (the analysis bundle) and this repo's
# test/vp generators; writes LaTeX fragments to $OUT and images to $OUT/figures.
#
#   PACK=~/vskipper-data OUT=generated bash scripts/reproduce_analysis.sh [--rescore]
#
# Fails closed: a missing input stops the run rather than leaving a half-regenerated set, which is
# how a paper ends up with one stale row.
set -euo pipefail

PACK=${PACK:?set PACK to the unpacked analysis bundle}
OUT=${OUT:-generated}
# --rescore: rebuild quality/loaded_all.json from the raw served responses (needs stock lm-eval 0.4.9.1)
RESCORE=0; for arg in "$@"; do case "$arg" in --rescore) RESCORE=1;; *) echo "unknown argument: $arg" >&2; exit 2;; esac; done
VP=${VP:-$(cd "$(dirname "$0")/../test/vp" && pwd)}
KNEES=(--knee gsm8k=11 --knee bbh_cot=25 --knee coqa=27)
mkdir -p "$OUT/figures"
say(){ printf '\n=== %s ===\n' "$*"; }

# The three pre-computed inputs: the paired analysis over the raw per-request arrays, the latency
# map, and the roofline dump from the served tree. Re-deriving these needs the FULL data tier.
for f in paired_report.json latency_map.json sweep_vstar.json; do
  [ -f "$OUT/$f" ] || cp "$PACK/precomputed/$f" "$OUT/$f"
done
for d in sweep sweep_v2 sweep_ungated; do
  [ -d "$OUT/$d" ] || cp -r "$PACK/precomputed/$d" "$OUT/$d"
done

say "1/6  headline, tails, absolutes, engagement"
python3 "$VP/paper_table.py" "$OUT/paired_report.json" "${KNEES[@]}" --emit --columns main \
  --macros "$OUT/headline_macros.tex" > "$OUT/headline_rows.tex"
python3 "$VP/abs_macros.py" "$OUT/headline_macros.tex" "$OUT/headline_macros_abs.tex"
python3 "$VP/paper_table.py" "$OUT/paired_report.json" "${KNEES[@]}" --emit --columns tails > "$OUT/tails_rows.tex"
python3 "$VP/absolutes_table.py" "$OUT/paired_report.json" "${KNEES[@]}" \
  --out "$OUT/absolute_rows.tex" --macros "$OUT/absolute_macros.tex"
python3 "$VP/engagement_table.py" "$PACK/headline" "${KNEES[@]}" \
  --rows "$OUT/engagement_rows.tex" --macros "$OUT/engagement_macros.tex"

say "2/6  companion banks, natural lengths, H100"
python3 "$VP/paper_table.py" "$PACK/headline-served-banks/paired_final_3rep_p95.json" "${KNEES[@]}" \
  --emit --columns backup --dataset-suffix " (served-arm banks)" > "$OUT/backup_rows.tex"
python3 "$VP/natural_lengths_table.py" "$PACK/banks/harvest-upstream" \
  --rows "$OUT/natural_lengths_rows.tex" --macros "$OUT/natural_lengths_macros.tex"
python3 "$VP/paper_table.py" "$PACK/h100/paired_report_fp16_n3.json" --knee gsm8k=17 \
  --emit --columns main --dataset-suffix " (H100)" > "$OUT/h100_rows.tex"

say "3/6  ablation (three datasets)"
for spec in "gsm8k:11:" "bbh_cot:25:Bbh" "coqa:27:Coqa"; do
  IFS=: read -r DS K PFX <<< "$spec"; R=()
  for a in stock vdec_fd vpre_binarycohort integrated_alwaysskip; do
    R+=(--report "$a=$PACK/ablation/$DS/paired_ablation_$a.json")
  done
  SUF=""; [ -n "$PFX" ] && SUF="_$DS"
  python3 "$VP/ablation_table.py" --headline "$OUT/paired_report.json" "${R[@]}" --dataset "$DS" --knee "$DS=$K" \
    --rows "$OUT/ablation_rows$SUF.tex" --macros "$OUT/ablation_macros$SUF.tex" \
    ${PFX:+--macro-prefix "vpAbl$PFX"}
done

say "4/6  Qwen boundary, loop attribution, natural lane, ladders, V*"
python3 "$VP/paper_table.py" "$PACK/qwen/serving/paired_report.json" --knee gsm8k=14 --emit --columns main \
  --dataset-suffix " (Qwen3-4B)" > "$OUT/qwen_serving_rows.tex"
python3 "$VP/paper_table.py" "$PACK/qwen/attest_paired_report.json" "$PACK/qwen/attest_rates_paired_report.json" \
  --knee gsm8k=14 --emit --columns main --dataset-suffix " (Qwen3-4B, always-route)" > "$OUT/qwen_alwaysroute_rows.tex"
python3 "$VP/paper_table.py" "$PACK/qwen/sharedband_paired_report.json" --knee gsm8k=14 --emit --columns main \
  --dataset-suffix " (Qwen3-4B, shared Llama band)" > "$OUT/qwen_sharedband_rows.tex"
python3 "$VP/qwen_serving_macros.py" "$PACK/qwen/serving/paired_report.json" "$PACK/qwen/attest_paired_report.json" \
  "$PACK/qwen/attest_rates_paired_report.json" --shared="$PACK/qwen/sharedband_paired_report.json" \
  > "$OUT/qwen_serving_macros.tex"
python3 "$VP/attribution_table.py" "$PACK/attribution/loopers-gsm8k-r10p45" \
  --population "$PACK/attribution/population500" \
  --rows "$OUT/attribution_rows.tex" --macros "$OUT/attribution_macros.tex"
# --alwaysroute is NOT optional: without it the table loses its third stack per knee cell, silently.
AR=$PACK/natural-lane/alwaysroute
python3 "$VP/natural_lane_table.py" "$PACK/natural-lane/summary.json" --lmeval "$PACK/natural-lane/lmeval_scores.json" \
  --alwaysroute "$AR/alwaysroute_natural_summary.json" --alwaysroute-lmeval "$AR/alwaysroute_lmeval_scores.json" \
  --rows "$OUT/natural_lane_rows.tex" --macros "$OUT/natural_lane_macros.tex"
# The served configuration scored at each load point (Table 2 block (b) + the appendix grid): the summarized
# lm-eval scores with per-document vectors, built off the serving path by score_natural_lane_lmeval.py.
# `--rescore` rebuilds that summary from the raw served responses shipped under quality-lane-raw/ (27 cells:
# upstream / hybrid / always-route x GSM8K / BBH-CoT / CoQA x three rates) against the frozen .qual suites
# in suites/; it needs lm-eval 0.4.9.1 importable (stock, unpatched) and takes a few minutes.
LOADED=$PACK/quality/loaded_all.json
if [ "$RESCORE" = 1 ]; then
  python3 "$VP/score_natural_lane_lmeval.py" --suites "$PACK/suites" --eval-split-only \
    --cells "$PACK"/quality-lane-raw/harvest-*/cell/*.qual_*_rep1.jsonl --out "$OUT/loaded_all.rescored.json"
  LOADED=$OUT/loaded_all.rescored.json
fi
python3 "$VP/f1_macros.py" "$PACK/F1" --macros "$OUT/f1_macros.tex"
python3 "$VP/loaded_shares.py" --raw "$PACK/quality-lane-raw" --macros "$OUT/loaded_shares_macros.tex" --json "$OUT/loaded_shares.json"
python3 "$VP/loaded_quality_table.py" "$LOADED" \
  --rows "$OUT/loaded_quality_rows.tex" --sweep-rows "$OUT/loaded_quality_sweep.tex" \
  --macros "$OUT/loaded_quality_macros.tex"
LAD=$PACK/ladders/upstream
python3 "$VP/ladder_table.py" "gsm8k=$LAD/ladder-gsm8k-upstream" "bbh_cot=$LAD/ladder-bbh_cot-upstream" \
  "coqa=$LAD/ladder-coqa-upstream" "${KNEES[@]}" --rows "$OUT/ladder_rows.tex" --macros "$OUT/ladder_macros.tex"
python3 "$VP/vstar_macros.py" "$OUT/sweep_vstar.json" --out "$OUT/sweep_vstar_macros.tex"

say "5/6  figures and the sweep maps"
python3 "$VP/latency_map_figure.py" "$OUT/latency_map.json" --pdf "$OUT/figures/latency_map.pdf" \
  --macros "$OUT/latency_map_macros.tex"
SW="--dataset gsm8k --suite gsm8k_eqw_r10p45 --headline $OUT/paired_report.json"
python3 "$VP/sweep_heatmap.py" "$OUT/sweep" $SW --png "$OUT/figures/sweep_heatmap.png" --macros "$OUT/sweep_macros.tex"
# the mean-only pass exists for the figure; its macros are a subset and are discarded
python3 "$VP/sweep_heatmap.py" "$OUT/sweep" $SW --metrics "E2E mean" --png "$OUT/figures/sweep_heatmap_mean.png" \
  --macros "$OUT/.sweep_macros_mean.discard"
python3 "$VP/sweep_heatmap.py" "$OUT/sweep_ungated" $SW --metrics "E2E mean" \
  --png "$OUT/figures/sweep_heatmap_ungated_mean.png" --macros "$OUT/sweep_macros_ungated.tex" --macro-prefix vpSweepUng
# ORDER MATTERS: the mean-only run writes the figure, the all-metrics run writes the macros, and the
# all-metrics output is a superset. Reversing these two silently drops the p95 macros.
python3 "$VP/sweep_heatmap.py" "$OUT/sweep_v2" $SW --metrics "E2E mean" \
  --png "$OUT/figures/sweep_heatmap_v2_mean.png" --macros "$OUT/sweep_macros_v2.tex" --macro-prefix vpSweepTwo
python3 "$VP/sweep_heatmap.py" "$OUT/sweep_v2" $SW --png "$OUT/figures/sweep_heatmap_v2.png" \
  --macros "$OUT/sweep_macros_v2.tex" --macro-prefix vpSweepTwo
python3 "$VP/sweep_prediction.py" --sweep "fixed=$OUT/sweep" --occupancy "fixed=$PACK/sweep-v1/occupancy.json" \
  --sweep "rule=$OUT/sweep_v2" --occupancy "rule=$PACK/sweep-v2/occupancy.json" \
  --sweep "ungated=$OUT/sweep_ungated" --occupancy "ungated=$PACK/sweep-ungated/occupancy.json" \
  --suite gsm8k_eqw_r10p45 --png "$OUT/figures/sweep_prediction.png" \
  --rows "$OUT/sweep_cells_rows.tex" --macros "$OUT/sweep_prediction_macros.tex"

say "6/6  quality: the 2x2 gates"
Q=${Q:-$OUT/.q2x2}
if [ ! -d "$Q/opt/vpipe/campaign" ]; then
  (cd "$PACK/raw" && sha256sum -c q2x2-all.tgz.sha256 >/dev/null)
  mkdir -p "$Q" && tar xzf "$PACK/raw/q2x2-all.tgz" -C "$Q"
fi
C=$Q/opt/vpipe/campaign; AB=$PACK/quality/a6000-20260910-bbh-2x2/campaign
rm -f "$OUT/quality_rows.tex"
python3 "$VP/paired_dod_2x2.py" --dataset gsm8k --arm "A=$C/q2x2-gsm8k/A" --arm "B=$C/q2x2-gsm8k/B" \
  --arm "C=$C/q2x2-corrected/upstream/gsm8k" --arm "D=$C/q2x2-corrected/integrated_alwaysskip/gsm8k" \
  --margin 1.0 --latex "$OUT/quality_rows.tex" --json "$OUT/paired_dod_gsm8k.json"
python3 "$VP/paired_dod_2x2.py" --dataset coqa --arm "A=$C/q2x2-coqa/A" --arm "B=$C/q2x2-coqa/B" \
  --arm "C=$C/q2x2-tokenized/upstream/coqa" --arm "D=$C/q2x2-tokenized/integrated_alwaysskip/coqa" \
  --margin 1.0 --latex "$OUT/quality_rows.tex" --json "$OUT/paired_dod_coqa.json"
python3 "$VP/paired_dod_2x2.py" --dataset bbh_cot --arm "A=$AB/q2x2-bbh-raw/A" --arm "B=$AB/q2x2-bbh-raw/B" \
  --arm "C=$C/q2x2-corrected/upstream/bbh_cot" --arm "D=$C/q2x2-corrected/integrated_alwaysskip/bbh_cot" \
  --margin 1.0 --latex "$OUT/quality_rows.tex" --json "$OUT/paired_dod_bbh_cot.json"
QQ=$PACK/qwen/quality; rm -f "$OUT/qwen_quality_rows.tex" "$OUT/qwen_quality_macros.tex"
for t in gsm8k coqa bbh_cot; do
  a=$(dirname "$(find "$QQ/qwen-quality-ab-bf16/qwen_base/$t" -name 'samples_*.jsonl' | head -1)")
  b=$(dirname "$(find "$QQ/qwen-quality-ab-bf16/qwen_fd/$t" -name 'samples_*.jsonl' | head -1)")
  python3 "$VP/paired_dod_2x2.py" --dataset "$t" --arm "A=$a" --arm "B=$b" \
    --arm "C=$QQ/qwen-quality-cd-bf16/qwen_upstream/$t/served" \
    --arm "D=$QQ/qwen-quality-cd-bf16/qwen_alwaysroute/$t/served" \
    --macro-prefix vpQwen --margin 1.0 --latex "$OUT/qwen_quality_rows.tex" \
    --macros "$OUT/qwen_quality_macros.tex" --json "$OUT/paired_dod_qwen_bf16_$t.json"
done

echo
echo "Done. $(ls "$OUT"/*.tex | wc -l) LaTeX fragments in $OUT."
# The committed generated/ is what the PDF is typeset from, so it cannot drift from the paper.
# A frozen versions/<x.y>/ snapshot can, and did: it went stale the moment a number was corrected.
PAPER=${PAPER:-../paper/vskipper}
if [ -d "$PAPER/generated" ]; then
  echo
  # Compare the LaTeX fragments only. bank_policy.tex is one line from a shell variable; the
  # figures are images; the .json files are inputs and superseded analyses, not artifacts.
  same=0; diff_n=0; missing=0
  for f in "$OUT"/*.tex; do
    b=$(basename "$f"); [ "$b" = bank_policy.tex ] && continue
    if [ ! -f "$PAPER/generated/$b" ]; then echo "  NOT IN PAPER: $b"; missing=$((missing+1))
    elif cmp -s "$f" "$PAPER/generated/$b"; then same=$((same+1))
    else echo "  DIFFERS: $b"; diff_n=$((diff_n+1)); fi
  done
  echo "  byte-identical $same | differing $diff_n | not in paper $missing"
  [ "$diff_n" = 0 ] && echo "REPRODUCED: every regenerated fragment matches $PAPER/generated"
  # Appendix M's promise, checked the other way round: every result literal typed in main.tex must be
  # backed by a generated fragment. Hand-typed literals are listed for verification against their artifact.
  python3 "$VP/check_paper_numbers.py" "$PAPER/main.tex" "$OUT"
else
  echo "Compare against the shipped set:  diff -r $OUT <paper>/generated"
fi
