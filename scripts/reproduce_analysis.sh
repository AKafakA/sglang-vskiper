#!/usr/bin/env bash
# v1.6 regeneration (2026-09-17, D-824/D-833): every headline number from the LADDER-CONTROL campaign (baseline = upstream SGLang
# with --cuda-graph-max-bs 1024, the same decode-graph ladder the fork captures; D-824). The raw cells live on the A100 node and
# CSD3 (openclaw has no room for the 13 GB mirror), so the raw-dependent analyses ran ON THE NODE (analysis_bundle_v16.sh, harness
# a5a61bf215) and this script consumes their outputs from the v1.6 data home; the report-driven steps run here as in regenerate.sh.
#   ./regenerate_v16.sh            # perf tables/macros/figures + quality tables from the block-B lm-eval scores
set -euo pipefail
# Stage 3 of the artifact: PACK = the unpacked data pack (analysis/, evidence/, identity-preserved/, qwen8b-selection/),
# VP = this repo's test/vp, OUT = where generated/ and figures/ land (the paper repo runs it with OUT = its own root).
PACK=${PACK:?set PACK to the unpacked data pack (e.g. ~/data/vskipper/v1.6)}
VP=${VP:-$(cd "$(dirname "$0")/../test/vp" && pwd)}
OUT=${OUT:-.}; mkdir -p "$OUT"; cd "$OUT"
AN=$PACK/analysis; EV=$PACK/evidence
BASE=upstream_g1024; TREAT=vskipper
KNEES=(--knee gsm8k=13 --knee bbh_cot=34 --knee coqa=25)   # g1024 knees, D-827 rule (A100-SXM4-80GB)
say(){ printf '\n=== %s ===\n' "$*"; }
mkdir -p generated figures

say "1  node-computed analyses -> generated/ (paired report over 6 reps, latency map, graph coverage, cache share, engagement, loaded shares, natural lengths, suite, ladders)"
for f in paired_report.json latency_map.json graph_coverage.json graph_coverage_macros.tex cache_share_macros.tex engagement_rows.tex engagement_macros.tex \
         loaded_shares.json loaded_shares_macros.tex natural_lengths_rows.tex natural_lengths_macros.tex suite_macros.tex ladder_rows.tex ladder_macros.tex; do
  [ -f $AN/$f ] || { echo "  MISSING $AN/$f"; exit 1; }; cp $AN/$f generated/$f; done
python3 - generated/paired_report.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); rows = r["rows"] if isinstance(r, dict) else r
n = sorted({row.get("n") for row in rows}); print("  paired report:", len(rows), "rows; n per row:", n, "; baseline", r.get("baseline"), "treatment", r.get("treatment"))
PY
printf '\\vpUpstreamBanks%s\n' true > generated/bank_policy.tex

say "2  headline rows + macros (main / tails), absolutes, latency-map figure"
python3 $VP/paper_table.py generated/paired_report.json "${KNEES[@]}" --emit --columns main --macros generated/headline_macros.tex > generated/headline_rows.tex
python3 $VP/abs_macros.py generated/headline_macros.tex generated/headline_macros_abs.tex
python3 $VP/paper_table.py generated/paired_report.json "${KNEES[@]}" --emit --columns tails > generated/tails_rows.tex
python3 $VP/absolutes_table.py generated/paired_report.json "${KNEES[@]}" --out generated/absolute_rows.tex --macros generated/absolute_macros.tex
python3 $VP/latency_map_figure.py generated/latency_map.json --pdf figures/latency_map.pdf --macros generated/latency_map_macros.tex | tail -1
grep -c '\\\\' generated/headline_rows.tex | xargs printf '  %s headline rows\n'

say "3  transfer rows: H100 500 W (Q*=14), RTX A6000, A100-40 (each from its own paired report when present)"
rm -f generated/h100_rows.tex generated/h100_macros.tex generated/a6000_rows.tex generated/a6000_macros.tex
H=$EV/h100-500w/ladder/h100_gsm8k/paired_report.upstream_g1024__vskipper.json
[ -f $H ] && python3 $VP/paper_table.py $H --knee gsm8k=14 --emit --columns main --dataset-suffix " (H100)" --macros generated/h100_macros.tex --macro-prefix vpHone > generated/h100_rows.tex && echo "  H100: $(grep -c '\\\\' generated/h100_rows.tex) rows"
A6=$AN/rtxa6000/paired_report.upstream_g128__vskipper.json   # 48 GB card: the same-ladder control is upstream_g128 (D-824 on this device)
[ -f $A6 ] && python3 $VP/paper_table.py $A6 --knee gsm8k=$(cat $AN/rtxa6000/qstar.txt) --emit --columns main --dataset-suffix " (RTX A6000)" --macros generated/a6000_macros.tex --macro-prefix vpAsix > generated/a6000_rows.tex && echo "  A6000: $(grep -c '\\\\' generated/a6000_rows.tex) rows" || echo "  A6000 row pending"

say "3b ladders of the transfer devices (rung logs pulled from the CSD3 mirror)"
LH=$EV/h100-500w/ladder/ladder-gsm8k-upstream_g1024; rm -f generated/ladder_h100_rows.tex generated/ladder_h100_macros.tex
[ -d $LH ] && python3 $VP/ladder_table.py h100_gsm8k=$LH --knee h100_gsm8k=14 --rows generated/ladder_h100_rows.tex --macros generated/ladder_h100_macros.tex | tail -1
LA=$AN/rtxa6000/ladder-gsm8k-upstream_g128; rm -f generated/ladder_a6000_rows.tex generated/ladder_a6000_macros.tex
[ -d $LA ] && python3 $VP/ladder_table.py a6000_gsm8k=$LA --knee a6000_gsm8k=$(cat $AN/rtxa6000/qstar.txt) --rows generated/ladder_a6000_rows.tex --macros generated/ladder_a6000_macros.tex | tail -1

say "3b2 kernel headroom table (tuned Triton count-GEMM vs cuBLAS, from the tile artifacts)"
python3 $VP/tile_ratio_table.py --device NVIDIA_A100=A100 --device "NVIDIA_RTX_A6000=RTX A6000" --rows generated/tile_ratio_rows.tex --macros generated/tile_ratio_macros.tex | tail -1

say "3c hardware-band appendix table (every device's band = the rule's output)"
A6K=$( [ -f $AN/rtxa6000/qstar.txt ] && cat $AN/rtxa6000/qstar.txt || echo 6 )
python3 $VP/hardware_bands_table.py --rows generated/hardware_bands_rows.tex --macros generated/hardware_bands_macros.tex --knee NVIDIA_A100=13 --knee NVIDIA_H100_HBM3=14 --knee NVIDIA_RTX_A6000=$A6K --used NVIDIA_A100,NVIDIA_H100_HBM3,NVIDIA_RTX_A6000 | tail -1

say "4  ablation (per dataset, from the node's paired reports; arms vdec_fd / vpre_binarycohort / integrated_alwaysskip vs $BASE)"
for DSK in "gsm8k:13:" "bbh_cot:34:_bbh_cot" "coqa:25:_coqa"; do IFS=: read -r ADS AK SFX <<< "$DSK"
  ABR=(); for arm in vdec_fd vpre_binarycohort integrated_alwaysskip; do f=$AN/ablation/$ADS/paired_report.${BASE}__${arm}.json; [ -f $f ] && ABR+=(--report $arm=$f); done
  rm -f generated/ablation_rows$SFX.tex generated/ablation_macros$SFX.tex; PFX=vpAbl; [ $ADS = bbh_cot ] && PFX=vpAblBbh; [ $ADS = coqa ] && PFX=vpAblCoqa
  python3 $VP/ablation_table.py --headline generated/paired_report.json "${ABR[@]}" --dataset $ADS --knee $ADS=$AK --rows generated/ablation_rows$SFX.tex --macros generated/ablation_macros$SFX.tex --macro-prefix $PFX 2>&1 | tail -1 || echo "  ablation $ADS: FAILED (see above)"
done

say "5  sweeps (Fig. 3): shared-band (b), own-band (c), ungated (a) from the node's per-arm reports"
for S in "sweep:vpSweep:sweep_macros.tex:sweep_heatmap.png" "sweep_v2:vpSweepTwo:sweep_macros_v2.tex:sweep_heatmap_v2.png" "sweep_ungated:vpSweepUng:sweep_macros_ungated.tex:sweep_heatmap_ungated.png"; do
  IFS=: read -r SD SP SM SPNG <<< "$S"; rm -rf generated/$SD; mkdir -p generated/$SD; cp $AN/$SD/paired_report.${BASE}__*.json generated/$SD/
  for f in generated/$SD/paired_report.${BASE}__*.json; do mv $f generated/$SD/paired_report.$(basename $f .json | sed "s/paired_report.${BASE}__//; s/_sharedband$//").json; done   # the heatmap keys arms as integrated_randomskip_r<R>_d<D>
  python3 $VP/sweep_heatmap.py generated/$SD --dataset gsm8k --suite gsm8k_eqw_r12p35 --headline generated/paired_report.json --png figures/$SPNG --macros generated/$SM --macro-prefix $SP 2>&1 | tail -1 || echo "  $SD: FAILED"
done

say "6  quality: block B (27 cells, lm-eval filters, per-document paired intervals) + the 2x2 C/D from the same knee cells"
[ -f $AN/loaded_all_v16.json ] && python3 $VP/loaded_quality_table.py $AN/loaded_all_v16.json --layout v16 --shares generated/loaded_shares.json --rows generated/loaded_quality_rows.tex --sweep-rows generated/loaded_quality_sweep.tex --macros generated/loaded_quality_macros.tex | tail -1 || echo "  block-B scores pending"

say "5b sweep occupancy (node-computed), the roofline prediction figure/table and the per-panel mean/TTFT macros"
python3 - $AN/sweep-occ <<'PY'
import json, sys, os
d = sys.argv[1]
def load(n): return json.load(open(os.path.join(d, f"occupancy_{n}.json")))
fixed = load("sharedband"); fix = load("sharedband_fix")
for tag in fixed:
    for arm in list(fixed[tag]):   # arms integrated_randomskip_r<R>_d<D>_sharedband -> the heatmap/prediction key
        if arm.endswith("_sharedband"): fixed[tag][arm[:-len("_sharedband")]] = fixed[tag].pop(arm)
    for arm, rec in fix.get(tag, {}).items():   # the tenth arm (r75_d75) from its own 2-arm root; keep only the sweep rate
        key = arm[:-len("_sharedband")] if arm.endswith("_sharedband") else arm
        if key not in fixed[tag]: fixed[tag][key] = {k: v for k, v in rec.items() if k.endswith("r12p35")}
json.dump({"fixed": fixed["sweep-v1"], "rule": load("ownband")["sweep-v2"], "ungated": load("ungated")["sweep-ungated"]}, open("generated/sweep_occupancy_v16.json", "w"), indent=1)
print("  occupancy: fixed", len(fixed["sweep-v1"]), "arms; rule", len(load("ownband")["sweep-v2"]), "; ungated", len(load("ungated")["sweep-ungated"]))
PY
python3 - <<'PY'
import json; o = json.load(open("generated/sweep_occupancy_v16.json"))
for tag in ("fixed", "rule", "ungated"): json.dump({tag: o[tag]}, open(f"generated/sweep_occupancy_{tag}.json", "w"), indent=1)
PY
SUITE=gsm8k_eqw_r12p35
python3 $VP/sweep_prediction.py --sweep fixed=generated/sweep --occupancy fixed=generated/sweep_occupancy_fixed.json --sweep rule=generated/sweep_v2 --occupancy rule=generated/sweep_occupancy_rule.json --sweep ungated=generated/sweep_ungated --occupancy ungated=generated/sweep_occupancy_ungated.json --suite $SUITE --png figures/sweep_prediction.png --rows generated/sweep_cells_rows.tex --macros generated/sweep_prediction_macros.tex 2>&1 | tail -1
python3 $VP/sweep_heatmap.py generated/sweep_v2 --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "E2E mean" --png figures/sweep_heatmap_v2_mean.png --macros /tmp/sweep_v2_mean.tex --macro-prefix vpSweepTwo | tail -1
python3 $VP/sweep_heatmap.py generated/sweep --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "E2E mean" --png figures/sweep_heatmap_mean.png --macros /tmp/sweep_mean.tex | tail -1
python3 $VP/sweep_heatmap.py generated/sweep_ungated --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "E2E mean" --png figures/sweep_heatmap_ungated_mean.png --macros /tmp/sweep_ung_mean.tex --macro-prefix vpSweepUng | tail -1
for M in "generated/sweep:vpSweep:sweep_ttft_macros.tex" "generated/sweep_ungated:vpSweepUng:sweep_ttft_macros_ungated.tex" "generated/sweep_v2:vpSweepTwo:sweep_ttft_macros_v2.tex"; do IFS=: read -r SD SP SM <<< "$M"
  python3 $VP/sweep_heatmap.py $SD --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "TTFT mean,TPOT mean" --png /tmp/sweep_ttft_$SP.png --macros generated/$SM --macro-prefix $SP | tail -1; done
[ -f generated/sweep_vstar.json ] && python3 $VP/vstar_macros.py generated/sweep_vstar.json --out generated/sweep_vstar_macros.tex 2>&1 | tail -1

say "5c Qwen rows (D-830, o8192 budget D-834): Qwen3-4B hybrid (3 reps, Q*=17) + shared Llama band (1 rep); Qwen3-8B hybrid (3 reps, Q*=14)"
QW=$AN/qwen; rm -f generated/qwen_serving_rows.tex generated/qwen_sharedband_rows.tex generated/qwen8b_serving_rows.tex generated/qwen_serving_macros_v16.tex
[ -f $QW/qwen4b_gsm8k_q4b/paired_report.upstream_g1024__vskipper_qwen3_4b.json ] && python3 $VP/paper_table.py $QW/qwen4b_gsm8k_q4b/paired_report.upstream_g1024__vskipper_qwen3_4b.json --knee gsm8k_q4b=17 --emit --columns main --dataset-suffix " (Qwen3-4B)" --macros /tmp/qwen4b_macros.tex --macro-prefix vpQwenFour > generated/qwen_serving_rows.tex && cat /tmp/qwen4b_macros.tex >> generated/qwen_serving_macros_v16.tex && echo "  Qwen3-4B: $(grep -c '\\\\' generated/qwen_serving_rows.tex) rows"
[ -f $QW/qwen4b_shared_gsm8k_q4b/paired_report.upstream_g1024__vskipper_qwen3_4b_sharedband.json ] && python3 $VP/paper_table.py $QW/qwen4b_shared_gsm8k_q4b/paired_report.upstream_g1024__vskipper_qwen3_4b_sharedband.json --knee gsm8k_q4b=17 --emit --columns main --dataset-suffix " (Qwen3-4B, shared Llama band)" --macros /tmp/qwen4bs_macros.tex --macro-prefix vpQwenFourShr > generated/qwen_sharedband_rows.tex && cat /tmp/qwen4bs_macros.tex >> generated/qwen_serving_macros_v16.tex && echo "  Qwen3-4B shared band: $(grep -c '\\\\' generated/qwen_sharedband_rows.tex) rows"
Q8ROOT=$QW/qwen8b_c5e5s10000_gsm8k_q8b; Q8ARM=vskipper_qwen3_8b_c5e5s10000   # v1.6.1: the ONE Qwen3-8B checkpoint (D-843 pick); the 5000/7500 roots stay in the pack as history
[ -f $Q8ROOT/paired_report.upstream_g1024__$Q8ARM.json ] && python3 $VP/paper_table.py $Q8ROOT/paired_report.upstream_g1024__$Q8ARM.json --knee gsm8k_q8b=14 --emit --columns main --dataset-suffix " (Qwen3-8B)" --macros /tmp/qwen8b_macros.tex --macro-prefix vpQwenEight > generated/qwen8b_serving_rows.tex && cat /tmp/qwen8b_macros.tex >> generated/qwen_serving_macros_v16.tex && echo "  Qwen3-8B: $(grep -c '\\\\' generated/qwen8b_serving_rows.tex) rows"

say "6e Qwen quality: block-B knee rows (qwen_quality_v16) + the 2x2 per checkpoint (A/B native: 4B from v1.5, 8B from CloudLab; C/D = block B)"
rm -f generated/qwen_blockb_rows.tex generated/qwen_blockb_macros.tex generated/qwen_quality_rows.tex generated/qwen_quality_macros.tex
QL=$AN/qwen_loaded_v16.json
if [ -f $QL ]; then
  python3 $VP/qwen_quality_v16.py --scores $QL --raw $AN/qwen_loaded_cells --ds gsm8k_q4b --knee r16p15 --arms up=upstream_g1024 hyb=vskipper_qwen3_4b alw=vskipper_qwen3_4b_alwaysroute --label "Qwen3-4B" --macro QwenFour --rows generated/qwen_blockb_rows.tex --macros generated/qwen_blockb_macros.tex | tail -1
  QS=$AN/qwen_loaded_sweep.json   # v1.6.1: the pick's block-B cells (upstream_g1024 / hybrid / always-route, one node, D-843)
  python3 $VP/qwen_quality_v16.py --scores $QS --raw $AN/qwen_loaded_cells_sweep --ds gsm8k_q8b --knee r13p3 --arms up=upstream_g1024 hyb=$Q8ARM alw=${Q8ARM}_alwaysroute --label "Qwen3-8B" --macro QwenEight --rows generated/qwen_blockb_rows.tex --macros generated/qwen_blockb_macros.tex | tail -1
  Q4=$PACK/identity-preserved/qwen-quality-ab-bf16
  # 6e2 native composite readings, one line per native run (the same rule as the served cells; never computed by hand): the 4B and 8B
  # A/B halves, plus the D-843 checkpoint-selection runs when the pack holds them (`qwen8b-selection/<label>/gsm8k`, skip shares in SKIPS).
  SEL=$PACK/qwen8b-selection; RUNS="--run Qwen3-8B-base=$SEL/base/gsm8k"; SKIPS=""
  [ -f $SEL/SKIPS.txt ] && for d in $SEL/*/gsm8k; do l=$(basename $(dirname $d)); [ "$l" = base ] && continue; RUNS="$RUNS --run $l=$d"; sk=$(grep "^$l=" $SEL/SKIPS.txt | cut -d= -f2); [ -n "$sk" ] && SKIPS="$SKIPS --skip $l=$sk"; done
  python3 $VP/native_composite.py $RUNS $SKIPS --base Qwen3-8B-base --rows generated/qwen_native_rows.tex --macros generated/qwen_native_macros.tex --macro-prefix vpNatQ --json generated/qwen_native.json | tail -n 6
  python3 $VP/paired_dod_2x2_v16.py --dataset gsm8k --arm A=$Q4/qwen_base/gsm8k --arm B=$Q4/qwen_fd/gsm8k --scores $QL --cell-c "loaded-upstream_g1024-gsm8k_q4b-r16p15/" --cell-d "loaded-vskipper_qwen3_4b_alwaysroute-gsm8k_q4b-r16p15/" --margin 1.0 --latex generated/qwen_quality_rows.tex --macros generated/qwen_quality_macros.tex --macro-prefix vpQwenFour --json generated/paired_dod_qwen4b.json | tail -1
  python3 $VP/paired_dod_2x2_v16.py --dataset gsm8k --arm A=$SEL/base/gsm8k --arm B=$SEL/c5e5-step10000/gsm8k --scores $QS --cell-c "loaded-upstream_g1024-gsm8k_q8b-r13p3/" --cell-d "loaded-${Q8ARM}_alwaysroute-gsm8k_q8b-r13p3/" --margin 1.0 --latex generated/qwen_quality_rows.tex --macros generated/qwen_quality_macros.tex --macro-prefix vpQwenEight --json generated/paired_dod_qwen8b.json | tail -1
  sed -i '1s/^[^&]*&/Qwen3-4B GSM8K \&/; 2s/^[^&]*&/Qwen3-8B GSM8K \&/' generated/qwen_quality_rows.tex   # row 1 = 4B, row 2 = 8B (the first sed used to hit both lines)
else echo "  Qwen block-B scores pending"; fi

rm -f generated/qwen8b_s7500_rows.tex   # v1.6.1: the 7,500 row is gone (one Qwen3-8B checkpoint, D-843)

say "6c2 GSM8K-only latency table, headline format, all three models (owner 16:0xZ Sep 17)"
rm -f generated/gsm8k_models_rows.tex
python3 $VP/paper_table.py generated/paired_report.json "${KNEES[@]}" --emit --columns main --exclude-datasets bbh_cot,coqa --dataset-suffix " (Llama-3-8B)" --macros /tmp/gsm8k_llama_macros.tex --macro-prefix vpGsmLlama > generated/gsm8k_models_rows.tex
cat generated/qwen_serving_rows.tex generated/qwen8b_serving_rows.tex >> generated/gsm8k_models_rows.tex 2>/dev/null; echo "  GSM8K all-models: $(grep -c '\\\\' generated/gsm8k_models_rows.tex) rows"

say "6d Pareto panels: block-B knee score (0.95 x Q*) vs mean E2E latency per served arm (D-829)"
python3 $VP/pareto_figure.py --headline generated/paired_report.json --ablation-dir $AN/ablation --scores $AN/loaded_all_v16.json --knee gsm8k=r12p35 --knee bbh_cot=r32p3 --knee coqa=r23p75 --pdf figures/pareto_knee.pdf --macros generated/pareto_macros.tex | head -1

say "6c the 2x2 gate: A/B native (v1.5 lm-eval samples, identity-preserved) + C/D = block-B knee cells, paired per document"
Q=$PACK/identity-preserved/q2x2-all; ABB=$PACK/identity-preserved/a6000-20260910-bbh-2x2
rm -f generated/quality_rows.tex generated/quality_macros.tex
for spec in "gsm8k:$Q/q2x2-gsm8k/A:$Q/q2x2-gsm8k/B:r12p35" "coqa:$Q/q2x2-coqa/A:$Q/q2x2-coqa/B:r23p75" "bbh_cot:$ABB/q2x2-bbh-raw/A:$ABB/q2x2-bbh-raw/B:r32p3"; do IFS=: read -r ds qa qb r <<< "$spec"
  python3 $VP/paired_dod_2x2_v16.py --dataset $ds --arm A=$qa --arm B=$qb --scores $AN/loaded_all_v16.json --cell-c "loaded-upstream_g1024-$ds-$r/" --cell-d "loaded-integrated_alwaysskip-$ds-$r/" --margin 1.0 --latex generated/quality_rows.tex --macros generated/quality_macros.tex --json generated/paired_dod_$ds.json | tail -1; done

# App. E.2's answer-extraction diagnostic (per-arm strict/flexible filter counts) is a property of the checkpoint's epilogue, read
# from the v1.5 lm-eval client cells (identity-preserved, paper snapshot 075d49b); only the macros the v1.6 gate does not emit are carried.
python3 - generated/quality_macros.tex $PACK/identity-preserved/quality_macros_v15.tex <<'PY'
import re, sys
have = set(re.findall(r"\\newcommand\{\\(\w+)\}", open(sys.argv[1]).read())); carried = []
for line in open(sys.argv[2]):
    m = re.match(r"\\newcommand\{\\(\w+)\}", line)
    if m and m.group(1) not in have and re.search(r"(Strict|Flex|FilterSpread)", m.group(1)): carried.append(line.rstrip())
open(sys.argv[1], "a").write("%% filter diagnostic (App. E.2), v1.5 lm-eval client cells, identity-preserved\n" + "\n".join(carried) + "\n"); print(f"  carried {len(carried)} filter-diagnostic macros from v1.5")
PY

say "6b natural lane (own-stop) table: upstream_g1024 harvests vs the served arm's natural cells (+ always-route at the knee), lm-eval scores"
NL=$AN/natural-lane; rm -f generated/natural_lane_rows.tex generated/natural_lane_macros.tex
python3 $VP/natural_lane_table.py $NL/summary.json --layout v16 --lmeval $NL/lmeval_scores.json --alwaysroute $NL/alwaysroute_summary.json --alwaysroute-lmeval $NL/lmeval_scores.json --rows generated/natural_lane_rows.tex --macros generated/natural_lane_macros.tex 2>&1 | tail -2

say "7  build"
latexmk -g -pdf -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
errs=$(grep -cE '^!|Misplaced|Undefined control' main.log || true); echo "  $(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}') pages, $errs error lines"
say "8  MECHANICAL CHECK -- the paper against the artifacts"
python3 $VP/paper_table.py generated/paired_report.json "${KNEES[@]}" --verify main.tex || echo "  (verify reports the v1.5 literals still in main.tex until the text is rewritten)"
