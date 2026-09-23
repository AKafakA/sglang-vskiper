#!/usr/bin/env bash
# Reproduce numerical outputs and data-driven plots from the frozen reviewer inputs.
set -euo pipefail
python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Exact reference reproduction uses CPython 3.12 (validated: 3.12.11). Activate the documented analysis environment.')
import numpy, matplotlib
PY
PACK=${PACK:?set PACK to the unpacked reviewer data pack}
PACK=$(cd -- "$PACK" && pwd -P)
VP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../test/vp" && pwd -P)
[ "$#" -eq 1 ] || { echo 'Usage: PACK=/path/to/pack bash reproduce_analysis.sh NEW_OUTPUT_DIRECTORY' >&2; exit 2; }
OUT=$1
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT";; esac
case "$PACK$VP$OUT" in *[[:space:]]*) echo 'Use paths without whitespace.' >&2; exit 2;; esac
[ ! -e "$OUT" ] && [ ! -L "$OUT" ] || { echo "Output path already exists: $OUT" >&2; exit 2; }
if command -v sha256sum >/dev/null 2>&1; then HASH=(sha256sum); else HASH=(shasum -a 256); fi
(cd "$PACK" && "${HASH[@]}" -c MANIFEST.sha256)
mkdir -p -- "$OUT/scratch"
exec > >(tee "$OUT/analysis.log") 2>&1
export PACK OUT PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$OUT/scratch/matplotlib"
cd -- "$OUT"
AN=$PACK/analysis; EV=$PACK/evidence
BASE=upstream_g1024; TREAT=vskipper
KNEES=(--knee gsm8k=13 --knee bbh_cot=34 --knee coqa=25)   # g1024 knees, D-827 rule (A100-SXM4-80GB)
say(){ printf '\n=== %s ===\n' "$*"; }
mkdir -p generated figures

say "0  identity-preserved generated files carried from v1.5 (App. G attribution, attested skip, client equivalence, F1 probe, sweep V*) -> generated/"
if [ -d $PACK/identity-preserved/paper-carried ]; then cp $PACK/identity-preserved/paper-carried/*.tex $PACK/identity-preserved/paper-carried/*.json generated/; echo "  carried $(ls $PACK/identity-preserved/paper-carried | grep -c -v README) files"; else echo "  (no paper-carried/ in the pack: the v1.5 identity-preserved macros must already be in generated/)"; fi

# v1.8 (D-849 add. 36): Figure 1 from its three runs (evidence/f1/rep<k>) replaces the carried one-run macros and adds the figure
if [ -d $EV/f1 ]; then say "0b Figure 1 (FLOPs saved, time lost): mean of the runs under evidence/f1"
  python3 $VP/f1_figure.py $EV/f1/rep* --pdf figures/f1_flops_vs_wallclock.pdf --macros generated/f1_macros.tex; fi

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
python3 $VP/paper_table.py generated/paired_report.json "${KNEES[@]}" --emit --columns main --macros generated/headline_macros.tex --group-rows --no-n-col --order gsm8k,bbh_cot,coqa > generated/headline_rows.tex   # v1.7 rc5: grouped by workload, n in the caption
python3 $VP/abs_macros.py generated/headline_macros.tex generated/headline_macros_abs.tex
python3 $VP/paper_table.py generated/paired_report.json "${KNEES[@]}" --emit --columns tails > generated/tails_rows.tex
python3 $VP/absolutes_table.py generated/paired_report.json "${KNEES[@]}" --out generated/absolute_rows.tex --macros generated/absolute_macros.tex
python3 $VP/latency_map_figure.py generated/latency_map.json --pdf figures/latency_map.pdf --macros generated/latency_map_macros.tex | tail -1
grep -c '\\\\' generated/headline_rows.tex | xargs printf '  %s headline rows\n'

say "3  transfer rows: H100 500 W (Q*=14), RTX A6000, A100-40 (each from its own paired report when present)"
rm -f generated/h100_rows.tex generated/h100_macros.tex generated/a6000_rows.tex generated/a6000_macros.tex
H=$EV/h100-500w/ladder/h100_gsm8k/paired_report.upstream_g1024__vskipper.json
[ -f $H ] && python3 $VP/paper_table.py $H --knee gsm8k=14 --emit --columns main --dataset-label "H100" --macros generated/h100_macros.tex --macro-prefix vpHone > generated/h100_rows.tex && echo "  H100: $(grep -c '\\\\' generated/h100_rows.tex) rows"
# 48 GB card: v1.8 measures it on the 256-row ladder with the same-ladder baseline upstream_g256 (D-849 add. 42); v1.7 used upstream_g128
A6=$AN/rtxa6000/paired_report.upstream_g256__vskipper.json; [ -f $A6 ] || A6=$AN/rtxa6000/paired_report.upstream_g128__vskipper.json
[ -f $A6 ] && python3 $VP/paper_table.py $A6 --knee gsm8k=$(cat $AN/rtxa6000/qstar.txt) --emit --columns main --dataset-label "RTX A6000" --macros generated/a6000_macros.tex --macro-prefix vpAsix > generated/a6000_rows.tex && echo "  A6000: $(grep -c '\\\\' generated/a6000_rows.tex) rows" || echo "  A6000 row pending"

say "3b ladders of the transfer devices (preserved rung logs)"
LH=$EV/h100-500w/ladder/ladder-gsm8k-upstream_g1024; rm -f generated/ladder_h100_rows.tex generated/ladder_h100_macros.tex
[ -d $LH ] && python3 $VP/ladder_table.py h100_gsm8k=$LH --knee h100_gsm8k=14 --rows generated/ladder_h100_rows.tex --macros generated/ladder_h100_macros.tex | tail -1
# v1.7 body: the compact transfer table (means only), H100 then RTX A6000, without the n column
rm -f generated/transfer_body_rows.tex
[ -f $H ] && python3 $VP/paper_table.py $H --knee gsm8k=14 --emit --columns means --dataset-label "H100" --macros ./scratch/transfer_h.tex --no-n-col >> generated/transfer_body_rows.tex
[ -f $A6 ] && python3 $VP/paper_table.py $A6 --knee gsm8k=$(cat $AN/rtxa6000/qstar.txt) --emit --columns means --dataset-label "RTX A6000" --macros ./scratch/transfer_a.tex --no-n-col >> generated/transfer_body_rows.tex
LA=$AN/rtxa6000/ladder-gsm8k-upstream_g128; rm -f generated/ladder_a6000_rows.tex generated/ladder_a6000_macros.tex
[ -d $LA ] && python3 $VP/ladder_table.py a6000_gsm8k=$LA --knee a6000_gsm8k=$(cat $AN/rtxa6000/qstar.txt) --rows generated/ladder_a6000_rows.tex --macros generated/ladder_a6000_macros.tex | tail -1

say "3b2 kernel headroom table (tuned Triton count-GEMM vs cuBLAS, from the tile artifacts)"
python3 $VP/tile_ratio_table.py --device NVIDIA_A100=A100 --device "NVIDIA_RTX_A6000=RTX A6000" --rows generated/tile_ratio_rows.tex --macros generated/tile_ratio_macros.tex | tail -1

say "3c hardware-band appendix table (every device's band = the rule's output)"
A6K=$( [ -f $AN/rtxa6000/qstar.txt ] && cat $AN/rtxa6000/qstar.txt || echo 6 )
python3 $VP/hardware_bands_table.py --only-used --rows generated/hardware_bands_rows.tex --macros generated/hardware_bands_macros.tex --knee NVIDIA_A100=13 --knee NVIDIA_H100_HBM3=14 --knee NVIDIA_RTX_A6000=$A6K --used NVIDIA_A100,NVIDIA_H100_HBM3,NVIDIA_RTX_A6000 | tail -1

say "4  ablation (per dataset, from the node's paired reports; arms vdec_fd / vpre_binarycohort / integrated_alwaysskip [/ vskipper_noupgrade, v1.8] vs $BASE)"
for DSK in "gsm8k:13:" "bbh_cot:34:_bbh_cot" "coqa:25:_coqa"; do IFS=: read -r ADS AK SFX <<< "$DSK"
  # v1.8 (owner 09-22 13:0xZ): the no-promotion arm (vskipper_noupgrade, the one v1.8 mechanism off) is a fourth column pair when its report exists
  ABR=(); for arm in vdec_fd vpre_binarycohort integrated_alwaysskip vskipper_noupgrade; do f=$AN/ablation/$ADS/paired_report.${BASE}__${arm}.json; [ -f $f ] && ABR+=(--report $arm=$f); done
  rm -f generated/ablation_rows$SFX.tex generated/ablation_macros$SFX.tex; PFX=vpAbl; [ $ADS = bbh_cot ] && PFX=vpAblBbh; [ $ADS = coqa ] && PFX=vpAblCoqa
  python3 $VP/ablation_table.py --headline generated/paired_report.json "${ABR[@]}" --dataset $ADS --knee $ADS=$AK --rows generated/ablation_rows$SFX.tex --macros generated/ablation_macros$SFX.tex --macro-prefix $PFX 2>&1 | tail -1 || echo "  ablation $ADS: FAILED (see above)"
done

say "5  sweeps (Fig. 3): shared-band (b), own-band (c), ungated (a) from the node's per-arm reports"
for S in "sweep:vpSweep:sweep_macros.tex:sweep_heatmap.png" "sweep_v2:vpSweepTwo:sweep_macros_v2.tex:sweep_heatmap_v2.png" "sweep_ungated:vpSweepUng:sweep_macros_ungated.tex:sweep_heatmap_ungated.png"; do
  IFS=: read -r SD SP SM SPNG <<< "$S"; rm -rf generated/$SD; mkdir -p generated/$SD; cp $AN/$SD/paired_report.${BASE}__*.json generated/$SD/
  for f in generated/$SD/paired_report.${BASE}__*.json; do mv $f generated/$SD/paired_report.$(basename $f .json | sed "s/paired_report.${BASE}__//; s/_sharedband$//").json; done   # the heatmap keys arms as integrated_randomskip_r<R>_d<D>
  python3 $VP/sweep_heatmap.py generated/$SD --dataset gsm8k --suite gsm8k_eqw_r12p35 --headline generated/paired_report.json --png figures/$SPNG --macros generated/$SM --macro-prefix $SP 2>&1 | tail -1 || echo "  $SD: FAILED"   # macros for mean + p95; the paper's App. H figure is the p95 panel below
done

say "6  quality: block B (27 cells, lm-eval filters, per-document paired intervals) + the 2x2 C/D from the same knee cells"
[ -f $AN/loaded_all_v16.json ] && python3 $VP/loaded_quality_table.py $AN/loaded_all_v16.json --layout v16 --no-filter-col --shares generated/loaded_shares.json --rows generated/loaded_quality_rows.tex --sweep-rows generated/loaded_quality_sweep.tex --macros generated/loaded_quality_macros.tex | tail -1 || echo "  block-B scores pending"

say "5b sweep occupancy (node-computed), the roofline prediction figure/table and the per-panel mean/TTFT macros"
python3 - $AN/sweep-occ <<'PY'
import json, sys, os
d = sys.argv[1]
def load(n): return json.load(open(os.path.join(d, f"occupancy_{n}.json")))
fixed = load("sharedband")
# v1.7 ran the tenth shared-band arm (r75_d75) in its own 2-arm root (occupancy_sharedband_fix.json); v1.8 runs all ten in the chains
fix = load("sharedband_fix") if os.path.exists(os.path.join(d, "occupancy_sharedband_fix.json")) else {}
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
python3 $VP/sweep_prediction.py --no-occupancy-bars --sweep fixed=generated/sweep --occupancy fixed=generated/sweep_occupancy_fixed.json --sweep rule=generated/sweep_v2 --occupancy rule=generated/sweep_occupancy_rule.json --sweep ungated=generated/sweep_ungated --occupancy ungated=generated/sweep_occupancy_ungated.json --suite $SUITE --png figures/sweep_prediction.png --rows generated/sweep_cells_rows.tex --macros generated/sweep_prediction_macros.tex 2>&1 | tail -1
python3 $VP/sweep_heatmap.py generated/sweep_v2 --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "E2E mean" --title "per-policy band" --png figures/sweep_heatmap_v2_mean.png --macros ./scratch/sweep_v2_mean.tex --macro-prefix vpSweepTwo | tail -1
python3 $VP/sweep_heatmap.py generated/sweep --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "E2E mean" --title "shared band" --png figures/sweep_heatmap_mean.png --macros ./scratch/sweep_mean.tex | tail -1
python3 $VP/sweep_heatmap.py generated/sweep_ungated --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "E2E mean" --title "always route" --png figures/sweep_heatmap_ungated_mean.png --macros ./scratch/sweep_ung_mean.tex --macro-prefix vpSweepUng | tail -1
python3 $VP/sweep_heatmap.py generated/sweep --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "E2E p95" --title "shared band, p95" --png figures/sweep_heatmap_p95.png --macros ./scratch/sweep_p95.tex | tail -1   # App. H: the p95 companion of the shared-band map
python3 $VP/sweep_heatmap.py generated/sweep_ungated --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "E2E p95" --title "always route, p95" --png figures/sweep_heatmap_ungated_p95.png --macros ./scratch/sweep_ung_p95.tex | tail -1   # App. I three-panel p95 (rc6)
python3 $VP/sweep_heatmap.py generated/sweep_v2 --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "E2E p95" --title "per-policy band, p95" --png figures/sweep_heatmap_v2_p95.png --macros ./scratch/sweep_v2_p95.tex --macro-prefix vpSweepTwo | tail -1
for M in "generated/sweep:vpSweep:sweep_ttft_macros.tex" "generated/sweep_ungated:vpSweepUng:sweep_ttft_macros_ungated.tex" "generated/sweep_v2:vpSweepTwo:sweep_ttft_macros_v2.tex"; do IFS=: read -r SD SP SM <<< "$M"
  python3 $VP/sweep_heatmap.py $SD --dataset gsm8k --suite $SUITE --headline generated/paired_report.json --metrics "TTFT mean,TPOT mean" --png ./scratch/sweep_ttft_$SP.png --macros generated/$SM --macro-prefix $SP | tail -1; done
[ -f generated/sweep_vstar.json ] && python3 $VP/vstar_macros.py generated/sweep_vstar.json --out generated/sweep_vstar_macros.tex 2>&1 | tail -1

say "5c Qwen rows (D-830, o8192 budget D-834): Qwen3-4B hybrid (3 reps, Q*=17) + shared Llama band (1 rep); Qwen3-8B, three checkpoints (3 reps each, Q*=14)"
QW=$AN/qwen; rm -f generated/qwen_serving_rows.tex generated/qwen_sharedband_rows.tex generated/qwen8b_serving_rows.tex generated/qwen8b_alt_serving_rows.tex generated/qwen8b_old_serving_rows.tex generated/qwen_serving_macros_v16.tex
[ -f $QW/qwen4b_gsm8k_q4b/paired_report.upstream_g1024__vskipper_qwen3_4b.json ] && python3 $VP/paper_table.py $QW/qwen4b_gsm8k_q4b/paired_report.upstream_g1024__vskipper_qwen3_4b.json --knee gsm8k_q4b=17 --emit --columns main --dataset-label "Qwen3-4B" --macros ./scratch/qwen4b_macros.tex --macro-prefix vpQwenFour > generated/qwen_serving_rows.tex && cat ./scratch/qwen4b_macros.tex >> generated/qwen_serving_macros_v16.tex && echo "  Qwen3-4B: $(grep -c '\\\\' generated/qwen_serving_rows.tex) rows"
[ -f $QW/qwen4b_shared_gsm8k_q4b/paired_report.upstream_g1024__vskipper_qwen3_4b_sharedband.json ] && python3 $VP/paper_table.py $QW/qwen4b_shared_gsm8k_q4b/paired_report.upstream_g1024__vskipper_qwen3_4b_sharedband.json --knee gsm8k_q4b=17 --emit --columns main --dataset-label "Qwen3-4B (Llama band)" --macros ./scratch/qwen4bs_macros.tex --macro-prefix vpQwenFourShr > generated/qwen_sharedband_rows.tex && cat ./scratch/qwen4bs_macros.tex >> generated/qwen_serving_macros_v16.tex && echo "  Qwen3-4B shared band: $(grep -c '\\\\' generated/qwen_sharedband_rows.tex) rows"
# v1.7 (D-847): the served Qwen3-8B checkpoint is c1e4s15000 (penalty 4e-4/4, step 15,000). App. K also reports the 2e-4/4 arm at
# steps 10,000 and 18,750. The label is display-only (macro names come from the dataset key, gsm8k_q8b -> QwenEight); the macro
# stems are vpQwenEight (served) / vpQwenEightAlt (2e-4, 18,750) / vpQwenEightOld (2e-4, 10,000). Labels carry no comma: the
# verify step's --ignore-datasets list is comma-separated.
Q8ROOT=$QW/qwen8b_c1e4s15000_gsm8k_q8b;  Q8ARM=vskipper_qwen3_8b_c1e4s15000;  Q8LABEL='Qwen3-8B $4{\times}10^{-4}$ 15k (served)'
Q8AROOT=$QW/qwen8b_c5e5s18750_gsm8k_q8b; Q8AARM=vskipper_qwen3_8b_c5e5s18750; Q8ALABEL='Qwen3-8B $2{\times}10^{-4}$ 18.75k'
Q8OROOT=$QW/qwen8b_c5e5s10000_gsm8k_q8b; Q8OARM=vskipper_qwen3_8b_c5e5s10000; Q8OLABEL='Qwen3-8B $2{\times}10^{-4}$ 10k'
q8rows(){ # root arm label macro-stem rows-file
  [ -f $1/paired_report.upstream_g1024__$2.json ] || { echo "  MISSING $1/paired_report.upstream_g1024__$2.json"; return 1; }
  python3 $VP/paper_table.py $1/paired_report.upstream_g1024__$2.json --knee gsm8k_q8b=14 --emit --columns main --dataset-label "$3" --macros ./scratch/$5.macros.tex --macro-prefix $4 > generated/$5.tex && cat ./scratch/$5.macros.tex >> generated/qwen_serving_macros_v16.tex && echo "  $3: $(grep -c '\\\\' generated/$5.tex) rows"; }
q8rows $Q8ROOT $Q8ARM "$Q8LABEL" vpQwenEight qwen8b_serving_rows
q8rows $Q8AROOT $Q8AARM "$Q8ALABEL" vpQwenEightAlt qwen8b_alt_serving_rows
q8rows $Q8OROOT $Q8OARM "$Q8OLABEL" vpQwenEightOld qwen8b_old_serving_rows

say "6e Qwen quality: block-B knee rows (qwen_quality_v16) + the 2x2 per checkpoint (A/B native: 4B from v1.5, 8B from the node's lm-eval runs; C/D: v1.8 lm-eval client runs, v1.7 block B)"
rm -f generated/qwen_blockb_rows.tex generated/qwen_blockb_macros.tex generated/qwen_quality_rows.tex generated/qwen_quality_macros.tex
QL=$AN/qwen_loaded_v16.json
if [ -f $QL ]; then
  python3 $VP/qwen_quality_v16.py --scores $QL --raw $AN/qwen_loaded_cells --ds gsm8k_q4b --knee r16p15 --arms up=upstream_g1024 hyb=vskipper_qwen3_4b alw=vskipper_qwen3_4b_alwaysroute --label "Qwen3-4B" --macro QwenFour --rows generated/qwen_blockb_rows.tex --macros generated/qwen_blockb_macros.tex | tail -1
  # Qwen3-8B block-B cells: the served and the 18,750 checkpoint share one node run (qwen_loaded_q8b.json, 2026-09-19, with its
  # upstream cell); the 10,000 checkpoint has its own run and upstream cell (qwen_loaded_q8b_step10000.json, 2026-09-17).
  QS=$AN/qwen_loaded_q8b.json; QSO=$AN/qwen_loaded_q8b_step10000.json; RAWO=$AN/qwen_loaded_cells_q8b_step10000
  # v1.8: all three checkpoints' block-B cells come from one node run -> one score file and one cells dir
  [ -f $QSO ] || QSO=$QS; [ -d $RAWO ] || RAWO=$AN/qwen_loaded_cells_q8b
  python3 $VP/qwen_quality_v16.py --scores $QSO --raw $RAWO --ds gsm8k_q8b --knee r13p3 --arms up=upstream_g1024 hyb=$Q8OARM alw=${Q8OARM}_alwaysroute --label "$Q8OLABEL" --macro QwenEightOld --rows generated/qwen_blockb_rows.tex --macros generated/qwen_blockb_macros.tex | tail -1
  python3 $VP/qwen_quality_v16.py --scores $QS --raw $AN/qwen_loaded_cells_q8b --ds gsm8k_q8b --knee r13p3 --arms up=upstream_g1024 hyb=$Q8AARM alw=${Q8AARM}_alwaysroute --label "$Q8ALABEL" --macro QwenEightAlt --rows generated/qwen_blockb_rows.tex --macros generated/qwen_blockb_macros.tex | tail -1
  python3 $VP/qwen_quality_v16.py --scores $QS --raw $AN/qwen_loaded_cells_q8b --ds gsm8k_q8b --knee r13p3 --arms up=upstream_g1024 hyb=$Q8ARM alw=${Q8ARM}_alwaysroute --label "$Q8LABEL" --macro QwenEight --rows generated/qwen_blockb_rows.tex --macros generated/qwen_blockb_macros.tex | tail -1
  Q4=$PACK/identity-preserved/qwen-quality-ab-bf16
  # 6e2 native composite readings, one line per native run (the same rule as the served cells; never computed by hand): the Qwen3-8B
  # base and the three reported checkpoints (`qwen8b-native/<label>/gsm8k`; SKIPS.txt = the training-time chat-template probe skip).
  # Each checkpoint also gets its SERVED knee reading (block B, the same scorer) beside the native one: the label `c<coef>-step<N>`
  # maps to the fork arm `vskipper_qwen3_8b_c<coef>s<N>`; the base maps to the upstream cell. Macro stems spell the digits out.
  SEL=$PACK/qwen8b-native; RUNS="--run Qwen3-8B-base=$SEL/base/gsm8k --served Qwen3-8B-base=$QS:loaded-upstream_g1024-gsm8k_q8b-r13p3/ --macro-key Qwen3-8B-base=Base"; SKIPS=""
  [ -f $SEL/SKIPS.txt ] && for d in $SEL/*/gsm8k; do l=$(basename $(dirname $d)); [ "$l" = base ] && continue; RUNS="$RUNS --run $l=$d"
    arm=vskipper_qwen3_8b_$(echo $l | sed 's/-step/s/'); for sf in $QS $QSO; do python3 -c "import json,sys; d=json.load(open('$sf'))['__per_row__']; sys.exit(0 if any('loaded-$arm-gsm8k_q8b-r13p3/' in k for k in d) else 1)" && { RUNS="$RUNS --served $l=$sf:loaded-$arm-gsm8k_q8b-r13p3/"; break; }; done
    key=$(echo $l | sed 's/c1e4/CoefOneEFour/; s/c5e5/CoefFiveEFive/; s/-step18750/StepEighteenKSevenFifty/; s/-step15000/StepFifteenK/; s/-step10000/StepTenK/; s/[^A-Za-z]//g'); RUNS="$RUNS --macro-key $l=$key"
    sk=$(grep "^$l=" $SEL/SKIPS.txt | cut -d= -f2); [ -n "$sk" ] && SKIPS="$SKIPS --skip $l=$sk"; done
  python3 $VP/native_composite.py $RUNS $SKIPS --base Qwen3-8B-base --rows generated/qwen_native_rows.tex --macros generated/qwen_native_macros.tex --macro-prefix vpNatQ --json generated/qwen_native.json | tail -n 6
  # C/D: v1.8 packs carry the served arms as lm-eval client runs (q2x2-cd/, D-849 add. 51/52, matched protocol); v1.7 packs
  # used the block-B knee cells. cdq <arm-D> <score file> <model key> prints the C/D arguments for either layout.
  CDQ=$PACK/q2x2-cd/qwen
  cdq(){ if [ -d $CDQ ]; then echo "--arm C=$CDQ/$3/upstream/gsm8k --arm D=$CDQ/$3/$1/gsm8k"
         else local r=r13p3 d=gsm8k_q8b; [ $3 = q4b ] && r=r16p15 d=gsm8k_q4b; echo "--scores $2 --cell-c loaded-upstream_g1024-$d-$r/ --cell-d loaded-$1-$d-$r/"; fi; }
  python3 $VP/paired_dod_2x2_v16.py --no-filter-col --no-gate-col --dataset gsm8k --arm A=$Q4/qwen_base/gsm8k --arm B=$Q4/qwen_fd/gsm8k $(cdq vskipper_qwen3_4b_alwaysroute $QL q4b) --margin 1.0 --latex generated/qwen_quality_rows.tex --macros generated/qwen_quality_macros.tex --macro-prefix vpQwenFour --json generated/paired_dod_qwen4b.json | tail -1
  python3 $VP/paired_dod_2x2_v16.py --no-filter-col --no-gate-col --dataset gsm8k --arm A=$SEL/base/gsm8k --arm B=$SEL/c5e5-step10000/gsm8k $(cdq ${Q8OARM}_alwaysroute $QSO q8b) --margin 1.0 --latex generated/qwen_quality_rows.tex --macros generated/qwen_quality_macros.tex --macro-prefix vpQwenEightOld --json generated/paired_dod_qwen8b_old.json | tail -1
  python3 $VP/paired_dod_2x2_v16.py --no-filter-col --no-gate-col --dataset gsm8k --arm A=$SEL/base/gsm8k --arm B=$SEL/c5e5-step18750/gsm8k $(cdq ${Q8AARM}_alwaysroute $QS q8b) --margin 1.0 --latex generated/qwen_quality_rows.tex --macros generated/qwen_quality_macros.tex --macro-prefix vpQwenEightAlt --json generated/paired_dod_qwen8b_alt.json | tail -1
  python3 $VP/paired_dod_2x2_v16.py --no-filter-col --no-gate-col --dataset gsm8k --arm A=$SEL/base/gsm8k --arm B=$SEL/c1e4-step15000/gsm8k $(cdq ${Q8ARM}_alwaysroute $QS q8b) --margin 1.0 --latex generated/qwen_quality_rows.tex --macros generated/qwen_quality_macros.tex --macro-prefix vpQwenEight --json generated/paired_dod_qwen8b.json | tail -1
  python3 - generated/qwen_quality_rows.tex "Qwen3-4B" "$Q8OLABEL" "$Q8ALABEL" "$Q8LABEL" <<'PY'
import sys, pathlib   # row i's first cell = label i (the d-o-d script writes the dataset key there); one row per call above, in order
p = pathlib.Path(sys.argv[1]); lines = p.read_text().splitlines()
assert len(lines) == len(sys.argv) - 2, (len(lines), sys.argv[2:])
p.write_text("".join(label + " &" + line.split("&", 1)[1] + "\n" for label, line in zip(sys.argv[2:], lines)))
PY
else echo "  Qwen block-B scores pending"; fi

say "6g served decode skip at each model's knee: the always-route arm's attestation on its block-B knee cell (Table 5 column, the V* sentence, App. K)"
python3 $VP/attested_skip.py --cell LlamaKnee=$AN/loaded_cells/loaded-integrated_alwaysskip-gsm8k-r12p35/server_info.after.json \
  --cell QwenFourKnee=$AN/qwen_loaded_cells/loaded-vskipper_qwen3_4b_alwaysroute-gsm8k_q4b-r16p15/server_info.after.json \
  --cell QwenEightKnee=$AN/qwen_loaded_cells_q8b/loaded-${Q8ARM}_alwaysroute-gsm8k_q8b-r13p3/server_info.after.json \
  --cell QwenEightAltKnee=$AN/qwen_loaded_cells_q8b/loaded-${Q8AARM}_alwaysroute-gsm8k_q8b-r13p3/server_info.after.json \
  --cell QwenEightOldKnee=${RAWO:-$AN/qwen_loaded_cells_q8b_step10000}/loaded-${Q8OARM}_alwaysroute-gsm8k_q8b-r13p3/server_info.after.json \
  --macros generated/served_skip_macros.tex && echo "  $(grep -c newcommand generated/served_skip_macros.tex) served-skip macros"
python3 - $PACK/probes generated/probe_skip_macros.tex <<'PY'
import json, sys, pathlib   # the 32-prompt routing probe's chat-template skip share (overall_skip_rate), the training-time selection signal (App. M)
root = pathlib.Path(sys.argv[1]); out = ["% GENERATED: chat-template skip share of the routing probe per reported Qwen3-8B checkpoint (probes/<arm>/probe_step<N>-chat.json)"]
for name, arm, step in (("QwenEight", "q8b-ste-c1e4-from7500", 15000), ("QwenEightAlt", "q8b-ste-c5e5-from7500", 18750), ("QwenEightOld", "q8b-ste-c5e5-from7500", 10000)):
    v = json.load(open(root / arm / f"probe_step{step}-chat.json"))["overall_skip_rate"]; out.append(f"\\newcommand{{\\vpProbeSkip{name}}}{{{v:.3f}}}")
pathlib.Path(sys.argv[2]).write_text("\n".join(out) + "\n"); print(f"  {len(out) - 1} probe-skip macros")
PY

say "6d Pareto panels: block-B knee score (0.95 x Q*) vs mean E2E latency per served arm (D-829)"
python3 $VP/pareto_figure.py --headline generated/paired_report.json --ablation-dir $AN/ablation --scores $AN/loaded_all_v16.json --knee gsm8k=r12p35 --knee bbh_cot=r32p3 --knee coqa=r23p75 --pdf figures/pareto_knee.pdf --macros generated/pareto_macros.tex | sed -n "1,1p"

say "6d2 cross-model figure (v1.7): GSM8K at each model's own knee, upstream vs vSkipper, three models, explicit keys"
QPK=$Q8ROOT/paired_report.upstream_g1024__$Q8ARM.json
if [ -f $QPK ] && [ -f $QS ]; then
  python3 $VP/pareto_models_figure.py \
    --point "Llama-3-8B=generated/paired_report.json:gsm8k:r12p35:$AN/loaded_all_v16.json:loaded-upstream_g1024-gsm8k-r12p35/:loaded-vskipper-gsm8k-r12p35/" \
    --point "Qwen3-4B=$QW/qwen4b_gsm8k_q4b/paired_report.upstream_g1024__vskipper_qwen3_4b.json:gsm8k_q4b:r16p15:$AN/qwen_loaded_v16.json:loaded-upstream_g1024-gsm8k_q4b-r16p15/:loaded-vskipper_qwen3_4b-gsm8k_q4b-r16p15/" \
    --point "Qwen3-8B=$QPK:gsm8k_q8b:r13p3:$QS:loaded-upstream_g1024-gsm8k_q8b-r13p3/:loaded-${Q8ARM}-gsm8k_q8b-r13p3/" \
    --macro-key "Llama-3-8B=LlamaGsm" --macro-key "Qwen3-4B=QwenFour" --macro-key "Qwen3-8B=QwenEight" \
    --pdf figures/pareto_models.pdf --macros generated/pareto_models_macros.tex | sed -n "1,1p"
else echo "  cross-model figure: Qwen3-8B inputs pending"; fi

say "6c the 2x2 gate: A/B native (v1.5 lm-eval samples, identity-preserved) + C/D served (v1.8: lm-eval client runs; v1.7: block-B knee cells), paired per document"
Q=$PACK/identity-preserved/q2x2-all; ABB=$PACK/identity-preserved/a6000-20260910-bbh-2x2; CDL=$PACK/q2x2-cd/llama
rm -f generated/quality_rows.tex generated/quality_macros.tex
for spec in "gsm8k:$Q/q2x2-gsm8k/A:$Q/q2x2-gsm8k/B:r12p35" "coqa:$Q/q2x2-coqa/A:$Q/q2x2-coqa/B:r23p75" "bbh_cot:$ABB/q2x2-bbh-raw/A:$ABB/q2x2-bbh-raw/B:r32p3"; do IFS=: read -r ds qa qb r <<< "$spec"
  if [ -d $CDL ]; then CD="--arm C=$CDL/upstream/$ds --arm D=$CDL/integrated_alwaysskip/$ds"
  else CD="--scores $AN/loaded_all_v16.json --cell-c loaded-upstream_g1024-$ds-$r/ --cell-d loaded-integrated_alwaysskip-$ds-$r/"; fi
  python3 $VP/paired_dod_2x2_v16.py --no-filter-col --no-gate-col --dataset $ds --arm A=$qa --arm B=$qb $CD --margin 1.0 --latex generated/quality_rows.tex --macros generated/quality_macros.tex --json generated/paired_dod_$ds.json | tail -1; done

# App. E.2's answer-extraction diagnostic (per-arm strict/flexible filter counts) is a property of the checkpoint's epilogue. v1.8:
# computed by paired_dod_2x2.py over the four GSM8K lm-eval arms (A/B native, C/D served); v1.7: carried from the v1.5 lm-eval client
# cells (identity-preserved, paper snapshot 075d49b). Only the macros the v1.6 gate does not emit are taken.
FSRC=$PACK/identity-preserved/quality_macros_v15.tex
if [ -d $CDL ]; then rm -f ./scratch/q2x2_filter_rows.tex ./scratch/q2x2_filter_macros.tex
  python3 $VP/paired_dod_2x2.py --dataset gsm8k --arm A=$Q/q2x2-gsm8k/A --arm B=$Q/q2x2-gsm8k/B --arm C=$CDL/upstream/gsm8k --arm D=$CDL/integrated_alwaysskip/gsm8k --margin 1.0 --latex ./scratch/q2x2_filter_rows.tex --macros ./scratch/q2x2_filter_macros.tex | tail -1
  FSRC=./scratch/q2x2_filter_macros.tex; fi
python3 - generated/quality_macros.tex $FSRC <<'PY'
import re, sys
have = set(re.findall(r"\\newcommand\{\\(\w+)\}", open(sys.argv[1]).read())); carried = []
for line in open(sys.argv[2]):
    m = re.match(r"\\newcommand\{\\(\w+)\}", line)
    if m and m.group(1) not in have and re.search(r"(Strict|Flex|FilterSpread)", m.group(1)): carried.append(line.rstrip())
src = "v1.5 lm-eval client cells, identity-preserved" if sys.argv[2].endswith("quality_macros_v15.tex") else "the four v1.8 GSM8K lm-eval arms"
open(sys.argv[1], "a").write(f"%% filter diagnostic (App. E.2), {src}\n" + "\n".join(carried) + "\n"); print(f"  took {len(carried)} filter-diagnostic macros from {src}")
PY

say "6b natural lane (own-stop) table: upstream_g1024 harvests vs the served arm's natural cells (+ always-route; v1.8: every rate + the two knee legs)"
NL=$AN/natural-lane; rm -f generated/natural_lane_rows.tex generated/natural_lane_macros.tex
# v1.8: always-route at every rate (the summary carries all nine) and the two knee legs (natural-lane/legs/, D-849 add. 65)
LEGS=""; if [ -d $NL/legs ]; then python3 $VP/natural_lane_summarize.py $NL/legs/* --out ./scratch/natural_legs_summary.json | tail -1; LEGS="--legs ./scratch/natural_legs_summary.json"; fi
python3 $VP/natural_lane_table.py --compact $NL/summary.json --layout v16 --lmeval $NL/lmeval_scores.json --alwaysroute $NL/alwaysroute_summary.json --alwaysroute-lmeval $NL/lmeval_scores.json $LEGS --rows generated/natural_lane_rows.tex --macros generated/natural_lane_macros.tex 2>&1 | tail -2

say "6f magnitude twins (\\<name>Abs) for the H100 / A6000 / Qwen row macros"
# v1.7: magnitude twins (\<name>Abs) for the H100 / A6000 / Qwen row macros, for "falls by X %" prose
cat generated/h100_macros.tex generated/a6000_macros.tex generated/qwen_serving_macros_v16.tex 2>/dev/null > ./scratch/transfer_macros_all.tex
python3 $VP/abs_macros.py ./scratch/transfer_macros_all.tex generated/transfer_macros_abs.tex


say "7  verify numerical reference outputs"
# a pack built before its reference outputs exist (the v1.8 internal pack) has no verifier yet; the reviewer pack always carries one
if [ -f "$PACK/verify_results.py" ]; then python3 "$PACK/verify_results.py" "$OUT"; else echo "  no verify_results.py in this pack: reference outputs not yet frozen"; fi
