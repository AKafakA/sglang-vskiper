#!/usr/bin/env bash
# Phase-A verdict at STEP (default 5000) across the two arms, by the owner's keep rule (2026-09-16):
#   holds  := every task >= stock - 0.10 AND chat empties == 0 (32 chat prompts all non-empty)
#   keep   := among holding arms, the one with the HIGHER chat skip (> 0.50 is not rejected)
#   none holds := the lower-coefficient arm continues ONLY if its chat skip is inside 0.30-0.50; else STOP.
# Prints the table + verdict; writes ${RUN_ROOT}/gates/PHASE_A_VERDICT.txt. Does not launch anything.
set -uo pipefail
STEP=${1:-5000}; ARMS=(${2:-q8b-ste-c1e5} ${3:-q8b-ste-c25e6})
readonly RUN_ROOT=/mydata/q8b
python3 - "${RUN_ROOT}" "${STEP}" "${ARMS[@]}" <<'PY' | tee "${RUN_ROOT}/gates/PHASE_A_VERDICT.txt"
import json, sys
root, step, arms = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
stock = json.load(open(f'{root}/gates/q8b-stock/gate_stepstock.json'))['tasks']
tasks = ['gsm8k', 'bbh', 'coqa_mt', 'coqa_flat']  # owner 2026-09-16: mmlu_pro/ifeval dropped, bbh added
coef = {'q8b-ste-c1e5': 1e-5, 'q8b-ste-c25e6': 2.5e-5}
rows = []
for a in arms:
    g = json.load(open(f'{root}/gates/{a}/gate_step{step}.json'))['tasks']
    c = json.load(open(f'{root}/gates/{a}/probe_step{step}-chat.json'))
    skip = c.get('chat_skip_rate') or c.get('overall_skip_rate')
    empties = sum(1 for q in c.get('prompts', []) if not (q.get('continuation') or '').strip())
    deltas = {t: (g[t]['accuracy'] - stock[t]['accuracy']) if (t in g and g[t].get('accuracy') is not None and stock.get(t, {}).get('accuracy') is not None) else None for t in tasks}
    holds = all(d is not None and d >= -0.10 for d in deltas.values()) and empties == 0
    rows.append((a, skip, empties, deltas, holds))
print(f'PHASE A @ step {step}  (stock: ' + ' '.join(f"{t}={stock[t]['accuracy']:.2f}" for t in tasks if stock.get(t,{}).get('accuracy') is not None) + ')')
print('arm            coef    chat_skip  empties  ' + '  '.join(f'{t:>9}' for t in tasks) + '  holds')
for a, skip, e, d, h in rows:
    print(f'{a:14} {coef.get(a,float("nan")):.1e}  {skip if skip is None else round(skip,3)!s:>9}  {e:>7}  ' + '  '.join(f'{("na" if d[t] is None else f"{d[t]:+.2f}"):>9}' for t in tasks) + f'  {h}')
holding = [r for r in rows if r[4] and r[1] is not None]
if holding:
    best = max(holding, key=lambda r: r[1]); print(f'VERDICT keep={best[0]} chat_skip={best[1]:.3f} (highest skip whose quality holds)')
else:
    low = min(rows, key=lambda r: coef.get(r[0], 1)); s = low[1] or 0
    if 0.30 <= s <= 0.50: print(f'VERDICT continue={low[0]} chat_skip={s:.3f} (no arm holds quality; lower coefficient is in band) -- owner informed')
    else: print(f'VERDICT STOP: no arm holds quality and the lower-coefficient arm is out of band (skip={s}) -- owner picks the next coefficient')
PY
