#!/usr/bin/env python3
"""Gate: the four arms of the 2x2 ran the SAME evaluation protocol (owner, 2026-09-22: "the consistency between arm
a b vs c d? do we have any checker or gate on that? e.g the template and scores?").

The 2x2 reads x_i = (D_i - C_i) - (B_i - A_i) per document. A/B are the base model and the checkpoint under lm-eval's
in-process HF backend; C/D are the same pair SERVED (genuine upstream, always-route) with lm-eval as the API client.
The difference-of-differences only cancels the stack if everything else is identical across the four runs, and until
now nothing compared them: `paired_dod_2x2.py` matches documents and refuses a missing one, the per-arm gates check
empties, upstream identity and executed skipping -- a served arm on a different chat template, stop list, few-shot
setting or filter set would have passed every one of them (the class of D-255 / D-256: a gate is blind to what it
normalizes). This gate compares the RAW fields lm-eval records for each run; nothing is normalized away.

REFUSES (exit 1) unless, across A, B, C, D:
  * the same lm-eval version, the same task set, and per task the same `versions`, `n-shot`, effective `n-samples`,
    and `task_hashes` within each backend pair (it hashes prompt_hash; see below);
  * the same task config fields that shape a prompt, a generation or a score (doc_to_text/target, delimiters,
    num_fewshot, fewshot_config/split, generation_kwargs, filter_list, metric_list, output_type, dataset/splits,
    description, repeats) and the same command-line `gen_kwargs`;
  * the same `fewshot_as_multiturn` and `system_instruction_sha`;
  * `chat_template_sha` equal for A == C and B == D (each model on both stacks; A vs B is reported);
  * GSM8K: both published filters (`strict-match`, `flexible-extract`) in every run's filter_list and on every
    document -- the composite rule the paper's GSM8K scores use (strict where it parses, flexible otherwise);
  * the same documents, and per document the same `doc_hash` and `target_hash` in all four runs;
  * per document the same `prompt_hash` within each backend pair (A == B native, C == D served). Across backends the
    request representation differs (a rendered string vs a message list / token ids), so that share is reported only.
Reported (not gating): per-arm mean score, and per-document agreement A vs C and B vs D -- the same weights on the two
stacks, so a large gap flags a protocol mismatch before the difference-of-differences can absorb it.

usage: verify_2x2_protocol.py --dataset gsm8k --arm A=<dir> --arm B=<dir> --arm C=<dir> --arm D=<dir> [--json out]
Exit 0 = PASS, 1 = REFUSED, 2 = input error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paired_dod_2x2 import SPEC, load_arm  # noqa: E402  (the 2x2's own loader: the gate scores what the DiD scores)

CONFIG_FIELDS = ("doc_to_text", "doc_to_target", "doc_to_choice", "target_delimiter", "fewshot_delimiter", "num_fewshot",
                 "fewshot_config", "fewshot_split", "generation_kwargs", "filter_list", "metric_list", "output_type",
                 "dataset_path", "dataset_name", "training_split", "test_split", "description", "repeats")
GSM8K_FILTERS = {"strict-match", "flexible-extract"}
ARMS = ("A", "B", "C", "D")


def canon(v) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def newest_results(root: str) -> dict:
    files = sorted(glob.glob(os.path.join(root, "**", "results_*.json"), recursive=True))
    if not files:
        sys.exit(f"INPUT ERROR: no results_*.json under {root}")
    return json.load(open(files[-1]))   # timestamped names: the newest run, the one load_arm's newest samples belong to


def samples_meta(root: str, prefix: str) -> dict[tuple[str, int], dict]:
    """Per (task, doc_id): doc/target/prompt hashes and the set of filters present -- read raw, one file per task."""
    newest: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(root, "**", f"{prefix}*.jsonl"), recursive=True)):
        newest[os.path.basename(f)[len("samples_"):].rsplit("_20", 1)[0]] = f
    if not newest:
        sys.exit(f"INPUT ERROR: no {prefix}*.jsonl under {root}")
    out: dict[tuple[str, int], dict] = {}
    for task, f in newest.items():
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                m = out.setdefault((task, int(r["doc_id"])), {"doc_hash": r.get("doc_hash"), "target_hash": r.get("target_hash"),
                                                             "prompt_hash": r.get("prompt_hash"), "filters": set()})
                m["filters"].add(r.get("filter", "none"))
                for k in ("doc_hash", "target_hash", "prompt_hash"):   # every filter row of one document must agree
                    if m[k] != r.get(k):
                        m[k] = "INCONSISTENT"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SPEC), required=True)
    ap.add_argument("--arm", action="append", default=[], help="A=<dir> B=<dir> C=<dir> D=<dir> (all four)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    roots = dict(x.split("=", 1) for x in a.arm)
    if set(roots) != set(ARMS):
        print(f"INPUT ERROR: need exactly --arm A= B= C= D=, got {sorted(roots)}"); return 2
    prefix = SPEC[a.dataset][0]
    res = {k: newest_results(roots[k]) for k in ARMS}
    meta = {k: samples_meta(roots[k], prefix) for k in ARMS}
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str, gating: bool = True) -> None:
        checks.append({"check": name, "ok": bool(ok), "gating": gating, "detail": detail})
        print(f"  {'OK    ' if ok else ('REFUSE' if gating else 'NOTE  ')} {name:44s} {detail}")

    print(f"2x2 protocol gate -- {a.dataset}")
    for k in ARMS:
        print(f"  {k}: {roots[k]}")
    vers = {k: res[k].get("lm_eval_version") for k in ARMS}
    check("lm-eval version", len(set(vers.values())) == 1, canon(vers))
    tasks = {k: sorted(t for t in res[k].get("configs", {}) if f"samples_{t}_".startswith(prefix) or t.startswith(prefix[len("samples_"):].rstrip("_")))
             for k in ARMS}
    check("task set", len({canon(v) for v in tasks.values()}) == 1 and bool(tasks["A"]), canon(tasks["A"]) + ("" if len({canon(v) for v in tasks.values()}) == 1 else " vs " + canon(tasks)))
    for t in tasks["A"]:
        for field in ("versions", "n-shot"):
            vals = {k: res[k].get(field, {}).get(t) for k in ARMS}
            check(f"{t}: {field}", len({canon(v) for v in vals.values()}) == 1, canon(vals["A"]) if len({canon(v) for v in vals.values()}) == 1 else canon(vals))
        # lm-eval 0.4.9.1 task_hash = hash("".join(doc_hash + prompt_hash + target_hash per sample)) (loggers/evaluation_tracker.py):
        # it inherits prompt_hash's backend dependence (rendered string vs message list on chat tasks), so it gates WITHIN each
        # backend pair like prompt_hash; its three components are compared individually below.
        th = {k: res[k].get("task_hashes", {}).get(t) for k in ARMS}
        check(f"{t}: task_hashes A == B (native pair)", th["A"] == th["B"] and th["A"] is not None, f"{str(th['A'])[:12]} / {str(th['B'])[:12]}")
        check(f"{t}: task_hashes C == D (served pair)", th["C"] == th["D"] and th["C"] is not None, f"{str(th['C'])[:12]} / {str(th['D'])[:12]}")
        check(f"{t}: task_hashes A == C (cross-backend)", True, "equal" if th["A"] == th["C"] else "differ (inherits prompt_hash representation; not gating)", gating=False)
        eff = {k: (res[k].get("n-samples", {}).get(t) or {}).get("effective") for k in ARMS}
        check(f"{t}: n-samples effective", len(set(eff.values())) == 1, canon(eff))
        for field in CONFIG_FIELDS:
            vals = {k: res[k]["configs"].get(t, {}).get(field) for k in ARMS}
            same = len({canon(v) for v in vals.values()}) == 1
            check(f"{t}: config.{field}", same, "equal" if same else canon(vals))
        if a.dataset == "gsm8k":
            names = {k: {f.get("name") for f in (res[k]["configs"].get(t, {}).get("filter_list") or [])} for k in ARMS}
            check(f"{t}: composite filters in filter_list", all(GSM8K_FILTERS <= v for v in names.values()),
                  canon({k: sorted(x for x in v if x) for k, v in names.items()}))
    cli = {k: (res[k].get("config") or {}).get("gen_kwargs") for k in ARMS}
    check("command-line gen_kwargs", len({canon(v) for v in cli.values()}) == 1, canon(cli))
    for field in ("fewshot_as_multiturn", "system_instruction_sha"):
        vals = {k: res[k].get(field) for k in ARMS}
        check(field, len({canon(v) for v in vals.values()}) == 1, canon(vals))
    tpl = {k: res[k].get("chat_template_sha") for k in ARMS}
    check("chat_template_sha A == C (base, both stacks)", tpl["A"] == tpl["C"], f"{str(tpl['A'])[:16]} / {str(tpl['C'])[:16]}")
    check("chat_template_sha B == D (checkpoint, both stacks)", tpl["B"] == tpl["D"], f"{str(tpl['B'])[:16]} / {str(tpl['D'])[:16]}")
    check("chat_template_sha A == B (base vs checkpoint)", tpl["A"] == tpl["B"], f"{str(tpl['A'])[:16]} / {str(tpl['B'])[:16]}", gating=False)

    docs = {k: set(meta[k]) for k in ARMS}
    same_docs = all(docs[k] == docs["A"] for k in ARMS)
    check("same documents", same_docs, f"n={len(docs['A'])}" + ("" if same_docs else " " + canon({k: len(v) for k, v in docs.items()})))
    common = sorted(set.intersection(*docs.values()))
    for field in ("doc_hash", "target_hash"):
        bad = [d for d in common if len({meta[k][d][field] for k in ARMS}) != 1 or meta["A"][d][field] in (None, "INCONSISTENT")]
        check(f"per-document {field}", not bad, f"{len(common) - len(bad)}/{len(common)} equal" + (f"; first {bad[0]}" if bad else ""))
    for x, y, label in (("A", "B", "native pair"), ("C", "D", "served pair")):
        bad = [d for d in common if meta[x][d]["prompt_hash"] != meta[y][d]["prompt_hash"] or meta[x][d]["prompt_hash"] in (None, "INCONSISTENT")]
        check(f"per-document prompt_hash {x} == {y} ({label})", not bad, f"{len(common) - len(bad)}/{len(common)} equal" + (f"; first {bad[0]}" if bad else ""))
    cross = sum(meta["A"][d]["prompt_hash"] == meta["C"][d]["prompt_hash"] for d in common)
    check("per-document prompt_hash A == C (cross-backend)", True, f"{cross}/{len(common)} equal (representations differ by client; not gating)", gating=False)
    if a.dataset == "gsm8k":
        bad = {k: sum(not (GSM8K_FILTERS <= meta[k][d]["filters"]) for d in common) for k in ARMS}
        check("every document carries both GSM8K filters", not any(bad.values()), canon(bad))

    scores = {k: load_arm(roots[k], a.dataset) for k in ARMS}   # exits FATAL on a document lacking a filter, as the DiD would
    report = {}
    for k in ARMS:
        v = [scores[k][d] for d in common if d in scores[k]]
        report[f"mean_{k}"] = round(100 * sum(v) / len(v), 2) if v else None
    for x, y in (("A", "C"), ("B", "D")):
        both = [d for d in common if d in scores[x] and d in scores[y]]
        agree = sum(abs(scores[x][d] - scores[y][d]) < 1e-9 for d in both)
        report[f"agree_{x}{y}"] = round(100 * agree / len(both), 2) if both else None
    print(f"  REPORT means A/B/C/D = {report['mean_A']} / {report['mean_B']} / {report['mean_C']} / {report['mean_D']} ; "
          f"per-document agreement A~C {report['agree_AC']} %, B~D {report['agree_BD']} %  (same weights, two stacks)")

    refused = [c["check"] for c in checks if c["gating"] and not c["ok"]]
    verdict = "PASS" if not refused else f"REFUSED ({len(refused)})"
    print(f"VERDICT-2x2-PROTOCOL {a.dataset}: {verdict}")
    if a.json:
        json.dump({"dataset": a.dataset, "arms": roots, "verdict": verdict, "refused": refused, "checks": checks, "report": report},
                  open(a.json, "w"), indent=1)
    return 0 if not refused else 1


if __name__ == "__main__":
    raise SystemExit(main())
