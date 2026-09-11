#!/usr/bin/env python3
"""Drive lm-eval for the quality lane, and GATE the result.

Why this exists. lm-eval has never been invoked from a committed file in this repo --
every quality number came from hand-written shell on an ephemeral box. That is how the
2026-09-09 quality lane spent a night measuring the no-skip body (D-627) and how a
51.6 %-empty run reported 0.0 % empty (lm-eval rewrites an empty generation as
"[invalid]", so `filtered_resps` hides it). The gates that catch both already existed and
had no callers.

Two rules this file exists to hold:

  * **lm-eval is UNMODIFIED and runs the FULL split.** `--limit` is refused outright.
    D-607, owner: *"we should not making any changes on the 3rd party evaluation harness
    so llm-eval for quality and sglang serving for the performances, that is a protection
    of our results."* Scoring is lm-eval's, never ours.
  * **The prompt protocol is READ FROM THE PERF SUITE, never chosen here.** The frozen
    suite records `prompt_kind` per request, so chat-vs-raw cannot drift between the two
    lanes -- which is a question that has already had to be re-answered by hand. gsm8k and
    bbh_cot are `chat_messages`; coqa is `raw_completion`, and forcing chat flags onto it
    would CREATE the mismatch rather than fix one.

Usage (served arm, e.g. C=stock or D=vSkipper):
  run_lmeval_quality.py --workload gsm8k --arm integrated_alwaysskip \\
      --suite-dir /dev/shm/vpipe/suites --output-dir OUT \\
      --base-url http://127.0.0.1:31910 --tokenizer /path/to/model

Usage (native PyTorch reference, e.g. A=raw base, B=released checkpoint):
  run_lmeval_quality.py --workload gsm8k --arm armB-released-fd \\
      --suite-dir /dev/shm/vpipe/suites --output-dir OUT \\
      --hf-model /path/to/checkpoint --hf-dtype float16 --trust-remote-code
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GATES = ROOT / "test/vp/gates"

# Our workload name -> the lm-eval task that defines its protocol.
# bbh uses `bbh_cot_fewshot`, NOT `bbh_fewshot`: the latter scores 0.0 through a
# leading-space artifact and is unusable as a control (D-616 + amendment).
TASKS = {
    "gsm8k": "gsm8k",
    "coqa": "coqa",
    "bbh_cot": "bbh_cot_fewshot",
}


def _fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _prompt_kind(suite_dir: Path, suite_name: str) -> str:
    """Read the protocol the PERF lane froze for this dataset.

    `suite_name` is the FILE PREFIX, which is not the workload name: the built suites are
    `coqa.d179`, `gsm8k.d179`, `bbh_cot.d179`. Defaults to the workload, overridable with
    --suite-name.

    One value for the whole suite: a suite that mixes prompt kinds has no single
    lm-eval protocol, and silently picking one would be the drift this guards against.
    """
    metadata = suite_dir / f"{suite_name}.metadata.jsonl"
    if not metadata.is_file():
        candidates = sorted(p.name.replace(".metadata.jsonl", "")
                            for p in suite_dir.glob("*.metadata.jsonl"))
        sys.exit(
            f"FATAL: no frozen suite at {metadata}. The quality protocol is read from "
            "the perf suite so the two lanes cannot drift.\n"
            f"       Suites present in {suite_dir}: {', '.join(candidates) or '(none)'}\n"
            "       Pass --suite-name with the right prefix, or build the suite first."
        )
    kinds = set()
    with metadata.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                kinds.add(str(json.loads(line).get("prompt_kind") or ""))
    if len(kinds) != 1:
        sys.exit(f"FATAL: {metadata} mixes prompt kinds {sorted(kinds)}")
    kind = kinds.pop()
    if kind not in ("chat_messages", "raw_completion"):
        sys.exit(f"FATAL: {metadata} has unknown prompt kind {kind!r}")
    return kind


def _lmeval_command(args: argparse.Namespace, task: str, kind: str) -> list[str]:
    command = [args.lmeval_python, "-m", "lm_eval"]
    if args.base_url:
        # lm-eval's chat template cannot be served over /v1/completions (D-233), so a
        # chat protocol must go to /v1/chat/completions.
        endpoint = "chat/completions" if kind == "chat_messages" else "completions"
        model = "local-chat-completions" if kind == "chat_messages" else "local-completions"
        model_args = (
            f"base_url={args.base_url.rstrip('/')}/v1/{endpoint},"
            f"model=served,tokenizer_backend=huggingface,"
            f"tokenizer={args.tokenizer},num_concurrent={args.num_concurrent},"
            f"max_retries=2,tokenized_requests=False"
        )
    else:
        model = "hf"
        model_args = f"pretrained={args.hf_model},device_map=cuda:0"
        if args.hf_dtype:
            model_args += f",dtype={args.hf_dtype}"
        if args.trust_remote_code:
            model_args += ",trust_remote_code=True"
    command += ["--model", model, "--model_args", model_args, "--tasks", task]
    if not args.base_url:
        # Served arms are batched by the server; only the in-process model needs this.
        command += ["--batch_size", str(args.batch_size)]
    if kind == "chat_messages":
        command += ["--apply_chat_template", "--fewshot_as_multiturn"]
    # --log_samples produces the samples_*.jsonl layout verify_zero_empty globs. Without
    # it there is nothing to gate, and the run is unverifiable.
    command += ["--output_path", str(args.output_dir), "--log_samples"]
    return command


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=sorted(TASKS), required=True)
    ap.add_argument("--arm", required=True, help="label recorded in the manifest")
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--suite-name", default="",
                    help="frozen-suite file prefix; defaults to --workload. "
                         "The built suites are coqa.d179 / gsm8k.d179 / bbh_cot.d179.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-url", default="", help="serving endpoint, for arms C/D")
    ap.add_argument("--tokenizer", default="", help="tokenizer path, with --base-url")
    ap.add_argument("--num-concurrent", type=int, default=16)
    ap.add_argument(
        "--upstream-baseline",
        action="store_true",
        help="The served endpoint is GENUINE upstream SGLang (a separate tree with no "
             "vpipe/ package), which is what arm C of the 2x2 must be: the gate condition "
             "reads 'no quality collapse vs UPSTREAM sglang', and arm C had always been "
             "ARMS['stock'] -- this fork with the skipper off -- so the comparison the gate "
             "names was never actually run. Inverts the attestation: vp_runtime must be "
             "ABSENT, and its presence refuses the run as a mislaunched fork.",
    )
    ap.add_argument("--hf-model", default="", help="model path, for native arms A/B")
    ap.add_argument("--hf-dtype", default="")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument(
        "--batch-size",
        default="32",
        help="lm-eval --batch_size for NATIVE arms. Default 32 (owner). Two MEASURED "
             "facts, one of which corrects this flag's original justification.\n"
             "  (1) 'auto' resolves to 7 on an 80 GiB A100. _detect_batch_size() probes at "
             "the model's MAXIMUM context (8192) and materialises (B, 8192, 128256) fp32 "
             "logits -- ~8.4 GiB per sequence -- so it stops at 7 no matter how short the "
             "real prompts are. It is a logits-buffer bound, not a KV bound.\n"
             "  (2) 32 is NOT the ~4.5x speedup that ratio suggests. CORRECTED: 32 and "
             "auto=7 finished BBH in the SAME wall time. HF's static batching pads every "
             "member to the longest member's generation, so on a heterogeneous-length task "
             "a wider batch buys throughput on the short members and then waits for the "
             "long one. Widen the batch for headroom, not for speed.\n"
             "Ignored for served arms, where the server batches. Batch size does not change "
             "greedy results: arm A returned 0.7741 at both 8 and 'auto'.",
    )
    ap.add_argument("--lmeval-python", default=sys.executable)
    ap.add_argument(
        "--min-skip-share",
        type=float,
        default=0.5,
        help="floor for the skipping gate on a routed served arm",
    )
    args = ap.parse_args()

    if bool(args.base_url) == bool(args.hf_model):
        ap.error("give exactly one of --base-url (served) or --hf-model (native)")
    if args.base_url and not args.tokenizer:
        ap.error("--base-url needs --tokenizer")

    task = TASKS[args.workload]
    suite_name = args.suite_name or args.workload
    kind = _prompt_kind(args.suite_dir, suite_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"workload={args.workload} suite={suite_name} task={task} protocol={kind} arm={args.arm}")

    routes_decode = False
    before = args.output_dir / "server_info.before.json"
    after = args.output_dir / "server_info.after.json"
    if args.base_url:
        sys.path.insert(0, str(ROOT / "test/vp"))
        from run_qps_evaluation import _arm_routes_decode  # noqa: PLC0415

        info = _fetch_json(f"{args.base_url.rstrip('/')}/server_info")
        before.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
        routes_decode = _arm_routes_decode(info, upstream_baseline=args.upstream_baseline)
        print("served arm: "
              + ("UPSTREAM (no vp_runtime, as required)" if args.upstream_baseline
                 else f"routes decode: {routes_decode}"))

    command = _lmeval_command(args, task, kind)
    print("+ " + " ".join(command), flush=True)
    lmeval_rc = subprocess.run(command, cwd=ROOT, check=False).returncode

    if args.base_url:
        after.write_text(
            json.dumps(_fetch_json(f"{args.base_url.rstrip('/')}/server_info"),
                       indent=2, sort_keys=True) + "\n"
        )

    failed: list[str] = []
    if lmeval_rc != 0:
        failed.append("lm_eval")

    # ZERO EMPTY on raw `resps`. Never filtered_resps: lm-eval rewrites an empty
    # generation as "[invalid]", so a 51.6 %-empty run reports 0.0 %.
    if subprocess.run(
        [sys.executable, str(GATES / "verify_zero_empty.py"),
         "--lmeval-dir", str(args.output_dir)],
        cwd=ROOT, check=False,
    ).returncode != 0:
        failed.append("zero_empty")

    # A served flag is not an active treatment (D-627): a whole night's quality numbers
    # described the production all-RUN body while every input gate passed.
    if routes_decode:
        if subprocess.run(
            [sys.executable, str(GATES / "verify_skipping_executed.py"),
             "--before", str(before), "--after", str(after),
             "--min-skip-share", str(args.min_skip_share)],
            cwd=ROOT, check=False,
        ).returncode != 0:
            failed.append("skipping_executed")

    manifest = {
        "workload": args.workload,
        "suite_name": suite_name,
        "lmeval_task": task,
        "prompt_kind": kind,
        "arm": args.arm,
        # Whether this cell was served by GENUINE upstream SGLang. Arm C of the 2x2 is
        # gated on "no quality collapse vs UPSTREAM sglang" (D-587), and for months arm C
        # was ARMS["stock"] -- this fork with the skipper off. The manifest recorded the
        # arm LABEL, which is what made that invisible to every downstream reader. Record
        # the fact, so the summariser can refuse a (D - C) computed against the wrong C.
        "upstream_baseline": bool(args.upstream_baseline),
        "routes_decode": routes_decode,
        "batch_size": (args.batch_size if not args.base_url else None),
        "lmeval_command": command,
        "lmeval_returncode": lmeval_rc,
        "failed_gates": failed,
        "status": "passed" if not failed else "refused",
    }
    (args.output_dir / "quality_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    if failed:
        print(f"\nREFUSED arm={args.arm} workload={args.workload} "
              f"gates={','.join(failed)}", flush=True)
        return 1
    print(f"\nOK arm={args.arm} workload={args.workload} protocol={kind}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
