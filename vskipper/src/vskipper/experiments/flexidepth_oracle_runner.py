#!/usr/bin/env python3
"""Official FlexiDepth oracle runner (W1.1).

Runs the OFFICIAL checkpoint (xuan-luo/FlexiDepth-Llama-3-8B-Instruct) through
its own trust_remote_code modeling path over a multi-item probe set and dumps,
per item: per-position router decisions for every routed layer (with the LAST
prompt position called out — H1), first-token logits top-k + argmax with a
generate() consistency check, the greedy continuation, and (optionally) the
per-generated-token router masks from a manual greedy loop (battery-2 route
tape). Design: vPipe-doc/codex/asplos-plan/2026-08-18-w11-oracle-runner-design.md.

Protocol code is REUSED from flexidepth_upstream_gsm8k_probe.py (published
HF multiturn protocol, workload single-turn, frozen-input-ids replay).

Facts grounded in the checkpoint's modeling_ddllama.py @ revision
2ce73595ad0467fedd539c113e5b9deed046df32:
- router_mask = (sigmoid(router_logits) > 0.5), mask 1 = RUN, threshold is the
  hard-coded 0.5 (no config field);
- DDLlamaModel.forward returns router_masks (one [batch, seq] tensor per
  routed layer); DDLlamaForCausalLM CONSUMES them without re-exposing, and its
  forward print()s a layer count to stdout every call — model calls run under
  redirect_stdout(devnull) and all runner output goes to files/stderr.

Route-tape convention (--per-step-masks): step_router_masks covers exactly the
generated tokens that were FED BACK as inputs — every generated token except
the final one (EOS or max-length alike; generation never feeds the final token
either). Invariant: len(step_router_masks) == max(0, completion_tokens - 1).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import msgspec

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flexidepth_upstream_gsm8k_probe import (  # noqa: E402
    build_upstream_messages,
    build_workload_messages,
    load_frozen_input_ids,
)

# modeling_ddllama.py:316 — router_mask = (router_weights > 0.5)
OFFICIAL_ROUTER_THRESHOLD = 0.5
# Audited checkpoint revision (modeling source read + threshold fact above).
DEFAULT_REVISION = "2ce73595ad0467fedd539c113e5b9deed046df32"


class FirstTokenEvidence(msgspec.Struct):
    argmax_id: int
    topk_ids: list[int]
    topk_logits: list[float]
    generate_first_id: int | None
    consistent: bool


class GenerationEvidence(msgspec.Struct):
    text: str
    completion_tokens: int
    last_token_id: int | None
    finish_empty: bool


class StepMask(msgspec.Struct):
    step: int
    token_id: int
    actions: dict[str, int]


class ProbeRecord(msgspec.Struct):
    target_id: str
    prompt_protocol: str
    model: str
    model_commit_hash: str | None
    dtype: str
    transformers_version: str
    torch_version: str
    prompt_tokens: int
    prompt_token_sha256: str
    max_new_tokens: int
    routing_layers: list[int]
    router_threshold: float
    prompt_router_masks: dict[str, list[int]]
    last_prompt_position_actions: dict[str, str]
    prompt_skip_rate: float
    first_token: FirstTokenEvidence
    generation: GenerationEvidence
    fewshot_seed: int | None = None
    num_fewshot: int | None = None
    fewshot_indices: list[int] | None = None
    frozen_requested_output_len: int | None = None
    step_router_masks: list[StepMask] | None = None
    step_generation_matches_generate: bool | None = None


class ItemStatus(msgspec.Struct):
    target_id: str
    ok: bool
    consistent: bool
    finish_empty: bool
    error: str | None = None


class RunManifest(msgspec.Struct):
    model: str
    revision_requested: str | None
    resolved_commit_hash: str | None
    dtype: str
    prompt_protocol: str
    transformers_version: str
    torch_version: str
    runner_git_commit: str
    runner_sha256: str
    probe_items_sha256: str
    requests_jsonl_sha256: str | None
    dataset_source: str | None
    generation_config: dict
    started_utc: str
    finished_utc: str
    args: dict[str, str]
    items: list[ItemStatus]
    n_items: int
    n_consistency_failures: int
    n_errors: int
    n_empty: int


def parse_spec_list(spec: str) -> list[str]:
    """Parse a comma-separated list, or '@path' meaning one entry per line."""
    if spec.startswith("@"):
        lines = Path(spec[1:]).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]
    return [part.strip() for part in spec.split(",") if part.strip()]


def normalize_eos_ids(eos_token_id: object) -> list[int]:
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    if isinstance(eos_token_id, (list, tuple)):
        return [int(token_id) for token_id in eos_token_id]
    raise TypeError(f"unsupported eos_token_id type: {type(eos_token_id)!r}")


def checked_mask_row(values: list, expected_len: int, context: str) -> list[int]:
    """Validate one mask row: exact length, strictly binary values."""
    if len(values) != expected_len:
        raise ValueError(
            f"{context}: mask length {len(values)} != expected {expected_len}"
        )
    row: list[int] = []
    for value in values:
        if value not in (0, 0.0, 1, 1.0):
            raise ValueError(f"{context}: non-binary mask value {value!r}")
        row.append(int(value))
    return row


def checked_layer_masks(
    masks: tuple, routing_layers: list[int], context: str
) -> None:
    if len(masks) != len(routing_layers):
        raise ValueError(
            f"{context}: {len(masks)} masks for {len(routing_layers)} routed layers"
        )


def actions_from_masks(
    routing_layers: list[int], per_layer_prompt_masks: dict[str, list[int]]
) -> tuple[dict[str, str], float]:
    """Last-position run/skip per layer (H1 field) + prompt-wide skip rate."""
    last_actions: dict[str, str] = {}
    run_slots = 0
    total_slots = 0
    for layer_index in routing_layers:
        mask = per_layer_prompt_masks[str(layer_index)]
        last_actions[str(layer_index)] = "run" if mask[-1] == 1 else "skip"
        run_slots += sum(mask)
        total_slots += len(mask)
    skip_rate = (total_slots - run_slots) / total_slots if total_slots else 0.0
    return last_actions, skip_rate


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def runner_git_commit() -> str:
    here = Path(__file__).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain", "--", str(here)],
            cwd=here.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return f"{commit}-dirty" if porcelain else commit
    except (subprocess.SubprocessError, OSError):
        return "unavailable"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="xuan-luo/FlexiDepth-Llama-3-8B-Instruct"
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"hub revision pin (default {DEFAULT_REVISION[:12]}…); "
        "pass 'local' for a local model directory",
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--prompt-protocol",
        choices=("upstream-multiturn", "workload-single-turn", "frozen-input-ids"),
        required=True,
    )
    parser.add_argument(
        "--target-indices",
        help="gsm8k test indices: comma list or @file (one per line)",
    )
    parser.add_argument(
        "--request-ids",
        help="frozen-suite request IDs: comma list or @file (one per line)",
    )
    parser.add_argument("--requests-jsonl", type=Path)
    parser.add_argument("--fewshot-seed", type=int, default=1234)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--per-step-masks", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.topk < 1:
        parser.error("--topk must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.num_fewshot < 0:
        parser.error("--num-fewshot must be >= 0")
    if args.prompt_protocol == "frozen-input-ids":
        if not args.request_ids or args.requests_jsonl is None:
            parser.error("frozen-input-ids requires --request-ids and --requests-jsonl")
        if args.target_indices:
            parser.error("--target-indices is not valid with frozen-input-ids")
        if not args.requests_jsonl.is_file():
            parser.error(f"--requests-jsonl not found: {args.requests_jsonl}")
        item_ids = parse_spec_list(args.request_ids)
    else:
        if not args.target_indices:
            parser.error(f"{args.prompt_protocol} requires --target-indices")
        if args.request_ids or args.requests_jsonl is not None:
            parser.error(
                "--request-ids/--requests-jsonl are only valid with frozen-input-ids"
            )
        item_ids = parse_spec_list(args.target_indices)
        for item_id in item_ids:
            if not item_id.lstrip("-").isdigit():
                parser.error(f"non-integer target index: {item_id!r}")
            if int(item_id) < 0:
                parser.error(f"negative target index: {item_id!r}")
    if not item_ids:
        parser.error("empty probe set")
    if len(item_ids) != len(set(item_ids)):
        parser.error("duplicate probe item ids")

    revision = None if args.revision == "local" else args.revision
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The checkpoint's modeling code needs the post-4.55 transformers APIs
    # (masking_utils, GradientCheckpointingLayer, TransformersKwargs), and this
    # runner passes the canonical `dtype=` kwarg — on older transformers that
    # kwarg is silently dropped and the model loads fp32 (2x memory). Fail
    # closed instead of degrading.
    version_tuple = tuple(
        int(part) for part in transformers.__version__.split(".")[:2]
    )
    if version_tuple < (4, 56):
        raise SystemExit(
            f"transformers {transformers.__version__} is too old for the "
            "official FlexiDepth modeling code; use the box server venv "
            "(transformers >= 4.56)."
        )

    train = test = None
    dataset_source = None
    if args.prompt_protocol != "frozen-input-ids":
        import datasets

        try:
            train = datasets.load_dataset("gsm8k", "main", split="train")
            test = datasets.load_dataset("gsm8k", "main", split="test")
            dataset_source = "gsm8k"
        except Exception as exc:
            log(f"dataset 'gsm8k' load failed ({exc!r}); trying 'openai/gsm8k'")
            train = datasets.load_dataset("openai/gsm8k", "main", split="train")
            test = datasets.load_dataset("openai/gsm8k", "main", split="test")
            dataset_source = "openai/gsm8k"
        for item_id in item_ids:
            if int(item_id) >= len(test):
                parser.error(
                    f"target index {item_id} out of range (test size {len(test)})"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    manifest_path = args.output_dir / "run_manifest.json"
    for existing in (records_path, manifest_path):
        if existing.exists():
            raise SystemExit(
                f"refusing to overwrite existing output: {existing} "
                "(use a fresh --output-dir)"
            )

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    log(f"loading {args.model} (revision={revision}, dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, revision=revision
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=True,
            revision=revision,
        )
        .to("cuda")
        .eval()
    )
    routing_layers = list(model.config.routing_layers)
    eos_ids = normalize_eos_ids(model.generation_config.eos_token_id)
    statuses: list[ItemStatus] = []

    # The official CausalLM forward print()s to stdout every call.
    devnull = open(os.devnull, "w")  # noqa: SIM115 — closed in finally below
    try:
        with records_path.open("wb") as records_file:
            for item_id in item_ids:
                log(f"item {item_id}")
                try:
                    status = run_item(
                        args=args,
                        item_id=item_id,
                        model=model,
                        tokenizer=tokenizer,
                        routing_layers=routing_layers,
                        eos_ids=eos_ids,
                        train=train,
                        test=test,
                        devnull=devnull,
                        records_file=records_file,
                        transformers_version=str(transformers.__version__),
                        torch_version=str(torch.__version__),
                    )
                except Exception as exc:  # per-item capture; run stays loud via exit code
                    log(f"  ITEM ERROR {item_id}: {exc!r}")
                    status = ItemStatus(
                        target_id=item_id,
                        ok=False,
                        consistent=False,
                        finish_empty=False,
                        error=repr(exc),
                    )
                statuses.append(status)
    finally:
        devnull.close()

    n_failures = sum(
        1 for status in statuses if status.error is None and not status.consistent
    )
    n_errors = sum(1 for status in statuses if status.error is not None)
    manifest = RunManifest(
        model=args.model,
        revision_requested=revision,
        resolved_commit_hash=model.config._commit_hash,
        dtype=args.dtype,
        prompt_protocol=args.prompt_protocol,
        transformers_version=str(transformers.__version__),
        torch_version=str(torch.__version__),
        runner_git_commit=runner_git_commit(),
        runner_sha256=hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        probe_items_sha256=sha256_text("\n".join(item_ids)),
        requests_jsonl_sha256=(
            hashlib.sha256(args.requests_jsonl.read_bytes()).hexdigest()
            if args.requests_jsonl is not None
            else None
        ),
        dataset_source=dataset_source,
        generation_config=json.loads(
            model.generation_config.to_json_string(use_diff=False)
        ),
        started_utc=started_utc,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        args={key: str(value) for key, value in vars(args).items()},
        items=statuses,
        n_items=len(statuses),
        n_consistency_failures=n_failures,
        n_errors=n_errors,
        n_empty=sum(1 for status in statuses if status.finish_empty),
    )
    manifest_path.write_bytes(
        msgspec.json.format(msgspec.json.encode(manifest), indent=2) + b"\n"
    )
    log(
        f"wrote {records_path} ({len(statuses)} records, {manifest.n_empty} empty, "
        f"{n_failures} consistency failures, {n_errors} item errors)"
    )
    if n_failures or n_errors:
        log("RUN NOT CLEAN: consistency failures or item errors present")
        return 1
    return 0


def run_item(
    *,
    args: argparse.Namespace,
    item_id: str,
    model,
    tokenizer,
    routing_layers: list[int],
    eos_ids: list[int],
    train,
    test,
    devnull,
    records_file,
    transformers_version: str,
    torch_version: str,
) -> ItemStatus:
    import torch

    fewshot_indices = None
    frozen_requested_output_len = None
    if args.prompt_protocol == "frozen-input-ids":
        prompt_ids, frozen_requested_output_len = load_frozen_input_ids(
            args.requests_jsonl, item_id
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        target_id = item_id
    else:
        target = test[int(item_id)]
        if args.prompt_protocol == "upstream-multiturn":
            messages, fewshot_indices = build_upstream_messages(
                train,
                target,
                fewshot_seed=args.fewshot_seed,
                num_fewshot=args.num_fewshot,
            )
        else:
            messages, fewshot_indices = build_workload_messages(
                train, target, num_fewshot=args.num_fewshot
            )
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        target_id = f"gsm8k:test:{int(item_id)}"

    prompt_len = int(input_ids.shape[1])
    prompt_token_sha = sha256_text(json.dumps(input_ids[0].tolist()))

    with torch.inference_mode(), contextlib.redirect_stdout(devnull):
        # Base-model forward: the ONLY place router_masks are exposed (the
        # CausalLM wrapper consumes them). use_cache=True so the per-step
        # loop can continue from this same forward.
        base_out = model.model(input_ids=input_ids, use_cache=True)
        masks = base_out.router_masks
        checked_layer_masks(masks, routing_layers, f"{target_id} prompt")
        per_layer_prompt_masks = {}
        for layer_index, mask in zip(routing_layers, masks):
            if mask.ndim != 2 or mask.shape[0] != 1:
                raise ValueError(
                    f"{target_id} prompt layer {layer_index}: unexpected mask "
                    f"shape {tuple(mask.shape)}"
                )
            per_layer_prompt_masks[str(layer_index)] = checked_mask_row(
                mask[0].tolist(),
                prompt_len,
                f"{target_id} prompt layer {layer_index}",
            )
        last_actions, prompt_skip_rate = actions_from_masks(
            routing_layers, per_layer_prompt_masks
        )
        last_logits = model.lm_head(base_out.last_hidden_state[:, -1, :]).float()[0]
        topk = torch.topk(last_logits, k=args.topk)
        argmax_id = int(last_logits.argmax())

        output_ids = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion_ids = [
            int(v) for v in output_ids[0, input_ids.shape[1]:].tolist()
        ]

        step_masks: list[StepMask] | None = None
        step_matches: bool | None = None
        if args.per_step_masks:
            step_masks = []
            step_ids: list[int] = []
            past = base_out.past_key_values
            next_id = argmax_id
            while True:
                step_ids.append(next_id)
                # Route-tape convention (see module docstring): the final
                # generated token — EOS or max-length alike — is never fed
                # back, so it gets no StepMask, matching generate() exactly.
                if next_id in eos_ids or len(step_ids) >= args.max_new_tokens:
                    break
                step_input = torch.tensor(
                    [[next_id]], dtype=torch.long, device="cuda"
                )
                step_out = model.model(
                    input_ids=step_input, past_key_values=past, use_cache=True
                )
                past = step_out.past_key_values
                step_index = len(step_ids) - 1
                checked_layer_masks(
                    step_out.router_masks,
                    routing_layers,
                    f"{target_id} step {step_index}",
                )
                actions = {}
                for layer_index, mask in zip(routing_layers, step_out.router_masks):
                    if tuple(mask.shape) != (1, 1):
                        raise ValueError(
                            f"{target_id} step {step_index} layer {layer_index}: "
                            f"unexpected mask shape {tuple(mask.shape)}"
                        )
                    actions[str(layer_index)] = checked_mask_row(
                        mask[0].tolist(),
                        1,
                        f"{target_id} step {step_index} layer {layer_index}",
                    )[0]
                step_masks.append(
                    StepMask(step=step_index, token_id=next_id, actions=actions)
                )
                next_id = int(
                    model.lm_head(step_out.last_hidden_state[:, -1, :])
                    .float()[0]
                    .argmax()
                )
            step_matches = step_ids == completion_ids
            assert len(step_masks) == max(0, len(step_ids) - 1), (
                "route-tape invariant violated"
            )

    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    generate_first_id = completion_ids[0] if completion_ids else None
    consistent = generate_first_id == argmax_id and (
        step_matches is None or step_matches
    )
    record = ProbeRecord(
        target_id=target_id,
        prompt_protocol=args.prompt_protocol,
        model=args.model,
        model_commit_hash=model.config._commit_hash,
        dtype=args.dtype,
        transformers_version=transformers_version,
        torch_version=torch_version,
        prompt_tokens=prompt_len,
        prompt_token_sha256=prompt_token_sha,
        max_new_tokens=args.max_new_tokens,
        routing_layers=routing_layers,
        router_threshold=OFFICIAL_ROUTER_THRESHOLD,
        prompt_router_masks=per_layer_prompt_masks,
        last_prompt_position_actions=last_actions,
        prompt_skip_rate=prompt_skip_rate,
        first_token=FirstTokenEvidence(
            argmax_id=argmax_id,
            topk_ids=[int(v) for v in topk.indices.tolist()],
            topk_logits=[float(v) for v in topk.values.tolist()],
            generate_first_id=generate_first_id,
            consistent=generate_first_id == argmax_id,
        ),
        generation=GenerationEvidence(
            text=text,
            completion_tokens=len(completion_ids),
            last_token_id=completion_ids[-1] if completion_ids else None,
            finish_empty=not text.strip(),
        ),
        fewshot_seed=(
            args.fewshot_seed
            if args.prompt_protocol == "upstream-multiturn"
            else None
        ),
        num_fewshot=(
            args.num_fewshot
            if args.prompt_protocol != "frozen-input-ids"
            else None
        ),
        fewshot_indices=fewshot_indices,
        frozen_requested_output_len=frozen_requested_output_len,
        step_router_masks=step_masks,
        step_generation_matches_generate=step_matches,
    )
    records_file.write(msgspec.json.encode(record) + b"\n")
    records_file.flush()
    log(
        f"  skip_rate={prompt_skip_rate:.3f} "
        f"last_pos_skips={sum(1 for a in last_actions.values() if a == 'skip')}"
        f"/{len(routing_layers)} consistent={consistent} "
        f"empty={not text.strip()} tokens={len(completion_ids)}"
    )
    return ItemStatus(
        target_id=target_id,
        ok=consistent,
        consistent=consistent,
        finish_empty=not text.strip(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
