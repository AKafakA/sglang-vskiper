#!/usr/bin/env python3
"""Run a selected GSM8K item with FlexiDepth's published HF eval protocol."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def build_upstream_messages(
    train: Any,
    target: dict[str, Any],
    *,
    fewshot_seed: int,
    num_fewshot: int,
) -> tuple[list[dict[str, str]], list[int]]:
    indices = random.Random(fewshot_seed).sample(range(len(train)), num_fewshot)
    messages: list[dict[str, str]] = []
    for index in indices:
        row = train[index]
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Question: {row['question']}\nAnswer:",
                },
                {"role": "assistant", "content": row["answer"]},
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": f"Question: {target['question']}\nAnswer:",
        }
    )
    return messages, indices


def build_workload_messages(
    train: Any,
    target: dict[str, Any],
    *,
    num_fewshot: int,
) -> tuple[list[dict[str, str]], list[int]]:
    indices = list(range(num_fewshot))
    examples = []
    for index in indices:
        row = train[index]
        examples.append(f"Question: {row['question']}\nAnswer: {row['answer']}")
    examples.append(f"Question: {target['question']}\nAnswer:")
    return [{"role": "user", "content": "\n\n".join(examples)}], indices


def load_frozen_input_ids(
    requests_path: Path, request_id: str
) -> tuple[list[int], int | None]:
    for line in requests_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("request_id") != request_id:
            continue
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"{request_id} has no tokenized prompt")
        return [int(token_id) for token_id in prompt], row.get("output_len")
    raise ValueError(f"request ID not found: {request_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-index", type=int, default=1289)
    parser.add_argument("--fewshot-seed", type=int, default=1234)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-protocol",
        choices=(
            "upstream-multiturn",
            "workload-single-turn",
            "frozen-input-ids",
        ),
        default="upstream-multiturn",
    )
    parser.add_argument("--requests-jsonl", type=Path)
    parser.add_argument("--request-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import datasets
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frozen_requested_output_len = None
    if args.prompt_protocol == "frozen-input-ids":
        if args.requests_jsonl is None or not args.request_id:
            parser.error(
                "frozen-input-ids requires --requests-jsonl and --request-id"
            )
        prompt_ids, frozen_requested_output_len = load_frozen_input_ids(
            args.requests_jsonl, args.request_id
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        fewshot_indices = None
        target_id = args.request_id
    else:
        try:
            train = datasets.load_dataset("gsm8k", "main", split="train")
            test = datasets.load_dataset("gsm8k", "main", split="test")
        except Exception:
            train = datasets.load_dataset("openai/gsm8k", "main", split="train")
            test = datasets.load_dataset("openai/gsm8k", "main", split="test")

        target = test[args.target_index]
        if args.prompt_protocol == "upstream-multiturn":
            messages, fewshot_indices = build_upstream_messages(
                train,
                target,
                fewshot_seed=args.fewshot_seed,
                num_fewshot=args.num_fewshot,
            )
        else:
            messages, fewshot_indices = build_workload_messages(
                train,
                target,
                num_fewshot=args.num_fewshot,
            )
        target_id = f"gsm8k:test:{args.target_index}"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()
    if args.prompt_protocol != "frozen-input-ids":
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion_ids = output_ids[0, input_ids.shape[1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    record = {
        "model": args.model,
        "target_id": target_id,
        "prompt_protocol": args.prompt_protocol,
        "fewshot_indices": fewshot_indices,
        "fewshot_seed": args.fewshot_seed,
        "num_fewshot": args.num_fewshot,
        "max_new_tokens": args.max_new_tokens,
        "frozen_requested_output_len": frozen_requested_output_len,
        "do_sample": False,
        "temperature": 0.0,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "prompt_tokens": int(input_ids.shape[1]),
        "completion_tokens": int(completion_ids.shape[0]),
        "last_token_id": int(completion_ids[-1]) if len(completion_ids) else None,
        "eos_token_id": model.generation_config.eos_token_id,
        "text": text,
    }
    rendered = json.dumps(record, indent=2, ensure_ascii=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
