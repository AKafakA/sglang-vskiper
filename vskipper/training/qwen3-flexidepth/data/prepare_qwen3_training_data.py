#!/usr/bin/env python3
"""Prepare manifest-closed Qwen3 data for the two FlexiDepth stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import datasets
from transformers import AutoTokenizer


STAGES = {
    "alignment": {
        "dataset": "allenai/tulu-3-sft-mixture",
        "revision": "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
    },
    "annealing": {
        "dataset": "mlabonne/open-perfectblend",
        "revision": "af60f3c18201652a83a93f46fcfee1b646ba3df7",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_checkpoint_manifest(model_path: Path, expected_sha256: str) -> None:
    manifest = model_path / "CHECKPOINT_SHA256SUMS"
    require(manifest.is_file(), "tokenizer checkpoint manifest")
    require(sha256(manifest) == expected_sha256, "tokenizer manifest hash")
    rows = [line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line]
    require(rows, "empty tokenizer checkpoint manifest")
    verified = set()
    for expected, relative in rows:
        relative_path = Path(relative.removeprefix("*").removeprefix("./"))
        require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe checkpoint path: {relative}",
        )
        if (
            relative_path.suffix == ".safetensors"
            or relative_path.name == "model.safetensors.index.json"
        ):
            continue
        path = model_path / relative_path
        require(path.is_file(), f"checkpoint file missing: {relative_path}")
        require(sha256(path) == expected, f"checkpoint hash: {relative_path}")
        verified.add(relative_path.name)
    require(
        {"config.json", "tokenizer.json", "tokenizer_config.json"} <= verified,
        "tokenizer checkpoint files were not verified",
    )


def convert_messages(batch: dict[str, list[Any]], stage: str) -> dict[str, list[Any]]:
    messages_batch: list[list[dict[str, str]]] = []
    unknown_turns: list[int] = []
    nonempty_assistant: list[bool] = []
    if stage == "alignment":
        source_rows = batch["messages"]
        role_map = {"system": "system", "user": "user", "assistant": "assistant"}
        role_key = "role"
        content_key = "content"
    else:
        source_rows = batch["conversations"]
        role_map = {"human": "user", "gpt": "assistant"}
        role_key = "from"
        content_key = "value"

    for turns in source_rows:
        converted = []
        unknown = 0
        for turn in turns:
            role = role_map.get(turn.get(role_key))
            content = turn.get(content_key)
            if role is None:
                unknown += 1
                continue
            require(isinstance(content, str), "message content is not text")
            converted.append({"role": role, "content": content})
        messages_batch.append(converted)
        unknown_turns.append(unknown)
        nonempty_assistant.append(
            any(
                turn["role"] == "assistant" and bool(turn["content"].strip())
                for turn in converted
            )
        )
    return {
        "vp_messages": messages_batch,
        "vp_unknown_turns": unknown_turns,
        "vp_nonempty_assistant": nonempty_assistant,
    }


def tokenize_batch(
    batch: dict[str, list[Any]], tokenizer: Any
) -> dict[str, list[Any]]:
    text = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        for messages in batch["vp_messages"]
    ]
    tokenized = tokenizer(
        text,
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
        return_attention_mask=False,
    )
    tokenized["num_tokens"] = [len(input_ids) for input_ids in tokenized["input_ids"]]
    tokenized["unknown_turns"] = batch["vp_unknown_turns"]
    tokenized["nonempty_assistant"] = batch["vp_nonempty_assistant"]
    return tokenized


def within_limit(row: dict[str, Any], max_length: int) -> bool:
    return row["num_tokens"] <= max_length


def output_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "DATASET_SHA256SUMS"
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--tokenizer-manifest-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--test-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-examples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = STAGES[args.stage]
    require(args.max_length == 2048, "FlexiDepth training length must be 2048")
    require(args.test_size == 0.01, "FlexiDepth evaluation split must be 0.01")
    require(args.seed == 42, "FlexiDepth data seed must be 42")
    require(args.num_proc > 0, "num_proc must be positive")
    require(args.max_source_examples >= 0, "max source examples must be nonnegative")
    require(len(args.tokenizer_revision) == 40, "full tokenizer revision")
    require(len(args.source_revision) == 40, "full source revision")
    require(len(args.source_archive_sha256) == 64, "source archive SHA-256")

    tokenizer_path = args.tokenizer.resolve()
    output_dir = args.output_dir.resolve()
    temp = output_dir.with_name(f"{output_dir.name}.tmp.{os.getpid()}")
    require(tokenizer_path.is_dir(), "tokenizer directory")
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    require(not temp.exists(), f"temporary output exists: {temp}")
    verify_checkpoint_manifest(tokenizer_path, args.tokenizer_manifest_sha256)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )

    raw = datasets.load_dataset(
        stage["dataset"], revision=stage["revision"], split="train"
    )
    source_total_examples = len(raw)
    if args.max_source_examples:
        raw = raw.select(range(min(source_total_examples, args.max_source_examples)))
    source_examples = len(raw)
    converted = raw.map(
        convert_messages,
        batched=True,
        fn_kwargs={"stage": args.stage},
        num_proc=args.num_proc,
        remove_columns=raw.column_names,
        desc=f"Convert {args.stage} messages",
    )
    tokenized = converted.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        remove_columns=converted.column_names,
        desc=f"Tokenize {args.stage}",
    )
    filtered = tokenized.filter(
        within_limit,
        fn_kwargs={"max_length": args.max_length},
        num_proc=args.num_proc,
        desc="Filter sequences over 2048 tokens",
    )
    require(len(filtered) > 1, "no usable examples")
    unknown_turn_count = sum(filtered["unknown_turns"])
    no_nonempty_assistant_count = len(filtered) - sum(filtered["nonempty_assistant"])
    token_count = sum(filtered["num_tokens"])
    max_observed_length = max(filtered["num_tokens"])
    min_observed_length = min(filtered["num_tokens"])
    final = filtered.remove_columns(
        ["num_tokens", "unknown_turns", "nonempty_assistant"]
    ).train_test_split(test_size=args.test_size, seed=args.seed)
    dataset_dict = datasets.DatasetDict(
        {"train": final["train"], "eval": final["test"]}
    )

    temp.mkdir(parents=True)
    dataset_dict.save_to_disk(temp)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "stage": args.stage,
        "dataset": stage["dataset"],
        "dataset_revision": stage["revision"],
        "source_revision": args.source_revision,
        "source_archive_sha256": args.source_archive_sha256,
        "tokenizer": str(tokenizer_path),
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_manifest_sha256": args.tokenizer_manifest_sha256,
        "max_length": args.max_length,
        "test_size": args.test_size,
        "seed": args.seed,
        "source_examples": source_examples,
        "source_total_examples": source_total_examples,
        "is_preflight": bool(args.max_source_examples),
        "retained_examples": len(filtered),
        "discarded_over_length": source_examples - len(filtered),
        "train_examples": len(dataset_dict["train"]),
        "eval_examples": len(dataset_dict["eval"]),
        "retained_tokens": token_count,
        "minimum_length": min_observed_length,
        "maximum_length": max_observed_length,
        "unknown_turns": unknown_turn_count,
        "rows_without_nonempty_assistant": no_nonempty_assistant_count,
        "train_fingerprint": dataset_dict["train"]._fingerprint,
        "eval_fingerprint": dataset_dict["eval"]._fingerprint,
    }
    (temp / "PREPROCESSING_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    hashes = output_hashes(temp)
    with (temp / "DATASET_SHA256SUMS").open("w") as handle:
        for relative, digest in sorted(hashes.items()):
            handle.write(f"{digest}  {relative}\n")
    for relative, digest in hashes.items():
        require(sha256(temp / relative) == digest, f"output hash: {relative}")
    temp.rename(output_dir)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
