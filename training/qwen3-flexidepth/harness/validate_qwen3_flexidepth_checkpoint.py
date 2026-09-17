#!/usr/bin/env python3
"""Validate a full dense or MoE Qwen3 FlexiDepth initialized checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_cache(output) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (layer.keys.detach().cpu(), layer.values.detach().cpu())
        for layer in output.past_key_values.layers
    )


def compare_cache(
    candidate,
    reference: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> None:
    require(len(candidate.past_key_values.layers) == len(reference), "K/V layers")
    for layer, (expected_keys, expected_values) in zip(
        candidate.past_key_values.layers, reference
    ):
        torch.testing.assert_close(
            layer.keys.detach().cpu(), expected_keys, rtol=0, atol=0
        )
        torch.testing.assert_close(
            layer.values.detach().cpu(), expected_values, rtol=0, atol=0
        )


def load_model(path: Path, *, config=None, trust_remote_code: bool):
    return AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-model-type",
        choices=("qwen3", "qwen3_moe", "qwen2", "llama", "mistral"),
        required=True,
    )
    parser.add_argument("--routing-layers", type=int, nargs="+", required=True)
    return parser.parse_args()


def router_masks_of(candidate: Any, output: Any, **forward_kwargs: Any) -> tuple:
    """Router masks for a forward. ddqwen* CausalLM outputs carry them; the
    ddllama CausalLM (released Llama source, used for the Mistral port, D-348)
    returns a plain CausalLMOutputWithPast, so re-run the inner model, which
    exposes DDModelOutputWithPast.router_masks. Same weights, same inputs, no
    grad -- identical decisions."""

    masks = getattr(output, "router_masks", None)
    if masks is not None:
        return tuple(masks)
    with torch.no_grad():
        inner = candidate.model(**forward_kwargs)
    masks = getattr(inner, "router_masks", None)
    require(masks is not None, "router masks are not exposed by the candidate model")
    return tuple(masks)


def main() -> None:
    args = parse_args()
    require(torch.cuda.is_available(), "full checkpoint gate requires CUDA")
    manifest_path = args.candidate / "INITIALIZATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "PASS", "initialization manifest status")
    require(manifest.get("base_model_type") == args.base_model_type, "base model type")
    require(manifest.get("routing_layers") == args.routing_layers, "routing layers")
    expected_custom_type = {
        "qwen3": "ddqwen3",
        "qwen3_moe": "ddqwen3_moe",
        "qwen2": "ddqwen2",
        "gemma3": "ddgemma3",
        "gemma3_text": "ddgemma3",
        # Llama-layout families initialize through the ddllama source (D-348).
        "llama": "ddllama",
        "mistral": "ddllama",
    }[args.base_model_type]
    require(
        manifest.get("custom_model_type") == expected_custom_type,
        "custom model type",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base, local_files_only=True, trust_remote_code=False
    )
    tokens = tokenizer(
        "Virtual pipelining preserves each layer cache.", return_tensors="pt"
    )
    input_ids = tokens.input_ids.to("cuda")
    attention_mask = tokens.attention_mask.to("cuda")
    moe = args.base_model_type == "qwen3_moe"
    torch.cuda.reset_peak_memory_stats()

    base = load_model(args.base, trust_remote_code=False)
    with torch.no_grad():
        reference = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_router_logits=moe,
        )
    reference_logits = reference.logits.detach().cpu()
    reference_cache = cpu_cache(reference)
    reference_expert_logits = (
        tuple(tensor.detach().cpu() for tensor in reference.router_logits)
        if moe
        else None
    )
    del reference, base
    gc.collect()
    torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(
        args.candidate, local_files_only=True, trust_remote_code=True
    )
    require(config.model_type == expected_custom_type, "candidate config type")
    config.router_force_mode = "run"
    candidate = load_model(
        args.candidate, config=config, trust_remote_code=True
    )
    with torch.no_grad():
        all_run = candidate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_router_logits=moe,
        )
    torch.testing.assert_close(
        all_run.logits.detach().cpu(), reference_logits, rtol=0, atol=0
    )
    compare_cache(all_run, reference_cache)
    run_masks = router_masks_of(
        candidate, all_run, input_ids=input_ids, attention_mask=attention_mask,
        use_cache=False,
    )
    require(len(run_masks) == len(args.routing_layers), "RUN router mask count")
    require(all(mask.eq(1).all() for mask in run_masks), "RUN masks")
    if moe:
        require(
            len(all_run.router_logits) == len(reference_expert_logits),
            "expert router count",
        )
        for actual, expected in zip(
            all_run.router_logits, reference_expert_logits
        ):
            torch.testing.assert_close(actual.detach().cpu(), expected, rtol=0, atol=0)
    del all_run
    torch.cuda.empty_cache()

    for layer_idx in config.routing_layers:
        candidate.model.layers[layer_idx].router_force_mode = "project"
    with torch.no_grad():
        project = candidate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    require(
        len(project.past_key_values.layers) == config.num_hidden_layers,
        "PROJECT K/V layer count",
    )
    project_masks = router_masks_of(
        candidate, project, input_ids=input_ids, attention_mask=attention_mask,
        use_cache=False,
    )
    require(len(project_masks) == len(args.routing_layers), "PROJECT router mask count")
    require(all(mask.eq(0).all() for mask in project_masks), "PROJECT masks")
    for layer in project.past_key_values.layers:
        require(torch.isfinite(layer.keys).all().item(), "PROJECT nonfinite keys")
        require(torch.isfinite(layer.values).all().item(), "PROJECT nonfinite values")
        require(layer.keys.abs().sum().item() > 0, "PROJECT zero keys")
        require(layer.values.abs().sum().item() > 0, "PROJECT zero values")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "base_model_type": args.base_model_type,
        "custom_model_type": expected_custom_type,
        "routing_layers": args.routing_layers,
        "initialization_manifest_sha256": sha256(manifest_path),
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "checks": [
            "forced_run_exact_logits",
            "forced_run_exact_own_layer_kv",
            "forced_project_complete_own_layer_kv",
        ],
    }
    if moe:
        report["checks"].append("forced_run_exact_expert_router_logits")
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    temp = args.output.with_name(f"{args.output.name}.tmp")
    require(not temp.exists(), f"temporary output exists: {temp}")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temp.rename(args.output)


if __name__ == "__main__":
    main()
