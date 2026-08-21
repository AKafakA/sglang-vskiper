#!/usr/bin/env python3
"""Collect a sealed AdaSkip fixed-profile calibration on a GPU host.

This reproduces the audited official E2E warm-up semantics for a selected
Llama checkpoint: first 20 qasper test examples whose source ``length`` is
greater than 4000, official prompt formatting, official head/tail truncation,
per-token float32 cosine means, and output/input norm-ratio compensation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any


OFFICIAL_REPOSITORY = "https://github.com/ASISys/AdaSkip"
OFFICIAL_REVISION = "4ef686d6fd7b94756e4959f3f80191708dcec9e3"
OFFICIAL_E2E_MODEL_SHA256 = (
    "a59b20c898986964c8c1152f9236d8f25a2f3727dc4856225e122fbf858c14da"
)
OFFICIAL_E2E_RUNNER_SHA256 = (
    "7154f212bf81e74c0b7fe4092c98d7e0c054f2c994307d8ecb72d6b6731ad729"
)
OFFICIAL_PROMPT_CONFIG_SHA256 = (
    "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30"
)
QASPER_PROMPT_SHA256 = (
    "0fbdd123fe7f83d6d6a9c583ca28cb29d68e523bf8fec01a0cf2dcd11037775d"
)
LONG_BENCH_DATASET_ID = "zai-org/LongBench"
LONG_BENCH_DATASET_REVISION = (
    "38b3b9ec6c9d887c2b9f1587dc1476f6c8ff1a1a"
)
QASPER_PARQUET_SHA256 = (
    "7c6bf3a2a402b557d001808ba345a23921a211c39bf2d36d925d1d70e21b3f03"
)
QASPER_PARQUET_PATH = "qasper/test-00000-of-00001.parquet"
QASPER_TEST_ROWS = 200
COSINE_EPS = 1e-8
QASPER_PROMPT = (
    "You are given a scientific article and a question. Answer the question "
    "as concisely as you can, using a single phrase or sentence if possible. "
    'If the question cannot be answered based on the information in the article, '
    'write "unanswerable". If the question is a yes/no question, answer "yes", '
    '"no", or "unanswerable". Do not provide any explanation.\n\n'
    "Article: {context}\n\n Answer the question based on the above article "
    "as concisely as you can, using a single phrase or sentence if possible. "
    'If the question cannot be answered based on the information in the article, '
    'write "unanswerable". If the question is a yes/no question, answer "yes", '
    '"no", or "unanswerable". Do not provide any explanation.\n\n'
    "Question: {input}\n\nAnswer:"
)
REQUEST_COUNT = 20
SOURCE_LENGTH_MIN_EXCLUSIVE = 4000


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(payload)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite calibration artifact: {path}") from error


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-id", default=LONG_BENCH_DATASET_ID)
    parser.add_argument(
        "--dataset-revision",
        default=LONG_BENCH_DATASET_REVISION,
    )
    parser.add_argument("--max-input-tokens", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_input_tokens <= 1:
        parser.error("--max-input-tokens must be greater than one")
    for name in ("model_revision", "dataset_revision"):
        if not str(getattr(args, name)).strip():
            parser.error(f"--{name.replace('_', '-')} must be nonempty")
    if args.dataset_id != LONG_BENCH_DATASET_ID:
        parser.error(
            f"--dataset-id must be the pinned {LONG_BENCH_DATASET_ID}"
        )
    if args.dataset_revision != LONG_BENCH_DATASET_REVISION:
        parser.error(
            "--dataset-revision must be the pinned qasper commit "
            f"{LONG_BENCH_DATASET_REVISION}"
        )
    return args


def _truncate_like_official(tokenizer: Any, prompt: str, limit: int) -> str:
    token_ids = tokenizer(
        prompt,
        truncation=False,
        return_tensors="pt",
    ).input_ids[0]
    if len(token_ids) <= limit:
        return prompt
    half = int(limit / 2)
    return tokenizer.decode(
        token_ids[:half], skip_special_tokens=True
    ) + tokenizer.decode(token_ids[-half:], skip_special_tokens=True)


class _LayerCollector:
    def __init__(self, layers: Any) -> None:
        import torch
        import torch.nn.functional as F

        self._torch = torch
        self._functional = F
        self._layers = layers
        self._pending_input: dict[int, Any] = {}
        self._pending_post_attention: dict[int, Any] = {}
        self._attention_similarity = [0.0] * len(layers)
        self._mlp_similarity = [0.0] * len(layers)
        self._attention_scale = [0.0] * len(layers)
        self._mlp_scale = [0.0] * len(layers)
        self._request_count = 0
        self._hooks = []
        for layer_id, layer in enumerate(layers):
            self._hooks.append(
                layer.register_forward_pre_hook(self._layer_input_hook(layer_id))
            )
            self._hooks.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    self._post_attention_hook(layer_id)
                )
            )
            self._hooks.append(layer.register_forward_hook(self._layer_output_hook(layer_id)))

    def _layer_input_hook(self, layer_id: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            hidden_states = args[0]
            if hidden_states.ndim == 3 and hidden_states.shape[-2] > 1:
                self._pending_input[layer_id] = hidden_states

        return hook

    def _post_attention_hook(self, layer_id: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            post_attention = args[0]
            if post_attention.ndim != 3 or post_attention.shape[-2] <= 1:
                return
            layer_input = self._pending_input.pop(layer_id)
            self._attention_similarity[layer_id] += float(
                self._functional.cosine_similarity(
                    layer_input.float(),
                    post_attention.float(),
                    dim=-1,
                    eps=COSINE_EPS,
                )
                .mean()
                .item()
            )
            self._attention_scale[layer_id] += float(
                self._torch.norm(post_attention).item()
                / self._torch.norm(layer_input).item()
            )
            self._pending_post_attention[layer_id] = post_attention

        return hook

    def _layer_output_hook(self, layer_id: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states.ndim != 3 or hidden_states.shape[-2] <= 1:
                return
            post_attention = self._pending_post_attention.pop(layer_id)
            self._mlp_similarity[layer_id] += float(
                self._functional.cosine_similarity(
                    post_attention.float(),
                    hidden_states.float(),
                    dim=-1,
                    eps=COSINE_EPS,
                )
                .mean()
                .item()
            )
            self._mlp_scale[layer_id] += float(
                self._torch.norm(hidden_states).item()
                / self._torch.norm(post_attention).item()
            )

        return hook

    def finish_request(self) -> None:
        if self._pending_input or self._pending_post_attention:
            raise RuntimeError("AdaSkip calibration hooks did not close a layer")
        self._request_count += 1

    def measurements(self) -> dict[str, list[float]]:
        if self._request_count != REQUEST_COUNT:
            raise RuntimeError(
                f"expected {REQUEST_COUNT} calibration requests; "
                f"observed {self._request_count}"
            )

        def average(values: list[float]) -> list[float]:
            return [value / self._request_count for value in values]

        return {
            "attention_similarity": average(self._attention_similarity),
            "mlp_similarity": average(self._mlp_similarity),
            "attention_scale": average(self._attention_scale),
            "mlp_scale": average(self._mlp_scale),
        }

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()


def main() -> None:
    args = _arguments()
    artifact_paths = {
        "prompts": args.output_dir / "selected_prompts.json",
        "calibration": args.output_dir / "calibration.json",
        "manifest": args.output_dir / "manifest.json",
    }
    existing = [str(path) for path in artifact_paths.values() if path.exists()]
    if existing:
        raise RuntimeError(
            "refusing to reuse calibration artifacts: " + ", ".join(existing)
        )
    if _sha256(QASPER_PROMPT.encode("utf-8")) != QASPER_PROMPT_SHA256:
        raise RuntimeError("embedded qasper prompt does not match the audited source")

    import datasets
    import torch
    import transformers
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parquet_path = Path(
        hf_hub_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            filename=QASPER_PARQUET_PATH,
            revision=args.dataset_revision,
        )
    )
    parquet_sha256 = _file_sha256(parquet_path)
    if parquet_sha256 != QASPER_PARQUET_SHA256:
        raise RuntimeError(
            "pinned qasper parquet hash mismatch: "
            f"expected {QASPER_PARQUET_SHA256}, observed {parquet_sha256}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    # Load the hash-verified parquet directly: datasets>=4 removed script
    # loaders (the pinned LongBench revision ships LongBench.py), and the
    # parquet's SHA-256 gate above already proves byte-identity of the
    # source — stronger provenance than the script loader it replaces.
    dataset = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    )
    if len(dataset) != QASPER_TEST_ROWS:
        raise RuntimeError(
            f"pinned qasper test split has {len(dataset)} rows; "
            f"expected {QASPER_TEST_ROWS}"
        )

    selected = []
    prompt_records = []
    for dataset_index, row in enumerate(dataset):
        if int(row["length"]) <= SOURCE_LENGTH_MIN_EXCLUSIVE:
            continue
        original_prompt = QASPER_PROMPT.format(**row)
        prompt = _truncate_like_official(
            tokenizer,
            original_prompt,
            args.max_input_tokens,
        )
        input_ids = tokenizer(
            prompt,
            truncation=False,
            return_tensors="pt",
        ).input_ids[0]
        selected.append(prompt)
        prompt_records.append(
            {
                "selection_index": len(selected) - 1,
                "dataset_index": dataset_index,
                "source_length": int(row["length"]),
                "input_tokens": int(input_ids.shape[0]),
                "prompt_sha256": _sha256(prompt.encode("utf-8")),
                "prompt": prompt,
            }
        )
        if len(selected) == REQUEST_COUNT:
            break
    if len(selected) != REQUEST_COUNT:
        raise RuntimeError(
            f"qasper supplied only {len(selected)} qualifying calibration requests"
        )
    prompt_payload = _canonical_json(prompt_records)
    prompt_set_sha256 = _sha256(prompt_payload)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    if resolved_model_revision and resolved_model_revision != args.model_revision:
        raise RuntimeError(
            "resolved model revision differs from --model-revision: "
            f"{resolved_model_revision}"
        )
    layers = model.model.layers
    collector = _LayerCollector(layers)
    try:
        with torch.inference_mode():
            for prompt in selected:
                model_inputs = tokenizer(
                    prompt,
                    truncation=False,
                    return_tensors="pt",
                ).to(args.device)
                model.generate(
                    **model_inputs,
                    max_new_tokens=1,
                    num_beams=1,
                    do_sample=False,
                )
                collector.finish_request()
        measurements = collector.measurements()
    finally:
        collector.close()

    dataset_identity = (
        f"{args.dataset_id}:qasper:test:length_gt_4000:first20:"
        f"prompt_set_sha256={prompt_set_sha256}:"
        f"max_input_tokens={args.max_input_tokens}:head_tail_retokenize"
    )
    calibration = {
        "schema": "vpipe-adaskip-calibration-v1",
        "model": {
            "id": args.model,
            "revision": args.model_revision,
            "num_hidden_layers": len(layers),
        },
        "source": {
            "repository": OFFICIAL_REPOSITORY,
            "revision": OFFICIAL_REVISION,
        },
        "calibration": {
            "dataset_id": dataset_identity,
            "dataset_revision": args.dataset_revision,
            "request_count": REQUEST_COUNT,
        },
        "measurements": measurements,
    }
    calibration_payload = _canonical_json(calibration)
    manifest = {
        "schema": "vpipe-adaskip-calibration-manifest-v1",
        "calibration_sha256": _sha256(calibration_payload),
        "selected_prompts_sha256": prompt_set_sha256,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset": args.dataset_id,
        "dataset_config": "qasper",
        "dataset_split": "test",
        "dataset_revision": args.dataset_revision,
        "dataset_qasper_parquet_path": QASPER_PARQUET_PATH,
        "dataset_qasper_parquet_expected_sha256": QASPER_PARQUET_SHA256,
        "dataset_qasper_parquet_observed_sha256": parquet_sha256,
        "dataset_qasper_test_rows": len(dataset),
        "request_count": REQUEST_COUNT,
        "source_length_min_exclusive": SOURCE_LENGTH_MIN_EXCLUSIVE,
        "max_input_tokens": args.max_input_tokens,
        "selection": "dataset_order_after_source_length_filter",
        "truncation": "official_head_tail_decode_retokenize",
        "prompt_template_sha256": QASPER_PROMPT_SHA256,
        "cosine_epsilon": COSINE_EPS,
        "official_source": {
            "repository": OFFICIAL_REPOSITORY,
            "revision": OFFICIAL_REVISION,
            "e2e_model_sha256": OFFICIAL_E2E_MODEL_SHA256,
            "e2e_runner_sha256": OFFICIAL_E2E_RUNNER_SHA256,
            "prompt_config_sha256": OFFICIAL_PROMPT_CONFIG_SHA256,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "device": str(args.device),
            "dtype": "bfloat16",
        },
    }

    _write_new(artifact_paths["prompts"], prompt_payload)
    _write_new(artifact_paths["calibration"], calibration_payload)
    _write_new(artifact_paths["manifest"], _canonical_json(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
