#!/usr/bin/env python3
"""Fail-closed two-stage FlexiDepth training for supported model families."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import datasets
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


STAGES = {
    "alignment": {
        "dataset": "allenai/tulu-3-sft-mixture",
        "longmix_dataset": "vpipe/longmix-alignment-r2",
        "penalty": 1e-4,
    },
    "annealing": {
        "dataset": "mlabonne/open-perfectblend",
        "longmix_dataset": "vpipe/longmix-annealing-r2",
        "penalty": 1e-5,
    },
}
GLOBAL_BATCH_SIZE = 32
IM_START_ID = 151644  # Qwen3 <|im_start|>
IM_END_ID = 151645  # Qwen3 <|im_end|>


class RouterStatsSFTTrainer(SFTTrainer):
    """DIAGNOSTIC (debug-20260902): rank-local router statistics folded into every log line."""

    _vp_stats = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        weights = getattr(outputs, "router_weights", None)
        if self.model.training and weights and "attention_mask" in inputs:
            with torch.no_grad():
                w = torch.stack(weights, dim=-1).float()
                att = inputs["attention_mask"].bool()
                wv = w[att]
                sup = (inputs["labels"] != -100)[att]
                s = self._vp_stats or {
                    "n": 0, "n_sup": 0, "npen": 0, "pen": 0.0,
                    "w": torch.zeros(w.shape[-1]), "run": torch.zeros(w.shape[-1]),
                    "run_sup": torch.zeros(w.shape[-1]),
                }
                s["n"] += int(wv.shape[0])
                s["w"] += wv.sum(0).cpu()
                s["run"] += (wv > 0.5).sum(0).cpu()
                s["n_sup"] += int(sup.sum())
                s["run_sup"] += (wv[sup] > 0.5).sum(0).cpu()
                pen = getattr(outputs, "router_penalty_loss", None)
                if pen is not None:
                    s["pen"] += float(pen)
                    s["npen"] += 1
                self._vp_stats = s
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        s = self._vp_stats
        if s and s["n"] and self.model.training:
            runf = s["run"] / s["n"]
            logs["vp_skip"] = round(float(1 - runf.mean()), 4)
            logs["vp_skip_sup"] = round(float(1 - (s["run_sup"] / max(1, s["n_sup"])).mean()), 4)
            logs["vp_w"] = round(float(s["w"].sum() / (s["n"] * s["w"].numel())), 4)
            logs["vp_pen"] = round(s["pen"] / max(1, s["npen"]), 7)
            logs["vp_run_layers"] = ",".join(f"{x:.2f}" for x in runf.tolist())
            self._vp_stats = None
        super().log(logs, start_time)


def add_completion_mask(batch, assistant_id):
    """DIAGNOSTIC P3: 1 for tokens inside <|im_start|>assistant ... <|im_end|> (incl. the <|im_end|>)."""
    masks = []
    for ids in batch["input_ids"]:
        m = [0] * len(ids)
        i, n = 0, len(ids)
        while i < n:
            if ids[i] == IM_START_ID and i + 1 < n and ids[i + 1] == assistant_id:
                j = i + 2
                while j < n and ids[j] != IM_END_ID:
                    j += 1
                for k in range(i + 2, min(j + 1, n)):
                    m[k] = 1
                i = j + 1
            else:
                i += 1
        masks.append(m)
    return {"completion_mask": masks}
MAX_SUPPORTED_LENGTH = 8192
CHECKPOINT_COMPLETE = "VP_CHECKPOINT_COMPLETE.json"
MODEL_SOURCE_FILES = {
    "ddllama": (
        "configuration_ddllama.py",
        "modeling_ddllama.py",
    ),
    "ddqwen3": (
        "configuration_ddqwen3.py",
        "modeling_ddqwen3.py",
    ),
    "ddqwen3_moe": (
        "configuration_ddqwen3_moe.py",
        "modeling_ddqwen3_moe.py",
    ),
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


def verify_hash_manifest(root: Path, name: str, expected_sha256: str) -> Path:
    manifest = root / name
    require(manifest.is_file(), f"missing manifest: {manifest}")
    require(sha256(manifest) == expected_sha256, f"manifest hash mismatch: {name}")
    rows = [line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line]
    require(rows, f"empty manifest: {name}")
    for expected, relative in rows:
        relative_path = Path(relative.removeprefix("*").removeprefix("./"))
        require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe manifest path: {relative}",
        )
        path = root / relative_path
        require(path.is_file(), f"manifest file missing: {relative_path}")
        require(sha256(path) == expected, f"file hash mismatch: {relative_path}")
    return manifest


def verify_model_manifest(model_path: Path, expected_sha256: str) -> Path:
    manifest = model_path / "INITIALIZATION_MANIFEST.json"
    if not manifest.is_file():
        manifest = model_path / "TRAINING_MANIFEST.json"
    require(manifest.is_file(), "model has no initialization/training manifest")
    require(sha256(manifest) == expected_sha256, "model manifest hash mismatch")
    payload = json.loads(manifest.read_text())
    require(payload.get("status") == "PASS", "input model manifest did not pass")
    output_files = payload.get("output_files")
    require(isinstance(output_files, dict) and output_files, "model output hash map")
    for relative, expected in sorted(output_files.items()):
        relative_path = Path(relative)
        require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe model manifest path: {relative}",
        )
        path = model_path / relative_path
        require(path.is_file(), f"model file missing: {relative}")
        require(sha256(path) == expected, f"model file hash mismatch: {relative}")
    return manifest


def expected_trainable_names(model: torch.nn.Module) -> set[str]:
    config = model.config
    result: set[str] = set()
    for layer_idx in config.routing_layers:
        prefix = f"model.layers.{layer_idx}"
        result.update(
            {
                f"{prefix}.router.router_enc.weight",
                f"{prefix}.router.router_norm.weight",
                f"{prefix}.router.router_dec.weight",
                f"{prefix}.router.router_head.weight",
                f"{prefix}.router_proj.gate_proj.weight",
                f"{prefix}.router_proj.down_proj.weight",
                f"{prefix}.router_proj.up_proj.weight",
            }
        )
        if getattr(config, "mlp_bias", False):
            result.update(
                {
                    f"{prefix}.router.router_enc.bias",
                    f"{prefix}.router.router_dec.bias",
                    f"{prefix}.router.router_head.bias",
                    f"{prefix}.router_proj.gate_proj.bias",
                    f"{prefix}.router_proj.down_proj.bias",
                    f"{prefix}.router_proj.up_proj.bias",
                }
            )
        if getattr(config, "router_head_bias_init", None) is not None:
            result.add(f"{prefix}.router.router_head.bias")
    return result


def freeze_and_validate(model: torch.nn.Module) -> dict[str, Any]:
    expected = expected_trainable_names(model)
    known = {name for name, _ in model.named_parameters()}
    require(expected <= known, f"missing router parameters: {sorted(expected - known)}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in expected
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    require(actual == expected, "trainable parameter set is not exact")
    return {
        "trainable_names": sorted(actual),
        "trainable_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in actual
        ),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def distributed_peak_cuda_memory(device: torch.device) -> dict[str, int]:
    peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        dtype=torch.int64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    return {
        "peak_cuda_memory_allocated_bytes": int(peaks[0].item()),
        "peak_cuda_memory_reserved_bytes": int(peaks[1].item()),
    }


def output_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "TRAINING_MANIFEST.json"
    }


def seal_work_contract(work_dir: Path, contract: dict[str, Any], rank: int) -> Path:
    path = work_dir / "TRAINING_INPUT_CONTRACT.json"
    encoded = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if rank == 0 and not path.exists():
        temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temp.write_text(encoded)
        temp.rename(path)
    for _ in range(600):
        if path.is_file():
            break
        time.sleep(0.1)
    require(path.is_file(), "training input contract was not published")
    require(path.read_text() == encoded, "training work contract changed")
    return path


def verify_complete_checkpoint(path: Path) -> int:
    require(path.is_dir(), f"checkpoint directory missing: {path}")
    try:
        step = int(path.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise ValueError(f"invalid checkpoint name: {path.name}") from error
    marker = path / CHECKPOINT_COMPLETE
    require(marker.is_file(), f"checkpoint is not sealed: {path}")
    payload = json.loads(marker.read_text())
    require(payload == {"global_step": step}, f"invalid checkpoint marker: {path}")
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        require((path / name).is_file(), f"incomplete checkpoint {path}: {name}")
    model_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    require(
        any((path / name).is_file() for name in model_files),
        f"incomplete checkpoint {path}: model weights",
    )
    return step


class SealedCheckpointCallback(TrainerCallback):
    def __init__(self, stop_after_step: int = 0):
        self.stop_after_step = stop_after_step

    def on_save(self, args, state, control, **kwargs):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
                require(
                    (checkpoint / name).is_file(),
                    f"checkpoint save did not publish {name}: {checkpoint}",
                )
            model_files = (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
            require(
                any((checkpoint / name).is_file() for name in model_files),
                f"checkpoint save did not publish model weights: {checkpoint}",
            )
            marker = checkpoint / CHECKPOINT_COMPLETE
            temp = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
            temp.write_text(json.dumps({"global_step": state.global_step}) + "\n")
            temp.rename(marker)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        verify_complete_checkpoint(checkpoint)
        if self.stop_after_step:
            require(
                state.global_step <= self.stop_after_step,
                "checkpoint probe passed its declared stop step",
            )
            if state.global_step == self.stop_after_step:
                control.should_training_stop = True
        return control


def resolve_resume(work_dir: Path, value: str) -> str | bool:
    if value == "none":
        return False
    if value != "auto":
        path = Path(value).resolve()
        require(work_dir in path.parents, "resume checkpoint is outside work directory")
        verify_complete_checkpoint(path)
        return str(path)
    checkpoints = []
    for path in work_dir.glob("checkpoint-*"):
        if (path / CHECKPOINT_COMPLETE).is_file():
            checkpoints.append((verify_complete_checkpoint(path), path))
    checkpoints.sort()
    return str(checkpoints[-1][1]) if checkpoints else False


def checkpoint_evidence(value: str | bool) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value)
    step = verify_complete_checkpoint(path)
    marker = path / CHECKPOINT_COMPLETE
    state = path / "trainer_state.json"
    return {
        "path": str(path),
        "step": step,
        "marker_sha256": sha256(marker),
        "trainer_state_sha256": sha256(state),
        "files": {
            child.name: child.stat().st_size
            for child in sorted(path.iterdir())
            if child.is_file()
        },
    }


def sealed_checkpoint_steps(work_dir: Path) -> list[int]:
    return sorted(
        verify_complete_checkpoint(path)
        for path in work_dir.glob("checkpoint-*")
        if (path / CHECKPOINT_COMPLETE).is_file()
    )


def verify_frozen_parameter_identity(
    model: torch.nn.Module,
    model_path: Path,
) -> dict[str, int | bool]:
    trainable = expected_trainable_names(model)
    current = dict(model.named_parameters())
    reference = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    checked_tensors = 0
    checked_parameters = 0
    for name, expected in reference.named_parameters():
        require(name in current, f"reference tensor missing from trained model: {name}")
        if name in trainable:
            continue
        actual = current[name].detach().cpu()
        require(
            torch.equal(actual, expected.detach()),
            f"frozen parameter changed: {name}",
        )
        checked_tensors += 1
        checked_parameters += expected.numel()
        del actual
    del reference
    gc.collect()
    require(checked_tensors > 0, "no frozen parameters were checked")
    return {
        "passed": True,
        "checked_tensors": checked_tensors,
        "checked_parameters": checked_parameters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-device-train-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--router-penalty", type=float, default=None)
    parser.add_argument("--router-penalty-accumulation-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--probe-manifest-only", action="store_true")
    parser.add_argument("--checkpoint-probe", action="store_true")
    parser.add_argument("--stop-after-checkpoint-step", type=int, default=0)
    parser.add_argument("--verify-frozen-tensors", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--completion-only-loss", action="store_true")
    parser.add_argument("--completion-mask-cache", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = STAGES[args.stage]
    require(
        args.dataset_name in {stage["dataset"], stage["longmix_dataset"]},
        "dataset does not match stage",
    )
    require(len(args.source_revision) == 40, "source revision must be a full git SHA")
    require(len(args.source_archive_sha256) == 64, "source archive SHA-256")
    require(args.learning_rate == 1e-4, "FlexiDepth learning rate must remain 1e-4")
    require(args.save_steps > 0 and args.logging_steps > 0, "step intervals")
    require(0 < args.max_length <= MAX_SUPPORTED_LENGTH, "unsupported max length")
    require(args.save_total_limit > 0, "save total limit must be positive")
    require(
        not (args.probe_manifest_only and args.checkpoint_probe),
        "manifest-only and checkpoint probes are mutually exclusive",
    )
    if args.probe_manifest_only:
        require(args.max_steps > 0, "manifest-only probe must have bounded steps")
        require(
            args.resume_from_checkpoint == "none",
            "manifest-only probe cannot resume",
        )
    if args.checkpoint_probe:
        require(args.max_steps > 0, "checkpoint probe must have bounded steps")
    if args.stop_after_checkpoint_step:
        require(args.checkpoint_probe, "checkpoint stop requires checkpoint probe")
        require(
            0 < args.stop_after_checkpoint_step < args.max_steps,
            "checkpoint stop must precede the final step",
        )
    if args.verify_frozen_tensors:
        require(args.checkpoint_probe, "frozen-tensor check requires checkpoint probe")
        require(
            args.resume_from_checkpoint != "none",
            "frozen-tensor check requires a resumed process",
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    require(0 <= rank < world_size, "distributed rank is outside world size")
    require(torch.cuda.is_available(), "FlexiDepth training requires CUDA")
    require(0 <= local_rank < torch.cuda.device_count(), "invalid local CUDA rank")
    torch.cuda.set_device(local_rank)
    global_batch = (
        world_size
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    require(global_batch == GLOBAL_BATCH_SIZE, "global batch must equal 32")

    model_path = args.model_path.resolve()
    dataset_dir = args.dataset_dir.resolve()
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    require(model_path.is_dir(), "model path")
    require(dataset_dir.is_dir(), "dataset directory")
    require(work_dir != output_dir, "work and output directories must differ")
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    work_contract = seal_work_contract(
        work_dir,
        {
            "schema_version": 1,
            "stage": args.stage,
            "source_revision": args.source_revision,
            "source_archive_sha256": args.source_archive_sha256,
            "model_path": str(model_path),
            "model_manifest_sha256": args.model_manifest_sha256,
            "dataset_dir": str(dataset_dir),
            "dataset_name": args.dataset_name,
            "dataset_revision": args.dataset_revision,
            "dataset_manifest_sha256": args.dataset_manifest_sha256,
            "world_size": world_size,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": global_batch,
            "learning_rate": args.learning_rate,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "max_steps": args.max_steps,
            "max_length": args.max_length,
            "probe_manifest_only": args.probe_manifest_only,
            "checkpoint_probe": args.checkpoint_probe,
            "seed": args.seed,
            "completion_only_loss": args.completion_only_loss,
            "router_penalty_override": args.router_penalty,
            "router_penalty_accumulation_override": args.router_penalty_accumulation_steps,
            "probe_source_patch": "debug-20260902 D1-D4",
        },
        rank,
    )

    input_model_manifest = verify_model_manifest(
        model_path, args.model_manifest_sha256
    )
    dataset_manifest = verify_hash_manifest(
        dataset_dir, "DATASET_SHA256SUMS", args.dataset_manifest_sha256
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,
    )
    model_source_files = MODEL_SOURCE_FILES.get(model.config.model_type)
    require(
        model_source_files is not None,
        "model is not a supported FlexiDepth model",
    )
    require(model.config.router_force_mode == "dynamic", "router force mode")
    require(
        args.max_length <= model.config.max_position_embeddings,
        "trainer length exceeds model context",
    )
    # None = the sealed stage-table / ga-coupled semantics of every prior run.
    effective_router_penalty = (
        stage["penalty"] if args.router_penalty is None else args.router_penalty
    )
    effective_penalty_accumulation = (
        args.gradient_accumulation_steps
        if args.router_penalty_accumulation_steps is None
        else args.router_penalty_accumulation_steps
    )
    require(effective_router_penalty >= 0, "router penalty must be nonnegative (0 = P1 control)")
    require(
        effective_penalty_accumulation >= 1,
        "router penalty accumulation must be >= 1",
    )
    model.config.router_penalty = effective_router_penalty
    model.config.router_penalty_gradient_accumulation_steps = (
        effective_penalty_accumulation
    )
    model.config.use_cache = False
    trainable = freeze_and_validate(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    dataset_dict = datasets.load_from_disk(dataset_dir)
    require("train" in dataset_dict, "dataset has no train split")
    train_dataset = dataset_dict["train"]
    require("input_ids" in train_dataset.column_names, "dataset is not tokenized")
    require(len(train_dataset) > 0, "empty train split")
    if args.completion_only_loss:
        assistant_id = tokenizer.convert_tokens_to_ids("assistant")
        require(isinstance(assistant_id, int) and assistant_id >= 0, "assistant token id")
        require(args.completion_mask_cache is not None, "completion mask cache path")
        train_dataset = train_dataset.map(
            add_completion_mask,
            batched=True,
            fn_kwargs={"assistant_id": assistant_id},
            num_proc=16,
            cache_file_name=f"{args.completion_mask_cache}.rank{rank}.arrow",
            desc="completion mask (assistant spans)",
        )
        require("completion_mask" in train_dataset.column_names, "completion mask column")
        sample_masks = train_dataset[:64]["completion_mask"]
        covered = sum(sum(m) for m in sample_masks) / max(1, sum(len(m) for m in sample_masks))
        require(0.2 < covered < 0.95, f"implausible assistant coverage {covered:.3f}")
    if "sequence_tokens" in train_dataset.column_names:
        require(
            max(train_dataset["sequence_tokens"]) <= args.max_length,
            "dataset sequence exceeds trainer length",
        )

    training_args = SFTConfig(
        output_dir=str(work_dir),
        overwrite_output_dir=False,
        do_eval=False,
        learning_rate=args.learning_rate,
        num_train_epochs=1.0,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        dataloader_drop_last=True,
        remove_unused_columns=True,
        save_strategy="no" if args.probe_manifest_only else "steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        seed=args.seed,
        max_length=args.max_length,
        report_to="none",
        ddp_find_unused_parameters=False,
        completion_only_loss=True if args.completion_only_loss else None,
    )
    trainer = RouterStatsSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=(
            []
            if args.probe_manifest_only
            else [
                SealedCheckpointCallback(
                    stop_after_step=args.stop_after_checkpoint_step
                )
            ]
        ),
    )
    resume_from_checkpoint = resolve_resume(
        work_dir, args.resume_from_checkpoint
    )
    resume_evidence = checkpoint_evidence(resume_from_checkpoint)
    result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )
    trainer.accelerator.wait_for_everyone()
    peak_cuda_memory = distributed_peak_cuda_memory(torch.device("cuda", local_rank))
    frozen_identity: dict[str, int | bool] | None = None
    if args.verify_frozen_tensors and trainer.is_world_process_zero():
        frozen_identity = verify_frozen_parameter_identity(model, model_path)
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        temp = output_dir.with_name(f"{output_dir.name}.tmp.{os.getpid()}")
        require(not temp.exists(), f"temporary output exists: {temp}")
        temp.mkdir(parents=True)
        if not args.probe_manifest_only and not args.checkpoint_probe:
            trainer.save_model(str(temp))
            tokenizer.save_pretrained(temp)
            for name in model_source_files:
                source = model_path / name
                require(source.is_file(), f"input model source missing: {name}")
                shutil.copy2(source, temp / name)
            for name in model_source_files:
                require((temp / name).is_file(), f"saved model source: {name}")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "stage": args.stage,
            "source_revision": args.source_revision,
            "source_archive_sha256": args.source_archive_sha256,
            "work_contract": str(work_contract),
            "work_contract_sha256": sha256(work_contract),
            "input_model_manifest": str(input_model_manifest),
            "input_model_manifest_sha256": args.model_manifest_sha256,
            "dataset": args.dataset_name,
            "dataset_revision": args.dataset_revision,
            "dataset_manifest": str(dataset_manifest),
            "dataset_manifest_sha256": args.dataset_manifest_sha256,
            "dataset_train_examples": len(train_dataset),
            "model_type": model.config.model_type,
            "router_penalty": effective_router_penalty,
            "router_penalty_stage_default": stage["penalty"],
            "router_penalty_override": args.router_penalty,
            "router_penalty_gradient_accumulation_steps": effective_penalty_accumulation,
            "router_penalty_accumulation_override": args.router_penalty_accumulation_steps,
            "world_size": world_size,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": global_batch,
            "learning_rate": args.learning_rate,
            "epochs": 1.0,
            "max_steps": args.max_steps,
            "completed_global_step": trainer.state.global_step,
            "max_length": args.max_length,
            "save_total_limit": args.save_total_limit,
            "probe_manifest_only": args.probe_manifest_only,
            "checkpoint_probe": args.checkpoint_probe,
            "stop_after_checkpoint_step": args.stop_after_checkpoint_step,
            "resume_evidence": resume_evidence,
            "sealed_checkpoint_steps": sealed_checkpoint_steps(work_dir),
            "frozen_parameter_identity": frozen_identity,
            "process_id": os.getpid(),
            "seed": args.seed,
            "train_metrics": result.metrics,
            **peak_cuda_memory,
            **trainable,
            "output_files": output_hashes(temp),
        }
        (temp / "TRAINING_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temp.rename(output_dir)
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        # Publish is rank0's own rename; remote ranks may see a stale NFS
        # attribute cache for up to acdirmax and must not fail the run on it.
        require(output_dir.is_dir(), "final model output was not published")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
