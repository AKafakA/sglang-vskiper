# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (eager_on_graph decorators on attention / mamba layers).
No torch.compile.
"""

from __future__ import annotations

import dataclasses
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphMixin,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class BreakableCudaGraphBackend(DedupedCudaGraphMixin, BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention / mamba boundaries;
    attention metadata is recomputed at replay outside captured segments.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        self._model_runner = cuda_graph_runner.model_runner
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._static_forward_batches: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._debug_eager = debug_eager
        self._bind_vp_runtime_identity = (
            os.environ.get("SGLANG_VP_ASYNC_KV", "0") == "1"
            or os.environ.get("SGLANG_FD_VP_ASYNC_KV", "0") == "1"
        )
        self._shared_output_buffer: Optional[Any] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        self._shared_output_buffer = None
        self.begin_cuda_graph_capture()
        try:
            with self.replay_session():
                yield
        finally:
            try:
                self.end_cuda_graph_capture()
            finally:
                self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()
            if self._bind_vp_runtime_identity:
                self._device_module.synchronize()
                _reset_vp_async_runtime_state()

        graph = BreakableCUDAGraph(self.deduped_cuda_graph)
        captured_fn = (
            eager_on_graph(True)(forward_fn) if self._debug_eager else forward_fn
        )
        size = shape_key.size
        with BreakableCUDAGraphCapture(
            cuda_graph=graph,
            pool=self._pool,
            stream=self._capture_stream,
        ):
            out = captured_fn()
            if self._shared_output_buffer is not None:
                self._copy_output_to_buffer(out, self._shared_output_buffer, size)

        if self._bind_vp_runtime_identity:
            self._device_module.synchronize()
            _reset_vp_async_runtime_state()

        if self._shared_output_buffer is None:
            self._shared_output_buffer = out
            stored = self._slice_output(out, size)
        else:
            stored = self._slice_output(self._shared_output_buffer, size)

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = stored
        if dummies is not None:
            self._static_forward_batches[shape_key] = dummies

    def _slice_output(self, output: Any, num_tokens: int) -> Any:
        if output is None:
            return None
        if torch.is_tensor(output):
            return output[:num_tokens]
        if isinstance(output, PPProxyTensors):
            return output[:num_tokens]
        if isinstance(output, tuple):
            return tuple(self._slice_output(item, num_tokens) for item in output)
        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        if dataclasses.is_dataclass(output):
            values = {}
            for field in dataclasses.fields(output):
                value = getattr(output, field.name)
                values[field.name] = (
                    value[:num_tokens] if torch.is_tensor(value) else value
                )
            return type(output)(**values)
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _copy_output_to_buffer(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        if output is None or output_buffer is None:
            if output is None and output_buffer is None:
                return
            raise ValueError(
                "BCG output structure changed between capture sizes: "
                f"{type(output)} vs {type(output_buffer)}"
            )
        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
        if isinstance(output, PPProxyTensors) and isinstance(
            output_buffer, PPProxyTensors
        ):
            if output.tensors.keys() != output_buffer.tensors.keys():
                raise ValueError(
                    "BCG output proxy structure changed between capture sizes: "
                    f"{output.tensors.keys()} != {output_buffer.tensors.keys()}"
                )
            for key, tensor in output.tensors.items():
                self._copy_output_to_buffer(
                    tensor, output_buffer.tensors[key], num_tokens
                )
            return
        if isinstance(output, (list, tuple)) and isinstance(
            output_buffer, type(output)
        ):
            if len(output) != len(output_buffer):
                raise ValueError(
                    "BCG output sequence structure changed between capture sizes: "
                    f"{len(output)} != {len(output_buffer)}"
                )
            for item, buffer in zip(output, output_buffer):
                self._copy_output_to_buffer(item, buffer, num_tokens)
            return
        if (
            dataclasses.is_dataclass(output)
            and type(output_buffer) is type(output)
        ):
            for field in dataclasses.fields(output):
                value = getattr(output, field.name)
                buffer = getattr(output_buffer, field.name)
                if torch.is_tensor(value) and torch.is_tensor(buffer):
                    buffer[:num_tokens].copy_(value[:num_tokens])
                elif value is None and buffer is None:
                    continue
                else:
                    setattr(output_buffer, field.name, value)
            return
        raise TypeError(
            "Unsupported BCG output buffer pair: "
            f"{type(output)} vs {type(output_buffer)}"
        )

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        with enable_breakable_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        if self._bind_vp_runtime_identity:
            captured_forward_batch = self._static_forward_batches.get(shape_key)
            if captured_forward_batch is not None:
                _bind_vp_runtime_identity(
                    captured_forward_batch,
                    static_forward_batch,
                )
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self.close()
        self._graphs.clear()
        self._outputs.clear()
        self._static_forward_batches.clear()
        self._pool = None
        self._shared_output_buffer = None


_VP_RUNTIME_IDENTITY_FIELDS = (
    "rids",
    "vp_req_pool_indices_cpu",
    "vp_token_epochs",
    "vp_kv_positions",
)


def _reset_vp_async_runtime_state() -> None:
    """Discard synthetic warmup/capture repair ownership before serving."""

    from sglang.srt.vpipe.kv_commit import (
        reset_batched_kv_queues,
    )
    from sglang.srt.vpipe.kv_commit import (
        reset_kv_readiness_trackers,
    )

    reset_batched_kv_queues()
    reset_kv_readiness_trackers()


def _bind_vp_runtime_identity(captured_forward_batch: Any, runtime_batch: Any) -> None:
    """Bind request-scoped K/V identity to a captured decode batch.

    Breakable graph closures execute against the capture-time ``ForwardBatch``.
    GPU inputs are refreshed through the graph buffer registry, but VP readiness
    also needs four CPU lists from the live batch. Keep synthetic padded rows
    stable and advance their epoch with the real token so dummy-only work cannot
    accumulate indefinitely in the readiness tracker.
    """

    mode = getattr(captured_forward_batch, "forward_mode", None)
    try:
        if mode is None or not mode.is_decode():
            return
    except Exception:
        return

    values = [getattr(runtime_batch, name, None) for name in _VP_RUNTIME_IDENTITY_FIELDS]
    if any(value is None for value in values):
        for name in _VP_RUNTIME_IDENTITY_FIELDS:
            setattr(captured_forward_batch, name, None)
        return

    raw_bs = int(getattr(runtime_batch, "batch_size", len(values[0])))
    padded_bs = int(getattr(captured_forward_batch, "batch_size", raw_bs))
    if raw_bs < 0 or raw_bs > padded_bs or any(len(value) < raw_bs for value in values):
        for name in _VP_RUNTIME_IDENTITY_FIELDS:
            setattr(captured_forward_batch, name, None)
        return

    rids, slots, epochs, positions = (list(value[:raw_bs]) for value in values)
    pad_epoch = max((int(value) for value in epochs), default=0)
    pad_position = max((int(value) for value in positions), default=0)
    for row in range(raw_bs, padded_bs):
        rids.append(f"__fdvp_bcg_padding_{padded_bs}_{row}")
        slots.append(-(row + 1))
        epochs.append(pad_epoch)
        positions.append(pad_position)

    captured_forward_batch.rids = rids
    captured_forward_batch.vp_req_pool_indices_cpu = [int(value) for value in slots]
    captured_forward_batch.vp_token_epochs = [int(value) for value in epochs]
    captured_forward_batch.vp_kv_positions = [int(value) for value in positions]
